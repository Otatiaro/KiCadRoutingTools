#!/usr/bin/env python3
"""A routing step never writes a DECLARED copper-to-hole rule down.

Every routing writeback (route.py, route_diff.py, repair_planes.py,
route_planes.py, the GUI tabs) goes through `compute_targets` and
`apply_targets_to_project` / `apply_targets_to_board`. When the step was
given no explicit hole clearance, `compute_targets` used to target
`min_hole_clearance` at the step's COPPER clearance, and the only-lower writer
then took a declared rule down to it: a chain on a board declaring 0.2 wrote
0.199 (route_diff), 0.182 (route.py, the distance of one tap via to an NPTH
wall) and then 0.09 (repair_planes), and every later step and checker graded
at the relaxed value. A declared rule is the board's spec; copper under it is
a violation for KiCad to report.

Rows:
  A. compute_targets emits no `min_hole_clearance` target without an
     explicit hole clearance; the copper-clearance fallback rides its own key
  B. fix_project_for_output at clearance 0.09 keeps a declared 0.2
  C. ...and writes 0.09 where the project declares no copper-to-hole rule
     (KiCad's factory 0.25 would otherwise grade it)
  D. an explicit hole clearance is still a request the writer honours
  E. the mid-run sync (apply_routed_floors) keeps a declared 0.2 too
  F. held_hole_clearance (route_diff, the diff tab) returns the held value
     whatever the step's own copper reached

    python3 tests/test_declared_hole_clearance_kept.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

import fix_kicad_drc_settings as fk  # noqa: E402
from fix_kicad_drc_settings import (apply_routed_floors, compute_targets,  # noqa: E402
                                    fix_project_for_output, held_hole_clearance)

# Read off the module so the rows below still grade the BEHAVIOUR on a
# writer that predates the key, instead of dying on the import.
UNDECLARED_HOLE_CLEARANCE = getattr(fk, 'UNDECLARED_HOLE_CLEARANCE', None)

FAILS = []


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def near(a, b, tol=1e-6):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


BOARD = '''(kicad_pcb
 (version 20240108)
 (generator "test")
 (general (thickness 1.6))
 (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
 (setup (pad_to_mask_clearance 0))
 (net 0 "")
 (net 1 "GND")
 (gr_rect (start 0 0) (end 20 20) (stroke (width 0.1) (type solid)) (layer "Edge.Cuts"))
 (segment (start 2 2) (end 6 2) (width 0.2) (layer "F.Cu") (net 1) (uuid "s1"))
)
'''


def project(rules):
    return {"board": {"design_settings": {"rules": dict(rules),
                                          "rule_severities": {}}},
            "net_settings": {"meta": {"version": 3},
                             "classes": [{"name": "Default", "clearance": 0.09,
                                          "track_width": 0.2, "via_diameter": 0.45,
                                          "via_drill": 0.3}]},
            "meta": {"filename": "b.kicad_pro", "version": 1}}


def stage(td, rules):
    pcb = os.path.join(td, 'b.kicad_pcb')
    with open(pcb, 'w', encoding='utf-8') as f:
        f.write(BOARD)
    with open(os.path.join(td, 'b.kicad_pro'), 'w', encoding='utf-8') as f:
        json.dump(project(rules), f)
    return pcb


def rule(pcb):
    with open(os.path.splitext(pcb)[0] + '.kicad_pro', encoding='utf-8') as f:
        return json.load(f)['board']['design_settings']['rules'].get('min_hole_clearance')


print('\n-- A. compute_targets --')
t = compute_targets(clearance=0.09)
check('no min_hole_clearance target without an explicit value',
      'min_hole_clearance' not in t, sorted(t))
check('the copper-clearance fallback rides its own key',
      near(t.get(UNDECLARED_HOLE_CLEARANCE), 0.09), sorted(t))
check('a step that drew no copper has no fallback either',
      UNDECLARED_HOLE_CLEARANCE not in compute_targets(clearance=0.09,
                                                       copper_holes=False))

print('\n-- B. declared 0.2, step at 0.09 --')
with tempfile.TemporaryDirectory() as td:
    pcb = stage(td, {"min_clearance": 0.09, "min_hole_clearance": 0.2})
    fix_project_for_output(pcb, clearance=0.09, verbose=False)
    check('min_hole_clearance stays 0.2', near(rule(pcb), 0.2), rule(pcb))

print('\n-- C. nothing declared --')
with tempfile.TemporaryDirectory() as td:
    pcb = stage(td, {"min_clearance": 0.09})
    fix_project_for_output(pcb, clearance=0.09, verbose=False)
    check('the copper clearance is written', near(rule(pcb), 0.09), rule(pcb))

print('\n-- D. explicit hole clearance --')
with tempfile.TemporaryDirectory() as td:
    pcb = stage(td, {"min_clearance": 0.09, "min_hole_clearance": 0.2})
    fix_project_for_output(pcb, clearance=0.09, hole_clearance=0.15, verbose=False)
    check('an explicit 0.15 is written', near(rule(pcb), 0.15), rule(pcb))

print('\n-- E. mid-run sync --')
with tempfile.TemporaryDirectory() as td:
    pcb = stage(td, {"min_clearance": 0.09, "min_hole_clearance": 0.2})
    apply_routed_floors(pcb, clearance=0.09)
    check('apply_routed_floors keeps 0.2', near(rule(pcb), 0.2), rule(pcb))

print('\n-- F. held_hole_clearance --')
check('own copper at 0.182 -> the held 0.2',
      near(held_hole_clearance(0.2, 0.09, 0.182), 0.2))
check('nothing declared -> None', held_hole_clearance(0.0, 0.09, None) is None)

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
