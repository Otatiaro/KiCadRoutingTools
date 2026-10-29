#!/usr/bin/env python3
"""A route boxed in by other nets' copper is not reported boxed in by static obstacles.

`static_boxin_hint` called a search that died early with nothing to rip
"boxed in by static obstacles (neighboring pads + clearance)" and advised a
finer grid or a narrower track -- and recorded the verdict `boxed_in_static`
in the summary. When nothing was rippable because the wall is copper this run
may not rip (an earlier step's nets, or the whole board when the end-of-run
reconciliation retries a net against it), the #301 hint printed next to it
named that copper and the --rip-existing-nets retry that opens it: two
contradicting diagnoses, and a static verdict a caller counting box-ins took
at its word. The #301 attribution now runs first, and with any copper named
no static verdict is given.

Checks:
  1. static_boxin_hint: an early failure gives the static verdict; the same
     failure with copper blockers named gives none.
  2. route.py on a pad walled in by a ring of an earlier step's net W (not
     rippable: KICAD_RIP_PREEXISTING=0): the #301 hint names W and no static
     hint is printed. The same ring with no net, or KiCad-locked (no flag rips
     it, #521), is still reported boxed in by static obstacles.

    python3 tests/test_boxed_in_static_needs_no_copper.py
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from run_utils import check as run_check, evidence, tool       # noqa: E402
from synth import RoutingBoard                                 # noqa: E402

failures = []
STATIC = 'boxed in by static obstacles'
COPPER = 'blocking copper belongs to'


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def unit():
    print("1. static_boxin_hint and copper blockers")
    from routing_diagnostics import static_boxin_hint
    cfg = SimpleNamespace(grid_step=0.1, clearance=0.2, track_width=0.2)
    early = {'iterations_forward': 600, 'iterations_backward': 5000}
    hint, verdict = static_boxin_hint(early, cfg, return_verdict=True)
    check("an early failure with nothing named is static",
          STATIC in hint and (verdict or {}).get('verdict') == 'boxed_in_static')
    hint, verdict = static_boxin_hint(early, cfg, return_verdict=True,
                                      copper_blockers=['W'])
    check("with copper blockers named it is not", hint == '' and verdict is None)


def board(wall, locked=False):
    """N's pad X1 inside a ring on both layers; the ring is net `wall`'s
    copper (an earlier step's, with a pad of its own), KiCad-locked or not,
    or carries no net."""
    b = RoutingBoard()
    b.pad('X1', 20, 15, 'N')
    b.pad('X2', 35, 25, 'N')
    for layer in ('F.Cu', 'B.Cu'):
        b.ring([(18.8, 13.8), (21.2, 13.8), (21.2, 16.2), (18.8, 16.2)], wall,
               layer=layer, locked=locked)
    if wall:
        b.pad('W1', 5, 5, wall)
        b.seg(5, 5, 18.8, 13.8, wall, w=0.2)
    return b.text()


def route(text, tmp, tag):
    inp = os.path.join(tmp, f'{tag}_in.kicad_pcb')
    out = os.path.join(tmp, f'{tag}_out.kicad_pcb')
    with open(inp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    evidence(inp, 'synthetic input board')
    env = dict(os.environ, KICAD_RIP_PREEXISTING='0')
    r = run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                   '--nets', 'N'], accept=True, env=env)
    return r.stdout + r.stderr


def whole_board():
    print("2. a pad walled in by an earlier step's copper")
    tmp = tempfile.mkdtemp(prefix='tboxcu_')
    try:
        log = route(board('W'), tmp, 'copper')
        if 'no rippable blockers found' not in log:
            print("  [BROKEN TEST] the walled pad never reached the "
                  "'no rippable blockers' failure")
            failures.append('broken fixture')
            return
        check("the #301 hint names the copper", COPPER in log and "'W'" in log)
        check("no static box-in is reported", STATIC not in log)
        log = route(board(''), tmp, 'netless')
        check("control: a ring with no net is still a static box-in",
              STATIC in log and COPPER not in log)
        log = route(board('W', locked=True), tmp, 'locked')
        check("a ring of KiCad-locked copper is still a static box-in (no flag "
              "rips it)", STATIC in log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    unit()
    whole_board()
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
