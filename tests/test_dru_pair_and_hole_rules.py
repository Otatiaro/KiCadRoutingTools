#!/usr/bin/env python3
"""The router holds a board's net-pair clearance rules and pad-scoped
hole_clearance rules from its .kicad_dru (dru_copper_rules).

Two shapes of custom rule bind a pair of items by WHO they are, and the
router used to announce both as skipped and route through them:

  * an isolation barrier, `(constraint clearance (min 1mm))` over a symmetric
    disjunction of `A.NetName == '...'` / `B.NetName == '...'` terms with
    negations and `!= ''` / `!= 'unconnected-*'` guards -- skipped as a
    "non-layer scope" (a via 0.96 mm from the other side on a real board);
  * `A.Type == 'Pad' && A.Pad_Type == 'Through-hole' && A.Hole_Size_X >=
    0.3mm` with `(constraint hole_clearance (min 0.28mm))` -- NOT honoured,
    "property Hole_Size_X not modelled" (vias 0.25-0.28 mm from through-hole
    leads on a real board).

Rows:
  A. the evaluator: Hole_Size_X against a number with a unit, KiCad's
     Pad_Type spelling, NetName wildcards (case-insensitive), &&, ||, !,
     parentheses; a size rule keyed on the hole stays announced
  B. derive: the barrier binds each isolated net against every other named
     net, not the isolated nets among themselves, not 'unconnected-*'
  C. route.py: a signal routed past an isolated net's copper keeps the rule's
     1 mm (it ran 0.35 mm off it before), and the run reports the rule
     honoured rather than skipped
  D. the via cells around a bound through-hole pad: a via whose drill is
     under 0.28 mm from the pad's copper, or whose copper is under 0.28 mm
     from its hole, is blocked -- in the routing stamp and in the plane-tap
     via map -- and the pad's own net is not held off it

    python3 tests/test_dru_pair_and_hole_rules.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from run_utils import check as run_check, evidence, tool        # noqa: E402
from synth import RoutingBoard                                  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


ISO = ("(A.NetName == 'ISO_*' && !(B.NetName == 'ISO_*') && B.NetName != '' "
       "&& B.NetName != 'unconnected-*') || (B.NetName == 'ISO_*' && "
       "!(A.NetName == 'ISO_*') && A.NetName != '' && A.NetName != 'unconnected-*')")
TH = "A.Type == 'Pad' && A.Pad_Type == 'Through-hole' && A.Hole_Size_X >= 0.3mm"
DRU = f'''(version 1)
(rule "isolation" (condition "{ISO}") (constraint clearance (min 1mm)))
(rule "leads" (condition "{TH}") (constraint hole_clearance (min 0.28mm)))
(rule "via size by hole" (condition "A.Type == 'Via' && A.Hole_Size_X < 0.3mm")
  (constraint via_diameter (min 0.45mm)))
'''


def project():
    return {'board': {'design_settings': {'rules': {
                'min_clearance': 0.1, 'min_track_width': 0.15,
                'min_via_diameter': 0.45, 'min_through_hole_diameter': 0.2,
                'min_hole_to_hole': 0.2, 'min_hole_clearance': 0.1,
                'min_copper_edge_clearance': 0.3}, 'rule_severities': {}}},
            'meta': {'filename': 'in.kicad_pro', 'version': 3},
            'net_settings': {'classes': [{'name': 'Default', 'clearance': 0.1,
                                          'track_width': 0.15, 'via_diameter': 0.45,
                                          'via_drill': 0.25}],
                             'meta': {'version': 4}}}


def board():
    b = RoutingBoard(w=40, h=30)
    # the signal, straight across the board at y = 15
    b.pad('J1', 6, 15, 'SIG', sx=1.0, sy=1.0)
    b.pad('J2', 34, 15, 'SIG', sx=1.0, sy=1.0)
    # isolated copper 0.6 mm above that line
    b.pad('K1', 16, 15.6, 'ISO_A', sx=0.4, sy=0.4)
    b.pad('K2', 24, 15.6, 'ISO_A', sx=0.4, sy=0.4)
    b.seg(16, 15.6, 24, 15.6, 'ISO_A', w=0.2)
    # a through-hole lead of another net (drill 0.8, pad 1.1: ring 0.15)
    b.pad('T1', 20, 24, 'OTHER', sx=1.1, sy=1.1, tht=True)
    b.pad('T2', 26, 24, 'OTHER', sx=1.1, sy=1.1, tht=True)
    b.pad('U1', 30, 27, 'unconnected-(U1-Pad1)', sx=0.6, sy=0.6)
    return b


def stage(d):
    pcb = os.path.join(d, 'in.kicad_pcb')
    with open(pcb, 'w', encoding='utf-8') as f:
        f.write(board().text())
    with open(os.path.join(d, 'in.kicad_pro'), 'w', encoding='utf-8') as f:
        json.dump(project(), f, indent=2)
    with open(os.path.join(d, 'in.kicad_dru'), 'w', encoding='utf-8') as f:
        f.write(DRU)
    return pcb


def seg_seg(a, b):
    """Edge-to-edge distance of two track segments."""
    def pt_seg(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
    d = min(pt_seg(a.start_x, a.start_y, b.start_x, b.start_y, b.end_x, b.end_y),
            pt_seg(a.end_x, a.end_y, b.start_x, b.start_y, b.end_x, b.end_y),
            pt_seg(b.start_x, b.start_y, a.start_x, a.start_y, a.end_x, a.end_y),
            pt_seg(b.end_x, b.end_y, a.start_x, a.start_y, a.end_x, a.end_y))
    return d - a.width / 2 - b.width / 2


def t_evaluator():
    print('\n-- A. the condition evaluator --')
    from design_rules import RuleItem, parse_condition, parse_dru, _eval
    pad = RuleItem(type='pad', pad_type='thru_hole', hole_x=0.8, hole_y=0.8)
    small = RuleItem(type='pad', pad_type='thru_hole', hole_x=0.2, hole_y=0.2)
    smd = RuleItem(type='pad', pad_type='smd', hole_x=0.0, hole_y=0.0)
    try:
        ast = parse_condition(TH)
    except Exception as e:                                       # noqa: BLE001
        check('the through-hole lead condition parses', False, repr(e))
        return
    check('the through-hole lead condition parses', True)
    check('a 0.8 mm through-hole pad matches', _eval(ast, pad, None, None))
    check('a 0.2 mm through-hole pad does not', not _eval(ast, small, None, None))
    check('an SMD pad does not', not _eval(ast, smd, None, None))
    check('a track does not', not _eval(ast, RuleItem(type='track'), None, None))
    check("12 mil (0.3048 mm) reads in mm",
          not _eval(parse_condition("A.Hole_Size_X >= 12mil"), small, None, None)
          and _eval(parse_condition("A.Hole_Size_X >= 12mil"), pad, None, None))
    iso = parse_condition(ISO)
    a = RuleItem(net_name='ISO_A')
    s = RuleItem(net_name='SIG')
    check('ISO_A vs SIG binds (both ways round)',
          _eval(iso, a, s, None) and _eval(iso, s, a, None))
    check("the wildcard is case-insensitive ('iso_b')",
          _eval(iso, RuleItem(net_name='iso_b'), s, None))
    check('ISO_A vs ISO_B does not bind',
          not _eval(iso, a, RuleItem(net_name='ISO_B'), None))
    check("ISO_A vs 'unconnected-(U1-Pad1)' does not bind",
          not _eval(iso, a, RuleItem(net_name='unconnected-(U1-Pad1)'), None))
    check("ISO_A vs a net-less item does not bind",
          not _eval(iso, a, RuleItem(net_name=''), None))
    rules, _ = parse_dru(DRU)
    by = {r.name: r for r in rules}
    check('the barrier and the lead rule are evaluated',
          by['isolation'].unsupported is None and by['leads'].unsupported is None,
          f"{by['isolation'].unsupported!r} {by['leads'].unsupported!r}")
    check('a via size rule keyed on the hole is still announced',
          bool(by['via size by hole'].unsupported), by['via size by hole'].unsupported)


def t_derive():
    print('\n-- B. derive --')
    from kicad_parser import parse_kicad_pcb
    from design_rules import DesignRules
    try:
        from dru_copper_rules import derive
    except ImportError as e:
        check('dru_copper_rules exists', False, repr(e))
        return
    with tempfile.TemporaryDirectory() as d:
        pcb_path = stage(d)
        pcb = parse_kicad_pcb(pcb_path)
        rules = DesignRules.from_project(pcb, pcb_path)
        entries = derive(rules, pcb)
    pair = next((e for e in entries if e['kind'] == 'net_pair'), None)
    hole = next((e for e in entries if e['kind'] == 'pad_hole'), None)
    check('a net-pair entry and a pad-hole entry', pair is not None and hole is not None,
          str([e['kind'] for e in entries]))
    if pair:
        g = pair['group_of']
        v = pair['value']
        check('ISO_A vs SIG: 1 mm', v.get((g['ISO_A'], g['SIG'])) == 1.0)
        check('ISO_A vs OTHER: 1 mm', v.get((g['ISO_A'], g['OTHER'])) == 1.0)
        check('SIG vs OTHER: unbound', (g['SIG'], g['OTHER']) not in v)
        check("ISO_A vs 'unconnected-(U1-Pad1)': unbound",
              (g['ISO_A'], g['unconnected-(U1-Pad1)']) not in v)
    if hole:
        check('the two through-hole leads are bound, at 0.28',
              sorted(p[0] for p in hole['pads']) == ['T1', 'T2']
              and abs(hole['clearance'] - 0.28) < 1e-9, str(hole['pads']))
    by = {r.name: r for r in rules.rules}
    check('both rules are reported honoured',
          bool(by['isolation'].honoured) and bool(by['leads'].honoured))


def t_route():
    print('\n-- C. route.py past the isolated copper --')
    from kicad_parser import parse_kicad_pcb
    tmp = tempfile.mkdtemp(prefix='druiso_')
    try:
        inp = stage(tmp)
        out = os.path.join(tmp, 'out.kicad_pcb')
        r = run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                       '--nets', 'SIG', '--escalation', 'off', '--clearance', '0.1',
                       '--track-width', '0.15', '--via-size', '0.45',
                       '--via-drill', '0.25', '--grid-step', '0.05'],
                      accept=True, env=dict(os.environ, KICAD_PLANE_FINALIZE='0'))
        log = (r.stdout or '') + (r.stderr or '')
        pcb = parse_kicad_pcb(evidence(out, 'routed board'))
        nid = {n.name: i for i, n in pcb.nets.items()}
        sig = [s for s in pcb.segments if s.net_id == nid['SIG']]
        iso = [s for s in pcb.segments if s.net_id == nid['ISO_A']]
        check('SIG routed', bool(sig), f'{len(sig)} segment(s)')
        if sig and iso:
            same = [seg_seg(a, b) for a in sig for b in iso if a.layer == b.layer]
            d = min(same) if same else float('inf')
            check('SIG keeps the 1 mm barrier off ISO_A copper', d >= 1.0 - 0.03,
                  f'{d:.3f} mm')
        check("the run reports the barrier honoured, not skipped",
              "'isolation': honoured" in log
              and "rule 'isolation': clearance 1.0mm has a non-layer scope" not in log)
        check("the hole-keyed via size rule is still announced",
              "'via size by hole': NOT honoured" in log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_hole_cells():
    print('\n-- D. via cells around a bound through-hole lead --')
    from kicad_parser import parse_kicad_pcb
    from design_rules import DesignRules
    from routing_config import GridRouteConfig, GridCoord
    try:
        from dru_copper_rules import derive, cells_for
        from net_keepouts import attach
        from plane_obstacle_builder import build_via_obstacle_map
    except ImportError as e:
        check('dru_copper_rules exists', False, repr(e))
        return
    with tempfile.TemporaryDirectory() as d:
        pcb_path = stage(d)
        pcb = parse_kicad_pcb(pcb_path)
        rules = DesignRules.from_project(pcb, pcb_path)
        attach(pcb, derive(rules, pcb))
        c = GridRouteConfig(layers=['F.Cu', 'B.Cu'], grid_step=0.05, clearance=0.1,
                            track_width=0.15, via_size=0.45, via_drill=0.25,
                            hole_to_hole_clearance=0.2)
        nid = {n.name: i for i, n in pcb.nets.items()}
        coord = GridCoord(c.grid_step)
        _cells, vias = cells_for(pcb, (nid['SIG'],), c)
        vset = {tuple(v) for v in vias.tolist()}
        own = {tuple(v) for v in cells_for(pcb, (nid['OTHER'],), c)[1].tolist()}

        def cell(dx):
            return coord.to_grid(20 + dx, 24)
        # T1 at (20, 24): copper r 0.55, hole r 0.4. A via's drill (r 0.125)
        # must sit 0.28 off the copper: centre >= 0.955 from the pad centre.
        check('a via 0.90 mm off (drill 0.225 from the copper) is blocked',
              cell(0.90) in vset)
        check('a via 1.00 mm off (drill 0.325 from the copper) is free',
              cell(1.00) not in vset)
        check("the pad's own net is not held off it", cell(0.90) not in own)
        tap = build_via_obstacle_map(pcb, c, nid['SIG'], verbose=False)
        gx, gy = cell(0.90)
        check('the plane-tap via map blocks it too', tap.is_via_blocked(gx, gy))


if __name__ == '__main__':
    for section in (t_evaluator, t_derive, t_route, t_hole_cells):
        try:
            section()
        except Exception as e:                                   # noqa: BLE001
            # an older router without the API under test: a failed row with
            # its reason, and the next section still runs
            check(f'{section.__name__} ran', False, repr(e))
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
    sys.exit(1 if FAILS else 0)
