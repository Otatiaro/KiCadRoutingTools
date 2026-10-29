#!/usr/bin/env python3
"""The BGA fanout's dog-bone and under-pad escapes keep track copper off a
part's NPTH holes at the NPTH floor, as the routing map does.

A Tag-Connect TC2030 footprint is six 0.79 mm copper pads on a 1.27 mm grid
and seven NPTH holes, four of them 2.37 mm. The net rescue's bare-ball escape
fans such a part as a ball field. The escape engine stamped every drilled pad
of the part at the BALL keep-out (the largest netted pad's radius), so a
2.37 mm hole was held off by less than its own radius, and its exact registry
skips pads without copper: escapes ran 0.07 mm from the hole walls, some of
them into the hole, where the routing map holds max(clearance, the 0.20 NPTH
fab floor, the board's min_hole_clearance).

Board (synthetic, 4 layers, project declaring clearance 0.09 and
min_hole_clearance 0.2): the TC2030 pad and hole pattern, one net per pad with
a partner pad, fanned by bga_fanout.py --escape-method dogbone and underpad.
Graded by check_drc at the declared floor: no track-hole row, and every pad
still escapes.

Standalone:  python3 tests/test_fanout_npth_floor.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from run_utils import run_step, tool                              # noqa: E402

HC, CL = 0.2, 0.09
# TC2030-IDC-FP: (x, y, diameter) of the NPTH holes and the copper pads, mm
NPTH = [(2.54, -2.54, 2.3749), (0.0, -2.54, 0.9906), (-2.54, -2.54, 2.3749), (2.54, 0.635, 2.3749),
        (-2.54, 0.635, 2.3749), (1.016, 2.54, 0.9906), (-1.016, 2.54, 0.9906)]
PADS = [(-0.635, -1.27), (0.635, -1.27), (-0.635, 0.0), (0.635, 0.0), (-0.635, 1.27), (0.635, 1.27)]
fails = []


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f" -- {detail}" if detail else ''))
    if not cond:
        fails.append(name)


def write_board(path):
    nets = [''] + [f'N{k + 1}' for k in range(len(PADS))]
    body = [f'  (pad "" np_thru_hole circle (at {x} {y}) (size {d} {d}) (drill {d}) (layers "*.Cu" "*.Mask"))'
            for x, y, d in NPTH]
    body += [f'  (pad "{k + 1}" connect circle (at {x} {y}) (size 0.7874 0.7874) (layers "F.Cu" "F.Mask") '
             f'(net {k + 1} "N{k + 1}"))' for k, (x, y) in enumerate(PADS)]
    partners = [f'  (pad "{k + 1}" smd rect (at {2 + 2 * k} 0) (size 0.4 0.6) (layers "F.Cu") '
                f'(net {k + 1} "N{k + 1}"))' for k in range(len(PADS))]

    def fp(ref, x, y, b):
        return (f' (footprint "t:{ref}" (layer "F.Cu") (at {x} {y})\n'
                f'  (property "Reference" "{ref}" (at 0 -4) (layer "F.SilkS"))\n' + '\n'.join(b) + '\n )')
    edge = '\n'.join(f' (gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (layer "Edge.Cuts") (width 0.1))'
                     for a, b in [((0, 0), (40, 0)), ((40, 0), (40, 40)), ((40, 40), (0, 40)), ((0, 40), (0, 0))])
    layers = ('(layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal) '
              '(25 "Edge.Cuts" user) (5 "F.SilkS" user) (1 "F.Mask" user) (3 "B.Mask" user))')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f'(kicad_pcb (version 20240108) (generator "test")\n {layers}\n'
                 + '\n'.join(f' (net {i} "{n}")' for i, n in enumerate(nets)) + f'\n{edge}\n'
                 + fp('J5', 20, 20, body) + '\n' + fp('J1', 0, 36, partners) + '\n)\n')
    with open(os.path.splitext(path)[0] + '.kicad_pro', 'w') as fh:
        json.dump({"board": {"design_settings": {"rules": {
                       "min_clearance": CL, "min_hole_clearance": HC, "min_hole_to_hole": 0.2}}},
                   "net_settings": {"classes": [{"name": "Default", "clearance": CL}]}}, fh)
    return path


def fan(tmp, method):
    print(f'TC2030 pattern, --escape-method {method}:')
    src = write_board(os.path.join(tmp, f'{method}.kicad_pcb'))
    out = os.path.join(tmp, f'{method}_fan.kicad_pcb')
    _rc, log = run_step([tool('bga_fanout.py'), src, '--component', 'J5', '--escape-method', method,
                         '--track-width', 0.09, '--clearance', CL, '--via-size', 0.45, '--via-drill', 0.3,
                         '--layers', 'F.Cu', 'B.Cu', '--output', out], outputs=[out])
    drc = os.path.join(tmp, f'{method}.drc.json')
    run_step([tool('check_drc.py'), out, '--clearance', CL, '--hole-clearance', HC,
              '--clearance-margin', 0, '--json', drc], outputs=[drc], findings_ok=True)
    with open(drc, encoding='utf-8') as fh:
        by_type = json.load(fh).get('by_type', {})
    check(f'{method}: no track inside the NPTH floor', not by_type.get('track-hole'), str(by_type))
    row = [l for l in log.splitlines() if l.startswith('JSON_SUMMARY')][-1]
    s = json.loads(row.split(':', 1)[1])
    check(f'{method}: every pad escapes', s.get('requested', 0) == len(PADS) and s.get('failed') == 0,
          f"{s.get('escaped')}/{s.get('requested')}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        fan(tmp, 'dogbone')
        fan(tmp, 'underpad')
    print()
    if fails:
        print(f'FAILED: {len(fails)} check(s): {fails}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
