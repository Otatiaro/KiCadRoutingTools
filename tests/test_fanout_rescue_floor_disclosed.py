#!/usr/bin/env python3
"""A BGA fanout ball rescued at the fab floor's clearance is disclosed and
written back, as qfn_fanout's escalated stubs are (#1328).

Under the default --escalation fab, the under-pad escape's fab-floor passes
(the shrink rescue, the rip-swap rescue, the plane drops at the floor) re-run
at the advanced tier's clearance, 0.09 mm. They shipped that copper with no
`design_rules` clearance row and without the clearance ledger, so the output
project kept the run's --clearance as its Default class and KiCad -- and
check_drc, which grades at the project -- reported the rescued escapes as
clearance violations.

Board (synthetic, 4 layers): an 8x8 BGA at 0.8 mm, one net per ball, fanned
under the pads at --clearance 0.15 with 0.15 tracks and 0.45/0.2 vias, which
the inner balls cannot escape at; the shrink rescue escapes them at the
floor. Checked: the rescue fired; each rescued net has a clearance row at
the delivered value; the output project's Default class is that value, with
the #1160 record naming the nets; check_drc at the project's own floor finds
nothing.

Standalone:  python3 tests/test_fanout_rescue_floor_disclosed.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from run_utils import run_step, tool                              # noqa: E402

CL, TW, VS, VD = 0.15, 0.15, 0.45, 0.2
fails = []


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f" -- {detail}" if detail else ''))
    if not cond:
        fails.append(name)


def write_board(path, n=8, pitch=0.8):
    """An n x n BGA at (10, 10), one net per ball, each with a partner pad."""
    nets, pads, partners = [''], [], []
    off = (n - 1) * pitch / 2.0
    rows = 'ABCDEFGHJK'
    for i in range(n):
        for j in range(n):
            name = f'U1_{rows[i]}{j + 1}'
            nets.append(name)
            pads.append(f'  (pad "{rows[i]}{j + 1}" smd circle (at {round(j * pitch - off, 4)} '
                        f'{round(i * pitch - off, 4)}) (size 0.35 0.35) (layers "F.Cu") '
                        f'(net {len(nets) - 1} "{name}"))')
            k = len(nets) - 2
            partners.append(f'  (pad "{k + 1}" smd rect (at {round(1.0 + k * 0.28, 4)} 0) '
                            f'(size 0.15 0.4) (layers "F.Cu") (net {len(nets) - 1} "{name}"))')

    def fp(ref, x, y, body):
        return (f' (footprint "t:{ref}" (layer "F.Cu") (at {x} {y})\n'
                f'  (property "Reference" "{ref}" (at 0 -2) (layer "F.SilkS"))\n'
                + '\n'.join(body) + '\n )')
    edge = '\n'.join(
        f' (gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (layer "Edge.Cuts") (width 0.1))'
        for a, b in [((0, 0), (20, 0)), ((20, 0), (20, 20)), ((20, 20), (0, 20)), ((0, 20), (0, 0))])
    layers = ('(layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) '
              '(2 "B.Cu" signal) (25 "Edge.Cuts" user) (5 "F.SilkS" user) '
              '(1 "F.Mask" user) (3 "B.Mask" user))')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f'(kicad_pcb (version 20240108) (generator "test")\n {layers}\n'
                 + '\n'.join(f' (net {i} "{m}")' for i, m in enumerate(nets)) + f'\n{edge}\n'
                 + fp('U1', 10, 10, pads) + '\n' + fp('J1', 0, 19, partners) + '\n)\n')
    proj = {"board": {"design_settings": {"rules": {
                "min_clearance": CL, "min_hole_to_hole": 0.25, "min_track_width": 0.09,
                "min_via_annular_width": 0.05, "min_via_diameter": 0.25,
                "min_through_hole_diameter": 0.15, "min_copper_edge_clearance": 0.2}}},
            "net_settings": {"classes": [{"name": "Default", "clearance": CL}]}}
    with open(os.path.splitext(path)[0] + '.kicad_pro', 'w') as fh:
        json.dump(proj, fh)
    return path


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = write_board(os.path.join(tmp, 'bga.kicad_pcb'))
        out = os.path.join(tmp, 'fan.kicad_pcb')
        _rc, log = run_step([tool('bga_fanout.py'), src, '--component', 'U1',
                             '--escape-method', 'underpad', '--track-width', TW, '--clearance', CL,
                             '--via-size', VS, '--via-drill', VD,
                             '--layers', 'F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu', '--output', out],
                            outputs=[out])
        rescue = [l for l in log.splitlines() if 'Under-pad shrink rescue' in l]
        check('the shrink rescue escaped balls at the fab floor',
              bool(rescue) and 'rescued 0 of' not in rescue[0], (rescue or ['none'])[0].strip())
        row = [l for l in log.splitlines() if l.startswith('JSON_SUMMARY')][-1]
        s = json.loads(row.split(':', 1)[1])
        dr = s.get('design_rules') or {}
        rows = [r for r in dr.get('narrowed') or () if r['kind'] == 'clearance']
        floor = min((r['delivered'] for r in rows), default=None)
        check('the rescued nets have a clearance row below --clearance',
              len(rows) > 0 and floor is not None and floor < CL,
              f"{len(rows)} row(s), smallest {floor}")
        check('min_delivered names the clearance', floor is not None
              and (dr.get('min_delivered') or {}).get('clearance') == floor,
              str(dr.get('min_delivered')))
        check('the summary reports it as the clearance used', s.get('min_clearance_used') == floor,
              str(s.get('min_clearance_used')))
        with open(os.path.splitext(out)[0] + '.kicad_pro', encoding='utf-8') as fh:
            proj = json.load(fh)
        default = [c.get('clearance') for c in proj['net_settings']['classes'] if c['name'] == 'Default']
        check('the output project carries the delivered clearance', default == [floor], str(default))
        rec = (proj.get('kicad_routing_tools') or {}).get('class_clearance_relaxed') or {}
        check('the relaxed Default class is recorded with its nets (#1160)',
              rec.get('from') == CL and rec.get('to') == floor
              and set(rec.get('nets') or ()) == {r['net_name'] for r in rows}, str(rec)[:200])
        drc = os.path.join(tmp, 'drc.json')
        run_step([tool('check_drc.py'), out, '--clearance-margin', 0, '--json', drc],
                 outputs=[drc], findings_ok=True)
        with open(drc, encoding='utf-8') as fh:
            by_type = json.load(fh).get('by_type', {})
        check('check_drc at the project floor finds nothing', not by_type, str(by_type))
    print()
    if fails:
        print(f'FAILED: {len(fails)} check(s): {fails}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
