#!/usr/bin/env python3
"""The BGA and QFN fanouts, and the cap pass after them, never write a
declared min_hole_clearance down in their output project.

The fanouts write their DRC floors back through compute_targets like every
routing step, and on a board declaring min_hole_clearance 0.25 they wrote it
down to the copper clearance they escaped at (0.25 -> 0.1), so the cap pass,
every later step and KiCad's own DRC then graded the hole floor at 0.1.

Board (synthetic, 4 layers, project declaring clearance 0.1 and
min_hole_clearance 0.25): an 8x8 BGA with four back-side caps, fanned by
bga_fanout.py and then place_fanout_clearance.py, and a QFN-40 fanned by
qfn_fanout.py. Each output project keeps 0.25.

Standalone:  python3 tests/test_fanout_keeps_declared_hole_clearance.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from run_utils import run_step, tool                              # noqa: E402

HC, CL = 0.25, 0.1
fails = []


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f" -- {detail}" if detail else ''))
    if not cond:
        fails.append(name)


class Board:
    """A 4-layer board as text, with the project declaring the floors."""

    def __init__(self):
        self.nets = ['']
        self.items = []

    def net(self, name):
        if name not in self.nets:
            self.nets.append(name)
        return self.nets.index(name)

    def pad(self, num, x, y, sx, sy, net, shape='rect', layer='F.Cu'):
        n = self.net(net)
        return (f'  (pad "{num}" smd {shape} (at {x} {y}) (size {sx} {sy}) '
                f'(layers "{layer}") (net {n} "{net}"))')

    def fp(self, ref, x, y, pads, layer='F.Cu'):
        self.items.append(f' (footprint "t:{ref}" (layer "{layer}") (at {x} {y})\n'
                          f'  (property "Reference" "{ref}" (at 0 -2) (layer "F.SilkS"))\n'
                          + '\n'.join(pads) + '\n )')

    def bga(self, ref, cx, cy, n=8, pitch=0.8):
        off = (n - 1) * pitch / 2.0
        rows = 'ABCDEFGHJK'
        self.fp(ref, cx, cy, [self.pad(f'{rows[i]}{j + 1}', round(j * pitch - off, 4),
                                       round(i * pitch - off, 4), 0.35, 0.35,
                                       f'{ref}_{rows[i]}{j + 1}', shape='circle')
                              for i in range(n) for j in range(n)])

    def qfn(self, ref, cx, cy, per_side=10, pitch=0.4, body=5.0):
        pads, k = [], 0
        span = (per_side - 1) * pitch / 2.0
        e = body / 2.0 - 0.35
        for side in range(4):
            for i in range(per_side):
                k += 1
                t = round(i * pitch - span, 4)
                x, y, sx, sy = [(-e, t, 0.7, 0.25), (t, e, 0.25, 0.7),
                                (e, -t, 0.7, 0.25), (-t, -e, 0.25, 0.7)][side]
                pads.append(self.pad(str(k), x, y, sx, sy, f'{ref}_{k}'))
        self.fp(ref, cx, cy, pads)

    def cap(self, ref, x, y, net_a, net_b):
        self.fp(ref, x, y, [self.pad('1', -0.48, 0, 0.56, 0.62, net_a, layer='B.Cu'),
                            self.pad('2', 0.48, 0, 0.56, 0.62, net_b, layer='B.Cu')],
                layer='B.Cu')

    def partners(self, ref, y, w=20):
        """One pad per net on a row, so every net is a connection to make."""
        nets = self.nets[1:]
        step = (w - 2.0) / max(1, len(nets))
        self.fp(ref, 0, y, [self.pad(str(k + 1), round(1.0 + k * step, 4), 0, 0.15, 0.4, n)
                            for k, n in enumerate(nets)])

    def write(self, path, w=20, h=20):
        edge = '\n'.join(
            f' (gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (layer "Edge.Cuts") (width 0.1))'
            for a, b in [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))])
        nets = '\n'.join(f' (net {i} "{n}")' for i, n in enumerate(self.nets))
        layers = ('(layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) '
                  '(2 "B.Cu" signal) (25 "Edge.Cuts" user) (5 "F.SilkS" user) '
                  '(1 "F.Mask" user) (3 "B.Mask" user))')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(f'(kicad_pcb (version 20240108) (generator "test")\n {layers}\n'
                     f'{nets}\n{edge}\n' + '\n'.join(self.items) + '\n)\n')
        rules = {"min_clearance": CL, "min_hole_clearance": HC, "min_hole_to_hole": 0.25,
                 "min_track_width": 0.09, "min_via_annular_width": 0.05,
                 "min_via_diameter": 0.25, "min_through_hole_diameter": 0.15,
                 "min_copper_edge_clearance": 0.2}
        proj = {"board": {"design_settings": {"rules": rules}},
                "net_settings": {"classes": [{
                    "name": "Default", "clearance": CL, "track_width": 0.12,
                    "via_diameter": 0.35, "via_drill": 0.2}]}}
        with open(os.path.splitext(path)[0] + '.kicad_pro', 'w') as fh:
            json.dump(proj, fh)
        return path


def rule(pcb):
    with open(os.path.splitext(pcb)[0] + '.kicad_pro', encoding='utf-8') as fh:
        return json.load(fh)['board']['design_settings']['rules'].get('min_hole_clearance')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        b = Board()
        b.bga('U1', 10, 10)
        for k, (x, y) in enumerate([(8.8, 8.4), (10.4, 9.2), (11.2, 10.8), (8.4, 11.6)]):
            b.cap(f'C{k + 1}', x, y, 'U1_C3', f'U1_F{k + 3}')
        b.partners('J1', 19)
        src = b.write(os.path.join(tmp, 'bga.kicad_pcb'))
        fan = os.path.join(tmp, 'bga_fan.kicad_pcb')
        run_step([tool('bga_fanout.py'), src, '--component', 'U1', '--escape-method', 'underpad',
                  '--track-width', 0.12, '--clearance', CL, '--via-size', 0.35, '--via-drill', 0.2,
                  '--layers', 'F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu', '--output', fan], outputs=[fan])
        check('bga_fanout keeps the declared 0.25', rule(fan) == HC, f'wrote {rule(fan)}')
        cap = os.path.join(tmp, 'bga_cap.kicad_pcb')
        run_step([tool('place_fanout_clearance.py'), fan, cap, '--clearance', CL], outputs=[cap])
        check('place_fanout_clearance keeps the declared 0.25', rule(cap) == HC, f'wrote {rule(cap)}')

        q = Board()
        q.qfn('U2', 10, 10)
        q.partners('J1', 19)
        qsrc = q.write(os.path.join(tmp, 'qfn.kicad_pcb'))
        qfan = os.path.join(tmp, 'qfn_fan.kicad_pcb')
        run_step([tool('qfn_fanout.py'), qsrc, '--component', 'U2', '--escape-method', 'underpad',
                  '--width', 0.12, '--clearance', CL, '--via-size', 0.35, '--via-drill', 0.2,
                  '--output', qfan], outputs=[qfan])
        check('qfn_fanout keeps the declared 0.25', rule(qfan) == HC, f'wrote {rule(qfan)}')
    print()
    if fails:
        print(f'FAILED: {len(fails)} check(s): {fails}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
