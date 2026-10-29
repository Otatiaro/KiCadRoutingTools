#!/usr/bin/env python3
"""The BGA and QFN fanouts, and the cap pass after them, hold a board's
min_hole_clearance at the vias they place, as KiCad grades it.

KiCad 10's hole_clearance rule holds foreign copper `min_hole_clearance` from
every hole wall, via drills included. A 0.35/0.2 via has a 0.075 mm ring, so
on a board declaring clearance 0.1 and min_hole_clearance 0.25 its copper must
keep 0.175 mm from copper of another net. The fanout engines priced their via
keep-outs at the copper clearance alone: on an 0.8 mm BGA a 0.12 mm track ran
between two drop vias 0.24 mm from each hole, and the cap pass parked cap pads
0.2 mm from a via hole.

Boards (synthetic, 4 layers, written below with their project):
  * an 8x8 BGA, 0.8 mm pitch, one net per ball, fanned by bga_fanout.py with
    the under-pad and the channel escapes;
  * a QFN-40, 0.4 mm pitch, fanned by qfn_fanout.py with the under-pad escape;
  * the BGA with movable 0402 caps on the back, under its vias, after
    place_fanout_clearance.py;
  * kicad_files/glasgow_revC.kicad_pcb's U30 at the 0.25 it declares, the
    board that found it, fanned and cap-passed as test_bga_underpad_diff does.
Each output is graded by check_drc at the declared 0.25 (the fanout's own
project writeback is not what is under test): no track-hole, via-hole or
pad-hole row, and the part still escapes.

Standalone:  python3 tests/test_fanout_plated_hole_clearance.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from run_utils import run_step, evidence, tool                   # noqa: E402

HC, CL, TW, VS, VD = 0.25, 0.1, 0.12, 0.35, 0.2
HOLE_TYPES = ('track-hole', 'via-hole', 'pad-hole')
LAYERS = ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu']
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

    def fp(self, ref, x, y, pads, layer='F.Cu'):
        self.items.append(f' (footprint "t:{ref}" (layer "{layer}") (at {x} {y})\n'
                          f'  (property "Reference" "{ref}" (at 0 -2) (layer "F.SilkS"))\n'
                          + '\n'.join(pads) + '\n )')

    def pad(self, num, x, y, sx, sy, net, shape='rect', layer='F.Cu'):
        n = self.net(net)
        return (f'  (pad "{num}" smd {shape} (at {x} {y}) (size {sx} {sy}) '
                f'(layers "{layer}") (net {n} "{net}"))')

    def bga(self, ref, cx, cy, n=8, pitch=0.8):
        off = (n - 1) * pitch / 2.0
        rows = 'ABCDEFGHJK'
        pads = [self.pad(f'{rows[i]}{j + 1}', round(j * pitch - off, 4),
                         round(i * pitch - off, 4), 0.35, 0.35,
                         f'{ref}_{rows[i]}{j + 1}', shape='circle')
                for i in range(n) for j in range(n)]
        self.fp(ref, cx, cy, pads)

    def qfn(self, ref, cx, cy, per_side=8, pitch=0.5, body=5.0):
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

    def partners(self, ref, y, w=20):
        """One pad per net on a row at `y`, so every net is a connection to
        make (a one-pad net is not fanned)."""
        nets = self.nets[1:]
        step = (w - 2.0) / max(1, len(nets))
        self.fp(ref, 0, y, [self.pad(str(k + 1), round(1.0 + k * step, 4), 0, 0.15, 0.4, n)
                            for k, n in enumerate(nets)])

    def cap(self, ref, x, y, net_a, net_b):
        self.fp(ref, x, y, [self.pad('1', -0.48, 0, 0.56, 0.62, net_a, layer='B.Cu'),
                            self.pad('2', 0.48, 0, 0.56, 0.62, net_b, layer='B.Cu')],
                layer='B.Cu')

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
                    "name": "Default", "clearance": CL, "track_width": TW,
                    "via_diameter": VS, "via_drill": VD}]}}
        with open(os.path.splitext(path)[0] + '.kicad_pro', 'w') as fh:
            json.dump(proj, fh)
        return path


def grade(board, tmp):
    """check_drc's hole rows on `board` at the declared floor, and its total."""
    out = os.path.join(tmp, os.path.basename(board) + '.drc.json')
    run_step([tool('check_drc.py'), board, '--clearance', CL, '--hole-clearance', HC,
              '--json', out], outputs=[out], findings_ok=True)
    doc = json.load(open(out, encoding='utf-8'))
    by_type = doc.get('by_type', {})
    return {t: by_type.get(t, 0) for t in HOLE_TYPES}, sum(by_type.values())


def summary(output):
    rows = [l for l in output.splitlines() if l.startswith('JSON_SUMMARY')]
    return json.loads(rows[-1].split(':', 1)[1]) if rows else {}


def bga_case(tmp, method):
    print(f'BGA 8x8 / 0.8 mm, --escape-method {method}:')
    b = Board()
    b.bga('U1', 10, 10)
    b.partners('J1', 19)
    src = b.write(os.path.join(tmp, f'bga_{method}.kicad_pcb'))
    out = os.path.join(tmp, f'bga_{method}_fan.kicad_pcb')
    _rc, log = run_step([tool('bga_fanout.py'), src, '--component', 'U1',
                         '--escape-method', method, '--track-width', TW, '--clearance', CL,
                         '--via-size', VS, '--via-drill', VD, '--layers', *LAYERS,
                         '--output', out], outputs=[out])
    s = summary(log)
    rows, _total = grade(out, tmp)
    check(f'{method}: no copper inside the hole floor', not any(rows.values()), str(rows))
    check(f'{method}: the part still escapes', s.get('requested', 0) > 0
          and s.get('escaped', 0) >= 0.8 * s['requested'],
          f"{s.get('escaped')}/{s.get('requested')}")
    return out


def qfn_case(tmp, drill, share):
    """With 0.35/0.2 vias the staggered drops sat 0.225 mm hole to copper, and
    a stub laid one 0.4 mm pitch beside a neighbour's via is 0.24 mm from its
    hole, so every other pad cannot escape at all: those are reported, not
    shipped. With a 0.15 drill (a 0.1 ring) the whole part fits the floor."""
    print(f'QFN-40 / 0.4 mm, {VS}/{drill} vias, --escape-method underpad:')
    b = Board()
    b.qfn('U2', 10, 10, per_side=10, pitch=0.4)
    b.partners('J1', 19)
    src = b.write(os.path.join(tmp, f'qfn{drill}.kicad_pcb'))
    out = os.path.join(tmp, f'qfn{drill}_fan.kicad_pcb')
    _rc, log = run_step([tool('qfn_fanout.py'), src, '--component', 'U2',
                         '--escape-method', 'underpad', '--width', TW, '--clearance', CL,
                         '--via-size', VS, '--via-drill', drill, '--output', out], outputs=[out])
    rows, _total = grade(out, tmp)
    check(f'qfn {drill}: no copper inside the hole floor', not any(rows.values()), str(rows))
    s = summary(log)
    check(f'qfn {drill}: at least {share:.0%} of the part escapes, the rest reported',
          s.get('requested', 0) > 0 and s.get('escaped', 0) >= share * s['requested']
          and s.get('escaped', 0) + s.get('failed', 0) == s['requested'],
          f"{s.get('escaped')}/{s.get('requested')}, {s.get('failed')} failed")


def cap_case(tmp):
    print('BGA + movable back-side caps, then place_fanout_clearance:')
    b = Board()
    b.bga('U1', 10, 10)
    for k, (x, y) in enumerate([(8.8, 8.4), (10.4, 9.2), (11.2, 10.8), (8.4, 11.6)]):
        b.cap(f'C{k + 1}', x, y, 'U1_C3', f'U1_F{k + 3}')
    b.partners('J1', 19)
    src = b.write(os.path.join(tmp, 'cap.kicad_pcb'))
    fan = os.path.join(tmp, 'cap_fan.kicad_pcb')
    run_step([tool('bga_fanout.py'), src, '--component', 'U1', '--escape-method', 'underpad',
              '--track-width', TW, '--clearance', CL, '--via-size', VS, '--via-drill', VD,
              '--layers', *LAYERS, '--output', fan], outputs=[fan])
    out = os.path.join(tmp, 'cap_opt.kicad_pcb')
    run_step([tool('place_fanout_clearance.py'), fan, out, '--clearance', CL], outputs=[out])
    rows, total = grade(out, tmp)
    check('cap pass: no cap pad inside a via hole floor', rows['pad-hole'] == 0, str(rows))
    check('cap pass: the board is DRC-clean', total == 0, f'{total} violation(s)')


def glasgow_case(tmp):
    """The real board that found it: glasgow_revC declares min_hole_clearance
    0.25, and its U30 (BGA-121, 0.8 mm) escaped under the pads with 0.35/0.2
    vias, then the cap pass, shipped 158 track-hole and 13 pad-hole rows at
    that floor. Balls the floor leaves no room for are reported, not shipped."""
    print('glasgow_revC U30 at its declared 0.25, under-pad escape + cap pass:')
    board = evidence(os.path.join(REPO, 'kicad_files', 'glasgow_revC.kicad_pcb'),
                     'the glasgow_revC fixture')
    fan = os.path.join(tmp, 'glasgow_fan.kicad_pcb')
    _rc, log = run_step([tool('bga_fanout.py'), board, '--component', 'U30',
                         '--escape-method', 'underpad', '--diff-pairs', '*',
                         '--track-width', TW, '--clearance', CL, '--via-size', VS,
                         '--via-drill', VD, '--diff-pair-gap', 0.101,
                         '--layers', *LAYERS, '--output', fan], outputs=[fan])
    out = os.path.join(tmp, 'glasgow_cap.kicad_pcb')
    run_step([tool('place_fanout_clearance.py'), fan, out, '--clearance', CL], outputs=[out])
    rows, _total = grade(out, tmp)
    check('glasgow: no copper inside the hole floor', not any(rows.values()), str(rows))
    s = summary(log)
    check('glasgow: at least 90% of U30 escapes, the rest reported',
          s.get('requested', 0) > 0 and s.get('escaped', 0) >= 0.9 * s['requested']
          and s.get('escaped', 0) + s.get('failed', 0) == s['requested'],
          f"{s.get('escaped')}/{s.get('requested')}, {s.get('failed')} failed")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bga_case(tmp, 'underpad')
        bga_case(tmp, 'channel')
        bga_case(tmp, 'dogbone')
        qfn_case(tmp, VD, 0.5)
        qfn_case(tmp, 0.15, 1.0)
        cap_case(tmp)
        glasgow_case(tmp)
    print()
    if fails:
        print(f'FAILED: {len(fails)} check(s): {fails}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
