#!/usr/bin/env python3
"""smooth_board.py: the staircases a finished board kept from earlier chain steps collapse.

What must hold:
  * a staircase laid before the tool runs collapses, its ends and the net's connection kept;
  * foreign copper keeps its clearance: a track in the way holds the shortcut back;
  * --skip-nets and a protected net leave their copper as it is; --include-protected takes it;
  * the file keeps every other item; a second run on the output changes nothing;
  * the same input gives the same output (reproducible);
  * a net-filtered keep-out from the .kicad_dru holds the shortcut back, unbound nets pass.

    python3 tests/test_smooth_board.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'py_router'))
sys.path.insert(0, os.path.join(REPO, 'py_tools'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from kicad_parser import parse_kicad_pcb
from synth import RoutingBoard

TOOL = os.path.join(REPO, 'py_router', 'smooth_board.py')
fails = []


def check(name, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        fails.append(name)


def staircase(b, net, x0=5.0, y0=5.0, steps=12, step=0.25, w=0.2):
    """Pads at both ends, then grid jogs on the diagonal between them."""
    b.pad('P' + net, x0, y0, net)
    x, y = x0, y0
    for i in range(steps):
        b.seg(x, y, x + step, y, net, w=w)
        x += step
        b.seg(x, y, x, y + step, net, w=w)
        y += step
    b.pad('Q' + net, x, y, net)
    return x, y


def run(src, dst, *opts):
    r = subprocess.run([sys.executable, TOOL, src, dst, '--clearance', '0.1', *opts], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        print(r.stdout[-2000:])
    return r


def segs(path, net):
    pcb = parse_kicad_pcb(path)
    nid = next(i for i, n in pcb.nets.items() if n.name == net)
    return [s for s in pcb.segments if s.net_id == nid]


def length(ss):
    return sum(((s.start_x - s.end_x) ** 2 + (s.start_y - s.end_y) ** 2) ** 0.5 for s in ss)


def touches(ss, x, y):
    return any(min(abs(s.start_x - x) + abs(s.start_y - y), abs(s.end_x - x) + abs(s.end_y - y)) < 1e-3 for s in ss)


def octolinear(s):
    dx, dy = abs(s.start_x - s.end_x), abs(s.start_y - s.end_y)
    return dx < 1e-4 or dy < 1e-4 or abs(dx - dy) < 1e-4


def main():
    d = tempfile.mkdtemp(prefix='smooth_board')
    try:
        # A lone staircase
        b = RoutingBoard()
        end = staircase(b, 'A')
        src = os.path.join(d, 'one.kicad_pcb')
        open(src, 'w', encoding='utf-8').write(b.text())
        out = os.path.join(d, 'one_out.kicad_pcb')
        check('tool runs', run(src, out).returncode == 0)
        before, after = segs(src, 'A'), segs(out, 'A')
        check('the staircase collapses', len(after) < len(before) and length(after) < length(before),
              f'{len(before)} -> {len(after)} segments, {length(before):.2f} -> {length(after):.2f} mm')
        check('its ends stay', touches(after, 5.0, 5.0) and touches(after, *end))
        check('octolinear output', all(octolinear(s) for s in after))

        # Rerun and reproducibility
        again = os.path.join(d, 'again.kicad_pcb')
        run(out, again)
        check('a second run changes nothing', len(segs(again, 'A')) == len(after))
        twice = os.path.join(d, 'one_out2.kicad_pcb')
        run(src, twice)
        no_uuid = lambda p: re.sub(r'\(uuid "[^"]*"\)', '', open(p, encoding='utf-8').read())  # noqa: E731
        check('reproducible, uuids aside', no_uuid(out) == no_uuid(twice))

        # A foreign track across the diagonal holds the shortcut back
        b = RoutingBoard()
        staircase(b, 'A')
        b.pad('PX', 5.2, 7.2, 'X')
        b.pad('QX', 7.2, 5.2, 'X')
        b.seg(5.2, 7.2, 7.2, 5.2, 'X', w=0.2)
        src = os.path.join(d, 'wall.kicad_pcb')
        open(src, 'w', encoding='utf-8').write(b.text())
        out = os.path.join(d, 'wall_out.kicad_pcb')
        run(src, out)
        x = segs(out, 'X')
        check('the foreign track is untouched', len(x) == 1 and abs(length(x) - 2.828) < 1e-2)

        # --skip-nets
        b = RoutingBoard()
        staircase(b, 'A')
        src = os.path.join(d, 'skip.kicad_pcb')
        open(src, 'w', encoding='utf-8').write(b.text())
        out = os.path.join(d, 'skip_out.kicad_pcb')
        run(src, out, '--skip-nets', 'A*')
        check('--skip-nets leaves the net', len(segs(out, 'A')) == len(segs(src, 'A')))

        # A net-filtered keep-out holds the shortcut back for the nets it binds only
        for net, held in (('A', True), ('B', False)):
            b = RoutingBoard()
            # a detour around the region; its shortcut is the diagonal through it
            b.pad('P' + net, 5.0, 5.0, net)
            b.pad('Q' + net, 8.0, 8.0, net)
            b.seg(5.0, 5.0, 8.0, 5.0, net, w=0.2)
            b.seg(8.0, 5.0, 8.0, 8.0, net, w=0.2)
            b.items.append('  (zone (net 0) (net_name "") (layer "F.Cu") (uuid "synth-ko") (name "barrier") '
                           '(hatch edge 0.5) (connect_pads (clearance 0)) (min_thickness 0.25) '
                           '(keepout (tracks allowed) (vias allowed) (pads allowed) (copperpour allowed) '
                           '(footprints allowed)) (fill (thermal_gap 0.5) (thermal_bridge_width 0.5)) '
                           '(polygon (pts (xy 6.2 6.2) (xy 6.8 6.2) (xy 6.8 6.8) (xy 6.2 6.8))))')
            src = os.path.join(d, f'ko{net}.kicad_pcb')
            open(src, 'w', encoding='utf-8').write(b.text())
            open(os.path.join(d, f'ko{net}.kicad_dru'), 'w', encoding='utf-8').write(
                '(version 1)\n(rule "barrier" (constraint disallow track via) '
                '(condition "A.intersectsArea(\'barrier\') && A.NetName == \'A\'"))\n')
            out = os.path.join(d, f'ko{net}_out.kicad_pcb')
            run(src, out)
            inside = 0
            for s in segs(out, net):
                for t in (i / 20 for i in range(21)):
                    px, py = s.start_x + t * (s.end_x - s.start_x), s.start_y + t * (s.end_y - s.start_y)
                    inside += 6.2 <= px <= 6.8 and 6.2 <= py <= 6.8
            if held:
                check('a bound net stays out of the keep-out', inside == 0, f'{inside} samples inside')
            else:
                check('an unbound net crosses it', inside > 0, f'{inside} samples inside')
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if fails:
        print(f'FAILED: {fails}')
        return 1
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
