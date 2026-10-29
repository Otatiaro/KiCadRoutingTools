#!/usr/bin/env python3
"""A ripped net's saved copper that sat exactly AT the clearance from a
neighbour is restored, not refused as a short (#134 guard).

The #134 guard (`_saved_route_colliders`) compared each pair's distance with a
strict `<` and no tie epsilon. A grid whose step divides track + clearance
(0.09 for a 0.09/0.09 board, 0.045, 0.18) lays two neighbours two cells apart,
which is exactly the clearance; the float distance comes out a hair under it
(0.0899999...), so nearly every restore beside a neighbour was refused, the net
was left ripped and re-routed at the end of the run, and some of those re-routes
failed. On a 0.05 grid the same neighbours sit 0.11 apart and no restore was
refused, which made the aligned grids look worse than they are.

1. Saved copper at exactly the clearance from foreign copper, on the 0.09 /
   0.045 / 0.18 grids (segment-segment, segment-via, via-via): not a collision.
2. The same copper a micron inside the clearance: still a collision.
3. route.py end to end: a crossbar routed at 0.09 that rips and restores
   beside neighbours at the clearance ships copper check_drc and
   check_connected pass.

    python3 tests/test_restore_at_clearance.py
"""
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))  # #522
sys.path.insert(0, os.path.join(ROOT, 'py_tools'))  # #522
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kicad_parser import Segment, Via  # noqa: E402
from rip_up_reroute import _saved_route_collides  # noqa: E402
from routing_config import GridCoord  # noqa: E402

fails = []
TRACK = CLR = 0.09


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        fails.append(name)


def _seg(x1, y1, x2, y2, net, w=TRACK, layer='F.Cu'):
    return Segment(start_x=x1, start_y=y1, end_x=x2, end_y=y2,
                   width=w, layer=layer, net_id=net)


def _via(x, y, net, size=0.45):
    return Via(x=x, y=y, size=size, drill=0.3, layers=['F.Cu', 'B.Cu'], net_id=net)


def _pcb(segments=(), vias=()):
    return SimpleNamespace(segments=list(segments), vias=list(vias))


def _collides(saved_segs, saved_vias, foreign_segs=(), foreign_vias=()):
    return _saved_route_collides({'new_segments': list(saved_segs), 'new_vias': list(saved_vias)},
                                 _pcb(foreign_segs, foreign_vias), [1], clearance=CLR)


def test_tie_is_legal():
    for step in (0.09, 0.045, 0.18):
        c = GridCoord(step)
        pitch = TRACK + CLR                       # two neighbours' centrelines
        k = int(round(pitch / step))
        for gy in (7, 1193, 2389):                # rows whose float coordinates carry noise
            _x0, y = c.to_float(0, gy)
            _x1, y2 = c.to_float(0, gy + k)
            x0, x1 = c.to_float(100, 0)[0], c.to_float(900, 0)[0]
            mine = _seg(x0, y, x1, y, net=1)
            check('step %g row %d: neighbour track at the clearance restores' % (step, gy),
                  not _collides([mine], [], [_seg(x0, y2, x1, y2, net=2)]))
            inside = _seg(x0, y2 - 0.001, x1, y2 - 0.001, net=2)
            check("step %g row %d: 1 um inside the clearance still refuses" % (step, gy),
                  _collides([mine], [], [inside]))
        # a via beside a track, centre at via/2 + clearance + track/2
        vx, vy = c.to_float(400, 31)
        gap = 0.45 / 2 + CLR + TRACK / 2
        ty = vy + gap
        check('step %g: track at the clearance from a restored via restores' % step,
              not _collides([], [_via(vx, vy, net=1)], [_seg(vx - 1, ty, vx + 1, ty, net=2)]))
        check('step %g: via at the clearance from a restored track restores' % step,
              not _collides([_seg(vx - 1, ty, vx + 1, ty, net=1)], [], [], [_via(vx, vy, net=2)]))
        vx2 = vx + 0.45 + CLR
        check('step %g: via at the clearance from a restored via restores' % step,
              not _collides([], [_via(vx, vy, net=1)], [], [_via(vx2, vy, net=2)]))


def _run(argv):
    return subprocess.run([sys.executable, '-X', 'utf8'] + argv, capture_output=True,
                          text=True, cwd=ROOT,
                          env=dict(os.environ, KICAD_PLANE_FINALIZE='0'))


def test_route_restores_at_clearance():
    from synth import RoutingBoard
    # A crossbar: eight nets left to right and eight top to bottom, each
    # set in a scrambled order, through a 3.5 mm square on F.Cu / B.Cu. It
    # rips and restores, with neighbours two 0.09 cells apart -- at exactly
    # the clearance -- and must still ship clean, fully connected copper.
    n, size = 8, 3.5
    b = RoutingBoard(w=size + 4, h=size + 4, layers=('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'))
    p = size / n
    for i in range(n):
        t = 2 + p * (i + 0.5)
        b.pad('H%da' % i, 1.0, t, 'H%d' % i, sx=0.25, sy=0.25)
        b.pad('H%db' % i, size + 3.0, 2 + p * (n - i - 0.5), 'H%d' % i, sx=0.25, sy=0.25)
        b.pad('V%da' % i, t, 1.0, 'V%d' % i, sx=0.25, sy=0.25)
        b.pad('V%db' % i, 2 + p * ((i * 3) % n + 0.5), size + 3.0, 'V%d' % i, sx=0.25, sy=0.25)
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, 'in.kicad_pcb'), os.path.join(d, 'out.kicad_pcb')
        with open(inp, 'w', encoding='utf-8') as f:
            f.write(b.text())
        r = _run(['py_router/route.py', inp, out, '--nets', '*', '--grid-step', '0.09',
                  '--track-width', str(TRACK), '--clearance', str(CLR),
                  '--via-size', '0.45', '--via-drill', '0.3',
                  '--fab-overrides', _overrides(d), '--layers', 'F.Cu', 'B.Cu'])
        check('route.py exit 0 (%d)' % r.returncode, r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return
        ripped = len(re.findall(r'Ripping|ripped', r.stdout))
        check('the scene rips and restores (%d rip lines)' % ripped, ripped > 0)
        drc = _run(['py_router/check_drc.py', out])
        check('check_drc clean', drc.returncode == 0)
        conn = _run(['py_router/check_connected.py', out])
        check('check_connected clean', conn.returncode == 0
              and 'ALL NETS FULLY CONNECTED' in conn.stdout)


def _overrides(d):
    p = os.path.join(d, 'fab.txt')
    with open(p, 'w') as f:
        f.write('clearance = %g\ntrack_width = %g\n' % (CLR, TRACK))
    return p


if __name__ == '__main__':
    test_tie_is_legal()
    test_route_restores_at_clearance()
    print('\n%s' % ('FAILED: %d' % len(fails) if fails else 'ALL PASSED'))
    sys.exit(1 if fails else 0)
