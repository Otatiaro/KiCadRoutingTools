#!/usr/bin/env python3
"""A rectangular board's edge keep-out is measured from the TRUE edge.

`_add_rectangular_edge_obstacles` counted its bands in whole cells from the
edge's ROUNDED cell (`GridCoord.to_grid`). An edge off the grid -- 3.08 mm on a
0.05 grid rounds out to 3.10, 223.52 on a 0.09 grid lies 0.04 past its cell --
left the first free row up to half a cell inside the edge clearance, and
route.py ran a track 0.015 mm into it (SEGMENT-BOARD-EDGE). The polygon path
already measured true distances; this pins the rectangle to the same rule.

1. Off-grid edges, several grid steps: no free track / via cell is closer to
   the true edge than its clearance, and the first free one is within a cell
   of it (the band does not over-block).
2. On-grid edges: the bands are the integer bands, cell for cell.
3. End to end: a corridor under a barrier whose only row lies inside the edge
   clearance. route.py must not take it: check_drc and check_connected clean.

    python3 tests/test_rect_edge_band_true_edge.py
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))  # #522
sys.path.insert(0, os.path.join(ROOT, 'py_tools'))  # #522
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obstacle_map import (GridObstacleMap, add_board_edge_obstacles,  # noqa: E402
                          _add_rectangular_edge_obstacles)
from routing_config import GridCoord, GridRouteConfig  # noqa: E402
from kicad_parser import BoardInfo  # noqa: E402
from synth import RoutingBoard, make_pcb  # noqa: E402

fails = []
EDGE, TRACK, VIA = 0.2, 0.09, 0.45
T_CLR, V_CLR = EDGE + TRACK / 2, EDGE + VIA / 2


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        fails.append(name)


def _bands(bounds, step, integer=False):
    """The edge bands route.py stamps for a rectangular board of `bounds`
    (`add_board_edge_obstacles`), or, `integer`, the whole-cell bands counted
    from the rounded edge cells."""
    coord = GridCoord(step)
    gmin_x, gmin_y = coord.to_grid(bounds[0], bounds[1])
    gmax_x, gmax_y = coord.to_grid(bounds[2], bounds[3])
    ob = GridObstacleMap(1)
    if integer:
        _add_rectangular_edge_obstacles(ob, coord, 1, gmin_x, gmin_y, gmax_x, gmax_y,
                                        coord.to_grid_dist(T_CLR),
                                        coord.to_grid_dist_safe(V_CLR))
    else:
        pcb = make_pcb(board_info=BoardInfo(layers={0: "F.Cu"}, copper_layers=["F.Cu"],
                                            board_bounds=bounds))
        cfg = GridRouteConfig(grid_step=step, layers=['F.Cu'], track_width=TRACK,
                              clearance=0.09, via_size=VIA, board_edge_clearance=EDGE)
        add_board_edge_obstacles(ob, pcb, cfg)
    return ob, coord, (gmin_x, gmin_y, gmax_x, gmax_y)


def _nearest_free(ob, step, bounds, g):
    """Smallest true distance to the edge of a free track / via cell, probed
    across the middle of each of the four edges."""
    gmin_x, gmin_y, gmax_x, gmax_y = g
    mx, my = (gmin_x + gmax_x) // 2, (gmin_y + gmax_y) // 2
    probes = ([(gx, my, gx * step - bounds[0]) for gx in range(gmin_x - 2, gmin_x + 25)]
              + [(gx, my, bounds[2] - gx * step) for gx in range(gmax_x - 25, gmax_x + 2)]
              + [(mx, gy, gy * step - bounds[1]) for gy in range(gmin_y - 2, gmin_y + 25)]
              + [(mx, gy, bounds[3] - gy * step) for gy in range(gmax_y - 25, gmax_y + 2)])
    t = min((d for gx, gy, d in probes if d > 0 and not ob.is_blocked(gx, gy, 0)), default=None)
    v = min((d for gx, gy, d in probes if d > 0 and not ob.is_via_blocked(gx, gy)), default=None)
    return t, v


def test_off_grid_edges():
    for bounds in [(0.0, 0.0, 12.0, 3.08), (25.4, 25.4, 223.52, 55.88),
                   (30.0, 30.00236, 75.99764, 76.0), (10.03, 7.013, 40.071, 31.0)]:
        for step in (0.025, 0.045, 0.05, 0.09, 0.1, 0.18):
            ob, _c, g = _bands(bounds, step)
            t, v = _nearest_free(ob, step, bounds, g)
            tag = '%s @ %g' % (bounds, step)
            check('%s: free track cell %.4f >= %.3f' % (tag, t, T_CLR), t >= T_CLR - 1e-9)
            check('%s: free via cell %.4f >= %.3f' % (tag, v, V_CLR), v >= V_CLR - 1e-9)
            check('%s: bands tight (within a cell)' % tag,
                  t < T_CLR + step + 1e-9 and v < V_CLR + step + 1e-9)


def test_on_grid_edges_unchanged():
    bounds = (10.0, 5.0, 40.0, 25.0)
    for step in (0.025, 0.05, 0.1):
        new, _c, (x0, y0, x1, y1) = _bands(bounds, step)
        old, _c, _g = _bands(bounds, step, integer=True)
        diff = [(gx, gy) for gx in range(x0 - 3, x1 + 4) for gy in (y0 - 3, y0 + 7, y0 + 12, y1 - 12, y1 + 3)
                if new.is_blocked(gx, gy, 0) != old.is_blocked(gx, gy, 0)
                or new.is_via_blocked(gx, gy) != old.is_via_blocked(gx, gy)]
        diff += [(gx, gy) for gy in range(y0 - 3, y1 + 4) for gx in (x0 + 2, x0 + 9, x0 + 14, x1 - 14, x1 - 2)
                 if new.is_blocked(gx, gy, 0) != old.is_blocked(gx, gy, 0)
                 or new.is_via_blocked(gx, gy) != old.is_via_blocked(gx, gy)]
        check('on-grid edges @ %g: same bands as the integer sweep (%d cells differ)'
              % (step, len(diff)), not diff)


def _run(argv):
    return subprocess.run([sys.executable, '-X', 'utf8'] + argv, capture_output=True,
                          text=True, cwd=ROOT)


def test_route_avoids_off_grid_corridor():
    # Bottom edge 3.08 rounds out to 3.10 on the 0.05 grid. The barrier's end
    # leaves one F.Cu row under it, y = 2.85, which is 0.23 mm from the edge
    # (< 0.245). The route must go round it on another layer.
    b = RoutingBoard(w=12, h=3.08, layers=('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'))
    b.pad('R1', 3, 1.5, 'A', sx=0.4, sy=0.4)
    b.pad('R2', 9, 1.5, 'A', sx=0.4, sy=0.4)
    b.seg(6, 0.5, 6, 2.5, 'W', w=0.4, locked=True)
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, 'in.kicad_pcb'), os.path.join(d, 'out.kicad_pcb')
        with open(inp, 'w', encoding='utf-8') as f:
            f.write(b.text())
        r = _run(['py_router/route.py', inp, out, '--nets', 'A', '--grid-step', '0.05',
                  '--track-width', str(TRACK), '--clearance', '0.09',
                  '--board-edge-clearance', str(EDGE), '--via-size', str(VIA), '--via-drill', '0.3'])
        check('route.py exit 0 (%d)' % r.returncode, r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return
        drc = _run(['py_router/check_drc.py', out])
        check('check_drc clean (no SEGMENT-BOARD-EDGE)',
              drc.returncode == 0 and 'BOARD-EDGE' not in drc.stdout)
        conn = _run(['py_router/check_connected.py', out])
        check('check_connected clean', conn.returncode == 0
              and 'ALL NETS FULLY CONNECTED' in conn.stdout)


if __name__ == '__main__':
    test_off_grid_edges()
    test_on_grid_edges_unchanged()
    test_route_avoids_off_grid_corridor()
    print('\n%s' % ('FAILED: %d' % len(fails) if fails else 'ALL PASSED'))
    sys.exit(1 if fails else 0)
