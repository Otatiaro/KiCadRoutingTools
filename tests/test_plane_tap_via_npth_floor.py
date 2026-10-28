#!/usr/bin/env python3
"""The plane-tap via map holds VIA copper off NPTH holes at the board's
copper-to-hole floor, as the signal map and the plane connector map do.

`plane_obstacle_builder.build_via_obstacle_map` is the map behind every
via a plane step places on its own: repair_planes' pad taps (and route.py's
in-run plane finalize, which runs them), the rip-and-retry tap path, and
route_planes' stitching and pad vias. It held a via's DRILL off an NPTH hole
at the hole-to-hole minimum and its copper off the hole's mask-sized pad at
copper clearance, but never applied the board's `min_hole_clearance` to the
via's copper -- the #1038 arm `obstacle_map.block_via_cells_near_npth_floors`
gives the signal map and the plane connector map. Found on a 4-layer board
routed at 0.09 with 0.45/0.3 vias and a declared 0.2: a GND tap via landed
0.182 mm from a connector's NPTH locating hole, which KiCad reports as
hole_clearance.

Geometry: NPTH hole at (15,15), drill 2.1 (wall at r=1.05), via 0.55/0.254,
copper clearance 0.09, h2h 0.25, 0.1 grid.
  * the pad-copper stamp reaches 1.05 + 0.275 + 0.09 (+ half-cell cushion)
    ~ 1.465 mm, the drill stamp is a disc of 15 cells (1.5 mm);
  * declared 0.25 -> via copper needs 1.05 + 0.25 + 0.275 = 1.575: a via at
    the cell (+1.3, +0.8), 1.526 mm off the centre (copper 0.20 off the wall),
    must be blocked; one at 1.6 mm stays free;
  * nothing declared -> that cell stays free (raise-only);
  * the hole nominally on the plane net changes nothing;
  * over a window around the hole, the tap map blocks every cell the signal
    map's NPTH via arms block.

    python3 tests/test_plane_tap_via_npth_floor.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from kicad_parser import parse_kicad_pcb  # noqa: E402
from obstacle_map import GridObstacleMap, add_drill_hole_obstacles  # noqa: E402
from plane_obstacle_builder import build_via_obstacle_map  # noqa: E402
from routing_config import GridRouteConfig, GridCoord  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f"  {detail}" if detail else ''))
    if not cond:
        fails.append(name)


def _rect_edge(x1, y1, x2, y2):
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return "\n".join(
        f' (gr_line (start {pts[i][0]} {pts[i][1]}) '
        f'(end {pts[(i + 1) % 4][0]} {pts[(i + 1) % 4][1]}) '
        f'(layer "Edge.Cuts") (width 0.1))' for i in range(4))


def board(hole_net=0):
    net_tok = f' (net {hole_net} "GND")' if hole_net else ''
    return f'''(kicad_pcb
 (version 20221018)
 (net 0 "")
 (net 1 "GND")
 (net 2 "SIG")
{_rect_edge(0, 0, 30, 30)}
 (footprint "t:mh" (layer "F.Cu") (at 15 15)
  (property "Reference" "MH1" (at 0 -2) (layer "F.SilkS"))
  (pad "" np_thru_hole circle (at 0 0) (size 2.1 2.1) (drill 2.1)
       (layers "F&B.Cu" "*.Mask"){net_tok})
 )
 (segment (start 2 2) (end 4 2) (width 0.2) (layer "F.Cu") (net 2) (uuid "s"))
)'''


def cfg(hole_clr=0.0):
    c = GridRouteConfig(layers=["F.Cu", "B.Cu"], grid_step=0.1, clearance=0.09,
                        track_width=0.2, via_size=0.55, via_drill=0.254,
                        hole_to_hole_clearance=0.25)
    c.hole_clearance = hole_clr
    return c


def tap_map(pcb, c, plane_net=1):
    return build_via_obstacle_map(pcb, c, plane_net, verbose=False)


def blocked(obs, c, xy):
    gx, gy = GridCoord(c.grid_step).to_grid(*xy)
    return obs.is_via_blocked(gx, gy)


def parse(txt):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'b.kicad_pcb')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(txt)
        return parse_kicad_pcb(p)


FLOOR = (16.3, 15.8)  # 1.526 mm off the hole centre: copper 0.20 off the wall

# 1. the declared floor holds the tap via's copper off the wall
pcb = parse(board())
c = cfg(0.25)
obs = tap_map(pcb, c)
check("declared 0.25: a tap via at 1.526 mm (copper 0.20 off the wall) is "
      "blocked", blocked(obs, c, FLOOR))
check("declared 0.25: a tap via at 1.6 mm stays free",
      not blocked(obs, c, (16.6, 15.0)))
sig = GridObstacleMap(len(c.layers))
add_drill_hole_obstacles(sig, pcb, c, set())
coord = GridCoord(c.grid_step)
g0x, g0y = coord.to_grid(12.4, 12.4)
g1x, g1y = coord.to_grid(17.6, 17.6)
missing = [(gx, gy) for gx in range(g0x, g1x + 1) for gy in range(g0y, g1y + 1)
           if sig.is_via_blocked(gx, gy) and not obs.is_via_blocked(gx, gy)]
check("declared 0.25: the tap map blocks every cell the signal map's NPTH "
      "via arms block", not missing, f"{len(missing)} cell(s), e.g. {missing[:3]}")

# 2. raise-only: nothing declared
c = cfg()
check("nothing declared: the tap via at 1.526 mm stays free",
      not blocked(tap_map(pcb, c), c, FLOOR))

# 3. net-independent: the hole nominally on the plane net
pcb = parse(board(hole_net=1))
c = cfg(0.25)
check("declared 0.25, GND-tied NPTH, GND tapped: still blocked",
      blocked(tap_map(pcb, c), c, FLOOR))

print("-" * 60)
if fails:
    print(f"{len(fails)} FAILED: " + "; ".join(fails))
    sys.exit(1)
print("ALL PASS")
