#!/usr/bin/env python3
"""A ripped net's copper is restored only where its via holes, and the foreign
vias beside it, keep the board's plated hole floor (KiCad's hole_clearance).

The restore guard (rip_up_reroute._saved_route_collides, #134) refuses saved
copper that would hit what moved into its corridor while it was ripped, but
priced every pair at the copper clearance. A shortcut laid past a ripped
net's vacated via (the octolinear smoother, which can only see the copper
that is on the board) then came back beside the restored via at the copper
clearance only: on a 4-layer board at clearance 0.09 with 0.45/0.3 vias and a
declared min_hole_clearance 0.2, a track 0.119 mm from the via's copper,
0.194 mm from its hole, which KiCad reports.

Rows (the geometry of that board, a via of net A at (54.8, 40.1)):
  1. A's via restored beside B's track 0.194 mm from the hole: refused;
  2. the same with the track 0.229 mm from the hole (the copper before the
     shortcut): restored;
  3. B's track restored beside A's via, the same 0.194: refused;
  4. two vias, one restored, 0.18 mm hole to copper: refused; 0.25: restored;
  5. no plated floor declared: the 0.194 case is restored, as before.

Standalone:  python3 tests/test_restore_holds_hole_floor.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'py_router'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from synth import make_pcb, make_seg, make_via, make_net           # noqa: E402
from routing_config import GridRouteConfig                         # noqa: E402
from rip_up_reroute import _saved_route_collides                   # noqa: E402

A, B = 1, 2
VX, VY = 54.8, 40.1
fails = []


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f" -- {detail}" if detail else ''))
    if not cond:
        fails.append(name)


def config(floor):
    cfg = GridRouteConfig(layers=['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'], track_width=0.09,
                          clearance=0.09, via_size=0.45, via_drill=0.3, grid_step=0.05)
    cfg.plated_hole_clearance = floor
    return cfg


def board(items):
    return make_pcb(nets={A: make_net(A, 'A'), B: make_net(B, 'B')},
                    vias=[i for i in items if hasattr(i, 'drill')],
                    segments=[i for i in items if not hasattr(i, 'drill')])


def track(start_x):
    """B's diagonal from (start_x, 40.1) to (58.05, 36.3)-ish, 0.09 wide."""
    return make_seg(start_x, 40.1, start_x + 3.8, 36.3, net_id=B, width=0.09)


def restores(saved, on_board, floor):
    pcb = board(on_board)
    return not _saved_route_collides(saved, pcb, [saved_net(saved)], 0.09, config=config(floor))


def saved_net(saved):
    items = (saved.get('new_segments') or []) + (saved.get('new_vias') or [])
    return items[0].net_id


def main():
    via = make_via(VX, VY, net_id=A, size=0.45, drill=0.3)
    print('1-2. a restored via beside a foreign track:')
    check('0.194 mm from the hole: refused',
          not restores({'new_vias': [via]}, [track(54.25)], 0.2))
    check('0.229 mm from the hole: restored',
          restores({'new_vias': [via]}, [track(54.2)], 0.2))
    print('3. a restored track beside a foreign via:')
    check('0.194 mm from the hole: refused',
          not restores({'new_segments': [track(54.25)]}, [via], 0.2))
    print('4. a restored via beside a foreign via:')
    for gap, ok in ((0.18, False), (0.25, True)):
        # centre distance for a hole-to-copper gap `gap`: drill/2 + gap + size/2
        other = make_via(VX + 0.15 + gap + 0.225, VY, net_id=B, size=0.45, drill=0.3)
        check(f'{gap} mm hole to copper: {"restored" if ok else "refused"}',
              restores({'new_vias': [via]}, [other], 0.2) == ok)
    print('5. no plated floor declared:')
    check('0.194 mm from the hole: restored, as before',
          restores({'new_vias': [via]}, [track(54.25)], 0.0))
    print()
    if fails:
        print(f'FAILED: {len(fails)} check(s): {fails}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
