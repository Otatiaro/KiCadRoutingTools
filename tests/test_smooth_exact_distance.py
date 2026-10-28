#!/usr/bin/env python3
"""The smoother's clearance probes measure foreign tracks exactly and do not
re-count the board's pads on every probe.

`make_segment_clears` asked `_seg_foreign_seg_dist` for its sampled sweep:
one point every 0.02 mm along the probe against every foreign track in the
window, so a probe cost grew with its length (a 30 mm shortcut: 1,500
points), and every probe re-summed the pad count of the whole board twice
(`_foreign_pad_arrays`, `_foreign_hole_caps`) to validate caches the pass
never invalidates.

What must hold (1 and 2 fail on the parent):
  1. clears() takes the exact segment-to-segment distance (the #1318 rule for
     a caller that admits new copper);
  2. during smooth_octolinear_chains the pad count is summed a handful of
     times (cache builds), not once per probe;
  3. a foreign pad added AFTER the pad caches were built still blocks a
     shortcut: the pass rebuilds them before it trusts them;
  4. on a board of staircases between foreign tracks, the smoothed copper is
     the same with the sampled sweep as with the exact distance.

    python3 tests/test_smooth_exact_distance.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'py_router'))

import single_ended_routing as ser
from kicad_parser import Pad, Net, PCBData, BoardInfo, Segment
from pcb_modification import smooth_octolinear_chains

W = 0.15
CLR = 0.1
fails = []


def check(name, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        fails.append(name)


class CountingPads(dict):
    """pads_by_net that counts full scans (the pad-count signature)."""
    scans = 0

    def values(self):
        CountingPads.scans += 1
        return super().values()


def _stair(x0, y0, n=8, step=0.5):
    pts = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        x += step
        pts.append((x, y))
        y += step
        pts.append((x, y))
    return pts


def _poly(points, net_id, w=W):
    return [Segment(start_x=a[0], start_y=a[1], end_x=b[0], end_y=b[1],
                    width=w, layer='F.Cu', net_id=net_id)
            for a, b in zip(points, points[1:])]


def _pad(x, y, net_id, size=0.6):
    return Pad(component_ref='U9', pad_number=str(net_id), global_x=x, global_y=y,
               local_x=0.0, local_y=0.0, size_x=size, size_y=size, shape='rect',
               layers=['F.Cu'], net_id=net_id, net_name=f'/P{net_id}', drill=0)


def board(stairs=6, pads_cls=dict):
    """Staircases (nets 1..stairs) side by side, a foreign track (net 99)
    between each pair, and a pad of net 98 near the first staircase."""
    nets = {98: Net(98, '/PAD'), 99: Net(99, '/WALL')}
    segs = []
    for k in range(stairs):
        nid = k + 1
        nets[nid] = Net(nid, f'/S{nid}')
        x0 = 10.0 + 6.0 * k
        segs += _poly(_stair(x0, 10.0), nid)
        # a foreign wall 0.45 mm right of the staircase's diagonal at mid-height
        segs += _poly([(x0 + 2.6, 11.0), (x0 + 4.6, 13.0)], 99)
    pads = pads_cls({98: [_pad(60.0, 60.0, 98)]})
    pcb = PCBData(footprints={}, nets=nets, segments=segs, vias=[],
                  board_info=BoardInfo(layers={}, copper_layers=['F.Cu', 'B.Cu'],
                                       board_bounds=(0.0, 0.0, 200.0, 200.0)),
                  pads_by_net=pads)
    return pcb, [{'new_segments': [s for s in segs if s.net_id != 99], 'new_vias': []}]


def key(pcb):
    return sorted((s.net_id, round(s.start_x, 6), round(s.start_y, 6),
                   round(s.end_x, 6), round(s.end_y, 6)) for s in pcb.segments)


def main():
    real = ser._seg_foreign_seg_dist

    # 1. exact distance
    seen = []

    def spy(*a, **k):
        seen.append(k.get('exact'))
        return real(*a, **k)
    ser._seg_foreign_seg_dist = spy
    try:
        pcb, results = board()
        smooth_octolinear_chains(results, pcb, clearance=CLR)
    finally:
        ser._seg_foreign_seg_dist = real
    check('1. clears() probes foreign tracks with the exact distance',
          seen and all(e is True for e in seen),
          f'{len(seen)} probes, exact={sorted(set(map(str, seen)))}')

    # 2. pad count not re-summed per probe
    CountingPads.scans = 0
    pcb, results = board(pads_cls=CountingPads)
    smooth_octolinear_chains(results, pcb, clearance=CLR)
    check('2. the pad count is summed per cache build, not per probe',
          CountingPads.scans <= 10, f'{CountingPads.scans} scans for {len(seen)} probes')

    # 3. a pad added after the caches were built still blocks
    pcb, results = board(stairs=1)
    ser._foreign_pad_arrays(pcb, 'F.Cu')          # caches built on the old pads
    ser._seg_foreign_hole_dist(pcb, 1, 0.0, 0.0, 1.0, 1.0)
    # just above-left of the staircase's diagonal: blocks it, clears the stairs
    pcb.pads_by_net[98].append(_pad(11.65, 12.0, 98, size=0.3))
    smooth_octolinear_chains(results, pcb, clearance=CLR)
    clash = min(ser._seg_foreign_pad_dist(pcb, 1, s.start_x, s.start_y, s.end_x, s.end_y,
                                          'F.Cu')
                for s in pcb.segments if s.net_id == 1)
    check('3. a pad added after the cache was built keeps the shortcut off it',
          clash >= CLR + W / 2 - 1e-4, f'closest {clash:.3f} mm')

    # 4. same copper with the sampled sweep
    pcb_e, res_e = board()
    smooth_octolinear_chains(res_e, pcb_e, clearance=CLR)
    ser._seg_foreign_seg_dist = lambda *a, **k: real(*a, **{**k, 'exact': None})
    try:
        pcb_s, res_s = board()
        smooth_octolinear_chains(res_s, pcb_s, clearance=CLR)
    finally:
        ser._seg_foreign_seg_dist = real
    changed = sum(1 for s in pcb_e.segments if s.net_id != 99) != sum(
        1 for s in board()[0].segments if s.net_id != 99)
    check('4. sampled sweep and exact distance smooth to the same copper',
          key(pcb_e) == key(pcb_s) and changed)

    if fails:
        print(f'FAIL ({len(fails)})')
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
