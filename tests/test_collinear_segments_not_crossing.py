#!/usr/bin/env python3
"""Collinear segments far apart are not reported as crossing.

`rip_up_reroute._segs_cross` decided a proper crossing from the signs of four
orientations. On COLLINEAR segments all four are zero up to rounding, so their
signs are noise: a restored diagonal track (25, 12.4)-(15.05, 22.35) and a
foreign stub (12.1, 25.3)-(12.05, 25.35) on the same 45-degree line, 4.17 mm
apart, were reported crossing -- `_seg_seg_dist_sq` returned 0 -- and the #134
restore test refused to put a ripped net back for a short that did not exist.
Orientations within 1e-9 mm^2 of zero are now no side at all, and collinear or
touching pairs take the exact endpoint distances.

Checks:
  1. the recorded pair: distance 4.1719 mm, not 0 (0 on the parent);
  2. random collinear disjoint pairs on 45-degree and arbitrary bearings: the
     distance is the gap along the line (the parent reports 0 for some);
  3. proper crossings, T-junctions, shared endpoints and overlapping collinear
     pairs still measure 0, and parallel offset pairs their offset;
  4. `_saved_route_collides` restores the recorded track next to the stub.

    python3 tests/test_collinear_segments_not_crossing.py
"""
import math
import os
import random
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, TESTS_DIR)

from routing_config import GridRouteConfig                  # noqa: E402
from synth import make_seg, make_pcb                        # noqa: E402
import rip_up_reroute as rr                                 # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def dist(a, b):
    return math.sqrt(rr._seg_seg_dist_sq(*a, *b))


REST = (25.0, 12.4, 15.05, 22.35)
STUB = (12.100000000000001, 25.3, 12.05, 25.35)


def main():
    d = dist(REST, STUB)
    check("the recorded collinear pair measures its 4.17 mm gap", abs(d - 4.171930009) < 1e-6,
          f"{d:.6f} mm")

    rnd = random.Random(2)
    bad = []
    for i in range(2000):
        x0, y0 = rnd.uniform(0, 150), rnd.uniform(0, 110)
        if rnd.random() < 0.5:
            ux, uy = rnd.choice([(1, 1), (1, -1), (-1, 1)])
            ux, uy = ux / math.sqrt(2), uy / math.sqrt(2)
        else:
            a = rnd.uniform(0, 2 * math.pi)
            ux, uy = math.cos(a), math.sin(a)
        l1, gap, l2 = rnd.uniform(0.05, 15), rnd.uniform(0.01, 5), rnd.uniform(0.05, 2)
        a_seg = (x0, y0, x0 + l1 * ux, y0 + l1 * uy)
        s = l1 + gap
        b_seg = (x0 + s * ux, y0 + s * uy, x0 + (s + l2) * ux, y0 + (s + l2) * uy)
        if abs(dist(a_seg, b_seg) - gap) > 1e-6:
            bad.append((i, dist(a_seg, b_seg), gap))
    check("collinear disjoint pairs measure their gap", not bad,
          f"{len(bad)}/2000 wrong, e.g. {bad[:2]}")

    cases = [
        ("proper crossing", (0, 0, 2, 2), (0, 2, 2, 0), 0.0),
        ("T-junction", (0, 0, 2, 0), (1, 0, 1, 1), 0.0),
        ("shared endpoint", (0, 0, 1, 1), (1, 1, 2, 0), 0.0),
        ("overlapping collinear", (0, 0, 2, 2), (1, 1, 3, 3), 0.0),
        ("parallel, 0.3 apart", (0, 0, 2, 0), (0, 0.3, 2, 0.3), 0.3),
        ("near-miss crossing", (0, 0, 2, 0), (1, 0.001, 1, 1), 0.001),
    ]
    for label, a, b, want in cases:
        got = dist(a, b)
        check(label, abs(got - want) < 1e-9, f"{got:.9f}, want {want}")

    cfg = GridRouteConfig(clearance=0.09, track_width=0.3, via_size=0.45, via_drill=0.3,
                          layers=['F.Cu', 'B.Cu'], grid_step=0.05)
    pcb = make_pcb(segments=[make_seg(*STUB, net_id=326, width=0.3)])
    # the rest of the saved route reaches past the stub, so the stub is inside
    # the prefilter box (as on the board it was recorded from)
    saved = {'new_segments': [make_seg(*REST, net_id=320, width=0.3),
                              make_seg(15.05, 22.35, 10.0, 30.0, net_id=320, width=0.3)],
             'new_vias': []}
    check("a saved route collinear with the stub, 0.84 mm from it at best, is restorable",
          not rr._saved_route_collides(saved, pcb, [320], 0.09, config=cfg))

    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
