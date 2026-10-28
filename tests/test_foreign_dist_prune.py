#!/usr/bin/env python3
"""The foreign-copper distance sweeps skip items that cannot hold the minimum.

_seg_foreign_seg_dist and _seg_foreign_pad_dist window foreign copper to 5 mm
around the probe, then evaluate every sample of the probe against every
windowed item. On a dense board the window holds hundreds of items, a few of
which can be the nearest -- and the cleanup passes ask tens of thousands of
these (smooth_octolinear_chains' clears(): ~120 k probes on a 300-net board).
Each sweep now first drops the items whose lower bound (bbox gap, less the
item's half-width or depth and its largest class excess) exceeds an upper
bound on the result (the probe's first endpoint, which is sample 0 of the
sweep), so the minimum is the same float.

Checks, on random boards (segments and vias of several widths, rect,
roundrect, circle and rotated pads, probes that start inside foreign pads or
cross foreign tracks, net classes and track rules, exact mode):
  1. _seg_foreign_seg_dist, sampled and exact, equals the unpruned sweep bit
     for bit;
  2. _seg_foreign_pad_dist equals the unpruned sweep bit for bit;
  3. the pruning is exercised (a windowed set above the threshold in most
     cases).

    python3 tests/test_foreign_dist_prune.py
"""
import math
import os
import random
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, TESTS_DIR)

from synth import make_seg, make_via, make_pad, make_net, make_pcb  # noqa: E402
import single_ended_routing as ser                                 # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def random_board(rnd):
    nets = {i: make_net(i, 'N%d' % i) for i in range(1, 13)}
    segs = []
    for _ in range(rnd.randint(20, 250)):
        x, y = rnd.uniform(0, 12), rnd.uniform(0, 12)
        L, a = rnd.uniform(0, 3), rnd.uniform(0, 2 * math.pi)
        segs.append(make_seg(x, y, x + L * math.cos(a), y + L * math.sin(a),
                             net_id=rnd.randint(1, 12), layer=rnd.choice(['F.Cu', 'B.Cu']),
                             width=rnd.choice([0.0, 0.1, 0.2, 0.5])))
    vias = [make_via(rnd.uniform(0, 12), rnd.uniform(0, 12), net_id=rnd.randint(1, 12),
                     size=rnd.choice([0.45, 0.6])) for _ in range(rnd.randint(0, 30))]
    pads_by_net = {}
    for k in range(rnd.randint(20, 200)):
        net = rnd.randint(1, 12)
        shape = rnd.choice(['rect', 'roundrect', 'circle', 'oval'])
        sx = rnd.choice([0.25, 0.5, 1.2]); sy = sx if shape == 'circle' else rnd.choice([0.25, 0.6, 1.5])
        kw = {}
        if rnd.random() < 0.3:
            kw['rect_rotation'] = rnd.choice([30.0, 45.0, 90.0])
        pad = make_pad(net, rnd.uniform(0, 12), rnd.uniform(0, 12), ref='U%d' % k,
                       size_x=sx, size_y=sy, shape=shape, **kw)
        if rnd.random() < 0.2:
            pad.local_clearance = rnd.choice([0.05, 0.2])
        pads_by_net.setdefault(net, []).append(pad)
    pcb = make_pcb(nets=nets, segments=segs, vias=vias, pads_by_net=pads_by_net)
    return pcb, segs, [p for v in pads_by_net.values() for p in v]


def probes(rnd, segs, pads, n):
    out = []
    for _ in range(n):
        if pads and rnd.random() < 0.3:          # starting inside a pad
            p = rnd.choice(pads)
            x, y = p.global_x + rnd.uniform(-0.1, 0.1), p.global_y + rnd.uniform(-0.1, 0.1)
        else:
            s = rnd.choice(segs)
            t = rnd.random()
            x = s.start_x + t * (s.end_x - s.start_x)
            y = s.start_y + t * (s.end_y - s.start_y)
        L, a = rnd.choice([0.05, 0.3, 1.0, 4.0]), rnd.uniform(0, 2 * math.pi)
        out.append((rnd.randint(1, 12), x, y, x + L * math.cos(a), y + L * math.sin(a),
                    rnd.choice(['F.Cu', 'B.Cu'])))
    return out


def run(pcb, cases, ncl, tcl, pruned):
    saved = ser._PRUNE_MIN_ITEMS
    if not pruned:
        ser._PRUNE_MIN_ITEMS = 10 ** 9
    try:
        res = []
        for net, x1, y1, x2, y2, layer in cases:
            res.append(ser._seg_foreign_seg_dist(pcb, net, x1, y1, x2, y2, layer,
                                                  net_clearances=ncl, base_clearance=0.1,
                                                  track_clearances=tcl))
            res.append(ser._seg_foreign_seg_dist(pcb, net, x1, y1, x2, y2, layer,
                                                  net_clearances=ncl, base_clearance=0.1,
                                                  exact=True))
            res.append(ser._seg_foreign_pad_dist(pcb, net, x1, y1, x2, y2, layer,
                                                  base_clearance=0.1 if ncl is not None else None,
                                                  net_clearances=ncl))
        return res
    finally:
        ser._PRUNE_MIN_ITEMS = saved


def main():
    rnd = random.Random(5)
    bad, total, windows = {0: 0, 1: 0, 2: 0}, 0, 0
    for _ in range(40):
        pcb, segs, pads = random_board(rnd)
        cases = probes(rnd, segs, pads, 60)
        ncl = rnd.choice([None, {n: rnd.choice([0.1, 0.15, 0.3]) for n in range(1, 13)}])
        tcl = rnd.choice([None, {3: 0.25}])
        want = run(pcb, cases, ncl, tcl, pruned=False)
        got = run(pcb, cases, ncl, tcl, pruned=True)
        for i, (a, b) in enumerate(zip(want, got)):
            if a != b:
                bad[i % 3] += 1
        total += len(cases)
        windows += sum(len(segs) + len(pcb.vias) > ser._PRUNE_MIN_ITEMS for _ in cases)
    print(f"{total} probes on 40 random boards")
    check("segment distance (sampled and exact) equals the unpruned sweep",
          bad[0] == 0 and bad[1] == 0, f"sampled {bad[0]}, exact {bad[1]} mismatches")
    check("pad distance equals the unpruned sweep", bad[2] == 0, f"{bad[2]} mismatches")
    check("the pruning is exercised", windows > total // 2, f"{windows}/{total}")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
