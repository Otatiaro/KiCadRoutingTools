#!/usr/bin/env python3
"""Frontier blocker attribution counts the same cells, faster.

find_route_blocker_from_frontier names the net whose copper accounts for the
most cells of a failed search's blocked frontier. It used to probe, for every
point of every nearby segment's walk, each cell of a (2e+1)-square window in a
set -- (2e+1)^2 set lookups per point, ~150 at a 0.05 mm grid -- and
preexisting_blocker_hint called it up to 16 times per failed net with the same
frontier, recounting identical tallies to pick the next blocker. The tally now
comes from a 2D prefix sum (frontier_block_counts), and the hint counts once
per frontier then picks blockers in turn (top_frontier_blocker).

Checks, on random boards (segments of every width and bearing, zero-length
segments, vias of several sizes, frontiers with off-layer and duplicate cells,
copper straddling the frontier bbox):
  1. frontier_block_counts equals the cell-by-cell reference tally, net by net
     and in the same first-seen order.
  2. find_route_blocker_from_frontier returns what the reference selection
     returns, for random protected sets.
  3. naming blockers in turn from one tally gives the same sequence as
     recounting before every pick (what the hint used to do).

    python3 tests/test_frontier_block_counts.py
"""
import io
import os
import random
import sys
from contextlib import redirect_stdout

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from synth import make_seg, make_via, make_net, make_pcb       # noqa: E402
from routing_config import GridRouteConfig, GridCoord          # noqa: E402
from bresenham_utils import walk_line                          # noqa: E402
from plane_blocker_detection import (                          # noqa: E402
    find_route_blocker_from_frontier, frontier_block_counts,
    top_frontier_blocker)

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def reference_counts(blocked_cells, pcb_data, config, exclude_net_id):
    """The tally as defined: every window cell of every walk point probed."""
    coord = GridCoord(config.grid_step)
    blocked = set(blocked_cells)
    counts = {}
    for seg in pcb_data.segments:
        if seg.net_id == exclude_net_id:
            continue
        half = (seg.width if seg.width > 0 else config.track_width) / 2
        e = max(1, coord.to_grid_dist(half + config.clearance + config.track_width / 2))
        gx1, gy1 = coord.to_grid(seg.start_x, seg.start_y)
        gx2, gy2 = coord.to_grid(seg.end_x, seg.end_y)
        n = sum((gx + ex, gy + ey, 0) in blocked
                for gx, gy in walk_line(gx1, gy1, gx2, gy2)
                for ex in range(-e, e + 1) for ey in range(-e, e + 1))
        if n:
            counts[seg.net_id] = counts.get(seg.net_id, 0) + n
    for via in pcb_data.vias:
        if via.net_id == exclude_net_id:
            continue
        r = (via.size if via.size > 0 else config.via_size) / 2
        e = max(1, coord.to_grid_dist(r + config.track_width / 2 + config.clearance))
        gx, gy = coord.to_grid(via.x, via.y)
        n = sum((gx + ex, gy + ey, 0) in blocked
                for ex in range(-e, e + 1) for ey in range(-e, e + 1)
                if ex * ex + ey * ey <= e * e)
        if n:
            counts[via.net_id] = counts.get(via.net_id, 0) + n
    return counts


def reference_pick(counts, protected):
    free = {k: v for k, v in counts.items() if k not in protected}
    return max(free, key=lambda k: free[k]) if free else None


def random_case(rnd):
    step = rnd.choice([0.05, 0.1])
    config = GridRouteConfig(grid_step=step, clearance=rnd.choice([0.09, 0.15]),
                             track_width=rnd.choice([0.1, 0.2]), via_size=0.45)
    nets = {i: make_net(i, 'N%d' % i) for i in range(1, 9)}
    segs = []
    for _ in range(rnd.randint(0, 40)):
        x1, y1 = rnd.uniform(0, 6), rnd.uniform(0, 6)
        if rnd.random() < 0.1:
            x2, y2 = x1, y1                              # zero-length
        else:
            x2, y2 = x1 + rnd.uniform(-3, 3), y1 + rnd.uniform(-3, 3)
        segs.append(make_seg(x1, y1, x2, y2, net_id=rnd.randint(1, 8),
                             width=rnd.choice([0.0, 0.1, 0.2, 0.5]),
                             layer=rnd.choice(['F.Cu', 'B.Cu'])))
    vias = [make_via(rnd.uniform(0, 6), rnd.uniform(0, 6), net_id=rnd.randint(1, 8),
                     size=rnd.choice([0.0, 0.3, 0.45, 0.8]))
            for _ in range(rnd.randint(0, 15))]
    pcb = make_pcb(nets=nets, segments=segs, vias=vias)
    # a frontier: a blob of cells around a centre, some off layer 0, with repeats
    cx, cy = rnd.randint(10, 50), rnd.randint(10, 50)
    cells = [(cx + rnd.randint(-15, 15), cy + rnd.randint(-15, 15),
              rnd.choice([0, 0, 0, 1, 2])) for _ in range(rnd.randint(0, 300))]
    cells += cells[:10]
    return pcb, config, cells, rnd.randint(1, 8)


def main():
    with redirect_stdout(io.StringIO()):   # the protected-blocker chatter
        result = run_cases()
    report(*result)


def run_cases():
    rnd = random.Random(1)
    n_cases, bad_counts, bad_pick, bad_seq, nonempty = 400, [], [], [], 0
    for case in range(n_cases):
        pcb, config, cells, exclude = random_case(rnd)
        ref = reference_counts(cells, pcb, config, exclude)
        got = frontier_block_counts(cells, pcb, config, exclude)
        nonempty += bool(ref)
        if list(ref.items()) != list(got.items()):
            bad_counts.append(case)
        protected = set(rnd.sample(range(1, 9), rnd.randint(0, 4)))
        want = reference_pick(ref, protected)
        if find_route_blocker_from_frontier(cells, pcb, config, exclude,
                                            set(protected)) != want:
            bad_pick.append(case)
        # the hint's loop: name blockers in turn, protecting each named one
        seq_old, seq_new, p_old, p_new = [], [], set(protected), set(protected)
        for _ in range(8):
            b = find_route_blocker_from_frontier(cells, pcb, config, exclude, p_old)
            if b is None:
                break
            seq_old.append(b); p_old.add(b)
        for _ in range(8):
            b = top_frontier_blocker(got, pcb, p_new)
            if b is None:
                break
            seq_new.append(b); p_new.add(b)
        if seq_old != seq_new:
            bad_seq.append(case)
    return n_cases, nonempty, bad_counts, bad_pick, bad_seq


def report(n_cases, nonempty, bad_counts, bad_pick, bad_seq):
    print(f"{n_cases} random cases, {nonempty} with a non-empty tally")
    check("counts equal the cell-by-cell tally, in first-seen order",
          not bad_counts and nonempty > n_cases // 2, f"mismatches {bad_counts[:5]}")
    check("the named blocker is the reference pick", not bad_pick,
          f"mismatches {bad_pick[:5]}")
    check("one tally names the same blockers in turn as recounting",
          not bad_seq, f"mismatches {bad_seq[:5]}")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
