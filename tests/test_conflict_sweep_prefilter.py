#!/usr/bin/env python3
"""The restore conflict sweep samples only the pairs a bbox gap leaves possible.

`rip_restore._conflict_sweep` (behind `_copper_conflicts` and
`conflict_owner_ids`, the terminal and victim-priority restores) sampled every
candidate segment every 0.2 mm against EVERY board segment and via on its
layer. It now skips, before sampling, a board item whose bbox gap to the
candidate is already at least the pair's required distance: no sample, all of
which lie in the candidate's bbox, can be nearer than that gap.

Checks, on random boards (segments of every width, layer and bearing, vias,
net classes, a .kicad_dru layer rule, in-flight copper):
  1. conflict_owner_ids equals the unfiltered sweep's (the skip disabled);
  2. _copper_conflicts' verdict equals the unfiltered sweep's.

    python3 tests/test_conflict_sweep_prefilter.py
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
from synth import make_seg, make_via, make_pcb              # noqa: E402
import rip_restore as rs                                    # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def random_case(rnd):
    layers = ['F.Cu', 'B.Cu']
    config = GridRouteConfig(clearance=0.1, track_width=0.15, via_size=0.45,
                             via_drill=0.3, layers=layers, grid_step=0.05)
    if rnd.random() < 0.5:
        config.set_net_clearances({n: rnd.choice([0.1, 0.2, 0.4]) for n in range(3, 9)},
                                  routed_net_ids=[1, 2])
    if rnd.random() < 0.3:
        config.layer_clearances = {'B.Cu': 0.08}

    def seg(net, x, y, L):
        a = rnd.uniform(0, 2 * math.pi)
        return make_seg(x, y, x + L * math.cos(a), y + L * math.sin(a), net_id=net,
                        layer=rnd.choice(layers), width=rnd.choice([0.1, 0.15, 0.25, 1.0]))
    board = [seg(rnd.randint(1, 8), rnd.uniform(0, 20), rnd.uniform(0, 20), rnd.uniform(0, 3))
             for _ in range(rnd.randint(0, 150))]
    bvias = [make_via(rnd.uniform(0, 20), rnd.uniform(0, 20), net_id=rnd.randint(1, 8),
                      size=rnd.choice([0.45, 0.8])) for _ in range(rnd.randint(0, 30))]
    pcb = make_pcb(segments=board, vias=bvias)
    if rnd.random() < 0.3:
        pcb._inflight_copper = [([seg(rnd.randint(3, 8), rnd.uniform(0, 20),
                                      rnd.uniform(0, 20), 2.0)], [])]
    own = rnd.choice([1, 2])
    cand = [seg(own, rnd.uniform(0, 20), rnd.uniform(0, 20), rnd.uniform(0, 8))
            for _ in range(rnd.randint(1, 12))]
    cvias = [make_via(rnd.uniform(0, 20), rnd.uniform(0, 20), net_id=own)
             for _ in range(rnd.randint(0, 4))]
    return pcb, config, {own}, cand, cvias


def sweep(pcb, config, own, cand, cvias, filtered):
    saved = rs._GAP_SLACK
    if not filtered:
        rs._GAP_SLACK = 1e9          # no pair is ever far enough: every one is sampled
    try:
        return (rs.conflict_owner_ids(pcb, config, own, cand, cvias),
                rs._copper_conflicts(pcb, config, own, cand, cvias))
    finally:
        rs._GAP_SLACK = saved


def main():
    rnd = random.Random(11)
    n, bad_owners, bad_verdict, some, none_ = 300, [], [], 0, 0
    for case in range(n):
        args = random_case(rnd)
        w_owners, w_any = sweep(*args, filtered=False)
        g_owners, g_any = sweep(*args, filtered=True)
        some += bool(w_owners); none_ += not w_owners
        if w_owners != g_owners:
            bad_owners.append(case)
        if w_any != g_any:
            bad_verdict.append(case)
    print(f"{n} random cases: {some} with conflicts, {none_} without")
    check("conflict owners equal the unfiltered sweep's",
          not bad_owners and some and none_, f"mismatches {bad_owners[:5]}")
    check("the boolean verdict equals the unfiltered sweep's", not bad_verdict,
          f"mismatches {bad_verdict[:5]}")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
