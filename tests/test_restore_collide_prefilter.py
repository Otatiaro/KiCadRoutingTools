#!/usr/bin/env python3
"""The restore collision test measures only the pairs that can collide.

`rip_up_reroute._saved_route_colliders` decides whether a ripped net's saved
copper may be put back. Inside its prefilter box it measured EVERY restored
item against EVERY foreign item -- a long saved route boxes most of the board,
so the pairs grew as (restored items) x (foreign items), ~20 M point-to-segment
distances in one rip-heavy route. Each restored item now keeps, in one numpy
pass, only the foreign items whose bbox gap is under the largest threshold a
pair can have, then measures those in the original order.

Checks, on random boards (segments of every width, layer and bearing, vias,
long saved routes, net classes, a .kicad_dru layer rule, no config):
  1. the hits -- items, kinds and order -- equal the full pairwise scan;
  2. the first_only verdict (_saved_route_collides) equals the full scan's.

    python3 tests/test_restore_collide_prefilter.py
"""
import os
import random
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, TESTS_DIR)

from routing_config import GridRouteConfig                  # noqa: E402
from synth import make_seg, make_via, make_pcb              # noqa: E402
import rip_up_reroute as rr                                 # noqa: E402
from geometry_utils import point_to_segment_dist_sq as p2s  # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def reference(saved, pcb, own_ids, clearance, config):
    """Every restored item against every foreign item, as defined."""
    own = set(own_ids)
    pairwise = config is not None and not config.pair_clearance_inert()

    def clr(own_net, o, layer, kind):
        return (config.pair_clearance(own_net, o.net_id, layer, kind=kind, base=clearance)
                if pairwise else clearance)
    o_segs = [s for s in pcb.segments if s.net_id not in own and s.net_id != 0]
    o_vias = [v for v in pcb.vias if v.net_id not in own and v.net_id != 0]
    hits, seen = [], set()

    def hit(kind, obj):
        if id(obj) not in seen:
            seen.add(id(obj)); hits.append((kind, obj))
    for s in saved['new_segments']:
        hw = s.width / 2.0
        for o in o_segs:
            if o.layer == s.layer:
                thr = hw + o.width / 2.0 + clr(s.net_id, o, s.layer, 'track')
                if rr._seg_seg_dist_sq(s.start_x, s.start_y, s.end_x, s.end_y,
                                       o.start_x, o.start_y, o.end_x, o.end_y) < thr * thr:
                    hit('segment', o)
        for v in o_vias:
            thr = hw + v.size / 2.0 + clr(s.net_id, v, s.layer, 'layer')
            if p2s(v.x, v.y, s.start_x, s.start_y, s.end_x, s.end_y) < thr * thr:
                hit('via', v)
    for vv in saved['new_vias']:
        vr = vv.size / 2.0
        for v in o_vias:
            thr = vr + v.size / 2.0 + clr(vv.net_id, v, None, 'stack')
            if (vv.x - v.x) ** 2 + (vv.y - v.y) ** 2 < thr * thr:
                hit('via', v)
        for o in o_segs:
            thr = vr + o.width / 2.0 + clr(vv.net_id, o, o.layer, 'layer')
            if p2s(vv.x, vv.y, o.start_x, o.start_y, o.end_x, o.end_y) < thr * thr:
                hit('segment', o)
    return hits


def random_case(rnd):
    layers = ['F.Cu', 'B.Cu']
    config = None
    if rnd.random() < 0.8:
        config = GridRouteConfig(clearance=0.1, track_width=0.15, via_size=0.45,
                                 via_drill=0.3, layers=layers, grid_step=0.05)
        if rnd.random() < 0.5:
            config.set_net_clearances({n: rnd.choice([0.1, 0.2, 0.4])
                                       for n in range(3, 9)}, routed_net_ids=[1, 2])
        if rnd.random() < 0.3:
            config.layer_clearances = {'B.Cu': 0.08}

    def seg(net, x, y, long_=False):
        L = rnd.uniform(5, 20) if long_ else rnd.uniform(0, 2)
        dx, dy = rnd.choice([(1, 0), (0, 1), (0.7071, 0.7071), (0.7071, -0.7071),
                             (rnd.random(), rnd.random())])
        return make_seg(x, y, x + L * dx, y + L * dy, net_id=net, layer=rnd.choice(layers),
                        width=rnd.choice([0.1, 0.15, 0.25, 1.0]))
    foreign = [seg(rnd.randint(0, 8), rnd.uniform(0, 25), rnd.uniform(0, 25))
               for _ in range(rnd.randint(0, 200))]
    fvias = [make_via(rnd.uniform(0, 25), rnd.uniform(0, 25), net_id=rnd.randint(0, 8),
                      size=rnd.choice([0.45, 0.6, 1.2])) for _ in range(rnd.randint(0, 40))]
    own_net = rnd.choice([1, 2])
    saved_segs = [seg(own_net, rnd.uniform(0, 25), rnd.uniform(0, 25), long_=rnd.random() < 0.3)
                  for _ in range(rnd.randint(0, 25))]
    saved_vias = [make_via(rnd.uniform(0, 25), rnd.uniform(0, 25), net_id=own_net,
                           size=rnd.choice([0.45, 0.6])) for _ in range(rnd.randint(0, 6))]
    if not saved_segs and not saved_vias:
        saved_vias = [make_via(12, 12, net_id=own_net)]
    pcb = make_pcb(segments=foreign, vias=fvias)
    return ({'new_segments': saved_segs, 'new_vias': saved_vias}, pcb, [1, 2],
            rnd.choice([0.1, 0.15]), config)


def main():
    rnd = random.Random(3)
    n, bad_hits, bad_first, with_hits = 300, [], [], 0
    for case in range(n):
        saved, pcb, own, clearance, config = random_case(rnd)
        want = reference(saved, pcb, own, clearance, config)
        got = rr._saved_route_colliders(saved, pcb, own, clearance, config=config)
        with_hits += bool(want)
        if [(k, id(o)) for k, o in want] != [(k, id(o)) for k, o in got]:
            bad_hits.append(case)
        if rr._saved_route_collides(saved, pcb, own, clearance, config=config) != bool(want):
            bad_first.append(case)
    print(f"{n} random cases, {with_hits} with a collision")
    check("hits, kinds and order equal the full pairwise scan",
          not bad_hits and 0 < with_hits < n, f"mismatches {bad_hits[:5]}")
    check("the first_only verdict equals the full scan's", not bad_first,
          f"mismatches {bad_first[:5]}")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
