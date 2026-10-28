"""Stub proximity costs equal the linear-falloff formula, cell for cell.

add_stub_proximity_costs_batch / _grouped compute each disk once per call
and stamp it row by row (grid_router 0.24.1). The costs must stay the
integers the per-cell formula gives -- (int)((1 - sqrt(d2) / r) * max_cost)
in f32, the max over a group's disks, then the max (batch, or inside a
max-zone rect) or the sum (grouped) into the map -- including disks that
cross tile edges (64 cells), negative coordinates and radii 0 and 1.

Run: python3 tests/test_stub_proximity_disk.py
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rust_router'))
import grid_router  # noqa: E402

f32 = np.float32


def disk(radius, max_cost):
    """{(dx, dy): cost} of one disk, the formula in f32 as the router has it."""
    out = {}
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            d2 = dx * dx + dy * dy
            if d2 <= radius * radius:
                with np.errstate(divide='ignore', invalid='ignore'):
                    p = f32(1.0) - np.sqrt(f32(d2)) / f32(radius)
                    v = p * f32(max_cost)
                out[(dx, dy)] = 0 if np.isnan(v) else int(v)
    return out


def reference_grouped(groups, rects):
    field_total = {}
    for points, radius, max_cost in groups:
        if radius <= 0 or not points:
            continue
        d = disk(radius, max_cost)
        field = {}
        for cx, cy in points:
            for (dx, dy), c in d.items():
                if c > 0 and c > field.get((cx + dx, cy + dy), 0):
                    field[(cx + dx, cy + dy)] = c
        for (x, y), c in field.items():
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in rects or ()):
                field_total[(x, y)] = max(field_total.get((x, y), 0), c)
            else:
                field_total[(x, y)] = field_total.get((x, y), 0) + c
    return field_total


def check(m, ref, box):
    x0, y0, x1, y1 = box
    bad = [(x, y, m.get_stub_proximity_cost(x, y), ref.get((x, y), 0))
           for x in range(x0, x1) for y in range(y0, y1)
           if m.get_stub_proximity_cost(x, y) != ref.get((x, y), 0)]
    assert not bad, '%d cells differ, e.g. (x, y, router, formula) %s' % (len(bad), bad[:5])
    assert m.get_stats()[2] == sum(1 for v in ref.values() if v), (m.get_stats()[2], len(ref))


BOX = (-90, -90, 260, 200)


def test_batch_is_the_max_of_the_disks():
    rnd = random.Random(1)
    for radius, max_cost in ((0, 500), (1, 500), (7, 1000), (23, 333), (40, 1000)):
        stubs = [(rnd.randrange(-60, 230), rnd.randrange(-60, 170)) for _ in range(25)] + [(63, 63), (64, -1)]
        m = grid_router.GridObstacleMap(2)
        m.add_stub_proximity_costs_batch(stubs, radius, max_cost)
        ref = {}
        for cx, cy in stubs:
            for (dx, dy), c in disk(radius, max_cost).items():
                if c > 0 and c > ref.get((cx + dx, cy + dy), 0):
                    ref[(cx + dx, cy + dy)] = c
        check(m, ref, BOX)


def test_grouped_sums_groups_and_takes_the_max_in_zones():
    rnd = random.Random(2)
    groups = [([(rnd.randrange(-60, 230), rnd.randrange(-60, 170)) for _ in range(rnd.randrange(1, 5))],
               rnd.choice((0, 1, 9, 30)), rnd.choice((200, 1000))) for _ in range(40)]
    rects = [(0, 0, 70, 90), (150, -50, 200, 20)]
    for zones in (rects, None):
        m = grid_router.GridObstacleMap(2)
        m.add_stub_proximity_costs_grouped(groups, zones)
        check(m, reference_grouped(groups, zones), BOX)


if __name__ == '__main__':
    test_batch_is_the_max_of_the_disks()
    test_grouped_sums_groups_and_takes_the_max_in_zones()
    print('test_stub_proximity_disk: all passed')
