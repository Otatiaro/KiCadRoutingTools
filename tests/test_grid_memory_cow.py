"""Obstacle-map clones share their tiles (grid_router 0.24, copy-on-write).

route.py clones the working map once per routed net (`clone_fresh()`), and
nested rip-up keeps several clones alive at once (8 measured on a 150 x
107.5 mm 4-layer board). Each clone used to copy the refcount and proximity
hash maps whole (~110-130 MB there); the tiles now stay shared until a clone
writes to one.

Cases:
  1. Eight clone_fresh() of a board-sized map, each unstamping a small net
     (what a route does first), together cost under a tenth of building the
     map once. (On the parent they cost about eight maps; the proximity a
     route then stamps is the clone's own data either way.)
  2. A clone is independent: unstamping in it leaves the source blocked,
     and the source's stats are unchanged.
  3. The clone shares every tile it has not written (`tiles_shared_with`).

Run: python3 tests/test_grid_memory_cow.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rust_router'))
import grid_router  # noqa: E402

W, H, LAYERS = 3000, 2150, 4  # 150 x 107.5 mm at 0.05 mm


def resident_mb():
    """This process's resident memory, or None where it cannot be read."""
    if sys.platform == 'win32':
        import ctypes
        import ctypes.wintypes as wt

        class C(ctypes.Structure):
            _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [
                (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'a', 'b', 'c', 'd',
                                              'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        c = C()
        c.cb = ctypes.sizeof(c)
        ctypes.WinDLL('psapi').GetProcessMemoryInfo(wt.HANDLE(-1), ctypes.byref(c), c.cb)
        return c.PrivateUsage / 2**20
    try:
        with open('/proc/self/statm') as fh:
            return int(fh.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20
    except OSError:
        return None


def board_map():
    """Pads and tracks over the whole board on every layer, and vias:
    ~2 M counted cells, like a dense board."""
    m = grid_router.GridObstacleMap(LAYERS)
    rows = [(gx, gy, gy + 24, layer)
            for layer in range(LAYERS) for gx in range(0, W, 2) for gy in range(layer * 7, H, 160)]
    m.add_blocked_cell_spans_batch(np.array(rows, dtype=np.int32))
    m.add_blocked_via_spans_batch(np.array([(gx, gy, gy + 10) for gx in range(0, W, 3)
                                            for gy in range(40, H, 200)], dtype=np.int32))
    return m


def test_clones_cost_less_than_one_map():
    before = resident_mb()
    base = board_map()
    built = resident_mb()
    clones = []
    for k in range(8):
        c = base.clone_fresh()
        net = np.array([(100 + 300 * k, 0, 24, 0)], dtype=np.int32)
        c.remove_blocked_cell_spans_batch(net)
        clones.append(c)
    after = resident_mb()
    cells = base.get_stats()[0]
    assert cells > 1_500_000, 'BROKEN TEST: only %d blocked cells' % cells
    if before is None:
        print('SKIP memory check: resident memory unreadable here')
        return
    one, eight = built - before, after - built
    print('  one map %.0f MB, eight routed clones %.0f MB' % (one, eight))
    assert one > 5, 'BROKEN TEST: building the map took %.1f MB' % one
    assert eight < one / 10, 'eight clones took %.0f MB against %.0f MB for the map itself' % (eight, one)


def test_clone_is_independent_and_shares_the_rest():
    base = board_map()
    stats = base.get_stats()
    c = base.clone_fresh()
    assert c.is_blocked(100, 0, 0) and base.is_blocked(100, 0, 0)
    c.remove_blocked_cell_spans_batch(np.array([(100, 0, 24, 0)], dtype=np.int32))
    c.add_stub_proximity_costs_batch([(50, 50)], 10, 500)
    assert not c.is_blocked(100, 0, 0), 'unstamp did not reach the clone'
    assert base.is_blocked(100, 0, 0), 'unstamping the clone unblocked the source'
    assert base.get_stub_proximity_cost(50, 50) == 0 and c.get_stub_proximity_cost(50, 50) == 500
    assert base.get_stats() == stats, 'the source changed: %s -> %s' % (stats, base.get_stats())
    shared, total = c.tiles_shared_with(base)
    # one counted tile copied (the unstamped column) and the new cost tiles
    assert total - shared <= 1 + c.tile_stats()[1], (shared, total, c.tile_stats())
    assert shared > 1000, (shared, total)


if __name__ == '__main__':
    test_clones_cost_less_than_one_map()
    test_clone_is_independent_and_shares_the_rest()
    print('test_grid_memory_cow: all passed')
