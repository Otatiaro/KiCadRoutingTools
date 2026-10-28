#!/usr/bin/env python3
"""Concurrent A* searches: the Rust searches release the GIL, a SearchCancel
stops one from another thread, and a connection runs its forward and backward
full searches at the same time with the sequential outcome.

1. GIL released: while a long search runs on a thread, the main thread keeps
   running Python (its longest stall stays far below the search's duration).
2. SearchCancel: a token cancelled before the call returns (None, 0, []); one
   cancelled mid-search ends it early with no path and no frontier.
3. _probe_route_with_frontier_once with the backward full search on a second
   thread returns exactly what the sequential order returns (the same search
   run lazily, only when its result is read), for a forward success, a forward
   failure rescued by the backward search, and a failure both ways.
"""
import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'py_router'))
sys.path.insert(0, os.path.join(REPO, 'rust_router'))

import numpy as np

from grid_router import GridObstacleMap, GridRouter
try:
    from grid_router import SearchCancel
except ImportError:  # a grid_router older than the concurrent searches
    SearchCancel = None
import single_ended_routing as ser
from routing_config import GridRouteConfig


def _boxed(size):
    """Blocked cells of a one-layer square field's boundary."""
    blocked = []
    for i in range(size + 1):
        blocked += [(i, 0, 0), (i, size, 0), (0, i, 0), (size, i, 0)]
    return blocked


def build_sealed(size=900, box=3):
    """Open field, target sealed in a box: the search floods the whole field."""
    obs = GridObstacleMap(1)
    blocked = _boxed(size)
    c = size // 2
    for x in range(c - box, c + box + 1):
        for y in range(c - box, c + box + 1):
            if abs(x - c) == box or abs(y - c) == box:
                blocked.append((x, y, 0))
    obs.add_blocked_cells_batch(np.asarray(blocked, dtype=np.int32))
    return obs, [(3, 3, 0)], [(c, c, 0)]


def build_cup(size=400, half=120):
    """A cup open away from the source, the target deep inside it: the forward
    search fills the cup's mouth side before going round (about twice the
    backward search's expansions), the backward search walks out and round."""
    obs = GridObstacleMap(1)
    blocked = _boxed(size)
    c = size // 2
    for y in range(c - half, c + half + 1):
        blocked.append((c - half, y, 0))
    for x in range(c - half, c + half + 1):
        blocked += [(x, c - half, 0), (x, c + half, 0)]
    obs.add_blocked_cells_batch(np.asarray(blocked, dtype=np.int32))
    return obs, [(10, c, 0)], [(c - half + 5, c, 0)]


def build_split(size=300):
    """A wall splits the field: each end floods its own half, neither routes."""
    obs = GridObstacleMap(1)
    blocked = _boxed(size) + [(size // 2, y, 0) for y in range(size + 1)]
    obs.add_blocked_cells_batch(np.asarray(blocked, dtype=np.int32))
    return obs, [(10, size // 2, 0)], [(size - 10, size // 2, 0)]


class _LazyFuture:
    def __init__(self, fn, args, kwargs):
        self._call = (fn, args, kwargs)

    def result(self):
        fn, args, kwargs = self._call
        return fn(*args, **kwargs)


class _LazyPool:
    """Runs a submitted search only when its result is read: the sequential
    order (forward first, then backward only if asked for)."""
    def submit(self, fn, *args, **kwargs):
        return _LazyFuture(fn, args, kwargs)


def check_gil_released(router, failures):
    obs, src, tgt = build_sealed()
    out = {}

    def run():
        out['t0'] = time.perf_counter()
        out['r'] = router.route_with_frontier(obs, src, tgt, 10_000_000)
        out['t1'] = time.perf_counter()

    # The main thread timestamps its own progress; the longest gap between
    # them inside the search's interval is how long Python could not run.
    ticks = [time.perf_counter()]
    th = threading.Thread(target=run)
    th.start()
    while th.is_alive():
        ticks.append(time.perf_counter())
    th.join()
    t0, t1 = out['t0'], out['t1']
    inside = [t0] + [t for t in ticks if t0 < t < t1] + [t1]
    stall = max(b - a for a, b in zip(inside, inside[1:]))
    if out['r'][0] is not None:
        failures.append('sealed target was reached')
    if t1 - t0 < 0.2:
        failures.append('search too short to judge the GIL: %.3fs' % (t1 - t0))
    elif stall > 0.5 * (t1 - t0):
        failures.append('main thread stalled %.3fs during a %.3fs search: GIL held'
                        % (stall, t1 - t0))


def check_cancel(router, failures):
    if SearchCancel is None:
        failures.append('grid_router has no SearchCancel')
        return
    obs, src, tgt = build_sealed()
    token = SearchCancel()
    token.cancel()
    r = router.route_with_frontier(obs, src, tgt, 10_000_000, cancel=token)
    if r != (None, 0, []):
        failures.append('pre-cancelled search returned %r' % (r[1:],))
    _p, full_iters, full_blocked = router.route_with_frontier(obs, src, tgt, 10_000_000)
    if not full_blocked:
        failures.append('uncancelled sealed search returned no frontier')
    token = SearchCancel()
    out = {}
    th = threading.Thread(target=lambda: out.setdefault(
        'r', router.route_with_frontier(obs, src, tgt, 10_000_000, cancel=token)))
    th.start()
    time.sleep(0.02)
    token.cancel()
    th.join()
    path, iters, blocked = out['r']
    if path is not None or blocked or not 0 < iters < full_iters:
        failures.append('mid-search cancel: path=%s iters=%d (full %d) frontier=%d'
                        % (path is not None, iters, full_iters, len(blocked)))


def check_probe_outcomes(router, failures):
    if not hasattr(ser, '_search_pool'):
        failures.append('single_ended_routing runs its full searches one after the other')
        return
    obs, src, tgt = build_cup()
    _p, fwd_iters, _b = router.route_with_frontier(obs, src, tgt, 10_000_000)
    _p, bwd_iters, _b = router.route_with_frontier(obs, tgt, src, 10_000_000)
    if not 5_000 < bwd_iters < fwd_iters:
        failures.append('cup scenario off: forward %d, backward %d expansions'
                        % (fwd_iters, bwd_iters))
        return
    mid = (fwd_iters + bwd_iters) // 2
    split_obs, s_src, s_tgt = build_split()
    cases = {
        'forward routes': (obs, src, tgt, fwd_iters + 1000),
        'backward rescues': (obs, src, tgt, mid),
        'both fail': (split_obs, s_src, s_tgt, 30_000),
    }
    real_pool = ser._search_pool
    try:
        for name, (o, s, t, cap) in cases.items():
            config = GridRouteConfig(max_iterations=cap, max_probe_iterations=1_000)
            ser._search_pool = real_pool
            concurrent = ser._probe_route_with_frontier_once(router, o, s, t, config)
            ser._search_pool = lambda: _LazyPool()
            sequential = ser._probe_route_with_frontier_once(router, o, s, t, config)
            if concurrent != sequential:
                failures.append('%s: concurrent result differs from sequential '
                                '(path %s/%s, iterations %d/%d)'
                                % (name, concurrent[0] is not None,
                                   sequential[0] is not None,
                                   concurrent[1], sequential[1]))
            want_path = name != 'both fail'
            if (sequential[0] is not None) != want_path:
                failures.append('%s: scenario off (path found: %s)'
                                % (name, sequential[0] is not None))
            if name == 'backward rescues' and not sequential[4]:
                failures.append('backward rescues: path did not come from the backward search')
            if name == 'both fail' and not (sequential[2] and sequential[3]):
                failures.append('both fail: a full search left no frontier')
    finally:
        ser._search_pool = real_pool


def main():
    failures = []
    router = GridRouter(500, 1.9, layer_costs=[1000])
    check_gil_released(router, failures)
    check_cancel(router, failures)
    check_probe_outcomes(router, failures)
    if failures:
        print('FAIL')
        for f in failures:
            print('  -', f)
        return 1
    print('PASS: searches release the GIL, cancel works, concurrent probe == sequential')
    return 0


if __name__ == '__main__':
    sys.exit(main())
