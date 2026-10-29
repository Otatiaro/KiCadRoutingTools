#!/usr/bin/env python3
"""The net rescue tries each distinct rung once.

_attempt_edge walks the rescue ladder (_rescue_rungs) and stops at the first
rung that routes the gap. Under --escalation off nothing may narrow, so the
clearance ladder collapses to one rung at the run's own clearance, track and
power widths -- the same config as rung 0 (the finer grid at nominal
geometry). Every gap that failed rung 0 was then searched a second time: the
same window map (a cache hit), the same seeds, the same A* to the same
failure. On a dense board the failed rescue searches are the long ones (up to
the 1 M-iteration backstop each way), so the repeat doubled them.

Checks, on the two-wall channel board of test_net_rescue (two victim pads
across net-2 walls, 2 layers):
  1. under --escalation off, _rescue_rungs holds no two equal rungs, and still
     starts with rung 0 (fails on the parent: the ladder rung repeats it);
  2. a gap no rung can route is searched once per distinct rung -- once under
     --escalation off (twice on the parent) -- and the board is untouched;
  3. the rescue's outcome is the one the full ladder gives, rung for rung:
     same summary, same copper, same clearance and width used, on the pinch
     (routes only by necking down) and on the impossible channel, under the
     default policy and under --escalation off.

    python3 tests/test_rescue_distinct_rungs.py
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from types import SimpleNamespace                                  # noqa: E402

from kicad_parser import BoardInfo                                 # noqa: E402
from routing_config import GridRouteConfig                         # noqa: E402
from synth import make_net, make_pad, make_pcb, make_seg           # noqa: E402
import fab_tiers                                                   # noqa: E402
import net_rescue                                                  # noqa: E402
import single_ended_routing                                        # noqa: E402

VICTIM, WALL = 1, 2
failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def board(wall_y):
    bi = BoardInfo(layers={0: 'F.Cu', 31: 'B.Cu'}, copper_layers=['F.Cu', 'B.Cu'],
                   board_bounds=(-1.0, -1.0, 4.0, 1.0))
    pads = [make_pad(VICTIM, -0.4, 0.0, ref='U1', num='1', net_name='VICTIM',
                     size_x=0.3, size_y=0.3),
            make_pad(VICTIM, 3.4, 0.0, ref='U2', num='1', net_name='VICTIM',
                     size_x=0.3, size_y=0.3)]
    walls = [make_seg(-1.0, y, 4.0, y, layer=lay, net_id=WALL, width=0.2)
             for y in (wall_y, -wall_y) for lay in ('F.Cu', 'B.Cu')]
    return make_pcb(nets={VICTIM: make_net(VICTIM, 'VICTIM'), WALL: make_net(WALL, 'WALL')},
                    segments=walls, pads_by_net={VICTIM: pads, WALL: []}, board_info=bi)


def cfg():
    c = GridRouteConfig()
    c.layers = ['F.Cu', 'B.Cu']
    c.grid_step = 0.05
    c.clearance = 0.15
    c.track_width = 0.15
    c.via_size = 0.5
    c.via_drill = 0.3
    return c


def rescue(wall_y, policy, distinct=True):
    """rescue_failed_nets on a fresh board: (summary, copper, searches)."""
    fab_tiers.set_escalation_policy(policy, {})
    pcb = board(wall_y)
    state = SimpleNamespace(pcb_data=pcb, config=cfg(), routed_results={}, results=[],
                            remaining_net_ids=[VICTIM], routed_net_ids=[],
                            net_history={}, route_index=0)
    searches = []
    real_search = single_ended_routing.route_net_with_obstacles
    real_distinct = getattr(net_rescue, '_distinct_rungs', None)   # None on the parent

    def counted(pcb_, net_id, c, *a, **k):
        searches.append((c.grid_step, c.clearance, c.track_width, c.via_size))
        return real_search(pcb_, net_id, c, *a, **k)

    single_ended_routing.route_net_with_obstacles = counted
    if not distinct and real_distinct is not None:
        net_rescue._distinct_rungs = lambda rungs: list(rungs)
    try:
        summary = net_rescue.rescue_failed_nets(state, [('VICTIM', VICTIM)])
    finally:
        single_ended_routing.route_net_with_obstacles = real_search
        if real_distinct is not None:
            net_rescue._distinct_rungs = real_distinct
        fab_tiers.set_escalation_policy(None, {})
    if summary:
        summary = {k: v for k, v in summary.items() if k != 'time'}
    copper = sorted((round(s.start_x, 6), round(s.start_y, 6), round(s.end_x, 6),
                     round(s.end_y, 6), s.layer, round(s.width, 6), s.net_id)
                    for s in pcb.segments)
    copper += sorted((round(v.x, 6), round(v.y, 6), round(v.size, 6)) for v in pcb.vias)
    return summary, copper, searches


def main():
    # 1. the ladder under --escalation off
    fab_tiers.set_escalation_policy('off', {})
    try:
        pcb = board(0.32)
        rungs = net_rescue._rescue_rungs(cfg(), 0.025, pcb, VICTIM)
    finally:
        fab_tiers.set_escalation_policy(None, {})
    dups = sum(1 for i, r in enumerate(rungs) if any(r == o for o in rungs[:i]))
    check("--escalation off: no rung repeats an earlier one", dups == 0,
          f"{len(rungs)} rung(s), {dups} repeated")
    check("--escalation off: rung 0 is still the finer grid at nominal geometry",
          bool(rungs) and rungs[0].grid_step == 0.025 and rungs[0].clearance == 0.15
          and rungs[0].track_width == 0.15)

    # 2. an impossible gap: searched once per distinct rung
    _, copper, searches = rescue(0.24, 'off')
    check("--escalation off: an unroutable gap is searched once", len(searches) == 1,
          f"{len(searches)} search(es): {searches}")
    check("the board comes back untouched",
          all(c[-1] == WALL for c in copper if len(c) == 7) and len(copper) == 4)
    _, _, searches = rescue(0.24, None)
    check("default policy: no two searches at the same geometry",
          len(set(searches)) == len(searches), f"{searches}")

    # 3. same outcome as the full ladder
    for wall_y, policy in ((0.32, None), (0.32, 'off'), (0.24, None), (0.24, 'off'),
                           (0.6, 'off')):
        got = rescue(wall_y, policy)
        want = rescue(wall_y, policy, distinct=False)
        tag = f"walls +/-{wall_y}, escalation {policy or 'default'}"
        check(f"{tag}: same summary and copper as the full ladder",
              got[0] == want[0] and got[1] == want[1],
              f"{(got[0] or {}).get('recovered')} vs {(want[0] or {}).get('recovered')}; "
              f"searches {len(got[2])} vs {len(want[2])}")

    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
