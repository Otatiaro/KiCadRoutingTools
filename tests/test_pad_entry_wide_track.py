#!/usr/bin/env python3
"""A WIDE track lands on a fine-pitch row pad through its end, along its axis.

The row-pad flanks (pad_entry.py) leave one grid column open along the
pad's axis. A power net routes wider than the map's stamps, and the A*
tests every cell within its track margin of each move: with the flanks
starting half a grid step off the axis, that margin reached them from the
open column, so no move could end on the pad. A search toward the pad then
explored until its iteration budget ran out, and the connection was won by
the search from the other side, which starts ON the pad (a search never
tests its own start) and leaves it by a via dropped inside the pad. On a
real board each such tap edge cost the full budget, about 2.4 s at
--max-iterations 1000000, and doubled a 15 s signal step.

The board: one row of eight 0.3 x 1.6 mm pads at 0.5 mm pitch, off the
0.05 mm grid; two power nets routed at 0.3 mm (track 0.1 mm): a two-pad
net and a three-pad net (the multipoint path), their other pads above the
row. The other pads of the row are on single-pad nets.

Checks:
  1. on a map holding only the net's stamped flanks, a search at the net's
     own track margin reaches each wired row pad's centre from a cell past
     its end (the search that spent its whole budget);
  2. route.py: every power net routes and connects every pad;
  3. no via sits inside a wired row pad, and the centreline over each
     wired row pad, its flanks and its axial stub (the wide track's width
     plus clearance past the end) runs on the pad's axis column;
  4. check_drc.py and check_connected.py are clean.

    python3 tests/test_pad_entry_wide_track.py
"""
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'rust_router'))

from run_utils import check as run_check, evidence, tool        # noqa: E402
from synth import RoutingBoard                                  # noqa: E402

failures = []
PITCH, PAD_W, PAD_L = 0.5, 0.3, 1.6
TRACK = CLEARANCE = 0.1
WIDE = 0.3
STEP = 0.05
BUDGET = 20000
ORIGIN = (20.019, 15.0)
# pad number -> (net, partner pads)
WIRED = {3: ('PWR_A', [(17.0, 19.5)]),
         6: ('PWR_B', [(12.0, 21.0), (30.0, 21.0)])}


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def board():
    b = RoutingBoard()
    ox, oy = ORIGIN
    pads = []
    for i in range(8):
        num, off = i + 1, round((i - 3.5) * PITCH, 3)
        net = WIRED[num][0] if num in WIRED else f'U1_NC{num}'
        pads.append((str(num), off, 0, PAD_W, PAD_L, net))
    b.part('U1', ox, oy, pads)
    for num, (net, partners) in WIRED.items():
        for k, (x, y) in enumerate(partners):
            b.pad(f'R{num}{k}', x, y, net, sx=0.8, sy=0.8)
    return b.text()


def load(path):
    from kicad_parser import parse_kicad_pcb
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_kicad_pcb(path)


def wired_pads(pcb):
    return [p for net in pcb.pads_by_net.values() for p in net
            if p.component_ref == 'U1' and int(p.pad_number) in WIRED]


def reach_from_past_the_end(pcb):
    """(pad, found, iterations) per wired row pad: a one-layer search at the
    net's track margin, on a map holding only the net's stamped flanks,
    from 1 mm past the pad's end and 1 mm aside to the pad's centre."""
    from grid_router import GridObstacleMap, GridRouter
    from routing_config import GridRouteConfig, GridCoord
    import pad_entry
    out = []
    for p in wired_pads(pcb):
        cfg = GridRouteConfig(layers=['F.Cu'], track_width=TRACK, clearance=CLEARANCE,
                              grid_step=STEP, power_net_widths={p.net_id: WIDE})
        margins = cfg.track_margins_for_net(p.net_id)
        coord = GridCoord(STEP)
        src = coord.to_grid(p.global_x + 1.0, p.global_y - p.size_y / 2 - 1.0)
        tgt = coord.to_grid(p.global_x, p.global_y)
        obs = GridObstacleMap(1)
        router = GridRouter(500, 1.9, layer_costs=[1000])
        with pad_entry.stamp(obs, pcb, p.net_id, cfg):
            path, iters, _ = router.route_with_frontier(
                obs, [(src[0], src[1], 0)], [(tgt[0], tgt[1], 0)], BUDGET,
                track_margin=margins)
        out.append((f'U1.{p.pad_number}', path is not None, iters))
    return out


def _points(s, step=0.01):
    n = max(1, int(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) / step))
    for i in range(n + 1):
        yield (s.start_x + (s.end_x - s.start_x) * i / n,
               s.start_y + (s.end_y - s.start_y) * i / n)


def off_axis(pcb):
    """(pad, point) where a wired row pad's own net has F.Cu centreline over
    the pad's flank region (along its length and the axial stub, out to
    where the wide copper clears the pad's long side) more than half a grid
    step off the axis; and (pad, 'via') for a via inside the pad."""
    hits = []
    for p in wired_pads(pcb):
        hw, hl = p.size_x / 2, p.size_y / 2         # vertical pads
        reach = hl + WIDE + CLEARANCE
        outer = hw + WIDE / 2 + CLEARANCE
        name = f'U1.{p.pad_number}'
        for v in pcb.vias:
            if (v.net_id == p.net_id and abs(v.x - p.global_x) <= hw
                    and abs(v.y - p.global_y) <= hl):
                hits.append((name, 'via'))
        for s in pcb.segments:
            if s.net_id != p.net_id or s.layer != 'F.Cu':
                continue
            for x, y in _points(s):
                lat, along = abs(x - p.global_x), abs(y - p.global_y)
                if along <= reach - STEP and STEP / 2 + 0.005 < lat <= outer:
                    hits.append((name, round(x, 2), round(y, 2)))
                    break
    return sorted(set(hits))


def main():
    tmp = tempfile.mkdtemp(prefix='tpadwide_')
    try:
        inp = os.path.join(tmp, 'in.kicad_pcb')
        out = os.path.join(tmp, 'out.kicad_pcb')
        js = os.path.join(tmp, 'out.json')
        with open(inp, 'w', encoding='utf-8') as fh:
            fh.write(board())
        evidence(inp, 'synthetic input board')
        reached = reach_from_past_the_end(load(inp))
        check("a search at the net's track margin reaches each walled row pad",
              len(reached) == 2 and all(ok for _n, ok, _i in reached),
              f"(pad, found, iterations of {BUDGET}): {reached}")
        env = dict(os.environ, KICAD_PLANE_FINALIZE='0')
        run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                   '--nets', 'PWR_A', 'PWR_B', '--json-out', js,
                   '--grid-step', str(STEP),
                   '--track-width', str(TRACK), '--clearance', str(CLEARANCE),
                   '--power-nets', 'PWR_A', 'PWR_B',
                   '--power-nets-widths', str(WIDE), str(WIDE),
                   '--via-size', '0.45', '--via-drill', '0.3',
                   '--fab-tier', 'standard', '--escalation', 'off'],
                  accept=True, env=env)
        evidence(out, 'routed output')
        with open(evidence(js, '--json-out'), encoding='utf-8') as fh:
            doc = json.load(fh)
        routed = sorted(doc.get('routed_single') or [])
        check("both power nets route", routed == ['PWR_A', 'PWR_B'], str(routed))
        check("the three-pad net connects every pad",
              doc.get('multipoint_pads_connected') == doc.get('multipoint_pads_total') == 3
              and not doc.get('failed_multipoint'),
              f"{doc.get('multipoint_pads_connected')}/{doc.get('multipoint_pads_total')}")
        hits = off_axis(load(out))
        check("each wired row pad is reached on its axis, with no via in it",
              not hits, str(hits))
        drc = run_check([sys.executable, '-X', 'utf8', tool('check_drc.py'), out],
                        accept=True)
        check("check_drc.py is clean", drc.returncode == 0,
              " | ".join((drc.stdout + drc.stderr).strip().splitlines()[-3:]))
        con = run_check([sys.executable, '-X', 'utf8', tool('check_connected.py'), out],
                        accept=True)
        check("check_connected.py is clean", con.returncode == 0,
              " | ".join((con.stdout + con.stderr).strip().splitlines()[-3:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
