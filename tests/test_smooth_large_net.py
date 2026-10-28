#!/usr/bin/env python3
"""The octolinear smoother reaches a net of any size, guarding connectivity once per net.

`smooth_octolinear_chains` (#536) skipped every net above 400 segments, and
graded connectivity over the WHOLE net after every chain it shortened -- a
cost of (shortened chains x net size), which is what made the cap necessary.
The nets above it are a board's power rails: on a 4-layer board, a 590-segment
supply net kept 345 grid staircases the pass would have collapsed, and the
ground net 44 more.

Checks:
  1. a 480-segment net of staircases is smoothed, not skipped: strictly
     octolinear output, shorter copper, both ends of every staircase kept;
  2. connectivity is graded twice for that net (before and after), not once
     per chain;
  3. when the net's combined result grades worse, the net is replayed chain
     by chain, so only the offending chain is reverted and the others stay
     smoothed;
  4. dry_run still mutates nothing on a large net;
  5. on the same board written as a file, beside a foreign-net track, the
     smoothed copper passes check_drc.py and check_connected.py.

    python3 tests/test_smooth_large_net.py
"""
import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'py_router'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

import check_connected  # noqa: E402
from kicad_parser import BoardInfo, Net, Pad, PCBData, Segment, parse_kicad_pcb  # noqa: E402
from pcb_modification import smooth_octolinear_chains  # noqa: E402
from run_utils import check as run_check, evidence  # noqa: E402

N_STAIRS, STEPS, PITCH, W = 40, 12, 0.2, 0.15
fails = []


def check(name, cond, detail=""):
    print(("  ok  " if cond else " FAIL ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def _pad(i, x, y):
    return Pad(component_ref=f'P{i}', pad_number='1', global_x=x, global_y=y,
               local_x=0.0, local_y=0.0, size_x=0.3, size_y=0.3, shape='rect',
               layers=['F.Cu'], net_id=1, net_name='/RAIL', drill=0)


def _board(n_stairs=N_STAIRS):
    """n_stairs separate staircases of STEPS H/V jogs, each pad to pad, one net."""
    segs, pads, ends = [], [], []
    for i in range(n_stairs):
        x, y = 1.0, 1.0 + 4.0 * i
        pads.append(_pad(2 * i, x, y))
        start = (x, y)
        for k in range(STEPS):
            nx, ny = (x + PITCH, y) if k % 2 == 0 else (x, y + PITCH)
            segs.append(Segment(start_x=x, start_y=y, end_x=nx, end_y=ny,
                                width=W, layer='F.Cu', net_id=1))
            x, y = nx, ny
        pads.append(_pad(2 * i + 1, x, y))
        ends.append((start, (x, y)))
    pcb = PCBData(footprints={}, nets={1: Net(1, '/RAIL')}, segments=segs, vias=[],
                  board_info=BoardInfo(layers={}, copper_layers=['F.Cu', 'B.Cu'],
                                       board_bounds=(0.0, 0.0, 200.0, 200.0)),
                  pads_by_net={1: pads})
    return pcb, ends


def _length(segs):
    return sum(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) for s in segs)


def _octolinear(s):
    dx, dy = abs(s.end_x - s.start_x), abs(s.end_y - s.start_y)
    return dx < 1e-6 or dy < 1e-6 or abs(dx - dy) < 1e-6


def _touches(segs, p):
    return any(math.hypot(s.start_x - p[0], s.start_y - p[1]) < 1e-6 or
               math.hypot(s.end_x - p[0], s.end_y - p[1]) < 1e-6 for s in segs)


class _Spy:
    """Counts check_net_connectivity calls; optionally grades a trial worse
    when it no longer holds `marker` (a segment of one chosen staircase)."""

    def __init__(self, marker=None):
        self.calls, self.marker, self.real = 0, marker, check_connected.check_net_connectivity

    def __call__(self, net_id, segs, *a, **k):
        self.calls += 1
        out = self.real(net_id, segs, *a, **k)
        if self.marker is not None and not any(s is self.marker for s in segs):
            out = dict(out, connected=False, num_components=99)
        return out

    def __enter__(self):
        check_connected.check_net_connectivity = self
        return self

    def __exit__(self, *exc):
        check_connected.check_net_connectivity = self.real


def _one_staircase():
    """What the pass makes of ONE staircase, on a net far below the old cap."""
    pcb, _ends = _board(1)
    smooth_octolinear_chains([], pcb, None, clearance=0.1)
    return len(pcb.segments)


def large_net_is_smoothed():
    print("1-2. a 480-segment net")
    per = _one_staircase()
    pcb, ends = _board()
    n0, len0 = len(pcb.segments), _length(pcb.segments)
    with _Spy() as spy:
        _n, nets, _strip, _added, stats = smooth_octolinear_chains([], pcb, None, clearance=0.1)
    check(f"the net is smoothed, not skipped ({n0} segments)",
          nets == 1 and stats.get('nets_skipped_large', 0) == 0, f"{stats}")
    check("every staircase is smoothed as it is on a small net",
          per < STEPS and len(pcb.segments) == per * N_STAIRS,
          f"{n0} -> {len(pcb.segments)} segments ({per} per staircase alone)")
    check("output strictly octolinear", all(_octolinear(s) for s in pcb.segments))
    check("copper is shorter", _length(pcb.segments) < len0 - 1e-6,
          f"{len0:.2f} -> {_length(pcb.segments):.2f} mm")
    check("both ends of every staircase kept",
          all(_touches(pcb.segments, a) and _touches(pcb.segments, b) for a, b in ends))
    check("connectivity graded once per net, not once per chain", spy.calls == 2,
          f"{spy.calls} call(s) for {stats.get('spans')} span(s)")


def worse_net_replays_chain_by_chain():
    print("3. a net whose combined result grades worse")
    per = _one_staircase()
    pcb, ends = _board()
    marker = pcb.segments[STEPS * 7 + 3]          # a jog of staircase #7
    with _Spy(marker=marker):
        _n, nets, _strip, _added, stats = smooth_octolinear_chains([], pcb, None, clearance=0.1)
    kept = [s for s in pcb.segments if s is marker]
    check("the offending chain is reverted", bool(kept) and stats.get('chains_reverted') == 1,
          f"reverted {stats.get('chains_reverted')}")
    stair7 = [s for s in pcb.segments if 1.0 + 4.0 * 7 - 1e-6 <= min(s.start_y, s.end_y)
              and max(s.start_y, s.end_y) <= 1.0 + 4.0 * 7 + STEPS * PITCH + 1e-6]
    check("  and keeps its staircase", len(stair7) == STEPS, f"{len(stair7)} segment(s)")
    check("the other chains stay smoothed",
          len(pcb.segments) == STEPS + per * (N_STAIRS - 1), f"{len(pcb.segments)} segments")


def dry_run_mutates_nothing():
    print("4. dry run")
    pcb, _ends = _board()
    before = list(pcb.segments)
    _n, _nets, strip, added, stats = smooth_octolinear_chains([], pcb, None, clearance=0.1,
                                                              dry_run=True)
    check("dry run measures the spans", stats.get('spans', 0) >= N_STAIRS, f"{stats.get('spans')}")
    check("  and changes nothing", pcb.segments == before and not strip and not added)


def _board_text(segments, foreign):
    """The _board() layout as a .kicad_pcb: pads as footprints, net 1 tracks,
    and a net-2 track 0.5 mm under each staircase's first jog."""
    pads = []
    for i in range(N_STAIRS):
        y = 1.0 + 4.0 * i
        pads += [(f'P{2 * i}', 1.0, y), (f'P{2 * i + 1}', 1.0 + PITCH * STEPS / 2,
                                          y + PITCH * STEPS / 2)]
    body = [f'''  (footprint "Synth:Pad" (layer "F.Cu") (at {x:.4f} {y:.4f})
    (property "Reference" "{r}" (at 0 -1 0) (layer "F.SilkS") (hide yes)
      (effects (font (size 0.3 0.3) (thickness 0.05))))
    (attr smd)
    (pad "1" smd rect (at 0 0) (size 0.3 0.3) (layers "F.Cu") (net 1 "RAIL"))
  )''' for r, x, y in pads]
    body += [f'  (segment (start {a:.6f} {b:.6f}) (end {c:.6f} {d:.6f}) (width {w}) '
             f'(layer "F.Cu") (net {n}))' for a, b, c, d, w, n in segments + foreign]
    return f'''(kicad_pcb (version 20240108) (generator "synth") (generator_version "8.0")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0))
  (net 0 "")
  (net 1 "RAIL")
  (net 2 "OTHER")
  (gr_rect (start 0 0) (end 10 170) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts"))
{chr(10).join(body)}
)
'''


def written_board_is_clean():
    print("5. the smoothed board, written and graded")
    pcb0, _ends = _board()
    foreign = [(1.0, 0.5 + 4.0 * i, 4.0, 0.5 + 4.0 * i, W, 2) for i in range(N_STAIRS)]
    # One net, so connected: a stub from each first pad to a trunk at x = 0.7.
    ys = [1.0 + 4.0 * i for i in range(N_STAIRS)]
    trunk = [(0.7, ys[i], 0.7, ys[i + 1], W, 1) for i in range(N_STAIRS - 1)]
    stubs = [(0.7, y, 1.0, y, W, 1) for y in ys]
    tracks = [(s.start_x, s.start_y, s.end_x, s.end_y, W, 1) for s in pcb0.segments]
    tracks += trunk + stubs
    py = [sys.executable, '-X', 'utf8']
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, 'in.kicad_pcb'), os.path.join(td, 'out.kicad_pcb')
        with open(src, 'w', encoding='utf-8') as f:
            f.write(_board_text(tracks, foreign))
        pcb = parse_kicad_pcb(src)
        n0 = sum(1 for s in pcb.segments if s.net_id == 1)
        smooth_octolinear_chains([], pcb, None, clearance=0.1)
        rail = [(s.start_x, s.start_y, s.end_x, s.end_y, W, 1) for s in pcb.segments
                if s.net_id == 1]
        check("the parsed board's net is smoothed", len(rail) < n0, f"{n0} -> {len(rail)}")
        with open(out, 'w', encoding='utf-8') as f:
            f.write(_board_text(rail, foreign))
        evidence(out, 'smoothed board')
        for name, path in (('input', src), ('smoothed', out)):
            d = run_check(py + [os.path.join(REPO, 'py_router', 'check_drc.py'), path,
                                '--clearance', '0.1'], accept=True)
            check(f"check_drc.py on the {name} board: clean",
                  'NO DRC VIOLATIONS FOUND' in d.stdout)
            run_check(py + [os.path.join(REPO, 'py_router', 'check_connected.py'), path],
                      accept=True)
            check(f"check_connected.py on the {name} board: clean (exit 0)", True)


if __name__ == '__main__':
    large_net_is_smoothed()
    worse_net_replays_chain_by_chain()
    dry_run_mutates_nothing()
    written_board_is_clean()
    print("-" * 60)
    if fails:
        print(f"{len(fails)} FAILED: " + "; ".join(fails))
        sys.exit(1)
    print("ALL PASS")
