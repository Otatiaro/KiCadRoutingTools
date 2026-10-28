#!/usr/bin/env python3
"""A plane tap ships as the few segments its corners need, on a net of any size.

A tap's trace is built from the A* grid path, which holds one point per grid
cell, and `_path_to_segments` emitted one segment per cell: a straight 2.9 mm
tap at a 0.05 mm grid was 58 segments. The end-of-run collinear merge (#811)
was the only thing joining them back, and it skipped any net above 4000
segments -- the ground net of a dense board, which is where the taps are.
Measured on a 4-layer board with ~250 plane pads: 5555 ground segments, 98 %
of them under 0.3 mm, 4608 bit-exact collinear joints left in the output.

Checks:
  1. `_path_to_segments` collapses collinear grid points before emitting,
     keeps every corner and every off-line stub, and moves no copper (same
     length, every per-cell capsule covered by an emitted one).
  2. `merge_collinear_segments` merges a net above the old 4000-segment cap,
     in linear time.
  3. End to end: `repair_planes.py` taps four boxed-in pads of a synthetic
     board with the final merge ablated (KICAD_MERGE_COLLINEAR=0), so only the
     emission is graded. Each tap is reported as one trace segment, the
     shipped ground copper carries no collinear joint, and `check_drc.py` and
     `check_connected.py` pass.

    python3 tests/test_tap_collinear_emission.py
"""
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from pcb_modification import merge_collinear_segments  # noqa: E402
from route_planes import _path_to_segments  # noqa: E402
from routing_config import GridCoord, GridRouteConfig  # noqa: E402
from run_utils import check as run_check, evidence, tool_env  # noqa: E402
from synth import make_pad, make_pcb, make_seg, make_via  # noqa: E402

GRID = 0.05
W = 0.2
fails = []


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f"  {detail}" if detail else ''))
    if not cond:
        fails.append(name)


def _cfg():
    return replace(GridRouteConfig(track_width=W, clearance=0.2, via_size=0.6, via_drill=0.3,
                                   grid_step=GRID, layers=['F.Cu', 'B.Cu']),
                   power_net_widths={})


def _length(segs):
    return math.fsum(math.hypot(s['end'][0] - s['start'][0], s['end'][1] - s['start'][1])
                     for s in segs)


def _covered(p, q, segs, tol=1e-6):
    """Does the capsule p-q lie on one of `segs`, within its extent?"""
    for s in segs:
        (ax, ay), (bx, by) = s['start'], s['end']
        L = math.hypot(bx - ax, by - ay)
        if L < 1e-12:
            continue
        ux, uy = (bx - ax) / L, (by - ay) / L
        ok = True
        for px, py in (p, q):
            if abs(-uy * (px - ax) + ux * (py - ay)) > tol:
                ok = False
            along = ux * (px - ax) + uy * (py - ay)
            if along < -tol or along > L + tol:
                ok = False
        if ok:
            return True
    return False


def _tap(cells, via_pos, pad_xy):
    """Emit a tap along grid `cells`, and the per-cell polyline it replaces."""
    coord = GridCoord(GRID)
    path = [(gx, gy, 0) for gx, gy in cells]
    pad = make_pad(net_id=1, x=pad_xy[0], y=pad_xy[1], size_x=0.3, size_y=0.3)
    segs = _path_to_segments(path, via_pos, pad, 'F.Cu', 1, _cfg(), coord)
    poly = [tuple(via_pos)] + [coord.to_float(gx, gy) for gx, gy in cells] + [pad_xy]
    pieces = [(poly[i], poly[i + 1]) for i in range(len(poly) - 1)
              if math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1]) > 1e-9]
    return segs, pieces


def _same_copper(label, segs, pieces):
    ref = math.fsum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in pieces)
    check(f"{label}: copper length unchanged", abs(_length(segs) - ref) < 1e-9,
          f"{ref:.6f} -> {_length(segs):.6f} mm")
    check(f"{label}: every grid step lies on an emitted segment",
          all(_covered(p, q, segs) for p, q in pieces))


def emission():
    print("1. _path_to_segments")
    # Straight 2.9 mm run, via and pad on the grid line: one segment.
    cells = [(x, 120) for x in range(128, 69, -1)]
    segs, pieces = _tap(cells, (6.4, 6.0), (3.5, 6.0))
    check("a straight tap is ONE segment (was one per grid cell)", len(segs) == 1,
          f"{len(segs)} segment(s) for {len(pieces)} grid steps")
    check("  from the via to the pad centre",
          len(segs) == 1 and segs[0]['start'] == (6.4, 6.0) and segs[0]['end'] == (3.5, 6.0))
    _same_copper("  straight", segs, pieces)

    # Off-grid via and pad, still on the line: the stubs merge too.
    segs, pieces = _tap([(x, 120) for x in range(128, 69, -1)], (6.42, 6.0), (3.48, 6.0))
    check("collinear stubs merge into the run", len(segs) == 1, f"{len(segs)}")
    _same_copper("  collinear stubs", segs, pieces)

    # Axis run then a 45-degree run: two segments, the corner kept.
    cells = [(x, 120) for x in range(140, 119, -1)] + [(119 - k, 121 + k) for k in range(20)]
    segs, pieces = _tap(cells, (7.0, 6.0), (5.0, 7.0))
    check("an axis + diagonal tap keeps its one corner", len(segs) == 2, f"{len(segs)}")
    _same_copper("  corner", segs, pieces)

    # A stub that is NOT on the line stays its own segment.
    segs, pieces = _tap([(x, 120) for x in range(128, 69, -1)], (6.41, 6.02), (3.5, 6.0))
    check("an off-line via stub is kept", len(segs) == 2, f"{len(segs)}")
    _same_copper("  off-line stub", segs, pieces)


def large_net():
    print("2. merge_collinear_segments on a net above 4000 segments")
    n_taps, pieces = 300, 15
    segs, vias, pads = [], [], []
    for t in range(n_taps):
        y = 1.0 + 0.5 * t
        vias.append(make_via(0.0, y, net_id=1))
        pads.append(make_pad(net_id=1, x=pieces * GRID, y=y, size_x=0.3, size_y=0.3,
                             ref=f'P{t}'))
        segs += [make_seg(k * GRID, y, (k + 1) * GRID, y, width=W) for k in range(pieces)]
    pcb = make_pcb(segments=segs, vias=vias, pads_by_net={1: pads})
    t0 = time.time()
    _n, _nets, _strip, _added, stats = merge_collinear_segments([], pcb, None)
    dt = time.time() - t0
    check(f"a {len(segs)}-segment net is merged, not skipped",
          len(pcb.segments) == n_taps, f"{len(segs)} -> {len(pcb.segments)} segments")
    check("  every tap spans via to pad",
          all(abs(s.start_x) < 1e-9 and abs(s.end_x - pieces * GRID) < 1e-9
              for s in pcb.segments))
    check("  in linear time", dt < 5.0, f"{dt:.2f} s")


BOARD = '''(kicad_pcb (version 20240108) (generator "synth") (generator_version "8.0")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0))
  (net 0 "")
  (net 1 "GND")
{nets}
  (gr_rect (start 0 0) (end 20 20) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts"))
  (zone (net 1) (net_name "GND") (layer "B.Cu") (hatch edge 0.5) (connect_pads (clearance 0.2))
    (min_thickness 0.2) (fill yes (thermal_gap 0.2) (thermal_bridge_width 0.3))
    (polygon (pts (xy 0.5 0.5) (xy 19.5 0.5) (xy 19.5 19.5) (xy 0.5 19.5))))
{body}
)
'''


def _board(path):
    """Four GND pads, each at the closed end of a foreign-net corridor too
    narrow for a via: the tap must run 2.9 mm along it to the open end."""
    nets, body = [], []
    for i, y in enumerate((6.0, 9.0, 12.0, 15.0)):
        net = i + 2
        nets.append(f'  (net {net} "S{i}")')
        for a, b, c, d in ((3.0, y - 0.5, 6.0, y - 0.5), (3.0, y + 0.5, 6.0, y + 0.5),
                           (3.0, y - 0.5, 3.0, y + 0.5)):
            body.append(f'  (segment (start {a} {b}) (end {c} {d}) (width {W}) '
                        f'(layer "F.Cu") (net {net}))')
        body.append(f'''  (footprint "Synth:Pad" (layer "F.Cu") (at 3.5 {y})
    (property "Reference" "P{i + 1}" (at 0 -1 0) (layer "F.SilkS") (hide yes)
      (effects (font (size 0.3 0.3) (thickness 0.1))))
    (attr smd)
    (pad "1" smd rect (at 0 0) (size 0.3 0.3) (layers "F.Cu") (net 1 "GND"))
  )''')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(BOARD.format(nets='\n'.join(nets), body='\n'.join(body)))


def _collinear_joints(path, net_id):
    from kicad_parser import parse_kicad_pcb
    pcb = parse_kicad_pcb(path)
    ends = {}
    for s in pcb.segments:
        if s.net_id != net_id:
            continue
        for p, q in (((s.start_x, s.start_y), (s.end_x, s.end_y)),
                     ((s.end_x, s.end_y), (s.start_x, s.start_y))):
            ends.setdefault((s.layer, round(p[0], 4), round(p[1], 4)), []).append((p, q))
    vias = {(round(v.x, 4), round(v.y, 4)) for v in pcb.vias if v.net_id == net_id}
    n = 0
    for (layer, x, y), legs in ends.items():
        if len(legs) != 2 or (x, y) in vias:
            continue
        (p, a), (_, b) = legs
        ua, ub = (a[0] - p[0], a[1] - p[1]), (b[0] - p[0], b[1] - p[1])
        if abs(ua[0] * ub[1] - ua[1] * ub[0]) < 1e-9 and ua[0] * ub[0] + ua[1] * ub[1] < 0:
            n += 1
    return n


def end_to_end():
    print("3. repair_planes.py on a synthetic board, final merge ablated")
    py = [sys.executable, '-X', 'utf8']
    env = tool_env(dict(os.environ, KICAD_MERGE_COLLINEAR='0'))
    with tempfile.TemporaryDirectory() as td:
        src, out = os.path.join(td, 'in.kicad_pcb'), os.path.join(td, 'out.kicad_pcb')
        _board(src)
        r = run_check(py + [os.path.join(ROOT, 'py_router', 'repair_planes.py'), src, out,
                            '--nets', 'GND', '--plane-layers', 'B.Cu', '--repair-pads',
                            '--clearance', '0.2', '--track-width', str(W), '--via-size', '0.6',
                            '--via-drill', '0.3', '--grid-step', str(GRID),
                            '--fab-tier', 'standard', '--escalation', 'off'],
                      accept=True, env=env)
        evidence(out, 'repair_planes output')
        taps = [int(m) for m in re.findall(r'placed via at \([^)]*\), (\d+) trace segment',
                                           r.stdout)]
        check("the pads were tapped through the corridor (the path under test ran)",
              len(taps) >= 3, f"{len(taps)} tap(s)")
        check("each straight tap is emitted as ONE trace segment",
              bool(taps) and all(n == 1 for n in taps), f"{taps}")
        joints = _collinear_joints(out, 1)
        check("the shipped GND copper has no collinear joint", joints == 0, f"{joints}")
        d = run_check(py + [os.path.join(ROOT, 'py_router', 'check_drc.py'), out], accept=True)
        check("check_drc.py: clean", 'NO DRC VIOLATIONS FOUND' in d.stdout)
        run_check(py + [os.path.join(ROOT, 'py_router', 'check_connected.py'), out], accept=True)
        check("check_connected.py: clean (exit 0)", True)


if __name__ == '__main__':
    emission()
    large_net()
    end_to_end()
    print("-" * 60)
    if fails:
        print(f"{len(fails)} FAILED: " + "; ".join(fails))
        sys.exit(1)
    print("ALL PASS")
