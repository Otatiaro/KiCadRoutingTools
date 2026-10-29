#!/usr/bin/env python3
"""An oracle link keeps its new via off the net's own SMD pads (#581).

`--same-net-pad-clearance` keeps every new via at pad edge + via/2 + the
clearance from the net's own SMD pads: no via in a pad, which is a paid
filled-and-capped option at most fabs. The engine's routers stamp that
keep-out (`add_same_net_via_clearance`), and repair_planes hands the value to
oracle_reconnect's config (#1290). The oracle's weld router never stamped it:
its obstacle maps come from build_base_obstacles alone. A link from a floating
plane pad to plane copper on the far side then dropped its layer change
straight into the pad, the shortest path; on a 4-layer board 24 such vias
shipped from one repair_planes run (0402 pads, vias 0.01-0.46 mm off their
centres), each reported as a VIA_IN_PAD fab note.

The board: net A's 0402-sized pad on F.Cu at P and a B.Cu track of net A
starting right under it. The link source and the refill are stubbed (as in
test_oracle_link_own_via_h2h); the router is the real one. Cases:

  * keepout -- same_net_pad_clearance 0.09: the link is routed, its via sits
               at least pad edge + via/2 + 0.09 from the pad, and the copper
               joins the pad to the track;
  * off     -- same_net_pad_clearance unset: the run is unchanged (control:
               the via may sit on the pad, the link is routed).

    python3 tests/test_oracle_link_same_net_pad_via.py
"""
import contextlib
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, TESTS)

from run_utils import evidence                          # noqa: E402

fails = []

PAD = (5.0, 10.0)
PAD_HALF = 0.3                  # the pad is 0.6 x 0.6
VIA = 0.45
SNPC = 0.09
A_END = (PAD[0], PAD[1], 'F.Cu')
B_END = (8.0, 10.0, 'B.Cu')
F, I1, I2, B = 'F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f"  {detail}" if detail else ''))
    if not cond:
        fails.append(name)


def _board(path):
    """A 4-layer KiCad-10 board: net A's floating SMD pad on F.Cu and a net-A
    B.Cu track that starts right under it (the link's two ends), no via."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'''(kicad_pcb (version 20250114) (generator "pcbnew") (generator_version "10.0")
  (general (thickness 1.6))
  (paper "A4")
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal)
    (25 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0))
  (gr_rect (start 0 0) (end 20 20) (stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts"))
  (footprint "T:P" (layer "F.Cu") (at {PAD[0]} {PAD[1]})
    (property "Reference" "R1" (at 0 -1.5 0) (layer "F.SilkS"))
    (attr smd)
    (pad "1" smd rect (at 0 0) (size 0.6 0.6) (layers "F.Cu" "F.Paste" "F.Mask") (net "A")))
  (segment (start {PAD[0]} {PAD[1]}) (end 12 10) (width 0.2) (layer "B.Cu") (net "A"))
)
''')


def _run(snpc):
    """One oracle round over a fresh board, the link source and the refill
    stubbed. Returns (pcb of the written board, oracle stdout)."""
    import kicad_oracle
    import kicad_exact_fill
    from kicad_parser import parse_kicad_pcb
    from routing_config import GridRouteConfig
    calls = {'exact': 0}

    def fake_exact(board_file, net_names=None, pcb_data=None,
                   verbose=False, project_from=None):
        calls['exact'] += 1
        if calls['exact'] > 1:
            return []
        return [('A', A_END + ('pad',), B_END + ('track',))]

    saved = (kicad_exact_fill.exact_unconnected, kicad_oracle.kicad_unconnected,
             kicad_exact_fill.refill_islands)
    kicad_exact_fill.exact_unconnected = fake_exact
    kicad_oracle.kicad_unconnected = lambda *a, **k: None
    kicad_exact_fill.refill_islands = lambda *a, **k: None
    out = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as td:
            board = os.path.join(td, 'b.kicad_pcb')
            _board(board)
            cfg = GridRouteConfig(clearance=0.1, track_width=0.2,
                                  via_size=VIA, via_drill=0.3, grid_step=0.05,
                                  layers=[F, I1, I2, B],
                                  same_net_pad_clearance=snpc)
            with contextlib.redirect_stdout(out):
                kicad_oracle.oracle_reconnect(
                    board, ['A'], cfg, track_via_clearance=0.1,
                    hole_to_hole_clearance=0.2, max_rounds=1)
            pcb = parse_kicad_pcb(evidence(board, 'the board the oracle wrote'))
    finally:
        (kicad_exact_fill.exact_unconnected, kicad_oracle.kicad_unconnected,
         kicad_exact_fill.refill_islands) = saved
    return pcb, out.getvalue()


def _joined(pcb, nid):
    """True when net nid's copper joins the pad to the B.Cu track: segment
    ends on one layer, vias on every layer, the pad's rectangle on F.Cu."""
    parent = {}

    def find(n):
        parent.setdefault(n, n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(p, q):
        parent[find(p)] = find(q)

    def key(x, y, layer):
        if layer == F and abs(x - PAD[0]) <= PAD_HALF + 1e-6 and abs(y - PAD[1]) <= PAD_HALF + 1e-6:
            return ('pad',)
        return (round(x, 3), round(y, 3), layer)
    for s in pcb.segments:
        if s.net_id == nid:
            union(key(s.start_x, s.start_y, s.layer), key(s.end_x, s.end_y, s.layer))
    for v in pcb.vias:
        if v.net_id == nid:
            for layer in (F, I1, I2, B):
                union(key(v.x, v.y, F), key(v.x, v.y, layer))
    return find(('pad',)) == find(key(PAD[0], PAD[1], B))


def _pad_gap(v):
    """Edge-to-edge distance from via v's copper to the pad's rectangle."""
    dx = max(abs(v.x - PAD[0]) - PAD_HALF, 0.0)
    dy = max(abs(v.y - PAD[1]) - PAD_HALF, 0.0)
    return (dx * dx + dy * dy) ** 0.5 - v.size / 2


for name, snpc in (('keepout', SNPC), ('off', -1.0)):
    print(f"{name}:")
    pcb, log = _run(snpc)
    ok_line = next((ln for ln in log.splitlines() if ' OK ' in ln and 'A:' in ln), '')
    check("the oracle reported the link OK", bool(ok_line), ok_line.strip() or log[-600:])
    nid = next(i for i, n in pcb.nets.items() if n.name == 'A')
    nv = [v for v in pcb.vias if v.net_id == nid]
    check("the link placed a via", bool(nv))
    if name == 'keepout':
        close = [((v.x, v.y), round(_pad_gap(v), 3)) for v in nv
                 if _pad_gap(v) < SNPC - 0.01]
        check(f"every via clears the pad by >= {SNPC} mm (edge to edge)", not close,
              f"{close}")
    check("the net's copper joins the pad to the track", _joined(pcb, nid))

print("-" * 60)
if fails:
    print(f"{len(fails)} FAILED: " + "; ".join(fails))
    sys.exit(1)
print("ALL PASS")
