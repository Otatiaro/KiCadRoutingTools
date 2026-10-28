#!/usr/bin/env python3
"""The cycle prune breaks every track loop of a net, through pads and on zoned
nets, and drops the branch that carries the least current.

prune_redundant_cycles keeps a net's track copper a tree. Three shapes of
loop survived it:

  * a loop CLOSED THROUGH AN SMD PAD -- two branches landing on the same pad
    at different points (a via tapped to a pad twice, a junction reaching a
    switch pad by two paths). Track ends were only merged when coincident or
    inside a via / through-hole pad, so the two landings were two nodes and
    the loop was invisible;
  * any loop on a ZONED net. The pass skipped zoned nets as 'meshes, not
    trees', so on a board with planes a 0.25 mm track running 30 mm beside a
    2 mm power trunk, or a tap laid twice by two repair passes, shipped.
    The pass reads tracks, vias and pads only and credits no zone, so a
    track loop is a loop whatever the planes do; a plane tap (a pad's leaf
    branch to its via) is on no loop and stays. A branch lying in a pour of
    its own net on its own layer may be what ties that pour together, so it
    stays too;
  * the WRONG branch went: the segment closing the loop was the LONGEST one,
    so a power trunk lost its long middle segment to a thin parallel track.
    The branch dropped is now the one of highest resistance (length outside
    pad and via copper over width), removed whole.

Each case is a small board graded on the pass directly; the last runs it
through route.py and grades the output with check_drc and check_connected.

    python3 tests/test_cycle_prune_pad_zone_loops.py
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from kicad_parser import BoardInfo, Zone                            # noqa: E402
from synth import (RoutingBoard, make_net, make_pad, make_pcb,      # noqa: E402
                   make_seg, make_via)
from run_utils import check as run_check, evidence, tool            # noqa: E402
from check_connected import check_net_connectivity                  # noqa: E402
import pcb_modification as pm                                       # noqa: E402

LAYERS = ['F.Cu', 'In1.Cu', 'B.Cu']
failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def _board(segs, vias, pads, zones=()):
    bi = BoardInfo(layers={0: 'F.Cu', 1: 'In1.Cu', 31: 'B.Cu'},
                   copper_layers=list(LAYERS),
                   board_bounds=(-20.0, -20.0, 60.0, 60.0))
    by_net = {}
    for p in pads:
        by_net.setdefault(p.net_id, []).append(p)
    return make_pcb(nets={1: make_net(1, 'A')}, segments=segs, vias=vias,
                    pads_by_net=by_net, board_info=bi, zones=list(zones))


def _zone(layer, x0=-10, y0=-10, x1=30, y1=10):
    return Zone(net_id=1, net_name='A', layer=layer,
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _prune(pcb):
    with contextlib.redirect_stdout(io.StringIO()):
        return pm.prune_redundant_cycles([], pcb, {1}, clearance=0.1)


def _left(pcb, segs):
    ids = {id(s) for s in pcb.segments}
    return [s for s in segs if id(s) in ids]


def _same_connectivity(pcb, before_segs):
    pads = pcb.pads_by_net.get(1, [])
    b = check_net_connectivity(1, before_segs, pcb.vias, pads)
    a = check_net_connectivity(1, pcb.segments, pcb.vias, pads)
    return (len(a.get('disconnected_pads') or []) <= len(b.get('disconnected_pads') or [])
            and (a.get('num_components') or 1) <= (b.get('num_components') or 1))


# --- a loop closed through an SMD pad ----------------------------------------

def case_pad_loop():
    print("loop closed through an SMD pad (two landings on one pad)")
    p1 = make_pad(1, 0, 0, ref='SW1', size_x=1.0, size_y=1.0)
    p2 = make_pad(1, 10, 0, ref='R1', size_x=1.0, size_y=1.0)
    feed = [make_seg(10, 0, 5, 3), make_seg(5, 3, 0, 3)]
    direct = make_seg(0, 3, 0, 0.3)                       # lands on the top edge
    detour = [make_seg(0, 3, -2, 3), make_seg(-2, 3, -2, 0),
              make_seg(-2, 0, -0.4, 0)]                    # lands on the left edge
    segs = feed + [direct] + detour
    pcb = _board(list(segs), [], [p1, p2])
    _prune(pcb)
    check("the detour branch is removed whole", not _left(pcb, detour),
          f"{len(_left(pcb, detour))} of 3 left")
    check("the feed and the direct landing stay",
          len(_left(pcb, feed + [direct])) == 3)
    check("connectivity unchanged", _same_connectivity(pcb, segs))


def case_via_tapped_twice():
    print("via tapped to its pad twice (a straight tap and a staircase)")
    p1 = make_pad(1, 0, 0, ref='C1', size_x=0.6, size_y=0.6)
    p2 = make_pad(1, 6, 0, ref='C2', size_x=0.6, size_y=0.6)
    v = make_via(0, 0.9, size=0.45, drill=0.3)
    straight = make_seg(0, 0.9, 0, 0, width=0.127)
    # The staircase's first vertex lies outside the via's copper: copper
    # inside a via or a pad is the strict collapse's, not this pass's.
    stairs = [make_seg(0, 0.9, 0.2, 0.6, width=0.09),
              make_seg(0.2, 0.6, 0.2, 0.4, width=0.09),
              make_seg(0.2, 0.4, 0.2, 0.25, width=0.09)]
    link = make_seg(0, 0, 6, 0, width=0.2)
    back = make_seg(0, 0.9, 0, 3, layer='B.Cu', width=0.2)  # the via's other side
    segs = [straight] + stairs + [link, back]
    pcb = _board(list(segs), [v], [p1, p2])
    _prune(pcb)
    left = _left(pcb, [straight] + stairs)
    check("one of the two taps is removed whole",
          left == [straight] or left == stairs, f"{len(left)} tap segment(s) left")
    check("the via keeps a tap to the pad", bool(left))
    check("connectivity unchanged", _same_connectivity(pcb, segs))


# --- which branch goes --------------------------------------------------------

def _trunk_and_thin():
    """A 2 mm trunk (pad -> via, B.Cu run, via -> pad) beside a 0.25 mm F.Cu
    track between the same two pads. The trunk's B.Cu run is the longest
    single segment of the loop."""
    p1 = make_pad(1, 0, 0, ref='J1', size_x=2.0, size_y=2.0)
    p2 = make_pad(1, 20, 0, ref='J2', size_x=2.0, size_y=2.0)
    v1, v2 = make_via(0, -3, size=0.8, drill=0.4), make_via(20, -3, size=0.8, drill=0.4)
    trunk = [make_seg(0, 0, 0, -3, width=2.0),
             make_seg(0, -3, 20, -3, width=2.0, layer='B.Cu'),
             make_seg(20, -3, 20, 0, width=2.0)]
    thin = [make_seg(0, 0, 10, 1.5, width=0.25), make_seg(10, 1.5, 20, 0, width=0.25)]
    return [p1, p2], [v1, v2], trunk, thin


def case_trunk_kept():
    print("power trunk beside a thin parallel track (multi-layer loop)")
    pads, vias, trunk, thin = _trunk_and_thin()
    segs = trunk + thin
    pcb = _board(list(segs), vias, pads)
    _prune(pcb)
    check("the 2 mm trunk stays whole", len(_left(pcb, trunk)) == 3,
          f"{len(_left(pcb, trunk))} of 3 left")
    check("the thin track is removed whole", not _left(pcb, thin),
          f"{len(_left(pcb, thin))} of 2 left")
    check("connectivity unchanged", _same_connectivity(pcb, segs))


# --- zoned nets ---------------------------------------------------------------

def case_zoned_net():
    print("the same loop on a net with a plane on In1.Cu")
    pads, vias, trunk, thin = _trunk_and_thin()
    pcb = _board(trunk + thin, vias, pads, [_zone('In1.Cu')])
    _prune(pcb)
    check("the thin track is removed on the zoned net", not _left(pcb, thin),
          f"{len(_left(pcb, thin))} of 2 left")
    check("the trunk stays", len(_left(pcb, trunk)) == 3)


def case_zone_on_branch_layer():
    print("a loop whose branches lie in a pour of their net on their layer")
    pads, vias, trunk, thin = _trunk_and_thin()
    pcb = _board(trunk + thin, vias, pads, [_zone('F.Cu', -5, -1, 25, 5)])
    _prune(pcb)
    check("the in-pour thin track is left alone (it may tie the pour)",
          len(_left(pcb, thin)) == 2)


def case_plane_tap_kept():
    print("plane taps on a zoned net (leaf branches, no loop)")
    p1 = make_pad(1, 0, 0, ref='U1', size_x=0.6, size_y=0.6)
    p2 = make_pad(1, 5, 0, ref='U2', size_x=0.6, size_y=0.6)
    v1, v2 = make_via(0, 1.0, size=0.45, drill=0.3), make_via(5, 1.0, size=0.45, drill=0.3)
    segs = [make_seg(0, 0, 0, 1.0), make_seg(5, 0, 5, 1.0), make_seg(0, 0, 5, 0)]
    pcb = _board(list(segs), [v1, v2], [p1, p2], [_zone('In1.Cu')])
    _prune(pcb)
    check("every tap and the link stay", len(_left(pcb, segs)) == 3)


def case_stacked_on_zoned_net():
    print("a tap laid twice, stacked, on a zoned net")
    p1 = make_pad(1, 0, 0, ref='U1', size_x=0.6, size_y=0.6)
    p2 = make_pad(1, 5, 0, ref='U2', size_x=0.6, size_y=0.6)
    v = make_via(0, 1.0, size=0.45, drill=0.3)
    a, b = make_seg(0, 0.4, 0, 1.0, width=0.09), make_seg(0, 0.4, 0, 1.0, width=0.09)
    # Inside pad U2: a pair stacked on one span (the narrower one goes), and
    # a lone in-pad segment, which is the strict collapse's to grade.
    wide, narrow = make_seg(4.85, 0, 5.15, 0, width=0.2), make_seg(5.15, 0, 4.85, 0, width=0.09)
    lone = make_seg(5.0, -0.2, 5.0, 0.2, width=0.09)
    segs = [a, b, make_seg(0, 0, 0, 0.4, width=0.09), make_seg(0, 0, 5, 0),
            narrow, wide, lone]
    pcb = _board(list(segs), [v], [p1, p2], [_zone('In1.Cu')])
    _prune(pcb)
    check("exactly one of the stacked pair is removed", len(_left(pcb, [a, b])) == 1)
    check("of the pair stacked inside a pad, the narrower goes",
          _left(pcb, [wide, narrow]) == [wide])
    check("a lone in-pad segment stays", _left(pcb, [lone]) == [lone])


def case_no_new_spur():
    print("off-grid branch ending on a vertex merged with its neighbour's")
    # Hand-drawn copper off a fine-pitch pad, as found on a routed board: a
    # 0.05 mm staircase 0.01 mm off the pad's axis, and a straight run on the
    # axis whose last two segments (a diagonal, then a 0.11 mm vertical) come
    # back onto the staircase. The vertical's foot is a staircase vertex; once
    # the loop's branch goes, nothing may be left with a free end it did not
    # have (the stub's free end overlaps the staircase, so the dead-end sweep
    # counts it anchored, while KiCad flags track_dangling).
    p1 = make_pad(1, 0, 0, ref='U9', size_x=1.6, size_y=0.3, shape='roundrect')
    p2 = make_pad(1, 1.9379, -0.3601, ref='C1', size_x=0.4, size_y=0.4)
    stairs = [make_seg(*xy, width=w) for *xy, w in SPUR_STAIRS]
    run = make_seg(0.0, 0.0, 0.85, 0.0, width=0.25)
    diagonal = make_seg(0.85, 0.0, 0.984908, -0.123033, width=0.25)
    vertical = make_seg(0.984908, -0.123033, 0.984908, -0.0101, width=0.25)
    segs = stairs + [run, diagonal, vertical]
    pcb = _board(list(segs), [], [p1, p2])
    _prune(pcb)
    check("the loop is broken", len(_left(pcb, segs)) < len(segs))
    check("no stub is left where its diagonal went",
          not (_left(pcb, [vertical]) and not _left(pcb, [diagonal])))
    check("connectivity unchanged", _same_connectivity(pcb, segs))


# (x1, y1, x2, y2, width): the staircase of case_no_new_spur.
SPUR_STAIRS = [
    (-0.0121, -0.0101, 0.0, 0.0, 0.25), (0.0379, -0.0101, -0.0121, -0.0101, 0.25),
    (0.0879, -0.0101, 0.0379, -0.0101, 0.25), (0.1379, -0.0101, 0.0879, -0.0101, 0.25),
    (0.1879, -0.0101, 0.1379, -0.0101, 0.25), (0.2379, -0.0101, 0.1879, -0.0101, 0.25),
    (0.2879, -0.0101, 0.2379, -0.0101, 0.25), (0.3379, -0.0101, 0.2879, -0.0101, 0.25),
    (0.3879, -0.0101, 0.3379, -0.0101, 0.25), (0.4379, -0.0101, 0.3879, -0.0101, 0.25),
    (0.4879, -0.0101, 0.4379, -0.0101, 0.25), (0.5379, -0.0101, 0.4879, -0.0101, 0.25),
    (0.5879, -0.0101, 0.5379, -0.0101, 0.25), (0.6379, -0.0101, 0.5879, -0.0101, 0.25),
    (0.6879, -0.0101, 0.6379, -0.0101, 0.25), (0.7379, -0.0101, 0.6879, -0.0101, 0.25),
    (0.7879, -0.0101, 0.7379, -0.0101, 0.25), (0.8379, -0.0101, 0.7879, -0.0101, 0.25),
    (0.8879, -0.0101, 0.8379, -0.0101, 0.25), (0.9379, -0.0101, 0.8879, -0.0101, 0.25),
    (0.9879, -0.0101, 0.9379, -0.0101, 0.25), (1.0379, -0.0101, 0.9879, -0.0101, 0.25),
    (1.0879, -0.0101, 1.0379, -0.0101, 0.25), (1.1379, -0.0101, 1.0879, -0.0101, 0.25),
    (1.1879, -0.0101, 1.1379, -0.0101, 0.25), (1.2379, -0.0101, 1.1879, -0.0101, 0.25),
    (1.2879, -0.0101, 1.2379, -0.0101, 0.23), (1.3379, -0.0601, 1.2879, -0.0101, 0.23),
    (1.3879, -0.0601, 1.3379, -0.0601, 0.17), (1.4379, -0.1101, 1.3879, -0.0601, 0.17),
    (1.4879, -0.1101, 1.4379, -0.1101, 0.17), (1.5379, -0.1601, 1.4879, -0.1101, 0.17),
    (1.5879, -0.1601, 1.5379, -0.1601, 0.21), (1.6379, -0.1601, 1.5879, -0.1601, 0.21),
    (1.6879, -0.1601, 1.6379, -0.1601, 0.21), (1.7379, -0.2101, 1.6879, -0.1601, 0.21),
    (1.7879, -0.2601, 1.7379, -0.2101, 0.25), (1.8379, -0.2601, 1.7879, -0.2601, 0.25),
    (1.9379, -0.3601, 1.8379, -0.2601, 0.25),
]


# --- end to end: route.py, graded by check_drc + check_connected -------------

def case_route_py():
    print("route.py on a board carrying a loop on a zoned net")
    b = RoutingBoard(w=40, h=20, layers=('F.Cu', 'In1.Cu', 'B.Cu'))
    b.pad('J1', 5, 10, 'VIN', sx=2.0, sy=2.0)
    b.pad('J2', 25, 10, 'VIN', sx=2.0, sy=2.0)
    b.seg(5, 10, 5, 6, 'VIN', w=1.0)
    b.via(5, 6, 'VIN', size=0.8, drill=0.4)
    b.seg(5, 6, 25, 6, 'VIN', w=1.0, layer='B.Cu')
    b.via(25, 6, 'VIN', size=0.8, drill=0.4)
    b.seg(25, 6, 25, 10, 'VIN', w=1.0)
    b.seg(5, 10, 15, 13, 'VIN', w=0.25, uid='thin-a')
    b.seg(15, 13, 25, 10, 'VIN', w=0.25, uid='thin-b')
    b.zone('In1.Cu', 1, 1, 39, 19, 'VIN')
    b.pad('S1', 5, 16, 'SIG')
    b.pad('S2', 35, 16, 'SIG')
    work = tempfile.mkdtemp(prefix='cycle_prune_')
    try:
        src = os.path.join(work, 'in.kicad_pcb')
        out = os.path.join(work, 'out.kicad_pcb')
        with open(src, 'w', encoding='utf-8') as fh:
            fh.write(b.text())
        run_check([sys.executable, tool('route.py'), src, out,
                   '--nets', 'VIN', 'SIG', '--clearance', '0.2',
                   '--layers', 'F.Cu', 'In1.Cu', 'B.Cu',
                   '--layer-costs', '1', '-1', '1'], accept=True)
        evidence(out)
        from kicad_parser import parse_kicad_pcb
        routed = parse_kicad_pcb(out)
        vin = next(i for i, n in routed.nets.items() if n.name == 'VIN')
        vin_segs = [s for s in routed.segments if s.net_id == vin]
        # The smoother may straighten the thin track, so grade the copper,
        # not the uuids: only the trunk's three 1 mm runs may remain.
        thin = [s for s in vin_segs if s.width < 0.5]
        check("the thin parallel track is gone", not thin,
              f"{len(thin)} thin VIN segment(s) left")
        check("the trunk stays whole",
              sum(1 for s in vin_segs if s.width >= 0.99) == 3)
        for name in ('check_drc.py', 'check_connected.py'):
            r = subprocess.run([sys.executable, tool(name), out], capture_output=True,
                               text=True, encoding='utf-8', errors='replace')
            check(f"{name} clean", r.returncode == 0,
                  '' if r.returncode == 0 else (r.stdout + r.stderr)[-600:])
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    for case in (case_pad_loop, case_via_tapped_twice, case_trunk_kept,
                 case_zoned_net, case_zone_on_branch_layer, case_plane_tap_kept,
                 case_stacked_on_zoned_net, case_no_new_spur, case_route_py):
        case()
    if failures:
        print(f"\nFAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nPASSED")
