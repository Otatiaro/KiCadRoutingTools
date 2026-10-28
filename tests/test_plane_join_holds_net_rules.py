#!/usr/bin/env python3
"""A plane's region join holds the board's net rules from its .kicad_dru.

repair_planes joins the islands of a split plane with tracks of its own A*
(plane_region_connector). That search ran on the base obstacle map without
the per-route stamps the other routing entries take (net_keepouts.stamp), and
repair_planes did not derive the net-filtered keep-out areas at all, so a
join honoured neither a .kicad_dru net-pair clearance nor a `disallow` rule
on a named area: on a real board a +VIN join ran 0.775 mm from an isolated
net held at 1 mm, through the area that keeps other nets off the isolated
side.

The boards: a GND pour on B.Cu cut in two by an ISO_W track wall, a GND
through-hole pad in each half, and on F.Cu an ISO_W wall with two gaps. A
join can only cross on F.Cu, through a gap.

Checks:
  1. pair rule: ISO_* held 1 mm from every other net; the gap facing the
     pads is 1 mm wide, the other 6 mm. The halves are joined and every GND
     track keeps 1 mm from ISO_W (base: 0.85 mm, through the narrow gap).
  2. area rule: one 6 mm gap, under a rule area that disallows tracks and
     vias of every net but ISO_*. No GND track or via enters the area, so
     the halves stay split -- the rule leaves no legal join (base: a 2 mm
     join and a via through the area).

    python3 tests/test_plane_join_holds_net_rules.py
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

from run_utils import evidence                                  # noqa: E402
from synth import RoutingBoard                                  # noqa: E402

failures = []

ISO = ("(A.NetName == 'ISO_*' && !(B.NetName == 'ISO_*') && B.NetName != '' "
       "&& B.NetName != 'unconnected-*') || (B.NetName == 'ISO_*' && "
       "!(A.NetName == 'ISO_*') && A.NetName != '' && A.NetName != 'unconnected-*')")
PAIR_DRU = f'''(version 1)
(rule "isolation" (condition "{ISO}") (constraint clearance (min 1mm)))
'''
AREA_DRU = '''(version 1)
(rule "barrier" (condition "A.intersectsArea('barrier') && !(A.NetName == 'ISO_*')")
  (constraint disallow track via))
'''
AREA = (15.0, 2.0, 25.0, 10.0)        # the rule area over the upper gap


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def project():
    return {'board': {'design_settings': {'rules': {
                'min_clearance': 0.1, 'min_track_width': 0.15,
                'min_via_diameter': 0.45, 'min_through_hole_diameter': 0.2,
                'min_hole_to_hole': 0.2, 'min_hole_clearance': 0.1,
                'min_copper_edge_clearance': 0.3}, 'rule_severities': {}}},
            'meta': {'filename': 'in.kicad_pro', 'version': 3},
            'net_settings': {'classes': [{'name': 'Default', 'clearance': 0.1,
                                          'track_width': 0.2, 'via_diameter': 0.45,
                                          'via_drill': 0.25}],
                             'meta': {'version': 4}}}


def board(pad_y, wall_f, area=None):
    b = RoutingBoard(w=40, h=30)
    b.pad('G1', 10, pad_y, 'GND', sx=1.6, sy=1.6, tht=True)
    b.pad('G2', 30, pad_y, 'GND', sx=1.6, sy=1.6, tht=True)
    b.pad('W1', 20, 0.6, 'ISO_W', sx=0.6, sy=0.6, tht=True)
    b.pad('W2', 20, 29.4, 'ISO_W', sx=0.6, sy=0.6, tht=True)
    b.seg(20, 0.6, 20, 29.4, 'ISO_W', w=0.3, layer='B.Cu')
    for y0, y1 in wall_f:
        b.seg(20, y0, 20, y1, 'ISO_W', w=0.3, layer='F.Cu')
    b.items.append(
        '  (zone (net %d) (net_name "GND") (layer "B.Cu") (uuid "synth-zone") '
        '(hatch edge 0.5) (connect_pads (clearance 0.25)) (min_thickness 0.2) '
        '(filled_areas_thickness no) (fill yes (thermal_gap 0.3) '
        '(thermal_bridge_width 0.3)) (polygon (pts (xy 1 1) (xy 39 1) '
        '(xy 39 29) (xy 1 29))))' % b.net('GND'))
    if area:
        x0, y0, x1, y1 = area
        b.items.append(
            '  (zone (net 0) (net_name "") (layers "F.Cu" "B.Cu") (uuid "synth-area") '
            '(name "barrier") (hatch edge 0.5) (connect_pads (clearance 0)) '
            '(min_thickness 0.25) (keepout (tracks allowed) (vias allowed) '
            '(pads allowed) (copperpour allowed) (footprints allowed)) '
            f'(polygon (pts (xy {x0} {y0}) (xy {x1} {y0}) (xy {x1} {y1}) (xy {x0} {y1}))))')
    return b.text()


def pt_seg(p, s):
    (x1, y1), (x2, y2) = s
    dx, dy = x2 - x1, y2 - y1
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / L))
    return math.hypot(p[0] - x1 - t * dx, p[1] - y1 - t * dy)


def seg_dist(a, b):
    """Centreline distance of two (non-crossing) segments."""
    return min(pt_seg(a[0], b), pt_seg(a[1], b), pt_seg(b[0], a), pt_seg(b[1], a))


def run(tmp, tag, text, dru):
    from kicad_parser import parse_kicad_pcb
    from repair_planes import repair_planes
    d = os.path.join(tmp, tag)
    os.makedirs(d)
    inp, out = os.path.join(d, 'in.kicad_pcb'), os.path.join(d, 'out.kicad_pcb')
    with open(inp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    with open(os.path.join(d, 'in.kicad_pro'), 'w', encoding='utf-8') as fh:
        json.dump(project(), fh, indent=2)
    with open(os.path.join(d, 'in.kicad_dru'), 'w', encoding='utf-8') as fh:
        fh.write(dru)
    evidence(inp, f'{tag}: synthetic input board')
    with contextlib.redirect_stdout(io.StringIO()):
        repair_planes(inp, out, net_names=['GND'], plane_layers=['B.Cu'])
    evidence(out, f'{tag}: repaired output')
    pcb = parse_kicad_pcb(out)
    name = {nid: n.name for nid, n in pcb.nets.items()}
    f_cu = [s for s in pcb.segments if s.layer == 'F.Cu']
    return ([s for s in f_cu if name.get(s.net_id) == 'GND'],
            [s for s in f_cu if name.get(s.net_id) == 'ISO_W'],
            [v for v in pcb.vias if name.get(v.net_id) == 'GND'])


def ends(s):
    return ((s.start_x, s.start_y), (s.end_x, s.end_y))


def main():
    tmp = tempfile.mkdtemp(prefix='tplanejoin_')
    try:
        print("1. pair rule: ISO_* 1 mm from every other net")
        joins, walls, _ = run(tmp, 'pair', board(15, [(1, 14.5), (15.5, 21), (27, 29)]),
                           PAIR_DRU)
        check("the halves are joined", bool(joins), f"{len(joins)} GND segment(s) on F.Cu")
        worst = min((seg_dist(ends(j), ends(w)) - j.width / 2 - w.width / 2
                     for j in joins for w in walls), default=float('inf'))
        check("every join keeps 1 mm from ISO_W", worst >= 1.0 - 1e-3, f"{worst:.3f} mm")

        print("2. area rule: the only gap is closed to GND")
        joins, _, vias = run(tmp, 'area', board(6, [(1, 3), (9, 29)], AREA), AREA_DRU)
        x0, y0, x1, y1 = AREA

        def within(x, y, r):
            return x0 - r <= x <= x1 + r and y0 - r <= y <= y1 + r
        inside = [s for s in joins for t in (0.0, 0.25, 0.5, 0.75, 1.0)
                  if within(s.start_x + t * (s.end_x - s.start_x),
                            s.start_y + t * (s.end_y - s.start_y), s.width / 2)]
        check("no GND track enters the area", not inside,
              f"{len(inside)} sample(s) inside")
        v_in = [v for v in vias if within(v.x, v.y, v.size / 2)]
        check("no GND via in the area", not v_in, f"{len(v_in)} via(s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
