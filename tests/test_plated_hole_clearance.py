#!/usr/bin/env python3
"""Copper of another net is held off PLATED holes -- via drills and
through-hole pad barrels -- at the board's min_hole_clearance, as KiCad does.

KiCad 10's hole_clearance rule grades every hole, plated or not, from its wall.
A 0.45/0.3 via has a 0.075 mm ring, so on a board declaring clearance 0.09 and
min_hole_clearance 0.2 its copper must keep 0.125 mm from foreign copper, not
0.09. Measured with kicad-cli 10 on the probe board built below (copper gaps
0.11 mm): it reports a track on F.Cu and on In1.Cu beside a via, two vias, an
SMD pad beside a via, and a track beside a thin-ring PTH pad, all at 0.185 /
0.16 mm. It reports nothing for the same pairs on one net, for a track whose
hole gap is 0.205 mm, nor for a track beside a PTH pad whose ring covers the
floor. With the copper gap at 0.05 mm (under clearance) it reports the track
and via pairs as clearance only, and the pad pairs for their hole as well.

Three parts:
  1. check_drc grades the probe board exactly that way.
  2. The obstacle map holds a track, a new via and a via near a pad far
     enough off that floor, once the board declares it, and is unchanged on a
     board that declares none.
  3. route.py, asked to route through a gap between two vias that is wide
     enough for the copper clearance but not for the hole floor, delivers a
     board check_drc grades clean -- and the output project keeps the
     declared floor.

Standalone:  python3 tests/test_plated_hole_clearance.py
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'py_router'))
sys.path.insert(0, os.path.join(REPO, 'tests'))

from run_utils import tool, tool_env, evidence                   # noqa: E402
from kicad_parser import parse_kicad_pcb                         # noqa: E402
from check_drc import run_drc                                    # noqa: E402
from routing_config import GridRouteConfig, GridCoord            # noqa: E402
from obstacle_map import build_base_obstacle_map                 # noqa: E402
from kicad_dru import install_layer_clearances                   # noqa: E402
import obstacle_map                                              # noqa: E402

fails = []
HOLE_TYPES = ('track-hole', 'via-hole', 'pad-hole')
VS, VD, TW = 0.45, 0.3, 0.1


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name
          + (f" -- {detail}" if detail and not cond else ''))
    if not cond:
        fails.append(name)


class Board:
    """A small 4-layer board written as text, with its project."""

    def __init__(self):
        self.nets = ['']
        self.items = []
        self.width = 10.0

    def net(self, name):
        self.nets.append(name)
        return len(self.nets) - 1

    def via(self, x, y, n):
        self.items.append(f' (via (at {x} {y}) (size {VS}) (drill {VD}) '
                          f'(layers "F.Cu" "B.Cu") (net {n}) (uuid "v{len(self.items)}"))')

    def seg(self, x1, y1, x2, y2, n, layer='F.Cu', w=TW):
        self.items.append(f' (segment (start {x1} {y1}) (end {x2} {y2}) (width {w}) '
                          f'(layer "{layer}") (net {n}) (uuid "s{len(self.items)}"))')

    def fp(self, ref, x, y, pads):
        body = '\n'.join(pads)
        self.items.append(f' (footprint "t:{ref}" (layer "F.Cu") (at {x} {y})\n'
                          f'  (property "Reference" "{ref}" (at 0 -2) (layer "F.SilkS"))\n'
                          f'{body}\n )')

    def write(self, path, hole_clearance=0.2):
        w, h = self.width, 10
        edge = '\n'.join(
            f' (gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (layer "Edge.Cuts") (width 0.1))'
            for a, b in [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))])
        nets = '\n'.join(f' (net {i} "{n}")' for i, n in enumerate(self.nets))
        layers = ('(layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) '
                  '(2 "B.Cu" signal) (25 "Edge.Cuts" user) (5 "F.SilkS" user) '
                  '(1 "F.Mask" user) (3 "B.Mask" user))')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(f'(kicad_pcb (version 20240108) (generator "test")\n {layers}\n'
                     f'{nets}\n{edge}\n' + '\n'.join(self.items) + '\n)\n')
        rules = {"min_clearance": 0.09, "min_hole_to_hole": 0.2,
                 "min_track_width": 0.09, "min_via_annular_width": 0.05,
                 "min_via_diameter": 0.25, "min_through_hole_diameter": 0.15,
                 "min_copper_edge_clearance": 0.2}
        if hole_clearance is not None:
            rules["min_hole_clearance"] = hole_clearance
        proj = {"board": {"design_settings": {"rules": rules}},
                "net_settings": {"classes": [{
                    "name": "Default", "clearance": 0.09, "track_width": TW,
                    "via_diameter": VS, "via_drill": VD}]}}
        with open(os.path.splitext(path)[0] + '.kicad_pro', 'w') as fh:
            json.dump(proj, fh)
        return path


def probe_board(gap):
    """The kicad-cli probe: one case per 4 mm column, copper gap `gap`."""
    b = Board()
    cases = {}
    x = [1.0]

    def col():
        x[0] += 4
        return x[0]

    yc = 5 + VS / 2 + gap + TW / 2
    c = col(); a, v = b.net('T1'), b.net('V1'); b.via(c, 5, v); b.seg(c - 1.5, yc, c + 1.5, yc, a)
    cases['track F.Cu / via'] = ('T1', 'V1')
    c = col(); a, v = b.net('T2'), b.net('V2'); b.via(c, 5, v); b.seg(c - 1.5, yc, c + 1.5, yc, a, 'In1.Cu')
    cases['track In1.Cu / via'] = ('T2', 'V2')
    c = col(); a, v = b.net('VA'), b.net('VB'); b.via(c, 5, a); b.via(c + VS + gap, 5, v)
    cases['via / via'] = ('VA', 'VB')
    c = col(); a = b.net('S4'); b.via(c, 5, a); b.seg(c - 1.5, yc, c + 1.5, yc, a)
    c = col(); a = b.net('S5'); b.via(c, 5, a); b.via(c + VS + gap, 5, a)
    c = col(); p, v = b.net('P6'), b.net('V6'); b.via(c, 5, v)
    b.fp('U6', c, 5 + VS / 2 + gap + 0.25,
         [f'  (pad "1" smd rect (at 0 0) (size 0.5 0.5) (layers "F.Cu" "F.Mask") (net {p} "P6"))'])
    cases['smd pad / via'] = ('P6', 'V6')
    c = col(); t, p = b.net('T7'), b.net('P7')
    b.fp('J7', c, 5, [f'  (pad "1" thru_hole circle (at 0 0) (size 1.0 1.0) (drill 0.6) '
                      f'(layers "*.Cu" "*.Mask") (net {p} "P7"))'])
    b.seg(c - 1.5, 5 + 0.5 + gap + TW / 2, c + 1.5, 5 + 0.5 + gap + TW / 2, t)
    c = col(); t, p = b.net('T8'), b.net('P8')
    b.fp('J8', c, 5, [f'  (pad "1" thru_hole circle (at 0 0) (size 0.7 0.7) (drill 0.6) '
                      f'(layers "*.Cu" "*.Mask") (net {p} "P8"))'])
    b.seg(c - 1.5, 5 + 0.35 + gap + TW / 2, c + 1.5, 5 + 0.35 + gap + TW / 2, t)
    cases['track / thin-ring PTH'] = ('T8', 'P8')
    c = col(); v, p = b.net('V9'), b.net('P9')
    b.fp('J9', c, 5, [f'  (pad "1" thru_hole circle (at 0 0) (size 1.0 1.0) (drill 0.6) '
                      f'(layers "*.Cu" "*.Mask") (net {p} "P9"))'])
    b.via(c + 0.5 + gap + VS / 2, 5, v)
    cases['PTH pad / via'] = ('P9', 'V9')
    c = col(); t, v = b.net('T10'), b.net('V10'); b.via(c, 5, v)
    y10 = 5 + VS / 2 + 0.13 + TW / 2
    b.seg(c - 1.5, y10, c + 1.5, y10, t)
    b.width = x[0] + 4
    return b, cases


def hole_pairs(viols):
    return {frozenset((v['net1'], v['net2'])) for v in viols if v['type'] in HOLE_TYPES}


def part1_check_drc(tmp):
    print("1. check_drc grades plated holes as kicad-cli 10 does")
    b, cases = probe_board(0.11)
    p = b.write(os.path.join(tmp, 'probe.kicad_pcb'))
    viols = run_drc(p, clearance=0.09, quiet=True, print_summary=False,
                    hole_to_hole_clearance=0.2, hole_clearance=0.2)
    got = hole_pairs(viols)
    want = {frozenset(v) for v in cases.values()}
    check("exactly the six pairs kicad-cli reports (not the same-net pairs, "
          "not the 0.205 mm control, not the PTH whose ring covers the floor)",
          got == want, f"got {sorted(map(sorted, got))}")
    for v in viols:
        if v['type'] in HOLE_TYPES:
            check(f"{v['type']} {v['net1']}/{v['net2']} required 0.2 mm",
                  abs(v.get('required_mm', 0) - 0.2) < 1e-9, str(v))
    viols0 = run_drc(p, clearance=0.09, quiet=True, print_summary=False,
                     hole_to_hole_clearance=0.2, hole_clearance=0.0)
    check("a board declaring no hole clearance grades no plated-hole item",
          not hole_pairs(viols0))

    b, _ = probe_board(0.05)
    p = b.write(os.path.join(tmp, 'probe05.kicad_pcb'))
    viols = run_drc(p, clearance=0.09, quiet=True, print_summary=False,
                    hole_to_hole_clearance=0.2, hole_clearance=0.2)
    check("copper under clearance: only the PAD pairs are graded for their "
          "hole (kicad-cli: the track and via pairs are clearance only)",
          hole_pairs(viols) == {frozenset(('P6', 'V6')), frozenset(('P9', 'V9'))},
          f"got {sorted(map(sorted, hole_pairs(viols)))}")


def _map_for(path, routed_net):
    pcb = parse_kicad_pcb(path)
    cfg = GridRouteConfig(layers=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"], grid_step=0.05,
                          clearance=0.09, track_width=TW, via_size=VS, via_drill=VD,
                          hole_to_hole_clearance=0.2)
    install_layer_clearances(cfg, None, path, pcb)
    nid = next(i for i, n in pcb.nets.items() if n.name == routed_net)
    return cfg, build_base_obstacle_map(pcb, cfg, [nid])


def part2_obstacle_map(tmp):
    print("2. the obstacle map holds copper off foreign via drills")
    # A foreign via at (5, 5), a foreign track along y = 2 and a foreign SMD pad
    # at (5, 8); the routed net SIG owns nothing near them.
    for declared in (0.25, None):
        obstacle_map._HOLE_CLR_CACHE.clear()
        b = Board()
        sig = b.net('SIG')
        v, t, p = b.net('V'), b.net('T'), b.net('P')
        b.via(5, 5, v)
        b.seg(2, 2, 8, 2, t)
        b.fp('U1', 5, 8, [f'  (pad "1" smd rect (at 0 0) (size 0.5 0.5) '
                          f'(layers "F.Cu" "F.Mask") (net {p} "P"))'])
        b.fp('U2', 1, 9, [f'  (pad "1" smd rect (at 0 0) (size 0.5 0.5) '
                          f'(layers "F.Cu" "F.Mask") (net {sig} "SIG"))'])
        name = 'map_hc.kicad_pcb' if declared else 'map_none.kicad_pcb'
        path = b.write(os.path.join(tmp, name), hole_clearance=declared)
        cfg, obs = _map_for(path, 'SIG')
        coord = GridCoord(cfg.grid_step)
        tag = f"declared {declared}" if declared else "declared nothing"

        def tb(x, y):
            gx, gy = coord.to_grid(x, y)
            return obs.is_blocked(gx, gy, 0)

        def vb(x, y):
            gx, gy = coord.to_grid(x, y)
            return obs.is_via_blocked(gx, gy)

        if declared:
            check(f"{tag}: installed as config.plated_hole_clearance",
                  abs(getattr(cfg, 'plated_hole_clearance', 0.0) - declared) < 1e-9,
                  f"{getattr(cfg, 'plated_hole_clearance', None)}")
        # Track centre 0.45 from the via centre: copper gap 0.175 clears 0.09
        # (keep-out 0.365 mm, 0.4125 on the grid), hole gap 0.25 sits on the
        # 0.25 floor's keep-out (0.45 mm, 0.4625 on the grid).
        check(f"{tag}: a track 0.45 mm from a foreign via centre is "
              + ("held off" if declared else "allowed"),
              tb(5.45, 5) == bool(declared))
        check(f"{tag}: a track 0.55 mm from it is allowed", not tb(5.55, 5))
        # A new via 0.60 from the foreign via: copper 0.15 clears 0.09, each
        # hole sits 0.225 from the other's copper (needs 0.625).
        check(f"{tag}: a new via 0.60 mm from a foreign via is "
              + ("held off" if declared else "allowed"),
              vb(5.60, 5) == bool(declared))
        check(f"{tag}: a new via 0.70 mm from it is allowed", not vb(5.70, 5))
        # A new via 0.40 from a foreign track centre (width 0.1): copper 0.125,
        # hole 0.20 (needs 0.45).
        check(f"{tag}: a new via 0.40 mm from a foreign track is "
              + ("held off" if declared else "allowed"),
              vb(5, 2.40) == bool(declared))
        # A new via whose centre is 0.36 from a foreign pad's edge: copper
        # 0.135, hole 0.21 (needs 0.40).
        check(f"{tag}: a new via 0.36 mm from a foreign pad edge is "
              + ("held off" if declared else "allowed"),
              vb(5, 8 - 0.25 - 0.36) == bool(declared))
        check(f"{tag}: a new via 0.45 mm from it is allowed",
              not vb(5, 8 - 0.25 - 0.45))


def part3_route(tmp):
    print("3. route.py routes past a via gap the hole floor closes")
    # Two foreign vias 0.75 mm apart across the direct path: a 0.1 track fits
    # between them at copper clearance 0.09 (needs 0.73), not at the hole
    # floor 0.2 (needs 0.80). The board is wide enough to go around.
    b = Board()
    sig, a, c = b.net('SIG'), b.net('A'), b.net('B')
    b.width = 12
    for ref, x in (('U1', 2), ('U2', 10)):
        b.fp(ref, x, 5, [f'  (pad "1" smd rect (at 0 0) (size 0.6 0.6) '
                         f'(layers "F.Cu" "F.Mask") (net {sig} "SIG"))'])
    b.via(6, 5 - 0.375, a)
    b.via(6, 5 + 0.375, c)
    for y in (2.0, 2.8, 7.2, 8.0):
        b.via(6, y, a)
    src = b.write(os.path.join(tmp, 'gap.kicad_pcb'))
    out = os.path.join(tmp, 'gap_routed.kicad_pcb')
    r = subprocess.run([sys.executable, '-X', 'utf8', tool('route.py'), src, out,
                        '--nets', 'SIG', '--layers', 'F.Cu', '--clearance', '0.09',
                        '--track-width', str(TW), '--via-size', str(VS),
                        '--via-drill', str(VD), '--grid-step', '0.05'],
                       capture_output=True, text=True, env=tool_env(), timeout=600,
                       encoding='utf-8', errors='replace')
    check("route.py ran", r.returncode == 0 and os.path.isfile(out),
          f"rc={r.returncode} {r.stdout[-800:]} {r.stderr[-800:]}")
    if not os.path.isfile(out):
        return
    evidence(out, 'routed board')
    pcb = parse_kicad_pcb(out)
    sig_id = next(i for i, n in pcb.nets.items() if n.name == 'SIG')
    check("SIG is routed", any(s.net_id == sig_id for s in pcb.segments))
    # Measured here, independently of the grader: SIG's copper against every
    # foreign via's hole wall.
    from geometry_utils import point_to_segment_distance
    gaps = [point_to_segment_distance(v.x, v.y, s.start_x, s.start_y, s.end_x, s.end_y)
            - s.width / 2 - v.drill / 2
            for s in pcb.segments if s.net_id == sig_id
            for v in pcb.vias if v.net_id != sig_id]
    check("SIG's copper sits at least 0.2 mm from every foreign via hole",
          gaps and min(gaps) >= 0.2 - 0.01, f"closest {min(gaps):.4f} mm" if gaps else '')
    viols = run_drc(out, clearance=0.09, quiet=True, print_summary=False,
                    hole_to_hole_clearance=0.2, hole_clearance=0.2)
    check("the route clears every foreign via drill by 0.2 mm",
          not [v for v in viols if v['type'] in HOLE_TYPES],
          str([(v['type'], v['net1'], v['net2'], round(v['overlap_mm'], 4))
               for v in viols if v['type'] in HOLE_TYPES]))
    with open(os.path.splitext(out)[0] + '.kicad_pro') as fh:
        rules = json.load(fh)['board']['design_settings']['rules']
    check("the output project still declares min_hole_clearance 0.2",
          abs(rules.get('min_hole_clearance', 0) - 0.2) < 1e-9,
          f"{rules.get('min_hole_clearance')}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        part1_check_drc(tmp)
        part2_obstacle_map(tmp)
        part3_route(tmp)
    print("-" * 60)
    if fails:
        print(f"{len(fails)} FAILED: " + "; ".join(fails))
        return 1
    print("ALL PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
