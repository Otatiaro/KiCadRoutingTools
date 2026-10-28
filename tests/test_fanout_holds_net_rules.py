#!/usr/bin/env python3
"""The QFN and BGA fanouts hold the board's net-pair and pad hole-clearance
rules from its .kicad_dru (dru_copper_rules).

Both fanouts checked their copper against maps priced at the layer clearance
only, so an isolation barrier (`(constraint clearance (min 1mm))` over the
nets of one side against every other net) or a through-hole lead's
`hole_clearance` rule was invisible to them: an escape could ship 0.3 mm
from a net the rest of the router keeps 1 mm away.

Rows:
  A. dru_copper_rules.via_clears / drop_fanout_violations on a synthetic
     board: a via under 1 mm from an isolated net, or whose drill is under
     0.28 mm from a bound through-hole pad's copper, is refused; a net any of
     whose copper breaks a rule is dropped whole (from the returned lists and
     from pcb_data), a clean one is kept.
  B. qfn_fanout on kicad_files/tigard.kicad_pcb (U3, QFN-64): an ISO_X track
     laid 0.5 mm beside one escape. Control (no rule): the escape keeps
     running there. With the rule: no escape copper within 1 mm of ISO_X.
  C. bga_fanout on kicad_files/interf_u_unrouted_placed.kicad_pcb (U9): the
     same, beside one escape track.

B and C skip cleanly if their corpus board is absent.

    python3 tests/test_fanout_holds_net_rules.py
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

from synth import RoutingBoard                                  # noqa: E402

FAILS = []

ISO = ("(A.NetName == 'ISO_*' && !(B.NetName == 'ISO_*') && B.NetName != '' "
       "&& B.NetName != 'unconnected-*') || (B.NetName == 'ISO_*' && "
       "!(A.NetName == 'ISO_*') && A.NetName != '' && A.NetName != 'unconnected-*')")
TH = "A.Type == 'Pad' && A.Pad_Type == 'Through-hole' && A.Hole_Size_X >= 0.3mm"
DRU = (f'(version 1)\n(rule "isolation" (condition "{ISO}") (constraint clearance (min 1mm)))\n'
       f'(rule "leads" (condition "{TH}") (constraint hole_clearance (min 0.28mm)))\n')


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def pt_seg(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
    return math.hypot(px - x1 - t * dx, py - y1 - t * dy)


def seg_seg(a, b):
    """Centreline distance of two non-crossing segments ((x1, y1), (x2, y2))."""
    (a1, a2), (b1, b2) = a, b
    return min(pt_seg(*a1, *b1, *b2), pt_seg(*a2, *b1, *b2),
               pt_seg(*b1, *a1, *a2), pt_seg(*b2, *a1, *a2))


# ---------------------------------------------------------------- A. unit --
def t_unit():
    print('\n-- A. via_clears / drop_fanout_violations --')
    from kicad_parser import parse_kicad_pcb
    import dru_copper_rules as dcr
    tmp = tempfile.mkdtemp(prefix='fanrules_')
    try:
        b = RoutingBoard(w=40, h=30)
        b.pad('K1', 10, 10, 'ISO_A', sx=0.6, sy=0.6)
        b.pad('K2', 20, 10, 'ISO_A', sx=0.6, sy=0.6)
        b.seg(10, 10, 20, 10, 'ISO_A', w=0.2)
        b.pad('T1', 30, 20, 'OTHER', sx=1.1, sy=1.1, tht=True)
        b.pad('S1', 5, 25, 'SIG', sx=0.6, sy=0.6)
        b.pad('S2', 35, 25, 'SIG', sx=0.6, sy=0.6)
        b.pad('Q1', 5, 2, 'QUIET', sx=0.6, sy=0.6)
        pcb_file = os.path.join(tmp, 'in.kicad_pcb')
        with open(pcb_file, 'w', encoding='utf-8') as f:
            f.write(b.text())
        with open(os.path.join(tmp, 'in.kicad_dru'), 'w', encoding='utf-8') as f:
            f.write(DRU)
        pcb = parse_kicad_pcb(pcb_file)
        with contextlib.redirect_stdout(io.StringIO()):
            dcr.ensure_attached(pcb)
        check('the rules attach from the board\'s .kicad_dru', bool(dcr.entries_of(pcb)),
              f'{len(dcr.entries_of(pcb))} entr(y/ies)')
        nid = {n.name: i for i, n in pcb.nets.items()}
        sig = nid['SIG']
        # via copper 0.6 mm from ISO_A's track edge: under the 1 mm rule
        check('a via 0.6 mm off the isolated track is refused',
              not dcr.via_clears(pcb, sig, 15, 10 + 0.1 + 0.6 + 0.225, 0.45, 0.3))
        check('a via 1.2 mm off it is admitted',
              dcr.via_clears(pcb, sig, 15, 10 + 0.1 + 1.2 + 0.225, 0.45, 0.3))
        # drill edge 0.2 mm from T1's copper (ring edge at 0.55 + 0.2 + 0.15)
        check("a via whose drill is 0.2 mm from a through-hole lead's copper is refused",
              not dcr.via_clears(pcb, sig, 30 + 0.55 + 0.2 + 0.15, 20, 0.45, 0.3))
        check('the lead\'s own net is not held off it',
              dcr.via_clears(pcb, nid['OTHER'], 30 + 0.55 + 0.2 + 0.15, 20, 0.45, 0.3))
        tracks = [{'start': (5, 25), 'end': (15, 10.6), 'width': 0.15, 'layer': 'F.Cu',
                   'net_id': sig},
                  {'start': (5, 2), 'end': (8, 2), 'width': 0.15, 'layer': 'F.Cu',
                   'net_id': nid['QUIET']}]
        vias = [{'x': 15, 'y': 10.6, 'size': 0.45, 'drill': 0.3, 'net_id': sig,
                 'layers': ['F.Cu', 'B.Cu']}]
        from kicad_parser import Segment, Via
        for t in tracks:
            pcb.segments.append(Segment(t['start'][0], t['start'][1], t['end'][0],
                                        t['end'][1], t['width'], t['layer'], t['net_id']))
        pcb.vias.append(Via(15, 10.6, 0.45, 0.3, ['F.Cu', 'B.Cu'], sig))
        kt, kv, dropped = dcr.drop_fanout_violations(pcb, tracks, vias)
        check('the breaking net is dropped, the clean one kept',
              dropped == ['SIG'] and [t['net_id'] for t in kt] == [nid['QUIET']] and not kv,
              f'dropped {dropped}, kept {len(kt)} track(s) {len(kv)} via(s)')
        check('and its copper leaves pcb_data',
              not [s for s in pcb.segments if s.net_id == sig]
              and not [v for v in pcb.vias if v.net_id == sig])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------- B / C. engines --
def stage(board_file, tmp, with_rule):
    """A copy of the corpus board (its .kicad_pro too) in `tmp`, with the
    rule file beside it when `with_rule`."""
    dst = os.path.join(tmp, os.path.basename(board_file))
    shutil.copy(board_file, dst)
    pro = os.path.splitext(board_file)[0] + '.kicad_pro'
    if os.path.exists(pro):
        shutil.copy(pro, os.path.splitext(dst)[0] + '.kicad_pro')
    if with_rule:
        with open(os.path.splitext(dst)[0] + '.kicad_dru', 'w', encoding='utf-8') as f:
            f.write(DRU)
    return dst


def inject_iso(pcb, beside, gap=0.5):
    """An ISO_X track parallel to `beside` (a track dict), `gap` mm edge to
    edge, on its layer. Returns its end points."""
    from kicad_parser import Net, Segment
    nid = max(pcb.nets) + 1
    pcb.nets[nid] = Net(nid, 'ISO_X')
    (x1, y1), (x2, y2) = beside['start'], beside['end']
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    nx, ny = -(y2 - y1) / L, (x2 - x1) / L
    off = beside['width'] / 2 + gap + 0.1
    a, b = (x1 + nx * off, y1 + ny * off), (x2 + nx * off, y2 + ny * off)
    pcb.segments.append(Segment(a[0], a[1], b[0], b[1], 0.2, beside['layer'], nid))
    return a, b, beside['layer']


def near_iso(tracks, vias, iso):
    """Escape copper (any net but ISO_X) within 1 mm, edge to edge, of the
    injected track."""
    a, b, layer = iso
    out = []
    for t in tracks:
        if t['layer'] == layer and seg_seg((t['start'], t['end']), (a, b)) \
                - t['width'] / 2 - 0.1 < 1.0 - 1e-3:
            out.append(t)
    for v in vias:
        if pt_seg(v['x'], v['y'], *a, *b) - v['size'] / 2 - 0.1 < 1.0 - 1e-3:
            out.append(v)
    return out


def t_engine(label, board_file, ref, run, longest_on):
    print(f'\n-- {label} --')
    if not os.path.exists(board_file):
        print(f'  SKIP: corpus board not found ({board_file})')
        return
    from kicad_parser import parse_kicad_pcb
    tmp = tempfile.mkdtemp(prefix='fanrules_')
    try:
        # where to put the isolated track: beside the longest escape segment
        # of a plain run
        plain = os.path.join(tmp, 'plain')
        os.makedirs(plain)
        pcb = parse_kicad_pcb(stage(board_file, plain, False))
        with contextlib.redirect_stdout(io.StringIO()):
            tracks, _v = run(pcb, ref)[:2]
        cand = [t for t in tracks if t['layer'] == longest_on]
        if not cand:
            check(f'{label}: the plain run escapes on {longest_on}', False)
            return
        beside = max(cand, key=lambda t: math.hypot(t['end'][0] - t['start'][0],
                                                    t['end'][1] - t['start'][1]))
        for tag, rule in (('control', False), ('rule', True)):
            d = os.path.join(tmp, tag)
            os.makedirs(d)
            pcb = parse_kicad_pcb(stage(board_file, d, rule))
            iso = inject_iso(pcb, beside)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tracks, vias, failed = run(pcb, ref)
            hits = near_iso(tracks, vias, iso)
            if tag == 'control':
                check(f'{label} control: without the rule the escape still runs '
                      f'within 1 mm of ISO_X', bool(hits), f'{len(hits)} item(s)')
            else:
                check(f'{label}: with the rule no escape copper within 1 mm of ISO_X',
                      not hits, f'{len(hits)} item(s); {len(tracks)} track(s), '
                      f'{len(failed)} failed')
                check(f'{label}: the rest of the part still escapes', len(tracks) > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def qfn_run(pcb, ref):
    from qfn_fanout import generate_qfn_fanout
    fp = pcb.footprints[ref]
    nets = sorted({p.net_name for p in fp.pads if p.net_name and 'GND' not in p.net_name})
    return generate_qfn_fanout(fp, pcb, net_filter=nets, layer='F.Cu', track_width=0.1,
                               clearance=0.1, grid_step=0.05, escape_method='stub')


def bga_run(pcb, ref):
    from bga_fanout import generate_bga_fanout
    fp = pcb.footprints[ref]
    nets = sorted({p.net_name for p in fp.pads
                   if p.net_name and p.net_name not in ('GND', 'VCC')})
    t, v, _rm, failed = generate_bga_fanout(
        fp, pcb, net_filter=nets, layers=['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'],
        track_width=0.2, clearance=0.2, via_size=0.5, via_drill=0.3)
    return t, v, failed


def main():
    t_unit()
    t_engine('B. qfn_fanout', os.path.join(ROOT, 'kicad_files', 'tigard.kicad_pcb'), 'U3',
             qfn_run, 'F.Cu')
    t_engine('C. bga_fanout',
             os.path.join(ROOT, 'kicad_files', 'interf_u_unrouted_placed.kicad_pcb'), 'U9',
             bga_run, 'F.Cu')
    print('\nFAILED: ' + ', '.join(FAILS) if FAILS else '\nALL PASS')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
