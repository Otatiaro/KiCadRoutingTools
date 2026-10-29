#!/usr/bin/env python3
"""In pocket-rip mode the slot-0 guard keeps a first rip that enlarged the pocket.

The SE ladder's slot-0 guard (audit, structural) restores the first rip and
tries the re-analysis pick ALONE when that pick differs from it: a wrong
first pick would otherwise be part of every combination the ladder tries.
A pad walled in twice over -- an inner wall, and an outer one past it --
defeats it: ripping the inner wall enlarges the drained side's search, the
re-analysis rightly names the outer wall, and the guard restores the inner
one. The pocket shrinks back, and once it is too big for the pocket-rip
signature (one side at most a fifth of the other's iterations) the pooled
re-analysis hands the remaining rungs to the other side's flood.

A bystander leaves the pocket exactly as it was, so a first rip that grew
the drained side's search took down one of its walls: in pocket-rip mode
(KICAD_POCKET_RIP=1) the guard now keeps it and extends with the next wall.

Checks:
  1. A target pad inside two rings of rippable nets (B1 inside B2, no via
     room: a B.Cu pad under both), the source inside three cups of decoy
     nets that its forward search fills: with KICAD_POCKET_RIP=1 the ladder
     keeps B1, rips B2, rips no decoy, and the net connects; the board is
     DRC-clean and connected, and a second run writes the same copper.
  2. Without KICAD_POCKET_RIP the guard is unchanged: no "grew" line.

    python3 tests/test_slot0_guard_keeps_pocket_rip.py
"""
import json
import os
import re
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from run_utils import check as run_check, evidence, tool       # noqa: E402
from synth import RoutingBoard                                 # noqa: E402

failures = []
GREW = re.compile(r"Slot-0 guard: (\S+)'s rip grew the (\w+) pocket")


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def _fp(b, ref, pads, layer='F.Cu'):
    txt = []
    for i, (px, py, sx, sy) in enumerate(pads):
        txt.append('(pad "%d" smd rect (at %s %s) (size %s %s) (layers "%s" "%s") '
                   '(uuid "%s"))' % (i + 1, px, py, sx, sy, layer,
                                     layer.replace('Cu', 'Mask'), b._uid()))
    b.items.append(
        '  (footprint "synth:FP" (layer "%s") (uuid "%s") (at 0 0)\n'
        '    (property "Reference" "%s" (at 0 -2 0) (layer "F.SilkS") (uuid "%s")\n'
        '      (effects (font (size 0.5 0.5) (thickness 0.1))))\n    %s\n  )'
        % (layer, b._uid(), ref, b._uid(), '\n    '.join(txt)))


def _path(b, net, pts, layer='F.Cu'):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        b.seg(round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4), net,
              w=0.1, layer=layer)


def board():
    """N: S1 (6, 14) to T1 (25, 15). T1 inside ring B1 (2.4 mm), inside
    ring B2 (7 mm); U3, an unconnected B.Cu pad, under both rings. S1 inside
    three nested cups X, Y, Z on both layers, open away from T1."""
    b = RoutingBoard(w=40, h=30)
    b.pad('S1', 6.0, 14.0, 'N', sx=0.6, sy=0.6)
    b.pad('T1', 25.0, 15.0, 'N', sx=0.6, sy=0.6)
    _fp(b, 'U3', [(25.0, 15.0, 8.0, 8.0)], layer='B.Cu')
    for name, h in (('B1', 1.2), ('B2', 3.5)):
        x0, x1, y0, y1 = 25.0 - h, 25.0 + h, 15.0 - h, 15.0 + h
        b.pad(name + 'a', x0, y0, name, sx=0.5, sy=0.5)
        b.pad(name + 'b', x1, y1, name, sx=0.5, sy=0.5)
        _path(b, name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
    for k, name in enumerate('XYZ'):
        m = 1.25 + 1.0 * k
        xl, xc, yt, yb = 6.75 - m, 6.0 + m, 14.0 - m, 14.0 + m
        cup = [(xl, yt), (xc, yt), (xc, yb), (xl, yb)]
        b.pad(name + '1', xl, yt, name, sx=0.5, sy=0.5)
        b.pad(name + '2', xl, yb, name, sx=0.5, sy=0.5)
        _path(b, name, cup)
        _path(b, name, cup, layer='B.Cu')
        b.via(xc, yt, name, size=0.45, drill=0.3)
    return b.text()


def _project(path):
    proj = {'board': {'design_settings': {'rules': {'min_clearance': 0.09,
                                                    'min_track_width': 0.09},
                                          'rule_severities': {}}},
            'meta': {'filename': os.path.basename(path), 'version': 1},
            'net_settings': {'classes': [{'name': 'Default', 'clearance': 0.09,
                                          'track_width': 0.09,
                                          'via_diameter': 0.45,
                                          'via_drill': 0.3}],
                             'meta': {'version': 3}}}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(proj, fh)


def route(tmp, tag, pocket_rip):
    inp = os.path.join(tmp, f'{tag}_in.kicad_pcb')
    out = os.path.join(tmp, f'{tag}_out.kicad_pcb')
    js = os.path.join(tmp, f'{tag}.json')
    with open(inp, 'w', encoding='utf-8') as fh:
        fh.write(board())
    _project(os.path.splitext(inp)[0] + '.kicad_pro')
    evidence(inp, 'synthetic input board')
    env = dict(os.environ, KICAD_POCKET_RIP='1' if pocket_rip else '0')
    r = run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                   '--nets', 'N', '--rip-existing-nets', 'B1', 'B2', 'X', 'Y', 'Z',
                   '--layers', 'F.Cu', 'B.Cu', '--grid-step', '0.05',
                   '--escalation', 'off', '--json-out', js], accept=True, env=env)
    evidence(out, 'routed output')
    with open(js, encoding='utf-8') as fh:
        doc = json.load(fh)
    log = r.stdout + r.stderr
    ripped = set(re.findall(r"PRE-EXISTING rip: '([^']+)'", log))
    return out, doc, ripped, log


def _copper(path):
    from kicad_parser import parse_kicad_pcb
    pcb = parse_kicad_pcb(path)
    return (sorted((s.net_id, s.layer, round(s.width, 4), round(s.start_x, 4),
                    round(s.start_y, 4), round(s.end_x, 4), round(s.end_y, 4))
                   for s in pcb.segments),
            sorted((v.net_id, round(v.x, 4), round(v.y, 4)) for v in pcb.vias))


def graded(out):
    try:
        for t in ('check_drc.py', 'check_connected.py'):
            run_check([sys.executable, '-X', 'utf8', tool(t), out], accept=True)
    except AssertionError as exc:
        if 'BROKEN TEST' in str(exc):
            raise
        return False
    return True


def main():
    tmp = tempfile.mkdtemp(prefix='tslot0_')
    try:
        print("1. two nested walls, pocket-rip mode")
        out, doc, ripped, log = route(tmp, 'pocket', True)
        if 'Ripping up B1' not in log:
            print("  [BROKEN TEST] the ladder never ripped the inner wall first")
            failures.append('broken fixture')
            return 1
        grew = GREW.search(log)
        check("the guard keeps B1, whose rip grew the pocket",
              bool(grew) and grew.group(1) == 'B1', grew.group(0) if grew else '')
        check("N connects", 'N' in (doc.get('routed_single') or []),
              str(doc.get('failed_single')))
        check("both walls are ripped and no decoy",
              {'B1', 'B2'} <= ripped and not ripped & {'X', 'Y', 'Z'},
              str(sorted(ripped)))
        check("the routed board is DRC-clean and connected", graded(out))
        out2, _, ripped2, _ = route(tmp, 'pocket2', True)
        check("a second run rips the same nets and writes the same copper",
              ripped2 == ripped and _copper(out2) == _copper(out))
        print("2. without pocket-rip mode")
        _, _, _, log = route(tmp, 'off', False)
        check("the guard is unchanged (no 'grew' line)", not GREW.search(log))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
