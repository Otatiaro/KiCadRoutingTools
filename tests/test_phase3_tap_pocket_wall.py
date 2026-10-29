#!/usr/bin/env python3
"""The phase-3 tap cascade rips the wall that holds a walled-in pad.

A tap edge to a row pin fails: the pin's own neighbours' escapes wall it in,
and the backward search, started at the pin, drains in a small pocket while
the forward search, started at the net's tree, runs on. Two things kept the
cascade from ripping that wall:

  - it attributed the blockers from whichever direction reported FEWER cells,
    not from the one that drained, so a forward flood smaller than the pocket
    named its bystanders instead of the wall;
  - from N=2 on it re-analyzed both directions pooled, and ripped the
    first-attempt list's next slot even when that re-analysis named another
    net: rung N ripped a stale pick.

The cascade now attributes every rung, first and later, to the drained side
(the iteration counts say which), puts the re-analysis pick in slot N-1 as
the SE, diff-pair and reroute ladders do (blocking_analysis.
place_blocker_at_rung), and among the pocket's walls prefers the ones the
flood also meets: they separate the pocket from the rest of the net.

Checks:
  1. _tap_attribution_entries picks the drained side even when it has more
     cells, falls back to the smaller side without the signature, and returns
     the flood only with it.
  2. place_blocker_at_rung moves a listed pick into slot N-1, appends a new
     one there, and keeps a diff pair as one entry; nets_meeting_cells reads
     the cache and records nothing.
  3. A row pin walled in by one net, its tree's forward search held by three
     nested cups of decoy nets: the cascade rips the wall alone, the pin
     connects, the board is DRC-clean and connected, and a second run writes
     the same copper.
  4. A pin walled at both ends by two nets of which only one holds it in (dead
     ground behind the other): the one the flood also meets is ripped, the
     other never.

    python3 tests/test_phase3_tap_pocket_wall.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from run_utils import check as run_check, evidence, tool       # noqa: E402
from synth import RoutingBoard                                 # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def attribution_sides():
    print("1. the drained side is the one attributed")
    from phase3_routing import _tap_attribution_entries
    pocket = [(i, 0, 0) for i in range(30)]
    flood = [(i, 9, 0) for i in range(10)]
    extra = [(99, 99, 0)]
    drained = {('a', 'b'): (pocket + flood, (1.0, 2.0),
                            {'fwd': flood, 'bwd': pocket, 'extra': extra,
                             'iters': (5000, 420)})}
    [(cells, tgt, fl)] = _tap_attribution_entries(drained, 5000)
    check("the drained backward side, though it has more cells",
          cells == pocket + extra and tgt == (1.0, 2.0), f"{len(cells)} cells")
    check("the other side comes back as the flood", fl == flood)
    no_sig = {('a', 'b'): (pocket + flood, (1.0, 2.0),
                           {'fwd': flood, 'bwd': pocket, 'extra': extra,
                            'iters': (5000, 5000)})}
    [(cells, _, fl)] = _tap_attribution_entries(no_sig, 5000)
    check("without the signature, the side with fewer cells and no flood",
          cells == flood + extra and fl is None)
    old = {('a', 'b'): (pocket + flood, (1.0, 2.0),
                        {'fwd': flood, 'bwd': pocket, 'extra': extra})}
    [(cells, _, fl)] = _tap_attribution_entries(old, 5000)
    check("an entry without counts keeps the fewer-cells side",
          cells == flood + extra and fl is None)


def rung_slot():
    print("2. the re-analysis pick is the one rung N rips")
    from blocking_analysis import (place_blocker_at_rung, nets_meeting_cells,
                                   _pack_cells)
    import numpy as np

    def canon(nid, pairs):
        return pairs[nid] if nid in pairs else nid

    b = {n: SimpleNamespace(net_id=n) for n in (1, 2, 3, 4, 5, 6)}
    lst, seen = [b[1], b[2], b[3]], {1, 2, 3}
    place_blocker_at_rung(lst, seen, b[3], 2, {}, canon)
    check("a pick already listed moves into slot N-1",
          [x.net_id for x in lst] == [1, 3, 2], str([x.net_id for x in lst]))
    place_blocker_at_rung(lst, seen, b[4], 3, {}, canon)
    check("a new pick is appended into slot N-1",
          [x.net_id for x in lst] == [1, 3, 4, 2] and 4 in seen,
          str([x.net_id for x in lst]))
    lst, seen = [b[1], b[5]], {1, 5}
    place_blocker_at_rung(lst, seen, b[6], 2, {5: 5, 6: 5}, canon)
    check("a diff pair's other half is the same entry",
          [x.net_id for x in lst] == [1, 5], str([x.net_id for x in lst]))

    cells = [(10, 10, 0), (11, 10, 0)]
    keys = frozenset(_pack_cells(np.array([(11, 10)]), np.array([0])).tolist())
    cache = {(7, 0.0): (keys, frozenset()), (8, 0.0): (frozenset(), frozenset())}
    before = dict(cache)
    check("nets_meeting_cells names the net whose cells meet the frontier",
          nets_meeting_cells([7, 8, 9], cells, cache) == {7})
    check("and leaves the cache as it found it", cache == before)


def _fp(b, ref, pads, layer='F.Cu'):
    """A footprint of several pads (RoutingBoard.pad makes single-pad ones)."""
    txt = []
    for i, (px, py, sx, sy, net) in enumerate(pads):
        n = '(net %d "%s") ' % (b.net(net), net) if net else ''
        txt.append('(pad "%d" smd rect (at %s %s) (size %s %s) (layers "%s" "%s") '
                   '%s(uuid "%s"))' % (i + 1, round(px, 4), round(py, 4), sx, sy,
                                       layer, layer.replace('Cu', 'Mask'), n,
                                       b._uid()))
    b.items.append(
        '  (footprint "synth:FP" (layer "%s") (uuid "%s") (at 0 0)\n'
        '    (property "Reference" "%s" (at 0 -2 0) (layer "F.SilkS") (uuid "%s")\n'
        '      (effects (font (size 0.5 0.5) (thickness 0.1))))\n    %s\n  )'
        % (layer, b._uid(), ref, b._uid(), '\n    '.join(txt)))


def _path(b, net, pts, layer='F.Cu'):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        b.seg(round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4), net,
              w=0.1, layer=layer)


def _row(b, nets, half=1.25):
    """U1: five pins 0.5 mm apart at x = 20, long axis along x."""
    _fp(b, 'U1', [(20, 13.0 + 0.5 * i, 2 * half, 0.3, n)
                  for i, n in enumerate(nets)])
    return 20 - half, 20 + half


def pocket_board():
    """M's tap to U1.3 (the short MST edge; A1-B1 is the main one). W1 leaves
    U1.2 outward, runs past the outer ends of pins 3-5 and under the row: the
    one wall. The inner side is a box of unconnected pads (U2 on F.Cu, U3
    under it on B.Cu, so no via fits). Around A1, three nested cups of decoy
    nets on both layers, open away from U1: the tap's forward search fills
    them before it reaches U1, and the route leaves through their open side."""
    b = RoutingBoard(w=40, h=30)
    xo, xi = _row(b, ['', 'W1', 'M', '', ''])
    xr = xi + 4.0
    _fp(b, 'U2', [((xi + xr) / 2, 12.3, xr - xi + 2.0, 0.6, ''),
                  (xr + 0.5, 13.6, 1.0, 3.2, '')])
    _fp(b, 'U3', [((xi + xr) / 2 + 0.2, 13.9, xr - xi + 0.4, 3.2, '')],
        layer='B.Cu')
    b.pad('A1', 6.0, 14.0, 'M', sx=0.6, sy=0.6)
    b.pad('B1', 6.0, 28.5, 'M', sx=0.6, sy=0.6)
    b.pad('W1P', xr + 3.0, 27.0, 'W1', sx=0.6, sy=0.6)
    _path(b, 'W1', [(xo, 13.5), (xo - 0.35, 13.5), (xo - 0.35, 15.35),
                    (xr + 3.0, 15.35), (xr + 3.0, 27.0)])
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


def tie_board():
    """U1.3 walled at its outer end by W1 (U1.2's escape, running on to
    W1P) and at its inner end by W2 (joining U1.4 to U1.1 along the inner
    ends), with unconnected pads boxing in the inner side: ripping W2 leaves
    dead ground, ripping W1 frees the pin. Both walls meet the pocket over
    the same cells; only W1 meets the flood. D, on B.Cu under the pin, is
    the via-in-pad decline's pick."""
    b = RoutingBoard(w=40, h=30)
    xo, xi = _row(b, ['W2', 'W1', 'M', 'W2', ''], half=0.75)
    _fp(b, 'U2', [(21.7, 14.0, 1.0, 3.6, ''), (20.4, 12.3, 2.2, 0.6, ''),
                  (20.4, 15.7, 2.2, 0.6, '')])
    b.pad('A1', 14.0, 14.0, 'M', sx=0.6, sy=0.6)
    b.pad('B1', 14.0, 27.0, 'M', sx=0.6, sy=0.6)
    b.pad('W1P', 19.05, 27.0, 'W1', sx=0.6, sy=0.6)
    _path(b, 'W1', [(xo, 13.5), (19.05, 13.5), (19.05, 27.0)])
    _path(b, 'W2', [(xi, 14.5), (20.95, 14.5), (20.95, 13.0), (xi, 13.0)])
    b.pad('D1', 17.0, 2.0, 'D', sx=0.6, sy=0.6)
    b.pad('D2', 23.0, 2.0, 'D', sx=0.6, sy=0.6)
    _path(b, 'D', [(17.0, 2.0), (17.0, 14.0), (23.0, 14.0), (23.0, 2.0)],
          layer='B.Cu')
    b.via(17.0, 2.0, 'D', size=0.45, drill=0.3)
    b.via(23.0, 2.0, 'D', size=0.45, drill=0.3)
    return b.text()


def _project(path):
    """The 0.09 mm floor the boards are drawn to, for the route and grading."""
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


def route(text, tmp, tag, rippable, extra=()):
    inp = os.path.join(tmp, f'{tag}_in.kicad_pcb')
    out = os.path.join(tmp, f'{tag}_out.kicad_pcb')
    js = os.path.join(tmp, f'{tag}.json')
    with open(inp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    _project(os.path.splitext(inp)[0] + '.kicad_pro')
    evidence(inp, 'synthetic input board')
    r = run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                   '--nets', 'M', '--rip-existing-nets', *rippable,
                   '--layers', 'F.Cu', 'B.Cu', '--grid-step', '0.05',
                   '--same-net-pad-clearance', '0.09', '--escalation', 'off',
                   '--json-out', js, *extra], accept=True)
    evidence(out, 'routed output')
    evidence(js, '--json-out')
    with open(js, encoding='utf-8') as fh:
        doc = json.load(fh)
    log = r.stdout + r.stderr
    ripped = set(re.findall(r"PRE-EXISTING rip: '([^']+)'", log))
    return out, doc, ripped, log


def _copper(path):
    """Tracks and vias as values (outputs carry fresh uuids per run)."""
    from kicad_parser import parse_kicad_pcb
    pcb = parse_kicad_pcb(path)
    segs = sorted((s.net_id, s.layer, round(s.width, 4), round(s.start_x, 4),
                   round(s.start_y, 4), round(s.end_x, 4), round(s.end_y, 4))
                  for s in pcb.segments)
    vias = sorted((v.net_id, round(v.size, 4), round(v.x, 4), round(v.y, 4))
                  for v in pcb.vias)
    return segs, vias


def graded(out):
    """check_drc.py and check_connected.py both exit 0 on `out`."""
    try:
        for t in ('check_drc.py', 'check_connected.py'):
            run_check([sys.executable, '-X', 'utf8', tool(t), out], accept=True)
    except AssertionError as exc:
        if 'BROKEN TEST' in str(exc):
            raise
        return False
    return True


def whole_boards():
    print("3. a pin walled in by one net, its tree held by decoy cups")
    tmp = tempfile.mkdtemp(prefix='tp3pw_')
    try:
        out, doc, ripped, log = route(pocket_board(), tmp, 'pocket',
                                      ['W1', 'X', 'Y', 'Z'])
        if 'M: Attempting rip-up retry' not in log and 'to retry tap route' not in log:
            print("  [BROKEN TEST] the tap never reached the phase-3 cascade")
            failures.append('broken fixture')
            return
        check("U1.3 connects", not doc.get('failed_multipoint')
              and 'M' in (doc.get('routed_single') or []),
              str(doc.get('failed_multipoint')))
        check("the wall is ripped and no decoy", ripped == {'W1'},
              str(sorted(ripped)))
        check("the routed board is DRC-clean and connected", graded(out))
        out2, _, ripped2, _ = route(pocket_board(), tmp, 'pocket2',
                                    ['W1', 'X', 'Y', 'Z'])
        check("a second run rips the same nets and writes the same copper",
              ripped2 == ripped and _copper(out2) == _copper(out))
        out, doc, ripped, _ = route(pocket_board(), tmp, 'two_rungs',
                                    ['W1', 'X', 'Y', 'Z'], ['--max-ripup', '2'])
        check("with two rungs (--max-ripup 2) the wall alone is ripped and "
              "U1.3 connects", ripped == {'W1'} and not doc.get('failed_multipoint')
              and graded(out),
              f"{doc.get('failed_multipoint')}, ripped {sorted(ripped)}")

        print("4. two walls, one with dead ground behind it")
        out, doc, ripped, log = route(tie_board(), tmp, 'tie', ['W1', 'W2', 'D'])
        check("U1.3 connects", not doc.get('failed_multipoint'),
              str(doc.get('failed_multipoint')))
        check("the separating wall W1 is ripped, W2 never",
              'W1' in ripped and 'W2' not in ripped, str(sorted(ripped)))
        check("the routed board is DRC-clean and connected", graded(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    attribution_sides()
    rung_slot()
    whole_boards()
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
