"""Declared pin and gate swaps (`py_placer/swap_pins.py`, `placement/pin_swap.py`).

A design brief's `pin_swaps[]` names gates of pins that may trade nets; the
tool picks, on a placed board, the assignment that lowers place_simple's
signal cost and rewrites only those pads.

Rows:
  - a declaration that cannot be honoured is refused with its reason: gates
    of different sizes, a pin in two gates, a pin both in a gate and common,
    a part in two rows, an `adjacent` naming no gate, a pin number written as
    a number -- by the brief loader too -- and on the board a missing pin, a
    pool of different footprints, gates whose pins are of different types;
  - nothing declared is a refusal, not a silent no-op;
  - one part, single-pin gates: a fan of four crossing lines comes out with
    no crossing, the written board carries exactly the planned nets, and
    the schematic symbol's pins are renumbered to match;
  - multi-pin gates (a quad receiver's A/B/Y triplets) move as one block,
    pin k onto pin k, and a common pin never changes;
  - a pool of two identical parts trades lines between them, and names both
    parts as not writable to the schematic as pin renumberings;
  - multipoint lines: only the declared part's pads change, every other pad
    of those nets keeps its net, and an unused (net-less) channel moves with
    its gate;
  - a protected net, a net with locked copper and a pad with copper on it
    keep their gates;
  - a differential pair through a flow-through ESD array (two-pin gates),
    which the unconstrained optimum would split, stays on two
    declared-adjacent gates, and does not move at all without `adjacent`;
  - an input already at its optimum is left byte-identical;
  - on kicad_files/tigard.kicad_pcb, declaring its 4x resistor arrays'
    elements lowers the signal cost (measured, printed).

    python3 tests/test_pin_swap.py
"""

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('py_router', 'py_placer', 'tests'):
    sys.path.insert(0, os.path.join(ROOT, _d))

from run_utils import check, evidence  # noqa: E402

from kicad_parser import parse_kicad_pcb  # noqa: E402
from placement import pin_swap as ps  # noqa: E402
from placement import design_brief as db  # noqa: E402

TOOL = os.path.join(ROOT, 'py_placer', 'swap_pins.py')


# --------------------------------------------------------------------------
# fixtures

class Board:
    """A placed board as TEXT: footprints with numbered pads at given local
    positions and nets, and tracks that may be locked."""

    def __init__(self):
        self.nets = ['']
        self.items = []
        self._n = 0

    def _uid(self):
        self._n += 1
        return 'pinswap-%04d' % self._n

    def net(self, name):
        if not name:
            return 0
        if name not in self.nets:
            self.nets.append(name)
        return self.nets.index(name)

    def part(self, ref, x, y, pads, name='test:QUAD', value='V', pintypes=None):
        """`pads`: [(number, local x, local y, net name or '')]."""
        body = []
        for num, lx, ly, net in pads:
            n = self.net(net)
            nt = ' (net %d "%s")' % (n, net) if n else ''
            pt = ' (pintype "%s")' % pintypes[num] if pintypes and num in pintypes else ''
            body.append('    (pad "%s" smd rect (at %s %s) (size 0.4 0.4) '
                        '(layers "F.Cu" "F.Mask")%s%s (uuid "%s"))'
                        % (num, lx, ly, nt, pt, self._uid()))
        self.items.append(
            '  (footprint "%s" (layer "F.Cu") (uuid "%s") (at %s %s)\n'
            '    (property "Reference" "%s" (at 0 -2 0) (layer "F.SilkS") '
            '(uuid "%s"))\n'
            '    (property "Value" "%s" (at 0 2 0) (layer "F.Fab") (uuid "%s"))\n'
            '%s\n  )' % (name, self._uid(), x, y, ref, self._uid(), value,
                         self._uid(), '\n'.join(body)))

    def seg(self, x1, y1, x2, y2, net, locked=False):
        self.items.append('  (segment (start %s %s) (end %s %s) (width 0.2)%s '
                          '(layer "F.Cu") (net %d) (uuid "%s"))'
                          % (x1, y1, x2, y2, ' (locked yes)' if locked else '',
                             self.net(net), self._uid()))

    def write(self, path):
        out = ['(kicad_pcb (version 20241229) (generator "synth") '
               '(generator_version "9.0")\n'
               '  (general (thickness 1.6) (legacy_teardrops no))\n'
               '  (paper "A4")\n'
               '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) '
               '(37 "F.SilkS" user) (39 "F.Mask" user) (44 "Edge.Cuts" user) '
               '(49 "F.Fab" user))\n'
               '  (setup (pad_to_mask_clearance 0))']
        out += ['  (net %d "%s")' % (i, n) for i, n in enumerate(self.nets)]
        out += self.items
        out.append('  (gr_rect (start 0 0) (end 60 40) (layer "Edge.Cuts") '
                   '(stroke (width 0.1) (type solid)) (fill none) '
                   '(uuid "pinswap-edge"))')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(out) + '\n)\n')
        return path


def brief(path, rows):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'schema': 1, 'kind': 'design-brief', 'pin_swaps': rows}, fh)
    return path


def fan(b, ref='U1', x=10, y=5, order=(3, 2, 1, 0), lines=4, conn='J1',
        cx=30, name='test:QUAD'):
    """`ref`'s single-pin channels 1..4 at y, y+1, ..; a connector's pins at
    the same heights to the right; channel k carries line `order[k]`, so
    (3, 2, 1, 0) crosses every line with every other."""
    b.part(ref, x, y, [(str(k + 1), 0, k, 'L%d' % order[k] if order[k] is not None else '')
                       for k in range(lines)], name=name)
    b.part(conn, cx, y, [(str(k + 1), 0, k, 'L%d' % k) for k in range(lines)],
           name='test:CONN')


def schematic(path):
    """A schematic whose U1 symbol has pins A..D numbered 1..4."""
    pins = ''.join('      (pin passive line (at 0 %d 0) (length 1) (name "%s") (number "%d"))\n'
                   % (k, n, k + 1) for k, n in enumerate('ABCD'))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('(kicad_sch (version 20231120) (generator "synth")\n'
                 '  (lib_symbols\n    (symbol "t:QUAD"\n     (symbol "QUAD_1_1"\n'
                 + pins + '     )\n    )\n  )\n'
                 '  (symbol\n    (lib_id "t:QUAD") (at 50 50 0) (unit 1)\n'
                 '    (property "Reference" "U1" (at 0 0 0))\n  )\n)\n')
    return path


def nets_of(board, ref):
    pcb = parse_kicad_pcb(board)
    return {str(p.pad_number): p.net_name or '' for p in pcb.footprints[ref].pads}


def tmpdir():
    return tempfile.mkdtemp(prefix='pinswap_')


def run_tool(board, out, *extra, accept=True):
    return check([sys.executable, '-X', 'utf8', TOOL, board, out, *extra],
                 accept=accept, timeout=300)


# --------------------------------------------------------------------------
# rows

def test_refused_declarations():
    """Each incoherent row is refused with its reason, by the row parser and
    by the brief loader that goes through it."""
    cases = [
        ([{'ref': 'U1', 'gates': [['1', '2'], ['3']]}], 'different sizes'),
        ([{'ref': 'U1', 'gates': [['1'], ['1']]}], "pin '1' is in gate 0 and gate 1"),
        ([{'ref': 'U1', 'gates': [['1'], ['2']], 'common': ['2']}], 'both in a gate and `common`'),
        ([{'ref': 'U1', 'gates': [['1'], ['2']]}, {'refs': ['U2', 'U1'], 'gates': [['1'], ['2']]}],
         'U1 is already in pin_swaps[0]'),
        ([{'ref': 'U1', 'gates': [['1'], ['2']], 'adjacent': [['1', '9']]}], "pin '9' is in no gate"),
        ([{'ref': 'U1', 'gates': [[1], [2]]}], 'is not a string'),
        ([{'ref': 'U1', 'refs': ['U2'], 'gates': [['1'], ['2']]}], 'exactly one of `ref`'),
        ([{'ref': 'U1', 'gates': [['1']]}], 'nothing to swap'),
        ([{'ref': 'U1', 'gates': [['1'], ['2']], 'reversible': True}], 'unknown key(s) reversible'),
    ]
    for rows, why in cases:
        try:
            ps.brief_rows(rows)
        except ValueError as exc:
            assert why in str(exc), (why, str(exc))
        else:
            raise AssertionError(f"accepted {rows}")
        try:
            db.brief_from_dict({'schema': 1, 'kind': 'design-brief', 'pin_swaps': rows})
        except db.BriefError as exc:
            assert why in str(exc), (why, str(exc))
        else:
            raise AssertionError(f"brief accepted {rows}")
    assert ps.brief_rows('unknown') == []
    br = db.brief_from_dict({'schema': 1, 'kind': 'design-brief', 'pin_swaps': 'unknown'})
    assert br.pin_swaps_unknown


def test_brief_carries_rows():
    """The brief reports its rows as declared and carried -- applied by
    swap_pins.py, graded by no floorplan rule -- and the ids parse."""
    br = db.brief_from_dict({'schema': 1, 'kind': 'design-brief', 'pin_swaps': [
        {'ref': 'U1', 'gates': [['1'], ['2']], 'why': 'ESD channels'}]})
    frag, rep = db.compile_brief(br)
    assert 'pin_swaps' not in frag, frag
    assert rep['declared'] == ['pin_swaps[0]'], rep['declared']
    assert rep['counts']['pin_swaps'] == 1
    rec = db.parse_clause_id('pin_swaps[0]')
    assert rec and rec['kind'] == 'pin_swaps' and rec['row'] == 0, rec
    cov = db.clause_coverage(rep, {}, rules_run=())
    row = cov['clauses'][0]
    assert row['state'] == 'carried' and 'swap_pins.py' in row['why'], row
    assert cov['complete'], cov
    assert 'pin-swap pool' in db.format_report(rep, path='b.design-brief.json')


def test_refused_on_board():
    d = tmpdir()
    try:
        b = Board()
        fan(b)
        b.part('U2', 10, 20, [('1', 0, 0, ''), ('2', 0, 1, '')], name='test:OTHER')
        b.part('U3', 40, 20, [('1', 0, 0, 'A'), ('2', 0, 1, 'B')], name='test:DIODES',
               pintypes={'1': 'input', '2': 'output'})
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        out = os.path.join(d, 'o.kicad_pcb')
        for rows, why in [
                ([{'ref': 'U1', 'gates': [['1'], ['9']]}], 'U1 has no pin 9'),
                ([{'refs': ['U1', 'U2'], 'gates': [['1'], ['2']]}], 'different footprints'),
                ([{'ref': 'U9', 'gates': [['1'], ['2']]}], 'U9 not on this board'),
                ([{'ref': 'U3', 'gates': [['1'], ['2']]}], 'pins of different types')]:
            brief(os.path.join(d, 'b.design-brief.json'), rows)
            check([sys.executable, '-X', 'utf8', TOOL, board, out], refuse=why, code=2)
            assert not os.path.exists(out)
        os.remove(os.path.join(d, 'b.design-brief.json'))
        check([sys.executable, '-X', 'utf8', TOOL, board, out], refuse='nothing declared', code=2)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# swaps

def test_single_part_uncrosses():
    d = tmpdir()
    try:
        b = Board()
        fan(b)
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        brief(os.path.join(d, 'b.design-brief.json'),
              [{'ref': 'U1', 'gates': [['1'], ['2'], ['3'], ['4']], 'why': 'ESD channels'}])
        out = os.path.join(d, 'o.kicad_pcb')
        rep_path = os.path.join(d, 'r.json')
        sch = os.path.join(d, 'sch')
        os.mkdir(sch)
        sch_file = schematic(os.path.join(sch, 'b.kicad_sch'))
        run_tool(board, out, '--json-out', rep_path, '--schematic-dir', sch)
        rep = json.load(open(evidence(rep_path), encoding='utf-8'))
        # the schematic follows: the symbol pin that carried L0 (named D,
        # numbered 4) is now numbered 1, and so on round
        import re
        text = open(sch_file, encoding='utf-8').read()
        pins = dict(re.findall(r'\(name "(\w)"\) \(number "(\d)"', text))
        assert pins == {'A': '4', 'B': '3', 'C': '2', 'D': '1'}, pins
        assert rep['schematic']['unwritable'] == [], rep['schematic']
        assert rep['before']['crossings'] == 6, rep['before']
        assert rep['after']['crossings'] == 0, rep['after']
        assert nets_of(evidence(out), 'U1') == {'1': 'L0', '2': 'L1', '3': 'L2', '4': 'L3'}
        assert nets_of(out, 'J1') == nets_of(board, 'J1')
        assert len(rep['changes']) == 4, rep['changes']
        assert {c['ref'] for c in rep['changes']} == {'U1'}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_triplet_gates_and_common_pin():
    """A quad receiver: gates (A, B, Y), an enable pin common to all. Each
    receiver's three nets go to three connectors placed in reverse."""
    d = tmpdir()
    try:
        b = Board()
        # SOIC-16-like: gates (2,1,3) (6,7,5) (10,9,11) (14,15,13); 4 = EN.
        gates = [('2', '1', '3'), ('6', '7', '5'), ('10', '9', '11'), ('14', '15', '13')]
        pads = []
        for k, g in enumerate(gates):
            for j, pin in enumerate(g):
                # receiver k's pins carry channel (3 - k)'s nets
                pads.append((pin, 0, k * 3 + j, 'C%d_%s' % (3 - k, 'ABY'[j])))
        pads.append(('4', 2, 0, 'EN'))
        b.part('U1', 10, 2, pads, name='test:SOIC16')
        for k in range(4):
            b.part('J%d' % (k + 1), 30, 2 + k * 3,
                   [(str(j + 1), 0, j, 'C%d_%s' % (k, 'ABY'[j])) for j in range(3)],
                   name='test:CONN3')
        b.part('R1', 12, 0, [('1', 0, 0, 'EN'), ('2', 1, 0, '')], name='test:R')
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        brief(os.path.join(d, 'b.design-brief.json'),
              [{'ref': 'U1', 'gates': [list(g) for g in gates], 'common': ['4']}])
        out = os.path.join(d, 'o.kicad_pcb')
        run_tool(board, out)
        n = nets_of(evidence(out), 'U1')
        for k, g in enumerate(gates):
            chans = {n[p][:2] for p in g}
            assert len(chans) == 1, (g, [n[p] for p in g])     # one block
            assert [n[p][3] for p in g] == ['A', 'B', 'Y'], [n[p] for p in g]
        assert [n[g[0]] for g in gates] == ['C0_A', 'C1_A', 'C2_A', 'C3_A'], n
        assert n['4'] == 'EN'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pool_between_parts():
    """Two identical two-channel parts, each wired to the other's connector:
    the pool trades the lines between them."""
    d = tmpdir()
    try:
        b = Board()
        b.part('U1', 10, 5, [('1', 0, 0, 'L2'), ('2', 0, 1, 'L3')], name='test:DUAL')
        b.part('U2', 10, 25, [('1', 0, 0, 'L0'), ('2', 0, 1, 'L1')], name='test:DUAL')
        b.part('J1', 30, 5, [('1', 0, 0, 'L0'), ('2', 0, 1, 'L1')], name='test:CONN')
        b.part('J2', 30, 25, [('1', 0, 0, 'L2'), ('2', 0, 1, 'L3')], name='test:CONN')
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        brief(os.path.join(d, 'b.design-brief.json'),
              [{'refs': ['U1', 'U2'], 'gates': [['1'], ['2']]}])
        out = os.path.join(d, 'o.kicad_pcb')
        rep_path = os.path.join(d, 'r.json')
        sch = os.path.join(d, 'sch')
        os.mkdir(sch)
        run_tool(board, out, '--json-out', rep_path, '--schematic-dir', sch)
        assert nets_of(evidence(out), 'U1') == {'1': 'L0', '2': 'L1'}
        assert nets_of(out, 'U2') == {'1': 'L2', '2': 'L3'}
        rep = json.load(open(evidence(rep_path), encoding='utf-8'))
        assert rep['after']['tree_mm'] < rep['before']['tree_mm'], rep
        # a net that came from the other part cannot be a pin renumbering
        assert sorted(rep['schematic']['unwritable']) == ['U1', 'U2'], rep['schematic']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_multipoint_and_unused_channel():
    """Each line also reaches a series resistor and a pull-up: only U1's pads
    change. Channel 4 is unused (no net) and moves with its gate."""
    d = tmpdir()
    try:
        b = Board()
        fan(b, order=(None, 2, 1, 0))
        for k in range(3):
            b.part('R%d' % (k + 1), 20, 5 + k, [('1', 0, 0, 'L%d' % k), ('2', 1, 0, 'X%d' % k)],
                   name='test:R')
            b.part('RP%d' % (k + 1), 25, 5 + k, [('1', 0, 0, 'L%d' % k), ('2', 1, 0, 'VCC')],
                   name='test:R')
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        brief(os.path.join(d, 'b.design-brief.json'),
              [{'ref': 'U1', 'gates': [['1'], ['2'], ['3'], ['4']]}])
        out = os.path.join(d, 'o.kicad_pcb')
        run_tool(board, out)
        before, after = parse_kicad_pcb(board), parse_kicad_pcb(evidence(out))
        for ref, fp in before.footprints.items():
            if ref == 'U1':
                continue
            assert ({p.pad_number: p.net_name for p in fp.pads}
                    == {p.pad_number: p.net_name for p in after.footprints[ref].pads}), ref
        n = nets_of(out, 'U1')
        assert sorted(n.values()) == sorted(nets_of(board, 'U1').values()), n
        assert n['1'] == 'L0' and n['2'] == 'L1' and n['3'] == 'L2' and n['4'] == '', n
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_locked_and_protected_kept():
    """A protected net, a net with locked copper elsewhere and a pad with a
    track on it each keep their gate; the free gate alone has nowhere to go."""
    d = tmpdir()
    try:
        b = Board()
        fan(b)
        b.seg(30, 6, 33, 6, 'L1', locked=True)           # L1: locked copper on J1
        b.seg(10, 8, 8, 8, 'L0')                          # copper on U1.4 (L0)
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        with open(os.path.join(d, 'b.kicad_pro'), 'w', encoding='utf-8') as fh:
            json.dump({'kicad_routing_tools': {'protected_nets': {'L3': 'match group'}}}, fh)
        brief(os.path.join(d, 'b.design-brief.json'),
              [{'ref': 'U1', 'gates': [['1'], ['2'], ['3'], ['4']]}])
        out = os.path.join(d, 'o.kicad_pcb')
        rep_path = os.path.join(d, 'r.json')
        run_tool(board, out, '--json-out', rep_path)
        rep = json.load(open(evidence(rep_path), encoding='utf-8'))
        reasons = {tuple(f['gate']): f['reason'] for f in rep['pools'][0]['frozen']}
        assert 'protected' in reasons[('1',)], reasons
        assert 'locked copper' in reasons[('3',)], reasons
        assert 'copper on it' in reasons[('4',)], reasons
        assert rep['changes'] == [], rep['changes']
        assert nets_of(evidence(out), 'U1') == nets_of(board, 'U1')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_diff_pair_stays_adjacent():
    """A flow-through ESD array (gates (1,10) (2,9) (4,7) (5,6), GND 3/8)
    carries a USB pair on gates 1 and 2 and two lines on 4 and 5, while the
    connector interleaves them (D+, S1, D-, S2): the cheapest assignment
    without the rule splits the pair. With `adjacent` declared the pair
    ends on (1,10)+(2,9) or (4,7)+(5,6); without it the pair does not move."""
    d = tmpdir()
    try:
        def build():
            b = Board()
            ys = {'1': 0, '2': 1, '4': 3, '5': 4}
            nets = {'1': 'USB_D+', '2': 'USB_D-', '4': 'S1', '5': 'S2'}
            pads = []
            for a, z in (('1', '10'), ('2', '9'), ('4', '7'), ('5', '6')):
                pads += [(a, 0, ys[a], nets[a]), (z, 1, ys[a], nets[a])]
            pads += [('3', 0, 2, 'GND'), ('8', 1, 2, 'GND')]
            b.part('U1', 10, 5, pads, name='test:DQA10')
            b.part('J1', 30, 5, [('1', 0, 0, 'USB_D+'), ('2', 0, 1, 'S1'),
                                 ('3', 0, 3, 'USB_D-'), ('4', 0, 4, 'S2')], name='test:CONN')
            return b
        gates = [['1', '10'], ['2', '9'], ['4', '7'], ['5', '6']]
        # the unconstrained optimum splits the pair: priced without pairs
        sub = os.path.join(d, 'free')
        os.mkdir(sub)
        board = build().write(os.path.join(sub, 'b.kicad_pcb'))
        pcb = parse_kicad_pcb(board)
        pools = ps.resolve(ps.brief_rows([{'ref': 'U1', 'gates': gates}]), pcb)
        free = ps.optimise(pcb, pools)
        moved = {(r, p): n for r, p, _o, n in free['changes']}
        dp = {nid for nid, n in pcb.nets.items() if n.name in ('USB_D+', 'USB_D-')}
        assert {p for (_r, p), n in moved.items() if n in dp} & {'4', '5'}, free['changes']
        for adjacent in ([['1', '2'], ['4', '5']], None):
            sub = os.path.join(d, 'a' if adjacent else 'n')
            os.mkdir(sub)
            board = build().write(os.path.join(sub, 'b.kicad_pcb'))
            row = {'ref': 'U1', 'gates': gates, 'common': ['3', '8']}
            if adjacent:
                row['adjacent'] = adjacent
            brief(os.path.join(sub, 'b.design-brief.json'), [row])
            out = os.path.join(sub, 'o.kicad_pcb')
            rep_path = os.path.join(sub, 'r.json')
            run_tool(board, out, '--json-out', rep_path)
            n = nets_of(evidence(out), 'U1')
            assert n['3'] == 'GND' and n['8'] == 'GND'
            for a, z in gates:
                assert n[a] == n[z], (a, z, n)          # a gate is one block
            p = next(a for a, _z in gates if n[a] == 'USB_D+')
            m = next(a for a, _z in gates if n[a] == 'USB_D-')
            rep = json.load(open(evidence(rep_path), encoding='utf-8'))
            if adjacent:
                assert {p, m} in ({'1', '2'}, {'4', '5'}), n
            else:
                assert (p, m) == ('1', '2'), n
                why = [f['reason'] for f in rep['pools'][0]['frozen']]
                assert any('no `adjacent`' in w for w in why), why
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_optimum_left_untouched():
    d = tmpdir()
    try:
        b = Board()
        fan(b, order=(0, 1, 2, 3))
        board = b.write(os.path.join(d, 'b.kicad_pcb'))
        brief(os.path.join(d, 'b.design-brief.json'),
              [{'ref': 'U1', 'gates': [['1'], ['2'], ['3'], ['4']]}])
        out = os.path.join(d, 'o.kicad_pcb')
        run_tool(board, out)
        with open(board, encoding='utf-8') as a, open(evidence(out), encoding='utf-8') as c:
            assert a.read() == c.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_real_board_tigard():
    """tigard's eight 4x resistor arrays: elements (1,8) (2,7) (3,6) (4,5)
    of each array are interchangeable. The tool lowers the signal cost on
    the board as the human placed it, and the write matches the plan."""
    src = os.path.join(ROOT, 'kicad_files', 'tigard.kicad_pcb')
    if not os.path.isfile(src):
        print('  SKIP tigard: board not in this checkout')
        return
    d = tmpdir()
    try:
        board = os.path.join(d, 'tigard.kicad_pcb')
        shutil.copyfile(src, board)
        refs = ['RN%d' % i for i in range(1, 9)]
        brief(os.path.join(d, 'tigard.design-brief.json'),
              [{'ref': r, 'gates': [['1', '8'], ['2', '7'], ['3', '6'], ['4', '5']]}
               for r in refs])
        out = os.path.join(d, 'o.kicad_pcb')
        rep_path = os.path.join(d, 'r.json')
        run_tool(board, out, '--json-out', rep_path)
        rep = json.load(open(evidence(rep_path), encoding='utf-8'))
        b, a = rep['before'], rep['after']
        print(f"  tigard: trees {b['tree_mm']} -> {a['tree_mm']} mm, crossings "
              f"{b['crossings']} -> {a['crossings']}, {len(rep['changes'])} pin(s) changed")
        assert a['cost'] <= b['cost']
        assert {c['ref'] for c in rep['changes']} <= set(refs)
    finally:
        shutil.rmtree(d, ignore_errors=True)


TESTS = [test_refused_declarations, test_brief_carries_rows, test_refused_on_board,
         test_single_part_uncrosses, test_triplet_gates_and_common_pin,
         test_pool_between_parts, test_multipoint_and_unused_channel,
         test_locked_and_protected_kept, test_diff_pair_stays_adjacent,
         test_optimum_left_untouched, test_real_board_tigard]


if __name__ == '__main__':
    fails = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {t.__name__}: {e}")
    print('ALL PASS' if not fails else f'{fails} FAILED')
    sys.exit(1 if fails else 0)
