#!/usr/bin/env python3
"""The victim guard: a route run never ships a net less connected than it got.

The board: a KiCad-locked wall splits it in two, with ONE slot that fits one
track. V is routed through the slot on the input; T's two pads sit on either
side of the wall and are bare. Routing T rips V (a small pre-existing net is a
rip candidate by default), T takes the slot, and V has nowhere left to go:
its reroute fails and its restore is refused because T now holds its corridor.
The run then shipped V broken and T connected, and the improvement gate (#600)
ACCEPTED it, since one net gained for one lost is a draw. Repeated over a
congested board, that is a chain of retry steps each connecting two nets and
breaking two others.

Rows:
  1. CLI, KICAD_VICTIM_GUARD=0 (the parent's behaviour, and the proof this
     board reaches the defect): the gate lists V lost and T gained.
  2. CLI, guard on: V ships with exactly its input copper, T returns to its
     input state (bare), the gate lists nothing lost; the log names both.
  3. CLI, guard on, plus U, a net that needs nothing of V's corridor: U ships
     routed. The guard undoes the trade, not the run.
  4. GUI front (batch_route with return_results): the same restore, as an
     empty change-set, and results_data carries the guard's report.
  5. victim_guard units: the closure pulls the corridor holder, stops at a
     frozen net, brings a diff-pair partner; the block scanner takes track
     arcs but not a zone outline's arc; both net dialects resolve.

    python3 tests/test_victim_guard.py
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

from run_utils import check as run_check, evidence, tool        # noqa: E402
from synth import RoutingBoard, make_net, make_pcb, make_seg    # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def slot_board(with_u=False):
    b = RoutingBoard(w=40, h=30)
    b.pad('T1', 10, 10, 'T')
    b.pad('T2', 30, 10, 'T')
    b.pad('V1', 10, 20, 'V')
    b.pad('V2', 30, 20, 'V')
    b.seg(10, 20, 30, 20, 'V', w=0.2)
    # The wall: free width 0.8 mm in the slot, one 0.2 mm track at 0.2 mm
    # clearance, not two.
    b.seg(20, 0.5, 20, 19.45, 'WALL', w=0.3, locked=True)
    b.seg(20, 20.55, 20, 29.5, 'WALL', w=0.3, locked=True)
    if with_u:
        b.pad('U1', 24, 4, 'U')
        b.pad('U2', 36, 4, 'U')
    return b.text()


ARGS = ['--layers', 'F.Cu', '--track-width', '0.2', '--clearance', '0.2',
        '--grid-step', '0.1', '--escalation', 'off', '--no-fix-drc-settings']


def route(text, tmp, tag, nets, guard=True):
    inp = os.path.join(tmp, f'{tag}_in.kicad_pcb')
    out = os.path.join(tmp, f'{tag}_out.kicad_pcb')
    with open(inp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    evidence(inp, 'synthetic input board')
    env = dict(os.environ, KICAD_PLANE_FINALIZE='0',
               KICAD_VICTIM_GUARD='1' if guard else '0')
    r = run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                   '--nets'] + list(nets) + ARGS, accept=True, env=env)
    evidence(out, 'routed output')
    log = r.stdout + r.stderr
    m = re.findall(r'JSON_IMPROVEMENT_GATE: (\{.*\})', log)
    if not m:
        raise AssertionError('BROKEN TEST: the improvement gate did not run\n'
                             + log[-1500:])
    return inp, out, log, json.loads(m[-1])


def signature(path):
    from improvement_gate import copper_signature
    from kicad_parser import parse_kicad_pcb
    pcb = parse_kicad_pcb(path)
    names = {i: n.name for i, n in pcb.nets.items()}
    return copper_signature(pcb.segments, pcb.vias, names.get)


def connected(path):
    from improvement_gate import net_connectivity_map
    from kicad_parser import parse_kicad_pcb
    pcb = parse_kicad_pcb(path)
    return {pcb.nets[i].name: c for i, (c, _d) in
            net_connectivity_map(pcb).items()}


def whole_board():
    tmp = tempfile.mkdtemp(prefix='tvguard_')
    try:
        print("1. guard off: the board reaches the defect")
        _i, _o, _log, gate = route(slot_board(), tmp, 'off', ['T'], guard=False)
        if gate.get('lost') != ['V'] or gate.get('gained') != ['T']:
            print(f"  [BROKEN TEST] without the guard the run did not trade V "
                  f"for T (gate {gate}): this board no longer reaches the "
                  f"defect")
            failures.append('broken fixture')
            return
        print("  [ok] without the guard: V lost, T gained, and the gate "
              f"accepts the draw ({gate.get('verdict')})")

        print("2. guard on: the victim ships as the input had it")
        inp, out, log, gate = route(slot_board(), tmp, 'on', ['T'])
        check("the gate lists nothing lost", gate.get('lost') == [], str(gate))
        sin, sout = signature(inp), signature(out)
        check("V carries exactly its input copper", sout.get('V') == sin.get('V'),
              f"in {sin.get('V')} out {sout.get('V')}")
        check("T is back to its input state (bare)", not sout.get('T'),
              str(sout.get('T')))
        check("V is connected on the shipped board", connected(out).get('V'))
        check("the log names the victim and the corridor holder",
              re.search(r'Victim guard: 1 net\(s\) .*\(V\).*\(T\)', log)
              is not None,
              next((l.strip() for l in log.splitlines()
                    if 'Victim guard' in l), 'no Victim guard line'))

        print("3. guard on: a gain that needs no victim's corridor is kept")
        inp, out, log, gate = route(slot_board(with_u=True), tmp, 'u',
                                    ['T', 'U'])
        check("the gate lists U gained and nothing lost",
              gate.get('gained') == ['U'] and gate.get('lost') == [], str(gate))
        sin, sout = signature(inp), signature(out)
        check("V carries exactly its input copper", sout.get('V') == sin.get('V'))
        check("U ships routed and connected",
              bool(sout.get('U')) and connected(out).get('U'))

        print("4. GUI front: the same restore, as an empty change-set")
        import fab_tiers
        from kicad_parser import parse_kicad_pcb
        from route import batch_route
        prev = fab_tiers.get_escalation_policy()
        os.environ['KICAD_PLANE_FINALIZE'] = '0'
        try:
            fab_tiers.set_escalation_policy('off')
            src = os.path.join(tmp, 'on_in.kicad_pcb')
            _ok, _bad, _t, res = batch_route(
                input_file=src, output_file='', net_names=['T'],
                layers=['F.Cu'], return_results=True,
                pcb_data=parse_kicad_pcb(src), track_width=0.2,
                clearance=0.2, grid_step=0.1)
        finally:
            fab_tiers.set_escalation_policy(*prev)
            os.environ.pop('KICAD_PLANE_FINALIZE', None)
        vg = res.get('victim_guard') or {}
        check("results_data reports V restored with T",
              vg.get('victims') == ['V'] and vg.get('restored') == ['T', 'V'],
              str(vg))
        added = [s for r in res.get('results') or []
                 for s in (r.get('new_segments') or []) + (r.get('new_vias') or [])]
        removed = (res.get('segments_to_remove') or []) + \
            (res.get('vias_to_remove') or [])
        check("the change-set adds and removes nothing",
              not added and not removed, f"{len(added)} added, "
              f"{len(removed)} removed")
        gate = res.get('improvement_gate') or {}
        check("the GUI gate lists nothing lost", gate.get('lost') == [],
              str(gate))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def units():
    print("5. victim_guard units")
    from victim_guard import (restore_closure, block_net_name,
                              _top_level_copper_blocks,
                              restore_input_copper_text)
    pcb = make_pcb(nets={1: make_net(1, 'V'), 2: make_net(2, 'T'),
                         3: make_net(3, 'Z'), 4: make_net(4, 'TN')})
    v_in = make_seg(10, 20, 30, 20, net_id=1)
    pcb.segments = [make_seg(10, 20.1, 30, 20.1, net_id=2),
                    make_seg(25, 0, 25, 30, net_id=3, layer='B.Cu')]
    inp = {1: ([v_in], []), 2: ([], [])}
    got, refused = restore_closure([1], inp, pcb, 0.2)
    check("the closure pulls the net holding the victim's corridor",
          got == {1, 2} and not refused, f"{got} {refused}")
    got, refused = restore_closure([1], inp, pcb, 0.2, frozen={2})
    check("a frozen holder refuses the victim", not got and refused == {1: 2},
          f"{got} {refused}")
    got, refused = restore_closure([1], inp, pcb, 0.2, partner_of={2: 4, 4: 2})
    check("a diff-pair holder brings its partner", got == {1, 2, 4},
          str(got))
    pcb.segments = [make_seg(10, 25, 30, 25, net_id=2)]
    got, _ = restore_closure([1], inp, pcb, 0.2)
    check("a clear corridor restores the victim alone", got == {1}, str(got))

    text = ('(kicad_pcb\n\t(net 1 "V")\n\t(net 2 "T")\n'
            '\t(zone (net 2) (polygon (pts (arc (start 0 0) (mid 1 1) '
            '(end 2 0)))))\n'
            '\t(segment (start 1 1) (end 2 2) (width 0.2) (layer "F.Cu") '
            '(net 1) (uuid "a"))\n'
            '\t(arc (start 0 0) (mid 1 1) (end 2 0) (width 0.2) '
            '(layer "F.Cu") (net "T") (uuid "b"))\n'
            '\t(via (at 3 3) (size 0.5) (drill 0.3) (layers "F.Cu" "B.Cu") '
            '(net 1) (uuid "c"))\n)\n')
    kinds = [k for _s, _e, k in _top_level_copper_blocks(text)]
    check("the scanner takes the track arc, not the zone outline's",
          kinds == ['segment', 'arc', 'via'], str(kinds))
    names = {1: 'V', 2: 'T'}
    nets = [block_net_name(text[s:e], names)
            for s, e, _k in _top_level_copper_blocks(text)]
    check("both net dialects resolve", nets == ['V', 'T', 'V'], str(nets))
    shipped = text.replace('(start 1 1) (end 2 2)', '(start 5 5) (end 6 6)')
    back, cut, add = restore_input_copper_text(shipped, text, {'V'},
                                               names, names)
    check("the input's own block replaces the shipped one, the rest stays",
          (cut, add) == (1, 1) and '(start 1 1) (end 2 2)' in back
          and '(start 5 5)' not in back and back.count('(via') == 1,
          f"cut {cut} add {add}")


def main():
    units()
    whole_board()
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
