#!/usr/bin/env python3
"""A rescue that would neck a gap below the requested clearance first tries
moving the neighbour in the way, so the gap ships at the rules asked for.

The rescue's rungs route through free space. A neighbour running down the middle
of a channel leaves room for a second track only below the run's clearance, while
the same channel holds both tracks at the run's rules if the neighbour hugs one
wall. The rescue used to take the neck-down: a completed net, delivered 0.127 at a
requested 0.2 (kicad_files/splitflap_driver at 0.2, GND past /SENSOR_F).

1. The gap routes at the run's clearance and width, the neighbour reconnects at
   them, and the write-list holds the moved copper, not the lifted track.
2. A neighbour that cannot make room (its own pad sits mid-channel) goes back as
   it was: the same objects on the board and on its result, and the neck-down ships.
3. A locked neighbour, and copper the run did not lay, are not moved. Nor is
   anything moved while another net still waits: the route at the run's rules
   takes more room than the neck-down, and that room may be the other net's way
   out (kicad_files/watchy: +3V3 past EN, then SDA stayed open).
4. Same input, same copper.
5. route.py end to end on a channel the run's grid cannot share: no neck-down,
   check_drc at the requested clearance and check_connected clean.

    python3 tests/test_rescue_moves_neighbour.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))  # #522
sys.path.insert(0, os.path.join(ROOT, 'py_tools'))  # #522
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kicad_parser import BoardInfo  # noqa: E402
from net_rescue import _net_component_info, rescue_failed_nets  # noqa: E402
from rip_up_reroute import _saved_route_colliders  # noqa: E402
from routing_config import GridRouteConfig  # noqa: E402
from synth import RoutingBoard, make_net, make_pad, make_pcb, make_seg  # noqa: E402

fails = []
OPEN, NEIGHBOUR, WALL = 1, 2, 3
TRACK = CLR = 0.15
# Beside a centred neighbour the channel takes a 0.127/0.127 neck-down (0.456),
# not a nominal track (0.525). With the neighbour against a wall both fit (0.375).
HALF = 0.5


def check(name, cond, detail=''):
    print(("  ok  " if cond else " FAIL ") + name + (f'  {detail}' if detail else ''))
    if not cond:
        fails.append(name)


def _scene(mid_pad=False, locked=False):
    """Two rooms joined by one channel at y=0, walled to the outline. The open
    net's pads are in the rooms; the neighbour runs straight down the channel."""
    bi = BoardInfo(layers={0: 'F.Cu'}, copper_layers=['F.Cu'],
                   board_bounds=(-3.0, -2.0, 6.0, 2.0))
    t = 2.0 - HALF
    walls = [make_seg(0.5, y * (HALF + t / 2), 2.5, y * (HALF + t / 2), net_id=WALL, width=t)
             for y in (1, -1)]
    pads_open = [make_pad(OPEN, -0.4, 0.6, ref='U1', num='1', net_name='OPEN',
                          size_x=0.3, size_y=0.3),
                 make_pad(OPEN, 3.4, 0.6, ref='U2', num='1', net_name='OPEN',
                          size_x=0.3, size_y=0.3)]
    pads_n = [make_pad(NEIGHBOUR, -1.4, 0.0, ref='U3', num='1', net_name='NEIGHBOUR',
                       size_x=0.3, size_y=0.3),
              make_pad(NEIGHBOUR, 4.4, 0.0, ref='U4', num='1', net_name='NEIGHBOUR',
                       size_x=0.3, size_y=0.3)]
    if mid_pad:
        pads_n.append(make_pad(NEIGHBOUR, 1.5, 0.0, ref='U5', num='1',
                               net_name='NEIGHBOUR', size_x=0.15, size_y=0.15))
    run = make_seg(-1.4, 0.0, 4.4, 0.0, net_id=NEIGHBOUR, width=TRACK, locked=locked)
    pcb = make_pcb(nets={OPEN: make_net(OPEN, 'OPEN'),
                         NEIGHBOUR: make_net(NEIGHBOUR, 'NEIGHBOUR'),
                         WALL: make_net(WALL, 'WALL')},
                   segments=walls + [run],
                   pads_by_net={OPEN: pads_open, NEIGHBOUR: pads_n, WALL: []},
                   board_info=bi)
    return pcb, run


def _cfg():
    c = GridRouteConfig()
    c.layers = ['F.Cu']
    c.grid_step = 0.05
    c.clearance = CLR
    c.track_width = TRACK
    c.via_size = 0.5
    c.via_drill = 0.3
    return c


def _rescue(pcb, run, owned=True):
    owner = {'net_name': 'NEIGHBOUR', 'net_id': NEIGHBOUR,
             'new_segments': [run], 'new_vias': []}
    state = SimpleNamespace(pcb_data=pcb, config=_cfg(),
                            routed_results={NEIGHBOUR: owner} if owned else {},
                            results=[owner] if owned else [],
                            remaining_net_ids=[OPEN], routed_net_ids=[NEIGHBOUR],
                            net_history={}, route_index=0, diff_pair_by_net_id={})
    summary = rescue_failed_nets(state, [('OPEN', OPEN)])
    return state, owner, summary


def _copper(pcb, net):
    return sorted((round(s.start_x, 4), round(s.start_y, 4), round(s.end_x, 4),
                   round(s.end_y, 4), s.width) for s in pcb.segments if s.net_id == net)


def _opens_segs(pcb):
    return [s for s in pcb.segments if s.net_id == OPEN]


def test_neighbour_moves():
    pcb, run = _scene()
    state, owner, summary = _rescue(pcb, run)
    check('1. the gap is recovered', summary is not None and summary['recovered'] == ['OPEN'])
    segs = _opens_segs(pcb)
    check('1. the gap ships at the run\'s width',
          bool(segs) and all(abs(s.width - TRACK) < 1e-6 for s in segs),
          str(sorted({s.width for s in segs})))
    used = getattr(pcb.board_info, 'min_clearance_used', None)
    check('1. no clearance below the run\'s is recorded',
          used is None or used >= CLR - 1e-9, str(used))
    check('1. the neighbour reconnected', _net_component_info(pcb, NEIGHBOUR)[0] == 1)
    check('1. the open net is connected', _net_component_info(pcb, OPEN)[0] == 1)
    hits = _saved_route_colliders({'new_segments': segs, 'new_vias': []}, pcb, [OPEN],
                                  CLR - 1e-6)
    check('1. nothing foreign inside the run\'s clearance', not hits, str(len(hits)))
    live = {id(s) for s in pcb.segments}
    check('1. the lifted track left the board and the write-list',
          id(run) not in live and all(s is not run for s in owner['new_segments']))
    theirs = [s for s in pcb.segments if s.net_id == NEIGHBOUR]
    check('1. the write-list holds the moved copper',
          bool(theirs) and {id(s) for s in theirs} <= {id(s) for s in owner['new_segments']})
    return _copper(pcb, OPEN), _copper(pcb, NEIGHBOUR)


def test_neighbour_that_cannot_make_room():
    pcb, run = _scene(mid_pad=True)
    before = [id(s) for s in pcb.segments if s.net_id != OPEN]
    state, owner, summary = _rescue(pcb, run)
    after = [id(s) for s in pcb.segments if s.net_id != OPEN]
    check('2. the neighbour goes back: same objects on the board',
          sorted(before) == sorted(after))
    check('2. ... and on its result', owner['new_segments'] == [run])
    used = getattr(pcb.board_info, 'min_clearance_used', None)
    check('2. the neck-down ships', summary is not None and summary['recovered'] == ['OPEN']
          and used is not None and used < CLR, str(used))


def test_not_moved():
    for name, kw, owned in (('locked', {'locked': True}, True),
                            ('not this run\'s copper', {}, False)):
        pcb, run = _scene(**kw)
        _state, _owner, summary = _rescue(pcb, run, owned=owned)
        used = getattr(pcb.board_info, 'min_clearance_used', None)
        check(f'3. {name}: left where it is', any(s is run for s in pcb.segments))
        check(f'3. {name}: the neck-down ships', summary is not None
              and summary['recovered'] == ['OPEN'] and used is not None and used < CLR)


def test_not_moved_while_another_net_waits():
    pcb, run = _scene()
    later = 4
    pcb.nets[later] = make_net(later, 'OPEN_LATER')
    pcb.pads_by_net[later] = [
        make_pad(later, -2.0, -1.2, ref='U6', num='1', net_name='OPEN_LATER',
                 size_x=0.3, size_y=0.3),
        make_pad(later, -1.0, -1.2, ref='U7', num='1', net_name='OPEN_LATER',
                 size_x=0.3, size_y=0.3)]
    owner = {'net_name': 'NEIGHBOUR', 'net_id': NEIGHBOUR,
             'new_segments': [run], 'new_vias': []}
    state = SimpleNamespace(pcb_data=pcb, config=_cfg(), routed_results={NEIGHBOUR: owner},
                            results=[owner], remaining_net_ids=[OPEN, later],
                            routed_net_ids=[NEIGHBOUR], net_history={}, route_index=0,
                            diff_pair_by_net_id={})
    summary = rescue_failed_nets(state, [('OPEN', OPEN), ('OPEN_LATER', later)])
    used = getattr(pcb.board_info, 'min_clearance_used', None)
    check('3. another net waiting: left where it is', any(s is run for s in pcb.segments))
    check('3. another net waiting: the neck-down ships',
          summary is not None and sorted(summary['recovered']) == ['OPEN', 'OPEN_LATER']
          and used is not None and used < CLR, str(used))


def _run(args):
    return subprocess.run([sys.executable, '-X', 'utf8'] + args, cwd=ROOT, capture_output=True,
                          text=True, encoding='utf-8', errors='replace')


def test_route_end_to_end():
    # The channel's centre sits half a cell off the 0.1 grid and its usable band is
    # narrower than two tracks' pitch plus a cell, so the run's grid fits one track
    # there and the main pass cannot finish, rip-ups included; the rescue's finer
    # grid fits both once the first has moved over.
    b = RoutingBoard(w=24, h=8, layers=('F.Cu', 'B.Cu'))
    half, cy = 0.51, 3.95
    edge = 0.25     # the walls stop at the fab's edge floor, too close for a track
    for lo, hi in ((edge, cy - half), (cy + half, 8 - edge)):
        b.seg(11, (lo + hi) / 2, 13, (lo + hi) / 2, 'WALL', w=hi - lo, locked=True)
    b.pad('W1', 2.0, 1.0, 'WALL', sx=0.3, sy=0.3)
    b.pad('A1', 6.0, 5.0, 'A', sx=0.3, sy=0.3)
    b.pad('A2', 18.0, 5.0, 'A', sx=0.3, sy=0.3)
    b.pad('B1', 5.0, 3.0, 'B', sx=0.3, sy=0.3)
    b.pad('B2', 19.0, 3.0, 'B', sx=0.3, sy=0.3)
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, 'in.kicad_pcb'), os.path.join(d, 'out.kicad_pcb')
        with open(inp, 'w', encoding='utf-8') as f:
            f.write(b.text())
        r = _run(['py_router/route.py', inp, out, '--nets', 'A', 'B', '--layers', 'F.Cu',
                  '--grid-step', '0.1', '--track-width', '0.2', '--clearance', '0.2'])
        check('5. route.py exit 0 (%d)' % r.returncode, r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return
        check('5. the scene reaches the move', 'to route the gap at clearance 0.2' in r.stdout)
        m = re.search(r'JSON_SHIPPED_RULES: (\{.*\})', r.stdout)
        used = json.loads(m.group(1)).get('min_clearance_used') if m else None
        check('5. no neck-down shipped', used is None or used >= 0.2 - 1e-9, str(used))
        drc = _run(['py_router/check_drc.py', out, '--clearance', '0.2', '--no-size-checks'])
        check('5. check_drc clean at the requested clearance', drc.returncode == 0)
        conn = _run(['py_router/check_connected.py', out])
        check('5. check_connected clean', conn.returncode == 0
              and 'ALL NETS FULLY CONNECTED' in conn.stdout)


if __name__ == '__main__':
    first = test_neighbour_moves()
    test_neighbour_that_cannot_make_room()
    test_not_moved()
    test_not_moved_while_another_net_waits()
    second =test_neighbour_moves()
    check('4. same input, same copper', first == second)
    test_route_end_to_end()
    print('\n%s' % ('FAILED: %d' % len(fails) if fails else 'ALL PASSED'))
    sys.exit(1 if fails else 0)
