#!/usr/bin/env python3
"""A via beside a through-hole pad of its own net goes when the pad already
reaches the layer it changes to.

A track leaves a through-hole pad on one layer, runs a short stub to a via
and continues on the other layer. The pad's barrel already joins both layers,
so the via buys nothing once the stub can run on the continuation's layer.
The router lays this shape honestly when another net blocks that layer beside
the pad -- the via is then the crossing -- and the shape outlives the reason
when a later step moves the blocker. No pass removed it: the via is reached on
two layers, so every via sweep counts it as load-bearing.

Rows, on `drop_vias_beside_own_tht` and then through route.py on a board file
(both its passes: the normal end of a run and the "nothing to route" return):

  * the stub moves onto the continuation's layer and the via goes
  * a foreign track across that layer keeps the via (it IS the crossing)
  * a stub that branches, an SMD pad, a locked via, protected copper, a net
    outside the scope and a net with a pour are left alone
  * route.py ships the board without the via, check_drc and check_connected
    clean

    python3 tests/test_via_beside_own_tht.py
"""
import os
import subprocess
import sys
import tempfile
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from synth import make_pad, make_seg, make_via, make_net, make_pcb  # noqa: E402
from kicad_parser import BoardInfo, Zone                             # noqa: E402
from check_connected import check_net_connectivity                  # noqa: E402
try:
    from pcb_modification import drop_vias_beside_own_tht           # noqa: E402
except ImportError:                 # a tree without the pass: rows A-C fail
    drop_vias_beside_own_tht = None

FAILS = []


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def tht(x, y, net=1, ref='J1'):
    return make_pad(net, x, y, ref=ref, num='1', size_x=1.7, size_y=1.7, shape='circle',
                    layers=('*.Cu',), drill=1.0, pad_type='thru_hole')


def smd(x, y, net=1, ref='U1', layer='B.Cu'):
    return make_pad(net, x, y, ref=ref, size_x=1.0, size_y=1.0, layers=(layer,))


def board(segs, vias, pads, zones=()):
    by_net = {}
    for p in pads:
        by_net.setdefault(p.net_id, []).append(p)
    b = make_pcb(nets={1: make_net(1, 'SIG'), 2: make_net(2, 'OTHER')},
                 segments=segs, vias=vias, pads_by_net=by_net,
                 board_info=BoardInfo(layers={}, copper_layers=['F.Cu', 'B.Cu']))
    b.zones = list(zones)
    return b


def connected(b, net=1):
    return check_net_connectivity(net, [s for s in b.segments if s.net_id == net],
                                  [v for v in b.vias if v.net_id == net],
                                  b.pads_by_net[net], [])['connected']


def drop(b, scope=frozenset({1}), **kw):
    if drop_vias_beside_own_tht is None:
        return None
    return drop_vias_beside_own_tht(b, set(scope), clearance=0.2, **kw)


check('drop_vias_beside_own_tht exists', drop_vias_beside_own_tht is not None)


# J1.1 at (10, 10); the stub runs on F.Cu down to a via at (10, 12), and the
# net continues on B.Cu to U1, an SMD pad on B.Cu.
def shape():
    stub = make_seg(10, 10, 10, 12, layer='F.Cu')
    via = make_via(10, 12, size=0.5, drill=0.3)
    cont = make_seg(10, 12, 20, 12, layer='B.Cu')
    return stub, via, cont, [tht(10, 10), smd(20, 12)]


print('A. the stub moves to the continuation layer and the via goes')
stub, via, cont, pads = shape()
b = board([stub, cont], [via], pads)
check('fixture: connected, through the via', connected(b) and len(b.vias) == 1)
rs, rv, added = drop(b) or ([], [], [])
check('the via and the F.Cu stub go', [id(x) for x in rv] == [id(via)]
      and [id(x) for x in rs] == [id(stub)], f'{len(rs)} segs, {len(rv)} vias')
check('the stub is laid on B.Cu instead, same span',
      len(added) == 1 and added[0].layer == 'B.Cu'
      and {(added[0].start_x, added[0].start_y), (added[0].end_x, added[0].end_y)}
      == {(10, 10), (10, 12)}, str([(s.layer, s.start_x, s.start_y, s.end_x, s.end_y)
                                    for s in added]))
check('no via, no F.Cu copper left, the net still connected',
      not b.vias and all(s.layer == 'B.Cu' for s in b.segments) and connected(b))

print('B. a foreign track across B.Cu beside the pad: the via is the crossing')
stub, via, cont, pads = shape()
blocker = make_seg(7, 11.3, 13, 11.3, layer='B.Cu', net_id=2)
b = board([stub, cont, blocker], [via],
          pads + [smd(7, 11.3, net=2, ref='U2'), smd(13, 11.3, net=2, ref='U3')])
check('kept', drop(b) == ([], [], []) and len(b.vias) == 1)

print('C. what is never touched')
stub, via, cont, pads = shape()
spur = make_seg(10, 12, 13, 15, layer='F.Cu')      # a second F.Cu branch at the via
b = board([stub, cont, spur], [via], pads + [smd(13, 15, ref='U4', layer='F.Cu')])
check('a via with two branches on the stub layer stays', drop(b) == ([], [], []))
stub, via, cont, pads = shape()
b = board([stub, cont], [via], [smd(10, 10, ref='U5', layer='F.Cu'), pads[1]])
check('a via beside an SMD pad stays', drop(b) == ([], [], []))
stub, via, cont, pads = shape()
via.locked = True
b = board([stub, cont], [via], pads)
check('a locked via stays', drop(b) == ([], [], []))
stub, via, cont, pads = shape()
b = board([stub, cont], [via], pads)
check('protected (input) copper stays',
      drop(b, protected_ids={id(stub)}) == ([], [], []))
b = board([stub, cont], [via], pads)
check('a net outside the scope stays', drop(b, scope={2}) == ([], [], []))
b = board([stub, cont], [via], pads)
check('a skipped net (protected pair, impedance) stays',
      drop(b, skip_net_ids={1}) == ([], [], []))
b = board([stub, cont], [via], pads,
          zones=[Zone(net_id=1, net_name='SIG', layer='B.Cu',
                      polygon=[(0, 0), (30, 0), (30, 30), (0, 30)])])
check('a net with a pour stays (the via may feed it)', drop(b) == ([], [], []))

print('D. route.py ships the board without the via, on both of its paths')
HEAD = """(kicad_pcb (version 20260206) (generator "pcbnew") (generator_version "10.0")
\t(general (thickness 1.6)) (paper "A4")
\t(layers (0 "F.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
\t(setup (pad_to_mask_clearance 0))
\t(net 0 "") (net 1 "SIG") (net 2 "OTHER")
\t(gr_rect (start 0 0) (end 30 24) (stroke (width 0.1) (type solid)) (fill none) (layer "Edge.Cuts"))
"""
THT = """\t(footprint "t:H" (layer "F.Cu") (uuid "aaaaaaaa-0000-0000-0000-0000000{i:05d}") (at {x} {y})
\t\t(property "Reference" "{ref}" (at 0 -2 0) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))
\t\t(pad "1" thru_hole circle (at 0 0) (size 1.7 1.7) (drill 1) (layers "*.Cu" "*.Mask") (net "{net}")
\t\t\t(uuid "cccccccc-0000-0000-0000-0000000{i:05d}")))
"""
SMD = """\t(footprint "t:P" (layer "B.Cu") (uuid "aaaaaaaa-0000-0000-0000-0000000{i:05d}") (at {x} {y})
\t\t(property "Reference" "{ref}" (at 0 -2 0) (layer "B.SilkS") (effects (font (size 1 1) (thickness 0.15))))
\t\t(pad "1" smd rect (at 0 0) (size 1 1) (layers "B.Cu" "B.Mask") (net "{net}")
\t\t\t(uuid "cccccccc-0000-0000-0000-0000000{i:05d}")))
"""
COPPER = """\t(segment (start 10 10) (end 10 12) (width 0.2) (layer "F.Cu") (net "SIG")
\t\t(uuid "bbbbbbbb-0000-0000-0000-000000000001"))
\t(segment (start 10 12) (end 20 12) (width 0.2) (layer "B.Cu") (net "SIG")
\t\t(uuid "bbbbbbbb-0000-0000-0000-000000000002"))
\t(via (at 10 12) (size 0.5) (drill 0.3) (layers "F.Cu" "B.Cu") (net "SIG")
\t\t(uuid "dddddddd-0000-0000-0000-000000000001"))
"""


def write_board(path, with_other):
    fps = THT.format(i=0, ref='J1', x=10, y=10, net='SIG') + \
        SMD.format(i=1, ref='U1', x=20, y=12, net='SIG')
    if with_other:
        fps += SMD.format(i=2, ref='U2', x=5, y=20, net='OTHER') + \
            SMD.format(i=3, ref='U3', x=25, y=20, net='OTHER')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(HEAD + fps + COPPER + ')\n')


def run(argv):
    return subprocess.run([sys.executable, '-X', 'utf8'] + argv,
                          capture_output=True, text=True, cwd=ROOT)


from kicad_parser import parse_kicad_pcb   # noqa: E402
tmp = tempfile.mkdtemp(prefix='tvbt_')
try:
    for tag, with_other, nets in (('nothing-to-route return', False, ['SIG']),
                                  ('normal end of a run', True, ['SIG', 'OTHER'])):
        src = os.path.join(tmp, f'in_{with_other}.kicad_pcb')
        out = os.path.join(tmp, f'out_{with_other}.kicad_pcb')
        write_board(src, with_other)
        r = run([os.path.join(ROOT, 'py_router', 'route.py'), src, out,
                 '--nets', *nets, '--track-width', '0.2', '--clearance', '0.2'])
        ok = os.path.isfile(out)
        b = parse_kicad_pcb(out) if ok else None
        sig = [s for s in b.segments if s.net_id == 1] if ok else []
        check(f'{tag}: the pass ran and says so',
              'Vias beside own through-hole pads (end of run): removed 1 via' in r.stdout,
              (r.stdout + r.stderr)[-600:])
        check(f'{tag}: no SIG via ships, SIG is all B.Cu and connected',
              ok and not [v for v in b.vias if v.net_id == 1] and sig
              and all(s.layer == 'B.Cu' for s in sig)
              and check_net_connectivity(1, sig, [], b.pads_by_net[1], [])['connected'],
              str([(s.layer, s.start_x, s.start_y, s.end_x, s.end_y) for s in sig]))
        if ok:
            drc = run([os.path.join(ROOT, 'py_router', 'check_drc.py'), out,
                       '--clearance', '0.2', '--quiet'])
            check(f'{tag}: check_drc clean', drc.returncode == 0,
                  (drc.stdout + drc.stderr)[-400:])
            cc = run([os.path.join(ROOT, 'py_router', 'check_connected.py'), out,
                      '--nets', *nets, '--quiet'])
            check(f'{tag}: check_connected clean', cc.returncode == 0,
                  (cc.stdout + cc.stderr)[-400:])
    # GUI front (return_results): the same pass, handed to the applier as a
    # change-set -- the input via and stub to remove, the B.Cu stub to add.
    import contextlib
    import io
    import route as _route
    for tag, with_other, nets in (('nothing-to-route return', False, ['SIG']),
                                  ('normal end of a run', True, ['SIG', 'OTHER'])):
        with contextlib.redirect_stdout(io.StringIO()):
            data = _route.batch_route(os.path.join(tmp, f'in_{with_other}.kicad_pcb'),
                                      '', nets, return_results=True,
                                      track_width=0.2, clearance=0.2)[-1]
        vias_out = data.get('vias_to_remove') or []
        segs_out = data.get('segments_to_remove') or []
        laid = [s for r in data.get('results') or [] if r.get('cleanup') == 'tht_via_drop'
                for s in r.get('new_segments') or []]
        check(f'GUI front, {tag}: the input via and its F.Cu stub leave',
              [(v.x, v.y) for v in vias_out] == [(10, 12)]
              and [(s.layer, s.start_y, s.end_y) for s in segs_out] == [('F.Cu', 10, 12)],
              f'{[(v.x, v.y) for v in vias_out]} {[(s.layer, s.start_y, s.end_y) for s in segs_out]}')
        check(f'GUI front, {tag}: the stub arrives on B.Cu',
              [(s.layer, s.start_x, s.start_y, s.end_x, s.end_y) for s in laid]
              == [('B.Cu', 10, 10, 10, 12)], str(laid))
    src = os.path.join(tmp, 'in_False.kicad_pcb')
    out = os.path.join(tmp, 'keep.kicad_pcb')
    run([os.path.join(ROOT, 'py_router', 'route.py'), src, out, '--nets', 'SIG',
         '--track-width', '0.2', '--clearance', '0.2', '--keep-input-copper'])
    check('--keep-input-copper: the input via stays',
          os.path.isfile(out) and len(parse_kicad_pcb(out).vias) == 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILURE(S): ' + ', '.join(FAILS))
sys.exit(1 if FAILS else 0)
