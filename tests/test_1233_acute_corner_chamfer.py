#!/usr/bin/env python3
"""route.py ships no acute corner between same-net tracks (#1233).

Two arms of one net meeting at less than 90 degrees -- at a shared end or at
a T on another track's body -- leave an acid trap. route.py laid four shapes
of it, all reproduced here: two arms leaving a pad at 45 degrees, a branch
leaving a trunk at 45 degrees (a Y at a shared end, or a T on the trunk's
body), and the off-grid stub from a grid point already past a pad's centre
folding back to it. acute_corners.chamfer_acute_corners now lands one arm on
the other through a cut perpendicular to it, in the shared cleanup pipeline
and again at the end of route.py's run.

  1. the four shapes: every acute corner is cut, the cut meets its arm at
     90 degrees, the pads stay connected;
  2. guards: a cut the clearance predicate refuses, a locked arm and a
     read-only arm leave the corner and the copper as they were, and a track
     held at the joint by its cap alone is not cut loose;
  3. the end-of-run pass on the written file (CLI) and on the write model
     (GUI), and the pipeline pass's write-list bookkeeping;
  4. route.py on two small boards it routes with acute corners: none ships,
     and check_drc + check_connected stay clean. Control: with the pass off
     (KICAD_ACUTE_CHAMFER=0) the same boards do carry them, so the boards
     still exercise the defect;
  5. route.py on splitflap_driver: no cut leaves a net in copper fragments
     (the fragment sweep would send it to the reconciliation).

    python3 tests/test_1233_acute_corner_chamfer.py
"""
import copy
import math
import os
import random
import subprocess
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for p in (ROOT, TESTS_DIR, os.path.join(ROOT, 'py_router')):
    if p not in sys.path:
        sys.path.insert(0, p)

from synth import RoutingBoard                                  # noqa: E402
from kicad_parser import parse_kicad_pcb                        # noqa: E402
from acute_corners import acute_corners, chamfer_acute_corners  # noqa: E402
from check_connected import check_net_connectivity              # noqa: E402
from pcb_modification import make_segment_clears                # noqa: E402
from routing_config import GridRouteConfig                      # noqa: E402
from run_utils import evidence, run_step                        # noqa: E402

FAILS = []
CLR = 0.2


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name
          + (f'  {detail}' if detail else ''))


def load(tmp, tag, build):
    rb = RoutingBoard(w=20, h=20)
    build(rb)
    path = os.path.join(tmp, f'{tag}.kicad_pcb')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(rb.text())
    return path, parse_kicad_pcb(path)


def bend(rb, locked=False):
    """P1 -> V(10,10), then back at 45 degrees towards P2: a bare bend."""
    rb.pad('P1', 5, 10, 'A')
    rb.pad('P2', 5, 5, 'A')
    rb.seg(5, 10, 10, 10, 'A', w=0.25)
    rb.seg(10, 10, 9, 9, 'A', w=0.25, locked=locked)
    rb.seg(9, 9, 5, 5, 'A', w=0.25)


def y_joint(rb):
    """A trunk P1-P2 through (10,10); a branch leaves (10,10) at 45 degrees
    to the trunk's east arm and turns north to P3."""
    rb.pad('P1', 5, 10, 'A')
    rb.pad('P2', 15, 10, 'A')
    rb.pad('P3', 11, 5, 'A')
    rb.seg(5, 10, 10, 10, 'A', w=0.25)
    rb.seg(10, 10, 15, 10, 'A', w=0.25)
    rb.seg(10, 10, 11, 9, 'A', w=0.25)
    rb.seg(11, 9, 11, 5, 'A', w=0.25)


def t_joint(rb):
    """The same branch, landing on the body of a one-piece trunk."""
    rb.pad('P1', 5, 10, 'A')
    rb.pad('P2', 15, 10, 'A')
    rb.pad('P3', 11, 5, 'A')
    rb.seg(5, 10, 15, 10, 'A', w=0.25)
    rb.seg(10, 10, 11, 9, 'A', w=0.25)
    rb.seg(11, 9, 11, 5, 'A', w=0.25)


def in_pad_fold(rb):
    """The route's last grid point (10.05, 10.05) is past the off-grid pad
    centre; the stub back to the centre folds at ~49 degrees."""
    rb.pad('P1', 5, 10.05, 'A')
    rb.pad('P2', 10.0187, 10.0143, 'A', sx=0.5, sy=0.6)
    rb.seg(5, 10.05, 10.05, 10.05, 'A', w=0.25)
    rb.seg(10.05, 10.05, 10.0187, 10.0143, 'A', w=0.25)


def soft_held(rb):
    """The bend, plus a third track from P3 whose end stops 0.01 mm short
    of V: it joins the net's copper by its cap alone, which no track end at
    V records. Cutting V..a off would leave it a fragment of its own."""
    bend(rb)
    rb.pad('P3', 10, 15, 'A')
    rb.seg(10, 15, 10, 10.01, 'A', w=0.25)


def fragments(pcb, nid):
    from check_connected import net_copper_fragments
    return net_copper_fragments(nid, [s for s in pcb.segments if s.net_id == nid],
                                [v for v in pcb.vias if v.net_id == nid],
                                pcb.pads_by_net.get(nid, []), [], pcb_data=pcb)['fragments']


def net_ok(pcb):
    nid = next(i for i, n in pcb.nets.items() if n.name == 'A')
    r = check_net_connectivity(nid, [s for s in pcb.segments if s.net_id == nid],
                               [v for v in pcb.vias if v.net_id == nid],
                               pcb.pads_by_net.get(nid, []), [], pcb_data=pcb)
    return bool(r.get('connected'))


def right_angle_cuts(pcb, pairs):
    """Every new piece that is not collinear with the segment it came from
    (the cut) meets a kept track end at 90 degrees or more."""
    for _h, pieces in pairs:
        for c in pieces:
            for s in pcb.segments:
                if s is c or s.net_id != c.net_id:
                    continue
                for (px, py), (qx, qy) in (((c.start_x, c.start_y), (c.end_x, c.end_y)),
                                           ((c.end_x, c.end_y), (c.start_x, c.start_y))):
                    for (sx, sy), (tx, ty) in (((s.start_x, s.start_y), (s.end_x, s.end_y)),
                                               ((s.end_x, s.end_y), (s.start_x, s.start_y))):
                        if math.hypot(sx - px, sy - py) > 1e-6:
                            continue
                        ax, ay, bx, by = qx - px, qy - py, tx - sx, ty - sy
                        cos = (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))
                        if cos > math.cos(math.radians(89.5)):
                            return False
    return True


def shapes(tmp):
    print('1. the four shapes are cut, connected, at right angles')
    for tag, build, want in (('bend', bend, 1), ('y', y_joint, 1),
                             ('t', t_joint, 1), ('fold', in_pad_fold, 1)):
        _p, pcb = load(tmp, tag, build)
        before = acute_corners(pcb.segments)
        st = {}
        pairs = chamfer_acute_corners(pcb, clears=make_segment_clears(
            pcb, None, CLR, None, 0.0), stats=st)
        after = acute_corners(pcb.segments)
        check(f'{tag}: {len(before)} acute corner(s) -> 0, connected',
              len(before) >= want and not after and st['cut'] >= 1 and net_ok(pcb),
              f'after {after}, stats {st}')
        check(f'{tag}: the cut meets its arm at 90 degrees or more',
              right_angle_cuts(pcb, pairs))


def guards(tmp):
    print('2. a refused cut, a locked arm, a read-only arm: left as they were')
    _p, pcb = load(tmp, 'refused', bend)
    n = len(pcb.segments)
    st = {}
    pairs = chamfer_acute_corners(pcb, clears=lambda *a: False, stats=st)
    check('every cut refused: nothing changes, the corner is counted left',
          not pairs and len(pcb.segments) == n and st['left'] >= 1
          and len(acute_corners(pcb.segments)) == 1)
    _p, pcb = load(tmp, 'soft', soft_held)
    nid = next(i for i, x in pcb.nets.items() if x.name == 'A')
    before = fragments(pcb, nid)
    chamfer_acute_corners(pcb, clears=make_segment_clears(pcb, None, CLR, None, 0.0))
    check('a track held at V only by its cap is not cut loose (strict fragments '
          'and connectivity as before)',
          net_ok(pcb) and fragments(pcb, nid) == before == 1,
          f'fragments {before} -> {fragments(pcb, nid)}')
    _p, pcb = load(tmp, 'locked', lambda rb: bend(rb, locked=True))
    pairs = chamfer_acute_corners(pcb, clears=make_segment_clears(pcb, None, CLR, None, 0.0))
    check('a locked arm keeps its corner', not pairs
          and len(acute_corners(pcb.segments)) == 1)
    _p, pcb = load(tmp, 'ro', bend)
    pairs = chamfer_acute_corners(pcb, clears=make_segment_clears(pcb, None, CLR, None, 0.0),
                                  readonly_ids={id(s) for s in pcb.segments})
    check('read-only copper keeps its corner', not pairs
          and len(acute_corners(pcb.segments)) == 1)


def late_and_pipeline(tmp):
    print('3. end-of-run pass (CLI file, GUI write model) and the pipeline')
    from route import _late_acute_chamfer1233
    from improvement_gate import copper_signature
    from cleanup_pipeline import run_post_route_cleanup
    cfg = GridRouteConfig(clearance=CLR, track_width=0.25)

    path, pcb = load(tmp, 'cli', y_joint)
    n = _late_acute_chamfer1233(None, path, False, None, None, {}, {'A'}, False, cfg)
    out = parse_kicad_pcb(evidence(path, 'the chamfered board'))
    check('CLI: the written file carries the cut and no acute corner',
          n == 1 and not acute_corners(out.segments) and net_ok(out),
          f'cut {n}, left {acute_corners(out.segments)}')
    path, pcb = load(tmp, 'keep', y_joint)
    sig = copper_signature(pcb.segments, pcb.vias,
                           {i: x.name for i, x in pcb.nets.items()}.get)
    n = _late_acute_chamfer1233(None, path, False, None, None, sig, {'A'}, True, cfg)
    check('CLI --keep-input-copper: input copper keeps its corner',
          n == 0 and len(acute_corners(parse_kicad_pcb(path).segments)) == 1)

    _p, pcb = load(tmp, 'gui', bend)
    nid = next(i for i, x in pcb.nets.items() if x.name == 'A')
    mine = list(pcb.segments)
    rd = {'results': [{'net_id': nid, 'new_segments': list(mine), 'new_vias': []}]}

    def write_model(r):
        return {nid: [s for x in r['results'] for s in x['new_segments']]}, {}
    n = _late_acute_chamfer1233(pcb, None, True, rd, write_model, {}, {'A'}, False, cfg)
    shipped = [s for x in rd['results'] for s in x['new_segments']]
    check('GUI: the cut rides the run\'s result, pcb_data matches, no input strip',
          n == 1 and not acute_corners(shipped) and not rd.get('segments_to_remove')
          and {id(s) for s in shipped} == {id(s) for s in pcb.segments},
          f'cut {n}')

    _p, pcb = load(tmp, 'pipe', t_joint)
    host = next(s for s in pcb.segments if s.start_x == 5)
    results = [{'net_id': nid, 'new_segments': [s for s in pcb.segments if s is not host],
                'new_vias': []}]
    outc = run_post_route_cleanup(results, pcb, {nid}, cfg)
    written = [s for s in pcb.segments if s is not host]
    check('pipeline: corner cut; the split INPUT trunk is stripped and its '
          'pieces ride a result',
          outc.counts.get('acute_corners_cut') == 1
          and not acute_corners(pcb.segments)
          and host in outc.input_strip_segments
          and {id(s) for r in results for s in r['new_segments']}
          == {id(s) for s in written} and net_ok(pcb),
          f"counts {outc.counts.get('acute_corners_cut')}")


def synth_board(seed):
    """Three 3-pad nets on off-grid pads, as drawn by seed."""
    r = random.Random(seed)
    rb = RoutingBoard(w=20, h=15)
    for n in range(3):
        for k in range(3):
            rb.pad('P%d%d' % (n, k), round(r.uniform(2, 18), 4),
                   round(r.uniform(2, 13), 4), 'N%d' % n, sx=0.5, sy=0.6)
    return rb.text()


def end_to_end(tmp):
    print('4. route.py ships no acute corner (control: the boards carry them '
          'with the pass off)')
    for seed in (1, 3):
        src = os.path.join(tmp, f's{seed}.kicad_pcb')
        with open(src, 'w', encoding='utf-8') as fh:
            fh.write(synth_board(seed))
        outs = {}
        for arm in ('0', '1'):
            dst = os.path.join(tmp, f's{seed}_{arm}.kicad_pcb')
            env = dict(os.environ, KICAD_ACUTE_CHAMFER=arm)
            r = subprocess.run([sys.executable, '-X', 'utf8', 'py_router/route.py',
                                src, dst, '--nets', '*', '--track-width', '0.25',
                                '--clearance', str(CLR)],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=600)
            check(f'seed {seed}, pass {"on" if arm == "1" else "off"}: route.py ran',
                  r.returncode == 0 and 'Traceback' not in r.stdout + r.stderr,
                  (r.stdout + r.stderr)[-400:] if r.returncode else '')
            outs[arm] = acute_corners(parse_kicad_pcb(evidence(dst, 'routed board')).segments)
            if arm == '1':
                run_step(['py_router/check_drc.py', dst])
                run_step(['py_router/check_connected.py', dst])
        check(f'seed {seed}: control -- acute corners with the pass off',
              len(outs['0']) >= 1, f'{len(outs["0"])}')
        check(f'seed {seed}: none with it on (DRC and connectivity clean above)',
              not outs['1'], f'{outs["1"]}')


def splitflap(tmp):
    print('5. splitflap_driver: no cut leaves a net in fragments')
    # On this board a cut once took an arm that a third track joined only by
    # its cap: connectivity graded the same, but the copper fell in two
    # fragments meeting through a pad, the fragment sweep sent /SENSOR_F to
    # the reconciliation, and its re-route shipped a loop and a 57-degree
    # corner. The pass now grades the strict fragments too.
    dst = os.path.join(tmp, 'splitflap.kicad_pcb')
    _rc, out = run_step(['py_router/route.py', 'kicad_files/splitflap_driver.kicad_pcb',
                         dst, '--nets', '*'], outputs=[dst])
    check('no net reported in fragments', 'FRAGMENT SWEEP' not in out)
    check('no acute corner shipped',
          not acute_corners(parse_kicad_pcb(dst).segments))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        shapes(tmp)
        guards(tmp)
        late_and_pipeline(tmp)
        end_to_end(tmp)
        splitflap(tmp)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
