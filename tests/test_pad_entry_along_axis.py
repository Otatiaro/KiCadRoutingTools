#!/usr/bin/env python3
"""A track lands on a fine-pitch row pad through its END, along its axis.

On a 0.5 mm pitch row (0.3 x 1.6 mm pads, 0.1 mm track and clearance) the
0.2 mm gap between two pads of the part leaves room for a track centreline
right beside the pad, its copper over the pad's edge and 0.13 mm from the
neighbour. The A* reached the pad centre from there -- down the gap,
alongside the pad, then a 45-degree jog into the centre -- because the pad's
own copper is free to its own net and the detour costs nothing on an octile
grid. The copper is DRC-clean and connected, but it runs between two pads
of the same package and enters the pad from its side, which is not how a
fine-pitch pin is landed.

The board: two rows of eight pads, one with vertical pads and one with
horizontal pads (each row's pitch off the 0.05 mm grid, as a placed part's
usually is). Four pads of the first row are wired to pads above and below
it at diagonal offsets; two pads of the second row are wired, one of them
on a three-pad net (the multipoint path). The other pads are on single-pad
nets.

Checks:
  1. no wired pad has its own net's copper over a long side, along its
     length (a track beside the pad or a jog into it from the gap);
  2. every wired pad is left along its axis: the copper over the pad runs
     parallel to its long axis and reaches at least one track width plus
     clearance past the end before it turns;
  3. every net routes, check_drc.py and check_connected.py are clean.

    python3 tests/test_pad_entry_along_axis.py
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

from run_utils import check as run_check, evidence, tool        # noqa: E402
from synth import RoutingBoard                                  # noqa: E402

failures = []
PITCH, PAD_W, PAD_L = 0.5, 0.3, 1.6
TRACK = CLEARANCE = 0.1
STEP = 0.05
# (ref, origin off the grid, pads vertical, pad number -> partner pad(s))
ROWS = [
    ('U1', (20.019, 15.0), True,
     {2: [(17.0, 19.5)], 4: [(22.5, 19.5)], 5: [(17.5, 10.5)], 7: [(23.0, 10.5)]}),
    ('U2', (32.0, 15.019), False,
     {3: [(28.0, 11.5)], 6: [(36.0, 18.5), (36.0, 12.0)]}),
]


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def net_name(ref, num, wired):
    return f'{ref}_SIG{num}' if num in wired else f'{ref}_NC{num}'


def board():
    b = RoutingBoard()
    for ref, (ox, oy), vertical, wired in ROWS:
        pads = []
        for i in range(8):
            num, off = i + 1, round((i - 3.5) * PITCH, 3)
            dx, dy, sx, sy = ((off, 0, PAD_W, PAD_L) if vertical
                              else (0, off, PAD_L, PAD_W))
            pads.append((str(num), dx, dy, sx, sy, net_name(ref, num, wired)))
        b.part(ref, ox, oy, pads)
        for num, partners in wired.items():
            for k, (x, y) in enumerate(partners):
                b.pad(f'R{ref}{num}{k}', x, y, net_name(ref, num, wired),
                      sx=0.6, sy=0.6)
    return b.text()


def _frame(p, x, y):
    """(|along|, |lat|, half-length, half-width) of point (x, y) in pad p's
    frame (long axis = along)."""
    dx, dy = x - p.global_x, y - p.global_y
    if p.size_y >= p.size_x:
        return abs(dy), abs(dx), p.size_y / 2, p.size_x / 2
    return abs(dx), abs(dy), p.size_x / 2, p.size_y / 2


def _points(s, step=0.01):
    n = max(1, int(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) / step))
    for i in range(n + 1):
        yield (s.start_x + (s.end_x - s.start_x) * i / n,
               s.start_y + (s.end_y - s.start_y) * i / n)


def load(path):
    from kicad_parser import parse_kicad_pcb
    with contextlib.redirect_stdout(io.StringIO()):
        pcb = parse_kicad_pcb(path)
    wired = {(ref, str(n)) for ref, _o, _v, w in ROWS for n in w}
    pads = [p for net in pcb.pads_by_net.values() for p in net
            if (p.component_ref, p.pad_number) in wired]
    segs = [s for s in pcb.segments if s.layer == 'F.Cu']
    return pads, segs


def side_entries(pads, segs):
    """(pad, point) where a wired pad's own net has copper over the pad's
    long side, along its length. 5 um of slack: the grid puts a centreline
    up to half a step off the pad's axis, inside the pad."""
    hits = []
    for p in pads:
        for s in segs:
            if s.net_id != p.net_id:
                continue
            r = s.width / 2
            for x, y in _points(s):
                along, lat, hl, hw = _frame(p, x, y)
                if along < hl - 0.05 and hw - r + 0.005 < lat <= hw + r + CLEARANCE:
                    hits.append((f'{p.component_ref}.{p.pad_number}',
                                 round(x, 2), round(y, 2)))
                    break
    return sorted(set(hits))


def off_axis_exits(pads, segs):
    """Wired pads whose copper does not leave along the axis: a segment over
    the pad that is not parallel to the long axis, or no axial run reaching
    track + clearance past an end (less one grid step: the turn is taken
    on the first cell past the flank)."""
    bad = []
    for p in pads:
        vertical = p.size_y >= p.size_x
        reach = 0.0
        for s in segs:
            if s.net_id != p.net_id:
                continue
            over = [_frame(p, x, y) for x, y in _points(s)]
            over = [f for f in over if f[0] <= f[2] and f[1] <= f[3]]
            if not over:
                continue
            parallel = (abs(s.end_x - s.start_x) < 1e-6 if vertical
                        else abs(s.end_y - s.start_y) < 1e-6)
            if not parallel:
                bad.append((f'{p.component_ref}.{p.pad_number}', 'oblique over the pad'))
                continue
            hl = over[0][2]
            reach = max(reach, max(_frame(p, x, y)[0] for x, y in
                                   ((s.start_x, s.start_y), (s.end_x, s.end_y))) - hl)
        if reach < TRACK + CLEARANCE - STEP:
            bad.append((f'{p.component_ref}.{p.pad_number}', f'axial run {reach:.2f} mm past the end'))
    return bad


def main():
    tmp = tempfile.mkdtemp(prefix='tpadentry_')
    try:
        inp = os.path.join(tmp, 'in.kicad_pcb')
        out = os.path.join(tmp, 'out.kicad_pcb')
        js = os.path.join(tmp, 'out.json')
        with open(inp, 'w', encoding='utf-8') as fh:
            fh.write(board())
        evidence(inp, 'synthetic input board')
        env = dict(os.environ, KICAD_PLANE_FINALIZE='0')
        run_check([sys.executable, '-X', 'utf8', tool('route.py'), inp, out,
                   '--nets', '*', '--json-out', js, '--grid-step', str(STEP),
                   '--track-width', str(TRACK), '--clearance', str(CLEARANCE),
                   '--via-size', '0.45', '--via-drill', '0.3',
                   '--fab-tier', 'standard', '--escalation', 'off'],
                  accept=True, env=env)
        evidence(out, 'routed output')
        with open(evidence(js, '--json-out'), encoding='utf-8') as fh:
            doc = json.load(fh)
        wired = sorted(net_name(ref, n, w) for ref, _o, _v, w in ROWS for n in w)
        routed = sorted(doc.get('routed_single') or [])
        check("every wired net routes", routed == wired, str(routed))
        check("the three-pad net connects every pad",
              doc.get('multipoint_pads_connected') == doc.get('multipoint_pads_total') == 3
              and not doc.get('failed_multipoint'),
              f"{doc.get('multipoint_pads_connected')}/{doc.get('multipoint_pads_total')}")
        pads, segs = load(out)
        hits = side_entries(pads, segs)
        check("no wired pad is entered from a long side",
              not hits, f"{len(hits)} side entr(ies) (pad, point): {hits}")
        bad = off_axis_exits(pads, segs)
        check("every wired pad is left along its axis", not bad, str(bad))
        drc = run_check([sys.executable, '-X', 'utf8', tool('check_drc.py'), out],
                        accept=True)
        check("check_drc.py is clean", drc.returncode == 0,
              " | ".join((drc.stdout + drc.stderr).strip().splitlines()[-3:]))
        con = run_check([sys.executable, '-X', 'utf8', tool('check_connected.py'), out],
                        accept=True)
        check("check_connected.py is clean", con.returncode == 0,
              " | ".join((con.stdout + con.stderr).strip().splitlines()[-3:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
