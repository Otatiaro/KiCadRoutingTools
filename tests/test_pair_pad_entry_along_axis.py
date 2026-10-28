#!/usr/bin/env python3
"""A differential pair leaves two neighbouring row pins through their ENDS.

A pair on two neighbouring pins of a fine-pitch row (SOIC 1.27 mm, a 0.5 mm
QFN/DFN row) was joined to its pads in two ways that both enter a pin from
its flank:

  - the DIRECT HYBRID escape launched its coupled middle at the closest
    clear cell to the terminal, which is the gap BETWEEN the pair's two pads
    (their copper is free on the diff-pair map), and its single-ended legs
    then reached each pad centre from the side, a 45-degree jog across the
    pad's long edge. A leg that did leave through the end dropped its via
    just past the end, and the soft-joint pass then "firmed" that via with a
    straight bridge to the pad centre, across the flank;
  - the coupled route's connectors left the pad end at 60 degrees and more
    toward the partner, their copper sweeping the pad's corner.

DRC-clean and connected either way, but a pin is landed through its end,
along its axis, and the pair joins beyond the pads.

Boards (0.2 mm track, 0.25 mm gap, 0.09 mm clearance, 0.05 mm grid, vias
kept off same-net pads):
  soic          SOIC-8 pair on pins 6/5 to a far pad pair: the coupled route;
  chain         the same pair through a second SOIC (pins 3/2), multipoint;
  soic_crowded  pins 6/5 behind a foreign track across the pin ends, with an
                exposed pad: one track fits in the channel, a pair does not,
                so the hybrid escape is taken;
  row_end       a 0.5 mm row, pair on its last two pins (7/8), a foreign
                track across the ends: hybrid;
  row_mid       the same row, pair on pins 4/5: hybrid.

Checks, per board:
  1. the pair routes (JSON_SUMMARY routed_diff_pairs), and the hybrid boards
     really take the hybrid escape, with every leg through a pad end (no
     "entering from a pad's side" fallback);
  2. no pair pad has its own net's copper over a long side, along its length;
  3. every pair pad is left along its axis: the copper over it runs within
     5 degrees of the axis (an off-grid pad centre tilts the first segment
     by a grid fraction) and reaches one track width plus clearance past the
     end, less a grid step, before it turns;
  4. check_drc.py and check_connected.py are clean.

    python3 tests/test_pair_pad_entry_along_axis.py
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
TRACK, GAP, CLEARANCE, STEP = 0.2, 0.25, 0.09, 0.05


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def soic(b, ref, x, y, wired, exposed_pad=None):
    """SOIC-8 at (x, y): 1.95 x 0.6 mm pads, 1.27 mm pitch, rows at +-2.7."""
    pads = []
    for i in range(4):
        dy = round((i - 1.5) * 1.27, 3)
        pads.append((str(i + 1), -2.7, dy, 1.95, 0.6, wired.get(i + 1, f'{ref}_NC{i + 1}')))
        pads.append((str(8 - i), 2.7, dy, 1.95, 0.6, wired.get(8 - i, f'{ref}_NC{8 - i}')))
    if exposed_pad:
        pads.append(('9', 0, 0, 2.4, 3.1, exposed_pad))
    b.part(ref, x, y, pads)


def row(b, ref, x, y, wired, n=8, pitch=0.5):
    """Two facing rows of n 0.3 x 0.55 mm pads (a 0.5 mm DFN), 0.77 mm apart."""
    pads = []
    for i in range(n):
        dx = round((i - (n - 1) / 2) * pitch, 3)
        pads.append((str(i + 1), dx, 0.385, 0.3, 0.55, wired.get(i + 1, f'{ref}_NC{i + 1}')))
        pads.append((str(2 * n - i), dx, -0.385, 0.3, 0.55,
                     wired.get(2 * n - i, f'{ref}_NC{2 * n - i}')))
    b.part(ref, x, y, pads)


def foreign_track(b, x1, y1, x2, y2):
    b.seg(x1, y1, x2, y2, 'X', w=TRACK)
    b.pad('TX1', x1, y1, 'X', sx=0.5, sy=0.5)
    b.pad('TX2', x2, y2, 'X', sx=0.5, sy=0.5)


def board(name):
    """(board text, pair pads as (ref, pad number), hybrid expected)."""
    b = RoutingBoard(w=40, h=30)
    if name == 'soic':
        soic(b, 'U1', 12.013, 15.019, {6: 'A_P', 5: 'A_N'})
        b.pad('J1', 30.0, 6.0, 'A_P', sx=0.8, sy=0.8)
        b.pad('J2', 31.0, 6.0, 'A_N', sx=0.8, sy=0.8)
        return b.text(), [('U1', '6'), ('U1', '5')], False
    if name == 'chain':
        soic(b, 'U1', 12.013, 15.019, {6: 'A_P', 5: 'A_N'})
        soic(b, 'U3', 24.013, 15.019, {3: 'A_P', 2: 'A_N'})
        b.pad('J1', 6.0, 4.0, 'A_P', sx=0.8, sy=0.8)
        b.pad('J2', 7.0, 4.0, 'A_N', sx=0.8, sy=0.8)
        return b.text(), [('U1', '6'), ('U1', '5'), ('U3', '3'), ('U3', '2')], False
    if name == 'soic_crowded':
        soic(b, 'U1', 12.013, 15.019, {6: 'A_P', 5: 'A_N'}, exposed_pad='GND')
        foreign_track(b, 16.35, 11.5, 16.35, 18.6)
        b.pad('J1', 30.0, 22.0, 'A_P', sx=0.8, sy=0.8)
        b.pad('J2', 31.0, 22.0, 'A_N', sx=0.8, sy=0.8)
        return b.text(), [('U1', '6'), ('U1', '5')], True
    k = {'row_end': 7, 'row_mid': 4}[name]
    row(b, 'U2', 14.013, 15.019, {k: 'A_P', k + 1: 'A_N'})
    foreign_track(b, 10.0, 16.3, 18.0, 16.3)
    b.pad('J1', 24.0, 25.0, 'A_P', sx=0.8, sy=0.8)
    b.pad('J2', 25.0, 25.0, 'A_N', sx=0.8, sy=0.8)
    return b.text(), [('U2', str(k)), ('U2', str(k + 1))], True


def _frame(p, x, y):
    """(|along|, |lat|, half-length, half-width) of (x, y) in pad p's frame."""
    dx, dy = x - p.global_x, y - p.global_y
    if p.size_y >= p.size_x:
        return abs(dy), abs(dx), p.size_y / 2, p.size_x / 2
    return abs(dx), abs(dy), p.size_x / 2, p.size_y / 2


def _points(s, step=0.01):
    n = max(1, int(math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) / step))
    for i in range(n + 1):
        yield (s.start_x + (s.end_x - s.start_x) * i / n,
               s.start_y + (s.end_y - s.start_y) * i / n)


def _own(p, segs):
    return [s for s in segs if s.net_id == p.net_id and s.layer in p.layers]


def side_entries(pads, segs):
    """(pad, point) where a pair pad's own net has copper over the pad's long
    side, along its length (5 um of slack for the grid's half-step)."""
    hits = []
    for p in pads:
        for s in _own(p, segs):
            r = s.width / 2
            for x, y in _points(s):
                along, lat, hl, hw = _frame(p, x, y)
                if along < hl - 0.05 and hw - r + 0.005 < lat <= hw + r + CLEARANCE:
                    hits.append((f'{p.component_ref}.{p.pad_number}', round(x, 2), round(y, 2)))
                    break
    return hits


def off_axis_exits(pads, segs):
    """Pair pads whose copper does not leave along the axis."""
    bad = []
    tilt = math.sin(math.radians(5))
    for p in pads:
        vertical = p.size_y >= p.size_x
        reach = None
        for s in _own(p, segs):
            over = [f for f in (_frame(p, x, y) for x, y in _points(s))
                    if f[0] <= f[2] and f[1] <= f[3]]
            if not over:
                continue
            length = math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) or 1.0
            across = abs(s.end_x - s.start_x) if vertical else abs(s.end_y - s.start_y)
            if across > length * tilt:
                bad.append((f'{p.component_ref}.{p.pad_number}', 'oblique over the pad'))
                continue
            past = max(_frame(p, x, y)[0] for x, y in
                       ((s.start_x, s.start_y), (s.end_x, s.end_y))) - over[0][2]
            reach = past if reach is None else max(reach, past)
        if reach is not None and reach < TRACK + CLEARANCE - STEP:
            bad.append((f'{p.component_ref}.{p.pad_number}',
                        f'axial run {reach:.2f} mm past the end'))
    return bad


def run(name, tmp):
    print(f"\n-- {name} --")
    text, wanted, hybrid = board(name)
    inp = os.path.join(tmp, f'{name}.kicad_pcb')
    out = os.path.join(tmp, f'{name}_out.kicad_pcb')
    js = os.path.join(tmp, f'{name}_out.json')
    with open(inp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    evidence(inp, 'synthetic input board')
    r = run_check([sys.executable, '-X', 'utf8', tool('route_diff.py'), inp, out,
                   '--nets', 'A_P', 'A_N', '--json-out', js,
                   '--track-width', str(TRACK), '--diff-pair-gap', str(GAP),
                   '--clearance', str(CLEARANCE), '--grid-step', str(STEP),
                   '--via-size', '0.45', '--via-drill', '0.3',
                   '--same-net-pad-clearance', '0.09', '--no-gnd-vias',
                   '--fab-tier', 'standard', '--escalation', 'off'], accept=True)
    txt = r.stdout + r.stderr
    evidence(out, 'routed output')
    with open(evidence(js, '--json-out'), encoding='utf-8') as fh:
        doc = json.load(fh)
    check(f"{name}: the pair routes", doc.get('routed_diff_pairs') == ['A'],
          str(doc.get('routed_diff_pairs')))
    took = 'HYBRID ESCAPE' in txt
    check(f"{name}: {'the hybrid escape' if hybrid else 'the coupled route'} is taken",
          took == hybrid, f"HYBRID ESCAPE in the log: {took}")
    check(f"{name}: no leg falls back to a pad's side", "from a pad's side" not in txt)
    from kicad_parser import parse_kicad_pcb
    with contextlib.redirect_stdout(io.StringIO()):
        pcb = parse_kicad_pcb(out)
    pads = [p for ps in pcb.pads_by_net.values() for p in ps
            if (p.component_ref, p.pad_number) in wanted]
    check(f"{name}: every pair pad found", len(pads) == len(wanted), str(len(pads)))
    hits = side_entries(pads, pcb.segments)
    check(f"{name}: no pair pad is entered from a long side", not hits,
          f"{len(hits)} side entr(ies) (pad, point): {hits}")
    bad = off_axis_exits(pads, pcb.segments)
    check(f"{name}: every pair pad is left along its axis", not bad, str(bad))
    for script in ('check_drc.py', 'check_connected.py'):
        c = run_check([sys.executable, '-X', 'utf8', tool(script), out], accept=True)
        check(f"{name}: {script} is clean", c.returncode == 0,
              " | ".join((c.stdout + c.stderr).strip().splitlines()[-3:]))


def main():
    tmp = tempfile.mkdtemp(prefix='tpairentry_')
    try:
        for name in ('soic', 'chain', 'soic_crowded', 'row_end', 'row_mid'):
            run(name, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('FAILED: ' + ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
