#!/usr/bin/env python3
"""A plane tap lands on a fine-pitch row pad through its END, along its axis.

repair_planes taps an SMD plane pad with a via and a short trace, or with a
trace to a nearby same-net via or pad. Its trace map left the pad's own
flanks open, so on a 0.5 mm row the trace ran down the gap between two pins
of the package -- or straight across it, from one GND pin to its GND
neighbour -- and entered the pad from its side. The tap now blocks the
flanks of the net's row pads in its trace map (pad_entry) and places no via
on a flank.

The board: two rows of eight 0.3 x 1.6 mm pads at 0.5 mm (one vertical, one
horizontal, both off the 0.05 mm grid), pads 1, 4, 5 and 8 of each on GND,
the others on single-pad nets. route_planes pours GND on B.Cu, then
repair_planes --repair-pads taps the pads, vias kept off the pads.

Checks:
  1. no GND pad has its own net's copper over a long side, along its length;
  2. no via sits beside a GND pad's long side;
  3. check_drc.py and check_connected.py are clean.

    python3 tests/test_pad_entry_plane_tap.py
"""
import contextlib
import io
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
import test_pad_entry_along_axis as along                       # noqa: E402

failures = []
GND = (1, 4, 5, 8)
ROWS = [('U1', (20.019, 15.0), True), ('U2', (30.0, 15.019), False)]
GEOMETRY = ['--grid-step', '0.05', '--clearance', '0.1', '--via-size', '0.45',
            '--via-drill', '0.3', '--fab-tier', 'standard', '--escalation', 'off',
            '--same-net-pad-clearance', '0.1']


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def board():
    b = RoutingBoard()
    for ref, (ox, oy), vertical in ROWS:
        pads = []
        for i in range(8):
            num, off = i + 1, round((i - 3.5) * along.PITCH, 3)
            dx, dy, sx, sy = ((off, 0, along.PAD_W, along.PAD_L) if vertical
                              else (0, off, along.PAD_L, along.PAD_W))
            pads.append((str(num), dx, dy, sx, sy,
                         'GND' if num in GND else f'{ref}_N{num}'))
        b.part(ref, ox, oy, pads)
    return b.text()


def vias_beside(path, pads):
    """(pad, via) of every via whose copper reaches over a GND pad's long
    side, along its length."""
    from kicad_parser import parse_kicad_pcb
    with contextlib.redirect_stdout(io.StringIO()):
        pcb = parse_kicad_pcb(path)
    out = []
    for v in pcb.vias:
        for p in pads:
            a, lat, hl, hw = along._frame(p, v.x, v.y)
            if a < hl and hw < lat < hw + v.size / 2 + along.CLEARANCE:
                out.append((f'{p.component_ref}.{p.pad_number}', v.x, v.y))
    return out


def main():
    tmp = tempfile.mkdtemp(prefix='tpadtap_')
    try:
        inp = os.path.join(tmp, 'in.kicad_pcb')
        planes = os.path.join(tmp, 'planes.kicad_pcb')
        out = os.path.join(tmp, 'out.kicad_pcb')
        with open(inp, 'w', encoding='utf-8') as fh:
            fh.write(board())
        evidence(inp, 'synthetic input board')
        run_check([sys.executable, '-X', 'utf8', tool('route_planes.py'), inp, planes,
                   '--nets', 'GND', '--plane-layers', 'B.Cu', '--track-width', '0.1']
                  + GEOMETRY, accept=True)
        evidence(planes, 'poured board')
        run_check([sys.executable, '-X', 'utf8', tool('repair_planes.py'), planes, out,
                   '--nets', 'GND', '--layers', 'F.Cu', 'B.Cu',
                   '--min-track-width', '0.1', '--max-track-width', '0.25',
                   '--track-via-clearance', '0.1', '--zone-clearance', '0.1']
                  + GEOMETRY, accept=True)
        evidence(out, 'tapped board')
        along.ROWS = [(ref, None, None, {n: None for n in GND}) for ref, _o, _v in ROWS]
        pads, segs = along.load(out)
        check("every GND pad of the rows is found", len(pads) == 2 * len(GND), str(len(pads)))
        hits = along.side_entries(pads, segs)
        check("no GND pad is entered from a long side",
              not hits, f"{len(hits)} side entr(ies) (pad, point): {hits}")
        beside = vias_beside(out, pads)
        check("no via sits beside a GND pad", not beside, str(beside))
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
