#!/usr/bin/env python3
"""A walled search that fails does not pay for its diagnostics.

route.py first tries a net's search on a map with its row-pad flanks
(pad_entry.py); when that fails, the whole unblock ladder runs again
without them. A failed search dumps why its ends are blocked, naming the
blocking nets -- a scan of the board's copper per dump. Computed on the
walled map, the dump was thrown away: the ladder repeats the search without
the flanks and reports on that map. On a 4-layer board with ~500 walled
failures, those dumps were 18 s of a 7-minute step, the branch's whole
slowdown there.

The board: a 0.5 mm pitch row of vertical 0.3 x 1.6 mm pads; pad 4 is wired
to a pad above the row, and a narrow pad of another net sits 0.3 mm past
each of its ends: too close for the track to leave along the axis for track
plus clearance before it turns, so the walled search fails, and the ladder
enters the pad from a side. One routing layer (a via in the pad would
otherwise be the way out).

Checks (route.py run in-process, the blocking-net scan counted):
  1. the wired net routes, from the pad's side (the console says so);
  2. no blocking-net scan runs while the net's flanks are on the map;
  3. check_drc.py and check_connected.py are clean.

    python3 tests/test_pad_entry_walled_diagnostics.py
"""
import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'rust_router'))

from run_utils import check as run_check, evidence, tool        # noqa: E402
from synth import RoutingBoard                                  # noqa: E402

failures = []
PITCH, PAD_W, PAD_L = 0.5, 0.3, 1.6
TRACK = CLEARANCE = 0.1
STEP = 0.05
ORIGIN = (20.019, 15.0)
WIRED = 4


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def board():
    b = RoutingBoard()
    ox, oy = ORIGIN
    pads = []
    for i in range(8):
        num, off = i + 1, round((i - 3.5) * PITCH, 3)
        pads.append((str(num), off, 0, PAD_W, PAD_L,
                     'SIG' if num == WIRED else f'U1_NC{num}'))
    b.part('U1', ox, oy, pads)
    x = ox + round((WIRED - 1 - 3.5) * PITCH, 3)
    b.pad('R1', x - 1.5, oy - 5.0, 'SIG', sx=0.8, sy=0.8)
    # A narrow pad of another net 0.3 mm past each end of the wired pad:
    # the gaps beside it stay open.
    for k, dy in enumerate((-1.3, 1.3)):
        b.pad(f'B{k}', x, oy + dy, f'BLOCK{k}', sx=0.16, sy=0.4)
    return b.text()


def walled_now():
    """True when some net's stamped flanks are on their map right now
    (stamped, not lifted)."""
    import pad_entry
    for obs, cells in list(pad_entry._ACTIVE.values()):
        sample = cells[:50]
        if len(sample) and sum(obs.is_blocked(int(x), int(y), int(l))
                               for x, y, l in sample) > len(sample) // 2:
            return True
    return False


def route_counting(argv):
    """Run route.py in-process with `argv`; (console, scans on the walled
    map, all scans)."""
    import single_ended_routing as ser
    real = ser._identify_blocking_obstacles
    seen = {'walled': 0, 'all': 0}

    def counted(*a, **k):
        seen['all'] += 1
        if walled_now():
            seen['walled'] += 1
        return real(*a, **k)

    ser._identify_blocking_obstacles = counted
    out = io.StringIO()
    old_argv = sys.argv
    sys.argv = [tool('route.py')] + argv
    try:
        with contextlib.redirect_stdout(out):
            try:
                runpy.run_path(tool('route.py'), run_name='__main__')
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise
    finally:
        sys.argv = old_argv
        ser._identify_blocking_obstacles = real
    return out.getvalue(), seen['walled'], seen['all']


def main():
    tmp = tempfile.mkdtemp(prefix='tpadwalldiag_')
    try:
        inp = os.path.join(tmp, 'in.kicad_pcb')
        out = os.path.join(tmp, 'out.kicad_pcb')
        js = os.path.join(tmp, 'out.json')
        with open(inp, 'w', encoding='utf-8') as fh:
            fh.write(board())
        evidence(inp, 'synthetic input board')
        os.environ['KICAD_PLANE_FINALIZE'] = '0'
        console, walled, scans = route_counting(
            [inp, out, '--nets', 'SIG', '--layers', 'F.Cu', '--json-out', js,
             '--grid-step', str(STEP),
             '--track-width', str(TRACK), '--clearance', str(CLEARANCE),
             '--via-size', '0.45', '--via-drill', '0.3',
             '--fab-tier', 'standard', '--escalation', 'off'])
        evidence(out, 'routed output')
        with open(evidence(js, '--json-out'), encoding='utf-8') as fh:
            doc = json.load(fh)
        check("the wired net routes", doc.get('routed_single') == ['SIG'],
              str(doc.get('routed_single')))
        check("the walled search failed and the ladder entered from a side",
              "no way in through a row pad's ends" in console)
        check("no blocking-net scan on the walled map", walled == 0,
              f"{walled} of {scans} scans ran with the flanks on the map")
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
