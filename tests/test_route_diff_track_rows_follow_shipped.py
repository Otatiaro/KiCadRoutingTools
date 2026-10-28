#!/usr/bin/env python3
"""route_diff's design_rules track rows describe the tracks that SHIP.

The pair partner neck (#318) and the terminal neck record a `track_width` row
per routing ATTEMPT. A candidate the ladder then throws away -- a connector
grazing a foreign pad (#165), P/N crossing in the final geometry, a hybrid leg
that fails and rips the committed ones -- kept its row, so a run whose written
copper was all at the requested 0.2 mm reported "4 feature(s) ... delivered
below the requested size (smallest track width 0.0889 mm)" under
--escalation off. route_diff now replaces every track_width row with a census
of the copper it ships (`routing_common.new_copper_width_shortfall`,
`fab_tiers.replace_track_rows`), as it already did for vias (#1371).

Unit half: the census and the row replacement on stand-in segments.
End to end, both fronts (the CLI on a written board, the GUI's in-process
`return_results` run on the parsed board), two arms:
  PHANTOM -- P/N pads 0.32 mm apart at both ends, routed at 0.2 / gap 0.25,
             a foreign pad just below the source pair: the attempts neck
             (note_narrowing is called) and are rejected, every shipped track
             is 0.2 (the parent reported 24 features at 0.127 mm), and no
             track row is reported;
  SHIPPED -- pads 0.4 mm apart and a foreign pad beside P's terminal: a
             necked track ships, and the rows are exactly that copper.
The input boards are DRC-clean (pad gaps at or above the clearance).

    python3 tests/test_route_diff_track_rows_follow_shipped.py
"""
import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for _d in ('py_router', 'rust_router', 'tests'):
    sys.path.insert(0, os.path.join(ROOT, _d))

import fab_tiers                                                    # noqa: E402
from kicad_parser import parse_kicad_pcb                            # noqa: E402
from routing_common import new_copper_width_shortfall, segment_identity  # noqa: E402
from run_utils import check as run_check, evidence, tool           # noqa: E402
from synth import RoutingBoard                                      # noqa: E402

RUN_ALL_TIMEOUT = 300
W, GAP, CLR = 0.2, 0.25, 0.15
PAD = 0.15          # P/N pad gap at a 0.3 pitch is the clearance: a clean input
ARGS = dict(layers=['F.Cu', 'B.Cu'], clearance=CLR, track_width=W,
            diff_pair_gap=GAP, via_size=0.5, via_drill=0.3, grid_step=0.05)
FAILS = []


def check(name, ok, detail=''):
    if not ok:
        FAILS.append(name)
    print(('  PASS ' if ok else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def unit_half():
    print('unit: census and row replacement')
    s = lambda net, x, w: SimpleNamespace(net_id=net, layer='F.Cu', width=w,  # noqa: E731
                                          start_x=x, start_y=0.0, end_x=x + 1.0,
                                          end_y=0.0)
    names = {1: 'A', 2: 'B'}
    old = s(1, 0.0, 0.1)
    inp = {segment_identity(old, 'A')}
    segs = [old,                 # on the input board: not this run's
            s(1, 2.0, 0.2),      # new, at the request
            s(2, 4.0, 0.15)]     # new, under it
    got = new_copper_width_shortfall(segs, names.get, inp, lambda n, l: W)
    check('input copper is not this run\'s', got['A']['under_mm'] == 0, f'{got}')
    check('new narrow copper is counted', got['B']['min_mm'] == 0.15
          and got['B']['under_mm'] == 1.0, f'{got}')
    fab_tiers.reset_ledger()
    for _ in range(3):
        fab_tiers.note_narrowing(1, 'track_width', W, 0.0889, 'pair partner neck')
    fab_tiers.note_narrowing(2, 'track_width', W, 0.127, 'terminal neck')
    fab_tiers.note_narrowing(1, 'via_diameter', 0.45, 0.3, 'via-in-pad clamp')
    fab_tiers.replace_track_rows(
        {1, 2}, [(n, names[n], r['requested_mm'], r['min_mm'], r['under_mm'])
                 for n, r in ((1, got['A']), (2, got['B']))],
        'track copper shipped under width')
    rows = fab_tiers.escalation_summary()['narrowed']
    tw = [(r['net'], r['delivered']) for r in rows if r['kind'] == 'track_width']
    check('per-attempt rows are replaced by the shipped copper', tw == [(2, 0.15)],
          f'{tw}')
    check('other kinds are kept', any(r['kind'] == 'via_diameter' for r in rows))
    fab_tiers.reset_ledger()


def board(path, pitch, foreign=None):
    b = RoutingBoard(w=20, h=20)
    for ref, y in (('U1', 3.0), ('J1', 13.0)):
        b.pad(ref + 'P', 6.0, y, '/D_P', sx=PAD, sy=PAD)
        b.pad(ref + 'N', 6.0 + pitch, y, '/D_N', sx=PAD, sy=PAD)
    if foreign:
        b.pad('R1', foreign[0], foreign[1], 'X', sx=0.3, sy=0.3)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(b.text())


def track_rows(summary):
    return sorted((r['net_name'], r['delivered'])
                  for r in ((summary or {}).get('design_rules') or {}).get('narrowed', [])
                  if r['kind'] == 'track_width')


def shipped_narrow(segments, names):
    """(net name, width) of every pair track under the requested width."""
    return sorted({(names[s.net_id], round(s.width, 4)) for s in segments
                   if names.get(s.net_id, '').startswith('/D_')
                   and s.width < W - 1e-3})


def pair_widths(segments, names):
    return sorted({round(s.width, 4) for s in segments
                   if names.get(s.net_id, '').startswith('/D_')})


def gui_front(src):
    """The Differential tab's call: in-process, return_results, live board."""
    from route_diff import batch_route_diff_pairs
    pcb = parse_kicad_pcb(src)
    calls = []
    real = fab_tiers.note_narrowing

    def spy(net_id, kind, requested, delivered, site, *a, **k):
        if kind == 'track_width' and delivered < requested - 1e-9:
            calls.append(site)
        return real(net_id, kind, requested, delivered, site, *a, **k)
    fab_tiers.note_narrowing = spy
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            batch_route_diff_pairs(src, '', ['/D_*'], return_results=True,
                                   pcb_data=pcb, **ARGS)
            fab_tiers.finish_design_rules(None, 0)
    finally:
        fab_tiers.note_narrowing = real
    out = buf.getvalue()
    m = re.findall(r'JSON_SUMMARY: (\{.*\})', out)
    names = {n: nn.name for n, nn in pcb.nets.items()}
    return (json.loads(m[-1]) if m else None), out, pcb.segments, names, calls


def cli_front(src, out):
    r = run_check([sys.executable, '-X', 'utf8', tool('route_diff.py'), src, out,
                   '--nets', '/D_*', '--track-width', str(W), '--diff-pair-gap',
                   str(GAP), '--clearance', str(CLR), '--via-size', '0.5',
                   '--via-drill', '0.3', '--grid-step', '0.05', '--layers',
                   'F.Cu', 'B.Cu'], accept=True)
    text = r.stdout + (r.stderr or '')
    m = re.findall(r'JSON_SUMMARY: (\{.*\})', text)
    pcb = parse_kicad_pcb(evidence(out))
    names = {n: nn.name for n, nn in pcb.nets.items()}
    return (json.loads(m[-1]) if m else None), text, pcb.segments, names


def arm(td, label, pitch, foreign):
    src = os.path.join(td, f'{label}.kicad_pcb')
    board(src, pitch, foreign)

    print(f'{label} (GUI front)')
    summary, out, segs, names, calls = gui_front(src)
    check('the run prints its JSON_SUMMARY', summary is not None)
    ship = shipped_narrow(segs, names)
    if label == 'PHANTOM':
        check('fixture: the attempts necked (note_narrowing called)', bool(calls),
              f'{len(calls)} call(s): {sorted(set(calls))}')
        check('fixture: every shipped pair track is at the request',
              bool(segs) and pair_widths(segs, names) == [W],
              f'widths {pair_widths(segs, names)}')
    else:
        check('fixture: a narrowed pair track ships', bool(ship), f'{ship}')
    check('track rows are the shipped narrow copper', track_rows(summary) == ship,
          f'rows {track_rows(summary)} shipped {ship}')
    line = next((ln for ln in out.splitlines() if 'Design rules [' in ln), '')
    check('the Design rules line follows the rows',
          ('track width' in line) == bool(ship), line.strip() or '(no line)')

    print(f'{label} (CLI front)')
    summary, text, segs, names = cli_front(src, os.path.join(td, f'{label}_o.kicad_pcb'))
    check('the run prints its JSON_SUMMARY', summary is not None)
    ship = shipped_narrow(segs, names)
    if label == 'PHANTOM':
        check('fixture: every written pair track is at the request',
              bool(segs) and pair_widths(segs, names) == [W],
              f'widths {pair_widths(segs, names)}')
    else:
        check('fixture: a narrowed pair track is written', bool(ship), f'{ship}')
    check('track rows are the written narrow copper', track_rows(summary) == ship,
          f'rows {track_rows(summary)} written {ship}')
    line = next((ln for ln in text.splitlines() if 'Design rules [' in ln), '')
    check('the Design rules line follows the rows',
          ('track width' in line) == bool(ship), line.strip() or '(no line)')


if __name__ == '__main__':
    unit_half()
    td = tempfile.mkdtemp(prefix='t_rdtw_')
    try:
        arm(td, 'PHANTOM', 0.32, (6.15, 3.45))
        arm(td, 'SHIPPED', 0.4, (5.6, 3.3))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILURE(S): {", ".join(FAILS)}')
        sys.exit(1)
    print('ALL PASS')
