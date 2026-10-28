#!/usr/bin/env python3
"""Under --escalation off the diff-pair partner neck narrows nothing.

`_neck_pair_partner_grazes` (#318) necked a member's segment that grazes its
partner without asking `fab_tiers.may_narrow()`, while the terminal neck
turns the same graze into a refusal under --escalation off (#842). A pair
whose terminals sit closer than the coupled pitch (pads 0.32 mm apart, routed
at width 0.2 / gap 0.25 / clearance 0.15) shipped two attach segments at
0.1396 mm with the policy off, and the Design rules line reported them as
delivered below the requested size.

Now, with the policy off, the graze keeps its width, the hard gate reports it
and the caller refuses the candidate; the warning names the policy, not a
short.

Rows:
  1. unit: a 10 um graze under `fab` is necked and recorded (control); under
     `off` it keeps its width, is hard, and records nothing;
  2. unit: a collinear end-to-end graze stays hard under both policies;
  3. both fronts (CLI on a written board, the GUI's in-process
     `return_results` run) on the 0.32 mm pair, DRC-clean input:
     - `fab` (control): a pair-partner neck ships, so the fixture reaches it;
     - `off`: no pair track under 0.2 ships, no track_width row is reported,
       and the refusal says --escalation off.

    python3 tests/test_pair_neck_escalation_off.py
"""
import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for _d in ('py_router', 'rust_router', 'tests'):
    sys.path.insert(0, os.path.join(ROOT, _d))

import diff_pair_routing                                            # noqa: E402
import fab_tiers                                                    # noqa: E402
from kicad_parser import BoardInfo, parse_kicad_pcb                 # noqa: E402
from routing_config import GridRouteConfig                          # noqa: E402
from run_utils import check as run_check, evidence, tool            # noqa: E402
from synth import RoutingBoard, make_pcb, make_seg                  # noqa: E402

RUN_ALL_TIMEOUT = 300
P, N = 16, 17
W, GAP, CLR = 0.2, 0.25, 0.15
PAD, PITCH = 0.15, 0.32     # P/N pad gap 0.17 >= clearance: a clean input
LAYERS = ['F.Cu', 'B.Cu']
SITE = 'pair partner neck'
FAILS = []


def check(name, ok, detail=''):
    if not ok:
        FAILS.append(name)
    print(('  PASS ' if ok else '  FAIL ') + name + (f'  {detail}' if detail else ''))


PCB = make_pcb(board_info=BoardInfo(layers={}, copper_layers=list(LAYERS),
                                    board_bounds=(0, 0, 200, 200)))


def neck(policy, p_segs, n_segs):
    fab_tiers.set_escalation_policy(policy)
    cfg = GridRouteConfig(layers=list(LAYERS), clearance=CLR, track_width=W,
                          diff_pair_gap=CLR)
    necked, hard = diff_pair_routing._neck_pair_partner_grazes(p_segs, n_segs, cfg, PCB)
    rows = [r for r in fab_tiers.escalation_summary()['narrowed'] if r['site'] == SITE]
    return necked, hard, rows


def graze():
    pitch = W + CLR - 0.01                   # edge gap 0.14 under a 0.15 rule
    return ([make_seg(30.0, 0.0, 32.0, 0.0, net_id=P)],
            [make_seg(30.0, pitch, 32.0, pitch, net_id=N)])


print('1. a graze a neck could fix')
p, n = graze()
necked, hard, rows = neck('fab', p, n)
check('fab (control): necked, not hard, recorded',
      necked == 1 and not hard and p[0].width < W and len(rows) == 1,
      f'necked={necked} hard={len(hard)} P {p[0].width} rows={len(rows)}')
p, n = graze()
necked, hard, rows = neck('off', p, n)
check('off: nothing is necked', necked == 0 and p[0].width == W and n[0].width == W,
      f'necked={necked} P {p[0].width} N {n[0].width}')
check('off: the graze is hard, so the caller refuses it',
      len(hard) == 1 and 0 < hard[0][2] < CLR, f'{[(round(h[2], 4)) for h in hard]}')
check('off: nothing is recorded', not rows, f'{rows}')

print('2. a graze no neck can clear')
for policy in ('fab', 'off'):
    p = [make_seg(40.0, 0.0, 41.0, 0.0, net_id=P)]
    n = [make_seg(41.2, 0.0, 43.0, 0.0, net_id=N)]   # 0.2 end to end: no width clears 0.15
    necked, hard, rows = neck(policy, p, n)
    check(f'{policy}: collinear members stay hard', bool(hard) and not rows,
          f'{len(hard)} hard, rows={len(rows)}')
fab_tiers.set_escalation_policy(None)


def board(path):
    b = RoutingBoard(w=20, h=20)
    for ref, y in (('U1', 3.0), ('J1', 13.0)):
        b.pad(ref + 'P', 6.0, y, '/D_P', sx=PAD, sy=PAD)
        b.pad(ref + 'N', 6.0 + PITCH, y, '/D_N', sx=PAD, sy=PAD)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(b.text())


def narrow(segments, names):
    return sorted(round(s.width, 4) for s in segments
                  if names.get(s.net_id, '').startswith('/D_') and s.width < W - 1e-3)


def track_rows(summary):
    return [r for r in ((summary or {}).get('design_rules') or {}).get('narrowed', [])
            if r['kind'] == 'track_width']


def gui_front(src, policy):
    """The Differential tab's call: in-process, return_results, live board."""
    from route_diff import batch_route_diff_pairs
    pcb = parse_kicad_pcb(src)
    sites = []
    real = fab_tiers.note_narrowing

    def spy(net_id, kind, requested, delivered, site, *a, **k):
        sites.append(site)
        return real(net_id, kind, requested, delivered, site, *a, **k)
    fab_tiers.note_narrowing = spy
    fab_tiers.set_escalation_policy(policy)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            batch_route_diff_pairs(src, '', ['/D_*'], return_results=True,
                                   pcb_data=pcb, layers=list(LAYERS), clearance=CLR,
                                   track_width=W, diff_pair_gap=GAP, via_size=0.5,
                                   via_drill=0.3, grid_step=0.05)
    finally:
        fab_tiers.note_narrowing = real
        fab_tiers.set_escalation_policy(None)
    out = buf.getvalue()
    m = re.findall(r'JSON_SUMMARY: (\{.*\})', out)
    names = {i: nn.name for i, nn in pcb.nets.items()}
    return (json.loads(m[-1]) if m else None), out, narrow(pcb.segments, names), sites


def cli_front(src, out, policy):
    r = run_check([sys.executable, '-X', 'utf8', tool('route_diff.py'), src, out,
                   '--nets', '/D_*', '--track-width', str(W), '--diff-pair-gap', str(GAP),
                   '--clearance', str(CLR), '--via-size', '0.5', '--via-drill', '0.3',
                   '--grid-step', '0.05', '--layers', *LAYERS, '--escalation', policy],
                  accept=True)
    text = r.stdout + (r.stderr or '')
    m = re.findall(r'JSON_SUMMARY: (\{.*\})', text)
    pcb = parse_kicad_pcb(evidence(out))
    names = {i: nn.name for i, nn in pcb.nets.items()}
    return (json.loads(m[-1]) if m else None), text, narrow(pcb.segments, names)


print('3. route_diff on a pair whose pads sit closer than the coupled pitch')
td = tempfile.mkdtemp(prefix='t_pnoff_')
try:
    src = os.path.join(td, 'pair.kicad_pcb')
    board(src)

    summary, out, shipped, sites = gui_front(src, 'fab')
    check('GUI fab (control): a pair-partner neck ships',
          SITE in sites and bool(shipped), f'sites {sorted(set(sites))}, narrow {shipped}')
    summary, out, shipped, sites = gui_front(src, 'off')
    check('GUI off: the run prints its JSON_SUMMARY', summary is not None)
    check('GUI off: no pair track under the request ships', not shipped, f'{shipped}')
    check('GUI off: no track_width row', not track_rows(summary), f'{track_rows(summary)}')
    check('GUI off: the refusal names the policy',
          '--escalation off forbids necking it -- rejecting the pair' in out)

    summary, text, shipped = cli_front(src, os.path.join(td, 'fab.kicad_pcb'), 'fab')
    check('CLI fab (control): a narrowed pair track is written', bool(shipped), f'{shipped}')
    summary, text, shipped = cli_front(src, os.path.join(td, 'off.kicad_pcb'), 'off')
    check('CLI off: the run prints its JSON_SUMMARY', summary is not None)
    check('CLI off: no pair track under the request is written', not shipped, f'{shipped}')
    check('CLI off: no track_width row', not track_rows(summary), f'{track_rows(summary)}')
    line = next((ln for ln in text.splitlines() if 'Design rules [' in ln), '')
    check('CLI off: no Design rules line claims a narrow track', 'track width' not in line,
          line.strip() or '(no line)')
finally:
    shutil.rmtree(td, ignore_errors=True)

print()
if FAILS:
    print(f'{len(FAILS)} FAILURE(S): {", ".join(FAILS)}')
    sys.exit(1)
print('ALL PASS')
