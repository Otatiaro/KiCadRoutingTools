#!/usr/bin/env python3
"""Swap declared equivalent pins and gates to untangle a placed board's ratsnest.

    python3 -X utf8 py_placer/swap_pins.py placed.kicad_pcb swapped.kicad_pcb \\
        [--brief board.design-brief.json] [--json-out swaps.json] \\
        [--schematic-dir DIR] [--dry-run]

Reads the design brief's `pin_swaps[]` (the brief beside the board, or
`--brief`): which gates of a part, or of a pool of identical parts, may trade
nets -- the channels of an ESD array, the receivers of a quad line receiver,
the halves of a dual op amp. On the placed board it picks the assignment of
nets to those gates that lowers place_simple's own signal cost (the Manhattan
MST of every signal net plus `--crossing-weight` x the crossings among the
trees; `placement/pin_swap.py`), rewrites the nets of the pads concerned and
nothing else, and checks the written board against the plan before it keeps
it. Run it between placement and routing.

Never touched: an undeclared pin; a gate with copper ending on one of its
pads; a net the board protects (`protected_nets`), carries locked or held
copper, or promised to a pour; a differential pair the row cannot keep on two
of its declared `adjacent` gates. A gate keeps its pin order, so no swap
inverts a polarity.

`--schematic-dir` writes the swaps into the schematic as pin renumberings
(`schematic_updater.py`); a part whose new nets came from another part of a
pool cannot be written that way and is named. Zones are not refilled: refill
before judging a pour.

Exit codes: 0 done (with or without a swap), 2 unreadable input, nothing
declared, or a declaration the board contradicts, 3 the written board did not
match the plan (nothing kept).
"""

#: #937 registry: which door(s) show this tool, and whether it changes
#: the board. Read by krt_registry.py -- by AST, never imported.
KRT_TOOL = {'scope': ['placement', 'routing', 'combined'], 'kind': 'actor'}

import _path  # noqa: F401  (py_placer -> py_router/py_tools on sys.path)

import argparse
import json
import os
import sys


def run(board, out, *, brief=None, crossing_weight=None, json_out=None,
        schematic_dir=None, dry_run=False, log=print):
    """The whole step; returns `(exit code, report)`."""
    from kicad_parser import parse_kicad_pcb
    from placement import pin_swap as ps

    try:
        rows, brief_path = ps.load_rows(board, brief)
    except (OSError, ValueError) as exc:
        log(f"ERROR: {exc}")
        return 2, None
    if not rows:
        log(f"ERROR: nothing declared -- {brief_path or 'no design brief beside the board'}"
            f" has no pin_swaps[] rows. This tool swaps only what a design "
            f"brief declares equivalent.")
        return 2, None
    pcb = parse_kicad_pcb(board)
    try:
        pools = ps.resolve(rows, pcb)
    except ps.PinSwapError as exc:
        log(f"ERROR: {os.path.basename(brief_path)}: {exc}")
        return 2, None

    from protected_nets import read_for_pcb_data, read_pour_served_for_pcb_data
    from list_nets import find_differential_pairs
    protected = read_for_pcb_data(pcb, board) or {}
    served = read_pour_served_for_pcb_data(pcb, board) or {}
    pairs = find_differential_pairs(pcb)
    frozen = ps.frozen_slots(pools, pcb, protected, served, pairs)
    res = ps.optimise(pcb, pools, crossing_weight=(
        ps.CROSSING_WEIGHT if crossing_weight is None else crossing_weight),
        frozen=frozen, diff_pairs=pairs)

    name = {nid: n.name for nid, n in pcb.nets.items()}
    changes = [{'ref': r, 'pin': p, 'old_net': name.get(o, '') if o else '',
                'new_net': name.get(n, '') if n else ''}
               for r, p, o, n in res['changes']]
    report = {'schema': 1, 'board': board, 'out': None if dry_run else out,
              'brief': brief_path, 'before': res['before'], 'after': res['after'],
              'crossing_weight': res['crossing_weight'], 'start': res['start'],
              'pools': res['pools'], 'changes': changes}

    log(f"design brief {os.path.basename(brief_path)}: {len(pools)} pin-swap pool(s)")
    for pool in res['pools']:
        log(f"  {pool['name']} ({', '.join(pool['refs'])}): {pool['free_gates']} of "
            f"{len(pool['refs']) * len(pool['gates'])} gate(s) free, "
            f"{pool['pins_changed']} pin(s) changed")
        for f in pool['frozen']:
            log(f"    kept {f['ref']} gate {'/'.join(f['gate'])}: {f['reason']}")
    b, a = res['before'], res['after']
    log(f"signal trees {b['tree_mm']} -> {a['tree_mm']} mm, crossings "
        f"{b['crossings']} -> {a['crossings']}, cost {b['cost']} -> {a['cost']}")
    for c in changes:
        log(f"  {c['ref']}.{c['pin']}: {c['old_net'] or '(no net)'} -> "
            f"{c['new_net'] or '(no net)'}")

    code = 0
    if not dry_run:
        with open(board, encoding='utf-8', newline='') as fh:
            content = fh.read()
        text = ps.apply(content, pcb, pools, res['changes'])
        tmp = out + '.swap_pins.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as fh:
            fh.write(text)
        # The written board is parsed back and compared with the plan: a pad
        # off a declared pin that changed, or a declared pin that did not get
        # its net, and nothing is kept.
        bad = ps.verify(pcb, parse_kicad_pcb(tmp), pools, res['changes'])
        if bad:
            os.remove(tmp)
            log("ERROR: the written board does not match the plan; nothing kept:")
            for line in bad[:20]:
                log(f"  {line}")
            report['verify'] = bad
            code = 3
        else:
            os.replace(tmp, out)
            from placement.portfolio import copy_siblings
            copy_siblings(board, out)
            log(f"wrote {out}")
    if code == 0 and schematic_dir and changes:
        from schematic_updater import apply_swaps_to_schematics
        swaps, unwritable = ps.schematic_swaps(pools, res['changes'])
        report['schematic'] = {'dir': schematic_dir, 'swaps': swaps,
                               'unwritable': unwritable}
        if dry_run:
            log(f"schematic: {len(swaps)} pin renumbering(s) would be written")
        else:
            done, failed = apply_swaps_to_schematics(schematic_dir, swaps)
            report['schematic'].update({'applied': done, 'failed': failed})
        for ref in unwritable:
            log(f"WARNING schematic: {ref} took nets from another part of its "
                f"pool -- a symbol only renumbers its own pins, so rewire it "
                f"by hand")
    if json_out:
        with open(json_out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1)
    return code, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('board', help='the placed board')
    ap.add_argument('out', help='the board to write (may be the input)')
    ap.add_argument('--brief', help='the design brief carrying pin_swaps[] '
                                    '(default: the one beside the board)')
    ap.add_argument('--crossing-weight', type=float, default=None,
                    help="mm of signal tree one ratsnest crossing is worth "
                         "(default: place_simple's own, 2.0)")
    ap.add_argument('--json-out', help='write the report here: every pin '
                                       'changed, old net -> new net')
    ap.add_argument('--schematic-dir', help='also renumber the swapped pins '
                                            'in the .kicad_sch files here')
    ap.add_argument('--dry-run', action='store_true',
                    help='report the swaps, write no board')
    a = ap.parse_args(argv)
    if not a.dry_run:
        try:
            from redo_record import record_invocation
            record_invocation()
        except Exception:                               # noqa: BLE001
            pass
    code, _ = run(a.board, a.out, brief=a.brief, crossing_weight=a.crossing_weight,
                  json_out=a.json_out, schematic_dir=a.schematic_dir,
                  dry_run=a.dry_run)
    return code


if __name__ == '__main__':
    sys.exit(main())
