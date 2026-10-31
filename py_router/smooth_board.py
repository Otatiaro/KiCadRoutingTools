#!/usr/bin/env python3
"""Collapse the staircase micro-jogs of a finished board, whichever step laid them.

route.py smooths only the nets its own run routes (#536), so copper from earlier
chain steps keeps its grid jogs: differential pairs, plane taps, and every net a
later retry left alone. This tool runs the same smoother over the whole board as
a last chain step:

    python3 smooth_board.py in.kicad_pcb out.kicad_pcb --clearance 0.09

Pads, vias and zones stay where they are; a span is replaced only by a shorter
octolinear connector that clears foreign copper (smooth_octolinear_chains), and
no net may grade worse afterwards. Protected nets (coupled pairs, length-matched
groups, impedance nets, KiCad-locked copper) are skipped unless
--include-protected names the choice. Zones are not refilled: refill before DRC.
"""

#: #937 registry: which door(s) show this tool, and whether it changes
#: the board. Read by krt_registry.py -- by AST, never imported.
KRT_TOOL = {'scope': ['routing'], 'kind': 'actor'}

import os
if __name__ == '__main__':  # one OpenBLAS thread: tests/test_blas_single_thread.py
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import fnmatch
import json
import shutil
import sys


def _sibling_copy(src, dst):
    from copy_board import SIBLING_EXTS
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
        s, d = os.path.splitext(src)[0], os.path.splitext(dst)[0]
        for ext in SIBLING_EXTS:
            if os.path.exists(s + ext):
                shutil.copyfile(s + ext, d + ext)


def _write(path, pcb, removed, added):
    """Take the replaced tracks out of the file and put the new ones in."""
    from kicad_parser import board_uses_name_nets
    from kicad_writer import generate_segment_sexpr, remove_segments_from_content
    name_of = {nid: n.name for nid, n in pcb.nets.items()}
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    named = board_uses_name_nets(content)
    unmatched = []
    content, n = remove_segments_from_content(content, removed, net_id_to_name=name_of if named else None,
                                              unmatched_out=unmatched, match_width=True)
    if unmatched:
        raise RuntimeError(f'{len(unmatched)} replaced track(s) not found in {path}: nothing written')
    sexprs = [generate_segment_sexpr((p.start_x, p.start_y), (p.end_x, p.end_y), p.width, p.layer, p.net_id,
                                     name_of.get(p.net_id) if named else None) for p in added]
    if sexprs:
        lp = content.rfind(')')
        content = content[:lp] + '\n'.join(sexprs) + '\n' + content[lp:]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    return n


def smooth_board(path, clearance, board_edge_clearance, include_protected=False, skip_patterns=(), passes=3,
                 dry_run=False):
    """Smooth the board file in place until a pass changes nothing. Returns the summed stats."""
    from cleanup_pipeline import _smooth_skip_net_ids
    from design_rules import install_design_rules
    from kicad_parser import parse_kicad_pcb
    from pcb_modification import smooth_octolinear_chains
    from routing_config import GridRouteConfig
    from list_nets import net_clearance_map_by_id
    from net_keepouts import attach as attach_keepouts
    total = {'passes': 0, 'spans': 0, 'segs_removed': 0, 'segs_added': 0, 'saved_mm': 0.0, 'chains_reverted': 0}
    for _ in range(passes):
        pcb = parse_kicad_pcb(path)
        config = GridRouteConfig(clearance=clearance, board_edge_clearance=board_edge_clearance)
        rules = install_design_rules(config, path, pcb, net_keepouts=True)
        attach_keepouts(pcb, getattr(rules, 'net_keepouts', None))
        clearances = net_clearance_map_by_id(path, {nid: n.name for nid, n in pcb.nets.items()}) or {}
        config.set_net_clearances(clearances, [])
        skip = set() if include_protected else set(_smooth_skip_net_ids(pcb))
        skip |= {nid for nid, net in pcb.nets.items()
                 if any(fnmatch.fnmatchcase(net.name or '', p) for p in skip_patterns)}
        _n, _nets, removed, added, stats = smooth_octolinear_chains(
            [], pcb, None, clearance=clearance, net_clearances=clearances or None,
            board_edge_clearance=board_edge_clearance, config=config, skip_net_ids=skip, dry_run=dry_run)
        total['passes'] += 1
        for k in ('spans', 'segs_removed', 'segs_added', 'saved_mm', 'chains_reverted'):
            total[k] += stats.get(k, 0)
        if dry_run or not removed:
            break
        _write(path, pcb, removed, added)
    total['saved_mm'] = round(total['saved_mm'], 3)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--clearance', type=float, default=0.1, help="the board's Default clearance in mm (default 0.1)")
    ap.add_argument('--board-edge-clearance', type=float, default=0.0, help='copper to Edge.Cuts in mm (default: --clearance)')
    ap.add_argument('--include-protected', action='store_true',
                    help='also smooth protected nets (coupled pairs, matched groups, impedance nets); their gap and skew change')
    ap.add_argument('--skip-nets', nargs='*', default=[], metavar='GLOB', help='nets to leave as they are')
    ap.add_argument('--passes', type=int, default=3, help='at most this many passes (default 3)')
    ap.add_argument('--dry-run', action='store_true', help='measure only: the output is an unchanged copy')
    a = ap.parse_args()
    _sibling_copy(a.input, a.output)
    stats = smooth_board(a.output, a.clearance, a.board_edge_clearance, a.include_protected, a.skip_nets, a.passes,
                         a.dry_run)
    print('JSON_SUMMARY: ' + json.dumps(stats))
    return 0


if __name__ == '__main__':
    sys.exit(main())
