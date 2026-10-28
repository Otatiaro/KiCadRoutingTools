"""Victim guard: a route run never ships a net less connected than it got it.

Why this exists
---------------
A rip is a trade: the ripper takes the victim's corridor, the victim is queued
for a reroute and its original copper is restored if the reroute fails. Both
halves of that custody can fail. The restore is refused when copper routed
since the rip now holds the corridor (#134), and a reroute that lands PARTIAL
is kept "honestly" when the ripper took the corridor. The victim then ships
broken while the ripper ships connected, and nothing re-checks the trade.

The improvement gate (#600) catches the case where the run ends worse on
balance and reverts the WHOLE run, discarding every real gain with it. It
cannot catch an exact trade: one net connected for one broken ships as a
draw. That is what fine-grid retry steps do on a congested board -- two nets
connected, two others broken, step after step.

The guard closes the trade per net, at the end of the run, on the board it
ships. Every net that was connected on the input and is broken now (or has
more disconnected pads than it started with) gets its INPUT copper back. Any
net whose new copper collides with that input copper returns to its input
state too, and so on until nothing collides: the closure. The input copper of
all those nets coexisted on the input board, so the closure always has a
legal fixed point, and every net in it ends exactly as it started. So no net
ends worse than it began, by construction, and the gains that do not depend on
a victim's corridor are kept.

What the guard does not touch (it leaves those victims to the gate):

- a net whose PADS changed (a pad swap, a moved footprint): its input copper
  no longer reaches its pads;
- a net that owns a pour: its connectivity is the fill's, which other nets'
  copper also shapes, so input copper alone does not restore it.

A victim whose closure reaches one of those is not restored; it is named.
"""
from __future__ import annotations

import copy
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

COPPER_KINDS = ('segment', 'arc', 'via')


def damaged_nets(before: Dict[int, Tuple[bool, int]],
                 after: Dict[int, Tuple[bool, int]]) -> List[int]:
    """Nets the run made worse, on improvement_gate.compare_connectivity's
    terms: connected before and broken after (`lost`), or more disconnected
    pads than before (`worsened`)."""
    out = []
    for net_id, (conn_b, dis_b) in before.items():
        if net_id not in after:
            continue
        conn_a, dis_a = after[net_id]
        if (conn_b and not conn_a) or dis_a > dis_b:
            out.append(net_id)
    return sorted(out)


def connectivity_of(board, net_ids) -> Dict[int, Tuple[bool, int]]:
    """improvement_gate.net_connectivity_map's grade, for `net_ids` only. A
    net whose copper the run did not change kept its connectivity, so the
    guard grades the changed nets alone (and the pours, which other nets'
    copper shapes) rather than the whole board three times."""
    from improvement_gate import grade_nets
    return {n: (not g['broken'], len(g['failed_pads']))
            for n, g in grade_nets(board, sorted(net_ids)).items()}


def restore_closure(seeds: Iterable[int],
                    input_copper: Dict[int, Tuple[list, list]],
                    after_board,
                    clearance: float,
                    config=None,
                    frozen: Iterable[int] = (),
                    partner_of: Optional[Dict[int, int]] = None
                    ) -> Tuple[Set[int], Dict[int, int]]:
    """The nets that must return to their input copper so that every seed
    does, and the seeds that cannot.

    `input_copper` is {net: (segments, vias)} on the input board,
    `after_board` the board the run would ship (its `segments`/`vias`), all
    in one net numbering. A net joins a seed's set when its shipped copper
    collides with the input copper of a net already in it
    (rip_up_reroute._saved_route_colliders, pairwise clearances when `config`
    is given), and a diff-pair net brings its partner. A seed whose set
    reaches a `frozen` net is refused: {seed: the frozen net}.

    Each seed grows on its own, so one refused seed does not cost the others
    their restore; the union of closed sets is closed.
    """
    from rip_up_reroute import _saved_route_colliders

    frozen = set(frozen)
    partner_of = partner_of or {}
    restore: Set[int] = set()
    refused: Dict[int, int] = {}
    for seed in sorted(set(seeds)):
        if seed in restore:
            continue
        group = {seed}
        pending = [seed]
        if partner_of.get(seed) is not None:
            group.add(partner_of[seed])
            pending.append(partner_of[seed])
        blocker = next((n for n in group if n in frozen), None)
        while pending and blocker is None:
            net = pending.pop()
            segs, vias = input_copper.get(net, ([], []))
            hits = _saved_route_colliders(
                {'new_segments': list(segs), 'new_vias': list(vias)},
                after_board, list(group), clearance, config=config)
            for _kind, obj in hits:
                other = obj.net_id
                if other in group:
                    continue
                if other in frozen:
                    blocker = other
                    break
                for n in (other, partner_of.get(other)):
                    if n is not None and n not in group:
                        if n in frozen:
                            blocker = n
                            break
                        group.add(n)
                        pending.append(n)
                if blocker is not None:
                    break
        if blocker is not None:
            refused[seed] = blocker
        else:
            restore |= group
    return restore, refused


def remap_copper(items, net_id_of_name, name_of_id) -> list:
    """Copies of `items` with each net id carried over by NAME to another
    numbering (two parses of one board may number nets differently). An item
    whose net has no name in the target numbering is dropped."""
    out = []
    for it in items:
        nid = net_id_of_name.get(name_of_id(it.net_id))
        if nid is None:
            continue
        c = copy.copy(it)
        c.net_id = nid
        out.append(c)
    return out


# ---- the written file ------------------------------------------------------

def _top_level_copper_blocks(content: str) -> List[Tuple[int, int, str]]:
    """(start, end, kind) of every top-level (segment|arc|via ...) block.

    Depth-aware, so a zone outline's `(arc (start) (mid) (end))` is not taken
    for a track arc, and string- and escape-aware like
    kicad_writer._sexpr_block_end, since net names may hold parentheses."""
    out = []
    n = len(content)
    depth = 0
    in_str = False
    i = 0
    start = kind = None
    while i < n:
        c = content[i]
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == '(':
            depth += 1
            if depth == 2:
                m = re.match(r'\((segment|arc|via)(?=[\s(])', content[i:i + 9])
                if m:
                    start, kind = i, m.group(1)
        elif c == ')':
            if depth == 2 and start is not None:
                out.append((start, i + 1, kind))
                start = kind = None
            depth -= 1
        i += 1
    return out


_NET_TOKEN = re.compile(r'\(net\s+(?:(\d+)(?:\s+"((?:[^"\\]|\\.)*)")?|"((?:[^"\\]|\\.)*)")\s*\)')


def block_net_name(block: str, name_of_id: Dict[int, str]) -> Optional[str]:
    """The net a copper block belongs to, by name, in either dialect:
    `(net 5)`, `(net 5 "name")` or `(net "name")`."""
    m = _NET_TOKEN.search(block)
    if not m:
        return None
    if m.group(3) is not None:
        return re.sub(r'\\(.)', r'\1', m.group(3))
    if m.group(2) is not None:
        return re.sub(r'\\(.)', r'\1', m.group(2))
    return name_of_id.get(int(m.group(1)))


def _blocks_by_net(content: str, names: Set[str],
                   name_of_id: Dict[int, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for start, end, _kind in _top_level_copper_blocks(content):
        block = content[start:end]
        name = block_net_name(block, name_of_id)
        if name in names:
            out.setdefault(name, []).append(block)
    return out


def _canon(block: str) -> str:
    return ' '.join(block.split())


def restore_input_copper_text(out_content: str, in_content: str,
                              names: Set[str],
                              out_name_of_id: Dict[int, str],
                              in_name_of_id: Dict[int, str]
                              ) -> Tuple[str, int, int]:
    """`out_content` with the copper of every net in `names` replaced by its
    blocks on the input file, verbatim: uuids, locks and via protection
    included. A block both files carry (same text up to whitespace) stays
    where it is. Returns (content, blocks removed, blocks added)."""
    in_blocks = _blocks_by_net(in_content, names, in_name_of_id)
    keep: Dict[str, int] = {}
    for blocks in in_blocks.values():
        for b in blocks:
            keep[_canon(b)] = keep.get(_canon(b), 0) + 1
    present = dict(keep)
    cuts = []
    for start, end, _kind in _top_level_copper_blocks(out_content):
        block = out_content[start:end]
        if block_net_name(block, out_name_of_id) not in names:
            continue
        k = _canon(block)
        if present.get(k, 0) > 0:
            present[k] -= 1          # input copper still on the board
            continue
        cuts.append((start, end))
    pieces = []
    last = 0
    for start, end in cuts:
        # The block and the indentation before it, so no blank line is left.
        line_start = out_content.rfind('\n', last, start)
        cut_from = (line_start if line_start >= 0
                    and not out_content[line_start + 1:start].strip()
                    else start)
        pieces.append(out_content[last:cut_from])
        last = end
    pieces.append(out_content[last:])
    content = ''.join(pieces)
    added = []
    for blocks in in_blocks.values():
        for b in blocks:
            k = _canon(b)
            if present.get(k, 0) > 0:
                present[k] -= 1
                added.append(b)
    if added:
        tail = content.rstrip()
        if not tail.endswith(')'):
            raise ValueError('the written board does not end with its closing paren')
        content = (tail[:-1].rstrip() + '\n'
                   + ''.join('\t' + b + '\n' for b in added) + ')\n')
    return content, len(cuts), len(added)
