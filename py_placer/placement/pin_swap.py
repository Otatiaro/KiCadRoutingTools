"""Declared pin and gate swaps: which nets go on which equivalent pins.

A part often has pins that do the same job: the four channels of an ESD
array, the four receivers of a quad line receiver, the two halves of a dual
op amp, the four resistors of an array. Which net goes on which channel is a
schematic convenience, and the layout pays for it in crossings when the
channels face the wrong way. Several identical parts side by side (five ESD
arrays guarding one 20-pin connector) make one bigger pool: which line goes
on which part.

Nothing in a board file says which pins are equivalent. That is a datasheet
fact, so it is DECLARED, in the design brief's `pin_swaps[]` (read here by
`brief_rows`, the one parser of the key; `design_brief` validates a brief
through it). One row is one pool:

    {"ref": "U3",                        # or "refs": ["U3", "U4"]: a pool
     "gates": [["1", "10"], ["2", "9"], ["4", "7"], ["5", "6"]],
     "common": ["3", "8"],               # never touched (optional)
     "adjacent": [["1", "2"], ["4", "5"]],  # gates side by side (optional)
     "why": "TPD4E05U06 flow-through channels"}

A GATE is an ordered list of pins that moves as one block: the k-th pin of a
gate takes the place of the k-th pin of the gate it trades with, so a gate
never turns round and no swap can invert a polarity. A gate's nets move with
it, unused ones (an empty channel, a no-connect) included. `adjacent` names
pairs of gates (each by any one of its pins) that sit side by side: a
differential pair (`list_nets.find_differential_pairs`) running through two
gates of a pool may only ever land on two adjacent gates of one part, and
without `adjacent` it does not move at all.

`optimise` then picks, on a placed board, the assignment of each pool's
bundles of nets to its gates that lowers place_simple's own signal cost
(`placement.simple`): the Manhattan MST of every signal net plus
`crossing_weight` x the crossings among those trees, rails left out as
place_simple leaves them out (`grouping.rail_net_ids`). An optimal assignment
on the tree lengths alone (`scipy`'s `linear_sum_assignment`) and the input's
own assignment each seed a descent over gate swaps (and swaps of two adjacent
pairs at once), priced on the whole cost, and the cheaper end is kept; the
input is kept unless something is strictly cheaper.

Only declared pins ever change, and of those never a pin with copper on it,
a net the board protects (`protected_nets`: matched groups, routed pairs), a
net carrying locked or held copper (`is_immutable_copper`) or a pad promised
to a pour: a gate holding any of them keeps its nets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

#: The design brief's key.
BRIEF_KEY = 'pin_swaps'
#: The author looked and cannot yet say which pins are equivalent.
BRIEF_UNKNOWN = 'unknown'
BRIEF_ROW_KEYS = ('ref', 'refs', 'gates', 'common', 'adjacent', 'name', 'why',
                  'requirement', 'note', 'context')
#: place_simple's own default weight of one ratsnest crossing, in mm of tree.
CROSSING_WEIGHT = 2.0
#: Accepted descent moves, all pools together, before the descent stops.
MAX_MOVES = 2000
_EPS = 1e-9


class PinSwapError(ValueError):
    """A `pin_swaps` declaration that cannot be honoured as written."""


# --------------------------------------------------------------------------
# the declaration

def _pins(value, at: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{at}: expected a non-empty list of pin numbers")
    for v in value:
        if not isinstance(v, str) or not v:
            # Numbers as strings, as in a proximity row's `pads`: "1" and 1
            # are different pads to a reader that compares them as written,
            # and BGA pins ("A1") are strings anyway.
            raise ValueError(f"{at}: pin {v!r} is not a string -- write pin "
                             f"numbers as strings (\"1\", \"A3\")")
    return tuple(value)


def brief_rows(value, where: str = BRIEF_KEY) -> List[dict]:
    """Validate a design brief's `pin_swaps` value; one row per pool.

    A list of rows, or "unknown" (no rows). Board-independent: a row is
    refused here when it cannot be honoured on ANY board (gates of different
    sizes, a pin in two gates, an `adjacent` naming no gate, a part in two
    rows); `resolve` checks the rest against a board. Returns
    `[{'row', 'refs', 'gates', 'common', 'adjacent', <prose keys>}]`, gates as
    tuples of pin strings and `adjacent` as pairs of gate indices."""
    if value is None or value == BRIEF_UNKNOWN:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a list of rows or "
                         f"{BRIEF_UNKNOWN!r}, got {type(value).__name__}")
    rows: List[dict] = []
    owner: Dict[str, int] = {}
    for i, entry in enumerate(value):
        at = f"{where}[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{at}: expected an object, got {entry!r}")
        bad = sorted(str(k) for k in set(entry) - set(BRIEF_ROW_KEYS))
        if bad:
            raise ValueError(f"{at}: unknown key(s) {', '.join(bad)} (a row "
                             f"carries {', '.join(BRIEF_ROW_KEYS)})")
        if entry.get('context') is not None and not isinstance(entry['context'], dict):
            raise ValueError(f"{at}.context: expected an object")
        if ('ref' in entry) == ('refs' in entry):
            raise ValueError(f"{at}: give exactly one of `ref` (one part) or "
                             f"`refs` (a pool of identical parts)")
        if 'ref' in entry:
            if not isinstance(entry['ref'], str) or not entry['ref']:
                raise ValueError(f"{at}.ref: expected a reference string")
            refs = (entry['ref'],)
        else:
            refs = _pins(entry['refs'], f"{at}.refs")
            if len(set(refs)) != len(refs):
                raise ValueError(f"{at}.refs: a part is listed twice")
        for r in refs:
            if r in owner:
                raise ValueError(f"{at}: {r} is already in {where}"
                                 f"[{owner[r]}] -- one part, one pool")
            owner[r] = i
        gates_raw = entry.get('gates')
        if not isinstance(gates_raw, list) or len(gates_raw) < 1:
            raise ValueError(f"{at}.gates: expected a list of gates, each a "
                             f"list of pins")
        gates = tuple(_pins(g, f"{at}.gates[{k}]") for k, g in enumerate(gates_raw))
        if len(gates) * len(refs) < 2:
            raise ValueError(f"{at}.gates: one gate on one part leaves nothing "
                             f"to swap")
        sizes = sorted({len(g) for g in gates})
        if len(sizes) > 1:
            raise ValueError(
                f"{at}.gates: gates of different sizes ({', '.join(map(str, sizes))} "
                f"pins) -- the k-th pin of a gate takes the place of the k-th "
                f"pin of another, so every gate has the same pins in the same "
                f"order")
        seen: Dict[str, int] = {}
        for k, g in enumerate(gates):
            for p in g:
                if p in seen:
                    raise ValueError(f"{at}.gates: pin {p!r} is in gate "
                                     f"{seen[p]} and gate {k}")
                seen[p] = k
        common = _pins(entry['common'], f"{at}.common") if entry.get('common') else ()
        both = sorted(set(common) & set(seen))
        if both:
            raise ValueError(f"{at}: pin(s) {', '.join(both)} are both in a "
                             f"gate and `common`")
        adjacent = []
        for j, pair in enumerate(entry.get('adjacent') or ()):
            pa = _pins(pair, f"{at}.adjacent[{j}]")
            if len(pa) != 2:
                raise ValueError(f"{at}.adjacent[{j}]: expected two pins, one "
                                 f"of each of two gates side by side")
            for p in pa:
                if p not in seen:
                    raise ValueError(f"{at}.adjacent[{j}]: pin {p!r} is in no "
                                     f"gate")
            ga, gb = seen[pa[0]], seen[pa[1]]
            if ga == gb:
                raise ValueError(f"{at}.adjacent[{j}]: both pins are in gate "
                                 f"{ga} -- name two gates")
            adjacent.append(tuple(sorted((ga, gb))))
        row = {'row': i, 'refs': refs, 'gates': gates, 'common': common,
               'adjacent': tuple(sorted(set(adjacent)))}
        for k in ('name', 'why', 'requirement', 'note', 'context'):
            if entry.get(k) is not None:
                row[k] = entry[k]
        rows.append(row)
    return rows


def load_rows(board_path: str, brief_path: Optional[str] = None
              ) -> Tuple[List[dict], str]:
    """`(rows, brief path)` off the brief beside the board, or `brief_path`.
    `([], '')` when there is no brief. A malformed brief raises its own
    `BriefError` (a ValueError)."""
    from . import design_brief
    path = brief_path or design_brief.discover_brief(board_path)
    if not path:
        return [], ''
    brief = design_brief.load_brief(path)
    return [dict(r) for r in brief.pin_swaps], path


# --------------------------------------------------------------------------
# the declaration on a board

@dataclass
class Slot:
    """One gate of one part: its pins' pads (twins included), in gate order."""
    ref: str
    gate: int
    pins: Tuple[str, ...]
    pads: Tuple[Tuple[object, ...], ...]

    @property
    def key(self):
        return (self.ref, self.gate)


@dataclass
class Pool:
    row: int
    name: str
    refs: Tuple[str, ...]
    gates: Tuple[Tuple[str, ...], ...]
    adjacent: Set[Tuple[int, int]]
    slots: List[Slot] = field(default_factory=list)


def resolve(rows: Sequence[dict], pcb) -> List[Pool]:
    """The rows on `pcb`: every part present, every pin a pad, a pool of
    one footprint and one value, and the pins at one place in every gate of
    one electrical type. Raises PinSwapError naming what does not hold."""
    pools = []
    for row in rows:
        at = f"{BRIEF_KEY}[{row['row']}]"
        missing = [r for r in row['refs'] if r not in pcb.footprints]
        if missing:
            raise PinSwapError(f"{at}: {', '.join(missing)} not on this board")
        fps = [pcb.footprints[r] for r in row['refs']]
        names = sorted({f.footprint_name for f in fps})
        if len(names) > 1:
            raise PinSwapError(f"{at}: a pool of different footprints "
                               f"({', '.join(names)}) -- only identical parts "
                               f"trade gates")
        values = sorted({f.value for f in fps if f.value})
        if len(values) > 1:
            raise PinSwapError(f"{at}: a pool of different values "
                               f"({', '.join(values)}) -- only identical parts "
                               f"trade gates")
        pool = Pool(row=row['row'], name=str(row.get('name') or at),
                    refs=tuple(row['refs']), gates=tuple(row['gates']),
                    adjacent=set(row['adjacent']))
        for f in fps:
            by_num: Dict[str, list] = {}
            for p in f.pads:
                by_num.setdefault(str(p.pad_number), []).append(p)
            for k, g in enumerate(row['gates']):
                absent = [p for p in g if p not in by_num]
                if absent:
                    raise PinSwapError(f"{at}: {f.reference} has no pin "
                                       f"{', '.join(absent)}")
                pool.slots.append(Slot(f.reference, k, tuple(g),
                                       tuple(tuple(by_num[p]) for p in g)))
            for p in row.get('common') or ():
                if p not in by_num:
                    raise PinSwapError(f"{at}: {f.reference} has no common "
                                       f"pin {p}")
        # A gate's k-th pins must do the same job. Pin types are the
        # library's word for it: an input among outputs is a part that is
        # not symmetric the way the row says (a BAT54S's anode and cathode).
        size = len(row['gates'][0])
        for k in range(size):
            kinds = {}
            for s in pool.slots:
                t = (getattr(s.pads[k][0], 'pintype', '') or '').split('+')[0]
                if t and t != 'no_connect':
                    kinds.setdefault(t, []).append(f"{s.ref}.{s.pins[k]}")
            if len(kinds) > 1:
                raise PinSwapError(
                    f"{at}: position {k} of the gates holds pins of different "
                    f"types ({'; '.join(f'{t}: {v[0]}' for t, v in sorted(kinds.items()))}) "
                    f"-- these gates are not equivalent")
        pools.append(pool)
    return pools


def _net_of(slot: Slot, k: int) -> Tuple[int, Optional[str]]:
    """`(net id, reason)`: the net of a gate's k-th pin, and why the slot
    cannot move when its twin pads disagree."""
    ids = {p.net_id or 0 for p in slot.pads[k]}
    if len(ids) > 1:
        return 0, f"pin {slot.pins[k]}'s pads carry different nets"
    return ids.pop(), None


def _touches(pad, x, y, reach) -> bool:
    return (abs(x - pad.global_x) <= pad.size_x / 2 + reach + _EPS
            and abs(y - pad.global_y) <= pad.size_y / 2 + reach + _EPS)


def frozen_slots(pools: Sequence[Pool], pcb, protected=None,
                 pour_served=None, diff_pairs=None) -> Dict[Tuple, str]:
    """`{(ref, gate): reason}` for every slot that keeps its nets.

    A slot is frozen when a pad of it has copper ending on it, when one of
    its nets is protected (`protected`: net name -> reason, the board's
    `protected_nets`), carries locked or held copper anywhere, or is promised
    to a pour on one of its pads (`pour_served`: "REF.PAD" keys), when its
    twin pads disagree, or when it holds half of a differential pair
    (`diff_pairs`: `[(p name, n name)]`) that this pool cannot keep on two
    adjacent gates."""
    from kicad_parser import is_immutable_copper
    protected = protected or {}
    pour_served = set(pour_served or ())
    name_of = {nid: n.name for nid, n in pcb.nets.items()}
    immutable = ({s.net_id for s in pcb.segments if is_immutable_copper(s)}
                 | {v.net_id for v in pcb.vias if is_immutable_copper(v)})
    out: Dict[Tuple, str] = {}
    for pool in pools:
        for s in pool.slots:
            why = None
            for k, pads in enumerate(s.pads):
                nid, bad = _net_of(s, k)
                why = why or bad
                name = name_of.get(nid, '') if nid else ''
                if not why and name in protected:
                    why = f"net {name} is protected ({protected[name]})"
                if not why and nid and nid in immutable:
                    why = f"net {name} carries locked copper"
                for p in pads:
                    if why:
                        break
                    if f"{s.ref}.{p.pad_number}" in pour_served:
                        why = f"pad {s.ref}.{p.pad_number} is promised to a pour"
                        break
                    for seg in pcb.segments:
                        if (_touches(p, seg.start_x, seg.start_y, seg.width / 2)
                                or _touches(p, seg.end_x, seg.end_y, seg.width / 2)):
                            why = f"pad {s.ref}.{p.pad_number} has copper on it"
                            break
                    if not why:
                        for v in pcb.vias:
                            if _touches(p, v.x, v.y, v.size / 2):
                                why = f"pad {s.ref}.{p.pad_number} has a via on it"
                                break
            if why:
                out[s.key] = why
        _freeze_pairs(pool, pcb, diff_pairs or (), out)
    return out


def pair_ids(pcb, diff_pairs):
    """`[(p name, n name, p id, n id)]` for the pairs whose nets are both on
    the board (`diff_pairs`: `[(p name, n name)]`)."""
    ids = {n.name: nid for nid, n in pcb.nets.items() if n.name}
    return [(pn, nn, ids[pn], ids[nn]) for pn, nn in diff_pairs
            if pn in ids and nn in ids]


def _pair_slots(pool: Pool, pairs, bundle_of):
    """`[(p name, n name, slots holding p, slots holding n)]` for each pair
    (`pair_ids`) whose two nets both run through this pool."""
    where = {}
    for i in range(len(pool.slots)):
        for nid in bundle_of(i):
            if nid:
                where.setdefault(nid, set()).add(i)
    out = []
    for pn, nn, pid, nid in pairs:
        a, b = where.get(pid), where.get(nid)
        if a and b:
            out.append((pn, nn, a, b))
    return out


def _adjacent(pool: Pool, si: int, sj: int) -> bool:
    a, b = pool.slots[si], pool.slots[sj]
    if si == sj:
        return True
    return a.ref == b.ref and tuple(sorted((a.gate, b.gate))) in pool.adjacent


def _freeze_pairs(pool: Pool, pcb, diff_pairs, out) -> None:
    """Freeze the slots of a pair this pool cannot move while keeping it on
    two adjacent gates: one with no `adjacent` declared, one that is not on
    adjacent gates already, one spread over more than one gate per net."""
    def bundle(i):
        return [_net_of(pool.slots[i], k)[0] for k in range(len(pool.gates[0]))]
    for pn, nn, a, b in _pair_slots(pool, pair_ids(pcb, diff_pairs), bundle):
        why = None
        if len(a) > 1 or len(b) > 1:
            why = f"pair {pn}/{nn} runs through more than two of its gates"
        elif not _adjacent(pool, next(iter(a)), next(iter(b))):
            why = (f"pair {pn}/{nn} is not on two adjacent gates"
                   + ("" if pool.adjacent else
                      " (the row declares no `adjacent` gates)"))
        if why:
            for i in a | b:
                out.setdefault(pool.slots[i].key, why)


# --------------------------------------------------------------------------
# the cost

class Board:
    """The signal cost of an assignment, priced as place_simple prices it:
    each signal net's Manhattan MST plus `crossing_weight` x the proper
    crossings among the trees of different nets. Rails take no part."""

    def __init__(self, pcb, pools: Sequence[Pool], crossing_weight: float):
        import numpy as np
        from .grouping import rail_net_ids
        self.np = np
        self.cw = crossing_weight
        self.pools = list(pools)
        owners: Dict[int, Set[str]] = {}
        for nid, net in pcb.nets.items():
            for p in net.pads:
                owners.setdefault(nid, set()).add(p.component_ref)
        for fp in pcb.footprints.values():
            for p in fp.pads:
                if p.net_id:
                    owners.setdefault(p.net_id, set()).add(fp.reference)
        name_of = {nid: n.name for nid, n in pcb.nets.items()}
        self.rails = rail_net_ids(owners, lambda n: name_of.get(n, ''),
                                  len(pcb.footprints))
        # pads that may change: (ref, pin) -> [points]; the rest per net
        self.swap_pts: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        for pool in self.pools:
            for s in pool.slots:
                for pin, pads in zip(s.pins, s.pads):
                    self.swap_pts[(s.ref, pin)] = [(p.global_x, p.global_y)
                                                   for p in pads]
        self.base: Dict[int, List[Tuple[float, float]]] = {}
        for fp in pcb.footprints.values():
            for p in fp.pads:
                if not p.net_id or (fp.reference, str(p.pad_number)) in self.swap_pts:
                    continue
                self.base.setdefault(p.net_id, []).append((p.global_x, p.global_y))
        # current net on each swappable pin, and the reverse
        self.at: Dict[Tuple[str, str], int] = {}
        for pool in self.pools:
            for s in pool.slots:
                for k, pin in enumerate(s.pins):
                    self.at[(s.ref, pin)] = _net_of(s, k)[0]
        self.pins_of: Dict[int, Set[Tuple[str, str]]] = {}
        for key, nid in self.at.items():
            self.pins_of.setdefault(nid, set()).add(key)
        self.tree_len: Dict[int, float] = {}
        self.edges: Dict[int, object] = {}
        for nid in self.signal_nets():
            self._price(nid)
        self._stack()
        self.crossings = self._all_crossings()

    def signal(self, nid) -> bool:
        return bool(nid) and nid not in self.rails

    def signal_nets(self):
        return sorted({n for n in set(self.base) | set(self.pins_of) if self.signal(n)})

    def _points(self, nid, at=None, pins_of=None):
        at = self.at if at is None else at
        pins = (self.pins_of if pins_of is None else pins_of).get(nid, ())
        pts = list(self.base.get(nid, ()))
        for key in sorted(pins):
            pts.extend(self.swap_pts[key])
        return pts

    def _tree(self, pts):
        from .simple import mst_l1
        length, edges = mst_l1(pts)
        return length, self.np.asarray(edges, dtype=float).reshape(-1, 4)

    def _price(self, nid):
        self.tree_len[nid], self.edges[nid] = self._tree(self._points(nid))

    def _stack(self):
        np = self.np
        nets = [n for n in self.edges if len(self.edges[n])]
        self.E = (np.concatenate([self.edges[n] for n in nets])
                  if nets else np.zeros((0, 4)))
        self.L = (np.concatenate([np.full(len(self.edges[n]), n) for n in nets])
                  if nets else np.zeros(0, dtype=int))

    def _all_crossings(self) -> int:
        from .simple import crossings
        total = 0
        for n, e in self.edges.items():
            if len(e):
                total += crossings(e, self.E[self.L > n])
        return total

    def cost(self, length=None, cross=None) -> float:
        length = sum(self.tree_len.values()) if length is None else length
        cross = self.crossings if cross is None else cross
        return length + self.cw * cross

    def totals(self):
        return {'tree_mm': round(sum(self.tree_len.values()), 3),
                'crossings': int(self.crossings),
                'cost': round(self.cost(), 3)}

    def trial(self, change: Dict[Tuple[str, str], int]):
        """Price `change` ({pin key: new net}) without keeping it:
        `(delta cost, new trees)`."""
        from .simple import crossings
        np = self.np
        pins_of = {}
        touched = set()
        for key, new in change.items():
            old = self.at[key]
            if old == new:
                continue
            for n in (old, new):
                if n not in pins_of:
                    pins_of[n] = set(self.pins_of.get(n, ()))
            pins_of[old].discard(key)
            pins_of[new].add(key)
            touched.update((old, new))
        nets = sorted(n for n in touched if self.signal(n))
        if not nets:
            return 0.0, {}
        trees = {n: self._tree(self._points(n, pins_of=pins_of)) for n in nets}
        rest = self.E[~np.isin(self.L, nets)]
        d_len = sum(trees[n][0] - self.tree_len.get(n, 0.0) for n in nets)
        d_cross = 0
        for i, n in enumerate(nets):
            old_e = self.edges.get(n, np.zeros((0, 4)))
            new_e = trees[n][1]
            d_cross += crossings(new_e, rest) - crossings(old_e, rest)
            for m in nets[i + 1:]:
                d_cross += (crossings(new_e, trees[m][1])
                            - crossings(old_e, self.edges.get(m, np.zeros((0, 4)))))
        return d_len + self.cw * d_cross, {'trees': trees, 'pins_of': pins_of,
                                           'd_cross': d_cross, 'change': change}

    def commit(self, priced):
        if not priced:
            return
        for key, new in priced['change'].items():
            self.at[key] = new
        for n, pins in priced['pins_of'].items():
            self.pins_of[n] = pins
        for n, (length, edges) in priced['trees'].items():
            self.tree_len[n], self.edges[n] = length, edges
        self.crossings += priced['d_cross']
        self._stack()


# --------------------------------------------------------------------------
# the search

def _bundles(pool: Pool) -> List[Tuple[int, ...]]:
    size = len(pool.gates[0])
    return [tuple(_net_of(s, k)[0] for k in range(size)) for s in pool.slots]


def _change(pool: Pool, placement: Dict[int, Tuple[int, ...]]):
    """{pin key: net} putting bundle `placement[i]` on slot i."""
    out = {}
    for i, b in placement.items():
        s = pool.slots[i]
        for pin, nid in zip(s.pins, b):
            out[(s.ref, pin)] = nid
    return out


def _pairs_ok(pool: Pool, pairs, content) -> bool:
    """Is every differential pair (`pair_ids`) through the pool on two
    adjacent gates, with `content[i]` the bundle on slot i."""
    if not pairs:
        return True
    for _pn, _nn, a, b in _pair_slots(pool, pairs, lambda i: content[i]):
        if len(a) > 1 or len(b) > 1:
            return False
        if not _adjacent(pool, next(iter(a)), next(iter(b))):
            return False
    return True


def _moves(pool: Pool, free: List[int]):
    """Candidate moves on one pool, as lists of slot transpositions: one
    swap of two gates, and for two declared adjacent pairs, both gates of
    one traded with both gates of the other (either way round) -- the move
    that carries a differential pair."""
    for x in range(len(free)):
        for y in range(x + 1, len(free)):
            yield [(free[x], free[y])]
    by_key = {s.key: i for i, s in enumerate(pool.slots)}
    pairs = []
    for ref in pool.refs:
        for ga, gb in sorted(pool.adjacent):
            i, j = by_key.get((ref, ga)), by_key.get((ref, gb))
            if i in free and j in free:
                pairs.append((i, j))
    for x in range(len(pairs)):
        for y in range(x + 1, len(pairs)):
            (a1, a2), (b1, b2) = pairs[x], pairs[y]
            if len({a1, a2, b1, b2}) == 4:
                yield [(a1, b1), (a2, b2)]
                yield [(a1, b2), (a2, b1)]


def _descend(board: Board, pools, frees, pairs, content, limit):
    """Best-improvement descent over every pool's moves until none lowers
    the cost. `content[p][i]` is the bundle on slot i of pool p, updated in
    place. Returns the moves taken."""
    taken = 0
    while taken < limit:
        best = None
        for p, pool in enumerate(pools):
            for mv in _moves(pool, frees[p]):
                new = list(content[p])
                for i, j in mv:
                    new[i], new[j] = new[j], new[i]
                if not _pairs_ok(pool, pairs, new):
                    continue
                touched = {i for t in mv for i in t}
                delta, priced = board.trial(
                    _change(pool, {i: new[i] for i in touched}))
                if delta < -_EPS and (best is None or delta < best[0] - _EPS):
                    best = (delta, p, new, priced)
        if best is None:
            break
        _delta, p, new, priced = best
        content[p] = new
        board.commit(priced)
        taken += 1
    return taken


def _assign(board: Board, pool: Pool, free: List[int], content):
    """The bundles on `free` slots placed by an optimal assignment on the
    tree lengths alone, each bundle priced on each slot against the board as
    it stands. Returns the new content of the pool."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n = len(free)
    cost = np.zeros((n, n))
    for bi, src in enumerate(free):
        bundle = content[src]
        for si, dst in enumerate(free):
            s = pool.slots[dst]
            total = 0.0
            for pin, nid in zip(s.pins, bundle):
                if not board.signal(nid):
                    continue
                pts = list(board.base.get(nid, ()))
                for key in sorted(board.pins_of.get(nid, ())):
                    if key[0] == pool.slots[src].ref and key[1] in pool.slots[src].pins:
                        continue
                    pts.extend(board.swap_pts[key])
                pts.extend(board.swap_pts[(s.ref, pin)])
                total += board._tree(pts)[0]
            cost[bi, si] = total
    rows, cols = linear_sum_assignment(cost)
    new = list(content)
    for bi, si in zip(rows, cols):
        new[free[si]] = content[free[bi]]
    return new


def optimise(pcb, pools: Sequence[Pool], *, crossing_weight: float = CROSSING_WEIGHT,
             frozen: Optional[Dict[Tuple, str]] = None, diff_pairs=(),
             max_moves: int = MAX_MOVES) -> dict:
    """Pick each pool's assignment of bundles to gates. PURE on `pcb`.

    Two starts -- the board as it is, and an optimal assignment of each
    pool's free bundles on the tree lengths alone -- each descended on the
    whole cost; the cheaper end is kept, the input on a tie. Returns
    `{'before', 'after', 'changes', 'pools'}` with `changes` as
    `[(ref, pin, old net id, new net id)]`, declared pins only."""
    frozen = frozen or {}
    frees = [[i for i, s in enumerate(pool.slots) if s.key not in frozen]
             for pool in pools]
    start = [_bundles(pool) for pool in pools]
    pairs = pair_ids(pcb, diff_pairs)

    def run(seed_hungarian):
        board = Board(pcb, pools, crossing_weight)
        before = board.totals()
        content = [list(c) for c in start]
        if seed_hungarian:
            for p, pool in enumerate(pools):
                if len(frees[p]) < 2:
                    continue
                new = _assign(board, pool, frees[p], content[p])
                if not _pairs_ok(pool, pairs, new):
                    continue
                delta, priced = board.trial(
                    _change(pool, {i: new[i] for i in frees[p]}))
                board.commit(priced)
                content[p] = new
        moves = _descend(board, pools, frees, pairs, content, max_moves)
        return board, before, content, moves

    b0, before, c0, m0 = run(False)
    b1, _b, c1, m1 = run(True)
    if b1.cost() < b0.cost() - _EPS:
        board, content, moves, start_kind = b1, c1, m1, 'assignment'
    else:
        board, content, moves, start_kind = b0, c0, m0, 'input'
    changes = []
    per_pool = []
    for p, pool in enumerate(pools):
        n_changes = 0
        for i, s in enumerate(pool.slots):
            for pin, old, new in zip(s.pins, start[p][i], content[p][i]):
                if old != new:
                    changes.append((s.ref, pin, old, new))
                    n_changes += 1
        per_pool.append({'row': pool.row, 'name': pool.name,
                         'refs': list(pool.refs), 'gates': [list(g) for g in pool.gates],
                         'free_gates': len(frees[p]),
                         'frozen': [{'ref': s.ref, 'gate': list(s.pins),
                                     'reason': frozen[s.key]}
                                    for s in pool.slots if s.key in frozen],
                         'pins_changed': n_changes})
    return {'before': before, 'after': board.totals(), 'changes': changes,
            'pools': per_pool, 'start': start_kind, 'moves': moves,
            'crossing_weight': crossing_weight}


# --------------------------------------------------------------------------
# writing it

def net_tokens(content: str, pcb, pools: Sequence[Pool]) -> Dict[int, Optional[str]]:
    """`{net id: (net ...) token}` as the board's own pads spell it, for
    every net on a declared pin; 0 maps to None (no net)."""
    from kicad_writer import pad_net_spans
    out: Dict[int, Optional[str]] = {0: None}
    for pool in pools:
        for s in pool.slots:
            for k, pin in enumerate(s.pins):
                nid = _net_of(s, k)[0]
                if nid in out:
                    continue
                spans = pad_net_spans(content, s.ref, pin)
                if spans:
                    out[nid] = spans[0][2]
    return out


def apply(content: str, pcb, pools, changes) -> str:
    """`content` with each change's pin given its new net."""
    from kicad_writer import set_pad_nets_in_content
    tokens = net_tokens(content, pcb, pools)
    return set_pad_nets_in_content(
        content, [(ref, pin, tokens[new]) for ref, pin, _old, new in changes])


def verify(before, after, pools, changes) -> List[str]:
    """What is wrong with `after` (a parse of the written board) against
    `before` (the input's): a pad off a declared pin that changed net, a
    declared pin that did not get the net it was given, a pool whose
    multiset of nets changed. Empty when the write is exactly the plan."""
    name = lambda pcb, nid: pcb.nets[nid].name if nid in pcb.nets else ''  # noqa: E731
    declared = {(s.ref, pin) for pool in pools for s in pool.slots for pin in s.pins}
    want = {(r, p): name(before, new) for r, p, _o, new in changes}
    bad = []
    for ref, fp in before.footprints.items():
        fa = after.footprints.get(ref)
        if fa is None:
            bad.append(f"{ref} missing from the output")
            continue
        old = {}
        for p in fp.pads:
            old.setdefault(str(p.pad_number), set()).add(p.net_name or '')
        new = {}
        for p in fa.pads:
            new.setdefault(str(p.pad_number), set()).add(p.net_name or '')
        for num in sorted(set(old) | set(new)):
            key = (ref, num)
            exp = {want[key]} if key in want else old.get(num)
            if new.get(num) != exp:
                bad.append(f"{ref}.{num}: {sorted(old.get(num) or [])} -> "
                           f"{sorted(new.get(num) or [])}"
                           + ("" if key in declared else " (NOT declared)"))
    return bad


def schematic_swaps(pools, changes) -> Tuple[List[dict], List[str]]:
    """`(swaps, refs not writable)`: the pin-number transpositions that
    carry the board's change into each part's schematic symbol
    (`schematic_updater.apply_swaps_to_schematics`), and the parts whose
    change cannot be written that way because a net came from another part
    of a pool (a symbol only renumbers its own pins)."""
    by_ref: Dict[str, List[Tuple[str, int, int]]] = {}
    for ref, pin, old, new in changes:
        by_ref.setdefault(ref, []).append((pin, old, new))
    swaps, unwritable = [], []
    slots = {s.key: s for pool in pools for s in pool.slots}
    for ref, rows in sorted(by_ref.items()):
        # old net -> pins that had it, on this part
        owners: Dict[int, List[str]] = {}
        for (r, _g), s in slots.items():
            if r != ref:
                continue
            for k, pin in enumerate(s.pins):
                owners.setdefault(_net_of(s, k)[0], []).append(pin)
        sigma: Dict[str, str] = {}
        ok = True
        used: Set[str] = set()
        for pin, _old, new in sorted(rows):
            src = [p for p in owners.get(new, []) if p not in used]
            if not src:
                ok = False
                break
            src_pin = src[0]
            used.add(src_pin)
            sigma[src_pin] = pin
        if not ok or sorted(sigma) != sorted(sigma.values()):
            unwritable.append(ref)
            continue
        cur = {p: p for p in sigma}
        for pid in sorted(sigma):
            target = sigma[pid]
            if cur[pid] == target:
                continue
            other = next(q for q, v in cur.items() if v == target)
            swaps.append({'component_ref': ref, 'pad1': cur[pid], 'pad2': target})
            cur[other], cur[pid] = cur[pid], target
    return swaps, unwritable

