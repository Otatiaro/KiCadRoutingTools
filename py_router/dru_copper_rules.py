"""Custom-rule copper clearances the router holds per routed net.

Two shapes of `.kicad_dru` rule bind a PAIR of items by who they are, not by
where they are, so neither the #498 layer map nor the #735 track channel can
carry them, and the router used to announce them as skipped:

* a NET-PAIR clearance: `(constraint clearance (min X))` whose condition reads
  only net facts of A and B -- `NetName` (KiCad's case-insensitive `*`/`?`
  string compare), `NetClass`, `hasNetclass()`, `hasExactNetclass()`,
  `inDiffPair()` -- joined by `&&`, `||`, `!` and parentheses. An isolation
  barrier is the usual one: the nets of one side against every other net;
* a PAD hole clearance: `(constraint hole_clearance (min X))` whose condition
  reads item A only and is true for pads alone (`A.Type == 'Pad' &&
  A.Pad_Type == 'Through-hole' && A.Hole_Size_X >= 0.3mm`). KiCad evaluates a
  condition both ways round, so it binds every pair that has such a pad: the
  pad's hole against foreign copper, and the pad's copper against a foreign
  hole (a via drill).

`derive` reads both out of the resolver's table (`design_rules.DesignRules`,
whose evaluator decides every condition) into plain entries keyed by net
NAME, so a nested sub-run can take its parent's list; it marks each rule it
honours (`rule.honoured`) and leaves every other one as the resolver and
`kicad_dru` report it. A net-pair rule is evaluated once per pair of net
GROUPS -- nets that answer every atom of the condition alike -- so an
isolation rule over hundreds of nets costs a handful of evaluations. For a
given pair of nets the last matching net-pair rule sets the value, as KiCad's
forward walk does.

Both are RAISE-ONLY, stamped by `stamp` while a net routes, on top of the
pair clearance the map already prices: the copper of every net a routed net
is bound to (pads, tracks, vias, as `pcb_data` holds them at that moment) at
the rule value, and around every bound pad of another net its hole (tracks
and via copper at the rule value from the wall) and its copper (a via's
drill at the rule value from it). The same cells are removed on exit, so
refcounts balance. `net_keepouts.stamp` calls it, which puts it at every
route entry the net-filtered keep-outs reach; the plane-tap via map
(`plane_obstacle_builder.build_via_obstacle_map`) stamps it for the tapped
net. A rule a later typed rule relaxes for some items is over-applied there;
nothing is under-applied by the stamp itself. KiCad's DRC of the rule stays
the authority.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

#: Condition atoms a net-pair rule may use: facts of the item's NET.
_NET_PROPS = {'netname', 'netclass', 'net'}
_NET_CALLS = {'hasnetclass', 'hasexactnetclass', 'indiffpair'}


class _NotNetScoped(Exception):
    pass


def _atoms(node, out: list) -> list:
    """The leaves of a condition AST under and/or/not, each a (node, item)."""
    op = node[0]
    if op in ('and', 'or'):
        _atoms(node[1], out)
        _atoms(node[2], out)
    elif op == 'not':
        _atoms(node[1], out)
    elif op == 'cmp':
        items = {x[1] for x in node[2:] if x[0] in ('prop', 'call')}
        props = [x for x in node[2:] if x[0] == 'prop']
        if len(items) != 1 or any(p[2] not in _NET_PROPS for p in props) or \
                any(x[0] == 'call' and x[2] not in _NET_CALLS for x in node[2:]):
            raise _NotNetScoped
        out.append((node, items.pop()))
    elif op == 'call':
        if node[2] not in _NET_CALLS or node[1] not in ('A', 'B'):
            raise _NotNetScoped
        out.append((node, node[1]))
    else:
        raise _NotNetScoped
    return out


def _uses_b(node) -> bool:
    if not isinstance(node, tuple):
        return False
    if node[0] in ('prop', 'call') and node[1] in ('B', 'AB'):
        return True
    return any(_uses_b(c) for c in node[1:] if isinstance(c, tuple))


def _min_of(rule, kind) -> Optional[float]:
    spec = rule.constraints.get(kind) or {}
    v = spec.get('min')
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def derive(rules, pcb_data) -> List[dict]:
    """The board's net-pair and pad hole-clearance entries (see the module
    docstring), marking `rule.honoured` on each rule they carry. [] when the
    board declares neither."""
    from design_rules import Unsupported, _eval, _rule_matches
    if rules is None or pcb_data is None:
        return []
    nets = {nid: n.name for nid, n in (pcb_data.nets or {}).items()
            if nid and getattr(n, 'name', None)}
    copper = list(getattr(pcb_data.board_info, 'copper_layers', None)
                  or rules.copper_layers or [])
    ctx = ('clearance', frozenset(nets.values()))
    out: List[dict] = []

    # ---- net-pair clearance rules ----
    pair_rules = []
    for rule in rules.rules:
        v = _min_of(rule, 'clearance')
        if (v is None or rule.unsupported or rule.condition is None
                or rule.layer_clause is not None or rule.coupled or rule.keep_apart):
            continue
        try:
            atoms = _atoms(rule.condition, [])
        except _NotNetScoped:
            continue
        pair_rules.append((rule, v, atoms))
    if pair_rules and nets:
        items = {nid: rules.item_for_net(nid, 'track') for nid in nets}
        all_atoms = [a for _r, _v, atoms in pair_rules for a in atoms]
        sig_of: Dict[int, tuple] = {}
        try:
            for nid, it in items.items():
                sig_of[nid] = tuple(bool(_eval(node, it, it, None, ctx))
                                    for node, _x in all_atoms)
        except Unsupported:
            sig_of = {}
        if sig_of:
            groups: Dict[tuple, int] = {}
            group_of: Dict[str, int] = {}
            reps: List[int] = []
            for nid in sorted(sig_of):
                g = groups.setdefault(sig_of[nid], len(groups))
                if g == len(reps):
                    reps.append(nid)
                group_of[nets[nid]] = g
            value: Dict[Tuple[int, int], float] = {}
            names: Dict[Tuple[int, int], str] = {}
            used = set()
            for gi, a in enumerate(reps):
                for gj, b in enumerate(reps):
                    for rule, v, _atoms_ in pair_rules:      # file order: last wins
                        if _rule_matches(rule, items[a], items[b], None, copper, ctx):
                            value[(gi, gj)] = v
                            names[(gi, gj)] = rule.name
                    if (gi, gj) in names:
                        used.add(names[(gi, gj)])
            for rule, v, _atoms_ in pair_rules:
                if rule.name not in used:
                    continue
                bound = {n for n, g in group_of.items()
                         if any(value.get((g, h)) and names.get((g, h)) == rule.name
                                for h in range(len(reps)))}
                rule.honoured = (f"honoured per routed net: {v:g} mm between the "
                                 f"nets it pairs ({len(bound)} of {len(nets)} net(s) "
                                 f"in a bound pair), tracks and vias, and plane-tap vias")
            if value:
                out.append({'kind': 'net_pair', 'group_of': group_of,
                            'value': value, 'rules': sorted(used)})

    # ---- pad-scoped hole_clearance rules ----
    pads = [p for fp in (pcb_data.footprints or {}).values() for p in fp.pads
            if any(str(l).endswith('.Cu') for l in (p.layers or ()))
            or (getattr(p, 'drill', 0) or 0) > 0]
    for rule in rules.rules:
        c = _min_of(rule, 'hole_clearance')
        if c is None or rule.unsupported or rule.layer_clause is not None:
            continue
        if rule.condition is None or _uses_b(rule.condition):
            rule.unsupported = ("hole_clearance with a condition on B or none at "
                                "all is not modelled by the router")
            continue
        try:
            probes = [rules.item_for_net(0, 'track'),
                      rules.item_for_net(0, 'via', via_type='through')]
            if any(_eval(rule.condition, p, None, None, ctx) for p in probes):
                raise Unsupported("it binds tracks or vias too")
            bound = [p for p in pads
                     if _eval(rule.condition, rules.item_for_pad(p), None, None, ctx)]
        except Unsupported as e:
            rule.unsupported = f"hole_clearance for the router: {e}"
            continue
        if not bound:
            rule.honoured = "binds no pad on this board"
            continue
        out.append({'kind': 'pad_hole', 'rule': rule.name, 'clearance': c,
                    'pads': [(p.component_ref, p.pad_number, round(p.global_x, 6),
                              round(p.global_y, 6)) for p in bound]})
        rule.honoured = (f"honoured around {len(bound)} pad(s): tracks and via "
                         f"copper {c:g} mm off their holes, via drills {c:g} mm off "
                         f"their copper, for every other net")
    return out


# --------------------------------------------------------------------------
# Stamping
# --------------------------------------------------------------------------

class _Recorder:
    """Captures the track and via cells an obstacle stamp adds."""

    def __init__(self):
        self.cells: List[np.ndarray] = []
        self.vias: List[np.ndarray] = []

    def add_blocked_cells_batch(self, arr):
        a = np.asarray(arr, dtype=np.int32).reshape(-1, 3)
        if len(a):
            self.cells.append(a.copy())

    def add_blocked_cell(self, gx, gy, layer):
        self.cells.append(np.array([[gx, gy, layer]], dtype=np.int32))

    def add_blocked_vias_batch(self, arr):
        a = np.asarray(arr, dtype=np.int32).reshape(-1, 2)
        if len(a):
            self.vias.append(a.copy())

    def add_blocked_via(self, gx, gy):
        self.vias.append(np.array([[gx, gy]], dtype=np.int32))

    def add_blocked_via_spans_batch(self, spans):
        from obstacle_cache import expand_via_spans
        a = np.asarray(spans, dtype=np.int32).reshape(-1, 3)
        if len(a):
            self.vias.append(np.asarray(expand_via_spans(a), dtype=np.int32).reshape(-1, 2))

    def add_blocked_vias_small_batch(self, arr):
        pass                    # stamp() mirrors the full-size via cells itself

    def arrays(self):
        # Not de-duplicated: a row stamped twice is removed twice, so the
        # refcounts balance, and sorting the rows cost more than the stamp.
        cells = (np.ascontiguousarray(np.concatenate(self.cells)) if self.cells
                 else np.empty((0, 3), dtype=np.int32))
        vias = (np.ascontiguousarray(np.concatenate(self.vias)) if self.vias
                else np.empty((0, 2), dtype=np.int32))
        return cells, vias


def _rule_config(config, v):
    """`config` pricing every pair at exactly `v` (raise-only above the map)."""
    return replace(config, clearance=v, net_clearances={}, net_clearance_floor=v,
                   layer_clearances={}, track_clearances={},
                   track_partner_clearances={}, coupled_pair_clearances={})


def _pad_lookup(pcb_data):
    cache = pcb_data.__dict__.get('_dru_copper_pads')
    if cache is None or cache[0] is not pcb_data.footprints:
        idx = {}
        for fp in (pcb_data.footprints or {}).values():
            for p in fp.pads:
                idx[(p.component_ref, p.pad_number, round(p.global_x, 6),
                     round(p.global_y, 6))] = p
        cache = (pcb_data.footprints, idx)
        pcb_data._dru_copper_pads = cache
    return cache[1]


def _stamper(pcb_data, entries, net_ids, extra):
    """`stamp(sink, config)`: every cell the entries add for a route of
    `net_ids`, at `config`'s geometry, plus `extra` (a diff-pair centreline)."""
    from obstacle_map import (_add_pad_obstacle, add_segments_list_as_obstacles,
                              add_vias_list_as_obstacles, build_layer_map)
    from routing_config import GridCoord
    routed = {nid for nid in net_ids if nid}
    names = {nid: n.name for nid, n in pcb_data.nets.items() if nid}
    id_of = {nm: nid for nid, nm in names.items() if nm}

    def run(sink, config):
        coord = GridCoord(config.grid_step)
        layer_map = build_layer_map(config.layers)
        all_layers = list(range(len(config.layers)))
        for e in entries:
            if e['kind'] == 'net_pair':
                gof, val = e['group_of'], e['value']
                rg = {gof[names[r]] for r in routed if names.get(r) in gof}
                if not rg:
                    continue
                by_v: Dict[float, List[int]] = {}
                for nm, g in gof.items():
                    o = id_of.get(nm)
                    if o is None or o in routed:
                        continue
                    v = max((val.get((r, g), 0.0) for r in rg), default=0.0)
                    if v <= 0:
                        continue
                    base = max(config.pair_clearance(r, o) for r in routed)
                    if v > base + 1e-9:
                        by_v.setdefault(v, []).append(o)
                if not by_v:
                    continue
                segs_of: Dict[int, list] = {}
                for s in pcb_data.segments:
                    segs_of.setdefault(s.net_id, []).append(s)
                vias_of: Dict[int, list] = {}
                for vv in pcb_data.vias:
                    vias_of.setdefault(vv.net_id, []).append(vv)
                for v, others in by_v.items():
                    cfg = _rule_config(config, v)
                    segs = [s for o in others for s in segs_of.get(o, ())]
                    vias = [x for o in others for x in vias_of.get(o, ())]
                    if segs:
                        add_segments_list_as_obstacles(sink, segs, cfg, extra)
                    if vias:
                        add_vias_list_as_obstacles(sink, vias, cfg, extra)
                    for o in others:
                        for p in pcb_data.pads_by_net.get(o, ()):
                            _add_pad_obstacle(sink, p, coord, layer_map, cfg, extra,
                                              clearance_override=v)
            elif e['kind'] == 'pad_hole':
                lookup = _pad_lookup(pcb_data)
                pads = [lookup.get(tuple(k)) for k in e['pads']]
                pads = [p for p in pads if p is not None
                        and not (p.net_id and p.net_id in routed)]
                # Static copper: one rasterisation per (pad set, geometry).
                ckey = (id(e), tuple(id(p) for p in pads), config.grid_step,
                        config.track_width, config.via_size, config.via_drill,
                        config.clearance, tuple(config.layers), extra)
                cache = pcb_data.__dict__.setdefault('_dru_copper_hole_cells', {})
                if ckey not in cache:
                    rec = _Recorder()
                    _pad_hole_cells(rec, pads, e['clearance'], config, coord,
                                    layer_map, all_layers, extra)
                    # Only what the pads' own stamps (at the clearance the
                    # map prices them) leave open: most of a ring lies under
                    # its pad's copper band, and re-adding it every route
                    # cost more than the route.
                    own = _Recorder()
                    for p in pads:
                        _add_pad_obstacle(own, p, coord, layer_map, config, extra,
                                          clearance_override=config.obstacle_clearance(
                                              p.net_id))
                    cells, vias = rec.arrays()
                    ocells, ovias = own.arrays()
                    cache[ckey] = (_minus(cells, ocells), _minus(vias, ovias))
                cells, vias = cache[ckey]
                if len(cells):
                    sink.add_blocked_cells_batch(cells)
                if len(vias):
                    sink.add_blocked_vias_batch(vias)
    return run


def _minus(rows, drop):
    """The distinct rows of `rows` not in `drop` (int32 cell rows)."""
    if not len(rows):
        return rows
    rows = np.unique(rows, axis=0)
    if not len(drop):
        return rows
    def key(a):
        a = a.astype(np.int64)
        k = a[:, 0] * 4000003 + a[:, 1]
        return k * 64 + a[:, 2] if a.shape[1] == 3 else k
    return np.ascontiguousarray(rows[~np.isin(key(rows), key(drop))])


def _pad_hole_cells(sink, pads, c, config, coord, layer_map, all_layers, extra):
    """The cells a pad hole-clearance rule `c` adds around `pads`: tracks and
    via copper `c` off each hole wall, and a via's drill `c` off each pad's
    copper (its copper `c` - ring)."""
    from obstacle_map import (_add_pad_obstacle, block_track_cells_near_drills,
                              block_via_cells_near_drills)
    from kicad_parser import pad_drill_circles
    ring = (config.via_size - config.via_drill) / 2.0
    holes = [h for p in pads if (getattr(p, 'drill', 0) or 0) > 0
             for h in pad_drill_circles(p)]
    if holes:
        block_track_cells_near_drills(sink, holes, config.track_width,
                                      c + extra, config.grid_step, all_layers)
        block_via_cells_near_drills(sink, holes, config.via_drill,
                                    c + ring + extra, config.grid_step)
    if pads and c - ring > config.clearance:
        cfg = _rule_config(config, c - ring)
        drop: Dict[int, list] = {}      # the pad's track cells: not this rule's
        vsink: List[np.ndarray] = []
        for p in pads:
            _add_pad_obstacle(sink, p, coord, layer_map, cfg, extra,
                              clearance_override=c - ring,
                              cell_sink=drop, via_sink=vsink)
        for a in vsink:
            sink.add_blocked_vias_batch(a)


def entries_of(pcb_data) -> List[dict]:
    return getattr(pcb_data, '_dru_copper_rules', None) or []


def cells_for(pcb_data, net_ids: Iterable[int], config, extra: float = 0.0):
    """(track cells (N, 3), via cells (M, 2)) the entries add for a route of
    `net_ids`. Empty arrays when nothing binds."""
    entries = entries_of(pcb_data)
    rec = _Recorder()
    if entries:
        _stamper(pcb_data, entries, tuple(net_ids), extra)(rec, config)
    return rec.arrays()


@contextlib.contextmanager
def stamp(obstacles, pcb_data, net_ids: Iterable[int], config, extra: float = 0.0):
    """Block what the board's net-pair and pad hole-clearance rules require of
    a route of `net_ids` in `obstacles` for the duration of the block, then
    remove exactly what was added. Via cells reach every via map, as
    `net_keepouts.stamp` does: rung 0, the #568 small rung and each #530
    per-net rung at its own via."""
    entries = entries_of(pcb_data)
    if obstacles is None or not entries:
        yield
        return
    net_ids = tuple(net_ids)
    if extra:
        # a diff-pair centreline map: price at the members' own clearance
        # and add the centreline offset to every band instead
        config = replace(config, clearance=config.clearance - extra)
    stamper = _stamper(pcb_data, entries, net_ids, extra)
    rec = _Recorder()
    stamper(rec, config)
    cells, vias = rec.arrays()
    from obstacle_map import _rung_small_armed, _stamp_rungs_at_own_via
    small = _rung_small_armed() and hasattr(obstacles, 'add_blocked_vias_small_batch')
    rungs = len(vias) and hasattr(obstacles, 'rung_count')
    if len(cells):
        obstacles.add_blocked_cells_batch(cells)
    if len(vias):
        obstacles.add_blocked_vias_batch(vias)
        if small:
            obstacles.add_blocked_vias_small_batch(vias)
    if rungs:
        _stamp_rungs_at_own_via(obstacles, config, stamper)
    try:
        yield
    finally:
        if len(cells):
            obstacles.remove_blocked_cells_batch(cells)
        if len(vias):
            obstacles.remove_blocked_vias_batch(vias)
            if small:
                obstacles.remove_blocked_vias_small_batch(vias)
        if rungs:
            _stamp_rungs_at_own_via(obstacles, config, stamper, remove=True)


# --------------------------------------------------------------------------
# Admission test for passes that move copper outside the A* (the octolinear
# smoother): a candidate track must keep the same rules the stamp holds.
# --------------------------------------------------------------------------

def _bound_copper(pcb_data, net_id):
    """The copper a track of `net_id` is held off, as numpy rows:
    ({v: (segments (N,5) x1 y1 x2 y2 half-width, their layers, vias (M,3)
    x y radius, pads (K,3) x y bounding radius, their layers)}, holes (H,4)
    x y radius c). Cached per net on the copper counts it was built from."""
    entries = entries_of(pcb_data)
    key = (net_id, len(pcb_data.segments), len(pcb_data.vias))
    cache = pcb_data.__dict__.setdefault('_dru_copper_bound', {})
    hit = cache.get(net_id)
    if hit is not None and hit[0] == key:
        return hit[1]
    names = {nid: n.name for nid, n in pcb_data.nets.items() if nid}
    id_of = {nm: nid for nid, nm in names.items() if nm}
    by_v: Dict[float, set] = {}
    holes = []
    for e in entries:
        if e['kind'] == 'net_pair':
            gof, val = e['group_of'], e['value']
            g = gof.get(names.get(net_id))
            if g is None:
                continue
            for nm, h in gof.items():
                o = id_of.get(nm)
                v = val.get((g, h))
                if o is not None and o != net_id and v:
                    by_v.setdefault(v, set()).add(o)
        elif e['kind'] == 'pad_hole':
            lookup = _pad_lookup(pcb_data)
            from kicad_parser import pad_drill_circles
            for k in e['pads']:
                p = lookup.get(tuple(k))
                if p is None or p.net_id == net_id or not (getattr(p, 'drill', 0) or 0):
                    continue
                for hx, hy, dd in pad_drill_circles(p):
                    holes.append((hx, hy, dd / 2.0, e['clearance']))
    groups = {}
    for v, others in by_v.items():
        segs = [s for s in pcb_data.segments if s.net_id in others]
        vias = [x for x in pcb_data.vias if x.net_id in others]
        pads = [p for o in others for p in pcb_data.pads_by_net.get(o, ())]
        groups[v] = (
            np.array([[s.start_x, s.start_y, s.end_x, s.end_y, s.width / 2.0]
                      for s in segs], dtype=float).reshape(-1, 5),
            [s.layer for s in segs],
            np.array([[x.x, x.y, x.size / 2.0] for x in vias], dtype=float).reshape(-1, 3),
            np.array([[p.global_x, p.global_y, math.hypot(p.size_x, p.size_y) / 2.0]
                      for p in pads], dtype=float).reshape(-1, 3),
            [tuple(p.layers or ()) for p in pads])
    out = (groups, np.array(holes, dtype=float).reshape(-1, 4))
    cache[net_id] = (key, out)
    return out


def _pt_seg(px, py, x1, y1, x2, y2):
    """Distance of points (px, py) to segments, vectorised over numpy arrays."""
    dx, dy = x2 - x1, y2 - y1
    L = dx * dx + dy * dy
    t = np.where(L > 0, ((px - x1) * dx + (py - y1) * dy) / np.where(L > 0, L, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def track_clears(pcb_data, net_id, x1, y1, x2, y2, layer, width) -> bool:
    """Whether a track of `net_id` from (x1, y1) to (x2, y2) on `layer`, of
    `width`, keeps the board's net-pair and pad hole-clearance rules against
    the copper `pcb_data` holds. True when nothing binds. Pads are taken as
    their bounding circle, which can refuse a legal track, never admit an
    illegal one."""
    if not entries_of(pcb_data) or not net_id:
        return True
    groups, holes = _bound_copper(pcb_data, net_id)
    hw = width / 2.0
    eps = 1e-4
    for v, (segs, seg_layers, vias, pads, pad_layers) in groups.items():
        if len(segs):
            on = np.array([l == layer for l in seg_layers])
            if on.any():
                s = segs[on]
                d = np.minimum.reduce([
                    _pt_seg(x1, y1, s[:, 0], s[:, 1], s[:, 2], s[:, 3]),
                    _pt_seg(x2, y2, s[:, 0], s[:, 1], s[:, 2], s[:, 3]),
                    _pt_seg(s[:, 0], s[:, 1], x1, y1, x2, y2),
                    _pt_seg(s[:, 2], s[:, 3], x1, y1, x2, y2)])
                if (d - s[:, 4] < v + hw - eps).any() or _crosses(x1, y1, x2, y2, s):
                    return False
        if len(vias):
            d = _pt_seg(vias[:, 0], vias[:, 1], x1, y1, x2, y2) - vias[:, 2]
            if (d < v + hw - eps).any():
                return False
        if len(pads):
            on = np.array([layer in ls or '*.Cu' in ls
                           or (layer in ('F.Cu', 'B.Cu') and 'F&B.Cu' in ls)
                           for ls in pad_layers])
            if on.any():
                p = pads[on]
                d = _pt_seg(p[:, 0], p[:, 1], x1, y1, x2, y2) - p[:, 2]
                if (d < v + hw - eps).any():
                    return False
    if len(holes):
        d = _pt_seg(holes[:, 0], holes[:, 1], x1, y1, x2, y2) - holes[:, 2]
        if (d < holes[:, 3] + hw - eps).any():
            return False
    return True


def _crosses(x1, y1, x2, y2, s) -> bool:
    """Whether the segment crosses any of segments `s` (N, >=4)."""
    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    o1 = orient(x1, y1, x2, y2, s[:, 0], s[:, 1])
    o2 = orient(x1, y1, x2, y2, s[:, 2], s[:, 3])
    o3 = orient(s[:, 0], s[:, 1], s[:, 2], s[:, 3], x1, y1)
    o4 = orient(s[:, 0], s[:, 1], s[:, 2], s[:, 3], x2, y2)
    return bool(((o1 * o2 < 0) & (o3 * o4 < 0)).any())
