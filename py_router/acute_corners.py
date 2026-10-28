"""Acute corners between same-net tracks: the measure and the repair (#1233).

Two arms of one net meeting at a joint -- two track ends at one point, or a
track end on another track's centreline (a T) -- at less than 90 degrees
leave a narrow wedge of bare laminate between them. That wedge is an acid
trap: etchant pools in it and over-etches the joint. Nothing in the router
looked at the angle, and three things laid such corners: the grid search (a
branch leaving a trunk at 45 degrees), the off-grid stub from the last grid
point to a pad centre (which folds back when that grid point is already past
the centre), and copper laid after every cleanup (plane taps, the oracle).

``acute_corners`` is the measure; ``chamfer_acute_corners`` the repair. The
repair moves one arm B off the joint V and lands it on the other arm A
through a segment PERPENDICULAR to A: the corners it leaves are 90 degrees on
A and 90 + theta on B, and octolinear copper stays octolinear. The landing
point b on B is the first of

  * the point on B's first segment far enough from V that the cut swallows
    the wedge's inner corner (``_cut_depth``), and
  * each vertex further along B's free chain (a staircase step ending near
    V is cut off whole),

whose cut clears foreign copper (``make_segment_clears``, the octolinear
smoother's predicate), strands nothing and leaves no acute corner at V, a or
b. Each repair strictly lowers the net's acute count, so the pass ends.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Optional

from kicad_parser import PCBData, Segment, is_immutable_copper

#: Two arms enclosing less than this (degrees) make an acute corner. Half a
#: degree under 90, so a right angle off by grid rounding is not one.
ACUTE_MAX_DEG = 89.5
#: Arms closer than this (degrees) run along each other: overlapping copper
#: (check_weird's stacked-copper class), not a corner -- no cut changes it.
ACUTE_MIN_DEG = 1.0
#: Two track ends within this (mm) are one joint.
_JOINT_TOL = 1e-4
#: A track end within this (mm) of another track's centreline, away from its
#: ends, is a T on it.
_ON_LINE = 1e-3
#: No new piece shorter than this (mm).
_MIN_PIECE = 1e-3
#: Spatial-index cell (mm).
_CELL = 1.0
#: How far along B's free chain a landing point is looked for.
_MAX_CHAIN_MM = 2.0
_MAX_CHAIN_SEGS = 8


def _length(s):
    return math.hypot(s.end_x - s.start_x, s.end_y - s.start_y)


def _angle(ax, ay, bx, by):
    """Angle (degrees) between two vectors."""
    n = math.hypot(ax, ay) * math.hypot(bx, by)
    if n <= 0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / n))))


class CornerIndex:
    """One net's tracks by layer and grid cell, for the joint queries."""

    def __init__(self, segs, cell: float = _CELL):
        self.cell = cell
        # A cell keeps its tracks in insertion order (a dict, not a set of
        # ids): a set iterates in address order, which changes run to run, and
        # near() feeds arms_at, whose order breaks ties between equal angles.
        self.cells = defaultdict(dict)
        self.segs = {}
        for s in segs:
            self.add(s)

    def _keys(self, s):
        c = self.cell
        x0, x1 = sorted((s.start_x, s.end_x))
        y0, y1 = sorted((s.start_y, s.end_y))
        for i in range(math.floor((x0 - _ON_LINE) / c), math.floor((x1 + _ON_LINE) / c) + 1):
            for j in range(math.floor((y0 - _ON_LINE) / c), math.floor((y1 + _ON_LINE) / c) + 1):
                yield (s.layer, i, j)

    def add(self, s):
        self.segs[id(s)] = s
        for k in self._keys(s):
            self.cells[k][id(s)] = None

    def remove(self, s):
        self.segs.pop(id(s), None)
        for k in self._keys(s):
            self.cells[k].pop(id(s), None)

    def near(self, x, y, layer):
        c = self.cell
        return [self.segs[i] for i in
                self.cells.get((layer, math.floor(x / c), math.floor(y / c)), ())]

    def arms_at(self, x, y, layer):
        """The arms leaving joint (x, y): ``(segment, far_x, far_y, is_t)``.
        A track ending there gives one arm; a track whose body passes through
        it (a T host) gives two, one per half."""
        arms = []
        for s in self.near(x, y, layer):
            L = _length(s)
            if L <= _JOINT_TOL:
                continue
            if math.hypot(s.start_x - x, s.start_y - y) <= _JOINT_TOL:
                arms.append((s, s.end_x, s.end_y, False))
            elif math.hypot(s.end_x - x, s.end_y - y) <= _JOINT_TOL:
                arms.append((s, s.start_x, s.start_y, False))
            else:
                dx, dy = (s.end_x - s.start_x) / L, (s.end_y - s.start_y) / L
                t = (x - s.start_x) * dx + (y - s.start_y) * dy
                if (_ON_LINE < t < L - _ON_LINE
                        and abs((x - s.start_x) * dy - (y - s.start_y) * dx) <= _ON_LINE):
                    arms.append((s, s.start_x, s.start_y, True))
                    arms.append((s, s.end_x, s.end_y, True))
        return arms

    def acute_pairs_at(self, x, y, layer, arms=None):
        """[(angle, arm1, arm2)] for the acute corners at joint (x, y)."""
        arms = self.arms_at(x, y, layer) if arms is None else arms
        out = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                if a[0] is b[0]:
                    continue
                ang = _angle(a[1] - x, a[2] - y, b[1] - x, b[2] - y)
                if ACUTE_MIN_DEG <= ang < ACUTE_MAX_DEG:
                    out.append((ang, a, b))
        return out

    def joints(self):
        """Every track end, once per (layer, point)."""
        seen = {}
        for s in self.segs.values():
            for x, y in ((s.start_x, s.start_y), (s.end_x, s.end_y)):
                seen.setdefault((s.layer, round(x, 4), round(y, 4)), (x, y))
        return [(layer, x, y) for (layer, _rx, _ry), (x, y) in seen.items()]


def acute_corners(segments):
    """Every acute corner among ``segments`` (same-net tracks are judged
    together, nets apart): ``[(net_id, layer, x, y, angle_deg)]``, one per
    pair of arms. Graphic copper (#337) is not a track and is skipped."""
    by_net = defaultdict(list)
    for s in segments:
        if not getattr(s, 'graphic', False):
            by_net[s.net_id].append(s)
    out = []
    for net_id, segs in by_net.items():
        idx = CornerIndex(segs)
        for layer, x, y in idx.joints():
            for ang, _a, _b in idx.acute_pairs_at(x, y, layer):
                out.append((net_id, layer, x, y, ang))
    return out


def _cut_depth(theta_deg, width):
    """Distance from V along B at which a cut perpendicular to A swallows the
    wedge's inner corner, for two arms of ``width`` enclosing ``theta``.

    The inner edges of the two arms cross on the bisector at distance
    (w/2) / tan(theta/2) from V, measured along A. A cut landing on B at d
    sits at d cos(theta) along A and its copper reaches w/2 past that, so
    it covers the crossing from d = (x_C - w/2) / cos(theta)."""
    th = math.radians(theta_deg)
    x_c = (width / 2.0) / math.tan(th / 2.0)
    return max(0.0, (x_c - width / 2.0) / math.cos(th)) + _MIN_PIECE


def _seg(proto, x1, y1, x2, y2, width=None):
    s = copy.copy(proto)
    s.start_x, s.start_y, s.end_x, s.end_y = x1, y1, x2, y2
    s.start_x_str = s.start_y_str = s.end_x_str = s.end_y_str = ""
    s.uuid = ""
    s.locked = False
    if width is not None:
        s.width = width
    return s


def chamfer_acute_corners(pcb_data: PCBData, scope_net_ids=None, clears=None,
                          readonly_ids=frozenset(), skip_net_ids=frozenset(),
                          stats: Optional[dict] = None):
    """Cut every acute corner the repair can clear (see the module docstring).

    ``clears(x1, y1, x2, y2, layer, net_id, width) -> bool`` judges a cut
    against foreign copper (``pcb_modification.make_segment_clears``). Tracks
    that are graphic copper (#337), locked or sense-held
    (is_immutable_copper), or in ``readonly_ids`` (object ids) are never
    changed, and neither are the nets in ``skip_net_ids``. A net whose
    connectivity (check_net_connectivity, no pour credit) would grade worse
    is put back whole.

    Mutates ``pcb_data.segments`` in place. Returns ``[(segment, pieces)]``
    in the shape of split_kicad_dangling_hosts: each segment that left the
    board with the segments that replaced it (the cuts ride with their arm
    B; a piece changed again is expanded, so ``pieces`` is what the board
    now holds). ``stats`` receives ``{'cut', 'nets', 'left'}``, ``left``
    counting the acute corners no cut could clear.
    """
    from check_connected import check_net_connectivity, net_copper_fragments
    from check_drc import point_to_pad_distance
    from geometry_utils import point_to_segment_distance, via_copper_layers

    st = {'cut': 0, 'nets': 0, 'left': 0}
    readonly = set(readonly_ids or ())
    skip = set(skip_net_ids or ())
    by_net = defaultdict(list)
    for s in pcb_data.segments:
        if s.net_id and (scope_net_ids is None or s.net_id in scope_net_ids):
            by_net[s.net_id].append(s)
    vias_by_net = defaultdict(list)
    for v in pcb_data.vias:
        vias_by_net[v.net_id].append(v)

    copper_layers = list(getattr(getattr(pcb_data, 'board_info', None),
                                 'copper_layers', None) or ['F.Cu', 'B.Cu'])
    pairs = []

    def grade(net_id, segs, vias, pads):
        c = check_net_connectivity(net_id, segs, vias, pads, [], pcb_data=pcb_data)
        f = net_copper_fragments(net_id, segs, vias, pads, [], pcb_data=pcb_data)
        return (0 if c.get('connected') else 1,
                len(c.get('disconnected_pads') or []),
                c.get('num_components') or 1, f['fragments'])

    def worse(before, after):
        return any(a > b for a, b in zip(after, before))

    def fixed(s):
        return (id(s) in readonly or getattr(s, 'graphic', False)
                or is_immutable_copper(s))

    for net_id in sorted(by_net):
        if net_id in skip:
            continue
        segs = [s for s in by_net[net_id] if not getattr(s, 'graphic', False)]
        if len(segs) < 2:
            continue
        idx = CornerIndex(segs)
        corners = []
        for layer, x, y in idx.joints():
            p = idx.acute_pairs_at(x, y, layer)
            if p:
                corners.append((min(a for a, _1, _2 in p), layer, x, y))
        if not corners:
            continue
        corners.sort()
        pads = pcb_data.pads_by_net.get(net_id, [])
        vias = vias_by_net.get(net_id, [])

        def on_layer(pad, layer):
            return layer in pad.layers or any('*' in L for L in pad.layers)

        def via_on(v, layer):
            return layer in via_copper_layers(v, copper_layers)

        def anchored(x, y, layer):
            """A via or pad of the net holds this point on ``layer``."""
            for v in vias:
                if math.hypot(v.x - x, v.y - y) <= v.size / 2.0 and via_on(v, layer):
                    return True
            return any(on_layer(p, layer) and point_to_pad_distance(x, y, p) <= 0
                       for p in pads)

        def samples(s):
            L = _length(s)
            n = max(1, int(L / 0.05))
            return [(s.start_x + q / n * (s.end_x - s.start_x),
                     s.start_y + q / n * (s.end_y - s.start_y)) for q in range(n + 1)]

        def touches_via(s, v):
            return (point_to_segment_distance(v.x, v.y, s.start_x, s.start_y,
                                              s.end_x, s.end_y)
                    <= v.size / 2.0 + s.width / 2.0)

        def touches_pad(s, p):
            return any(point_to_pad_distance(px, py, p) <= s.width / 2.0
                       for px, py in samples(s))

        def holds(s, ex, ey):
            """Track ``s`` has (ex, ey) as an end or on its centreline."""
            return point_to_segment_distance(ex, ey, s.start_x, s.start_y,
                                              s.end_x, s.end_y) <= _ON_LINE

        def strands(removed, added, layer):
            """The edit leaves a via, a pad or another track's end that the
            removed copper held with nothing kept holding it."""
            gone_ids = {id(s) for s in removed}

            def kept_near(x, y):
                return [s for s in idx.near(x, y, layer) if id(s) not in gone_ids] + added

            for v in vias:
                if not via_on(v, layer):
                    continue
                if any(touches_via(s, v) for s in removed) and not any(
                        touches_via(s, v) for s in kept_near(v.x, v.y)):
                    return True
            for p in pads:
                r = math.hypot(p.size_x, p.size_y) / 2.0
                if not on_layer(p, layer) or all(
                        point_to_segment_distance(p.global_x, p.global_y, s.start_x,
                                                  s.start_y, s.end_x, s.end_y)
                        > r + s.width for s in removed):
                    continue
                if not any(touches_pad(s, p) for s in removed):
                    continue
                near = {id(s): s for cx, cy in ((p.global_x, p.global_y),
                                                 (p.global_x - r, p.global_y - r),
                                                 (p.global_x + r, p.global_y - r),
                                                 (p.global_x - r, p.global_y + r),
                                                 (p.global_x + r, p.global_y + r))
                        for s in kept_near(cx, cy)}
                if not any(touches_pad(s, p) for s in near.values()):
                    return True
            for r in removed:
                cands = {id(c): c for px, py in samples(r)
                         for c in idx.near(px, py, layer) if id(c) not in gone_ids}
                for c in cands.values():
                    for ex, ey in ((c.start_x, c.start_y), (c.end_x, c.end_y)):
                        if holds(r, ex, ey) and not any(
                                holds(k, ex, ey) for k in kept_near(ex, ey) if k is not c):
                            return True
            return False

        def free_vertex(x, y, layer, arms):
            return (len(arms) == 2 and not any(a[3] for a in arms)
                    and not anchored(x, y, layer))

        def chain(arm, x, y, layer):
            """B's free chain from V: [(segment, from_xy, to_xy)]."""
            out = []
            s, fx, fy, _t = arm
            px, py, total = x, y, 0.0
            while True:
                out.append((s, (px, py), (fx, fy)))
                total += math.hypot(fx - px, fy - py)
                if total >= _MAX_CHAIN_MM or len(out) >= _MAX_CHAIN_SEGS:
                    return out
                arms = idx.arms_at(fx, fy, layer)
                if not free_vertex(fx, fy, layer, arms):
                    return out
                nxt = [a for a in arms if a[0] is not s]
                if not nxt or abs(nxt[0][0].width - s.width) > 1e-6 or fixed(nxt[0][0]):
                    return out
                px, py = fx, fy
                s, fx, fy, _t = nxt[0]

        def plan(A, B, x, y, layer, arms_v):
            """The edit landing B on A: ``(removed, [(piece, source)], cut)``,
            or None when no landing point passes."""
            if B[3] or fixed(B[0]) or fixed(A[0]):
                return None
            LA = math.hypot(A[1] - x, A[2] - y)
            if LA <= _MIN_PIECE:
                return None
            ux, uy = (A[1] - x) / LA, (A[2] - y) / LA
            ch = chain(B, x, y, layer)
            s0, _p0, (q0x, q0y) = ch[0]
            L0 = math.hypot(q0x - x, q0y - y)
            vx, vy = (q0x - x) / L0, (q0y - y) / L0
            w = max(A[0].width, B[0].width)
            wb = B[0].width
            if len(arms_v) == 2 and not A[3]:
                modes = (True, False) if anchored(x, y, layer) else (True,)
            else:
                modes = (False,)
            cands = []
            d = _cut_depth(_angle(ux, uy, vx, vy), w)
            if d < L0 - _MIN_PIECE:
                cands.append((x + d * vx, y + d * vy, 0, True))
            for k, (_s, _f, (tx, ty)) in enumerate(ch):
                cands.append((tx, ty, k, False))
            for bx, by, k, partial in cands:
                t = (bx - x) * ux + (by - y) * uy
                if t <= _MIN_PIECE or t > LA + 1e-9:
                    continue
                if LA - t < _MIN_PIECE:
                    ax, ay = A[1], A[2]
                else:
                    ax, ay = x + t * ux, y + t * uy
                cut_len = math.hypot(bx - ax, by - ay)
                if cut_len < _MIN_PIECE:
                    continue
                if not partial:
                    # Where B goes on past b, a cut thinner than the tracks
                    # leaves A's copper and the onward leg's crossing at
                    # their own angle: that is the same trap, further out.
                    onward = [o for o in idx.arms_at(bx, by, layer)
                              if o[0] is not ch[k][0]]
                    if cut_len < w and any(
                            _angle(ux, uy, o[1] - bx, o[2] - by) < ACUTE_MAX_DEG
                            for o in onward):
                        continue
                if partial:
                    base_rm = [s0]
                    base_add = [(_seg(s0, bx, by, q0x, q0y), s0)]
                else:
                    base_rm = [c[0] for c in ch[:k + 1]]
                    base_add = []
                cut = _seg(B[0], ax, ay, bx, by, width=wb)
                base_add.append((cut, B[0]))
                if clears is not None and not clears(ax, ay, bx, by, layer, net_id, wb):
                    continue
                Aseg = A[0]
                at_far = (ax, ay) == (A[1], A[2])
                # A bend (two arms) loses V..a too; a joint where more meet
                # keeps V..a for them, and so may a bend on a via or pad.
                for drop_va in modes:
                    removed, added = list(base_rm), list(base_add)
                    if A[3]:
                        # A T host stays whole, split at a.
                        removed.append(Aseg)
                        added.append((_seg(Aseg, Aseg.start_x, Aseg.start_y, ax, ay), Aseg))
                        added.append((_seg(Aseg, ax, ay, Aseg.end_x, Aseg.end_y), Aseg))
                    elif drop_va:
                        removed.append(Aseg)
                        if not at_far:
                            added.append((_seg(Aseg, ax, ay, A[1], A[2]), Aseg))
                    elif not at_far:
                        removed.append(Aseg)
                        added.append((_seg(Aseg, x, y, ax, ay), Aseg))
                        added.append((_seg(Aseg, ax, ay, A[1], A[2]), Aseg))
                    if strands(removed, [s for s, _src in added], layer):
                        continue
                    return removed, added, cut
            return None

        def count_at(points, layer):
            return sum(len(idx.acute_pairs_at(px, py, layer)) for px, py in points)

        origin = {}         # id(piece) -> the board segment it came from
        edits = []
        cuts = left = 0
        for _ang, layer, x, y in corners:
            while True:
                arms_v = idx.arms_at(x, y, layer)
                acute = sorted(idx.acute_pairs_at(x, y, layer, arms_v),
                               key=lambda p: p[0])
                if not acute:
                    break
                done = False
                for _a, p1, p2 in acute:
                    for A, B in ((p1, p2), (p2, p1)):
                        e = plan(A, B, x, y, layer, arms_v)
                        if e is None:
                            continue
                        removed, added, cut = e
                        pts = [(x, y), (cut.start_x, cut.start_y),
                               (cut.end_x, cut.end_y)]
                        before = count_at(pts, layer)
                        for s in removed:
                            idx.remove(s)
                        for s, _src in added:
                            idx.add(s)
                        new_ids = {id(s) for s, _src in added}
                        after = count_at(pts, layer)
                        touched = any(id(a[0]) in new_ids
                                      for px, py in pts[1:]
                                      for _g, a1, a2 in idx.acute_pairs_at(px, py, layer)
                                      for a in (a1, a2))
                        if after < before and not touched:
                            for s, src in added:
                                origin[id(s)] = origin.get(id(src), src)
                            edits.append((removed, [s for s, _src in added]))
                            cuts += 1
                            done = True
                            break
                        for s, _src in added:
                            idx.remove(s)
                        for s in removed:
                            idx.add(s)
                    if done:
                        break
                if not done:
                    left += len(acute)
                    break
        st['left'] += left
        if not cuts:
            continue
        # Every edit re-shapes copper between joints it keeps, under guards
        # against stranding; the net is graded anyway (connectivity, and the
        # strict copper fragments route.py's fragment sweep counts: a cut can
        # leave two tracks meeting only through a pad's copper). If it reads
        # worse, the edits are replayed one by one and only those that keep
        # it whole are kept.
        base = grade(net_id, segs, vias, pads)
        final = list(idx.segs.values())
        if worse(base, grade(net_id, final, vias, pads)):
            state, kept = list(segs), 0
            for removed, added in edits:
                ids = {id(x) for x in state}
                if not all(id(x) in ids for x in removed):
                    continue
                gone_ids = {id(x) for x in removed}
                trial = [x for x in state if id(x) not in gone_ids] + added
                if not worse(base, grade(net_id, trial, vias, pads)):
                    state, kept = trial, kept + 1
            st['left'] += cuts - kept
            cuts, final = kept, state
            if not cuts:
                continue
        st['cut'] += cuts
        st['nets'] += 1
        final_ids = {id(x) for x in final}
        seg_ids = {id(x) for x in segs}
        pieces = defaultdict(list)
        for x in final:
            if id(x) not in seg_ids:
                pieces[id(origin[id(x)])].append(x)
        for x in segs:
            if id(x) not in final_ids:
                pairs.append((x, pieces.get(id(x), [])))
        pcb_data.segments = ([x for x in pcb_data.segments
                              if x.net_id != net_id or id(x) in final_ids
                              or id(x) not in seg_ids]
                             + [x for x in final if id(x) not in seg_ids])
        if hasattr(pcb_data, '_foreign_seg_arr_cache'):
            pcb_data._foreign_seg_arr_cache = None
    if pairs:
        # A cut is added copper, which carves pours the cached fill models
        # do not know about (the octolinear smoother's rule).
        try:
            delattr(pcb_data, '_plane_fill_models')
        except AttributeError:
            pass
        try:
            import plane_fill_model as _pfm
            _pfm._MODELS_BY_ZONE_ID.clear()
        except Exception:
            pass
    if stats is not None:
        stats.update(st)
    return pairs
