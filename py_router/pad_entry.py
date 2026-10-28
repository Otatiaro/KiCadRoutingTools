"""Land a track on a row pad through one of its ENDS, along its long axis.

A pad of a fine-pitch row (QFP, QFN, SOIC, SSOP, an SMD header) is free
copper to its own net, so the A* reaches its centre from any side. On an
octile grid the cheapest way in is often down the gap between two pads of
the package, the centreline right beside the pad and its copper over the
pad's edge, then a short jog into the centre: DRC-clean and connected, but
a track squeezed between two pins of one package, entering the pin from its
flank. A pin is landed through its end.

`stamp` blocks, for the net being routed and for the duration of its route,
the FLANK cells of each of its row pads: along the pad's length, every
track-centreline cell off the grid column nearest the pad's axis, out to
where the net's copper would be clear of the pad's long side. The flanks run
on past each end by one track width plus clearance, so the track leaves the
pad straight along its axis before it turns (a 45-degree exit off the end
sweeps its copper across the gap at the neighbours' corners). The axis
column alone stays open over the pad as well: the cleanup drops the copper
inside a pad, so a jog the path took inside it onto the axis would be left
as the visible exit. The octolinear smoother refuses a shortcut across a
flank (`segment_crosses_flank`), which would otherwise cut the corner the
route was made to turn past the end. A plane pad tap blocks the same flanks
in its window's trace map (`block_window_flanks`) and keeps its new via off
them (`via_site_on_flank`).

The flanks never cost a connection. While stamped they are on record for
their map (`active`); a search that finds no way in through any end lifts
them (`lifted`) and routes as before, from a pad's side, saying so on the
console (single_ended_routing._route_with_via_unblock). A plane tap does the
same with its whole parameter ladder (plane_pad_tap.tap_pad_with_escalation),
and so does a KiCad-oracle link, started from the centre of the row pad it is
anchored on (`seeds_to_pad_centres`, `walled_copy`; kicad_oracle).

A differential pair's DIRECT HYBRID escape (diff_pair_routing) routes each
terminal leg single-ended to its coupled middle: the leg is walled the same
way, and lifted the same way when no end is free (_route_hybrid_leg). Its
coupled middle launches clear of the pair's row pads and their flanks
(`entry_rect`), beyond the ends: launched in the gap between the pair's
own two pads, the middle would leave its legs no way in but a flank.

A pad is a row pad when it is an SMD rectangle elongated along one axis
(length >= 1.5 x width) with another pad of its footprint BESIDE it --
across its long side, within one pad length. A two-pad passive's pads sit
end to end and are left alone, as is a lone or square pad, which has no
long side to enter from.
"""
import contextlib
import math
from typing import Iterable, List

import numpy as np

#: A pad is elongated when its length is at least this many widths.
ELONGATION = 1.5


def _axis(pad):
    """(half-width, half-length, long axis is x) of an axis-aligned pad."""
    hx, hy = pad.size_x / 2, pad.size_y / 2
    return (hy, hx, True) if hx >= hy else (hx, hy, False)


def is_row_pad(pad, footprint_pads) -> bool:
    """True when `pad` is an elongated SMD pad with a pad of its own footprint
    beside it, across its long side (see the module docstring)."""
    if getattr(pad, 'drill', 0) or getattr(pad, 'polygons', None):
        return False
    if getattr(pad, 'rect_rotation', 0.0):
        return False
    hw, hl, along_x = _axis(pad)
    if hw <= 0 or hl < ELONGATION * hw:
        return False
    for other in footprint_pads:
        if other is pad:
            continue
        dx, dy = other.global_x - pad.global_x, other.global_y - pad.global_y
        d_along, d_lat = (dx, dy) if along_x else (dy, dx)
        if abs(d_along) < hl and 0 < abs(d_lat) <= 2 * hl:
            return True
    return False


def row_pads(pcb_data, net_id: int) -> List:
    """The net's row pads."""
    out = []
    for pad in pcb_data.pads_by_net.get(net_id, []):
        fp = pcb_data.footprints.get(pad.component_ref)
        if fp is not None and is_row_pad(pad, fp.pads):
            out.append(pad)
    return out


def flank_cells(pad, layer_indices: Iterable[int], config) -> np.ndarray:
    """(N, 3) track cells of `pad`'s two flanks on `layer_indices`.

    A centreline cell is a flank cell when it lies along the pad's length,
    extended past each end by the axial stub (|along| <= half-length +
    track + clearance), off the grid column nearest the axis
    (|lat| > half a grid step), and not yet clear of the pad's long side
    (|lat| <= half-width + track/2 + clearance)."""
    step = config.grid_step
    tw = config.track_width
    hw, hl, along_x = _axis(pad)
    column = step / 2 + 1e-6
    outer = hw + tw / 2 + config.clearance
    reach = hl + tw + config.clearance
    cx, cy = pad.global_x, pad.global_y
    # Grid ranges covering the pad's length, its stubs and both flank bands.
    ext_x, ext_y = (reach, outer) if along_x else (outer, reach)
    gx0, gx1 = math.floor((cx - ext_x) / step), math.ceil((cx + ext_x) / step)
    gy0, gy1 = math.floor((cy - ext_y) / step), math.ceil((cy + ext_y) / step)
    gxs = np.arange(gx0, gx1 + 1)
    gys = np.arange(gy0, gy1 + 1)
    GX, GY = np.meshgrid(gxs, gys, indexing='ij')
    dx, dy = GX * step - cx, GY * step - cy
    d_along, d_lat = (dx, dy) if along_x else (dy, dx)
    lat = np.abs(d_lat)
    mask = (np.abs(d_along) <= reach + 1e-9) & (lat > column) & (lat <= outer + 1e-9)
    xy = np.stack([GX[mask], GY[mask]], axis=1).astype(np.int32)
    if not len(xy):
        return np.empty((0, 3), dtype=np.int32)
    rows = [np.hstack([xy, np.full((len(xy), 1), li, dtype=np.int32)])
            for li in layer_indices]
    return np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.int32)


def flank_rects(pad, config, track_width=None) -> List[tuple]:
    """`pad`'s two flanks as board-frame rectangles (x0, y0, x1, y1): the
    region `flank_cells` rasterises, for copper that is not on the grid.
    `track_width` overrides config.track_width (a cleanup config may carry
    only clearance and grid_step)."""
    tw = config.track_width if track_width is None else track_width
    hw, hl, along_x = _axis(pad)
    column = config.grid_step / 2 + 1e-6
    outer = hw + tw / 2 + config.clearance
    reach = hl + tw + config.clearance
    cx, cy = pad.global_x, pad.global_y
    out = []
    for lo, hi in ((column, outer), (-outer, -column)):
        if along_x:
            out.append((cx - reach, cy + lo, cx + reach, cy + hi))
        else:
            out.append((cx + lo, cy - reach, cx + hi, cy + reach))
    return out


def entry_rect(pad, config) -> tuple:
    """`pad` with its two flanks and the axial stub past each end, as one
    board-frame rectangle (x0, y0, x1, y1): the region a track of its net
    crosses only along the axis column."""
    tw = config.track_width
    hw, hl, along_x = _axis(pad)
    outer = hw + tw / 2 + config.clearance
    reach = hl + tw + config.clearance
    ex, ey = (reach, outer) if along_x else (outer, reach)
    return (pad.global_x - ex, pad.global_y - ey, pad.global_x + ex, pad.global_y + ey)


def net_flank_rects(pcb_data, net_id: int, config, track_width=None) -> List[tuple]:
    """(layers, rect) of every flank of `net_id`'s row pads."""
    return [(set(pad.layers), r) for pad in row_pads(pcb_data, net_id)
            for r in flank_rects(pad, config, track_width)]


def segment_crosses_flank(rects, x1, y1, x2, y2, layer) -> bool:
    """True when the centreline (x1, y1)-(x2, y2) on `layer` passes through
    the interior of one of `rects` (from `net_flank_rects`)."""
    eps = 1e-6
    dx, dy = x2 - x1, y2 - y1
    for layers, (rx0, ry0, rx1, ry1) in rects:
        if layer not in layers:
            continue
        # Liang-Barsky clip of the segment against the slightly shrunk rectangle.
        t0, t1, inside = 0.0, 1.0, True
        for p, q in ((-dx, x1 - (rx0 + eps)), (dx, (rx1 - eps) - x1),
                     (-dy, y1 - (ry0 + eps)), (dy, (ry1 - eps) - y1)):
            if p == 0:
                if q < 0:
                    inside = False
                    break
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                inside = False
                break
        if inside:
            return True
    return False


def net_flank_cells(pcb_data, net_id: int, config) -> np.ndarray:
    """(N, 3) flank cells of every row pad of `net_id`, unique."""
    layers = list(config.layers)
    chunks = []
    for pad in row_pads(pcb_data, net_id):
        idx = [layers.index(l) for l in pad.layers if l in layers]
        if idx:
            cells = flank_cells(pad, idx, config)
            if len(cells):
                chunks.append(cells)
    if not chunks:
        return np.empty((0, 3), dtype=np.int32)
    return np.unique(np.concatenate(chunks), axis=0)


def window_row_pads(pcb_data, net_id: int, layer: str, cx: float, cy: float,
                    half: float) -> List:
    """The net's row pads on `layer` whose centre lies in the square window of
    half-size `half` around (cx, cy)."""
    return [p for p in row_pads(pcb_data, net_id) if layer in p.layers
            and abs(p.global_x - cx) <= half and abs(p.global_y - cy) <= half]


def block_window_flanks(obstacles, pads, config, layer_index: int = 0) -> None:
    """Block the flanks of `pads` on `layer_index` of a throw-away map (a plane
    tap's local window map: built per tap, so nothing is removed)."""
    chunks = [c for c in (flank_cells(p, [layer_index], config) for p in pads) if len(c)]
    if chunks:
        obstacles.add_blocked_cells_batch(np.unique(np.concatenate(chunks), axis=0))


def via_flank_zones(pads, via_size: float, config) -> List[tuple]:
    """Per pad of `pads`, what `via_site_on_flank` tests a via site against:
    (cx, cy, half-width, half-length, long axis is x, axis column, the two
    flank rectangles grown by the via's radius). Built once per tap; the
    via search asks for thousands of sites."""
    r = via_size / 2
    column = config.grid_step / 2 + 1e-6
    zones = []
    for p in pads:
        hw, hl, along_x = _axis(p)
        grown = [(x0 - r, y0 - r, x1 + r, y1 + r) for x0, y0, x1, y1 in flank_rects(p, config)]
        zones.append((p.global_x, p.global_y, hw, hl, along_x, column, grown))
    return zones


def via_site_on_flank(zones, x: float, y: float) -> bool:
    """True when a via at (x, y) would put its copper over a flank of one of
    the pads `zones` describes (`via_flank_zones`) without sitting inside that
    pad: its trace would leave the pad from the side. A via on the axis column
    past the end stays legal."""
    for cx, cy, hw, hl, along_x, column, grown in zones:
        if not any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in grown):
            continue
        dx, dy = x - cx, y - cy
        along, lat = (abs(dx), abs(dy)) if along_x else (abs(dy), abs(dx))
        if along <= hl and lat <= hw:
            continue              # a via in the pad needs no trace
        if lat <= column:
            continue              # on the axis column
        return True
    return False


#: (id(map), net id) -> (map, cells): the flanks a `stamp` holds on a map
#: right now, so the route can lift them when no end is free. The map is
#: pinned in the value so its id cannot be reused while the entry lives.
_ACTIVE = {}


@contextlib.contextmanager
def stamp(obstacles, pcb_data, net_id: int, config):
    """Block `net_id`'s row-pad flanks in `obstacles` for the duration of the
    block, then remove exactly what was added. While the block runs the
    cells are on record for `lifted` (`active`)."""
    if obstacles is None:
        yield
        return
    cells = net_flank_cells(pcb_data, net_id, config)
    if not len(cells):
        yield
        return
    obstacles.add_blocked_cells_batch(cells)
    key = (id(obstacles), net_id)
    _ACTIVE[key] = (obstacles, cells)
    try:
        yield
    finally:
        _ACTIVE.pop(key, None)
        obstacles.remove_blocked_cells_batch(cells)


@contextlib.contextmanager
def also_on(copy, original, net_id: int):
    """Put the flanks `original` holds for `net_id` on record for `copy`, a
    clone taken while they were stamped (it carries the same cells)."""
    entry = _ACTIVE.get((id(original), net_id))
    if entry is None or copy is original:
        yield
        return
    key = (id(copy), net_id)
    _ACTIVE[key] = (copy, entry[1])
    try:
        yield
    finally:
        _ACTIVE.pop(key, None)


def active(obstacles, net_id: int):
    """The flank cells a `stamp` holds on `obstacles` for `net_id`, else None."""
    entry = _ACTIVE.get((id(obstacles), net_id))
    return None if entry is None else entry[1]


@contextlib.contextmanager
def lifted(obstacles, net_id: int):
    """Take the flanks `stamp` holds on `obstacles` for `net_id` off the map
    for the duration of the block (a route that found no way in through a
    pad's end), and put them back."""
    cells = active(obstacles, net_id)
    if cells is not None:
        obstacles.remove_blocked_cells_batch(cells)
    try:
        yield
    finally:
        if cells is not None:
            obstacles.add_blocked_cells_batch(cells)


#: BlockedCellTracker.get_blocked (rust_router) samples a frontier past this
#: many cells; a frontier that long is not known to miss the flanks.
FRONTIER_CAP = 10000


def touched(obstacles, net_id: int, *frontiers) -> bool:
    """True when a failed search's blocked-cell frontier (lists of (gx, gy,
    layer)) reached a flank cell `stamp` holds on `obstacles` for `net_id`,
    or may have (a sampled frontier); False when it provably did not."""
    cells = active(obstacles, net_id)
    if cells is None:
        return False
    if any(len(f) >= FRONTIER_CAP for f in frontiers if f):
        return True
    held = {(int(x), int(y), int(l)) for x, y, l in cells}
    return any((int(c[0]), int(c[1]), int(c[2])) in held
               for f in frontiers if f for c in f)


def seeds_to_pad_centres(pcb_data, net_id: int, seeds, config):
    """`seeds` ((x, y) or (x, y, layer) points on the net's copper) with every
    point lying on one of its row pads moved to that pad's centre -- the same
    copper, so the same cluster -- and the row pads found. A link the oracle
    anchors at its nearest approach starts on the FLANK of a row pad facing
    its neighbour; started from the centre, a walled search leaves the pad
    through an end. Returns (seeds, pads), seeds de-duplicated in order."""
    pads, out, seen = [], [], set()
    rows = row_pads(pcb_data, net_id)
    for s in seeds:
        x, y = s[0], s[1]
        layer = s[2] if len(s) > 2 else None
        for p in rows:
            if layer is not None and layer not in p.layers:
                continue
            if (abs(x - p.global_x) <= p.size_x / 2 + config.track_width / 2
                    and abs(y - p.global_y) <= p.size_y / 2 + config.track_width / 2):
                x, y = p.global_x, p.global_y
                if p not in pads:
                    pads.append(p)
                break
        moved = (x, y) if layer is None else (x, y, layer)
        if moved not in seen:
            seen.add(moved)
            out.append(moved)
    return out, pads


def walled_copy(obstacles, pcb_data, net_id: int, layers, config, seeds,
                margin: float = 2.0):
    """A clone of `obstacles` (layer i = `layers[i]`) with the flanks of the
    net's row pads within `margin` of the `seeds`' box blocked: the map a
    plane link is routed on first, so it lands through the pads' ends."""
    xs = [s[0] for s in seeds]
    ys = [s[1] for s in seeds]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    chunks = []
    for p in row_pads(pcb_data, net_id):
        if not (x0 <= p.global_x <= x1 and y0 <= p.global_y <= y1):
            continue
        idx = [i for i, l in enumerate(layers) if l in p.layers]
        if idx:
            c = flank_cells(p, idx, config)
            if len(c):
                chunks.append(c)
    if not chunks:
        return None
    walled = obstacles.clone()
    walled.add_blocked_cells_batch(np.unique(np.concatenate(chunks), axis=0))
    return walled
