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
route was made to turn past the end.

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


def flank_rects(pad, config) -> List[tuple]:
    """`pad`'s two flanks as board-frame rectangles (x0, y0, x1, y1): the
    region `flank_cells` rasterises, for copper that is not on the grid."""
    hw, hl, along_x = _axis(pad)
    column = config.grid_step / 2 + 1e-6
    outer = hw + config.track_width / 2 + config.clearance
    reach = hl + config.track_width + config.clearance
    cx, cy = pad.global_x, pad.global_y
    out = []
    for lo, hi in ((column, outer), (-outer, -column)):
        if along_x:
            out.append((cx - reach, cy + lo, cx + reach, cy + hi))
        else:
            out.append((cx + lo, cy - reach, cx + hi, cy + reach))
    return out


def net_flank_rects(pcb_data, net_id: int, config) -> List[tuple]:
    """(layers, rect) of every flank of `net_id`'s row pads."""
    return [(set(pad.layers), r) for pad in row_pads(pcb_data, net_id)
            for r in flank_rects(pad, config)]


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


@contextlib.contextmanager
def stamp(obstacles, pcb_data, net_id: int, config):
    """Block `net_id`'s row-pad flanks in `obstacles` for the duration of the
    block, then remove exactly what was added."""
    if obstacles is None:
        yield
        return
    cells = net_flank_cells(pcb_data, net_id, config)
    if len(cells):
        obstacles.add_blocked_cells_batch(cells)
    try:
        yield
    finally:
        if len(cells):
            obstacles.remove_blocked_cells_batch(cells)
