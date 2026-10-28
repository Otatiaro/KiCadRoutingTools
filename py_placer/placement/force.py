"""Force-directed global placement: a start for the anneal (`place_simple
--forced`).

Every movable part is a rigid body with a position and a CONTINUOUS turn.
One energy is lowered by Nesterov's accelerated gradient with a
Barzilai-Borwein step (the ePlace optimiser); every force is minus its
gradient, applied at the point it acts on (a pad, a piece of courtyard), so
each part feels a force on its origin AND a torque from where its pads sit:

    E = sum over wired nets of w_n x a smooth net length    (ATTRACTION)
      + k_claim x each claimed pad's spring to its partner pad, and each
        decap's rail pad to the supply pin it is matched to, past a free
        length                                               (PROXIMITY)
      + k_pad x the short-range overlap of every two pads of different
        parts within their escape reach                      (PAD REPULSION)
      + lambda x the electrostatic energy of the density     (SPREADING)
      + k_edge x how far a part is outside the board, a zone that holds it,
        or inside a keep-out that binds it                   (BOUNDS)

ATTRACTION. A net's length is the weighted-average (`wa`) or log-sum-exp
(`lse`) smoothing of its half-perimeter (ePlace), with a smoothing length
`gamma` that falls as the parts spread: from far a pad is pulled at a nearly
constant force, near its net's other pads the pull fades smoothly. `gauss`
(a pin's well round its net's centre, `s^2 (1 - exp(-d^2 / 2 s^2))`) is the
alternative profile: linear near, fading past `s`. Nets poured as planes
(ground, the board's rails, `plane_nets`) do NOT attract: each of their pads
drops a via to its plane, so instead the pad's escape reach grows by a via
and it repels its neighbours further and weighs more in the density.

PROXIMITY. The intent's claims (a decap at its named pin, a crystal at its
oscillator pins) and the model's decaps (each matched to one supply pin of its
rail, the pins taken one cap each before any takes two) are stiff linear
springs (quadratic energy) past a free length: they hold whatever the
density does.

PAD REPULSION. Short range only: two pads of different parts on a shared
face repel while their centres are closer than the sum of their reaches
(`reach` = half the pad plus its escape: a track and its clearance, or for a
plane net's pad a via and its clearance), with a C1 profile: `(r0 - d)^2`
(`poly`), or a Gaussian of sigma r0/2 cut at r0 and shifted to meet zero
there (`gauss`).

SPREADING. Each face's density on a grid of `bin_mm` cells: each part's
courtyard (sampled at points, so the field turns it too), plus each pad's
routing demand (a track's escape; a via for a plane-net pad). The area the
movable parts cannot use (off the board, under a fixed part, inside a
keep-out that binds every part) is filled to the target density, so the
parts spread to an even density over the free area. Poisson's equation is
solved by DCT (Neumann boundary), the field smoothed by a Gaussian of
`smooth_mm`, and lambda rises each step from a fraction of the wire force
(blocks form first, then spread) until the overflow falls under `overflow`.

TURNS. A part turns freely until the overflow falls under `turn_overflow`;
then every turn is rounded to the nearest quarter of its own lattice and
`relax` more steps run with turns held.

The template members ride rigidly on their chip (seated as their template
puts them for its pose), and a part held on an edge (`edge_seat`) is
projected back onto it after each step, its turn held.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Where the profiles and ranges default; `place_simple --force-param` sets
#: any of them by name.


@dataclass
class ForceParams:
    attraction: str = 'wa'          # wa | lse | gauss
    gamma_scale: float = 1.0        # x the ePlace gamma schedule (wa / lse)
    gauss_mm: float = 8.0           # gauss attraction's reach (about a block)
    repulsion: str = 'poly'         # poly | gauss
    pad_weight: float = 2.0         # k_pad, per mm^2 of reach overlap
    claim_weight: float = 4.0       # k of a claim's or decap's spring, per mm^2
    claim_free_mm: float = 0.5      # a spring's free length
    claim_free_share: float = 0.5   # ... or this share of its claim's limit, the longer
    group_weight: float = 0.0       # k_group: a functional block held together (0: off)
    group_model: str = 'centroid'   # centroid: each member to its block's centre past the
    #                                 block's radius; hpwl: the members pulled as one net
    group_free: float = 1.0         # the block's free radius, x its radius at the allowed fill
    group_source: str = 'auto'      # auto: the declared groups, else the sheets; sheet: the sheets
    bin_mm: float = 1.0             # density grid cell
    smooth_mm: float = 0.0          # Gaussian smoothing of the field
    density: str = 'poisson'        # poisson | cap | both
    screen_mm: float = 0.0          # poisson: the field's reach (0: the whole board)
    spread_weight: float = 0.1      # both: the screened field's weight against the cap's
    cap_ratio: float = 1.5          # cap: the density allowed, x the board's mean
    cap_smooth_mm: float = 2.0      # cap: the reach over which a cell's excess is felt
    demand_weight: float = 1.0      # x each pad's routing demand in the density
    escape_mm: float = 1.0          # a track's escape length for its demand
    lambda_start: float = 0.0001    # lambda's start, x |grad W| / |grad D|
    lambda_rate: float = 1.005      # lambda's growth per step
    overflow: float = 0.10          # stop when the overflow is under this
    target_density: float = 1.0     # a cell's capacity for the overflow, x its free area
    turn_overflow: float = 0.25     # turns rounded once the overflow is under this
    pad_on: float = 0.5             # pad repulsion ramps in from this overflow to `overflow`
    iters: int = 3000               # most global steps
    relax: int = 400                # most steps once spread, the turns held
    max_step_mm: float = 2.0        # the most any point moves in one step
    relax_step_mm: float = 0.25     # ... once spread
    overlap_weight: float = 1.0     # k_ov once the parts are spread, per mm^2 of overlap
    overlap_growth: float = 1.03    # k_ov's growth per step from then on
    overlap_max: float = 1000.0     # k_ov's ceiling, x overlap_weight
    overlap_soft_mm: float = 0.05   # the softplus width of an overlap
    overlap_tol_mm2: float = 0.5    # relax ends once the overlap is under this
    skin_mm: float = 1.0            # Verlet skin of the neighbour lists
    swap_every: int = 25            # steps between two swap passes (0: none)
    swap_from: float = 0.5          # swaps once the overflow is under this, until the relax
    swap_max: int = 50              # most moves one pass takes
    move_every: int = 0             # steps between two part-move passes (0: none)
    move_from: float = 0.25         # part moves once the overflow is under this, relax included
    move_max: int = 50              # most part moves one pass takes
    cross_mm: float = 2.0           # a crossing of two nets' trees, in mm of wire (the anneal's)
    cross_weight: float = 0.0       # the smoothed crossing term, mm of wire per crossing (0: off)
    cross_eps0: float = 2.0         # its smoothing width (mm) as the pads come in ...
    cross_eps1: float = 0.3         # ... falling to this once they are in
    cross_every: int = 25           # steps between two rebuilds of the trees and their near pairs
    cross_margin_mm: float = 1.0    # a pair is near when its boxes are within this (+ 3 eps)
    snapshot_every: int = 0         # steps between two snapshots (0: phase changes only)
    edge_weight: float = 10.0       # k_edge, per mm^2 outside
    spread_sigma: float = 0.04      # start: jitter round the centre, x the board span
    via_mm: float = 0.0             # 0: the board's via (place_simple passes it)
    track_mm: float = 0.0           # 0: the board's clearance as track
    clearance_mm: float = 0.0       # 0: the board's clearance

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    def set(self, key, value):
        if key not in self.names():
            raise ValueError(f"unknown force parameter {key!r}; known: {', '.join(self.names())}")
        cur = getattr(self, key)
        setattr(self, key, type(cur)(value) if not isinstance(cur, str) else str(value))


# -- profiles: energy and its derivative along d -----------------------------

def repulsion_profile(kind, d, r0):
    """(energy, dE/dd) of a pair at centre distance `d` with reach `r0`,
    both zero at and past r0 with a zero slope there (C1)."""
    inside = d < r0
    if kind == 'poly':
        g = np.where(inside, r0 - d, 0.0)
        return g * g, -2.0 * g
    if kind == 'gauss':
        s = r0 / 2.0
        e = np.exp(-d * d / (2 * s * s))
        ec = np.exp(-r0 * r0 / (2 * s * s))
        # shifted and tilted so energy and slope meet zero at r0
        slope_c = -r0 / (s * s) * ec
        en = (e - ec - slope_c * (d - r0)) * (s * s)
        de = (-d / (s * s) * e - slope_c) * (s * s)
        return np.where(inside, en, 0.0), np.where(inside, de, 0.0)
    raise ValueError(f"unknown repulsion profile {kind!r}")


def _segments(order_net, n_nets):
    starts = np.searchsorted(order_net, np.arange(n_nets))
    return starts


def wa_grad(x, net, starts, w, gamma, kind='wa'):
    """Weighted-average (or log-sum-exp) half-perimeter along one axis: the
    length per net and d(length)/dx per pin. `net` sorted, `starts` its
    segment starts."""
    xmax = np.maximum.reduceat(x, starts)[net]
    xmin = np.minimum.reduceat(x, starts)[net]
    a = np.exp((x - xmax) / gamma)
    b = np.exp((xmin - x) / gamma)
    sa = np.add.reduceat(a, starts)
    sb = np.add.reduceat(b, starts)
    if kind == 'lse':
        L = gamma * np.log(sa) + xmax[starts] + gamma * np.log(sb) - xmin[starts]
        g = a / sa[net] - b / sb[net]
        return float(np.dot(w, L / 1.0)), g * w[net]
    sxa = np.add.reduceat(x * a, starts)
    sxb = np.add.reduceat(x * b, starts)
    pa, pb = sxa / sa, sxb / sb
    L = pa - pb
    ga = a / sa[net] * (1.0 + (x - pa[net]) / gamma)
    gb = b / sb[net] * (1.0 - (x - pb[net]) / gamma)
    return float(np.dot(w, L)), (ga - gb) * w[net]


def gauss_grad(x, y, net, starts, counts, w, s):
    """Each pin's Gaussian well round its net's centre: energy and its
    gradient per pin (the centre's own dependence included)."""
    cx = (np.add.reduceat(x, starts) / counts)[net]
    cy = (np.add.reduceat(y, starts) / counts)[net]
    dx, dy = x - cx, y - cy
    e = np.exp(-(dx * dx + dy * dy) / (2 * s * s))
    E = float(np.sum(w[net] * s * s * (1.0 - e)))
    gx_p, gy_p = w[net] * e * dx, w[net] * e * dy
    # the centre moves with each pin by 1/count: subtract the mean pull
    gx = gx_p - (np.add.reduceat(gx_p, starts) / counts)[net]
    gy = gy_p - (np.add.reduceat(gy_p, starts) / counts)[net]
    return E, gx, gy


# -- density ------------------------------------------------------------------

class Density:
    """Charges on a grid; Poisson by DCT; the field's gradient at points."""

    def __init__(self, x0, y0, x1, y1, h, screen_mm=0.0):
        self.x0, self.y0, self.h = x0, y0, h
        self.nx = max(4, int(math.ceil((x1 - x0) / h)))
        self.ny = max(4, int(math.ceil((y1 - y0) / h)))
        wu = np.pi * np.arange(self.nx) / self.nx
        wv = np.pi * np.arange(self.ny) / self.ny
        k2 = (wu[:, None] ** 2 + wv[None, :] ** 2)
        # screened (Yukawa) when `screen_mm`: the potential of a charge
        # falls as exp(-r / screen_mm), so a part is pushed off its
        # neighbours within that reach and not across the whole board
        k2 += (h / screen_mm) ** 2 if screen_mm > 0 else 0.0
        k2[0, 0] = k2[0, 0] or 1.0
        self.inv = 1.0 / k2
        if screen_mm <= 0:
            self.inv[0, 0] = 0.0

    def _cic(self, px, py):
        fx = (px - self.x0) / self.h - 0.5
        fy = (py - self.y0) / self.h - 0.5
        i0 = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        return i0, j0, tx, ty

    def deposit(self, px, py, q):
        g = np.zeros((self.nx, self.ny))
        if len(px) == 0:
            return g
        i0, j0, tx, ty = self._cic(px, py)
        np.add.at(g, (i0, j0), q * (1 - tx) * (1 - ty))
        np.add.at(g, (i0 + 1, j0), q * tx * (1 - ty))
        np.add.at(g, (i0, j0 + 1), q * (1 - tx) * ty)
        np.add.at(g, (i0 + 1, j0 + 1), q * tx * ty)
        return g

    def potential(self, rho, smooth_cells=0.0):
        from scipy import fft as sfft
        from scipy.ndimage import gaussian_filter
        a = sfft.dctn(rho - rho.mean(), type=2, norm="ortho", workers=-1)
        psi = sfft.idctn(a * self.inv, type=2, norm="ortho", workers=-1)
        if smooth_cells > 0:
            psi = gaussian_filter(psi, smooth_cells, mode='nearest')
        gx, gy = np.gradient(psi, self.h)
        return psi, gx, gy

    def cic(self, px, py):
        """Each point's four cells (flat indices, (4, n)) and weights."""
        i0, j0, tx, ty = self._cic(px, py)
        c = i0 * self.ny + j0
        cell = np.stack([c, c + self.ny, c + 1, c + self.ny + 1])
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
        return cell, w

    def deposit_cic(self, cell, w, q):
        n = self.nx * self.ny
        return np.bincount(cell.ravel(), (w * q).ravel(), n).reshape(self.nx, self.ny)

    def sample_cic(self, fx, fy, cell, w):
        return (fx.ravel()[cell] * w).sum(0), (fy.ravel()[cell] * w).sum(0)

    def sample(self, field, px, py):
        i0, j0, tx, ty = self._cic(px, py)
        return (field[i0, j0] * (1 - tx) * (1 - ty) + field[i0 + 1, j0] * tx * (1 - ty)
                + field[i0, j0 + 1] * (1 - tx) * ty + field[i0 + 1, j0 + 1] * tx * ty)


def _rect_cover(den, rect, out, weight=1.0):
    """Add each cell's covered fraction of `rect` (x0, y0, x1, y1) to `out`."""
    h = den.h
    i0 = max(0, int((rect[0] - den.x0) // h))
    i1 = min(den.nx - 1, int((rect[2] - den.x0) // h))
    j0 = max(0, int((rect[1] - den.y0) // h))
    j1 = min(den.ny - 1, int((rect[3] - den.y0) // h))
    if i1 < i0 or j1 < j0:
        return
    xs = den.x0 + h * np.arange(i0, i1 + 2)
    ys = den.y0 + h * np.arange(j0, j1 + 2)
    ox = np.clip(np.minimum(xs[1:], rect[2]) - np.maximum(xs[:-1], rect[0]), 0, h) / h
    oy = np.clip(np.minimum(ys[1:], rect[3]) - np.maximum(ys[:-1], rect[1]), 0, h) / h
    out[i0:i1 + 1, j0:j1 + 1] += weight * ox[:, None] * oy[None, :]


# -- the bodies ---------------------------------------------------------------

def _side_mask(sides):
    m = 0
    for s in sides:
        m |= 1 if str(s).upper().startswith('F') else 2
    return m or 1


def _samples(b, step):
    """Points over the local box `b` (x0, y0, x1, y1), each carrying its
    share of the box's area."""
    w, h = max(1e-3, b[2] - b[0]), max(1e-3, b[3] - b[1])
    nx, ny = max(1, int(math.ceil(w / step))), max(1, int(math.ceil(h / step)))
    xs = b[0] + w * (np.arange(nx) + 0.5) / nx
    ys = b[1] + h * (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel(), np.full(nx * ny, w * h / (nx * ny))


def _to_local(px, py, pose):
    """Global points into the frame of a part at `pose` (x, y, rot deg)."""
    phi = math.radians(-pose[2])
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = np.asarray(px) - pose[0], np.asarray(py) - pose[1]
    return c * dx + s * dy, -s * dx + c * dy


def _to_global(lx, ly, pose):
    phi = math.radians(-pose[2])
    c, s = math.cos(phi), math.sin(phi)
    return pose[0] + c * lx - s * ly, pose[1] + s * lx + c * ly


def _lattice_round(rot, base):
    """`rot` (deg) to the nearest of base + k x 90."""
    k = round((rot - base) / 90.0)
    return (base + 90.0 * k) % 360.0



# -- the problem as plain arrays ---------------------------------------------

@dataclass
class Problem:
    """Everything the kernel reads, as arrays: no KiCad object, no dict.

    Bodies b = 0..nb-1 (origin X, Y and turn PHI = radians(-rot)). A point
    on a body is (body, lx, ly) in its frame; a fixed point is global."""
    nb: int
    # pins (every connected pad of a body): frame, reach, faces, demand
    pin_body: np.ndarray
    pin_lx: np.ndarray
    pin_ly: np.ndarray
    pad_r: np.ndarray
    pad_side: np.ndarray
    pad_dem: np.ndarray
    # fixed pads
    fx_x: np.ndarray
    fx_y: np.ndarray
    fx_r: np.ndarray
    fx_side: np.ndarray
    fx_dem: np.ndarray
    # courtyard samples
    smp_body: np.ndarray
    smp_lx: np.ndarray
    smp_ly: np.ndarray
    smp_q: np.ndarray
    smp_side: np.ndarray
    # attraction: entries sorted by net; kind 0 pin, 1 fixed pad, 2 body origin
    a_kind: np.ndarray
    a_ref: np.ndarray
    a_net: np.ndarray
    a_starts: np.ndarray
    a_counts: np.ndarray
    a_w: np.ndarray
    # springs: one row per (spring, candidate target); the nearest is used
    sp_id: np.ndarray       # spring index of the row
    sp_subj: np.ndarray     # the subject pin
    sp_tpin: np.ndarray     # the target pin, -1 for a fixed target
    sp_tx: np.ndarray       # the fixed target's position
    sp_ty: np.ndarray
    sp_free: np.ndarray     # per spring
    # bounds: per body box in its frame; usable box; zone / keep-out rects
    body_box: np.ndarray
    turnable: np.ndarray
    usable: Tuple[float, float, float, float]
    bt_body: np.ndarray
    bt_in: np.ndarray
    bt_rect: np.ndarray
    rg: np.ndarray          # radius of gyration: mm per radian of turn
    # courtyard boxes (grown by half the clearance): a body's own and its
    # riders', in its frame (centre, half sizes); the fixed parts' global
    bx_body: np.ndarray
    bx_cx: np.ndarray
    bx_cy: np.ndarray
    bx_a: np.ndarray
    bx_b: np.ndarray
    bx_side: np.ndarray
    fb_rect: np.ndarray
    fb_side: np.ndarray
    # density
    grid: Tuple[float, float, float]    # x0, y0, h
    shape: Tuple[int, int]
    blocked: np.ndarray     # (2, nx, ny): 0 free .. 1 unusable, per face F, B
    static: np.ndarray      # (2, nx, ny): the fixed charge per face
    target: np.ndarray      # (2,): target charge per mm^2 per face
    # functional blocks (centroid model): one row per (block, member body),
    # sorted by block; each block's free radius
    gm_body: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    gm_grp: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    gm_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    g_R: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _corners(prob, X, Y, PHI):
    c, s = np.cos(PHI), np.sin(PHI)
    bb = prob.body_box
    cx = np.stack([c * bb[:, ix] - s * bb[:, iy] for ix, iy in ((0, 1), (2, 1), (0, 3), (2, 3))])
    cy = np.stack([s * bb[:, ix] + c * bb[:, iy] for ix, iy in ((0, 1), (2, 1), (0, 3), (2, 3))])
    return cx, cy


class SwapMoves:
    """Declared trades of nets between pins, for `optimise` to try.

    `moves` is `[(key, [(pin a, pin b), ...])]` in the problem's pin indices
    (`build_problem`'s `meta['pin_key']`): taking a move trades the nets of
    each pair at once. `allowed(key)` says whether a move may be taken now
    (a caller's validity rule, such as a differential pair kept on adjacent
    gates) and `commit(key)` is told each move taken. The base allows
    every move."""

    def __init__(self, moves):
        self.moves = list(moves)

    def allowed(self, key) -> bool:
        return True

    def commit(self, key) -> None:
        pass


def cross_energy(A, B, C, D, eps):
    """The smoothed crossing indicator of segment pairs AB, CD (each n x 2):
    E = sum sigma(-s_C s_D / eps^2) sigma(-t_A t_B / eps^2), s the signed
    distance to line AB, t to line CD -- 1 for a clear crossing, 0 for a pair
    well apart, smooth between. Returns (E, dE/dA, dE/dB, dE/dC, dE/dD).

    A point Q's distance to line P0P1 is s = n.(Q - P0), n the unit normal;
    with p = Q's projection along P0P1 as a share of its length, ds/dQ = n,
    ds/dP1 = -p n (the line turns about P0), ds/dP0 = -(1 - p) n."""
    from scipy.special import expit

    def side(P0, P1, Q):
        e = P1 - P0
        L = np.hypot(e[:, 0], e[:, 1]) + 1e-9
        tau = e / L[:, None]
        n = np.column_stack([-tau[:, 1], tau[:, 0]])
        rel = Q - P0
        return (n * rel).sum(1), (tau * rel).sum(1) / L, n

    e2 = eps * eps
    s1, pC, n1 = side(A, B, C)
    s2, pD, _ = side(A, B, D)
    t1, pA, n2 = side(C, D, A)
    t2, pB, _ = side(C, D, B)
    su, sv = expit(-s1 * s2 / e2), expit(-t1 * t2 / e2)
    du, dv = su * (1 - su) * sv, sv * (1 - sv) * su
    ds1, ds2 = -du * s2 / e2, -du * s1 / e2
    dt1, dt2 = -dv * t2 / e2, -dv * t1 / e2
    gA = -(ds1 * (1 - pC) + ds2 * (1 - pD))[:, None] * n1 + dt1[:, None] * n2
    gB = -(ds1 * pC + ds2 * pD)[:, None] * n1 + dt2[:, None] * n2
    gC = ds1[:, None] * n1 - (dt1 * (1 - pA) + dt2 * (1 - pB))[:, None] * n2
    gD = ds2[:, None] * n1 - (dt1 * pA + dt2 * pB)[:, None] * n2
    return float((su * sv).sum()), gA, gB, gC, gD


class Ratsnest:
    """The nets' trees: each net's Manhattan spanning tree over its pins and
    fixed pads, plane nets aside (the attraction's nets, its block
    pseudo-nets left out). An ENTRY is one such point."""

    def __init__(self, prob: 'Problem'):
        self.p = prob
        keep = prob.a_kind != 2
        self.e_kind, self.e_ref = prob.a_kind[keep], prob.a_ref[keep]
        self.e_net = prob.a_net[keep]
        pin = self.e_kind == 0
        self.e_pin = pin
        self._pin_ref = np.where(pin, self.e_ref, 0)
        fr = np.where(pin, 0, self.e_ref)
        self._fx = np.where(pin, 0.0, prob.fx_x[fr] if prob.fx_x.size else 0.0)
        self._fy = np.where(pin, 0.0, prob.fx_y[fr] if prob.fx_y.size else 0.0)
        self.e_body = np.where(pin, prob.pin_body[self._pin_ref], -1)
        self.n_nets = int(prob.a_w.size)
        self.w = prob.a_w
        self.net_ents = [np.nonzero(self.e_net == n)[0] for n in range(self.n_nets)]

    def _points(self, px, py):
        """Every entry's position: a pin's at (px, py), a fixed pad's own."""
        pin = self.e_pin
        return np.where(pin, px[self._pin_ref], self._fx), np.where(pin, py[self._pin_ref], self._fy)

    def trees(self, px, py, nets):
        """The spanning trees of `nets` at pin positions (px, py), one scipy
        call over the block-diagonal graph: (segments n x 4, their net, their
        entries a / b, each net's tree length)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        ents = [self.net_ents[n] for n in nets]
        ents = [e for e in ents if e.size >= 2]
        length = np.zeros(self.n_nets)
        if not ents:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), length
        I, J = [], []
        for e in ents:
            i, j = np.triu_indices(e.size, 1)
            I.append(e[i])
            J.append(e[j])
        I, J = np.concatenate(I), np.concatenate(J)
        allx, ally = self._points(px, py)
        d = np.abs(allx[I] - allx[J]) + np.abs(ally[I] - ally[J]) + 1e-6
        n = self.e_net.size
        t = minimum_spanning_tree(coo_matrix((d, (I, J)), shape=(n, n))).tocoo()
        a, b = t.row, t.col
        seg = np.column_stack([allx[a], ally[a], allx[b], ally[b]])
        net = self.e_net[a]
        np.add.at(length, net, t.data - 1e-6)
        return seg, net, a, b, length


class PartMoves(Ratsnest):
    """The discrete moves the gradient cannot make, for `optimise` to try:
    two parts trading places, or a two-pad part turned end for end. Pads and
    courtyards repel, so two passives whose nets cross never pass each other
    by sliding; a trade or a half turn does it in one step.

    The candidates come from the crossings of the nets' trees (each net's
    Manhattan spanning tree over its pins and fixed pads, plane nets aside):
    each crossing's parts, a part that is no edge seat and carries no
    template member. A trade takes two parts of one footprint (same box,
    same pads) in one block, each to the other's pose, a two-pad one also
    turned or not; a half turn takes a two-pad part. A move is priced
    exactly on what it touches -- its nets' weighted tree length, the
    crossings of their trees with every other tree at `cross_mm` each, the
    springs -- and taken when it lowers that."""

    def __init__(self, prob: 'Problem', P: 'ForceParams', block_of):
        super().__init__(prob)
        self.P = P
        nb = prob.nb
        self.body_nets = [np.unique(self.e_net[self.e_body == b]) for b in range(nb)]
        pins_of = [np.nonzero(prob.pin_body == b)[0] for b in range(nb)]
        self.pins_of = pins_of
        anchors = np.bincount(prob.bx_body, minlength=nb) > 1
        self.free = prob.turnable & ~anchors
        self.two_pad = np.array([len(pins_of[b]) == 2 for b in range(nb)])
        sig = {}
        self.sig = np.zeros(nb, int)
        for b in range(nb):
            pts = sorted(zip(np.round(prob.pin_lx[pins_of[b]], 2), np.round(prob.pin_ly[pins_of[b]], 2)))
            key = (tuple(np.round(prob.body_box[b], 2)), tuple(pts))
            self.sig[b] = sig.setdefault(key, len(sig))
        self.block = np.asarray(block_of, int)

    def candidates(self, seg, a, b, cross_i, cross_j):
        """The moves the crossings suggest: ('turn', b) and ('trade', b1, b2, f1, f2)."""
        out = set()
        free, two = self.free, self.two_pad
        for i, j in zip(cross_i, cross_j):
            Bi = {int(x) for x in self.e_body[[a[i], b[i]]] if x >= 0 and free[x]}
            Bj = {int(x) for x in self.e_body[[a[j], b[j]]] if x >= 0 and free[x]}
            for x in Bi | Bj:
                if two[x]:
                    out.add(('turn', x))
            for x in Bi:
                for y in Bj:
                    if x == y or self.sig[x] != self.sig[y] or self.block[x] != self.block[y]:
                        continue
                    x_, y_ = min(x, y), max(x, y)
                    for fx_ in ((0, 1) if two[x_] else (0,)):
                        for fy_ in ((0, 1) if two[y_] else (0,)):
                            out.add(('trade', x_, y_, fx_, fy_))
        return sorted(out)

    @staticmethod
    def apply(move, X, Y, PHI):
        """`move` on the poses (in place); returns the bodies it moved."""
        if move[0] == 'turn':
            b = move[1]
            PHI[b] += math.pi
            return (b,)
        _k, x, y, fx_, fy_ = move
        X[x], X[y] = X[y], X[x]
        Y[x], Y[y] = Y[y], Y[x]
        PHI[x], PHI[y] = PHI[y] + math.pi * fx_, PHI[x] + math.pi * fy_
        return (x, y)

    def run(self, K, X, Y, PHI, log=None, step=0):
        """One pass on (X, Y, PHI), in place: every candidate priced at the
        same start, then the best taken first, skipping any whose nets an
        earlier one has touched. Returns the moves taken [(move, delta)]."""
        from .simple import grid_router as gr
        P = self.P
        px, py, _qx, _qy = K.positions(X, Y, PHI)
        seg, net, a, b, length = self.trees(px, py, range(self.n_nets))
        if not seg.shape[0]:
            return []
        segc = np.ascontiguousarray(seg)
        ci, cj = gr.seg_cross_pairs(segc, None, np.ascontiguousarray(net, dtype=np.int64))
        cands = self.candidates(seg, a, b, ci, cj)
        e0 = K.spring_energy(px, py)
        priced = []
        for mv in cands:
            Xc, Yc, Pc = X.copy(), Y.copy(), PHI.copy()
            moved = self.apply(mv, Xc, Yc, Pc)
            nets = np.unique(np.concatenate([self.body_nets[m] for m in moved]))
            if not nets.size and mv[0] == 'turn':
                continue
            qx, qy, _a, _b = K.positions(Xc, Yc, Pc)
            touched = np.isin(net, nets)
            old, rest = segc[touched], segc[~touched]
            nseg, nnet, _na, _nb, nlen = self.trees(qx, qy, nets)
            nseg = np.ascontiguousarray(nseg)

            def cross(S, L):
                if not S.shape[0]:
                    return 0
                c = gr.seg_crossings(S, np.ascontiguousarray(rest)) if rest.shape[0] else 0
                return c + gr.seg_crossings(S, None, np.ascontiguousarray(L, dtype=np.int64))
            dcross = cross(nseg, nnet) - cross(old, net[touched])
            dwire = float(np.dot(self.w[nets], nlen[nets] - length[nets]))
            dspring = K.spring_energy(qx, qy) - e0
            delta = dwire + P.cross_mm * dcross + dspring
            if delta < -1e-6:
                priced.append((delta, mv, set(nets.tolist()), moved, dcross))
        priced.sort(key=lambda t: t[0])
        taken, busy_nets, busy_bodies = [], set(), set()
        for delta, mv, nets, moved, dcross in priced:
            if len(taken) >= P.move_max:
                break
            if nets & busy_nets or busy_bodies.intersection(moved):
                continue
            self.apply(mv, X, Y, PHI)
            busy_nets |= nets
            busy_bodies.update(moved)
            taken.append((mv, round(delta, 3), int(dcross)))
        if log and taken:
            log(f"force step {step:4d}: {len(taken)} part move(s) of {len(cands)} candidate(s) "
                f"over {len(ci)} crossing(s), {sum(t[1] for t in taken):+.1f} mm, "
                f"crossings {sum(t[2] for t in taken):+d}")
        return taken


class Kernel:
    """E(X, Y, PHI) and its gradient, by term, on a `Problem`."""

    def __init__(self, prob: Problem, P: ForceParams):
        self.p, self.P = prob, P
        x0, y0, h = prob.grid
        self.den = Density(x0, y0, x0 + h * prob.shape[0], y0 + h * prob.shape[1], h,
                           P.screen_mm if P.density in ('poisson', 'both') else 0.0)
        self.cap = P.target_density * h * h * (1.0 - prob.blocked)
        if P.density not in ('poisson', 'cap', 'both'):
            raise ValueError(f"unknown density model {P.density!r}")
        # cap: the charge a cell may hold before it pushes, per face (the
        # board's mean fill x cap_ratio over the free part of the cell), and
        # the fixed parts' own demand counted against it
        self.allow = (P.cap_ratio * prob.target[:, None, None] * h * h * (1.0 - prob.blocked))
        self.fixed_q = prob.static - prob.target[:, None, None] * h * h * prob.blocked
        self.faces = [f for f in (0, 1)
                      if (prob.smp_side & (1 << f)).any() or (prob.pad_side & (1 << f)).any()]
        self.rmax = 2.0 * float(max(prob.pad_r.max(initial=0.0), prob.fx_r.max(initial=0.0)))
        self.time = {}
        from scipy.spatial import cKDTree
        self._tree = cKDTree
        self.n_springs = int(prob.sp_free.size)
        self.rep = 1.0      # the pad repulsion's share now (`optimise` ramps it)
        self.kov = 0.0      # the courtyard overlap's weight now (`optimise` raises it)
        # Verlet lists: the pairs within reach plus a skin, rebuilt only when
        # a point has moved half the skin since they were built
        self.skin = P.skin_mm
        self._pads = None       # (positions at build, i, j)
        self._boxes = None      # (centres at build, i, j, fixed i, fixed j)
        self.rebuilds = {'pads': 0, 'boxes': 0}
        self.measure_overlap = False    # price the overlap area even at k_ov 0
        # the smoothed crossings: the trees, their near pairs (entries a, b of
        # one segment, c, d of the other), the weight and width now
        self.rn = Ratsnest(prob) if P.cross_weight > 0 else None
        self._xp = None
        self.cross_k, self.cross_eps = 0.0, P.cross_eps0
        # each face's charges: courtyard samples, then pad demands
        self._charges = {}
        for f in self.faces:
            bit = 1 << f
            ms = (prob.smp_side & bit) > 0
            mp = (prob.pad_side & bit) > 0
            q = np.concatenate([prob.smp_q[ms], P.demand_weight * prob.pad_dem[mp]])
            body = np.concatenate([prob.smp_body[ms], prob.pin_body[mp]])
            self._charges[f] = (ms, mp, q, body)
        nfx = prob.fx_x.size
        self._allr = np.concatenate([prob.pad_r, prob.fx_r])
        self._alls = np.concatenate([prob.pad_side, prob.fx_side])
        self._allb = np.concatenate([prob.pin_body, np.full(nfx, -1)])
        self._bR = np.hypot(prob.bx_a, prob.bx_b)
        fr = prob.fb_rect
        self._fbc = np.column_stack([(fr[:, 0] + fr[:, 2]) / 2, (fr[:, 1] + fr[:, 3]) / 2])
        self._fbR = np.hypot((fr[:, 2] - fr[:, 0]) / 2, (fr[:, 3] - fr[:, 1]) / 2)

    def _t(self, k, t0):
        import time as _time
        now = _time.perf_counter()
        self.time[k] = self.time.get(k, 0.0) + now - t0
        return now

    def _pad_pairs(self, px, py, allx, ally):
        """The pad pairs within reach + skin of different parts on a shared
        face, at least one movable; rebuilt when a pad moved half the skin."""
        v = self._pads
        if v is not None and v[0].size == px.size:
            moved = float(np.max(np.hypot(px - v[0], py - v[1]), initial=0.0))
            if moved < self.skin / 2.0:
                return v[2], v[3]
        pr = self._tree(np.column_stack([allx, ally])).query_pairs(
            self.rmax + self.skin, output_type='ndarray')
        i, j = (pr[:, 0], pr[:, 1]) if len(pr) else (np.zeros(0, int), np.zeros(0, int))
        b, sd = self._allb, self._alls
        keep = (b[i] != b[j]) & ((sd[i] & sd[j]) > 0) & ((b[i] >= 0) | (b[j] >= 0))
        i, j = i[keep], j[keep]
        self._pads = (px.copy(), py.copy(), i, j)
        self.rebuilds['pads'] += 1
        return i, j

    def _box_geometry(self, X, Y, PHI):
        """Each box's centre and half extents (axis-aligned bound of the
        turned box), and their derivatives along the turn."""
        p = self.p
        b = p.bx_body
        c, s = np.cos(PHI)[b], np.sin(PHI)[b]
        cx = X[b] + c * p.bx_cx - s * p.bx_cy
        cy = Y[b] + s * p.bx_cx + c * p.bx_cy
        ac, as_ = np.abs(c), np.abs(s)
        hx = ac * p.bx_a + as_ * p.bx_b
        hy = as_ * p.bx_a + ac * p.bx_b
        sc, ss = np.sign(c), np.sign(s)
        dcx = -(s * p.bx_cx + c * p.bx_cy)
        dcy = c * p.bx_cx - s * p.bx_cy
        dhx = -sc * s * p.bx_a + ss * c * p.bx_b
        dhy = ss * c * p.bx_a - sc * s * p.bx_b
        return cx, cy, hx, hy, dcx, dcy, dhx, dhy

    def _box_pairs(self, cx, cy):
        v = self._boxes
        if v is not None:
            moved = float(np.max(np.hypot(cx - v[0], cy - v[1]), initial=0.0))
            if moved < self.skin / 2.0:
                return v[2:]
        p = self.p
        R = self._bR
        pts = np.column_stack([cx, cy])
        # most boxes are small: a KD-tree at twice the 90th percentile reach;
        # the few big ones against every box
        r90 = float(np.percentile(R, 90)) if R.size else 0.0
        big = np.nonzero(R > r90)[0]
        pr = self._tree(pts).query_pairs(2 * r90 + self.skin, output_type='ndarray')
        i, j = (pr[:, 0], pr[:, 1]) if len(pr) else (np.zeros(0, int), np.zeros(0, int))
        if big.size:
            # the KD pairs with a big box are found again below: drop them
            isbig = np.zeros(R.size, bool)
            isbig[big] = True
            m = ~(isbig[i] | isbig[j])
            i, j = i[m], j[m]
            bi = np.repeat(big, R.size)
            bj = np.tile(np.arange(R.size), big.size)
            m = (bi != bj) & ~(isbig[bj] & (bj < bi))      # big-big once
            i = np.concatenate([i, bi[m]])
            j = np.concatenate([j, bj[m]])
        d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
        keep = ((p.bx_body[i] != p.bx_body[j]) & ((p.bx_side[i] & p.bx_side[j]) > 0)
                & (d < R[i] + R[j] + self.skin))
        i, j = i[keep], j[keep]
        # movable boxes against the fixed ones
        fi, fj = np.zeros(0, int), np.zeros(0, int)
        if p.fb_side.size:
            reach = R[:, None] + self._fbR[None, :] + self.skin
            d = np.hypot(cx[:, None] - self._fbc[None, :, 0], cy[:, None] - self._fbc[None, :, 1])
            fi, fj = np.nonzero((d < reach) & ((p.bx_side[:, None] & p.fb_side[None, :]) > 0))
        self._boxes = (cx.copy(), cy.copy(), i, j, fi, fj)
        self.rebuilds['boxes'] += 1
        return i, j, fi, fj

    def _overlap(self, X, Y, PHI, g):
        """k_ov x sum over box pairs of soft(px) x soft(py): along each axis
        the PENETRATION -- how far one box must move to clear the other,
        the overlap while they straddle and more once one holds the other,
        so a box inside another is still pushed to its nearer side --
        through a softplus of width `overlap_soft_mm` (smooth at the
        touch). The product is the overlap area while they straddle.
        Gradient on both boxes' bodies, in translation and turn."""
        p, P = self.p, self.P
        cx, cy, hx, hy, dcx, dcy, dhx, dhy = self._box_geometry(X, Y, PHI)
        i, j, fi, fj = self._box_pairs(cx, cy)
        beta = P.overlap_soft_mm
        k = self.kov
        nb = p.nb
        E, area, npair = 0.0, 0.0, 0

        def soft(o):
            z = np.clip(o / beta, -50, 50)
            return beta * np.logaddexp(0.0, z), 1.0 / (1.0 + np.exp(-z))

        def axis(ca, ha, cb, hb):
            # penetration: the lesser of a's high end past b's low end and
            # b's high end past a's low end; d/d(ca), d/d(ha), d/d(cb), d/d(hb)
            up = (ca + ha) - (cb - hb)
            dn = (cb + hb) - (ca - ha)
            first = up <= dn
            pen = np.where(first, up, dn)
            sgn = np.where(first, 1.0, -1.0)
            real = np.minimum(ca + ha, cb + hb) - np.maximum(ca - ha, cb - hb)
            return pen, real, sgn

        def apply(boxes, ex, ey, sx_, sy_):
            b = p.bx_body[boxes]
            g[0] += np.bincount(b, ex * sx_, nb)
            g[1] += np.bincount(b, ey * sy_, nb)
            g[2] += np.bincount(b, ex * (sx_ * dcx[boxes] + dhx[boxes])
                                + ey * (sy_ * dcy[boxes] + dhy[boxes]), nb)

        for kind in ('mov', 'fix'):
            if kind == 'mov':
                if not i.size:
                    continue
                a, b = i, j
                cxb, cyb, hxb, hyb = cx[b], cy[b], hx[b], hy[b]
            else:
                if not fi.size:
                    continue
                a, b = fi, fj
                R = p.fb_rect[b]
                cxb, cyb = (R[:, 0] + R[:, 2]) / 2, (R[:, 1] + R[:, 3]) / 2
                hxb, hyb = (R[:, 2] - R[:, 0]) / 2, (R[:, 3] - R[:, 1]) / 2
            pxv, rx, sgx = axis(cx[a], hx[a], cxb, hxb)
            pyv, ry, sgy = axis(cy[a], hy[a], cyb, hyb)
            real = (rx > 0) & (ry > 0)
            area += float(np.sum(np.where(real, rx * ry, 0.0)))
            npair += int(real.sum())
            if k <= 0:
                continue
            sx, dsx = soft(pxv)
            sy, dsy = soft(pyv)
            E += float(k * np.sum(sx * sy))
            ex = k * dsx * sy           # dE/d(penetration x)
            ey = k * dsy * sx
            # a: d pen / d ca = sgn, d pen / d ha = 1; b: -sgn and 1
            apply(a, ex, ey, sgx, sgy)
            if kind == 'mov':
                apply(b, ex, ey, -sgx, -sgy)
        return E, area, npair

    def density_map(self, X, Y, PHI, face=0):
        """The movable parts' density on face `face` (0 front, 1 back) per
        cell, as a share of the cell's area, and the cells' origin and size:
        `(grid, x0, y0, h)`."""
        if face not in self._charges:
            return None
        px, py, qx, qy = self.positions(X, Y, PHI)
        ms, mp, q, _body = self._charges[face]
        cell, w = self.den.cic(np.concatenate([qx[ms], px[mp]]), np.concatenate([qy[ms], py[mp]]))
        h = self.den.h
        return self.den.deposit_cic(cell, w, q) / (h * h), self.den.x0, self.den.y0, h

    def _entry_of_pin(self):
        e = getattr(self, '_eop', None)
        if e is None:
            p = self.p
            e = np.full(p.pin_body.size, -1)
            m = p.a_kind == 0
            e[p.a_ref[m]] = np.nonzero(m)[0]
            self._eop = e
        return e

    def swap_delta(self, pairs, px, py):
        """The weighted half-perimeter change of the nets a move touches,
        `pairs` [(pin a, pin b)] each trading its net with the other."""
        p = self.p
        eop = self._entry_of_pin()
        ent = [e for a, b in pairs for e in (eop[a], eop[b]) if e >= 0]
        if not ent:
            return 0.0
        nets = np.unique(p.a_net[ent])
        idx = np.nonzero(np.isin(p.a_net, nets))[0]
        ref = p.a_ref[idx].copy()
        kind = p.a_kind[idx]
        net = p.a_net[idx]
        fx = p.fx_x if p.fx_x.size else np.zeros(1)
        fy = p.fx_y if p.fx_y.size else np.zeros(1)
        # the nets' segments in `idx` (sorted by net already)
        starts = np.searchsorted(net, nets)

        def hpwl(rf):
            m0, m1 = kind == 0, kind == 1
            x = np.zeros(rf.size)
            y = np.zeros(rf.size)
            x[m0], y[m0] = px[rf[m0]], py[rf[m0]]
            x[m1], y[m1] = fx[rf[m1]], fy[rf[m1]]
            span = (np.maximum.reduceat(x, starts) - np.minimum.reduceat(x, starts)
                    + np.maximum.reduceat(y, starts) - np.minimum.reduceat(y, starts))
            return float(np.dot(p.a_w[nets], span))
        before = hpwl(ref)
        pos = {int(e): i for i, e in enumerate(idx)}
        for a, b in pairs:
            ea, eb = eop[a], eop[b]
            if ea >= 0:
                ref[pos[int(ea)]] = b
            if eb >= 0:
                ref[pos[int(eb)]] = a
        return hpwl(ref) - before

    def apply_swap(self, pairs):
        """Trade the nets of each (pin a, pin b): their attraction entries,
        reaches and routing demands."""
        p = self.p
        eop = self._entry_of_pin()
        for a, b in pairs:
            ea, eb = eop[a], eop[b]
            if ea >= 0:
                p.a_ref[ea] = b
            if eb >= 0:
                p.a_ref[eb] = a
            eop[a], eop[b] = eb, ea
            p.pad_r[[a, b]] = p.pad_r[[b, a]]
            p.pad_dem[[a, b]] = p.pad_dem[[b, a]]
        self._allr = np.concatenate([p.pad_r, p.fx_r])
        self._charges = {f: (ms, mp, np.concatenate([p.smp_q[ms], self.P.demand_weight * p.pad_dem[mp]]),
                             body) for f, (ms, mp, _q, body) in self._charges.items()}

    def positions(self, X, Y, PHI):
        p = self.p
        c, s = np.cos(PHI), np.sin(PHI)
        b = p.pin_body
        px = X[b] + c[b] * p.pin_lx - s[b] * p.pin_ly
        py = Y[b] + s[b] * p.pin_lx + c[b] * p.pin_ly
        b = p.smp_body
        qx = X[b] + c[b] * p.smp_lx - s[b] * p.smp_ly
        qy = Y[b] + s[b] * p.smp_lx + c[b] * p.smp_ly
        return px, py, qx, qy

    def _spring_rows(self, px, py):
        """Each spring's nearest candidate row at pin positions (px, py):
        (row, dx, dy, d, over), `over` its stretch past its free length."""
        p = self.p
        mv = p.sp_tpin >= 0
        tx = np.where(mv, px[np.maximum(p.sp_tpin, 0)], p.sp_tx)
        ty = np.where(mv, py[np.maximum(p.sp_tpin, 0)], p.sp_ty)
        dx, dy = px[p.sp_subj] - tx, py[p.sp_subj] - ty
        d2 = dx * dx + dy * dy
        # the nearest row per spring: sort by (spring, d2), first of each
        o = np.lexsort((d2, p.sp_id))
        first = np.ones(o.size, bool)
        first[1:] = p.sp_id[o][1:] != p.sp_id[o][:-1]
        row = o[first]
        d = np.sqrt(d2[row]) + 1e-9
        return row, dx[row], dy[row], d, np.maximum(0.0, d - p.sp_free[p.sp_id[row]])

    def spring_energy(self, px, py):
        """The springs' energy at pin positions (px, py)."""
        if not self.p.sp_id.size:
            return 0.0
        over = self._spring_rows(px, py)[4]
        return float(self.P.claim_weight * np.sum(over * over))

    def rebuild_crossings(self, px, py):
        """The trees at pin positions (px, py) and their pairs of segments of
        two nets whose boxes are within `cross_margin_mm` + 3 eps."""
        rn = self.rn
        seg, net, a, b, _l = rn.trees(px, py, range(rn.n_nets))
        if seg.shape[0] < 2:
            self._xp = None
            return 0
        lo = np.minimum(seg[:, :2], seg[:, 2:])
        hi = np.maximum(seg[:, :2], seg[:, 2:])
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        m = self.P.cross_margin_mm + 3.0 * self.cross_eps
        tree = self._tree(mid)
        reach = np.hypot(half[:, 0], half[:, 1]) + float(np.hypot(half[:, 0], half[:, 1]).max()) + m
        I, J = [], []
        for i, js in enumerate(tree.query_ball_point(mid, reach)):
            js = np.asarray(js, int)
            js = js[js > i]
            I.append(np.full(js.size, i))
            J.append(js)
        i, j = np.concatenate(I), np.concatenate(J)
        keep = ((net[i] != net[j])
                & (np.abs(mid[i, 0] - mid[j, 0]) <= half[i, 0] + half[j, 0] + m)
                & (np.abs(mid[i, 1] - mid[j, 1]) <= half[i, 1] + half[j, 1] + m))
        i, j = i[keep], j[keep]
        self._xp = (a[i], b[i], a[j], b[j])
        return i.size

    def _acc(self, g, fx, fy, x, y, body, X, Y):
        nb = self.p.nb
        g[0] += np.bincount(body, fx, nb)
        g[1] += np.bincount(body, fy, nb)
        g[2] += np.bincount(body, -fx * (y - Y[body]) + fy * (x - X[body]), nb)

    def __call__(self, X, Y, PHI, lam, gamma):
        import time as _time
        p, P = self.p, self.P
        g = np.zeros((3, p.nb))
        T = {}
        t0 = _time.perf_counter()
        px, py, qx, qy = self.positions(X, Y, PHI)
        t0 = self._t('positions', t0)
        # attraction
        if p.a_net.size:
            k = p.a_kind
            ref = p.a_ref
            ax = np.empty(ref.size)
            ay = np.empty(ref.size)
            m0, m1, m2 = k == 0, k == 1, k == 2
            ax[m0], ay[m0] = px[ref[m0]], py[ref[m0]]
            ax[m1], ay[m1] = p.fx_x[ref[m1]], p.fx_y[ref[m1]]
            ax[m2], ay[m2] = X[ref[m2]], Y[ref[m2]]
            if P.attraction in ('wa', 'lse'):
                ex, gx = wa_grad(ax, p.a_net, p.a_starts, p.a_w, gamma, P.attraction)
                ey, gy = wa_grad(ay, p.a_net, p.a_starts, p.a_w, gamma, P.attraction)
                T['wire'] = ex + ey
            elif P.attraction == 'gauss':
                T['wire'], gx, gy = gauss_grad(ax, ay, p.a_net, p.a_starts, p.a_counts, p.a_w, P.gauss_mm)
            else:
                raise ValueError(f"unknown attraction profile {P.attraction!r}")
            r0 = ref[m0]
            self._acc(g, gx[m0], gy[m0], px[r0], py[r0], p.pin_body[r0], X, Y)
            if m2.any():
                g[0] += np.bincount(ref[m2], gx[m2], p.nb)
                g[1] += np.bincount(ref[m2], gy[m2], p.nb)
        t0 = self._t('attraction', t0)
        # springs: each to its nearest candidate
        T['springs'] = 0.0
        if p.sp_id.size:
            row, dx, dy, d, over = self._spring_rows(px, py)
            T['springs'] = float(P.claim_weight * np.sum(over * over))
            f = 2 * P.claim_weight * over / d
            s = p.sp_subj[row]
            self._acc(g, f * dx, f * dy, px[s], py[s], p.pin_body[s], X, Y)
            tp = p.sp_tpin[row]
            m = tp >= 0
            if m.any():
                k2 = tp[m]
                self._acc(g, -(f * dx)[m], -(f * dy)[m], px[k2], py[k2], p.pin_body[k2], X, Y)
        t0 = self._t('springs', t0)
        # smoothed crossings of the nets' trees, on the pairs found at the last rebuild
        T['crossings'] = 0.0
        if self.cross_k > 0 and self._xp is not None and self._xp[0].size:
            rn = self.rn
            ex, ey = rn._points(px, py)
            pts = [np.column_stack([ex[e], ey[e]]) for e in self._xp]
            E, *G = cross_energy(*pts, self.cross_eps)
            T['crossings'] = self.cross_k * E
            for e, ge in zip(self._xp, G):
                m = rn.e_pin[e]
                pin = rn._pin_ref[e[m]]
                self._acc(g, self.cross_k * ge[m, 0], self.cross_k * ge[m, 1], px[pin], py[pin],
                          p.pin_body[pin], X, Y)
        t0 = self._t('crossings', t0)
        # functional blocks: E = k_group sum w_i (|x_i - c| - R)+^2, c the
        # block's weighted centre (its own dependence on x_i included), so
        # a member strayed out of its block is pulled home and one inside
        # its block's disc feels nothing
        T['groups'] = 0.0
        if P.group_weight > 0 and p.gm_body.size:
            b, gi, w = p.gm_body, p.gm_grp, p.gm_w
            ng = p.g_R.size
            W = np.bincount(gi, w, ng)
            cx = (np.bincount(gi, w * X[b], ng) / W)[gi]
            cy = (np.bincount(gi, w * Y[b], ng) / W)[gi]
            dx, dy = X[b] - cx, Y[b] - cy
            r = np.hypot(dx, dy) + 1e-9
            over = np.maximum(0.0, r - p.g_R[gi])
            T['groups'] = float(P.group_weight * np.sum(w * over * over))
            fx = 2 * P.group_weight * w * over * dx / r
            fy = 2 * P.group_weight * w * over * dy / r
            # the centre moves by w_i / W with each member
            fx -= (w / W[gi]) * np.bincount(gi, fx, ng)[gi]
            fy -= (w / W[gi]) * np.bincount(gi, fy, ng)[gi]
            g[0] += np.bincount(b, fx, p.nb)
            g[1] += np.bincount(b, fy, p.nb)
        t0 = self._t('groups', t0)
        # short-range pad repulsion, on a Verlet list
        T['pads'] = 0.0
        kp = P.pad_weight * self.rep
        if kp > 0 and p.pin_body.size:
            allx = np.concatenate([px, p.fx_x])
            ally = np.concatenate([py, p.fx_y])
            i, j = self._pad_pairs(px, py, allx, ally)
            if i.size:
                allr, allb = self._allr, self._allb
                dx, dy = allx[i] - allx[j], ally[i] - ally[j]
                d = np.hypot(dx, dy) + 1e-9
                e, de = repulsion_profile(P.repulsion, d, allr[i] + allr[j])
                T['pads'] = float(kp * e.sum())
                fx = kp * de * dx / d
                fy = kp * de * dy / d
                mi = allb[i] >= 0
                ki = i[mi]
                self._acc(g, fx[mi], fy[mi], allx[ki], ally[ki], allb[ki], X, Y)
                mj = allb[j] >= 0
                kj = j[mj]
                self._acc(g, -fx[mj], -fy[mj], allx[kj], ally[kj], allb[kj], X, Y)
        t0 = self._t('pads', t0)
        # courtyard overlap (same face), on a Verlet list
        T['overlap'], T['overlap_mm2'], T['overlap_pairs'] = 0.0, 0.0, 0
        if p.bx_body.size and (self.kov > 0 or self.measure_overlap):
            Eo, area, npair = self._overlap(X, Y, PHI, g)
            T['overlap'], T['overlap_mm2'], T['overlap_pairs'] = Eo, area, npair
        t0 = self._t('overlap', t0)
        # density, per face
        den = self.den
        h2 = den.h * den.h
        Ed, over, tot = 0.0, 0.0, 0.0
        for f in self.faces:
            ms, mp, q, body = self._charges[f]
            # every charge of the face, courtyard samples then pad demands
            x = np.concatenate([qx[ms], px[mp]])
            y = np.concatenate([qy[ms], py[mp]])
            cell, w = den.cic(x, y)
            rho_m = den.deposit_cic(cell, w, q)
            if P.density in ('cap', 'both'):
                # E = 1/2 sum (G * rho - allow)+^2 / h^2, G a Gaussian of
                # cap_smooth_mm: only the excess over the allowed fill
                # pushes, felt over the Gaussian's reach
                from scipy.ndimage import gaussian_filter
                sc = P.cap_smooth_mm / den.h
                ex = np.maximum(0.0, gaussian_filter(rho_m + self.fixed_q[f], sc, mode='constant')
                                - self.allow[f])
                Ed += 0.5 * float(np.sum(ex * ex)) / h2
                psi = gaussian_filter(ex, sc, mode='constant') / h2
                gxf, gyf = np.gradient(psi, den.h)
            if P.density in ('poisson', 'both'):
                # both: the cap holds a block together at short range, a
                # weak screened field spreads the blocks over the board
                ws = 1.0 if P.density == 'poisson' else P.spread_weight
                rho = rho_m + p.static[f]
                psi2, gx2, gy2 = den.potential(rho / h2, P.smooth_mm / den.h)
                Ed += ws * 0.5 * float(np.sum(rho * psi2))
                if P.density == 'poisson':
                    gxf, gyf = gx2, gy2
                else:
                    gxf, gyf = gxf + ws * gx2, gyf + ws * gy2
            over += float(np.maximum(0.0, rho_m - self.cap[f]).sum())
            tot += float(rho_m.sum())
            if lam > 0:
                fx, fy = den.sample_cic(gxf, gyf, cell, w)
                lq = lam * q
                self._acc(g, lq * fx, lq * fy, x, y, body, X, Y)
        T['density'] = Ed
        T['overflow'] = over / max(1e-9, tot)
        t0 = self._t('density', t0)
        # bounds (translation only): the usable box, zones, keep-outs
        cx, cy = _corners(p, X, Y, PHI)
        bx0, bx1 = X + cx.min(0), X + cx.max(0)
        by0, by1 = Y + cy.min(0), Y + cy.max(0)
        kb = P.edge_weight
        kk = P.edge_weight + self.kov       # keep-outs and zones harden with the overlap
        u = p.usable
        Eb = 0.0
        for gi, lo, hi, a0, a1 in ((0, bx0, bx1, u[0], u[2]), (1, by0, by1, u[1], u[3])):
            o0 = np.where(p.turnable, np.maximum(0.0, a0 - lo), 0.0)
            o1 = np.where(p.turnable, np.maximum(0.0, hi - a1), 0.0)
            Eb += float(kb * np.sum(o0 * o0 + o1 * o1))
            g[gi] += 2 * kb * (o1 - o0)
        if p.bt_body.size:
            b, R = p.bt_body, p.bt_rect
            lx, hx, ly, hy = bx0[b], bx1[b], by0[b], by1[b]
            # inside a zone: the escape on each side
            i0, i1 = np.maximum(0.0, R[:, 0] - lx), np.maximum(0.0, hx - R[:, 2])
            j0, j1 = np.maximum(0.0, R[:, 1] - ly), np.maximum(0.0, hy - R[:, 3])
            ins = p.bt_in
            Eb += float(kk * np.sum(np.where(ins, i0 * i0 + i1 * i1 + j0 * j0 + j1 * j1, 0.0)))
            gxz = np.where(ins, 2 * kk * (i1 - i0), 0.0)
            gyz = np.where(ins, 2 * kk * (j1 - j0), 0.0)
            # out of a keep-out: by its nearest exit that stays on the
            # board (a band along the edge is left inward, never outward)
            ox = np.minimum(hx, R[:, 2]) - np.maximum(lx, R[:, 0])
            oy = np.minimum(hy, R[:, 3]) - np.maximum(ly, R[:, 1])
            hit = (~ins) & (ox > 0) & (oy > 0)
            ex = np.stack([hx - R[:, 0], R[:, 2] - lx, hy - R[:, 1], R[:, 3] - ly])
            ok = np.stack([lx - ex[0] >= u[0] - 1e-6, hx + ex[1] <= u[2] + 1e-6,
                           ly - ex[2] >= u[1] - 1e-6, hy + ex[3] <= u[3] + 1e-6])
            way = np.argmin(np.where(ok, ex, np.inf), axis=0)
            dd = np.where(hit, ex[way, np.arange(way.size)], 0.0)
            Eb += float(kk * np.sum(dd * dd))
            # moving left / down lowers exits 0 and 2; right / up, 1 and 3
            sgn = np.array([1.0, -1.0, 1.0, -1.0])[way]
            gxz += np.where(hit & (way < 2), 2 * kk * dd * sgn, 0.0)
            gyz += np.where(hit & (way >= 2), 2 * kk * dd * sgn, 0.0)
            g[0] += np.bincount(b, gxz, p.nb)
            g[1] += np.bincount(b, gyz, p.nb)
        T['bounds'] = Eb
        g[2][~p.turnable] = 0.0
        self._t('bounds', t0)
        return T, g


# -- from the model to the problem ---------------------------------------------

def block_list(model, movable, sheets=False):
    """The functional blocks held together: the declared groups when there
    are any, else each schematic sheet instance the board's footprints come
    from (their `(path ...)`), so a board needs nothing beyond its schematic."""
    if model.groups and not sheets:
        return list(model.groups)
    from .groups import _from_sheet
    return [(n, m) for n, m in sorted(_from_sheet(model.pcb, movable).items()) if len(m) >= 2]


def build_problem(model, movable, P: ForceParams, plane_nets=(), seed: int = 0):
    """(problem, X, Y, PHI, meta) for `movable` on `model`: the arrays the
    kernel reads, the start, and what `place` needs to write the result
    back. Template members are seated now: on a movable chip they ride on it
    (its points), on a fixed one they stay at their template pose."""
    from . import simple_anneal as sa
    from .edge_seat import project
    st, pcb = model.state, model.pcb
    nrng = np.random.default_rng(seed)
    clr = P.clearance_mm or st.clearance
    track = P.track_mm or clr
    via = P.via_mm or 0.5
    mov = [r for r in movable if r in st.parts]
    movset = set(mov)
    seats = dict(getattr(model, 'edge_seats', {}) or {})
    containers = set(getattr(st, 'container_refs', ()) or ())
    planes = set(plane_nets) | set(model.rnets) | set(model.ground)
    ux0, uy0, ux1, uy1 = st.usable

    rider_of: Dict[str, Tuple[str, int]] = {}
    for i, (anc, m) in enumerate(model.tmpl):
        r = m['ref']
        if r not in movset:
            continue
        if anc in movset:
            rider_of[r] = (anc, i)
        else:
            x, y, rot = model.template_target(i)
            st.parts[r].ensure_rotation(rot)
            sa._place(st, r, x, y, rot)
    stay = {m['ref'] for a, m in model.tmpl if a not in movset}
    bodies = [r for r in mov if r not in rider_of and r not in stay]
    bidx = {r: i for i, r in enumerate(bodies)}
    nb = len(bodies)
    fixedset = set(st.parts) - set(bodies) - set(rider_of)

    cx, cy = (ux0 + ux1) / 2.0, (uy0 + uy1) / 2.0
    X = cx + nrng.normal(0, P.spread_sigma * (ux1 - ux0), nb)
    Y = cy + nrng.normal(0, P.spread_sigma * (uy1 - uy0), nb)
    rot0 = np.array([st.parts[r].rot for r in bodies], dtype=float)
    PHI = np.radians(-rot0)
    turnable = np.array([r not in seats for r in bodies], dtype=bool)
    for r in seats:
        i = bidx.get(r)
        if i is not None:
            x, y, rot = project(seats[r], (X[i], Y[i], rot0[i]), rot0[i])
            X[i], Y[i], PHI[i] = x, y, math.radians(-rot)

    def reach(p, net):
        half = max(p.size_x, p.size_y) / 2.0
        return half + ((via + clr) if net in planes else (track + clr)) / 2.0

    def demand(p, net):
        if net in planes:
            return (via + clr) ** 2 + (track + clr) * 0.5 * P.escape_mm
        return (track + clr) * P.escape_mm

    pins = {k: [] for k in ('body', 'lx', 'ly', 'r', 'side', 'dem', 'net')}
    fxs = {k: [] for k in ('x', 'y', 'r', 'side', 'dem', 'net')}
    smp = {k: [] for k in ('body', 'lx', 'ly', 'q', 'side')}
    boxes = {k: [] for k in ('body', 'cx', 'cy', 'a', 'b', 'side')}
    fboxes = []
    pin_key: Dict[Tuple[str, str], List[int]] = {}
    fixed_pin: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    sample_step = P.bin_mm / 2.0
    for r, part in st.parts.items():
        fpads = [p for p in pcb.footprints[r].pads if p.net_id > 0] if r in pcb.footprints else []
        if len(fpads) != len(part.pads_local):
            fpads = [None] * len(part.pads_local)
        side = _side_mask(part.sides)
        here = _side_mask((part.side,))
        if r in bidx or r in rider_of:
            if r in bidx:
                b, frame = bidx[r], None
            else:
                anc, i = rider_of[r]
                b = bidx[anc]
                A = st.parts[anc]
                frame = ((A.x, A.y, A.rot), model.template_target(i))
            lxs = np.array([q[0] for q in part.pads_local])
            lys = np.array([q[1] for q in part.pads_local])
            if frame is not None and lxs.size:
                gx, gy = _to_global(lxs, lys, frame[1])
                lxs, lys = _to_local(gx, gy, frame[0])
            for (lx, ly, (_x, _y, net)), p in zip(zip(lxs, lys, part.pads_local), fpads):
                pin_key.setdefault((r, str(p.pad_number) if p is not None else ''), []).append(len(pins['body']))
                pins['body'].append(b)
                pins['lx'].append(float(lx))
                pins['ly'].append(float(ly))
                pins['net'].append(net)
                pins['r'].append(reach(p, net) if p is not None else 0.3)
                pins['side'].append(side if (p is None or p.drill) else here)
                pins['dem'].append(demand(p, net) if p is not None else 0.0)
            if r in containers:
                continue
            b0 = part.bounds_by_rot[0.0]
            kx = np.array([b0[0], b0[2], b0[0], b0[2]])
            ky = np.array([b0[1], b0[1], b0[3], b0[3]])
            if frame is not None:
                gx, gy = _to_global(kx, ky, frame[1])
                kx, ky = _to_local(gx, gy, frame[0])
            boxes['body'].append(b)
            boxes['cx'].append(float(kx.min() + kx.max()) / 2.0)
            boxes['cy'].append(float(ky.min() + ky.max()) / 2.0)
            boxes['a'].append(float(kx.max() - kx.min()) / 2.0)
            boxes['b'].append(float(ky.max() - ky.min()) / 2.0)
            boxes['side'].append(side)
            sx, sy, sq = _samples(part.bounds_by_rot[0.0], sample_step)
            if frame is not None:
                gx, gy = _to_global(sx, sy, frame[1])
                sx, sy = _to_local(gx, gy, frame[0])
            smp['body'] += [b] * len(sx)
            smp['lx'] += list(sx)
            smp['ly'] += list(sy)
            smp['q'] += list(sq)
            smp['side'] += [side] * len(sx)
        else:
            if r not in containers:
                fboxes.append(tuple(part.rect()) + (side,))
            for (gx, gy, net), p in zip(part.pad_globals(), fpads):
                fixed_pin.setdefault((r, str(p.pad_number) if p is not None else ''), []).append((gx, gy))
                fxs['x'].append(gx)
                fxs['y'].append(gy)
                fxs['net'].append(net)
                fxs['r'].append(reach(p, net) if p is not None else 0.3)
                fxs['side'].append(side if (p is None or p.drill) else here)
                fxs['dem'].append(demand(p, net) if p is not None else 0.0)
    A = lambda v, t=float: np.asarray(v, dtype=t)  # noqa: E731
    pin_body, pin_lx, pin_ly = A(pins['body'], int), A(pins['lx']), A(pins['ly'])
    smp_body, smp_lx, smp_ly, smp_q = A(smp['body'], int), A(smp['lx']), A(smp['ly']), A(smp['q'])
    rg = np.ones(nb)
    body_box = np.zeros((nb, 4))
    if smp_body.size:
        wsum = np.bincount(smp_body, smp_q, nb)
        r2 = np.bincount(smp_body, smp_q * (smp_lx ** 2 + smp_ly ** 2), nb)
        rg = np.maximum(0.5, np.sqrt(r2 / np.maximum(wsum, 1e-9)))
    bx_body = A(boxes['body'], int)
    bx_cx, bx_cy, bx_a, bx_b = A(boxes['cx']), A(boxes['cy']), A(boxes['a']), A(boxes['b'])
    if bx_body.size:
        for k, (fn, src) in enumerate(((np.minimum, bx_cx - bx_a), (np.minimum, bx_cy - bx_b),
                                       (np.maximum, bx_cx + bx_a), (np.maximum, bx_cy + bx_b))):
            init = np.full(nb, np.inf if fn is np.minimum else -np.inf)
            fn.at(init, bx_body, src)
            body_box[:, k] = np.where(np.isfinite(init), init, 0.0)
    fb = A(fboxes).reshape(-1, 5)

    # attraction: every pin of a wired net that is not a plane, fixed pads too
    pin_net, fx_net = pins['net'], fxs['net']
    wnets = sorted({n for n in pin_net if n and n not in planes})
    widx = {n: i for i, n in enumerate(wnets)}
    w_of = [1.0 if n in model.signal else model.rw for n in wnets]
    kinds, refs, nets = [], [], []
    for k, n in enumerate(pin_net):
        if n in widx:
            kinds.append(0); refs.append(k); nets.append(widx[n])   # noqa: E702
    for k, n in enumerate(fx_net):
        if n in widx:
            kinds.append(1); refs.append(k); nets.append(widx[n])   # noqa: E702
    if P.group_weight > 0 and P.group_model == 'hpwl':
        for gi, (_n, members) in enumerate(model.groups):
            bs = sorted({bidx[r] for r in members if r in bidx}
                        | {bidx[rider_of[r][0]] for r in members if r in rider_of})
            if len(bs) >= 2:
                for b in bs:
                    kinds.append(2); refs.append(b); nets.append(len(wnets) + gi)  # noqa: E702
    a_kind, a_ref, a_net = A(kinds, int), A(refs, int), A(nets, int)
    a_w = np.zeros(0)
    a_starts = np.zeros(0, int)
    a_counts = np.zeros(0)
    if a_net.size:
        cnt = np.bincount(a_net)
        keep = cnt[a_net] >= 2
        # a net with only fixed pins pulls nothing
        has_mov = np.bincount(a_net, (a_kind != 1).astype(float))
        keep &= has_mov[a_net] > 0
        a_kind, a_ref, a_net = a_kind[keep], a_ref[keep], a_net[keep]
        o = np.argsort(a_net, kind='stable')
        a_kind, a_ref, a_net = a_kind[o], a_ref[o], a_net[o]
        uniq, a_net = np.unique(a_net, return_inverse=True)
        a_w = A([w_of[u] if u < len(wnets) else P.group_weight for u in uniq])
        a_starts = np.searchsorted(a_net, np.arange(len(uniq)))
        a_counts = np.bincount(a_net).astype(float)

    # springs: the claims, and the decaps (matched later, `match_decaps`)
    def pts_of(ref, num):
        if ref in bidx or ref in rider_of:
            return [(k, 0.0, 0.0) for k in pin_key.get((ref, str(num)), [])]
        return [(-1, x, y) for x, y in fixed_pin.get((ref, str(num)), [])]
    claim_rows = []     # (subject pin, [(tpin, tx, ty)], free length)
    for c in model.claims:
        s, p, legs = c
        # a claim's spring is free up to its limit's share: a decap at 2 mm
        # is pulled home, a protector within 8 mm of its connector only kept
        # in reach (its spring would otherwise pull it onto the pins)
        lim = model._cmax.get(id(c))
        free = max(P.claim_free_mm, P.claim_free_share * lim) if lim else P.claim_free_mm
        for spad, ppads in legs:
            subj = [t[0] for t in pts_of(s, spad) if t[0] >= 0]
            tg = [t for q in ppads for t in pts_of(p, q)]
            if subj and tg:
                claim_rows.append((subj[0], tg, free))
            else:       # the subject fixed: its partner's pad is pulled to it
                sub2 = [t[0] for q in ppads for t in pts_of(p, q) if t[0] >= 0]
                tg2 = pts_of(s, spad)
                if sub2 and tg2:
                    claim_rows.append((sub2[0], tg2, free))
    decap_rows = []     # (cap pin, [(tpin, tx, ty)], set key)
    for cap, rail in model.decaps:
        dp = model.decap_pins.get(cap) or model.supply.get(rail) or []
        if cap not in st.parts or not dp or cap not in pcb.footprints:
            continue
        cp = []
        for num in [str(q.pad_number) for q in pcb.footprints[cap].pads if q.net_id == rail]:
            cp = [t[0] for t in pts_of(cap, num) if t[0] >= 0]
            if cp:
                break
        tg = [t for ic, n in dp for t in pts_of(ic, n)]
        if cp and tg:
            decap_rows.append((cp[0], tg, tuple(sorted(dp))))

    # zones and keep-outs binding each body (rect terms)
    bt_body, bt_in, bt_rect = [], [], []
    ko_count: Dict[Tuple, int] = {}
    for r in bodies:
        for t in (st.intent_spec_for(r) or ()):
            rect = t.rect if t.rect is not None else (t.entry or {}).get('rect')
            circ = (t.entry or {}).get('circle') if t.rule == 'keepout' else None
            if rect is None and circ:       # a round keep-out, by its square
                rect = (circ[0] - circ[2], circ[1] - circ[2], circ[0] + circ[2], circ[1] + circ[2])
            if rect is None:
                continue
            rect = tuple(float(v) for v in rect)
            if t.rule == 'zone_containment':
                bt_body.append(bidx[r]); bt_in.append(True); bt_rect.append(rect)  # noqa: E702
            elif t.rule in ('keepout', 'zone_exclusive'):
                bt_body.append(bidx[r]); bt_in.append(False); bt_rect.append(rect)  # noqa: E702
                if t.rule == 'keepout':
                    ko_count[rect] = ko_count.get(rect, 0) + 1

    # density: what the movable parts cannot use, the fixed charge, targets
    bb = pcb.board_info.board_bounds or (ux0, uy0, ux1, uy1)
    den = Density(bb[0], bb[1], bb[2], bb[3], P.bin_mm)
    nx, ny = den.nx, den.ny
    inside = np.zeros((nx, ny))
    _rect_cover(den, (ux0, uy0, ux1, uy1), inside)
    gate = st.edge_gate if getattr(st, 'edge_gate', None) is not None and st.edge_gate.active else None
    if gate is not None:
        xs = den.x0 + den.h * np.arange(nx)
        ys = den.y0 + den.h * np.arange(ny)
        for i, j in zip(*np.nonzero(inside > 0)):
            cell = (xs[i], ys[j], xs[i] + den.h, ys[j] + den.h)
            if not gate.rect_bbox_decides(cell) and gate.rect_outside_amount(cell) > 1e-6:
                inside[i, j] = 0.0
    inside = np.clip(inside, 0.0, 1.0)
    fx_x, fx_y = A(fxs['x']), A(fxs['y'])
    fx_side, fx_dem = A(fxs['side'], int), A(fxs['dem'])
    pad_side, pad_dem = A(pins['side'], int), A(pins['dem'])
    smp_side = A(smp['side'], int)
    blocked = np.zeros((2, nx, ny))
    static = np.zeros((2, nx, ny))
    target = np.zeros(2)
    h2 = den.h * den.h
    for f in (0, 1):
        bit = 1 << f
        cover = np.zeros((nx, ny))
        for r in fixedset:
            if r not in containers and _side_mask(st.parts[r].sides) & bit:
                _rect_cover(den, st.parts[r].rect(), cover)
        for rect, c in ko_count.items():
            if c >= 0.9 * nb:
                _rect_cover(den, rect, cover)
        blocked[f] = 1.0 - np.clip(inside - cover, 0.0, 1.0)
        q = float(smp_q[(smp_side & bit) > 0].sum() + P.demand_weight * pad_dem[(pad_side & bit) > 0].sum())
        target[f] = q / max(1e-6, h2 * (1.0 - blocked[f]).sum())
        static[f] = target[f] * h2 * blocked[f]
        m = (fx_side & bit) > 0
        if m.any() and P.demand_weight > 0:
            static[f] += P.demand_weight * den.deposit(fx_x[m], fx_y[m], fx_dem[m])

    prob = Problem(
        nb=nb, pin_body=pin_body, pin_lx=pin_lx, pin_ly=pin_ly, pad_r=A(pins['r']),
        pad_side=pad_side, pad_dem=pad_dem, fx_x=fx_x, fx_y=fx_y, fx_r=A(fxs['r']),
        fx_side=fx_side, fx_dem=fx_dem, smp_body=smp_body, smp_lx=smp_lx, smp_ly=smp_ly,
        smp_q=smp_q, smp_side=smp_side, a_kind=a_kind, a_ref=a_ref, a_net=a_net,
        a_starts=a_starts, a_counts=a_counts, a_w=a_w,
        sp_id=np.zeros(0, int), sp_subj=np.zeros(0, int), sp_tpin=np.zeros(0, int),
        sp_tx=np.zeros(0), sp_ty=np.zeros(0), sp_free=np.zeros(0),
        body_box=body_box, turnable=turnable, usable=(ux0, uy0, ux1, uy1),
        bt_body=A(bt_body, int), bt_in=A(bt_in, bool),
        bt_rect=A(bt_rect).reshape(-1, 4), rg=rg,
        bx_body=bx_body, bx_cx=bx_cx, bx_cy=bx_cy, bx_a=bx_a + clr / 2.0, bx_b=bx_b + clr / 2.0,
        bx_side=A(boxes['side'], int), fb_rect=fb[:, :4] + np.array([-1, -1, 1, 1]) * clr / 2.0,
        fb_side=fb[:, 4].astype(int),
        grid=(den.x0, den.y0, den.h), shape=(nx, ny), blocked=blocked, static=static,
        target=target)
    if P.group_weight > 0 and P.group_model == 'centroid':
        # each block's members (a template member counts as its chip, which
        # carries it) weighted by their courtyard area; the block's free
        # radius is that of a disc holding its area at the allowed fill
        mass = np.bincount(smp_body, smp_q, nb) if smp_body.size else np.ones(nb)
        fill = max(1e-3, min(1.0, P.cap_ratio * float(target[0])))
        gb, gg, gr = [], [], []
        for _n, members in block_list(model, mov, P.group_source == 'sheet'):
            bs = sorted({bidx[r] for r in members if r in bidx}
                        | {bidx[rider_of[r][0]] for r in members if r in rider_of})
            if len(bs) < 2:
                continue
            gi = len(gr)
            gb += bs
            gg += [gi] * len(bs)
            gr.append(P.group_free * math.sqrt(float(mass[bs].sum()) / (math.pi * fill)))
        prob.gm_body, prob.gm_grp = A(gb, int), A(gg, int)
        prob.gm_w, prob.g_R = mass[prob.gm_body] if gb else np.zeros(0), A(gr)
    # each body's block (-1: none), for the part moves: a trade stays in one
    block_of = np.full(nb, -1)
    for gi, (_n, members) in enumerate(block_list(model, mov, P.group_source == 'sheet')):
        for r in members:
            b = bidx.get(r)
            if b is not None and block_of[b] < 0:
                block_of[b] = gi
    meta = {'bodies': bodies, 'bidx': bidx, 'rider_of': rider_of, 'seats': seats,
            'pin_key': pin_key, 'block_of': block_of,
            'base': np.array([st.parts[r].orig_rot % 90.0 for r in bodies]),
            'claim_rows': claim_rows, 'decap_rows': decap_rows}
    return prob, X, Y, PHI, meta


def set_springs(prob: Problem, rows):
    """`rows` [(subject pin, [(tpin, tx, ty)], free)] into the problem's spring arrays."""
    sid, subj, tp, tx, ty = [], [], [], [], []
    for i, (k, tg, _free) in enumerate(rows):
        for t in tg:
            sid.append(i); subj.append(k); tp.append(t[0]); tx.append(t[1]); ty.append(t[2])  # noqa: E702
    prob.sp_id = np.asarray(sid, int)
    prob.sp_subj = np.asarray(subj, int)
    prob.sp_tpin = np.asarray(tp, int)
    prob.sp_tx = np.asarray(tx, float)
    prob.sp_ty = np.asarray(ty, float)
    prob.sp_free = np.asarray([r[2] for r in rows], float)


def match_decaps(decap_rows, px, py, free):
    """Each decap to ONE pin of its rail's set: the nearest (cap, pin) pairs
    first, every pin taking one cap before any takes two (the anneal's
    spread over the pins)."""
    out = []
    by_set: Dict[tuple, List[int]] = {}
    for i, row in enumerate(decap_rows):
        by_set.setdefault(row[2], []).append(i)
    for rows in by_set.values():
        tg = decap_rows[rows[0]][1]
        T = np.array([(px[t[0]], py[t[0]]) if t[0] >= 0 else (t[1], t[2]) for t in tg])
        caps = np.array([decap_rows[i][0] for i in rows])
        D = np.hypot(T[None, :, 0] - px[caps][:, None], T[None, :, 1] - py[caps][:, None])
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        used = np.zeros(len(tg), int)
        got = {}
        level = 0
        while len(got) < len(rows):
            for ci, tj in order:
                if ci not in got and used[tj] <= level:
                    got[ci] = tj
                    used[tj] += 1
            level += 1
        for ci, tj in got.items():
            out.append((int(caps[ci]), [tg[tj]], free))
    return out


def optimise(prob: Problem, P: ForceParams, X, Y, PHI, *, base, claim_rows=(), decap_rows=(),
             project=None, log=None, swaps=None, snapshot=None, parts=None):
    """Lower the energy from (X, Y, PHI): Nesterov with a Barzilai-Borwein
    step, lambda rising each step, the turns rounded to `base` + k x 90
    once the overflow falls under `turn_overflow`, then `relax` steps with
    turns held. `project(X, Y, PHI)` is applied in place after each step.

    `swaps` (a `SwapMoves`) is tried every `swap_every` steps once the
    blocks have formed and before the relax (the overflow under
    `swap_from`): each allowed move priced on the nets it touches alone,
    the best that lowers the wire taken, until none does. `parts` (a
    `PartMoves`) is run every `move_every` steps once the overflow is under
    `move_from`, the relax included; the momentum restarts after any move.
    `snapshot(info)` is called every `snapshot_every` steps and at each
    phase change; its time is kept out of the measured time.
    Returns (X, Y, PHI, summary)."""
    import time as _time
    K = Kernel(prob, P)
    nb, rg = prob.nb, prob.rg
    snap_s = [0.0]
    swap_log: List[dict] = []
    swap_s = [0.0]
    move_log: List[dict] = []
    move_s = [0.0]

    def emit(phase, step, u_, lam_, T_, gv_):
        if snapshot is None:
            return
        t = _time.perf_counter()
        snapshot({'phase': phase, 'step': step, 'X': u_[:nb].copy(), 'Y': u_[nb:2 * nb].copy(),
                  'PHI': u_[2 * nb:] / rg, 'lambda': lam_, 'kov': K.kov,
                  'terms': dict(T_), 'force': -np.vstack([gv_[:nb], gv_[nb:2 * nb],
                                                           gv_[2 * nb:] * rg]),
                  'kernel': K})
        snap_s[0] += _time.perf_counter() - t

    def try_swaps(u_, step, lam_):
        """Greedy best-improvement over the allowed moves, priced on the
        weighted half-perimeter of the nets each touches."""
        t = _time.perf_counter()
        px, py, _qx, _qy = K.positions(u_[:nb], u_[nb:2 * nb], u_[2 * nb:] / rg)
        taken = 0
        while taken < P.swap_max:
            best = None
            for key, pairs in swaps.moves:
                if not swaps.allowed(key):
                    continue
                d = K.swap_delta(pairs, px, py)
                if d < -1e-9 and (best is None or d < best[0]):
                    best = (d, key, pairs)
            if best is None:
                break
            K.apply_swap(best[2])
            swaps.commit(best[1])
            taken += 1
            swap_log.append({'step': step, 'move': best[1], 'delta_mm': round(best[0], 3)})
            if log:
                log(f"force step {step:4d}: swap {best[1]} taken, wire {best[0]:+.2f} mm")
            swap_s[0] += _time.perf_counter() - t
            emit('swap ' + str(best[1]), step, u_, lam_, {'swap_delta_mm': best[0]},
                 np.zeros(3 * nb))
            t = _time.perf_counter()
        swap_s[0] += _time.perf_counter() - t
        return taken

    def springs_at(Xv, Yv, PHv):
        px, py, _qx, _qy = K.positions(Xv, Yv, PHv)
        set_springs(prob, list(claim_rows)
                    + (match_decaps(decap_rows, px, py, P.claim_free_mm) if decap_rows else []))

    def gamma_for(tau):
        return P.gamma_scale * 8.0 * P.bin_mm * 10 ** ((min(tau, 1.0) - 0.1) * 20.0 / 9.0 - 1.0)

    def proj(v, held):
        Xv, Yv, PHv = v[:nb].copy(), v[nb:2 * nb].copy(), v[2 * nb:] / rg
        if held is not None:
            PHv = held.copy()
        if project is not None:
            project(Xv, Yv, PHv)
        return np.concatenate([Xv, Yv, PHv * rg])

    def grad(v, lam, gamma, held):
        T, g = K(v[:nb], v[nb:2 * nb], v[2 * nb:] / rg, lam, gamma)
        gv = np.concatenate([g[0], g[1], g[2] / rg])
        if held is not None:
            gv[2 * nb:] = 0.0
        return T, gv

    t_start = _time.perf_counter()
    if project is not None:
        project(X, Y, PHI)
    springs_at(X, Y, PHI)

    def ramp(tau):
        span = max(1e-6, P.pad_on - P.overflow)
        return float(min(1.0, max(0.0, (P.pad_on - tau) / span)))
    T0, g0 = K(X, Y, PHI, 0.0, 1.0)
    gamma = gamma_for(T0['overflow'])
    K.rep = ramp(T0['overflow'])
    T0, g0 = K(X, Y, PHI, 0.0, gamma)
    _T1, g1 = K(X, Y, PHI, 1.0, gamma)
    gw, gd = float(np.abs(g0[:2]).sum()), float(np.abs(g1[:2] - g0[:2]).sum())
    lam = max(1e-12, P.lambda_start * gw / max(1e-12, gd))
    held = None
    u = np.concatenate([X, Y, PHI * rg])
    v = u.copy()
    a_k = 1.0
    T, gv = grad(v, lam, gamma, held)
    emit('start', 0, u, lam, T, gv)
    v_prev = g_prev = None
    alpha = 0.0
    relax_left = None
    step = 0
    rounded_at = None
    K.time.clear()
    while step < P.iters + P.relax:
        step += 1
        if v_prev is not None:
            dg = float(np.linalg.norm(gv - g_prev))
            if dg > 1e-12:
                alpha = float(np.linalg.norm(v - v_prev)) / dg
        gmax = float(np.abs(gv).max()) or 1e-12
        cap_mm = P.max_step_mm if relax_left is None else P.relax_step_mm
        if alpha <= 0 or gmax * alpha > cap_mm:
            alpha = cap_mm / gmax
        u_new = proj(v - alpha * gv, held)
        a_next = (1 + math.sqrt(4 * a_k * a_k + 1)) / 2
        v_new = proj(u_new + (a_k - 1) / a_next * (u_new - u), held)
        u, a_k, v_prev, g_prev, v = u_new, a_next, v, gv, v_new
        if step % 10 == 0 and decap_rows:
            springs_at(u[:nb], u[nb:2 * nb], u[2 * nb:] / rg)
        kov_used = K.kov
        T, gv = grad(v, lam, gamma, held)
        tau = T['overflow']
        gamma = gamma_for(tau)
        # pad repulsion and overlap come in as the parts spread, and stay
        K.rep = 1.0 if K.rep >= 1.0 else ramp(tau)
        # the smoothed crossings come in with them, their width narrowing
        if K.rn is not None and K.rep > 0:
            K.cross_k = P.cross_weight * K.rep
            K.cross_eps = P.cross_eps0 + (P.cross_eps1 - P.cross_eps0) * K.rep
            if K._xp is None or step % P.cross_every == 0:
                px_, py_, _q, _r = K.positions(u[:nb], u[nb:2 * nb], u[2 * nb:] / rg)
                n_pairs = K.rebuild_crossings(px_, py_)
                if log and step % 250 == 0:
                    log(f"force step {step:4d}: {n_pairs} near pair(s) of tree segments, "
                        f"eps {K.cross_eps:.2f} mm")
        # the courtyard overlap: off while the parts spread, then rising
        # each step until it dominates (the turns are held by then)
        if relax_left is not None:
            K.kov = min(P.overlap_weight * P.overlap_max,
                        max(K.kov, P.overlap_weight) * P.overlap_growth)
        if relax_left is None:
            lam *= P.lambda_rate
        if (swaps is not None and P.swap_every and step % P.swap_every == 0
                and relax_left is None and tau < P.swap_from):
            if try_swaps(u, step, lam):
                T, gv = grad(v, lam, gamma, held)
        if parts is not None and P.move_every and step % P.move_every == 0 and tau < P.move_from:
            t = _time.perf_counter()
            Xm, Ym, PHm = u[:nb].copy(), u[nb:2 * nb].copy(), u[2 * nb:] / rg
            taken = parts.run(K, Xm, Ym, PHm, log=log, step=step)
            if taken:
                if held is not None:
                    for mv, _d, _c in taken:
                        for b_ in ((mv[1],) if mv[0] == 'turn' else mv[1:3]):
                            held[b_] = PHm[b_]
                u = proj(np.concatenate([Xm, Ym, PHm * rg]), held)
                v = u.copy()
                a_k, v_prev, g_prev, alpha = 1.0, None, None, 0.0
                T, gv = grad(v, lam, gamma, held)
                move_log.extend({'step': step, 'move': list(map(str, mv)), 'delta_mm': d_, 'crossings': c_}
                                for mv, d_, c_ in taken)
            move_s[0] += _time.perf_counter() - t
        if snapshot is not None and P.snapshot_every and step % P.snapshot_every == 0:
            emit('spread' if relax_left is None else 'overlap rises', step, u, lam, T, gv)
        if log and (step == 1 or step % 50 == 0):
            log(f"force step {step:4d}: overflow {tau:.3f} lambda {lam:.3g} gamma {gamma:.2f} "
                f"wire {T.get('wire', 0.0):.0f} springs {T['springs']:.1f} crossings {T.get('crossings', 0.0):.1f} groups {T['groups']:.1f} pads {T['pads']:.1f} "
                f"bounds {T['bounds']:.2f} overlap {T['overlap_mm2']:.1f} mm2 "
                f"({T['overlap_pairs']} pairs, k {K.kov:.3g})")
        if held is None and (tau < P.turn_overflow or step >= P.iters):
            PHu = u[2 * nb:] / rg
            rots = np.array([_lattice_round(-math.degrees(p_), b0) for p_, b0 in zip(PHu, base)])
            held = np.where(prob.turnable, np.radians(-rots), PHu)
            u = proj(u, held)
            v = u.copy()
            a_k, v_prev, g_prev, alpha = 1.0, None, None, 0.0
            T, gv = grad(v, lam, gamma, held)
            rounded_at = step
            if log:
                log(f"force step {step:4d}: turns rounded to quarter turns at overflow {tau:.3f}")
            emit('turns rounded', step, u, lam, T, gv)
        if held is not None and relax_left is None and (tau < P.overflow or step >= P.iters):
            relax_left = P.relax
            emit('spread; overlap rises', step, u, lam, T, gv)
        if relax_left is not None:
            relax_left -= 1
            if relax_left <= 0 or (kov_used > 0 and T['overlap_mm2'] < P.overlap_tol_mm2):
                break
    if held is None:
        PHu = u[2 * nb:] / rg
        held = np.radians(-np.array([_lattice_round(-math.degrees(p_), b0) for p_, b0 in zip(PHu, base)]))
    u = proj(u, held)
    Xf, Yf = u[:nb], u[nb:2 * nb]
    K.measure_overlap = True
    Tf, gf = K(Xf, Yf, held, lam, gamma)
    total = _time.perf_counter() - t_start - snap_s[0]
    emit('end', step, u, lam, Tf, np.concatenate([gf[0], gf[1], gf[2] / rg]))
    return Xf, Yf, held, {
        'steps': step, 'turns_rounded_at': rounded_at, 'overflow': round(Tf['overflow'], 4),
        'lambda': lam, 'gamma': round(gamma, 4), 'seconds': round(total, 2),
        'snapshot_seconds': round(snap_s[0], 2), 'swap_seconds': round(swap_s[0], 2),
        'swaps': swap_log, 'part_moves': move_log, 'part_move_seconds': round(move_s[0], 2),
        'ms_per_step': round(1000.0 * (total - swap_s[0] - move_s[0]) / max(1, step), 2),
        'seconds_by_term': {k: round(v_, 2) for k, v_ in K.time.items()},
        'rebuilds': dict(K.rebuilds),
        'terms': {k: round(float(v_), 3) for k, v_ in Tf.items()}}


def place(model, movable: Sequence[str], *, seed: int = 0, params: Optional[ForceParams] = None,
          plane_nets=(), log=None, swaps=None, snapshot=None) -> Dict:
    """Lay `movable` out by forces on `model.state` and leave them there,
    each turn a quarter of its own lattice, the template members at their
    template poses. `swaps(meta)` may return a `SwapMoves` over the
    problem's pins (`meta['pin_key']`: (ref, pad number) -> pin indices);
    `snapshot(info)` gets `optimise`'s info plus `poses`, every movable
    part's (x, y, rot). Returns `optimise`'s summary."""
    from . import simple_anneal as sa
    from .edge_seat import project as seat_project
    P = params or ForceParams()
    st = model.state
    prob, X, Y, PHI, meta = build_problem(model, movable, P, plane_nets, seed)
    if prob.nb == 0:
        return {'steps': 0}
    bidx, seats = meta['bidx'], meta['seats']
    seated = [(bidx[r], s) for r, s in seats.items() if r in bidx]
    ux0, uy0, ux1, uy1 = prob.usable

    def project(Xv, Yv, PHv):
        for b, seat in seated:
            rot = -math.degrees(PHv[b])
            x, y, rot = seat_project(seat, (Xv[b], Yv[b], rot), rot)
            Xv[b], Yv[b], PHv[b] = x, y, math.radians(-rot)
        cx, cy = _corners(prob, Xv, Yv, PHv)
        lo_x, hi_x = ux0 - cx.min(0), ux1 - cx.max(0)
        lo_y, hi_y = uy0 - cy.min(0), uy1 - cy.max(0)
        m = prob.turnable & (lo_x <= hi_x)
        Xv[m] = np.clip(Xv[m], lo_x[m], hi_x[m])
        m = prob.turnable & (lo_y <= hi_y)
        Yv[m] = np.clip(Yv[m], lo_y[m], hi_y[m])
    from .templates import member_target

    def poses_of(Xv, Yv, PHv):
        out = {r: (float(Xv[b]), float(Yv[b]), (-math.degrees(PHv[b])) % 360.0)
               for r, b in bidx.items()}
        for r, (anc, i) in meta['rider_of'].items():
            out[r] = member_target(out[anc], model.tmpl[i][1])
        return out

    def snap(info):
        info['poses'] = poses_of(info['X'], info['Y'], info['PHI'])
        info['bodies'] = meta['bodies']
        snapshot(info)
    X, Y, PHI, summary = optimise(prob, P, X, Y, PHI, base=meta['base'],
                                  claim_rows=meta['claim_rows'], decap_rows=meta['decap_rows'],
                                  project=project, log=log,
                                  swaps=swaps(meta) if swaps is not None else None,
                                  snapshot=snap if snapshot is not None else None,
                                  parts=PartMoves(prob, P, meta['block_of']) if P.move_every else None)
    for r, b in bidx.items():
        rot = round((-math.degrees(PHI[b])) % 360.0, 6) % 360.0
        st.parts[r].ensure_rotation(rot)
        sa._place(st, r, float(X[b]), float(Y[b]), rot)
    for r, (_anc, i) in meta['rider_of'].items():
        x, y, rot = model.template_target(i)
        st.parts[r].ensure_rotation(rot)
        sa._place(st, r, x, y, rot)
    model.reprice_all()
    summary.update({'bodies': prob.nb, 'riders': len(meta['rider_of']),
                    'claims': len(meta['claim_rows']), 'decaps': len(meta['decap_rows'])})
    return summary
