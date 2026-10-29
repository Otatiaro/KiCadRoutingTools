#!/usr/bin/env python3
"""_seg_foreign_pad_dist measures the segment, not samples of it.

The foreign-pad distance of a segment (the terminal graze checks, the
smoother's clears(), the via refit) used to sample the segment every 0.02 mm
and take the rounded-rect distance of each sample to each windowed pad: a cost
of (length / 0.02) x pads per call, and a reading up to half a step long
wherever the nearest point of the segment fell between two samples -- a corner
graze, or a corner the segment cuts through without a sample landing inside.
It now takes the exact distance from the segment to each pad (to the pad's
inner rect less its corner radius in the pad's frame, to a custom pad's
polygons), so a long probe costs the same as a short one.

Checks:
  1. on random boards (rect, roundrect, circle, oval, rotated and custom pads;
     probes of every length, inside pads, degenerate), the distance to a
     lone foreign pad equals check_drc.segment_pad_gap, the exact gap the
     DRC and the connectivity check use (clamped at 0 the same way);
  2. it is never above the old 0.02 mm sweep, and below it by at most half
     a step;
  3. a probe passing a corner between two samples reads the true gap, and a
     probe cutting a corner between two samples reads as inside the pad --
     the sweep read them 0.7 um and 4 um long (fails on the parent);
  4. a custom pad cut between two samples reads 0, not 7 um (fails on the
     parent).

    python3 tests/test_seg_foreign_pad_dist_exact.py
"""
import math
import os
import random
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, TESTS_DIR)

import numpy as np                                                 # noqa: E402
from synth import make_pad, make_net, make_pcb                     # noqa: E402
from check_drc import segment_pad_gap                              # noqa: E402
import single_ended_routing as ser                                 # noqa: E402

failures = []


def check(label, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def board(pad):
    return make_pcb(nets={1: make_net(1, 'A'), 2: make_net(2, 'B')},
                    pads_by_net={2: [pad]})


def random_pad(rnd):
    x, y = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
    if rnd.random() < 0.15:
        n = rnd.randint(3, 6)
        r = rnd.uniform(0.2, 0.8)
        poly = [(x + r * rnd.uniform(0.5, 1) * math.cos(2 * math.pi * k / n),
                 y + r * rnd.uniform(0.5, 1) * math.sin(2 * math.pi * k / n)) for k in range(n)]
        return make_pad(2, x, y, size_x=2 * r, size_y=2 * r, shape='custom', polygons=[poly])
    shape = rnd.choice(['rect', 'roundrect', 'circle', 'oval'])
    sx = rnd.uniform(0.2, 1.6)
    sy = sx if shape == 'circle' else rnd.uniform(0.2, 1.6)
    kw = {}
    if shape == 'roundrect':
        kw['roundrect_rratio'] = rnd.choice([0.1, 0.25, 0.5])
    if rnd.random() < 0.4:
        kw['rect_rotation'] = rnd.uniform(0, 180)
    return make_pad(2, x, y, size_x=sx, size_y=sy, shape=shape, **kw)


def random_probe(rnd, pad):
    if rnd.random() < 0.3:      # from inside or at the pad
        x, y = pad.global_x + rnd.uniform(-0.3, 0.3), pad.global_y + rnd.uniform(-0.3, 0.3)
    else:
        x, y = rnd.uniform(-3, 3), rnd.uniform(-3, 3)
    L = rnd.choice([0.0, 0.01, 0.07, 0.5, 2.0, 6.0])
    a = rnd.uniform(0, 2 * math.pi)
    return x, y, x + L * math.cos(a), y + L * math.sin(a)


def sampled(pcb, x1, y1, x2, y2):
    """The 0.02 mm sweep the function used to run (same pads, same model)."""
    nids, cx, cy, hx, hy, cr, rc, rs, ex, ey, plc, custom = ser._foreign_pad_arrays(pcb, 'F.Cu')
    n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 0.02) + 1)
    t = np.linspace(0.0, 1.0, n + 1)
    sx, sy = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    best = ser._custom_pad_min_dist(custom, 1, list(zip(sx, sy)))
    if cx.size:
        ddx = sx[:, None] - cx[None, :]
        ddy = sy[:, None] - cy[None, :]
        lx = np.abs(ddx * rc + ddy * rs)
        ly = np.abs(-ddx * rs + ddy * rc)
        d = np.hypot(np.maximum(lx - (hx - cr), 0.0), np.maximum(ly - (hy - cr), 0.0)) - cr
        best = min(best, float(d.min()))
    return best


def main():
    rnd = random.Random(11)
    n = worst_gap = 0
    bad_exact, bad_above, below = 0, 0, 0.0
    for _ in range(600):
        pad = random_pad(rnd)
        pcb = board(pad)
        for _ in range(8):
            x1, y1, x2, y2 = random_probe(rnd, pad)
            got = ser._seg_foreign_pad_dist(pcb, 1, x1, y1, x2, y2, 'F.Cu')
            want = segment_pad_gap(x1, y1, x2, y2, pad)
            if want > ser._FOREIGN_PAD_WINDOW:
                continue                     # outside the window: not measured
            n += 1
            if abs(max(got, 0.0) - want) > 1e-9:
                bad_exact += 1
                worst_gap = max(worst_gap, abs(max(got, 0.0) - want))
            old = sampled(pcb, x1, y1, x2, y2)
            if got > old + 1e-12:
                bad_above += 1
            below = max(below, old - got)
    print(f"{n} probes against 600 random pads")
    check("equals check_drc.segment_pad_gap (exact)", bad_exact == 0,
          f"{bad_exact} differ, worst {worst_gap:.2e} mm")
    check("never above the 0.02 mm sweep", bad_above == 0, f"{bad_above} above")
    check("below the sweep by at most half a step", below <= 0.01 + 1e-12,
          f"largest {below * 1000:.2f} um")

    # A 1 x 1 mm rect at the origin; a diagonal probe whose nearest point to the
    # (0.5, 0.5) corner falls between two samples (n = 71 samples, odd).
    pad = make_pad(2, 0.0, 0.0, size_x=1.0, size_y=1.0)
    pcb = board(pad)
    c, h = 0.05, 0.5
    x1, y1, x2, y2 = 0.5 + c - h, 0.5 + c + h, 0.5 + c + h, 0.5 + c - h
    got = ser._seg_foreign_pad_dist(pcb, 1, x1, y1, x2, y2, 'F.Cu')
    true = c * math.sqrt(2)
    check("a corner passed between two samples reads its true gap",
          abs(got - true) < 1e-12, f"{got:.6f} vs {true:.6f} mm")
    # The same probe moved 3 um past the corner cuts it: inside the copper.
    x1, y1, x2, y2 = 0.5 - 0.003 - h, 0.5 - 0.003 + h, 0.5 - 0.003 + h, 0.5 - 0.003 - h
    got = ser._seg_foreign_pad_dist(pcb, 1, x1, y1, x2, y2, 'F.Cu')
    check("a corner cut between two samples reads as inside the pad", got <= 0.0,
          f"{got * 1000:.2f} um")

    # A custom triangle whose tip a probe cuts between two samples.
    tri = [(0.0, 0.0), (1.0, 0.002), (0.0, 0.004)]
    pad = make_pad(2, 0.3, 0.002, size_x=1.0, size_y=0.004, shape='custom', polygons=[tri])
    pcb = board(pad)
    got = ser._seg_foreign_pad_dist(pcb, 1, 0.995, -0.5, 0.995, 0.51, 'F.Cu')
    check("a custom pad cut between two samples reads 0", got == 0.0, f"{got * 1000:.2f} um")

    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
