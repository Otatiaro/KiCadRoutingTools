"""Snapshots of a forced start as it converges (`place_simple --snapshot-every`).

Each snapshot is the board at that step drawn by `render_placement` (parts
tinted by group, outline, the intent's keep-outs and zones), with three
layers of its own: the movable parts' density under the parts (white where
the board is empty, red where a cell is full), the strongest forces on the
parts over them (an arrow from each origin, its length the force, an arc
for the torque) and a legend: step, phase, each energy term and the
courtyard overlap left. The frames are numbered in order and put together
as an animation (`film`): an MP4 by ffmpeg when it is on the PATH, else a
GIF by Pillow.

Rendering is slow next to a step (a board write and a render each), so it
is off by default and its time is kept out of the force pass's own
(`optimise`'s `snapshot_seconds`).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

#: The arrows drawn per frame: the parts with the strongest force.
ARROWS = 40
#: mm of arrow for the strongest force of the frame.
ARROW_MM = 8.0


def _heat(grid, x0, y0, h):
    """`fn(draw, renderer)`: each cell with parts in it, white to red by its
    density (1 = full)."""
    def draw(d, r):
        tf = r.tf
        nx, ny = grid.shape
        for i, j in zip(*np.nonzero(grid > 0.02)):
            v = float(min(1.0, grid[i, j]))
            c = (255, int(255 * (1 - 0.8 * v)), int(255 * (1 - 0.8 * v)))
            a = tf.pt(x0 + i * h, y0 + j * h)
            b = tf.pt(x0 + (i + 1) * h, y0 + (j + 1) * h)
            d.rectangle([a, b], fill=c)
    return draw


def _arrows(poses, force, bodies):
    """`fn(draw, renderer)`: the strongest forces, from each part's origin."""
    fx, fy, tq = force
    mag = np.hypot(fx, fy)
    if not mag.size or mag.max() <= 0:
        return lambda d, r: None
    top = np.argsort(-mag)[:ARROWS]
    scale = ARROW_MM / float(mag.max())
    tmax = float(np.abs(tq).max()) or 1.0

    def draw(d, r):
        tf = r.tf
        w = max(1, int(tf.length(0.25)))
        for b in top:
            x, y, _rot = poses[bodies[b]]
            ex, ey = x + fx[b] * scale, y + fy[b] * scale
            p0, p1 = tf.pt(x, y), tf.pt(ex, ey)
            d.line([p0, p1], fill=(20, 60, 200), width=w)
            ang = math.atan2(ey - y, ex - x)
            for s in (-0.5, 0.5):
                q = tf.pt(ex - 1.2 * math.cos(ang + s), ey - 1.2 * math.sin(ang + s))
                d.line([p1, q], fill=(20, 60, 200), width=w)
            k = abs(float(tq[b])) / tmax
            if k > 0.05:
                rad = tf.length(1.0 + 2.0 * k)
                box = [p0[0] - rad, p0[1] - rad, p0[0] + rad, p0[1] + rad]
                start = 0 if tq[b] > 0 else 180
                d.arc(box, start, start + 270 * k, fill=(150, 0, 150), width=w)
    return draw


def _legend(lines):
    def draw(d, r):
        from route_render import load_font
        size = max(12, int(r.tf.length(2.2)))
        font = load_font(size)
        x, y = r.tf.pt(r.tf.min_x + 2, r.tf.min_y + 2)
        widest = max(d.textlength(t, font=font) for t in lines)
        d.rectangle([x - 6, y - 6, x + widest + 8, y + (size + 4) * len(lines) + 6],
                    fill=(255, 255, 255), outline=(0, 0, 0))
        for k, t in enumerate(lines):
            d.text((x, y + k * (size + 4)), t, fill=(0, 0, 0), font=font)
    return draw


class Snapshots:
    """The frames of one forced start, written to `out_dir`."""

    def __init__(self, board, out_dir, *, groups=None, intent=None, every=0, size=1400):
        self.board, self.out_dir = board, out_dir
        self.groups, self.intent = groups, intent
        self.every, self.size = every, size
        self.frames: List[str] = []
        os.makedirs(out_dir, exist_ok=True)
        self._tmp = tempfile.mkdtemp(prefix='force_film_')

    def _render(self, board, name, under=(), over=()):
        import render_placement
        out = os.path.join(self.out_dir, name)
        argv = [board, '--flat', '--no-ratsnest', '--no-legality', '--no-labels', '--quiet',
                '--size', str(self.size), '-o', out]
        if self.groups:
            argv += ['--groups', self.groups]
        if self.intent:
            argv += ['--intent', self.intent]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            render_placement.main(argv, overlays=under, top_overlays=over)
        self.frames.append(out)
        return out

    def __call__(self, info):
        from .writer import write_placed_output
        poses = info['poses']
        tmp = os.path.join(self._tmp, 'frame.kicad_pcb')
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            write_placed_output(self.board, tmp, [
                {'reference': r, 'new_x': round(x, 4), 'new_y': round(y, 4),
                 'new_rotation': round(rot % 360.0, 4)} for r, (x, y, rot) in poses.items()])
        K = info['kernel']
        under = []
        dm = K.density_map(info['X'], info['Y'], info['PHI'], 0)
        if dm is not None:
            under.append(_heat(*dm))
        t = info['terms']
        lines = [f"step {info['step']}  {info['phase']}",
                 f"lambda {info['lambda']:.3g}  k_overlap {info['kov']:.3g}"]
        if 'overflow' in t:
            lines.append(f"overflow {t['overflow']:.3f}  overlap {t.get('overlap_mm2', 0):.1f} mm2 "
                         f"({int(t.get('overlap_pairs', 0))} pairs)")
            lines.append("E: " + "  ".join(f"{k} {t[k]:.0f}" for k in
                                            ('wire', 'springs', 'groups', 'pads', 'density', 'overlap', 'bounds')
                                            if k in t))
        else:
            lines += [f"{k} {v}" for k, v in t.items()]
        over = [_arrows(poses, info['force'], info['bodies']), _legend(lines)]
        safe = ''.join(c if c.isalnum() else '_' for c in info['phase'])[:30]
        return self._render(tmp, f"f{len(self.frames):04d}_s{info['step']:05d}_{safe}.png",
                            under, over)

    def still(self, board, label):
        """A frame of a whole board (before or after the legalization)."""
        lines = [label]
        return self._render(board, f"f{len(self.frames):04d}_{label.replace(' ', '_')}.png",
                            (), [_legend(lines)])

    def film(self, path_stem: str, fps: int = 6) -> Optional[str]:
        """The frames as an MP4 (ffmpeg) or a GIF (Pillow); the path written."""
        if not self.frames:
            return None
        ff = shutil.which('ffmpeg')
        if ff:
            lst = os.path.join(self._tmp, 'frames.txt')
            with open(lst, 'w', encoding='utf-8') as fh:
                for f in self.frames:
                    fh.write(f"file '{os.path.abspath(f)}'\nduration {1.0 / fps:.4f}\n")
            out = path_stem + '.mp4'
            r = subprocess.run([ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                                '-i', lst, '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                '-pix_fmt', 'yuv420p', out], capture_output=True)
            if r.returncode == 0:
                return out
        from PIL import Image
        imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in self.frames]
        out = path_stem + '.gif'
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return out

    def close(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
