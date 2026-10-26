"""ends_nudge.py OUT LANE X,Y [--kind=via|track] [--layer=L] [--via-at=X,Y] -- a CROWDED END's copper nudged clear on
the round's fanout bench (BENCH0, else BENCH): the loop stopped on a lane found too close, at its tooth, to the escape
copper of another bus lane (LANE, its via with --kind=via, or its track on layer L with --kind=track), and that lane's
via or its tooth -- its stub's free end -- moves by the smallest step that stands its copper clear of the point X,Y (the
finding's, in the board's frame) by a lane's clearance and a margin. With --via-at, the via standing at X,Y moves
instead, any net's -- a plane ball's drop -- its own tracks ending at it with it (LANE then only names the finding).
Every move is checked on the fanout's own obstacle model (braid.build_obstacles: every foreign pad, track and via, at
the escapes' track and clearance) on every layer a via stands through, and hole to hole against every other drill; a
tooth never steps back into its array. Writes OUT and its plan sidecar (the bench's, the tooth where it now stands).
Exit 0 moved; 3 no step clears the point; 4 the copper already stands clear of it as laid, so no move of it answers
what the loop found (the finding is another copper's)."""
import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'py_router')))
import awx_settings  # noqa: E402
from kicad_parser import parse_kicad_pcb  # noqa: E402
from kicad_writer import add_tracks_and_vias_to_pcb, remove_segments_from_content, remove_vias_from_content  # noqa: E402
import braid as te  # noqa: E402
import rules as _rules  # noqa: E402
from coherent_nets import by_key, namer  # noqa: E402  the whole route's names for nets

MARGIN = 0.05                     # past the lane's clearance: the audit's bar, and a grid's rounding off it
DIRS16 = [(math.cos(2 * math.pi * k / 16), math.sin(2 * math.pi * k / 16)) for k in range(16)]
TOL = 0.005                       # two ends this close are one point
STEP = 0.025
# (the farthest a move goes: a bus lane's own stub 0.25 mm, another net's via -- a drop no lane's order holds -- 0.5)
MOVE_MAX, VIA_AT_MAX = 0.25, 0.5

opt = {a.split('=')[0]: (a.split('=', 1)[1] if '=' in a else '1') for a in sys.argv[1:] if a.startswith('--')}
args = [a for a in sys.argv[1:] if not a.startswith('--')]
KIND = opt.get('--kind', 'all')       # the copper the crowded lane met: its via (any layer), its track (on --layer)
LAYER = opt.get('--layer')
VIA_AT = opt.get('--via-at')
STEPS = [STEP * k for k in range(1, int(round((VIA_AT_MAX if VIA_AT else MOVE_MAX) / STEP)) + 1)]
out, lane, pts_s = args[0], args[1], args[2]
pts = [tuple(float(v) for v in p.split(',')) for p in pts_s.split(';') if p]
R = _rules.active()
FT, FC = R.fan_track, R.fan_clear
H2H = R.hole_to_hole or 0.25
bench = awx_settings.get('BENCH0') or awx_settings.req('BENCH')
side = json.load(open(os.path.splitext(bench)[0] + '.plan.json'))
pcb = parse_kicad_pcb(bench)
n2n = {i: n.name for i, n in pcb.nets.items()}
byname = by_key(pcb)
if VIA_AT:
    _vx, _vy = (float(v) for v in VIA_AT.split(','))
    _v0 = min(pcb.vias, key=lambda v: math.hypot(v.x - _vx, v.y - _vy))
    nid, lane = _v0.net_id, namer(pcb)(n2n[_v0.net_id])
    tooth = tdir = tlayer = None
else:
    nid = byname[lane][0]
    tooth = tuple(side['ends'][lane][0])
    tdir = tuple(side['tooth_dir'][lane])
    tlayer = side['tooth_layer'][lane]
layers = list(pcb.board_info.copper_layers)
# (a stub track's layer as the loop saw it: the relayed bench's (BENCH) -- on more routing layers than two the relayer
# puts a via end's stub on the plan's layer, so a finding on that layer names a track the fanout bench has on another)
_relaid = {}
_key = lambda s_: tuple(sorted(((round(s_.start_x, 3), round(s_.start_y, 3)), (round(s_.end_x, 3), round(s_.end_y, 3)))))
if LAYER and awx_settings.get('BENCH') and os.path.abspath(awx_settings.get('BENCH')) != os.path.abspath(bench):
    _relaid = {_key(s_): s_.layer for s_ in parse_kicad_pcb(awx_settings.get('BENCH')).segments if s_.net_id == nid}
eff_layer = lambda s_: _relaid.get(_key(s_), s_.layer)
bar = te.TRACK / 2 + te.CLEAR + MARGIN
same = lambda p, q: abs(p[0] - q[0]) + abs(p[1] - q[1]) < TOL


def seg_d(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    L2 = (bx - ax) ** 2 + (by - ay) ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / L2))
    return math.hypot(ax + t * (bx - ax) - p[0], ay + t * (by - ay) - p[1])


# ---- the copper that moves: a lane's stub, from its ball's pad along its own copper to its tooth; or the via at X,Y
# and its net's tracks ending at it
segs = [s for s in pcb.segments if s.net_id == nid]
vias = [v for v in pcb.vias if v.net_id == nid]
if VIA_AT:
    pad, stub_v, frontier, seen = None, [_v0], [], set()
    stub_s = [s for s in segs if same((s.start_x, s.start_y), (_v0.x, _v0.y)) or same((s.end_x, s.end_y), (_v0.x, _v0.y))]
else:
    pad = min((p for fp in pcb.footprints.values() for p in fp.pads if p.net_id == nid),
              key=lambda p: math.hypot(p.global_x - tooth[0], p.global_y - tooth[1]))
    stub_s, stub_v, frontier, seen = [], [], [(pad.global_x, pad.global_y)], set()
while frontier:
    q = frontier.pop()
    for s in segs:
        if id(s) in seen:
            continue
        a, b = (s.start_x, s.start_y), (s.end_x, s.end_y)
        if same(a, q) or same(b, q) or (q == (pad.global_x, pad.global_y) and
                                        (math.hypot(a[0] - q[0], a[1] - q[1]) <= max(pad.size_x, pad.size_y) / 2 or
                                         math.hypot(b[0] - q[0], b[1] - q[1]) <= max(pad.size_x, pad.size_y) / 2)):
            seen.add(id(s))
            stub_s.append(s)
            nxt = b if (same(a, q) or math.hypot(a[0] - q[0], a[1] - q[1]) < math.hypot(b[0] - q[0], b[1] - q[1])) else a
            if not same(nxt, tooth):
                frontier.append(nxt)
    for v in vias:
        if id(v) not in seen and same((v.x, v.y), q):
            seen.add(id(v))
            stub_v.append(v)
print(f'ends_nudge: {lane}{" via at " + VIA_AT if VIA_AT else ""} tooth {tooth} {tlayer} dir {tdir}; stub '
      f'{len(stub_s)} track(s), {len(stub_v)} via(s); points {pts}; bar {bar:.3f}')
if not stub_s and not VIA_AT:
    print('ends_nudge: no stub found')
    sys.exit(2)

OBS = {}


def obs(L, margin):
    k = (L, margin)
    if k not in OBS:
        OBS[k] = te.build_obstacles(pcb, nid, {nid}, L, margin=margin)
    return OBS[k]


# (hole to hole is the fab's, whatever the nets: every other drill but the moved via's own)
drills = [(v.x, v.y, v.drill) for v in pcb.vias if all(v is not w for w in stub_v)] + \
    [(p.global_x, p.global_y, p.drill) for fp in pcb.footprints.values() for p in fp.pads if (p.drill or 0) > 0]


def try_move(obj, v):
    """the copper with `obj` ('via' or 'tooth') moved by v: (tracks, vias, new tooth), or None where it breaks a rule"""
    if obj == 'via':
        if not stub_v:
            return None
        v0 = stub_v[0]
        old_pt, new_pt = (v0.x, v0.y), (v0.x + v[0], v0.y + v[1])
    else:
        old_pt, new_pt = tooth, (tooth[0] + v[0], tooth[1] + v[1])
        if (v[0] * tdir[0] + v[1] * tdir[1]) < -1e-9:
            return None                                   # (never back toward the array)
    tracks = []
    for s in stub_s:
        a, b = (s.start_x, s.start_y), (s.end_x, s.end_y)
        a2 = new_pt if same(a, old_pt) else a
        b2 = new_pt if same(b, old_pt) else b
        if (a2, b2) != (a, b):
            if not obs(s.layer, FC + s.width / 2).seg_clear(a2, b2):
                return None
        tracks.append((s, a2, b2))
    vs = []
    for v_ in stub_v:
        p2 = new_pt if (obj == 'via' and same((v_.x, v_.y), old_pt)) else (v_.x, v_.y)
        if p2 != (v_.x, v_.y):
            for L in layers:
                if obs(L, FC + FT / 2).point_violation(p2, pad=v_.size / 2 - FT / 2) is not None:
                    return None
            if any(math.hypot(x - p2[0], y - p2[1]) < v_.drill / 2 + d / 2 + H2H for x, y, d in drills):
                return None
        vs.append((v_, p2))
    t2 = None if tooth is None else (new_pt if obj == 'tooth' else (new_pt if same(tooth, old_pt) else tooth))
    return tracks, vs, t2


def goal(tracks, vs):
    """(met, its clearance): the copper named -- its tracks on LAYER, its vias -- clear of the points by the bar"""
    d = 9e9
    for p in pts:
        if KIND in ('all', 'track'):
            for s, a2, b2 in tracks:
                if LAYER is None or eff_layer(s) == LAYER:
                    d = min(d, seg_d(p, a2, b2) - s.width / 2)
        if KIND in ('all', 'via'):
            for v_, p2 in vs:
                d = min(d, math.hypot(p2[0] - p[0], p2[1] - p[1]) - v_.size / 2)
    return d >= bar, d


base = goal([(s, (s.start_x, s.start_y), (s.end_x, s.end_y)) for s in stub_s], [(v_, (v_.x, v_.y)) for v_ in stub_v])
print(f'  as laid: goal met {base[0]}, clearance {base[1]:.3f}')
if base[0]:
    # (clear as laid: the copper named is not what stands at the point -- a move it calls clear changes nothing the
    # loop found, and the loop would run again for nothing)
    print('ends_nudge: the goal is met as laid -- nothing at the points this nudge can move')
    sys.exit(4)
best = None
for step in STEPS:
    for obj in (('via',) if VIA_AT else ('tooth', 'via')):
        for dx, dy in DIRS16:
            got = try_move(obj, (dx * step, dy * step))
            if got is None:
                continue
            ok, d = goal(got[0], got[1])
            if ok and (best is None or d > best[0]):
                best = (d, obj, (round(dx * step, 4), round(dy * step, 4)), got)
    if best is not None:
        break
if best is None:
    print('ends_nudge: NO nudge clears the points')
    sys.exit(3)
d, obj, v, (tracks, vs, t2) = best
print(f'ends_nudge: {lane} {obj} moved by {v} ({math.hypot(*v):.3f} mm): clearance {d:.3f}'
      + (f'; tooth -> ({t2[0]:.3f}, {t2[1]:.3f})' if t2 else ''))
# ---- written: the moved tracks and via in place of the old
moved_s = [s for s, a2, b2 in tracks if (a2, b2) != ((s.start_x, s.start_y), (s.end_x, s.end_y))]
moved_v = [v_ for v_, p2 in vs if p2 != (v_.x, v_.y)]
content = open(bench, encoding='utf-8').read()
content, n_s = remove_segments_from_content(content, moved_s, n2n)
content, n_v = remove_vias_from_content(content, moved_v, n2n)
if n_s != len(moved_s) or n_v != len(moved_v):
    print(f'ends_nudge: WARNING strip matched {n_s}/{len(moved_s)} tracks, {n_v}/{len(moved_v)} vias')
tmp = out + '.tmp'
open(tmp, 'w', encoding='utf-8').write(content)
add_tracks_and_vias_to_pcb(
    tmp, out,
    [{'start': a2, 'end': b2, 'width': s.width, 'layer': s.layer, 'net_id': nid} for s, a2, b2 in tracks if s in moved_s],
    [{'x': p2[0], 'y': p2[1], 'size': v_.size, 'drill': v_.drill, 'layers': list(v_.layers), 'net_id': nid,
      'tenting_attrs': getattr(v_, 'tenting_attrs', {}), 'inherit_when_unspecified': True}
     for v_, p2 in vs if v_ in moved_v], net_id_to_name=n2n)
os.remove(tmp)
if t2 is not None:
    side['ends'][lane][0] = [round(t2[0], 4), round(t2[1], 4)]
json.dump(side, open(os.path.splitext(out)[0] + '.plan.json', 'w'), indent=1)
for ext in ('.kicad_pro', '.kicad_dru'):
    s_ = os.path.splitext(bench)[0] + ext
    if os.path.isfile(s_):
        shutil.copy(s_, os.path.splitext(out)[0] + ext)
print(f'ends_nudge: wrote {out}')
