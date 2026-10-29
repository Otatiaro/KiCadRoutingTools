#!/usr/bin/env python3
"""CLI/GUI parity for the fanout's hole floor: the fanout tab holds a board's
min_hole_clearance at the vias it places exactly as bga_fanout does.

The engines read the floor themselves (`kicad_dru.plated_hole_floor`), from
the project on the CLI and from the live board's design settings in the GUI
(`obstacle_map.declared_hole_clearance`), so the two fronts must agree on the
value and on the copper it shapes.

Fixture: the synthetic 8x8 BGA of tests/test_fanout_plated_hole_clearance.py
(0.8 mm pitch, 4 layers, project declaring clearance 0.1, min_hole_clearance
0.25), loaded by pcbnew. What it pins, driving the REAL FanoutTab on a REAL
headless RoutingDialog against the CLI's text-parsed engine call with the
tab's OWN kwargs (the under-pad escape, 0.35/0.2 vias, 0.12 track):

  1. both fronts read the declared 0.25 floor;
  2. the emitted copper (tracks, vias, failed nets) is identical;
  3. no track or via of either front is inside the floor of a via hole.

Needs KiCad python (wx + pcbnew); skips cleanly without it.

Run:  python3 tests/gui_parity/test_fanout_hole_floor_gui.py
"""
import glob
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

os.environ.setdefault('WXSUPPRESS_SIZER_FLAGS_CHECK', '1')

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = 'U1'
sys.path.insert(0, os.path.join(REPO, 'py_router'))
from kicad_locate import path_version_key  # noqa: E402
del sys.path[0]    # this file orders its own sys.path further down
KICAD_PYTHONS = [
    "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3",
    "/usr/bin/python3",
    *sorted(glob.glob(r"C:\Program Files\KiCad\*\bin\python.exe"),
            key=path_version_key, reverse=True),
]


def _reexec_into_kicad():
    for cand in KICAD_PYTHONS:
        if cand != sys.executable and os.path.exists(cand):
            if subprocess.run([cand, '-c', 'import wx, pcbnew'],
                              capture_output=True).returncode == 0:
                argv = [cand, os.path.abspath(__file__)] + sys.argv[1:]
                if os.name == 'nt':
                    # os.execv re-splits argv on spaces on Windows, and the
                    # interpreter lives under "Program Files".
                    sys.exit(subprocess.run(argv).returncode)
                os.execv(cand, argv)
    print("SKIP: no python with wx + pcbnew found")
    sys.exit(0)


def _ok(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def _track_key(tracks):
    return sorted((round(t['start'][0], 6), round(t['start'][1], 6),
                   round(t['end'][0], 6), round(t['end'][1], 6),
                   t['net_id'], t['layer'], round(t['width'], 6))
                  for t in tracks)


def _via_key(vias):
    return sorted((round(v['x'], 6), round(v['y'], 6), v['net_id'],
                   round(v.get('size', 0), 6), round(v.get('drill', 0), 6))
                  for v in vias)


def _inside_floor(tracks, vias, floor):
    """Tracks and vias of another net closer than `floor` to a via's drill."""
    from geometry_utils import point_to_segment_distance
    bad = 0
    for v in vias:
        r = v['drill'] / 2.0
        for t in tracks:
            if t['net_id'] != v['net_id'] and point_to_segment_distance(
                    v['x'], v['y'], t['start'][0], t['start'][1],
                    t['end'][0], t['end'][1]) - t['width'] / 2.0 - r < floor - 1e-4:
                bad += 1
        for o in vias:
            if o['net_id'] != v['net_id'] and math.hypot(o['x'] - v['x'], o['y'] - v['y']) \
                    - o['size'] / 2.0 - r < floor - 1e-4:
                bad += 1
    return bad


def make_fixture(tmpdir):
    spec = importlib.util.spec_from_file_location(
        'fanout_floor', os.path.join(REPO, 'tests', 'test_fanout_plated_hole_clearance.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = mod.Board()
    b.bga(REF, 10, 10)
    b.partners('J1', 19)
    return b.write(os.path.join(tmpdir, 'bga.kicad_pcb')), mod.HC


def run_gui(pcbnew, board_path):
    import bga_fanout
    from kicad_dru import plated_hole_floor
    from kicad_parser import build_pcb_data_from_board
    from kicad_routing_plugin import swig_gui

    board = pcbnew.LoadBoard(board_path)
    pcbnew.GetBoard = lambda: board
    pcb_data = build_pcb_data_from_board(board)
    dialog = swig_gui.RoutingDialog(None, pcb_data, board_path)

    real_engine = bga_fanout.generate_bga_fanout
    seen, captured = {}, {}

    def _spy(footprint, pcb, **kwargs):
        seen['kwargs'] = dict(kwargs)
        seen['floor'] = plated_hole_floor(pcb)
        return real_engine(footprint, pcb, **kwargs)

    def _capture(tracks, vias, failed_nets=None, **kw):
        captured['tracks'] = tracks
        captured['vias'] = vias
        captured['failed'] = failed_nets or []

    try:
        bga_fanout.generate_bga_fanout = _spy
        tab = dialog.fanout_tab
        tab._apply_fanout_results = _capture
        fp = pcb_data.footprints[REF]
        nets = sorted({p.net_name for p in fp.pads if p.net_id and p.net_name})
        cfg = tab.bga_options.get_config()
        cfg.update(escape_method='underpad', track_width=0.12, clearance=0.1,
                   via_size=0.35, via_drill=0.2)
        tab._run_bga_fanout(fp, nets, cfg)
        from wx_pump import run_until
        run_until(lambda: not getattr(tab, '_running', False), 600)
        kwargs = seen.get('kwargs')
        if kwargs is not None:
            kwargs.pop('progress_callback', None)
            kwargs.pop('cancel_check', None)
        return ({'tracks': captured.get('tracks') or [], 'vias': captured.get('vias') or [],
                 'failed': captured.get('failed') or [], 'floor': seen.get('floor')}, kwargs)
    finally:
        bga_fanout.generate_bga_fanout = real_engine
        dialog.Destroy()


def run_cli(board_path, kwargs):
    from kicad_dru import plated_hole_floor
    from kicad_parser import parse_kicad_pcb
    from bga_fanout import generate_bga_fanout
    pcb = parse_kicad_pcb(board_path)
    floor = plated_hole_floor(pcb)
    tracks, vias, _rm, failed = generate_bga_fanout(pcb.footprints[REF], pcb, **kwargs)
    return {'tracks': tracks, 'vias': vias, 'failed': failed, 'floor': floor}


def main():
    try:
        import wx, pcbnew  # noqa: F401
    except ImportError:
        _reexec_into_kicad()
    import wx
    import pcbnew

    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, 'py_router'))
    sys.path.insert(0, os.path.join(REPO, 'py_tools'))
    app = wx.App(False)  # noqa: F841
    wx.MessageBox = lambda *a, **k: wx.OK
    r = []
    with tempfile.TemporaryDirectory() as td:
        board_path, declared = make_fixture(td)
        print(f"\nGUI front: real FanoutTab on {REF}")
        gui, kwargs = run_gui(pcbnew, board_path)
        r.append(_ok("captured the tab's engine kwargs (the CLI leg replays these)",
                     bool(kwargs)))
        if not kwargs:
            print("\nThe tab never reached generate_bga_fanout -- nothing to compare")
            return 1
        r.append(_ok(f"the tab ran the under-pad escape with 0.35/0.2 vias "
                     f"(got {kwargs.get('escape_method')}, {kwargs.get('via_size')}/"
                     f"{kwargs.get('via_drill')})",
                     kwargs.get('escape_method') == 'underpad'
                     and kwargs.get('via_size') == 0.35 and kwargs.get('via_drill') == 0.2))
        print(f"\nCLI front: text-parsed PCBData, replaying {len(kwargs)} kwargs from the tab")
        cli = run_cli(board_path, kwargs)

        print("\n[1] the floor both fronts read")
        r.append(_ok(f"GUI {gui['floor']}, CLI {cli['floor']}, declared {declared}",
                     gui['floor'] == cli['floor'] == declared))

        print("\n[2] emitted copper parity")
        print(f"      CLI: {len(cli['tracks'])} tracks, {len(cli['vias'])} vias, {len(cli['failed'])} failed")
        print(f"      GUI: {len(gui['tracks'])} tracks, {len(gui['vias'])} vias, {len(gui['failed'])} failed")
        r.append(_ok("fanout actually produced copper", len(cli['tracks']) > 0 and len(cli['vias']) > 0))
        r.append(_ok("track sets identical across fronts",
                     _track_key(cli['tracks']) == _track_key(gui['tracks'])))
        r.append(_ok("via sets identical across fronts",
                     _via_key(cli['vias']) == _via_key(gui['vias'])))
        r.append(_ok("failed-net lists identical across fronts",
                     sorted(cli['failed']) == sorted(gui['failed'])))

        print("\n[3] nothing inside the floor of a via hole")
        for label, res in (("CLI", cli), ("GUI", gui)):
            n = _inside_floor(res['tracks'], res['vias'], declared)
            r.append(_ok(f"{label}: {n} item(s) inside {declared} of a via hole", n == 0))

    passed = sum(r)
    print(f"\n{passed}/{len(r)} fanout hole-floor parity checks passed")
    print("=" * 60)
    return 0 if passed == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
