#!/usr/bin/env python3
"""CLI/GUI parity for a BGA fanout ball rescued at the fab floor: the fanout
tab discloses the rescue's clearance exactly as bga_fanout does.

The rescue records its clearance in the run's ledgers (`design_rules` rows,
`clearance_ledger`), and both fronts write the Default class from
`clearance_ledger.effective` (the CLI main, and the tab's writeback through
`fix_kicad_drc_settings.fanout_fix_kwargs`). So the two fronts must agree on
the copper and on what the ledgers say after the run.

Fixture: the synthetic 8x8 BGA of tests/test_fanout_rescue_floor_disclosed.py
(0.8 mm pitch, 4 layers, --clearance 0.15, 0.15 tracks, 0.45/0.2 vias, under
which the inner balls escape only at the fab floor), loaded by pcbnew. What it
pins, driving the REAL FanoutTab on a REAL headless RoutingDialog against the
CLI's text-parsed engine call with the tab's OWN kwargs:

  1. the emitted copper (tracks, vias, failed nets) is identical;
  2. the clearance ledger and the design_rules clearance rows are identical,
     and below the run clearance (the rescue fired and was recorded).

Needs KiCad python (wx + pcbnew); skips cleanly without it.

Run:  python3 tests/gui_parity/test_fanout_rescue_floor_gui.py
"""
import glob
import importlib.util
import os
import subprocess
import sys
import tempfile

os.environ.setdefault('WXSUPPRESS_SIZER_FLAGS_CHECK', '1')

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = 'U1'
CL, TW, VS, VD = 0.15, 0.15, 0.45, 0.2
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


def _ledgers():
    """(clearance ledger minimum, sorted clearance rows) after a run."""
    import clearance_ledger
    from fab_tiers import escalation_summary
    rows = sorted((r['net_name'], r['requested'], r['delivered'])
                  for r in escalation_summary()['narrowed'] if r['kind'] == 'clearance')
    return clearance_ledger.get_min(), rows


def _reset_ledgers():
    import clearance_ledger
    from fab_tiers import reset_ledger
    clearance_ledger.reset()
    reset_ledger()


def make_fixture(tmpdir):
    spec = importlib.util.spec_from_file_location(
        'rescue_floor', os.path.join(REPO, 'tests', 'test_fanout_rescue_floor_disclosed.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_board(os.path.join(tmpdir, 'bga.kicad_pcb'))


def run_gui(pcbnew, board_path):
    import bga_fanout
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
        out = real_engine(footprint, pcb, **kwargs)
        seen['ledgers'] = _ledgers()
        return out

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
        cfg.update(escape_method='underpad', track_width=TW, clearance=CL,
                   via_size=VS, via_drill=VD)
        _reset_ledgers()
        tab._run_bga_fanout(fp, nets, cfg)
        from wx_pump import run_until
        run_until(lambda: not getattr(tab, '_running', False), 600)
        kwargs = seen.get('kwargs')
        if kwargs is not None:
            kwargs.pop('progress_callback', None)
            kwargs.pop('cancel_check', None)
        return ({'tracks': captured.get('tracks') or [], 'vias': captured.get('vias') or [],
                 'failed': captured.get('failed') or [], 'ledgers': seen.get('ledgers')}, kwargs)
    finally:
        bga_fanout.generate_bga_fanout = real_engine
        dialog.Destroy()


def run_cli(board_path, kwargs):
    from kicad_parser import parse_kicad_pcb
    from bga_fanout import generate_bga_fanout
    pcb = parse_kicad_pcb(board_path)
    _reset_ledgers()
    tracks, vias, _rm, failed = generate_bga_fanout(pcb.footprints[REF], pcb, **kwargs)
    return {'tracks': tracks, 'vias': vias, 'failed': failed, 'ledgers': _ledgers()}


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
        board_path = make_fixture(td)
        print(f"\nGUI front: real FanoutTab on {REF}")
        gui, kwargs = run_gui(pcbnew, board_path)
        r.append(_ok("captured the tab's engine kwargs (the CLI leg replays these)",
                     bool(kwargs)))
        if not kwargs:
            print("\nThe tab never reached generate_bga_fanout -- nothing to compare")
            return 1
        print(f"\nCLI front: text-parsed PCBData, replaying {len(kwargs)} kwargs from the tab")
        cli = run_cli(board_path, kwargs)

        print("\n[1] emitted copper parity")
        print(f"      CLI: {len(cli['tracks'])} tracks, {len(cli['vias'])} vias, {len(cli['failed'])} failed")
        print(f"      GUI: {len(gui['tracks'])} tracks, {len(gui['vias'])} vias, {len(gui['failed'])} failed")
        r.append(_ok("fanout actually produced copper", len(cli['tracks']) > 0))
        r.append(_ok("track sets identical across fronts",
                     _track_key(cli['tracks']) == _track_key(gui['tracks'])))
        r.append(_ok("via sets identical across fronts",
                     _via_key(cli['vias']) == _via_key(gui['vias'])))
        r.append(_ok("failed-net lists identical across fronts",
                     sorted(cli['failed']) == sorted(gui['failed'])))

        print("\n[2] the rescue's clearance in the ledgers")
        (cmin, crows), (gmin, grows) = cli['ledgers'], gui['ledgers'] or (None, [])
        r.append(_ok(f"clearance ledger: CLI {cmin}, GUI {gmin}, below {CL}",
                     cmin is not None and cmin == gmin and cmin < CL))
        r.append(_ok(f"design_rules clearance rows identical ({len(crows)} on the CLI)",
                     len(crows) > 0 and crows == grows))

    passed = sum(r)
    print(f"\n{passed}/{len(r)} fanout rescue-floor parity checks passed")
    print("=" * 60)
    return 0 if passed == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
