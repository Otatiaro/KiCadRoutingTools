"""Every KRT command-line tool loads OpenBLAS with ONE thread.

numpy and scipy each load their OWN OpenBLAS (two DLLs in the wheels), and
each one commits a work buffer per thread at import -- one thread per core.
On a 20-core machine that is 1.22 GB of commit per process before any board
is read (numpy 626 MB, scipy.spatial another 635 MB; measured with
GetProcessMemoryInfo). The router makes no BLAS calls, and the placer's are
small dense solves, so the buffers are never used; Windows still charges
them to the commit limit, which is what makes a process fail to allocate
when several run side by side. With one thread the same imports commit
40 MB, and a 150 x 107.5 mm 4-layer route at 0.05 mm peaks at 1.49 GB
instead of 2.71 GB with an identical JSON_SUMMARY.

OpenBLAS reads OPENBLAS_NUM_THREADS once, when its library loads, so each
entry script sets it (setdefault: a caller's value wins) BEFORE its first
import, and only when it is the program being run -- an import (the GUI
plugin inside KiCad, a test) leaves the host process's environment alone:

    import os
    if __name__ == '__main__':  # one OpenBLAS thread: tests/test_blas_single_thread.py
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

Cases:
  1. Every script with a `__main__` block in py_router/, py_placer/,
     py_tools/ and awx/ carries the guard, ahead of every import but
     `from __future__` (static: the AST). A new tool without it fails here.
  2. Run as the program (`--help`), a sample of them loads numpy with
     OPENBLAS_NUM_THREADS=1 (an import hook records the value at numpy's
     first import; a tool whose --help exits before numpy, like the
     placer's, is read at exit instead).
  3. An explicit caller value (3) is kept.
  4. route.py IMPORTED leaves the environment alone.

Run: python3 tests/test_blas_single_thread.py
"""

import ast
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_utils import evidence, tool  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ('py_router', 'py_placer', 'py_tools', 'awx')
SAMPLE = ('py_router/route.py', 'py_router/route_diff.py', 'py_router/route_planes.py',
          'py_router/repair_planes.py', 'py_router/check_drc.py', 'py_router/check_connected.py',
          'py_router/bga_fanout.py', 'py_placer/place_simple.py', 'awx/whole_route.py')


def _is_main_block(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
            and any(isinstance(c, ast.Constant) and c.value == '__main__' for c in ast.walk(node.test)))


def _is_guard(node):
    return _is_main_block(node) and any(
        isinstance(c, ast.Constant) and c.value == 'OPENBLAS_NUM_THREADS' for c in ast.walk(node))


def entry_scripts():
    for d in DIRS:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if not f.endswith('.py'):
                continue
            rel = '%s/%s' % (d, f)
            with open(os.path.join(ROOT, d, f), encoding='utf-8') as fh:
                tree = ast.parse(fh.read())
            if any(_is_main_block(n) for n in tree.body):
                yield rel, tree


def test_every_entry_script_guards_before_its_imports():
    bad = []
    n = 0
    for rel, tree in entry_scripts():
        n += 1
        for node in tree.body:
            if _is_guard(node):
                break
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                continue
            if isinstance(node, ast.Import) and [a.name for a in node.names] == ['os']:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bad.append('%s: line %d imports before the guard' % (rel, node.lineno))
                break
        else:
            bad.append('%s: no OPENBLAS_NUM_THREADS guard' % rel)
    assert n > 50, 'BROKEN TEST: only %d entry scripts found' % n
    assert not bad, '\n'.join(bad)


_PROBE = textwrap.dedent('''
    import os, runpy, sys
    class Probe:
        def find_spec(self, name, path=None, target=None):
            if name == 'numpy' and 'numpy' not in sys.modules:
                print('NUMPY_LOADS_WITH=%s' % os.environ.get('OPENBLAS_NUM_THREADS'), flush=True)
                sys.meta_path.remove(self)
            return None
    sys.meta_path.insert(0, Probe())
    script, as_main = sys.argv[1], sys.argv[2] == '1'
    here = os.path.dirname(script)
    sys.path.insert(0, here)
    sys.argv = [script, '--help']
    try:
        if as_main:
            runpy.run_path(script, run_name='__main__')
        else:
            __import__(os.path.basename(script)[:-3])
    except SystemExit:
        pass
    if 'numpy' not in sys.modules:
        print('ENV_AT_EXIT=%s' % os.environ.get('OPENBLAS_NUM_THREADS'), flush=True)
''')


def _numpy_loads_with(rel, as_main=True, preset=None):
    env = dict(os.environ)
    env.pop('OPENBLAS_NUM_THREADS', None)
    if preset is not None:
        env['OPENBLAS_NUM_THREADS'] = preset
    d, name = rel.split('/')
    # awx/ is not a tool directory run_utils.tool() resolves
    script = evidence(os.path.join(ROOT, d, name), 'script') if d == 'awx' else tool(name)
    r = subprocess.run([sys.executable, '-c', _PROBE, script, '1' if as_main else '0'],
                       capture_output=True, text=True, env=env, timeout=300, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith(('NUMPY_LOADS_WITH=', 'ENV_AT_EXIT=')):
            return line.split('=', 1)[1]
    raise AssertionError('BROKEN TEST: %s reported nothing (exit %d)\n%s\n%s'
                         % (rel, r.returncode, r.stdout[-2000:], r.stderr[-2000:]))


def test_cli_loads_numpy_single_threaded():
    for rel in SAMPLE:
        got = _numpy_loads_with(rel)
        assert got == '1', '%s: numpy loaded with OPENBLAS_NUM_THREADS=%s' % (rel, got)


def test_caller_value_kept():
    got = _numpy_loads_with('py_router/route.py', preset='3')
    assert got == '3', 'caller OPENBLAS_NUM_THREADS=3 overridden to %s' % got


def test_import_leaves_host_environment():
    got = _numpy_loads_with('py_router/route.py', as_main=False)
    assert got == 'None', 'importing route.py set OPENBLAS_NUM_THREADS=%s' % got


if __name__ == '__main__':
    test_every_entry_script_guards_before_its_imports()
    test_cli_loads_numpy_single_threaded()
    test_caller_value_kept()
    test_import_leaves_host_environment()
    print('test_blas_single_thread: all passed')
