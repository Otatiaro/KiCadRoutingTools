"""The routing CLIs load OpenBLAS with ONE thread (py_router/blas_threads.py).

numpy and scipy each commit an OpenBLAS work buffer per core at import --
1.22 GB per process on a 20-core machine -- for BLAS calls the router never
makes. OpenBLAS reads OPENBLAS_NUM_THREADS once, when its library loads, so
the only thing that matters is the variable's value at the moment numpy is
FIRST imported. Each case runs an entry script as the program (`--help`, so
it parses no board) under an import hook that records that value.

Cases:
  1. route.py / route_diff.py / route_planes.py / repair_planes.py run as the
     program: numpy loads with OPENBLAS_NUM_THREADS=1.
  2. An explicit caller value (3) is kept.
  3. route.py IMPORTED (as the GUI plugin and the tests do) leaves the host
     process's environment alone.

Run: python3 tests/test_blas_single_thread.py
"""

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTER = os.path.join(ROOT, 'py_router')
SCRIPTS = ('route.py', 'route_diff.py', 'route_planes.py', 'repair_planes.py')

_PROBE = textwrap.dedent('''
    import os, runpy, sys
    class Probe:
        def find_spec(self, name, path=None, target=None):
            if name == 'numpy' and 'numpy' not in sys.modules:
                print('NUMPY_LOADS_WITH=%s' % os.environ.get('OPENBLAS_NUM_THREADS'), flush=True)
                sys.meta_path.remove(self)
            return None
    sys.meta_path.insert(0, Probe())
    router, script, as_main = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
    sys.path.insert(0, router)
    sys.argv = [script, '--help']
    try:
        if as_main:
            runpy.run_path(os.path.join(router, script), run_name='__main__')
        else:
            __import__(script[:-3])
    except SystemExit:
        pass
''')


def _numpy_loads_with(script, as_main=True, preset=None):
    env = dict(os.environ)
    env.pop('OPENBLAS_NUM_THREADS', None)
    if preset is not None:
        env['OPENBLAS_NUM_THREADS'] = preset
    r = subprocess.run([sys.executable, '-c', _PROBE, ROUTER, script, '1' if as_main else '0'],
                       capture_output=True, text=True, env=env, timeout=300, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith('NUMPY_LOADS_WITH='):
            return line.split('=', 1)[1]
    raise AssertionError('BROKEN TEST: %s never imported numpy (exit %d)\n%s\n%s'
                         % (script, r.returncode, r.stdout[-2000:], r.stderr[-2000:]))


def test_cli_loads_numpy_single_threaded():
    for script in SCRIPTS:
        got = _numpy_loads_with(script)
        assert got == '1', '%s: numpy loaded with OPENBLAS_NUM_THREADS=%s' % (script, got)


def test_caller_value_kept():
    got = _numpy_loads_with('route.py', preset='3')
    assert got == '3', 'caller OPENBLAS_NUM_THREADS=3 overridden to %s' % got


def test_import_leaves_host_environment():
    got = _numpy_loads_with('route.py', as_main=False)
    assert got == 'None', 'importing route.py set OPENBLAS_NUM_THREADS=%s' % got


if __name__ == '__main__':
    test_cli_loads_numpy_single_threaded()
    test_caller_value_kept()
    test_import_leaves_host_environment()
    print('test_blas_single_thread: all passed')
