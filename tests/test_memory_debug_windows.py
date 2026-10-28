"""--debug-memory measures on Windows without psutil (memory_debug.py).

psutil is not a dependency and Windows has no `resource` module, so
get_process_memory_mb() returned 0.0 there and every [MEMORY] line of
route.py / route_diff.py --debug-memory read "0.0 MB". It now falls back to
the process working set read through psapi.

Cases (Windows only; elsewhere the fallback is not reached and the test
reports a skip):
  1. With psutil hidden, the reading is positive and grows when 64 MB are
     allocated and touched.

Run: python3 tests/test_memory_debug_windows.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'py_router'))

import memory_debug


def test_reads_working_set_without_psutil():
    if sys.platform != 'win32':
        print('SKIP: Windows-only fallback')
        return
    saved = memory_debug._HAS_PSUTIL
    memory_debug._HAS_PSUTIL = False
    try:
        before = memory_debug.get_process_memory_mb()
        assert before > 0, 'no psutil: get_process_memory_mb() = %r' % before
        block = bytearray(64 * 1024 * 1024)
        for i in range(0, len(block), 4096):
            block[i] = 1
        after = memory_debug.get_process_memory_mb()
        assert after >= before + 48, 'working set %.1f -> %.1f MB after 64 MB touched' % (before, after)
        del block
    finally:
        memory_debug._HAS_PSUTIL = saved


if __name__ == '__main__':
    test_reads_working_set_without_psutil()
    print('test_memory_debug_windows: all passed')
