"""One OpenBLAS thread for the routing CLIs.

numpy and scipy each load their OWN OpenBLAS (two DLLs in the wheels), and
each one commits a work buffer per thread at import -- one thread per core.
On a 20-core machine that is 1.22 GB of commit per routing process before
any board is read (numpy 626 MB, scipy.spatial another 635 MB; measured with
GetProcessMemoryInfo). The router never calls BLAS (no matmul, solve or
lstsq on its paths), so the buffers are never used; Windows still charges
them to the commit limit, which is what makes a routing process fail to
allocate when several run side by side. With one thread the same imports
commit 40 MB, and a 150 x 107.5 mm 4-layer route at 0.05 mm peaks at
1.55 GB instead of 2.80 GB with an identical JSON_SUMMARY.

OpenBLAS reads the variable once, when its library loads, so this must run
before the first `import numpy` -- i.e. at the top of an entry script, above
`startup_checks` (which imports numpy). It only acts when the script is the
program being run (`__main__`): an IMPORT of route.py (the GUI plugin, a
test) leaves the host process's environment alone. `setdefault` keeps an
explicit OPENBLAS_NUM_THREADS from the caller.
"""
from __future__ import annotations

import os


def single_thread_if_main(module_name: str) -> None:
    if module_name == '__main__':
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
