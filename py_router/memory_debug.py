"""Memory debugging utilities for tracking routing memory usage."""
from __future__ import annotations

import sys
from typing import Dict, Any, Optional

# Try to import memory tracking libraries (graceful fallback)
_HAS_RESOURCE = False
_HAS_PSUTIL = False

try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    pass

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    pass


def _windows_working_set_bytes() -> int:
    """This process's working set (what psutil reports as rss on Windows),
    read through psapi with ctypes. psutil is not a dependency and Windows
    has no `resource` module, so without this --debug-memory printed 0 MB
    at every stage on Windows."""
    import ctypes
    import ctypes.wintypes as wt

    class _Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
        _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]

    c = _Counters()
    c.cb = ctypes.sizeof(c)
    # HANDLE(-1) is the current-process pseudo handle; a plain int argument
    # is passed as a 32-bit value on 64-bit and the call fails.
    if not ctypes.WinDLL('psapi').GetProcessMemoryInfo(wt.HANDLE(-1), ctypes.byref(c), c.cb):
        return 0
    return c.WorkingSetSize


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB.

    Tries multiple methods in order of reliability:
    1. psutil (most accurate, cross-platform)
    2. psapi through ctypes (Windows)
    3. resource (Unix only)
    """
    if _HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    if sys.platform == 'win32':
        try:
            return _windows_working_set_bytes() / (1024 * 1024)
        except (OSError, AttributeError):
            return 0.0

    if _HAS_RESOURCE:
        # maxrss is in KB on Linux, bytes on macOS
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return maxrss / (1024 * 1024)  # bytes to MB
        return maxrss / 1024  # KB to MB

    return 0.0  # Unable to measure


def format_memory_stats(label: str, memory_mb: float, delta_mb: Optional[float] = None) -> str:
    """Format memory statistics for display."""
    if delta_mb is not None and abs(delta_mb) > 0.1:
        sign = "+" if delta_mb > 0 else ""
        return f"[MEMORY] {label}: {memory_mb:.1f} MB ({sign}{delta_mb:.1f} MB)"
    return f"[MEMORY] {label}: {memory_mb:.1f} MB"


def estimate_net_obstacles_cache_mb(cache: Dict) -> float:
    """Estimate memory size of net_obstacles_cache (Dict[int, NetObstacleData])."""
    if not cache:
        return 0.0

    total = sys.getsizeof(cache)
    for net_id, data in cache.items():
        total += sys.getsizeof(net_id)
        total += sys.getsizeof(data)
        if hasattr(data, 'blocked_cells'):
            # numpy arrays: use nbytes for actual memory usage
            if hasattr(data.blocked_cells, 'nbytes'):
                total += data.blocked_cells.nbytes
            else:
                total += sys.getsizeof(data.blocked_cells)
                total += len(data.blocked_cells) * 24  # tuple of 3 ints (legacy)
        if hasattr(data, 'blocked_vias'):
            if hasattr(data.blocked_vias, 'nbytes'):
                total += data.blocked_vias.nbytes
            else:
                total += sys.getsizeof(data.blocked_vias)
                total += len(data.blocked_vias) * 16  # tuple of 2 ints (legacy)
        # #815: segment keep-outs live in the SPAN arrays, so a report that
        # counts only the cell arrays now under-reports the cache badly.
        for _attr in ('blocked_cell_spans', 'blocked_via_spans', 'blocked_vias_small'):
            _a = getattr(data, _attr, None)
            if _a is not None and hasattr(_a, 'nbytes'):
                total += _a.nbytes
    return total / (1024 * 1024)


def estimate_track_proximity_cache_mb(cache: Dict) -> float:
    """Estimate memory size of track_proximity_cache (numpy arrays)."""
    if not cache:
        return 0.0

    total = sys.getsizeof(cache)
    for net_id, arr in cache.items():
        total += sys.getsizeof(net_id)
        if hasattr(arr, 'nbytes'):
            # numpy array: 4 columns * 4 bytes (int32) = 16 bytes per row
            total += arr.nbytes + 128  # array overhead
        else:
            # Fallback for old dict format
            total += sys.getsizeof(arr)
    return total / (1024 * 1024)


def estimate_routed_paths_mb(paths: Dict) -> float:
    """Estimate memory size of routed_net_paths (Dict[int, List[Tuple]])."""
    if not paths:
        return 0.0

    total = sys.getsizeof(paths)
    for net_id, path in paths.items():
        total += sys.getsizeof(net_id) + sys.getsizeof(path)
        if path:
            total += len(path) * 24  # tuple of 3 ints per coordinate
    return total / (1024 * 1024)


def format_obstacle_map_stats(obstacles) -> str:
    """Format Rust GridObstacleMap statistics."""
    if obstacles is None or not hasattr(obstacles, 'get_stats'):
        return "[MEMORY] Obstacle map: N/A (no get_stats method)"

    stats = obstacles.get_stats()
    # get_stats() returns 7 counts (free_vias was added later); tolerate 6 or 7.
    blocked_cells, blocked_vias, stub_prox, layer_prox, cross_layer, source_target = stats[:6]
    free_vias = stats[6] if len(stats) > 6 else 0

    # Blocked cells/vias and proximity costs are 64 x 64 tiles (grid_router
    # 0.24): tile_stats() gives their bytes, tiles shared with clones
    # included. The rest are hash sets/maps, ~16-24 B per entry.
    MAP_B, SET_B = 24, 16
    tile_bytes = obstacles.tile_stats()[2] if hasattr(obstacles, 'tile_stats') else 0
    estimated_mb = (tile_bytes + cross_layer * MAP_B
                    + (source_target + free_vias) * SET_B) / (1024 * 1024)

    return (f"[MEMORY] Rust obstacle map: ~{estimated_mb:.1f} MB estimated\n"
            f"         blocked_cells: {blocked_cells:,}, blocked_vias: {blocked_vias:,}\n"
            f"         stub_proximity: {stub_prox:,}, layer_proximity: {layer_prox:,}\n"
            f"         cross_layer: {cross_layer:,}, source_target: {source_target:,}, free_vias: {free_vias:,}")
