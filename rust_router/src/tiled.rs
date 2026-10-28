//! Copy-on-write tiled grids for the obstacle map's per-cell state.
//!
//! The refcounted blocked cells / vias and the proximity costs used to live in
//! `FxHashMap<u64, _>`s: ~25-35 bytes per entry, and every `clone_fresh()`
//! (one per routed net, several alive at once during nested rip-up) copied
//! them whole. Here a grid is a directory of 64 x 64 tiles behind `Arc`s,
//! allocated on first write: a clone copies the directory (one pointer per
//! tile) and shares every tile until one side writes to it, when
//! `Arc::make_mut` copies that tile alone. A clone that removes one net and
//! stamps its own costs therefore owns only the tiles it touched.
//!
//! A grid holds `planes` layers in ONE directory (tile slot = plane, then
//! row, then column), so the hot test is the grid's inline header, one
//! directory load, one tile load -- no per-layer indirection.
//!
//! Semantics are those of the maps they replace: a refcount cell is present
//! while its count is above zero; a cost cell reads 0 when never set (every
//! cost the obstacle map stores is > 0, so "absent" and "0" never differed).

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::{pack_xy, unpack_xy};

pub const TILE_SHIFT: i32 = 6;
pub const TILE_SIDE: i32 = 1 << TILE_SHIFT;
const TILE_MASK: i32 = TILE_SIDE - 1;
const TILE_CELLS: usize = (TILE_SIDE * TILE_SIDE) as usize;
/// Directory growth margin, in tiles (amortizes regrowth).
const DIR_MARGIN: i32 = 4;
/// Directory size cap, in tiles per plane (1 << 20 tiles = 4 G cells = 8 MB
/// of pointers). Tiles beyond it go to a hash map: far-off coordinates cost
/// speed, never correctness.
const DIR_MAX_TILES: i64 = 1 << 20;

/// Cell index within a tile: row-major, so mask word `ly` is the tile's row.
#[inline(always)]
fn local(gx: i32, gy: i32) -> usize {
    (((gy & TILE_MASK) << TILE_SHIFT) | (gx & TILE_MASK)) as usize
}

/// Dense directory of shared tiles over `planes` layers, growing to cover
/// what is written.
#[derive(Clone)]
struct TileDir<T: Clone> {
    min_tx: i32,
    min_ty: i32,
    w: i32, // 0 = empty
    h: i32,
    planes: usize,
    tiles: Vec<Option<Arc<T>>>,
    far: FxHashMap<(u64, usize), Arc<T>>,
}

impl<T: Clone> TileDir<T> {
    fn new(planes: usize) -> Self {
        Self { min_tx: 0, min_ty: 0, w: 0, h: 0, planes, tiles: Vec::new(), far: FxHashMap::default() }
    }

    #[inline(always)]
    fn slot(&self, tx: i32, ty: i32, p: usize) -> Option<usize> {
        let dx = tx.wrapping_sub(self.min_tx);
        let dy = ty.wrapping_sub(self.min_ty);
        if (dx as u32) < self.w as u32 && (dy as u32) < self.h as u32 {
            Some((p * self.h as usize + dy as usize) * self.w as usize + dx as usize)
        } else {
            None
        }
    }

    #[inline(always)]
    fn get(&self, tx: i32, ty: i32, p: usize) -> Option<&T> {
        match self.slot(tx, ty, p) {
            Some(i) => self.tiles[i].as_deref(),
            None => {
                if self.far.is_empty() { None } else { self.far.get(&(pack_xy(tx, ty), p)).map(|a| &**a) }
            }
        }
    }

    /// Mutable tile if it exists (copied first if shared).
    fn get_mut(&mut self, tx: i32, ty: i32, p: usize) -> Option<&mut T> {
        match self.slot(tx, ty, p) {
            Some(i) => self.tiles[i].as_mut().map(Arc::make_mut),
            None => self.far.get_mut(&(pack_xy(tx, ty), p)).map(Arc::make_mut),
        }
    }

    /// Mutable tile, created with `make` when absent (copied first if shared).
    fn get_or_create(&mut self, tx: i32, ty: i32, p: usize, make: impl FnOnce() -> T) -> &mut T {
        if self.slot(tx, ty, p).is_none() {
            self.grow_to(tx, ty);
        }
        match self.slot(tx, ty, p) {
            Some(i) => Arc::make_mut(self.tiles[i].get_or_insert_with(|| Arc::new(make()))),
            None => Arc::make_mut(self.far.entry((pack_xy(tx, ty), p)).or_insert_with(|| Arc::new(make()))),
        }
    }

    fn grow_to(&mut self, tx: i32, ty: i32) {
        let (x0, y0, x1, y1) = if self.w == 0 {
            (tx - DIR_MARGIN, ty - DIR_MARGIN, tx + DIR_MARGIN, ty + DIR_MARGIN)
        } else {
            (self.min_tx.min(tx - DIR_MARGIN), self.min_ty.min(ty - DIR_MARGIN),
             (self.min_tx + self.w - 1).max(tx + DIR_MARGIN), (self.min_ty + self.h - 1).max(ty + DIR_MARGIN))
        };
        let nw = x1 as i64 - x0 as i64 + 1;
        let nh = y1 as i64 - y0 as i64 + 1;
        if nw * nh > DIR_MAX_TILES {
            return; // caller falls back to `far`
        }
        let (nw, nh) = (nw as usize, nh as usize);
        let mut nt: Vec<Option<Arc<T>>> = vec![None; self.planes * nw * nh];
        let (ow, oh) = (self.w as usize, self.h as usize);
        for p in 0..self.planes {
            for dy in 0..oh {
                for dx in 0..ow {
                    let old = &mut self.tiles[(p * oh + dy) * ow + dx];
                    if old.is_some() {
                        let ny = (self.min_ty + dy as i32 - y0) as usize;
                        let nx = (self.min_tx + dx as i32 - x0) as usize;
                        nt[(p * nh + ny) * nw + nx] = old.take();
                    }
                }
            }
        }
        self.min_tx = x0;
        self.min_ty = y0;
        self.w = nw as i32;
        self.h = nh as i32;
        self.tiles = nt;
        // far tiles now inside the window move in
        if !self.far.is_empty() {
            let inside: Vec<(u64, usize)> = self.far.keys().copied()
                .filter(|&(k, p)| { let (x, y) = unpack_xy(k); self.slot(x, y, p).is_some() })
                .collect();
            for key in inside {
                let (x, y) = unpack_xy(key.0);
                let i = self.slot(x, y, key.1).unwrap();
                self.tiles[i] = self.far.remove(&key);
            }
        }
    }

    /// Every allocated tile with its tile coordinates and plane.
    fn for_each(&self, mut f: impl FnMut(i32, i32, usize, &T)) {
        let (w, h) = (self.w as usize, self.h as usize);
        for p in 0..self.planes {
            for dy in 0..h {
                for dx in 0..w {
                    if let Some(t) = &self.tiles[(p * h + dy) * w + dx] {
                        f(self.min_tx + dx as i32, self.min_ty + dy as i32, p, t);
                    }
                }
            }
        }
        for (&(k, p), t) in self.far.iter() {
            let (x, y) = unpack_xy(k);
            f(x, y, p, t);
        }
    }

    /// (tiles not already in `seen`, tiles in all), for memory diagnostics:
    /// a tile shared with a grid counted before is not counted again.
    fn count_tiles(&self, seen: &mut FxHashSet<usize>) -> (usize, usize) {
        let mut own = 0;
        let mut all = 0;
        for t in self.tiles.iter().flatten().chain(self.far.values()) {
            all += 1;
            if seen.insert(Arc::as_ptr(t) as *const u8 as usize) {
                own += 1;
            }
        }
        (own, all)
    }
}

/// One tile of reference counts: a presence mask (one u64 per row) for the
/// hot test and a byte count per cell (255 = "at least 255, see `over`").
#[derive(Clone)]
struct RefTile {
    mask: [u64; TILE_SIDE as usize],
    count: [u8; TILE_CELLS],
}

impl RefTile {
    fn new() -> Self {
        Self { mask: [0; TILE_SIDE as usize], count: [0; TILE_CELLS] }
    }
}

/// Reference-counted cells over `planes` layers (a cell is present while > 0).
#[derive(Clone)]
pub struct RefGrid {
    dir: TileDir<RefTile>,
    /// count above 254 for cells whose byte reads 255 (rare: >= 255 nets
    /// stamping one cell)
    over: FxHashMap<(u64, usize), u32>,
    len: Vec<usize>,
}

impl RefGrid {
    pub fn new(planes: usize) -> Self {
        Self { dir: TileDir::new(planes), over: FxHashMap::default(), len: vec![0; planes] }
    }

    pub fn planes(&self) -> usize {
        self.dir.planes
    }

    /// Present cells on plane `p` (the old per-layer map's `len()`).
    #[inline]
    pub fn len(&self, p: usize) -> usize {
        self.len[p]
    }

    /// Present cells over all planes.
    pub fn total_len(&self) -> usize {
        self.len.iter().sum()
    }

    /// The is_blocked / is_via_blocked hot path. Each mask word is one
    /// 64-cell row of the tile, so the row is `gy & 63` and the bit `gx & 63`.
    #[inline(always)]
    pub fn contains(&self, gx: i32, gy: i32, p: usize) -> bool {
        match self.dir.get(gx >> TILE_SHIFT, gy >> TILE_SHIFT, p) {
            Some(t) => (t.mask[(gy & TILE_MASK) as usize] >> (gx & TILE_MASK)) & 1 != 0,
            None => false,
        }
    }

    pub fn get(&self, gx: i32, gy: i32, p: usize) -> u32 {
        match self.dir.get(gx >> TILE_SHIFT, gy >> TILE_SHIFT, p) {
            Some(t) => {
                let c = t.count[local(gx, gy)];
                if c == 255 { 254 + self.over.get(&(pack_xy(gx, gy), p)).copied().unwrap_or(1) } else { c as u32 }
            }
            None => 0,
        }
    }

    /// Add `by` (> 0) to the count; true on the 0 -> nonzero transition.
    pub fn add(&mut self, gx: i32, gy: i32, p: usize, by: u32) -> bool {
        if by == 0 {
            return false;
        }
        let t = self.dir.get_or_create(gx >> TILE_SHIFT, gy >> TILE_SHIFT, p, RefTile::new);
        let i = local(gx, gy);
        let c = t.count[i] as u32;
        let was_zero = c == 0;
        if was_zero {
            t.mask[i >> TILE_SHIFT] |= 1u64 << (i & TILE_MASK as usize);
        }
        if c == 255 {
            *self.over.get_mut(&(pack_xy(gx, gy), p)).unwrap() += by;
        } else if c + by >= 255 {
            t.count[i] = 255;
            self.over.insert((pack_xy(gx, gy), p), c + by - 254);
        } else {
            t.count[i] = (c + by) as u8;
        }
        if was_zero {
            self.len[p] += 1;
        }
        was_zero
    }

    /// Decrement a present cell; true on the 1 -> 0 transition. An absent
    /// cell is left alone (the maps' `get_mut` miss).
    pub fn dec(&mut self, gx: i32, gy: i32, p: usize) -> bool {
        let (tx, ty) = (gx >> TILE_SHIFT, gy >> TILE_SHIFT);
        let i = local(gx, gy);
        // read-only probe first: decrementing an absent cell must not copy a
        // shared tile
        match self.dir.get(tx, ty, p) {
            Some(t) if t.count[i] != 0 => {}
            _ => return false,
        }
        let t = self.dir.get_mut(tx, ty, p).unwrap();
        let c = t.count[i];
        if c == 255 {
            let k = (pack_xy(gx, gy), p);
            let e = self.over.get_mut(&k).unwrap();
            if *e > 1 {
                *e -= 1;
            } else {
                // 255 -> 254: the byte becomes exact again
                self.over.remove(&k);
                t.count[i] = 254;
            }
            false
        } else if c > 1 {
            t.count[i] = c - 1;
            false
        } else {
            t.count[i] = 0;
            t.mask[i >> TILE_SHIFT] &= !(1u64 << (i & TILE_MASK as usize));
            self.len[p] -= 1;
            true
        }
    }

    /// Every present cell of plane `p` with its count.
    pub fn for_each(&self, p: usize, mut f: impl FnMut(i32, i32, u32)) {
        let over = &self.over;
        self.dir.for_each(|tx, ty, tp, t| {
            if tp != p {
                return;
            }
            for (row, &m) in t.mask.iter().enumerate() {
                let mut m = m;
                while m != 0 {
                    let col = m.trailing_zeros() as i32;
                    m &= m - 1;
                    let gx = (tx << TILE_SHIFT) + col;
                    let gy = (ty << TILE_SHIFT) + row as i32;
                    let c = t.count[local(gx, gy)];
                    let n = if c == 255 { 254 + over.get(&(pack_xy(gx, gy), p)).copied().unwrap_or(1) } else { c as u32 };
                    f(gx, gy, n);
                }
            }
        });
    }

    pub fn count_tiles(&self, seen: &mut FxHashSet<usize>) -> (usize, usize) {
        self.dir.count_tiles(seen)
    }
}

/// One tile of costs.
#[derive(Clone)]
struct CostTile {
    v: [i32; TILE_CELLS],
}

/// Per-cell i32 costs over `planes` layers, 0 when never written.
#[derive(Clone)]
pub struct CostGrid {
    dir: TileDir<CostTile>,
    len: Vec<usize>,
}

impl CostGrid {
    pub fn new(planes: usize) -> Self {
        Self { dir: TileDir::new(planes), len: vec![0; planes] }
    }

    /// Nonzero cells over all planes (the old maps' `len()`: they held no zeros).
    pub fn total_len(&self) -> usize {
        self.len.iter().sum()
    }

    #[inline(always)]
    pub fn get(&self, gx: i32, gy: i32, p: usize) -> i32 {
        match self.dir.get(gx >> TILE_SHIFT, gy >> TILE_SHIFT, p) {
            Some(t) => t.v[local(gx, gy)],
            None => 0,
        }
    }

    #[inline]
    fn cell_mut(dir: &mut TileDir<CostTile>, gx: i32, gy: i32, p: usize) -> &mut i32 {
        &mut dir.get_or_create(gx >> TILE_SHIFT, gy >> TILE_SHIFT, p,
                               || CostTile { v: [0; TILE_CELLS] }).v[local(gx, gy)]
    }

    /// Raise a cell to `cost` (> 0) if below it.
    #[inline]
    pub fn raise(&mut self, gx: i32, gy: i32, p: usize, cost: i32) {
        if cost <= 0 || self.get(gx, gy, p) >= cost {
            return;
        }
        let c = Self::cell_mut(&mut self.dir, gx, gy, p);
        if *c == 0 {
            self.len[p] += 1;
        }
        *c = cost;
    }

    /// `raise` over the run of cells (gx0 + i, gy), i in 0..costs.len(),
    /// one tile lookup per tile the run crosses instead of two per cell. A
    /// tile is created or copied only when some cell of it actually rises,
    /// exactly as cell-by-cell `raise` would.
    pub fn raise_row(&mut self, gx0: i32, gy: i32, p: usize, costs: &[i32]) {
        let ty = gy >> TILE_SHIFT;
        let mut i = 0usize;
        while i < costs.len() {
            let gx = gx0 + i as i32;
            let tx = gx >> TILE_SHIFT;
            let n = ((TILE_SIDE - (gx & TILE_MASK)) as usize).min(costs.len() - i);
            let seg = &costs[i..i + n];
            let rises = match self.dir.get(tx, ty, p) {
                Some(t) => {
                    let base = local(gx, gy);
                    seg.iter().enumerate().any(|(k, &c)| c > 0 && c > t.v[base + k])
                }
                None => seg.iter().any(|&c| c > 0),
            };
            if rises {
                let t = self.dir.get_or_create(tx, ty, p, || CostTile { v: [0; TILE_CELLS] });
                let base = local(gx, gy);
                for (k, &c) in seg.iter().enumerate() {
                    let v = &mut t.v[base + k];
                    if c > 0 && c > *v {
                        if *v == 0 {
                            self.len[p] += 1;
                        }
                        *v = c;
                    }
                }
            }
            i += n;
        }
    }

    /// Add `cost` to a cell.
    #[inline]
    pub fn add(&mut self, gx: i32, gy: i32, p: usize, cost: i32) {
        if cost == 0 {
            return;
        }
        let c = Self::cell_mut(&mut self.dir, gx, gy, p);
        let was = *c;
        *c += cost;
        if was == 0 && *c != 0 {
            self.len[p] += 1;
        } else if was != 0 && *c == 0 {
            self.len[p] -= 1;
        }
    }

    pub fn clear(&mut self) {
        *self = Self::new(self.dir.planes);
    }

    /// Every nonzero cell of plane `p` with its cost.
    pub fn for_each(&self, p: usize, mut f: impl FnMut(i32, i32, i32)) {
        self.dir.for_each(|tx, ty, tp, t| {
            if tp != p {
                return;
            }
            for (i, &c) in t.v.iter().enumerate() {
                if c != 0 {
                    f((tx << TILE_SHIFT) + (i as i32 & TILE_MASK), (ty << TILE_SHIFT) + (i as i32 >> TILE_SHIFT), c);
                }
            }
        });
    }

    pub fn count_tiles(&self, seen: &mut FxHashSet<usize>) -> (usize, usize) {
        self.dir.count_tiles(seen)
    }
}

/// Bytes of one tile of each kind, for memory diagnostics.
pub const REF_TILE_BYTES: usize = std::mem::size_of::<RefTile>();
pub const COST_TILE_BYTES: usize = std::mem::size_of::<CostTile>();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refcounts_match_a_map() {
        let mut g = RefGrid::new(3);
        let mut m: FxHashMap<(u64, usize), u32> = FxHashMap::default();
        let mut s: u64 = 12345;
        for _ in 0..300_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // a few far-off coordinates exercise the overflow directory
            let gx = ((s >> 33) % 300) as i32 - 150 + if (s >> 20) & 7 == 0 { 40_000_000 } else { 0 };
            let gy = ((s >> 13) % 300) as i32 - 150;
            let p = ((s >> 40) % 3) as usize;
            let k = (pack_xy(gx, gy), p);
            if (s >> 60) < 9 {
                let by = if (s >> 50) & 15 == 0 { 300 } else { 1 };
                let was0 = !m.contains_key(&k);
                *m.entry(k).or_insert(0) += by;
                assert_eq!(g.add(gx, gy, p, by), was0);
            } else {
                let became0 = match m.get_mut(&k) {
                    Some(c) if *c > 1 => { *c -= 1; false }
                    Some(_) => { m.remove(&k); true }
                    None => false,
                };
                assert_eq!(g.dec(gx, gy, p), became0);
            }
            assert_eq!(g.get(gx, gy, p), m.get(&k).copied().unwrap_or(0));
            assert_eq!(g.contains(gx, gy, p), m.contains_key(&k));
        }
        assert_eq!(g.total_len(), m.len());
        let mut seen = 0;
        for p in 0..3 {
            assert_eq!(g.len(p), m.keys().filter(|k| k.1 == p).count());
            g.for_each(p, |x, y, c| { seen += 1; assert_eq!(m[&(pack_xy(x, y), p)], c); });
        }
        assert_eq!(seen, m.len());
    }

    #[test]
    fn clone_shares_until_written() {
        let mut a = RefGrid::new(2);
        for x in 0..640 { a.add(x, 3, 1, 1); }
        let mut b = a.clone();
        let mut seen = FxHashSet::default();
        let (own_a, _) = a.count_tiles(&mut seen);
        let (own_b, all_b) = b.count_tiles(&mut seen);
        assert_eq!((own_a, own_b, all_b), (10, 0, 10));
        b.dec(5, 3, 1);
        assert!(a.contains(5, 3, 1) && !b.contains(5, 3, 1));
        let mut seen = FxHashSet::default();
        a.count_tiles(&mut seen);
        assert_eq!(b.count_tiles(&mut seen).0, 1);
        // decrementing an absent cell copies nothing
        let c = a.clone();
        let mut d = c.clone();
        d.dec(7, 50, 1);
        d.dec(7, 3, 0);
        let mut seen = FxHashSet::default();
        c.count_tiles(&mut seen);
        assert_eq!(d.count_tiles(&mut seen).0, 0);
    }

    #[test]
    fn raise_row_is_raise_per_cell() {
        let mut a = CostGrid::new(2);
        let mut b = CostGrid::new(2);
        let mut s: u64 = 99;
        for _ in 0..20_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let gx0 = ((s >> 33) % 400) as i32 - 200;
            let gy = ((s >> 13) % 300) as i32 - 150;
            let p = ((s >> 45) % 2) as usize;
            let n = 1 + ((s >> 50) % 150) as usize;
            let costs: Vec<i32> = (0..n).map(|k| ((s >> (k % 40)) % 9) as i32 - 2).collect();
            a.raise_row(gx0, gy, p, &costs);
            for (k, &c) in costs.iter().enumerate() {
                b.raise(gx0 + k as i32, gy, p, c);
            }
        }
        assert_eq!(a.total_len(), b.total_len());
        for p in 0..2 {
            for gx in -210..260 {
                for gy in -150..150 {
                    assert_eq!(a.get(gx, gy, p), b.get(gx, gy, p));
                }
            }
        }
        // a row that raises nothing copies no shared tile
        let c = a.clone();
        let mut d = c.clone();
        d.raise_row(-200, 0, 0, &[0, -1, 0]);
        let mut seen = FxHashSet::default();
        c.count_tiles(&mut seen);
        assert_eq!(d.count_tiles(&mut seen).0, 0);
    }

    #[test]
    fn costs_raise_and_add() {
        let mut g = CostGrid::new(2);
        g.raise(-3, 70, 1, 5);
        g.raise(-3, 70, 1, 2);
        assert_eq!(g.get(-3, 70, 1), 5);
        g.add(-3, 70, 1, 4);
        g.add(1000, 1000, 0, 7);
        assert_eq!((g.get(-3, 70, 1), g.get(-3, 70, 0), g.get(1000, 1000, 0), g.get(0, 0, 0), g.total_len()),
                   (9, 0, 7, 0, 2));
    }
}
