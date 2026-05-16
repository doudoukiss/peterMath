use crate::metrics::Metrics;
use crate::palette;
use crate::simulation::{wrap_index, RenderStyle};

pub struct BitGrid {
    len: usize,
    words: Vec<u64>,
}

impl BitGrid {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            words: vec![0; len.div_ceil(64)],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn fill(&mut self, value: bool) {
        let fill = if value { u64::MAX } else { 0 };
        self.words.fill(fill);
        if value {
            self.clear_unused_tail_bits();
        }
    }

    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        let word = self.words[idx / 64];
        (word & (1_u64 << (idx % 64))) != 0
    }

    pub fn set(&mut self, idx: usize, value: bool) {
        debug_assert!(idx < self.len);
        let mask = 1_u64 << (idx % 64);
        let word = &mut self.words[idx / 64];
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    fn clear_unused_tail_bits(&mut self) {
        let used_bits = self.len % 64;
        if used_bits == 0 {
            return;
        }
        if let Some(last) = self.words.last_mut() {
            *last &= (1_u64 << used_bits) - 1;
        }
    }
}

pub struct LifeSim {
    w: usize,
    h: usize,
    cells: BitGrid,
    next: BitGrid,
    age: Vec<f32>,
    previous_age: Vec<f32>,
    pub random_density: f32,
    pub seed: u64,
}

impl LifeSim {
    pub fn new(w: usize, h: usize, seed: u64) -> Self {
        let mut sim = Self {
            w,
            h,
            cells: BitGrid::new(w * h),
            next: BitGrid::new(w * h),
            age: vec![0.0; w * h],
            previous_age: vec![0.0; w * h],
            random_density: 0.18,
            seed,
        };
        sim.reset_preset("symmetric_seed");
        sim
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub fn reset_preset(&mut self, preset: &str) {
        self.cells.fill(false);
        self.next.fill(false);
        self.age.fill(0.0);
        match preset {
            "symmetric_seed" => self.seed_symmetric(),
            _ => self.reset_random(),
        }
    }

    pub fn reset_random(&mut self) {
        let mut rng = fastrand::Rng::with_seed(self.seed + 17);
        for i in 0..self.cells.len() {
            self.cells.set(i, rng.f32() < self.random_density);
        }
        for (i, age) in self.age.iter_mut().enumerate() {
            *age = if self.cells.get(i) { 1.0 } else { 0.0 };
        }
    }

    fn seed_symmetric(&mut self) {
        let cx = self.w / 2;
        let cy = self.h / 2;
        let pattern = [
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 1),
            (1, 2),
            (-8, -5),
            (-8, -4),
            (-8, -3),
            (-7, -3),
            (-6, -4),
        ];
        for &(dx, dy) in &pattern {
            for sx in [-1, 1] {
                let x = cx as isize + dx * sx;
                let y = cy as isize + dy;
                let idx = wrap_index(x, y, self.w, self.h);
                self.cells.set(idx, true);
                self.age[idx] = 1.0;
            }
        }
    }

    pub fn step(&mut self) {
        self.previous_age.copy_from_slice(&self.age);
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let n = self.neighbor_count(x, y);
                let alive = self.cells.get(idx);
                self.next.set(
                    idx,
                    matches!((alive, n), (true, 2) | (true, 3) | (false, 3)),
                );
            }
        }
        std::mem::swap(&mut self.cells, &mut self.next);
        for i in 0..self.cells.len() {
            self.age[i] = if self.cells.get(i) {
                (self.age[i] + 0.08).clamp(0.0, 1.0)
            } else {
                (self.age[i] * 0.90).clamp(0.0, 1.0)
            };
        }
    }

    fn neighbor_count(&self, x: usize, y: usize) -> u8 {
        let mut count = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let idx = wrap_index(x as isize + dx, y as isize + dy, self.w, self.h);
                if self.cells.get(idx) {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn render_rgba(&self, style: RenderStyle, out: &mut [u8]) {
        for (i, px) in out.chunks_exact_mut(4).enumerate() {
            let v = self.age[i];
            let rgba = match style {
                RenderStyle::RawMath => {
                    if self.cells.get(i) {
                        [255, 255, 255, 255]
                    } else {
                        [16, 16, 16, 255]
                    }
                }
                RenderStyle::Artistic => palette::scientific(v),
            };
            px.copy_from_slice(&rgba);
        }
    }

    pub fn metrics(&self) -> Metrics {
        Metrics::from_scalar_grid(&self.age, Some(&self.previous_age), self.w, self.h)
    }
}
