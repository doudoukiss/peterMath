use crate::metrics::Metrics;
use crate::palette;
use crate::simulation::{wrap_index, RenderStyle};

pub struct ReactionDiffusionSim {
    w: usize,
    h: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    next_a: Vec<f32>,
    next_b: Vec<f32>,
    previous_b: Vec<f32>,
    pub diff_a: f32,
    pub diff_b: f32,
    pub feed: f32,
    pub kill: f32,
    pub dt: f32,
    pub seed: u64,
}

impl ReactionDiffusionSim {
    pub fn new(w: usize, h: usize, seed: u64) -> Self {
        let mut sim = Self {
            w,
            h,
            a: vec![1.0; w * h],
            b: vec![0.0; w * h],
            next_a: vec![1.0; w * h],
            next_b: vec![0.0; w * h],
            previous_b: vec![0.0; w * h],
            diff_a: 0.16,
            diff_b: 0.08,
            feed: 0.0367,
            kill: 0.0649,
            dt: 1.0,
            seed,
        };
        sim.reset_preset("labyrinth");
        sim
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub fn field(&self) -> &[f32] {
        &self.b
    }

    pub fn reset_preset(&mut self, preset: &str) {
        self.a.fill(1.0);
        self.b.fill(0.0);
        self.previous_b.fill(0.0);
        match preset {
            "mitosis" => {
                self.feed = 0.0367;
                self.kill = 0.0649;
                self.seed_dots(9, 7.0);
            }
            "labyrinth" => {
                self.feed = 0.030;
                self.kill = 0.055;
                self.seed_dots_full(150, 4.8);
            }
            "spots" => {
                self.feed = 0.022;
                self.kill = 0.051;
                self.seed_dots_full(80, 3.8);
            }
            "waves" => {
                self.feed = 0.014;
                self.kill = 0.047;
                self.seed_dots_full(54, 5.8);
            }
            _ => self.seed_dots(11, 6.5),
        }
        self.previous_b.copy_from_slice(&self.b);
    }

    fn seed_dots(&mut self, count: usize, radius: f32) {
        let mut rng = fastrand::Rng::with_seed(self.seed);
        for _ in 0..count {
            let cx = (0.18 + rng.f32() * 0.64) * self.w as f32;
            let cy = (0.18 + rng.f32() * 0.64) * self.h as f32;
            let r = radius as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    if (dx * dx + dy * dy) as f32 <= radius * radius {
                        let idx = wrap_index(cx as isize + dx, cy as isize + dy, self.w, self.h);
                        self.b[idx] = 1.0;
                        self.a[idx] = 0.0;
                    }
                }
            }
        }
    }

    fn seed_dots_full(&mut self, count: usize, radius: f32) {
        let mut rng = fastrand::Rng::with_seed(self.seed);
        for _ in 0..count {
            let cx = (0.06 + rng.f32() * 0.88) * self.w as f32;
            let cy = (0.06 + rng.f32() * 0.88) * self.h as f32;
            let r = radius as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    if (dx * dx + dy * dy) as f32 <= radius * radius {
                        let idx = wrap_index(cx as isize + dx, cy as isize + dy, self.w, self.h);
                        self.b[idx] = 1.0;
                        self.a[idx] = 0.0;
                    }
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.previous_b.copy_from_slice(&self.b);
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let a = self.a[idx];
                let b = self.b[idx];
                let reaction = a * b * b;
                let lap_a = self.laplacian(&self.a, x, y);
                let lap_b = self.laplacian(&self.b, x, y);
                self.next_a[idx] = (a + self.dt
                    * (self.diff_a * lap_a - reaction + self.feed * (1.0 - a)))
                    .clamp(0.0, 1.0);
                self.next_b[idx] = (b + self.dt
                    * (self.diff_b * lap_b + reaction - (self.kill + self.feed) * b))
                    .clamp(0.0, 1.0);
            }
        }
        std::mem::swap(&mut self.a, &mut self.next_a);
        std::mem::swap(&mut self.b, &mut self.next_b);
    }

    fn laplacian(&self, grid: &[f32], x: usize, y: usize) -> f32 {
        let c = -grid[y * self.w + x];
        let n = grid[wrap_index(x as isize, y as isize - 1, self.w, self.h)] * 0.20;
        let s = grid[wrap_index(x as isize, y as isize + 1, self.w, self.h)] * 0.20;
        let e = grid[wrap_index(x as isize + 1, y as isize, self.w, self.h)] * 0.20;
        let w = grid[wrap_index(x as isize - 1, y as isize, self.w, self.h)] * 0.20;
        let ne = grid[wrap_index(x as isize + 1, y as isize - 1, self.w, self.h)] * 0.05;
        let nw = grid[wrap_index(x as isize - 1, y as isize - 1, self.w, self.h)] * 0.05;
        let se = grid[wrap_index(x as isize + 1, y as isize + 1, self.w, self.h)] * 0.05;
        let sw = grid[wrap_index(x as isize - 1, y as isize + 1, self.w, self.h)] * 0.05;
        c + n + s + e + w + ne + nw + se + sw
    }

    pub fn render_rgba(&self, style: RenderStyle, out: &mut [u8]) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let v = (self.b[i] * 1.4 - self.a[i] * 0.15).clamp(0.0, 1.0);
                let rgba = match style {
                    RenderStyle::RawMath => palette::raw_gray(self.b[i]),
                    RenderStyle::Artistic => {
                        let gx = self.b[wrap_index(x as isize + 1, y as isize, self.w, self.h)]
                            - self.b[wrap_index(x as isize - 1, y as isize, self.w, self.h)];
                        let gy = self.b[wrap_index(x as isize, y as isize + 1, self.w, self.h)]
                            - self.b[wrap_index(x as isize, y as isize - 1, self.w, self.h)];
                        let edge = (gx * gx + gy * gy).sqrt() * 2.5;
                        palette::reaction_field(v, edge)
                    }
                };
                out[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
    }

    pub fn metrics(&self) -> Metrics {
        Metrics::from_scalar_grid(&self.b, Some(&self.previous_b), self.w, self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_labyrinth_changes_over_time() {
        let mut sim = ReactionDiffusionSim::new(96, 96, 2001);
        let initial = sim.metrics();
        let mut checkpoints = Vec::new();
        for step in 1..=900 {
            sim.step();
            if matches!(step, 50 | 300 | 900) {
                checkpoints.push(sim.metrics());
            }
        }

        assert_eq!(checkpoints.len(), 3);
        for metrics in checkpoints {
            let delta = (metrics.mass - initial.mass).abs()
                + (metrics.entropy - initial.entropy).abs()
                + (metrics.active as f32 - initial.active as f32).abs() / 10_000.0;
            assert!(
                delta > 0.002,
                "reaction-diffusion checkpoint did not visibly diverge: {delta}"
            );
        }
    }
}
