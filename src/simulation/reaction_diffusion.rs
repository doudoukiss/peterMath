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

#[derive(Debug, Clone, Copy, Default)]
pub struct FieldStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub delta_mean: f32,
    pub activity: f32,
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
        sim.reset_preset("mitosis");
        sim
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub fn field(&self) -> &[f32] {
        &self.b
    }

    pub fn reset_preset(&mut self, preset: &str) {
        self.diff_a = 0.16;
        self.diff_b = 0.08;
        self.dt = 1.0;
        self.a.fill(1.0);
        self.b.fill(0.0);
        self.next_a.fill(1.0);
        self.next_b.fill(0.0);
        match preset {
            "spots" => {
                self.feed = 0.042;
                self.kill = 0.060;
                self.seed_dots_full(54, 3.4);
            }
            "waves" => {
                self.feed = 0.026;
                self.kill = 0.051;
                self.seed_wave_bands();
            }
            "mitosis" => {
                self.feed = 0.0367;
                self.kill = 0.0649;
                self.seed_mitosis_islands();
            }
            "labyrinth" => {
                self.feed = 0.029;
                self.kill = 0.057;
                self.seed_dots_full(132, 4.4);
            }
            _ => {
                self.feed = 0.0367;
                self.kill = 0.0649;
                self.seed_mitosis_islands();
            }
        }
        self.next_a.copy_from_slice(&self.a);
        self.next_b.copy_from_slice(&self.b);
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

    fn seed_wave_bands(&mut self) {
        let cx = self.w as f32 * 0.5;
        let cy = self.h as f32 * 0.5;
        let max_r = self.w.min(self.h) as f32 * 0.42;
        for y in 0..self.h {
            for x in 0..self.w {
                let fx = x as f32 - cx;
                let fy = y as f32 - cy;
                let r = (fx * fx + fy * fy).sqrt();
                let diagonal = (fx * 0.72 + fy * 0.42).abs();
                let ring = ((r / 9.5).fract() - 0.5).abs() < 0.055 && r < max_r;
                let stripe = ((diagonal / 13.0).fract() - 0.5).abs() < 0.045;
                if ring || stripe {
                    let idx = y * self.w + x;
                    self.b[idx] = 0.94;
                    self.a[idx] = 0.10;
                }
            }
        }
        self.seed_dots(6, 5.5);
    }

    fn seed_mitosis_islands(&mut self) {
        let centers = [
            (0.30, 0.30, 9.0),
            (0.64, 0.31, 8.0),
            (0.44, 0.55, 10.0),
            (0.72, 0.66, 7.0),
            (0.23, 0.72, 8.5),
        ];
        for &(xr, yr, radius) in &centers {
            let cx = (xr * self.w as f32) as isize;
            let cy = (yr * self.h as f32) as isize;
            let r = radius as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let d2 = (dx * dx + dy * dy) as f32;
                    if d2 <= radius * radius {
                        let idx = wrap_index(cx + dx, cy + dy, self.w, self.h);
                        let edge = (d2.sqrt() / radius).clamp(0.0, 1.0);
                        self.b[idx] = (0.98 - edge * 0.18).clamp(0.0, 1.0);
                        self.a[idx] = (0.12 + edge * 0.18).clamp(0.0, 1.0);
                    }
                }
            }
        }
        self.seed_dots(10, 4.8);
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
        let stats = self.field_stats();
        let span = (stats.max - stats.min).max(0.035);
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let rgba = match style {
                    RenderStyle::RawMath => palette::raw_gray(self.b[i]),
                    RenderStyle::Artistic => {
                        let gx = self.b[wrap_index(x as isize + 1, y as isize, self.w, self.h)]
                            - self.b[wrap_index(x as isize - 1, y as isize, self.w, self.h)];
                        let gy = self.b[wrap_index(x as isize, y as isize + 1, self.w, self.h)]
                            - self.b[wrap_index(x as isize, y as isize - 1, self.w, self.h)];
                        let edge = (gx * gx + gy * gy).sqrt() * 5.0;
                        let normalized = ((self.b[i] - stats.min) / span).clamp(0.0, 1.0);
                        let concentration = (self.b[i] * 1.8).clamp(0.0, 1.0);
                        let v = (normalized * 0.72 + concentration * 0.22 + edge * 0.12)
                            .clamp(0.0, 1.0);
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

    pub fn field_stats(&self) -> FieldStats {
        if self.b.is_empty() {
            return FieldStats::default();
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0;
        let mut delta_sum = 0.0;
        let mut active = 0usize;
        for (value, previous) in self.b.iter().zip(self.previous_b.iter()) {
            min = min.min(*value);
            max = max.max(*value);
            sum += *value;
            delta_sum += (*value - *previous).abs();
            if *value > 0.035 {
                active += 1;
            }
        }
        let n = self.b.len() as f32;
        FieldStats {
            min,
            max,
            mean: sum / n,
            delta_mean: delta_sum / n,
            activity: active as f32 / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_show_measurable_metric_change() {
        for preset in ["spots", "waves", "labyrinth", "mitosis"] {
            let mut sim = ReactionDiffusionSim::new(96, 96, 2001);
            sim.reset_preset(preset);
            let initial = sim.metrics();
            let mut step_count = 0;
            for target in [50, 300, 900] {
                while step_count < target {
                    sim.step();
                    step_count += 1;
                }
                let current = sim.metrics();
                let changed = (current.mass - initial.mass).abs() > 0.0005
                    || (current.entropy - initial.entropy).abs() > 0.0005
                    || (current.vitality - initial.vitality).abs() > 0.0005;
                assert!(changed, "{preset} did not change by step {target}");
            }
        }
    }

    #[test]
    fn artistic_render_has_visible_contrast_for_all_presets() {
        for preset in ["spots", "waves", "labyrinth", "mitosis"] {
            let mut sim = ReactionDiffusionSim::new(96, 96, 2001);
            sim.reset_preset(preset);
            for _ in 0..180 {
                sim.step();
            }
            let mut pixels = vec![0; 96 * 96 * 4];
            sim.render_rgba(RenderStyle::Artistic, &mut pixels);
            let mut min_luma = u8::MAX;
            let mut max_luma = u8::MIN;
            for px in pixels.chunks_exact(4) {
                let luma = ((px[0] as u16 + px[1] as u16 + px[2] as u16) / 3) as u8;
                min_luma = min_luma.min(luma);
                max_luma = max_luma.max(luma);
            }
            assert!(
                max_luma.saturating_sub(min_luma) > 36,
                "{preset} render contrast too low: {min_luma}..{max_luma}"
            );
        }
    }
}
