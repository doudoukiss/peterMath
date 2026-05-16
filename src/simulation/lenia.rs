use crate::metrics::Metrics;
use crate::palette;
use crate::simulation::{wrap_index, RenderStyle};

#[derive(Clone)]
pub struct LeniaState {
    pub w: usize,
    pub h: usize,
    pub field: Vec<f32>,
    pub previous: Vec<f32>,
    pub radius: usize,
    pub growth_center: f32,
    pub growth_width: f32,
    pub dt: f32,
    pub decay: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LeniaInspection {
    pub x: usize,
    pub y: usize,
    pub value: f32,
    pub previous: f32,
    pub delta: f32,
    pub gradient: f32,
    pub convolution: f32,
    pub growth: f32,
    pub estimated_next: f32,
}

pub struct LeniaSim {
    w: usize,
    h: usize,
    field: Vec<f32>,
    next: Vec<f32>,
    previous: Vec<f32>,
    kernel: Vec<(isize, isize, f32)>,
    pub radius: usize,
    pub growth_center: f32,
    pub growth_width: f32,
    pub dt: f32,
    pub decay: f32,
    pub seed: u64,
}

impl LeniaSim {
    pub fn new(w: usize, h: usize, seed: u64) -> Self {
        let mut sim = Self {
            w,
            h,
            field: vec![0.0; w * h],
            next: vec![0.0; w * h],
            previous: vec![0.0; w * h],
            kernel: Vec::new(),
            radius: 9,
            growth_center: 0.31,
            growth_width: 0.052,
            dt: 0.060,
            decay: 0.003,
            seed,
        };
        sim.rebuild_kernel();
        sim.reset_preset("orbital_field");
        sim
    }

    pub fn from_state(state: &LeniaState) -> Self {
        let mut sim = Self::new(state.w, state.h, state.seed);
        sim.restore(state);
        sim
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub fn field(&self) -> &[f32] {
        &self.field
    }

    pub fn previous_field(&self) -> &[f32] {
        &self.previous
    }

    pub fn kernel_entries(&self) -> &[(isize, isize, f32)] {
        &self.kernel
    }

    pub fn kernel_profile(&self, samples: usize) -> Vec<f32> {
        let samples = samples.max(2);
        let mut profile = Vec::with_capacity(samples);
        let mut max_weight = 0.0_f32;
        for i in 0..samples {
            let d = i as f32 / (samples - 1) as f32;
            let ring = (-(d - 0.55).powi(2) / 0.10).exp();
            let hollow = 1.0 - (-(d).powi(2) / 0.05).exp() * 0.35;
            let weight = ring * hollow;
            max_weight = max_weight.max(weight);
            profile.push(weight);
        }
        if max_weight > 0.0 {
            for value in &mut profile {
                *value /= max_weight;
            }
        }
        profile
    }

    pub fn snapshot(&self) -> LeniaState {
        LeniaState {
            w: self.w,
            h: self.h,
            field: self.field.clone(),
            previous: self.previous.clone(),
            radius: self.radius,
            growth_center: self.growth_center,
            growth_width: self.growth_width,
            dt: self.dt,
            decay: self.decay,
            seed: self.seed,
        }
    }

    pub fn restore(&mut self, state: &LeniaState) {
        self.w = state.w;
        self.h = state.h;
        self.field = state.field.clone();
        self.previous = state.previous.clone();
        self.next = vec![0.0; self.w * self.h];
        self.radius = state.radius;
        self.growth_center = state.growth_center;
        self.growth_width = state.growth_width;
        self.dt = state.dt;
        self.decay = state.decay;
        self.seed = state.seed;
        self.rebuild_kernel();
    }

    pub fn set_radius(&mut self, radius: usize) {
        self.radius = radius.max(3);
        self.rebuild_kernel();
    }

    fn rebuild_kernel(&mut self) {
        self.kernel.clear();
        let r = self.radius as isize;
        let rf = self.radius as f32;
        let mut total = 0.0;
        for dy in -r..=r {
            for dx in -r..=r {
                let d = ((dx * dx + dy * dy) as f32).sqrt() / rf;
                if d <= 1.0 {
                    let ring = (-(d - 0.55).powi(2) / 0.10).exp();
                    let hollow = 1.0 - (-(d).powi(2) / 0.05).exp() * 0.35;
                    let weight = ring * hollow;
                    self.kernel.push((dx, dy, weight));
                    total += weight;
                }
            }
        }
        if total > 0.0 {
            for item in &mut self.kernel {
                item.2 /= total;
            }
        }
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        self.w = w.max(32);
        self.h = h.max(32);
        self.field = vec![0.0; self.w * self.h];
        self.previous = vec![0.0; self.w * self.h];
        self.next = vec![0.0; self.w * self.h];
        self.rebuild_kernel();
    }

    pub fn reset_preset(&mut self, preset: &str) {
        self.field.fill(0.0);
        self.previous.fill(0.0);
        let mut rng = fastrand::Rng::with_seed(self.seed);
        match preset {
            "orbital_field" => {
                self.configure_rule(9, 0.31, 0.052, 0.060, 0.003);
                for i in 0..28 {
                    let t = i as f32 / 28.0;
                    let a = i as f32 * 2.399_963 + 0.21 * (i as f32 * 1.7).sin();
                    let r = 0.035 + 0.31 * t.sqrt();
                    let wobble = 1.0 + 0.10 * (i as f32 * 0.83).cos();
                    let cx = self.w as f32 * (0.50 + r * wobble * a.cos());
                    let cy = self.h as f32 * (0.50 + r * a.sin());
                    let sigma = 3.6 + (i % 5) as f32 * 1.15;
                    let amplitude = 0.24 + 0.18 * (1.0 - t) + 0.10 * ((i * 7 % 11) as f32 / 10.0);
                    self.add_blob(cx, cy, sigma, amplitude);
                }
            }
            "twin_organisms" => {
                self.configure_rule(10, 0.30, 0.046, 0.055, 0.0025);
                for side in [-1.0, 1.0] {
                    for i in 0..18 {
                        let t = i as f32 / 18.0;
                        let a = i as f32 * 2.399_963 + side * 0.45;
                        let r = 0.035 + 0.18 * t.sqrt();
                        let cx = self.w as f32 * (0.50 + side * 0.19 + r * a.cos());
                        let cy = self.h as f32 * (0.50 + r * a.sin());
                        self.add_blob(cx, cy, 3.4 + (i % 4) as f32, 0.24 + 0.18 * (1.0 - t));
                    }
                }
            }
            "coral_drift" => {
                self.configure_rule(7, 0.25, 0.065, 0.050, 0.002);
                for i in 0..52 {
                    let t = i as f32 / 52.0;
                    let a = 0.55 + t * std::f32::consts::TAU * 2.3;
                    let r = 0.07 + 0.33 * t;
                    let cx = self.w as f32 * (0.50 + r * a.cos() * 0.95);
                    let cy = self.h as f32 * (0.50 + r * a.sin() * 0.55);
                    self.add_blob(cx, cy, 2.7 + (i % 6) as f32 * 0.55, 0.22 + 0.16 * rng.f32());
                }
            }
            "kernel_ring" => {
                self.configure_rule(14, 0.34, 0.040, 0.047, 0.0025);
                for i in 0..44 {
                    let a = i as f32 / 44.0 * std::f32::consts::TAU;
                    let r = if i % 2 == 0 { 0.22 } else { 0.31 };
                    let cx = self.w as f32 * (0.50 + r * a.cos());
                    let cy = self.h as f32 * (0.50 + r * a.sin());
                    self.add_blob(cx, cy, 3.2 + (i % 3) as f32, 0.22 + 0.12 * rng.f32());
                }
            }
            "sparse_soup" => {
                self.configure_rule(8, 0.32, 0.055, 0.062, 0.003);
                self.randomize_density_inner(&mut rng, 0.18);
            }
            "dense_bloom" => {
                self.configure_rule(11, 0.28, 0.070, 0.040, 0.004);
                self.randomize_density_inner(&mut rng, 0.52);
            }
            _ => {
                self.configure_rule(9, 0.31, 0.052, 0.060, 0.003);
                for _ in 0..18 {
                    let cx = rng.f32() * self.w as f32;
                    let cy = rng.f32() * self.h as f32;
                    self.add_blob(cx, cy, 6.0 + rng.f32() * 14.0, 0.35 + rng.f32() * 0.40);
                }
            }
        }
    }

    fn configure_rule(&mut self, radius: usize, center: f32, width: f32, dt: f32, decay: f32) {
        self.radius = radius;
        self.growth_center = center;
        self.growth_width = width;
        self.dt = dt;
        self.decay = decay;
        self.rebuild_kernel();
    }

    pub fn clear(&mut self) {
        self.field.fill(0.0);
        self.previous.fill(0.0);
        self.next.fill(0.0);
    }

    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn randomize_density(&mut self, seed: u64, density: f32) {
        self.seed = seed;
        self.field.fill(0.0);
        self.previous.fill(0.0);
        self.next.fill(0.0);
        let mut rng = fastrand::Rng::with_seed(seed);
        self.randomize_density_inner(&mut rng, density);
    }

    fn randomize_density_inner(&mut self, rng: &mut fastrand::Rng, density: f32) {
        let density = density.clamp(0.02, 0.85);
        let count = (12.0 + density * 120.0) as usize;
        for _ in 0..count {
            let cx = rng.f32() * self.w as f32;
            let cy = rng.f32() * self.h as f32;
            let sigma = 2.4 + rng.f32() * (4.0 + density * 18.0);
            let amplitude = 0.12 + rng.f32() * (0.25 + density * 0.42);
            self.add_blob(cx, cy, sigma, amplitude);
        }
    }

    pub fn paint_brush(&mut self, x: f32, y: f32, radius: f32, strength: f32) {
        self.apply_brush(x, y, radius, strength.clamp(0.0, 1.0), 1.0);
    }

    pub fn erase_brush(&mut self, x: f32, y: f32, radius: f32, strength: f32) {
        self.apply_brush(x, y, radius, strength.clamp(0.0, 1.0), -1.0);
    }

    pub fn apply_stamp(&mut self, x: f32, y: f32, stamp_id: &str, radius: f32, strength: f32) {
        self.previous.copy_from_slice(&self.field);
        let radius = radius.max(2.0);
        let strength = strength.clamp(0.05, 1.0);
        match stamp_id {
            "ring_seed" => {
                for i in 0..18 {
                    let a = i as f32 / 18.0 * std::f32::consts::TAU;
                    let cx = x + radius * 0.72 * a.cos();
                    let cy = y + radius * 0.72 * a.sin();
                    self.add_blob(cx, cy, radius * 0.16, 0.32 * strength);
                }
            }
            "twin_seed" => {
                self.add_blob(x - radius * 0.36, y, radius * 0.28, 0.62 * strength);
                self.add_blob(x + radius * 0.36, y, radius * 0.28, 0.62 * strength);
            }
            "arc_seed" => {
                for i in 0..13 {
                    let a = -1.15 + i as f32 / 12.0 * 2.30;
                    let cx = x + radius * 0.76 * a.cos();
                    let cy = y + radius * 0.76 * a.sin();
                    self.add_blob(cx, cy, radius * 0.14, 0.38 * strength);
                }
            }
            "noise_patch" => {
                let salt = ((x.round() as u64) << 32) ^ y.round() as u64 ^ self.seed;
                let mut rng = fastrand::Rng::with_seed(salt);
                for _ in 0..20 {
                    let a = rng.f32() * std::f32::consts::TAU;
                    let d = radius * rng.f32().sqrt();
                    let cx = x + d * a.cos();
                    let cy = y + d * a.sin();
                    self.add_blob(
                        cx,
                        cy,
                        radius * (0.08 + rng.f32() * 0.16),
                        0.20 + 0.36 * strength * rng.f32(),
                    );
                }
            }
            _ => {
                self.add_blob(x, y, radius * 0.34, 0.75 * strength);
            }
        }
    }

    fn apply_brush(&mut self, x: f32, y: f32, radius: f32, strength: f32, direction: f32) {
        if radius <= 0.0 || strength <= 0.0 {
            return;
        }

        self.previous.copy_from_slice(&self.field);
        let radius = radius.max(0.5);
        let sigma2 = 2.0 * (radius * 0.45).max(0.5).powi(2);
        let extent = radius.ceil() as isize;
        let cx = x.round() as isize;
        let cy = y.round() as isize;

        for dy in -extent..=extent {
            for dx in -extent..=extent {
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > radius * radius {
                    continue;
                }
                let idx = wrap_index(cx + dx, cy + dy, self.w, self.h);
                let falloff = (-d2 / sigma2).exp();
                let amount = (strength * falloff).clamp(0.0, 1.0);
                self.field[idx] = if direction > 0.0 {
                    self.field[idx] + amount * (1.0 - self.field[idx])
                } else {
                    self.field[idx] * (1.0 - amount)
                }
                .clamp(0.0, 1.0);
            }
        }
    }

    fn add_blob(&mut self, cx: f32, cy: f32, sigma: f32, amplitude: f32) {
        let r = (sigma * 3.0) as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let x = cx as isize + dx;
                let y = cy as isize + dy;
                let idx = wrap_index(x, y, self.w, self.h);
                let d2 = (dx * dx + dy * dy) as f32;
                let v = amplitude * (-d2 / (2.0 * sigma * sigma)).exp();
                self.field[idx] = (self.field[idx] + v).clamp(0.0, 1.0);
            }
        }
    }

    pub fn step(&mut self) {
        self.previous.copy_from_slice(&self.field);
        for y in 0..self.h {
            for x in 0..self.w {
                let neighborhood = self.convolution_at(x, y);
                let growth = self.growth_response(neighborhood);
                let idx = y * self.w + x;
                let value = self.field[idx] + self.dt * growth - self.decay * self.field[idx];
                self.next[idx] = value.clamp(0.0, 1.0);
            }
        }
        std::mem::swap(&mut self.field, &mut self.next);
    }

    pub fn convolution_at(&self, x: usize, y: usize) -> f32 {
        let x = x.min(self.w.saturating_sub(1));
        let y = y.min(self.h.saturating_sub(1));
        let mut neighborhood = 0.0;
        for &(dx, dy, weight) in &self.kernel {
            let idx = wrap_index(x as isize + dx, y as isize + dy, self.w, self.h);
            neighborhood += self.field[idx] * weight;
        }
        neighborhood
    }

    pub fn growth_response(&self, x: f32) -> f32 {
        let sigma2 = 2.0 * self.growth_width * self.growth_width;
        2.0 * (-(x - self.growth_center).powi(2) / sigma2).exp() - 1.0
    }

    pub fn inspect_point(&self, x: usize, y: usize) -> LeniaInspection {
        let x = x.min(self.w.saturating_sub(1));
        let y = y.min(self.h.saturating_sub(1));
        let idx = y * self.w + x;
        let value = self.field[idx];
        let previous = self.previous[idx];
        let gx = self.field[wrap_index(x as isize + 1, y as isize, self.w, self.h)]
            - self.field[wrap_index(x as isize - 1, y as isize, self.w, self.h)];
        let gy = self.field[wrap_index(x as isize, y as isize + 1, self.w, self.h)]
            - self.field[wrap_index(x as isize, y as isize - 1, self.w, self.h)];
        let gradient = (gx * gx + gy * gy).sqrt();
        let convolution = self.convolution_at(x, y);
        let growth = self.growth_response(convolution);
        let estimated_next = (value + self.dt * growth - self.decay * value).clamp(0.0, 1.0);

        LeniaInspection {
            x,
            y,
            value,
            previous,
            delta: value - previous,
            gradient,
            convolution,
            growth,
            estimated_next,
        }
    }

    pub fn render_rgba(&self, style: RenderStyle, out: &mut [u8]) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let v = self.field[i];
                let rgba = match style {
                    RenderStyle::RawMath => palette::raw_gray(v),
                    RenderStyle::Artistic => {
                        let gx = self.field[wrap_index(x as isize + 1, y as isize, self.w, self.h)]
                            - self.field[wrap_index(x as isize - 1, y as isize, self.w, self.h)];
                        let gy = self.field[wrap_index(x as isize, y as isize + 1, self.w, self.h)]
                            - self.field[wrap_index(x as isize, y as isize - 1, self.w, self.h)];
                        let edge = (gx * gx + gy * gy).sqrt() * 3.0;
                        palette::life_field_delta(
                            (v * 1.30).clamp(0.0, 1.0),
                            edge,
                            v,
                            v - self.previous[i],
                        )
                    }
                };
                out[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
    }

    pub fn metrics(&self) -> Metrics {
        Metrics::from_scalar_grid(&self.field, Some(&self.previous), self.w, self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_variant_produces_metric_delta_from_same_baseline() {
        let mut source = LeniaSim::new(96, 96, 2001);
        for _ in 0..8 {
            source.step();
        }

        let snapshot = source.snapshot();
        let mut baseline = LeniaSim::from_state(&snapshot);
        let mut variant = LeniaSim::from_state(&snapshot);
        variant.growth_center = (variant.growth_center + 0.055).clamp(0.05, 0.95);

        for _ in 0..32 {
            baseline.step();
            variant.step();
        }

        let base = baseline.metrics();
        let changed = variant.metrics();
        let delta = (changed.mass - base.mass).abs()
            + (changed.entropy - base.entropy).abs()
            + (changed.stability - base.stability).abs()
            + (changed.vitality - base.vitality).abs();

        assert!(
            delta > 0.001,
            "expected variant metrics to diverge, got delta {delta}"
        );
    }
}
