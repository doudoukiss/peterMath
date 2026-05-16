#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    pub mass: f32,
    pub entropy: f32,
    pub symmetry: f32,
    pub stability: f32,
    pub vitality: f32,
    pub active: usize,
}

impl Metrics {
    pub fn from_scalar_grid(grid: &[f32], previous: Option<&[f32]>, w: usize, h: usize) -> Self {
        if grid.is_empty() {
            return Self::default();
        }

        let n = grid.len() as f32;
        let mut sum = 0.0;
        let mut active = 0;
        let mut bins = [0usize; 16];
        for &v in grid {
            let x = v.clamp(0.0, 1.0);
            sum += x;
            if x > 0.08 {
                active += 1;
            }
            let bin = ((x * 15.999) as usize).min(15);
            bins[bin] += 1;
        }

        let mass = (sum / n).clamp(0.0, 1.0);
        let mut entropy = 0.0;
        for &count in &bins {
            if count > 0 {
                let p = count as f32 / n;
                entropy -= p * p.log2();
            }
        }
        entropy = (entropy / 4.0).clamp(0.0, 1.0);

        let mut mirror_error = 0.0;
        for y in 0..h {
            for x in 0..w {
                let a = grid[y * w + x].clamp(0.0, 1.0);
                let b = grid[y * w + (w - 1 - x)].clamp(0.0, 1.0);
                mirror_error += (a - b).abs();
            }
        }
        let symmetry = (1.0 - mirror_error / n).clamp(0.0, 1.0);

        let stability = if let Some(prev) = previous {
            if prev.len() == grid.len() {
                let mut diff = 0.0;
                for (a, b) in grid.iter().zip(prev.iter()) {
                    diff += (a - b).abs();
                }
                (1.0 - diff / n).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let vitality = (0.30 * entropy + 0.25 * symmetry + 0.25 * mass + 0.20 * (1.0 - stability))
            .clamp(0.0, 1.0);

        Self {
            mass,
            entropy,
            symmetry,
            stability,
            vitality,
            active,
        }
    }
}
