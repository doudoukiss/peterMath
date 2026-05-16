use crate::metrics::Metrics;
use crate::palette;
use crate::simulation::{wrap_index, RenderStyle};

pub struct LifeRlePattern {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<(usize, usize)>,
}

impl LifeRlePattern {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut header_width = None;
        let mut header_height = None;
        let mut body = String::new();

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.contains('=') && line.contains(',') {
                for part in line.split(',') {
                    let mut pieces = part.split('=');
                    let key = pieces.next().map(str::trim).unwrap_or_default();
                    let value = pieces.next().map(str::trim).unwrap_or_default();
                    match key {
                        "x" => header_width = value.parse::<usize>().ok(),
                        "y" => header_height = value.parse::<usize>().ok(),
                        _ => {}
                    }
                }
            } else {
                body.push_str(line);
            }
        }

        if body.is_empty() {
            return Err(anyhow::anyhow!("RLE body is empty"));
        }

        let mut cells = Vec::new();
        let mut x = 0usize;
        let mut y = 0usize;
        let mut count = 0usize;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for ch in body.chars() {
            if ch.is_ascii_digit() {
                count = count * 10 + ch.to_digit(10).unwrap_or_default() as usize;
                continue;
            }

            let run = count.max(1);
            count = 0;
            match ch {
                'b' => {
                    x += run;
                }
                'o' => {
                    for dx in 0..run {
                        cells.push((x + dx, y));
                    }
                    max_x = max_x.max(x + run);
                    max_y = max_y.max(y + 1);
                    x += run;
                }
                '$' => {
                    y += run;
                    x = 0;
                    max_y = max_y.max(y);
                }
                '!' => break,
                c if c.is_whitespace() => {}
                other => return Err(anyhow::anyhow!("unsupported RLE token: {other}")),
            }
        }

        let width = header_width.unwrap_or(max_x).max(max_x);
        let height = header_height.unwrap_or(max_y).max(max_y);
        Ok(Self {
            width,
            height,
            cells,
        })
    }
}

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
        self.previous_age.fill(0.0);
        match preset {
            "symmetric_seed" => self.seed_symmetric(),
            _ => self.reset_random(),
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(false);
        self.next.fill(false);
        self.age.fill(0.0);
        self.previous_age.fill(0.0);
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

    pub fn get_cell(&self, x: usize, y: usize) -> bool {
        self.cells.get(y * self.w + x)
    }

    pub fn set_cell(&mut self, x: usize, y: usize, alive: bool) {
        if x >= self.w || y >= self.h {
            return;
        }
        let idx = y * self.w + x;
        self.cells.set(idx, alive);
        self.age[idx] = if alive { 1.0 } else { 0.0 };
    }

    pub fn apply_rle_centered(&mut self, pattern: &LifeRlePattern) {
        self.clear();
        let x0 = self.w.saturating_sub(pattern.width) / 2;
        let y0 = self.h.saturating_sub(pattern.height) / 2;
        for &(x, y) in &pattern.cells {
            let tx = x0 + x;
            let ty = y0 + y;
            if tx < self.w && ty < self.h {
                self.set_cell(tx, ty, true);
            }
        }
    }

    pub fn export_rle(&self) -> String {
        let Some((min_x, min_y, max_x, max_y)) = self.live_bounds() else {
            return "x = 0, y = 0, rule = B3/S23\n!\n".to_owned();
        };

        let width = max_x - min_x + 1;
        let height = max_y - min_y + 1;
        let mut body = String::new();
        for y in min_y..=max_y {
            let mut run_char = None;
            let mut run_count = 0usize;
            for x in min_x..=max_x {
                let ch = if self.get_cell(x, y) { 'o' } else { 'b' };
                if run_char == Some(ch) {
                    run_count += 1;
                } else {
                    append_rle_run(&mut body, run_char, run_count);
                    run_char = Some(ch);
                    run_count = 1;
                }
            }
            append_rle_run(&mut body, run_char, run_count);
            if y != max_y {
                body.push('$');
            }
        }
        body.push('!');

        format!("x = {width}, y = {height}, rule = B3/S23\n{body}\n")
    }

    fn live_bounds(&self) -> Option<(usize, usize, usize, usize)> {
        let mut min_x = self.w;
        let mut min_y = self.h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut found = false;
        for y in 0..self.h {
            for x in 0..self.w {
                if self.get_cell(x, y) {
                    found = true;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        found.then_some((min_x, min_y, max_x, max_y))
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

fn append_rle_run(out: &mut String, ch: Option<char>, count: usize) {
    let Some(ch) = ch else {
        return;
    };
    if count > 1 {
        out.push_str(&count.to_string());
    }
    out.push(ch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_rle_body() {
        let pattern = LifeRlePattern::parse("#N Glider\nx = 3, y = 3, rule = B3/S23\nbob$2bo$3o!")
            .expect("valid glider RLE");
        assert_eq!(pattern.width, 3);
        assert_eq!(pattern.height, 3);
        assert_eq!(pattern.cells.len(), 5);
    }

    #[test]
    fn exports_active_bounding_box_as_rle() {
        let mut sim = LifeSim::new(8, 8, 1);
        sim.clear();
        sim.set_cell(2, 2, true);
        sim.set_cell(3, 2, true);
        sim.set_cell(4, 2, true);
        let rle = sim.export_rle();
        assert!(rle.starts_with("x = 3, y = 1, rule = B3/S23"));
        assert!(rle.contains("3o!"));
    }
}
