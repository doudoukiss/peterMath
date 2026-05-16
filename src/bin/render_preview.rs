#![allow(dead_code)]

#[path = "../export.rs"]
mod export;
#[path = "../metrics.rs"]
mod metrics;
#[path = "../palette.rs"]
mod palette;
#[path = "../simulation/mod.rs"]
mod simulation;

use simulation::lenia::LeniaSim;
use simulation::life::LifeSim;
use simulation::reaction_diffusion::ReactionDiffusionSim;
use simulation::RenderStyle;
use std::fs;

const PREVIEW_SIZE: usize = 512;
const JUDGE_GAP: usize = 32;
const EXPLANATION_PANEL_WIDTH: usize = 300;

fn main() -> anyhow::Result<()> {
    fs::create_dir_all("peterMath_exports/previews")?;
    render_lenia()?;
    render_reaction_diffusion()?;
    render_judge_reference()?;
    render_show_mode_storyboard()?;
    render_long_show_storyboard()?;
    render_major_cases_gallery()?;
    render_reaction_comparison()?;
    render_life_teaching_reference()?;
    render_lenia_explanation_reference()?;
    println!("Wrote peterMath_exports/previews/lenia_hero.png");
    println!("Wrote peterMath_exports/previews/reaction_diffusion_texture.png");
    println!("Wrote peterMath_exports/previews/judge_mode_reference.png");
    println!("Wrote peterMath_exports/previews/lenia_showcase.png");
    println!("Wrote peterMath_exports/previews/reaction_diffusion_showcase.png");
    println!("Wrote peterMath_exports/previews/show_mode_storyboard.png");
    println!("Wrote peterMath_exports/previews/show_mode_long_storyboard.png");
    println!("Wrote peterMath_exports/previews/major_cases_gallery.png");
    println!("Wrote peterMath_exports/previews/reaction_diffusion_comparison.png");
    println!("Wrote peterMath_exports/previews/life_teaching_reference.png");
    println!("Wrote peterMath_exports/previews/lenia_explanation_reference.png");
    Ok(())
}

fn render_lenia() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(128, 128, 1001);
    for _ in 0..120 {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    let hero = upscale_rgba(&pixels, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    export::save_png(
        "peterMath_exports/previews/lenia_hero.png",
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &hero,
    )?;
    export::save_png(
        "peterMath_exports/previews/lenia_showcase.png",
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &hero,
    )
}

fn render_reaction_diffusion() -> anyhow::Result<()> {
    let mut sim = ReactionDiffusionSim::new(128, 128, 2001);
    sim.reset_preset("labyrinth");
    for _ in 0..520 {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    let texture = upscale_rgba(&pixels, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    export::save_png(
        "peterMath_exports/previews/reaction_diffusion_texture.png",
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &texture,
    )?;
    export::save_png(
        "peterMath_exports/previews/reaction_diffusion_showcase.png",
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &texture,
    )
}

fn render_judge_reference() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(128, 128, 1001);
    for _ in 0..90 {
        sim.step();
    }

    let (w, h) = sim.size();
    let gap = JUDGE_GAP;
    let mut raw = vec![0; w * h * 4];
    let mut art = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::RawMath, &mut raw);
    sim.render_rgba(RenderStyle::Artistic, &mut art);
    let raw = upscale_rgba(&raw, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    let art = upscale_rgba(&art, w, h, PREVIEW_SIZE, PREVIEW_SIZE);

    let explanation = render_explanation_panel(&sim, EXPLANATION_PANEL_WIDTH, PREVIEW_SIZE);
    let out_w = PREVIEW_SIZE * 2 + EXPLANATION_PANEL_WIDTH + gap * 2;
    let mut combined = vec![0; out_w * PREVIEW_SIZE * 4];
    blit_rgba(&raw, PREVIEW_SIZE, PREVIEW_SIZE, &mut combined, out_w, 0);
    blit_rgba(
        &art,
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &mut combined,
        out_w,
        PREVIEW_SIZE + gap,
    );
    blit_rgba(
        &explanation,
        EXPLANATION_PANEL_WIDTH,
        PREVIEW_SIZE,
        &mut combined,
        out_w,
        PREVIEW_SIZE * 2 + gap * 2,
    );
    export::save_png(
        "peterMath_exports/previews/judge_mode_reference.png",
        out_w,
        PREVIEW_SIZE,
        &combined,
    )
}

fn render_show_mode_storyboard() -> anyhow::Result<()> {
    let tile = 256;
    let gap = 18;
    let out_w = tile * 3 + gap * 4;
    let out_h = tile * 2 + gap * 3;
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);

    let tiles = [
        storyboard_lenia("orbital_field", 24),
        storyboard_life_structure(32),
        storyboard_reaction("labyrinth", 320),
        storyboard_reaction("mitosis", 240),
        storyboard_lenia("orbital_field", 36),
        storyboard_lenia("dense_bloom", 4),
    ];
    let accent = [
        [100, 232, 218, 255],
        [216, 240, 139, 255],
        [255, 157, 102, 255],
        [255, 118, 168, 255],
        [154, 185, 255, 255],
        [255, 219, 128, 255],
    ];

    for (i, image) in tiles.iter().enumerate() {
        let col = i % 3;
        let row = i / 3;
        let x = gap + col * (tile + gap);
        let y = gap + row * (tile + gap);
        fill_rect(
            &mut out,
            out_w,
            x - 3,
            y - 3,
            tile + 6,
            tile + 6,
            [16, 24, 28, 255],
        );
        fill_rect(&mut out, out_w, x - 3, y - 3, tile + 6, 8, accent[i]);
        blit_rgba_at(image, tile, tile, &mut out, out_w, x, y);
    }

    export::save_png(
        "peterMath_exports/previews/show_mode_storyboard.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_long_show_storyboard() -> anyhow::Result<()> {
    let tile = 160;
    let gap = 12;
    let cols = 5;
    let rows = 4;
    let out_w = tile * cols + gap * (cols + 1);
    let out_h = tile * rows + gap * (rows + 1);
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);

    let tiles = [
        storyboard_lenia_sized("orbital_field", 24, tile),
        storyboard_life_preset("structure_showcase", 8, tile),
        storyboard_reaction_sized("labyrinth", 220, tile),
        storyboard_lenia_sized("orbital_field", 26, tile),
        storyboard_lenia_sized("twin_organisms", 26, tile),
        storyboard_life_preset("still_lifes", 4, tile),
        storyboard_life_preset("oscillators", 8, tile),
        storyboard_life_preset("glider_lane", 24, tile),
        storyboard_reaction_sized("spots", 160, tile),
        storyboard_reaction_sized("labyrinth", 220, tile),
        storyboard_reaction_sized("waves", 180, tile),
        storyboard_reaction_sized("mitosis", 180, tile),
        storyboard_lenia_sized("orbital_field", 34, tile),
        storyboard_lenia_sized("twin_organisms", 32, tile),
        storyboard_lenia_sized("kernel_ring", 32, tile),
        storyboard_lenia_sized("dense_bloom", 4, tile),
        storyboard_lenia_sized("coral_drift", 28, tile),
    ];

    for (i, image) in tiles.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x = gap + col * (tile + gap);
        let y = gap + row * (tile + gap);
        fill_rect(
            &mut out,
            out_w,
            x - 2,
            y - 2,
            tile + 4,
            tile + 4,
            [16, 24, 28, 255],
        );
        let accent = if i < 5 {
            [100, 232, 218, 255]
        } else if i < 8 {
            [216, 240, 139, 255]
        } else if i < 12 {
            [255, 157, 102, 255]
        } else {
            [255, 118, 168, 255]
        };
        fill_rect(&mut out, out_w, x - 2, y - 2, tile + 4, 6, accent);
        blit_rgba_at(image, tile, tile, &mut out, out_w, x, y);
    }

    export::save_png(
        "peterMath_exports/previews/show_mode_long_storyboard.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_major_cases_gallery() -> anyhow::Result<()> {
    let tile = 192;
    let gap = 14;
    let cols = 4;
    let rows = 3;
    let out_w = tile * cols + gap * (cols + 1);
    let out_h = tile * rows + gap * (rows + 1);
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);
    let tiles = [
        storyboard_life_preset("still_lifes", 3, tile),
        storyboard_life_preset("oscillators", 8, tile),
        storyboard_life_preset("glider_lane", 28, tile),
        storyboard_reaction_sized("spots", 180, tile),
        storyboard_reaction_sized("labyrinth", 240, tile),
        storyboard_reaction_sized("waves", 200, tile),
        storyboard_reaction_sized("mitosis", 200, tile),
        storyboard_lenia_sized("orbital_field", 36, tile),
        storyboard_lenia_sized("twin_organisms", 34, tile),
        storyboard_lenia_sized("kernel_ring", 34, tile),
        storyboard_lenia_sized("dense_bloom", 4, tile),
        storyboard_lenia_sized("coral_drift", 30, tile),
    ];
    for (i, image) in tiles.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x = gap + col * (tile + gap);
        let y = gap + row * (tile + gap);
        fill_rect(
            &mut out,
            out_w,
            x - 2,
            y - 2,
            tile + 4,
            tile + 4,
            [16, 24, 28, 255],
        );
        blit_rgba_at(image, tile, tile, &mut out, out_w, x, y);
    }
    export::save_png(
        "peterMath_exports/previews/major_cases_gallery.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_reaction_comparison() -> anyhow::Result<()> {
    let tile = 224;
    let gap = 18;
    let out_w = tile * 4 + gap * 5;
    let out_h = tile + gap * 2;
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);
    for (i, preset) in ["spots", "labyrinth", "waves", "mitosis"]
        .iter()
        .enumerate()
    {
        let image = storyboard_reaction_sized(preset, 220, tile);
        let x = gap + i * (tile + gap);
        let y = gap;
        fill_rect(
            &mut out,
            out_w,
            x - 2,
            y - 2,
            tile + 4,
            tile + 4,
            [16, 24, 28, 255],
        );
        blit_rgba_at(&image, tile, tile, &mut out, out_w, x, y);
    }
    export::save_png(
        "peterMath_exports/previews/reaction_diffusion_comparison.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_life_teaching_reference() -> anyhow::Result<()> {
    let tile = 224;
    let gap = 18;
    let out_w = tile * 3 + gap * 4;
    let out_h = tile + gap * 2;
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);
    for (i, (preset, steps)) in [("still_lifes", 3), ("oscillators", 9), ("glider_lane", 36)]
        .iter()
        .enumerate()
    {
        let image = storyboard_life_preset(preset, *steps, tile);
        let x = gap + i * (tile + gap);
        let y = gap;
        fill_rect(
            &mut out,
            out_w,
            x - 2,
            y - 2,
            tile + 4,
            tile + 4,
            [16, 24, 28, 255],
        );
        blit_rgba_at(&image, tile, tile, &mut out, out_w, x, y);
    }
    export::save_png(
        "peterMath_exports/previews/life_teaching_reference.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_lenia_explanation_reference() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(128, 128, 1001);
    sim.reset_preset("kernel_ring");
    for _ in 0..56 {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    let art = upscale_rgba(&pixels, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    let explanation = render_explanation_panel(&sim, EXPLANATION_PANEL_WIDTH, PREVIEW_SIZE);
    let gap = JUDGE_GAP;
    let out_w = PREVIEW_SIZE + EXPLANATION_PANEL_WIDTH + gap;
    let mut out = vec![0; out_w * PREVIEW_SIZE * 4];
    blit_rgba(&art, PREVIEW_SIZE, PREVIEW_SIZE, &mut out, out_w, 0);
    blit_rgba(
        &explanation,
        EXPLANATION_PANEL_WIDTH,
        PREVIEW_SIZE,
        &mut out,
        out_w,
        PREVIEW_SIZE + gap,
    );
    export::save_png(
        "peterMath_exports/previews/lenia_explanation_reference.png",
        out_w,
        PREVIEW_SIZE,
        &out,
    )
}

fn storyboard_lenia(preset: &str, steps: usize) -> Vec<u8> {
    storyboard_lenia_sized(preset, steps, 256)
}

fn storyboard_lenia_sized(preset: &str, steps: usize, size: usize) -> Vec<u8> {
    let source_size = if preset == "dense_bloom" { 128 } else { 64 };
    let mut sim = LeniaSim::new(source_size, source_size, 1001);
    sim.reset_preset(preset);
    for _ in 0..steps {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    upscale_rgba(&pixels, w, h, size, size)
}

fn storyboard_life_structure(steps: usize) -> Vec<u8> {
    storyboard_life_preset("structure_showcase", steps, 256)
}

fn storyboard_life_preset(preset: &str, steps: usize, size: usize) -> Vec<u8> {
    let mut sim = LifeSim::new(96, 96, 3001);
    sim.reset_preset(preset);
    for _ in 0..steps {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    nearest_upscale_rgba(&pixels, w, h, size, size)
}

fn storyboard_reaction(preset: &str, steps: usize) -> Vec<u8> {
    storyboard_reaction_sized(preset, steps, 256)
}

fn storyboard_reaction_sized(preset: &str, steps: usize, size: usize) -> Vec<u8> {
    let mut sim = ReactionDiffusionSim::new(128, 128, 2001);
    sim.reset_preset(preset);
    for _ in 0..steps {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    upscale_rgba(&pixels, w, h, size, size)
}

fn render_explanation_panel(sim: &LeniaSim, panel_w: usize, panel_h: usize) -> Vec<u8> {
    let mut out = vec![0; panel_w * panel_h * 4];
    fill_rect(&mut out, panel_w, 0, 0, panel_w, panel_h, [8, 12, 14, 255]);
    fill_rect(
        &mut out,
        panel_w,
        18,
        24,
        panel_w - 36,
        98,
        [13, 20, 23, 255],
    );
    fill_rect(
        &mut out,
        panel_w,
        18,
        144,
        panel_w - 36,
        162,
        [13, 20, 23, 255],
    );
    fill_rect(
        &mut out,
        panel_w,
        18,
        334,
        panel_w - 36,
        142,
        [13, 20, 23, 255],
    );

    let profile = sim.kernel_profile(64);
    draw_profile(
        &mut out,
        panel_w,
        (36, 44, panel_w - 72, 58),
        &profile,
        [100, 232, 218, 255],
    );

    let metrics = sim.metrics();
    let bars = [
        (metrics.mass, [100, 232, 218, 255]),
        (metrics.entropy, [255, 157, 102, 255]),
        (metrics.stability, [154, 185, 255, 255]),
        (metrics.vitality, [255, 111, 167, 255]),
    ];
    for (row, (value, color)) in bars.iter().enumerate() {
        let y = 166 + row * 32;
        fill_rect(
            &mut out,
            panel_w,
            36,
            y,
            panel_w - 72,
            10,
            [32, 44, 48, 255],
        );
        fill_rect(
            &mut out,
            panel_w,
            36,
            y,
            ((panel_w - 72) as f32 * value.clamp(0.0, 1.0)) as usize,
            10,
            *color,
        );
    }

    let (w, h) = sim.size();
    let inspection = sim.inspect_point(w / 2, h / 2);
    let signals = [
        inspection.value,
        inspection.delta.abs().min(1.0),
        inspection.gradient.min(1.0),
        inspection.convolution.clamp(0.0, 1.0),
        ((inspection.growth + 1.0) * 0.5).clamp(0.0, 1.0),
        inspection.estimated_next,
    ];
    for (i, value) in signals.iter().enumerate() {
        let x = 36 + i * 38;
        let height = (96.0 * value.clamp(0.0, 1.0)) as usize;
        fill_rect(
            &mut out,
            panel_w,
            x,
            436 - height,
            22,
            height,
            [255, 118, 168, 255],
        );
    }
    draw_circle(
        &mut out,
        panel_w,
        panel_w / 2,
        388,
        28,
        [100, 232, 218, 255],
    );
    draw_circle(&mut out, panel_w, panel_w / 2, 388, 4, [255, 118, 168, 255]);

    out
}

fn upscale_rgba(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<u8> {
    let mut out = vec![0; target_w * target_h * 4];
    for y in 0..target_h {
        let gy = if target_h > 1 {
            y as f32 * (source_h - 1) as f32 / (target_h - 1) as f32
        } else {
            0.0
        };
        let y0 = gy.floor() as usize;
        let y1 = (y0 + 1).min(source_h - 1);
        let ty = gy - y0 as f32;
        for x in 0..target_w {
            let gx = if target_w > 1 {
                x as f32 * (source_w - 1) as f32 / (target_w - 1) as f32
            } else {
                0.0
            };
            let x0 = gx.floor() as usize;
            let x1 = (x0 + 1).min(source_w - 1);
            let tx = gx - x0 as f32;
            let target_i = (y * target_w + x) * 4;
            for channel in 0..4 {
                let c00 = source[(y0 * source_w + x0) * 4 + channel] as f32;
                let c10 = source[(y0 * source_w + x1) * 4 + channel] as f32;
                let c01 = source[(y1 * source_w + x0) * 4 + channel] as f32;
                let c11 = source[(y1 * source_w + x1) * 4 + channel] as f32;
                let top = c00 + (c10 - c00) * tx;
                let bottom = c01 + (c11 - c01) * tx;
                out[target_i + channel] = (top + (bottom - top) * ty).round() as u8;
            }
        }
    }
    out
}

fn nearest_upscale_rgba(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<u8> {
    let mut out = vec![0; target_w * target_h * 4];
    for y in 0..target_h {
        let sy = (y * source_h / target_h).min(source_h - 1);
        for x in 0..target_w {
            let sx = (x * source_w / target_w).min(source_w - 1);
            let target_i = (y * target_w + x) * 4;
            let source_i = (sy * source_w + sx) * 4;
            out[target_i..target_i + 4].copy_from_slice(&source[source_i..source_i + 4]);
        }
    }
    out
}

fn blit_rgba(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    target: &mut [u8],
    target_w: usize,
    x_offset: usize,
) {
    for y in 0..source_h {
        let source_start = y * source_w * 4;
        let target_start = (y * target_w + x_offset) * 4;
        target[target_start..target_start + source_w * 4]
            .copy_from_slice(&source[source_start..source_start + source_w * 4]);
    }
}

fn blit_rgba_at(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    target: &mut [u8],
    target_w: usize,
    x_offset: usize,
    y_offset: usize,
) {
    for y in 0..source_h {
        let source_start = y * source_w * 4;
        let target_start = ((y + y_offset) * target_w + x_offset) * 4;
        target[target_start..target_start + source_w * 4]
            .copy_from_slice(&source[source_start..source_start + source_w * 4]);
    }
}

fn fill_rect(
    target: &mut [u8],
    target_w: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: [u8; 4],
) {
    let target_h = target.len() / target_w / 4;
    let x_end = (x + w).min(target_w);
    let y_end = (y + h).min(target_h);
    for yy in y..y_end {
        for xx in x..x_end {
            put_pixel(target, target_w, xx as i32, yy as i32, color);
        }
    }
}

fn draw_profile(
    target: &mut [u8],
    target_w: usize,
    rect: (usize, usize, usize, usize),
    values: &[f32],
    color: [u8; 4],
) {
    if values.len() < 2 {
        return;
    }
    let (x, y, w, h) = rect;
    let mut last = None;
    for (i, value) in values.iter().enumerate() {
        let tx = i as f32 / (values.len() - 1) as f32;
        let px = x as f32 + tx * w as f32;
        let py = y as f32 + (1.0 - value.clamp(0.0, 1.0)) * h as f32;
        let point = (px.round() as i32, py.round() as i32);
        if let Some(previous) = last {
            draw_line(target, target_w, previous, point, color);
        }
        last = Some(point);
    }
}

fn draw_circle(
    target: &mut [u8],
    target_w: usize,
    cx: usize,
    cy: usize,
    radius: usize,
    color: [u8; 4],
) {
    if radius <= 5 {
        for y in cy.saturating_sub(radius)..=cy + radius {
            for x in cx.saturating_sub(radius)..=cx + radius {
                let dx = x as isize - cx as isize;
                let dy = y as isize - cy as isize;
                if dx * dx + dy * dy <= (radius * radius) as isize {
                    put_pixel(target, target_w, x as i32, y as i32, color);
                }
            }
        }
        return;
    }

    let steps = 160;
    let mut previous = None;
    for i in 0..=steps {
        let a = i as f32 / steps as f32 * std::f32::consts::TAU;
        let point = (
            cx as i32 + (radius as f32 * a.cos()).round() as i32,
            cy as i32 + (radius as f32 * a.sin()).round() as i32,
        );
        if let Some(previous) = previous {
            draw_line(target, target_w, previous, point, color);
        }
        previous = Some(point);
    }
}

fn draw_line(target: &mut [u8], target_w: usize, from: (i32, i32), to: (i32, i32), color: [u8; 4]) {
    let (mut x0, mut y0) = from;
    let (x1, y1) = to;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        put_pixel(target, target_w, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn put_pixel(target: &mut [u8], target_w: usize, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as usize;
    let y = y as usize;
    let target_h = target.len() / target_w / 4;
    if x >= target_w || y >= target_h {
        return;
    }
    let i = (y * target_w + x) * 4;
    target[i..i + 4].copy_from_slice(&color);
}
