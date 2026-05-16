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
use simulation::RenderStyle;
use std::fs;

const PREVIEW_SIZE: usize = 512;
const GAP: usize = 24;
const PANEL_W: usize = 320;

fn main() -> anyhow::Result<()> {
    fs::create_dir_all("peterMath_exports/previews")?;
    render_lenia_hero()?;
    render_lenia_showcase()?;
    render_judge_reference()?;
    render_show_mode_storyboard()?;
    render_major_cases_gallery()?;
    render_lenia_explanation_reference()?;
    println!("Wrote peterMath_exports/previews/lenia_hero.png");
    println!("Wrote peterMath_exports/previews/lenia_showcase.png");
    println!("Wrote peterMath_exports/previews/judge_mode_reference.png");
    println!("Wrote peterMath_exports/previews/show_mode_storyboard.png");
    println!("Wrote peterMath_exports/previews/major_cases_gallery.png");
    println!("Wrote peterMath_exports/previews/lenia_explanation_reference.png");
    Ok(())
}

fn render_lenia_hero() -> anyhow::Result<()> {
    let hero = render_preset_tile("orbital_field", 56, PREVIEW_SIZE);
    export::save_png(
        "peterMath_exports/previews/lenia_hero.png",
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &hero,
    )
}

fn render_lenia_showcase() -> anyhow::Result<()> {
    let tile = 256;
    let presets = [
        ("orbital_field", 34, [100, 232, 218, 255]),
        ("twin_organisms", 34, [154, 185, 255, 255]),
        ("kernel_ring", 30, [255, 219, 128, 255]),
        ("sparse_soup", 38, [216, 240, 139, 255]),
        ("dense_bloom", 16, [255, 118, 168, 255]),
        ("coral_drift", 36, [255, 177, 116, 255]),
    ];
    let out_w = tile * 3 + GAP * 4;
    let out_h = tile * 2 + GAP * 3;
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);
    for (i, (preset, steps, accent)) in presets.iter().enumerate() {
        let col = i % 3;
        let row = i / 3;
        let x = GAP + col * (tile + GAP);
        let y = GAP + row * (tile + GAP);
        fill_rect(
            &mut out,
            out_w,
            x - 3,
            y - 3,
            tile + 6,
            tile + 6,
            [16, 24, 28, 255],
        );
        fill_rect(&mut out, out_w, x - 3, y - 3, tile + 6, 8, *accent);
        let image = render_preset_tile(preset, *steps, tile);
        blit_rgba_at(&image, tile, tile, &mut out, out_w, x, y);
    }
    export::save_png(
        "peterMath_exports/previews/lenia_showcase.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_judge_reference() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(96, 96, 1001);
    for _ in 0..44 {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut raw = vec![0; w * h * 4];
    let mut art = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::RawMath, &mut raw);
    sim.render_rgba(RenderStyle::Artistic, &mut art);
    let raw = upscale_rgba(&raw, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    let art = upscale_rgba(&art, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    let panel = render_explanation_panel(&sim, PANEL_W, PREVIEW_SIZE);
    let out_w = PREVIEW_SIZE * 2 + PANEL_W + GAP * 2;
    let mut out = vec![0; out_w * PREVIEW_SIZE * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, PREVIEW_SIZE, [5, 8, 10, 255]);
    blit_rgba_at(&raw, PREVIEW_SIZE, PREVIEW_SIZE, &mut out, out_w, 0, 0);
    blit_rgba_at(
        &art,
        PREVIEW_SIZE,
        PREVIEW_SIZE,
        &mut out,
        out_w,
        PREVIEW_SIZE + GAP,
        0,
    );
    blit_rgba_at(
        &panel,
        PANEL_W,
        PREVIEW_SIZE,
        &mut out,
        out_w,
        PREVIEW_SIZE * 2 + GAP * 2,
        0,
    );
    export::save_png(
        "peterMath_exports/previews/judge_mode_reference.png",
        out_w,
        PREVIEW_SIZE,
        &out,
    )
}

fn render_show_mode_storyboard() -> anyhow::Result<()> {
    let tile = 180;
    let presets = [
        ("orbital_field", 10, [100, 232, 218, 255]),
        ("orbital_field", 28, [100, 232, 218, 255]),
        ("twin_organisms", 28, [154, 185, 255, 255]),
        ("kernel_ring", 28, [255, 219, 128, 255]),
        ("dense_bloom", 14, [255, 118, 168, 255]),
        ("coral_drift", 32, [255, 177, 116, 255]),
        ("coral_drift", 42, [216, 240, 139, 255]),
    ];
    let cols = 4;
    let rows = 2;
    let out_w = tile * cols + GAP * (cols + 1);
    let out_h = tile * rows + GAP * (rows + 1);
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);
    for (i, (preset, steps, accent)) in presets.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x = GAP + col * (tile + GAP);
        let y = GAP + row * (tile + GAP);
        fill_rect(
            &mut out,
            out_w,
            x - 3,
            y - 3,
            tile + 6,
            tile + 6,
            [16, 24, 28, 255],
        );
        fill_rect(&mut out, out_w, x - 3, y - 3, tile + 6, 7, *accent);
        let image = render_preset_tile(preset, *steps, tile);
        blit_rgba_at(&image, tile, tile, &mut out, out_w, x, y);
    }
    export::save_png(
        "peterMath_exports/previews/show_mode_storyboard.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_major_cases_gallery() -> anyhow::Result<()> {
    let tile = 220;
    let presets = [
        ("orbital_field", 34, [100, 232, 218, 255]),
        ("twin_organisms", 34, [154, 185, 255, 255]),
        ("kernel_ring", 30, [255, 219, 128, 255]),
        ("sparse_soup", 38, [216, 240, 139, 255]),
        ("dense_bloom", 16, [255, 118, 168, 255]),
        ("coral_drift", 36, [255, 177, 116, 255]),
    ];
    let cols = 3;
    let rows = 2;
    let out_w = tile * cols + GAP * (cols + 1);
    let out_h = tile * rows + GAP * (rows + 1);
    let mut out = vec![0; out_w * out_h * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, out_h, [5, 8, 10, 255]);
    for (i, (preset, steps, accent)) in presets.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x = GAP + col * (tile + GAP);
        let y = GAP + row * (tile + GAP);
        fill_rect(
            &mut out,
            out_w,
            x - 3,
            y - 3,
            tile + 6,
            tile + 6,
            [16, 24, 28, 255],
        );
        fill_rect(&mut out, out_w, x - 3, y - 3, tile + 6, 8, *accent);
        let image = render_preset_tile(preset, *steps, tile);
        blit_rgba_at(&image, tile, tile, &mut out, out_w, x, y);
    }
    export::save_png(
        "peterMath_exports/previews/major_cases_gallery.png",
        out_w,
        out_h,
        &out,
    )
}

fn render_lenia_explanation_reference() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(96, 96, 1001);
    for _ in 0..52 {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut art = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut art);
    let art = upscale_rgba(&art, w, h, PREVIEW_SIZE, PREVIEW_SIZE);
    let panel = render_explanation_panel(&sim, PANEL_W, PREVIEW_SIZE);
    let out_w = PREVIEW_SIZE + PANEL_W + GAP;
    let mut out = vec![0; out_w * PREVIEW_SIZE * 4];
    fill_rect(&mut out, out_w, 0, 0, out_w, PREVIEW_SIZE, [5, 8, 10, 255]);
    blit_rgba_at(&art, PREVIEW_SIZE, PREVIEW_SIZE, &mut out, out_w, 0, 0);
    blit_rgba_at(
        &panel,
        PANEL_W,
        PREVIEW_SIZE,
        &mut out,
        out_w,
        PREVIEW_SIZE + GAP,
        0,
    );
    export::save_png(
        "peterMath_exports/previews/lenia_explanation_reference.png",
        out_w,
        PREVIEW_SIZE,
        &out,
    )
}

fn render_preset_tile(preset: &str, steps: usize, size: usize) -> Vec<u8> {
    let mut sim = LeniaSim::new(96, 96, seed_for_preset(preset));
    sim.reset_preset(preset);
    for _ in 0..steps {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
    upscale_rgba(&pixels, w, h, size, size)
}

fn seed_for_preset(preset: &str) -> u64 {
    match preset {
        "twin_organisms" => 1101,
        "kernel_ring" => 1201,
        "sparse_soup" => 1301,
        "dense_bloom" => 1401,
        "coral_drift" => 1501,
        _ => 1001,
    }
}

fn render_explanation_panel(sim: &LeniaSim, w: usize, h: usize) -> Vec<u8> {
    let metrics = sim.metrics();
    let inspection = sim.inspect_point(sim.size().0 / 2, sim.size().1 / 2);
    let mut panel = vec![0; w * h * 4];
    fill_rect(&mut panel, w, 0, 0, w, h, [8, 12, 14, 255]);
    fill_rect(&mut panel, w, 0, 0, 8, h, [100, 232, 218, 255]);
    fill_rect(&mut panel, w, 26, 30, w - 52, 4, [255, 219, 128, 255]);
    fill_rect(
        &mut panel,
        w,
        26,
        74,
        ((w - 52) as f32 * metrics.mass).round() as usize,
        12,
        [100, 232, 218, 255],
    );
    fill_rect(
        &mut panel,
        w,
        26,
        108,
        ((w - 52) as f32 * metrics.entropy).round() as usize,
        12,
        [255, 118, 168, 255],
    );
    fill_rect(
        &mut panel,
        w,
        26,
        142,
        ((w - 52) as f32 * metrics.stability).round() as usize,
        12,
        [255, 219, 128, 255],
    );
    fill_rect(
        &mut panel,
        w,
        26,
        176,
        ((w - 52) as f32 * metrics.vitality).round() as usize,
        12,
        [216, 240, 139, 255],
    );
    let kernel = sim.kernel_profile(72);
    let chart = (26, 230, w - 52, 90);
    fill_rect(
        &mut panel,
        w,
        chart.0,
        chart.1,
        chart.2,
        chart.3,
        [5, 8, 10, 255],
    );
    for i in 1..kernel.len() {
        let x0 = chart.0 + (i - 1) * chart.2 / (kernel.len() - 1);
        let x1 = chart.0 + i * chart.2 / (kernel.len() - 1);
        let y0 = chart.1 + chart.3 - (kernel[i - 1] * chart.3 as f32) as usize;
        let y1 = chart.1 + chart.3 - (kernel[i] * chart.3 as f32) as usize;
        draw_line(
            &mut panel,
            w,
            x0 as isize,
            y0 as isize,
            x1 as isize,
            y1 as isize,
            [100, 232, 218, 255],
        );
    }
    let response_y = (360.0 - inspection.growth.clamp(-1.0, 1.0) * 46.0).round() as usize;
    fill_rect(&mut panel, w, 26, 356, w - 52, 2, [32, 48, 54, 255]);
    fill_rect(
        &mut panel,
        w,
        26,
        response_y,
        w - 52,
        3,
        [255, 118, 168, 255],
    );
    panel
}

fn upscale_rgba(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<u8> {
    let mut out = vec![0; target_w * target_h * 4];
    if source_w == 0 || source_h == 0 || target_w == 0 || target_h == 0 {
        return out;
    }
    for y in 0..target_h {
        let fy = if target_h > 1 {
            y as f32 * (source_h - 1) as f32 / (target_h - 1) as f32
        } else {
            0.0
        };
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(source_h - 1);
        let ty = fy - y0 as f32;
        for x in 0..target_w {
            let fx = if target_w > 1 {
                x as f32 * (source_w - 1) as f32 / (target_w - 1) as f32
            } else {
                0.0
            };
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(source_w - 1);
            let tx = fx - x0 as f32;
            let dst = (y * target_w + x) * 4;
            for c in 0..4 {
                let a = source[(y0 * source_w + x0) * 4 + c] as f32;
                let b = source[(y0 * source_w + x1) * 4 + c] as f32;
                let c0 = source[(y1 * source_w + x0) * 4 + c] as f32;
                let d = source[(y1 * source_w + x1) * 4 + c] as f32;
                let top = a + (b - a) * tx;
                let bottom = c0 + (d - c0) * tx;
                out[dst + c] = (top + (bottom - top) * ty).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn blit_rgba_at(
    source: &[u8],
    source_w: usize,
    source_h: usize,
    out: &mut [u8],
    out_w: usize,
    x0: usize,
    y0: usize,
) {
    for y in 0..source_h {
        let dst_y = y0 + y;
        let dst_start = (dst_y * out_w + x0) * 4;
        let src_start = y * source_w * 4;
        out[dst_start..dst_start + source_w * 4]
            .copy_from_slice(&source[src_start..src_start + source_w * 4]);
    }
}

fn fill_rect(out: &mut [u8], out_w: usize, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
    let out_h = out.len() / 4 / out_w;
    for yy in y..(y + h).min(out_h) {
        for xx in x..(x + w).min(out_w) {
            let i = (yy * out_w + xx) * 4;
            out[i..i + 4].copy_from_slice(&color);
        }
    }
}

fn draw_line(
    out: &mut [u8],
    out_w: usize,
    mut x0: isize,
    mut y0: isize,
    x1: isize,
    y1: isize,
    color: [u8; 4],
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        set_pixel(out, out_w, x0, y0, color);
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

fn set_pixel(out: &mut [u8], out_w: usize, x: isize, y: isize, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as usize;
    let y = y as usize;
    let out_h = out.len() / 4 / out_w;
    if x >= out_w || y >= out_h {
        return;
    }
    let i = (y * out_w + x) * 4;
    out[i..i + 4].copy_from_slice(&color);
}
