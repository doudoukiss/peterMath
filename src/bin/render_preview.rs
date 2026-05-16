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
use simulation::reaction_diffusion::ReactionDiffusionSim;
use simulation::RenderStyle;
use std::fs;

const PREVIEW_SIZE: usize = 512;
const JUDGE_GAP: usize = 32;

fn main() -> anyhow::Result<()> {
    fs::create_dir_all("peterMath_exports/previews")?;
    render_lenia()?;
    render_reaction_diffusion()?;
    render_judge_reference()?;
    println!("Wrote peterMath_exports/previews/lenia_hero.png");
    println!("Wrote peterMath_exports/previews/reaction_diffusion_texture.png");
    println!("Wrote peterMath_exports/previews/judge_mode_reference.png");
    println!("Wrote peterMath_exports/previews/lenia_showcase.png");
    println!("Wrote peterMath_exports/previews/reaction_diffusion_showcase.png");
    Ok(())
}

fn render_lenia() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(192, 192, 1001);
    for _ in 0..240 {
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
    let mut sim = ReactionDiffusionSim::new(192, 192, 2001);
    sim.reset_preset("labyrinth");
    for _ in 0..1200 {
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
    let mut sim = LeniaSim::new(192, 192, 1001);
    for _ in 0..160 {
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

    let out_w = PREVIEW_SIZE * 2 + gap;
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
    export::save_png(
        "peterMath_exports/previews/judge_mode_reference.png",
        out_w,
        PREVIEW_SIZE,
        &combined,
    )
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
