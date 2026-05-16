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
    export::save_png("peterMath_exports/previews/lenia_hero.png", w, h, &pixels)?;
    export::save_png(
        "peterMath_exports/previews/lenia_showcase.png",
        w,
        h,
        &pixels,
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
    export::save_png(
        "peterMath_exports/previews/reaction_diffusion_texture.png",
        w,
        h,
        &pixels,
    )?;
    export::save_png(
        "peterMath_exports/previews/reaction_diffusion_showcase.png",
        w,
        h,
        &pixels,
    )
}

fn render_judge_reference() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(192, 192, 1001);
    for _ in 0..160 {
        sim.step();
    }

    let (w, h) = sim.size();
    let gap = 16;
    let mut raw = vec![0; w * h * 4];
    let mut art = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::RawMath, &mut raw);
    sim.render_rgba(RenderStyle::Artistic, &mut art);

    let out_w = w * 2 + gap;
    let mut combined = vec![0; out_w * h * 4];
    blit_rgba(&raw, w, h, &mut combined, out_w, 0);
    blit_rgba(&art, w, h, &mut combined, out_w, w + gap);
    export::save_png(
        "peterMath_exports/previews/judge_mode_reference.png",
        out_w,
        h,
        &combined,
    )
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
