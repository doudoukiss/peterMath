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
    println!("Wrote peterMath_exports/previews/lenia_showcase.png");
    println!("Wrote peterMath_exports/previews/reaction_diffusion_showcase.png");
    Ok(())
}

fn render_lenia() -> anyhow::Result<()> {
    let mut sim = LeniaSim::new(192, 192, 1001);
    for _ in 0..180 {
        sim.step();
    }
    let (w, h) = sim.size();
    let mut pixels = vec![0; w * h * 4];
    sim.render_rgba(RenderStyle::Artistic, &mut pixels);
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
        "peterMath_exports/previews/reaction_diffusion_showcase.png",
        w,
        h,
        &pixels,
    )
}
