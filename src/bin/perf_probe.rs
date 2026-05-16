#![allow(dead_code)]

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
use std::time::{Duration, Instant};

fn main() {
    println!("peterMath performance probe");
    println!("system,size,step_ms,render_ms");
    for size in [128, 256, 512] {
        probe_lenia(size);
        probe_reaction_diffusion(size);
        probe_life(size);
    }
}

fn probe_lenia(size: usize) {
    let mut sim = LeniaSim::new(size, size, 1001);
    for _ in 0..4 {
        sim.step();
    }
    let step_iters = iterations_for(size, 40, 16, 4);
    let step_ms = measure_steps(step_iters, || sim.step());
    let mut pixels = vec![0; size * size * 4];
    let render_iters = iterations_for(size, 18, 8, 3);
    let render_ms = measure_steps(render_iters, || {
        sim.render_rgba(RenderStyle::Artistic, &mut pixels)
    });
    print_result("Lenia", size, step_ms, render_ms);
}

fn probe_reaction_diffusion(size: usize) {
    let mut sim = ReactionDiffusionSim::new(size, size, 2001);
    sim.reset_preset("labyrinth");
    for _ in 0..12 {
        sim.step();
    }
    let step_iters = iterations_for(size, 80, 28, 8);
    let step_ms = measure_steps(step_iters, || sim.step());
    let mut pixels = vec![0; size * size * 4];
    let render_iters = iterations_for(size, 18, 8, 3);
    let render_ms = measure_steps(render_iters, || {
        sim.render_rgba(RenderStyle::Artistic, &mut pixels)
    });
    print_result("Reaction-Diffusion", size, step_ms, render_ms);
}

fn probe_life(size: usize) {
    let mut sim = LifeSim::new(size, size, 3001);
    for _ in 0..8 {
        sim.step();
    }
    let step_iters = iterations_for(size, 120, 48, 16);
    let step_ms = measure_steps(step_iters, || sim.step());
    let mut pixels = vec![0; size * size * 4];
    let render_iters = iterations_for(size, 24, 10, 4);
    let render_ms = measure_steps(render_iters, || {
        sim.render_rgba(RenderStyle::Artistic, &mut pixels)
    });
    print_result("Game of Life", size, step_ms, render_ms);
}

fn iterations_for(size: usize, small: usize, medium: usize, large: usize) -> usize {
    match size {
        0..=128 => small,
        129..=256 => medium,
        _ => large,
    }
}

fn measure_steps(iterations: usize, mut work: impl FnMut()) -> f32 {
    let start = Instant::now();
    for _ in 0..iterations {
        work();
    }
    average_ms(start.elapsed(), iterations)
}

fn average_ms(duration: Duration, iterations: usize) -> f32 {
    duration.as_secs_f32() * 1000.0 / iterations.max(1) as f32
}

fn print_result(system: &str, size: usize, step_ms: f32, render_ms: f32) {
    println!("{system},{size}x{size},{step_ms:.3},{render_ms:.3}");
}
