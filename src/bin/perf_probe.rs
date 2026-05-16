#![allow(dead_code)]

#[path = "../metrics.rs"]
mod metrics;
#[path = "../palette.rs"]
mod palette;
#[path = "../simulation/mod.rs"]
mod simulation;

use simulation::lenia::LeniaSim;
use simulation::RenderStyle;
use std::time::{Duration, Instant};

fn main() {
    println!("peterMath Lenia performance probe");
    println!("system,size,step_ms,render_ms,mass,entropy,stability");
    for size in [128, 256, 512] {
        probe_lenia(size);
    }
}

fn probe_lenia(size: usize) {
    let mut sim = LeniaSim::new(size, size, 1001);
    for _ in 0..8 {
        sim.step();
    }
    let step_iters = iterations_for(size, 48, 18, 5);
    let step_ms = measure_steps(step_iters, || sim.step());
    let mut pixels = vec![0; size * size * 4];
    let render_iters = iterations_for(size, 24, 10, 3);
    let render_ms = measure_steps(render_iters, || {
        sim.render_rgba(RenderStyle::Artistic, &mut pixels)
    });
    let metrics = sim.metrics();
    println!(
        "Lenia,{size}x{size},{step_ms:.3},{render_ms:.3},{:.4},{:.4},{:.4}",
        metrics.mass, metrics.entropy, metrics.stability
    );
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
