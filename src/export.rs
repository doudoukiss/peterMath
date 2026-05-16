use crate::metrics::Metrics;
use serde::Serialize;
use std::fs;
use std::path::Path;

pub fn save_png(path: &str, w: usize, h: usize, rgba: &[u8]) -> anyhow::Result<()> {
    image::save_buffer(
        Path::new(path),
        rgba,
        w as u32,
        h as u32,
        image::ColorType::Rgba8,
    )?;
    Ok(())
}

#[derive(Serialize)]
struct SnapshotMetadata<'a> {
    project: &'a str,
    mode: &'a str,
    render_style: &'a str,
    seed: u64,
    step_count: u64,
    metrics: SerializableMetrics,
}

#[derive(Serialize)]
struct SerializableMetrics {
    mass: f32,
    entropy: f32,
    symmetry: f32,
    stability: f32,
    vitality: f32,
    active: usize,
}

pub fn save_json(
    path: &str,
    mode: &str,
    render_style: &str,
    seed: u64,
    step_count: u64,
    metrics: Metrics,
) -> anyhow::Result<()> {
    let metadata = SnapshotMetadata {
        project: "peterMath",
        mode,
        render_style,
        seed,
        step_count,
        metrics: SerializableMetrics {
            mass: metrics.mass,
            entropy: metrics.entropy,
            symmetry: metrics.symmetry,
            stability: metrics.stability,
            vitality: metrics.vitality,
            active: metrics.active,
        },
    };
    let json = serde_json::to_string_pretty(&metadata)?;
    fs::write(path, json)?;
    Ok(())
}
