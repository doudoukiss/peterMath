use crate::metrics::Metrics;
use serde::Serialize;
use serde_json::Value;
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
    backend: &'a str,
    seed: u64,
    step_count: u64,
    grid_width: usize,
    grid_height: usize,
    parameters: Value,
    metrics: SerializableMetrics,
}

pub struct SnapshotExport<'a> {
    pub mode: &'a str,
    pub render_style: &'a str,
    pub backend: &'a str,
    pub seed: u64,
    pub step_count: u64,
    pub grid_width: usize,
    pub grid_height: usize,
    pub parameters: Value,
    pub metrics: Metrics,
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

pub fn save_json(path: &str, snapshot: SnapshotExport<'_>) -> anyhow::Result<()> {
    let metadata = SnapshotMetadata {
        project: "peterMath",
        mode: snapshot.mode,
        render_style: snapshot.render_style,
        backend: snapshot.backend,
        seed: snapshot.seed,
        step_count: snapshot.step_count,
        grid_width: snapshot.grid_width,
        grid_height: snapshot.grid_height,
        parameters: snapshot.parameters,
        metrics: SerializableMetrics {
            mass: snapshot.metrics.mass,
            entropy: snapshot.metrics.entropy,
            symmetry: snapshot.metrics.symmetry,
            stability: snapshot.metrics.stability,
            vitality: snapshot.metrics.vitality,
            active: snapshot.metrics.active,
        },
    };
    let json = serde_json::to_string_pretty(&metadata)?;
    fs::write(path, json)?;
    Ok(())
}
