use crate::metrics::Metrics;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;

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
    schema_version: u32,
    project: &'a str,
    mode: &'a str,
    render_style: &'a str,
    backend: &'a str,
    seed: u64,
    step_count: u64,
    grid_width: usize,
    grid_height: usize,
    teaching_mission: Option<Value>,
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
    pub teaching_mission: Option<Value>,
    pub parameters: Value,
    pub metrics: Metrics,
}

#[derive(Serialize, Clone, Copy)]
struct SerializableMetrics {
    mass: f32,
    entropy: f32,
    symmetry: f32,
    stability: f32,
    vitality: f32,
    active: usize,
}

#[derive(Serialize)]
pub struct ShareState<'a> {
    schema_version: u32,
    project: &'a str,
    mode: &'a str,
    render_style: &'a str,
    backend: &'a str,
    seed: u64,
    step_count: u64,
    grid_width: usize,
    grid_height: usize,
    teaching_mission: Option<Value>,
    parameters: Value,
    metrics: SerializableMetrics,
}

pub struct ShareStateExport<'a> {
    pub mode: &'a str,
    pub render_style: &'a str,
    pub backend: &'a str,
    pub seed: u64,
    pub step_count: u64,
    pub grid_width: usize,
    pub grid_height: usize,
    pub teaching_mission: Option<Value>,
    pub parameters: Value,
    pub metrics: Metrics,
}

pub struct EvidencePack {
    pub dir: PathBuf,
    pub snapshot_png: PathBuf,
    pub parameters_json: PathBuf,
    pub share_state_json: PathBuf,
    pub summary_md: PathBuf,
}

pub fn save_json(path: &str, snapshot: SnapshotExport<'_>) -> anyhow::Result<()> {
    let metadata = SnapshotMetadata {
        schema_version: SCHEMA_VERSION,
        project: "peterMath",
        mode: snapshot.mode,
        render_style: snapshot.render_style,
        backend: snapshot.backend,
        seed: snapshot.seed,
        step_count: snapshot.step_count,
        grid_width: snapshot.grid_width,
        grid_height: snapshot.grid_height,
        teaching_mission: snapshot.teaching_mission,
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

pub fn save_share_state(path: impl AsRef<Path>, state: ShareStateExport<'_>) -> anyhow::Result<()> {
    let share_state = share_state(state);
    let json = serde_json::to_string_pretty(&share_state)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn create_evidence_pack(
    dir: impl AsRef<Path>,
    snapshot_name: &str,
    w: usize,
    h: usize,
    rgba: &[u8],
    state: ShareStateExport<'_>,
) -> anyhow::Result<EvidencePack> {
    let dir = dir.as_ref().to_path_buf();
    fs::create_dir_all(&dir)?;
    let snapshot_png = dir.join(format!("{snapshot_name}_snapshot.png"));
    let parameters_json = dir.join(format!("{snapshot_name}_parameters.json"));
    let share_state_json = dir.join("peterMath_share_state.json");
    let summary_md = dir.join("SUMMARY.md");

    save_png(path_str(&snapshot_png)?, w, h, rgba)?;
    save_json(
        path_str(&parameters_json)?,
        SnapshotExport {
            mode: state.mode,
            render_style: state.render_style,
            backend: state.backend,
            seed: state.seed,
            step_count: state.step_count,
            grid_width: state.grid_width,
            grid_height: state.grid_height,
            teaching_mission: state.teaching_mission.clone(),
            parameters: state.parameters.clone(),
            metrics: state.metrics,
        },
    )?;
    save_share_state(&share_state_json, state)?;
    fs::write(&summary_md, evidence_summary(snapshot_name, w, h))?;

    Ok(EvidencePack {
        dir,
        snapshot_png,
        parameters_json,
        share_state_json,
        summary_md,
    })
}

fn share_state(state: ShareStateExport<'_>) -> ShareState<'_> {
    ShareState {
        schema_version: SCHEMA_VERSION,
        project: "peterMath",
        mode: state.mode,
        render_style: state.render_style,
        backend: state.backend,
        seed: state.seed,
        step_count: state.step_count,
        grid_width: state.grid_width,
        grid_height: state.grid_height,
        teaching_mission: state.teaching_mission,
        parameters: state.parameters,
        metrics: SerializableMetrics {
            mass: state.metrics.mass,
            entropy: state.metrics.entropy,
            symmetry: state.metrics.symmetry,
            stability: state.metrics.stability,
            vitality: state.metrics.vitality,
            active: state.metrics.active,
        },
    }
}

fn evidence_summary(snapshot_name: &str, w: usize, h: usize) -> String {
    format!(
        "# peterMath Lenia Evidence Pack\n\n\
This folder contains a reproducible Lenia teaching-game snapshot from the native peterMath app.\n\n\
- Snapshot: `{snapshot_name}_snapshot.png`\n\
- Parameters: `{snapshot_name}_parameters.json`\n\
- Share state: `peterMath_share_state.json`\n\
- Image size: {w}x{h}\n\n\
Use the JSON files to verify mission progress, seed, Lenia parameters, metrics, inspector data, active region, show-mode context, and performance diagnostics for the same visible state.\n"
    )
}

fn path_str(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("path is not valid UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_json_path(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("peterMath_{name}_{stamp}.json"))
    }

    #[test]
    fn snapshot_export_includes_teaching_mission_top_level() {
        let path = temp_json_path("snapshot");
        let mission = json!({
            "mission_id": "wake_field",
            "title_zh": "唤醒生命场",
            "status": "completed",
            "progress": 1.0,
            "completed_missions": ["wake_field"],
            "takeaway_zh": "Lenia 每一步都由同一个局部规则实时计算。"
        });

        save_json(
            path.to_str().unwrap(),
            SnapshotExport {
                mode: "连续生命场 Lenia",
                render_style: "艺术表达图",
                backend: "CPU 参考",
                seed: 1001,
                step_count: 60,
                grid_width: 16,
                grid_height: 16,
                teaching_mission: Some(mission.clone()),
                parameters: json!({"kernel_radius": 9}),
                metrics: Metrics {
                    mass: 0.1,
                    entropy: 0.2,
                    symmetry: 0.3,
                    stability: 0.4,
                    vitality: 0.5,
                    active: 42,
                },
            },
        )
        .unwrap();

        let exported: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(exported["teaching_mission"], mission);
        assert_eq!(exported["seed"], 1001);
        assert_eq!(exported["parameters"]["kernel_radius"], 9);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn share_state_export_includes_teaching_mission_top_level() {
        let path = temp_json_path("share");
        let mission = json!({
            "mission_id": "evidence_report",
            "title_zh": "生成证据报告",
            "status": "completed",
            "progress": 1.0,
            "completed_missions": ["wake_field", "evidence_report"],
            "takeaway_zh": "证据文件记录同一帧状态。"
        });

        save_share_state(
            &path,
            ShareStateExport {
                mode: "连续生命场 Lenia",
                render_style: "艺术表达图",
                backend: "CPU 参考",
                seed: 1001,
                step_count: 120,
                grid_width: 16,
                grid_height: 16,
                teaching_mission: Some(mission.clone()),
                parameters: json!({"kernel_radius": 9}),
                metrics: Metrics {
                    mass: 0.1,
                    entropy: 0.2,
                    symmetry: 0.3,
                    stability: 0.4,
                    vitality: 0.5,
                    active: 42,
                },
            },
        )
        .unwrap();

        let exported: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(exported["teaching_mission"], mission);
        assert_eq!(exported["step_count"], 120);
        let _ = fs::remove_file(path);
    }
}
