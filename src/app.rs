use crate::analysis::{self, ActiveRegionAnalysis, PopulationPhaseAnalysis};
use crate::export;
use crate::gpu::{self, GpuLeniaArt, GpuLeniaParams};
use crate::metrics::Metrics;
use crate::simulation::lenia::{LeniaInspection, LeniaSim, LeniaState};
use crate::simulation::life::{LifeRlePattern, LifeSim, LifeStateHash, PatternDetectionReport};
use crate::simulation::reaction_diffusion::ReactionDiffusionSim;
use crate::simulation::{RenderStyle, SimMode};
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use serde_json::{json, Value};
use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

const HISTORY_LIMIT: usize = 32;
const METRIC_HISTORY_LIMIT: usize = 180;
const TARGET_TICK: Duration = Duration::from_millis(66);
const MAX_FRAME_DELTA: Duration = Duration::from_millis(250);
const MAX_UPDATE_BATCHES: usize = 3;
const GPU_CPU_REFERENCE_SYNC_INTERVAL: usize = 4;

pub struct PeterMathApp {
    screen: AppScreen,
    mode: SimMode,
    overview_focus: SimMode,
    render_style: RenderStyle,
    lenia: LeniaSim,
    reaction: ReactionDiffusionSim,
    life: LifeSim,
    overview_lenia: LeniaSim,
    overview_reaction: ReactionDiffusionSim,
    overview_life: LifeSim,
    overview_step: u64,
    overview_lenia_texture: Option<TextureHandle>,
    overview_reaction_texture: Option<TextureHandle>,
    overview_life_texture: Option<TextureHandle>,
    gpu_lenia: Option<GpuLeniaArt>,
    prefer_gpu_lenia: bool,
    running: bool,
    judge_mode: bool,
    tool: InteractionTool,
    active_preset: LeniaPreset,
    active_reaction_preset: ReactionPreset,
    active_life_preset: LifePreset,
    active_stamp: LeniaStamp,
    grid_profile: GridProfile,
    random_density: f32,
    brush_radius: f32,
    brush_strength: f32,
    life_brush_radius: f32,
    reaction_brush_radius: f32,
    reaction_brush_strength: f32,
    undo_stack: Vec<LeniaHistorySnapshot>,
    redo_stack: Vec<LeniaHistorySnapshot>,
    pointer_edit_active: bool,
    inspected_lenia: Option<LeniaInspection>,
    show_kernel_overlay: bool,
    metric_history: Vec<MetricHistorySample>,
    dev_diagnostics: bool,
    performance: PerformanceStats,
    cpu_texture_dirty: bool,
    tick_accumulator: Duration,
    gpu_cpu_sync_interval: usize,
    gpu_cpu_sync_counter: usize,
    steps_per_frame: usize,
    step_count: u64,
    pixels: Vec<u8>,
    texture: Option<TextureHandle>,
    life_rle_input: String,
    life_rle_output: String,
    active_region_history: Vec<(f32, f32)>,
    life_hash_history: Vec<(u64, LifeStateHash)>,
    last_glider_centroid: Option<(f32, f32)>,
    show_active_region_overlay: bool,
    comparison_baseline: Option<LeniaState>,
    comparison_parameter: VariantParameter,
    comparison_value: f32,
    comparison_steps: usize,
    comparison_result: Option<RuleVariantComparison>,
    comparison_baseline_texture: Option<TextureHandle>,
    comparison_variant_texture: Option<TextureHandle>,
    status: String,
    last_interaction: String,
    last_tick: Instant,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppScreen {
    Overview,
    Experiment,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InteractionTool {
    Draw,
    Erase,
    Stamp,
    Pan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LeniaPreset {
    OrbitalField,
    TwinOrganisms,
    CoralDrift,
    KernelRing,
    SparseSoup,
    DenseBloom,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LeniaStamp {
    SoftCell,
    RingSeed,
    TwinSeed,
    ArcSeed,
    NoisePatch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReactionPreset {
    Labyrinth,
    Mitosis,
    Spots,
    Waves,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LifePreset {
    StructureShowcase,
    Glider,
    Oscillator,
    RandomSoup,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GridProfile {
    Reference192,
    Detail256,
    GpuPreview512,
}

#[derive(Clone)]
struct LeniaHistorySnapshot {
    state: LeniaState,
    step_count: u64,
    active_preset: LeniaPreset,
    grid_profile: GridProfile,
    random_density: f32,
}

#[derive(Clone, Copy)]
struct MetricHistorySample {
    step_count: u64,
    mass: f32,
    entropy: f32,
    stability: f32,
    vitality: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LeniaPhase {
    Sparse,
    Blooming,
    Drifting,
    Stabilizing,
    Turbulent,
    Dense,
    Fading,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VariantParameter {
    KernelRadius,
    GrowthCenter,
    GrowthWidth,
    Damping,
}

struct RuleVariantComparison {
    parameter: VariantParameter,
    value: f32,
    steps: usize,
    baseline_metrics: Metrics,
    variant_metrics: Metrics,
    width: usize,
    height: usize,
    baseline_pixels: Vec<u8>,
    variant_pixels: Vec<u8>,
}

struct GpuLeniaExportState {
    size: usize,
    pixels: Vec<u8>,
    metrics: Metrics,
    parameters: Value,
}

struct OverviewCard<'a> {
    title: &'a str,
    plain_rule: &'a str,
    goal: &'a str,
    try_action: &'a str,
    stage: &'a str,
    metrics: Metrics,
    conclusion: &'a str,
}

#[derive(Clone, Copy, Default)]
struct FrameTimingSample {
    frame_ms: f32,
    update_ms: f32,
    render_ms: f32,
    cpu_sync_ms: f32,
}

#[derive(Default)]
struct PerformanceStats {
    latest: FrameTimingSample,
    fps_estimate: f32,
    frame_samples: usize,
    source_grid: (usize, usize),
    gpu_grid: Option<usize>,
    pending_steps: u32,
    cpu_sync_interval: usize,
}

impl InteractionTool {
    const ALL: [Self; 4] = [Self::Draw, Self::Erase, Self::Stamp, Self::Pan];

    fn label(self) -> &'static str {
        match self {
            Self::Draw => "绘制",
            Self::Erase => "擦除",
            Self::Stamp => "盖章",
            Self::Pan => "观察",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::Draw => "draw",
            Self::Erase => "erase",
            Self::Stamp => "stamp",
            Self::Pan => "pan",
        }
    }
}

impl AppScreen {
    fn label_zh(self) -> &'static str {
        match self {
            Self::Overview => "三系统总览",
            Self::Experiment => "深入实验",
        }
    }
}

impl LeniaPreset {
    const ALL: [Self; 6] = [
        Self::OrbitalField,
        Self::TwinOrganisms,
        Self::CoralDrift,
        Self::KernelRing,
        Self::SparseSoup,
        Self::DenseBloom,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::OrbitalField => "轨道场",
            Self::TwinOrganisms => "双生命体",
            Self::CoralDrift => "珊瑚漂移",
            Self::KernelRing => "核环",
            Self::SparseSoup => "稀疏汤",
            Self::DenseBloom => "密集开花",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::OrbitalField => "orbital_field",
            Self::TwinOrganisms => "twin_organisms",
            Self::CoralDrift => "coral_drift",
            Self::KernelRing => "kernel_ring",
            Self::SparseSoup => "sparse_soup",
            Self::DenseBloom => "dense_bloom",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::OrbitalField => "螺旋种子显示连续场如何旋转、漂移并形成柔软边界。",
            Self::TwinOrganisms => "两个镜像团块展示同一规则如何产生不同生命形态。",
            Self::CoralDrift => "分枝路径强调脊线生长、衰减和边界竞争。",
            Self::KernelRing => "环形质量让邻域卷积核的半径和影响范围更清楚。",
            Self::SparseSoup => "低密度随机场测试少量岛屿能否自组织。",
            Self::DenseBloom => "高密度质量展示饱和、湍动和衰退的风险。",
        }
    }
}

impl LeniaStamp {
    const ALL: [Self; 5] = [
        Self::SoftCell,
        Self::RingSeed,
        Self::TwinSeed,
        Self::ArcSeed,
        Self::NoisePatch,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::SoftCell => "软细胞",
            Self::RingSeed => "环形种子",
            Self::TwinSeed => "双种子",
            Self::ArcSeed => "弧形种子",
            Self::NoisePatch => "噪声块",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::SoftCell => "soft_cell",
            Self::RingSeed => "ring_seed",
            Self::TwinSeed => "twin_seed",
            Self::ArcSeed => "arc_seed",
            Self::NoisePatch => "noise_patch",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::SoftCell => "单个高斯团块，用来测试局部增长响应。",
            Self::RingSeed => "环形盖章，对应卷积核的圆形采样结构。",
            Self::TwinSeed => "成对团块会在同一规则下合并、排斥或绕行。",
            Self::ArcSeed => "局部弧线，用来观察不对称梯度流。",
            Self::NoisePatch => "带种子的微结构，用来激发局部不稳定和纹理。",
        }
    }
}

impl ReactionPreset {
    const ALL: [Self; 4] = [Self::Labyrinth, Self::Mitosis, Self::Spots, Self::Waves];

    fn label(self) -> &'static str {
        match self {
            Self::Labyrinth => "迷宫生长",
            Self::Mitosis => "细胞分裂",
            Self::Spots => "斑点膜",
            Self::Waves => "波纹扩散",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::Labyrinth => "labyrinth",
            Self::Mitosis => "mitosis",
            Self::Spots => "spots",
            Self::Waves => "waves",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Labyrinth => "密集种子快速形成迷宫边界，适合评委现场观察。",
            Self::Mitosis => "小圆点会扩张、分裂，展示反应项的放大作用。",
            Self::Spots => "稳定斑点强调扩散速度差产生的空间纹理。",
            Self::Waves => "稀疏扰动生成缓慢波前，适合看传播过程。",
        }
    }
}

impl LifePreset {
    const ALL: [Self; 4] = [
        Self::StructureShowcase,
        Self::Glider,
        Self::Oscillator,
        Self::RandomSoup,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::StructureShowcase => "结构展示",
            Self::Glider => "滑翔机",
            Self::Oscillator => "振荡器",
            Self::RandomSoup => "随机汤",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::StructureShowcase => "structure_showcase",
            Self::Glider => "glider",
            Self::Oscillator => "oscillator",
            Self::RandomSoup => "random_soup",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::StructureShowcase => "同屏展示静态结构、周期振荡和滑翔移动。",
            Self::Glider => "多个滑翔机会沿对角线移动，便于观察质心漂移。",
            Self::Oscillator => "闪烁、蟾蜍和信标展示离散周期。",
            Self::RandomSoup => "随机细胞汤会快速淘汰并留下少数稳定结构。",
        }
    }
}

impl GridProfile {
    const ALL: [Self; 3] = [Self::Reference192, Self::Detail256, Self::GpuPreview512];

    fn label(self) -> &'static str {
        match self {
            Self::Reference192 => "192 参考",
            Self::Detail256 => "256 细节",
            Self::GpuPreview512 => "512 GPU 预览",
        }
    }

    fn size(self) -> usize {
        match self {
            Self::Reference192 => 192,
            Self::Detail256 => 256,
            Self::GpuPreview512 => 512,
        }
    }
}

impl MetricHistorySample {
    fn from_metrics(step_count: u64, metrics: Metrics) -> Self {
        Self {
            step_count,
            mass: metrics.mass,
            entropy: metrics.entropy,
            stability: metrics.stability,
            vitality: metrics.vitality,
        }
    }
}

impl LeniaPhase {
    fn from_metrics(metrics: Metrics, mass_trend: f32) -> Self {
        if metrics.mass < 0.012 || metrics.active < 24 {
            return Self::Sparse;
        }
        if mass_trend < -0.010 && metrics.vitality < 0.32 {
            return Self::Fading;
        }
        if metrics.mass > 0.44 {
            return Self::Dense;
        }
        if metrics.stability > 0.965 && metrics.vitality < 0.42 {
            return Self::Stabilizing;
        }
        if (1.0 - metrics.stability) > 0.18 && metrics.entropy > 0.42 {
            return Self::Turbulent;
        }
        if mass_trend > 0.006 || metrics.vitality > 0.58 {
            return Self::Blooming;
        }
        Self::Drifting
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Blooming => "blooming",
            Self::Drifting => "drifting",
            Self::Stabilizing => "stabilizing",
            Self::Turbulent => "turbulent",
            Self::Dense => "dense",
            Self::Fading => "fading",
        }
    }

    fn label_zh(self) -> &'static str {
        match self {
            Self::Sparse => "稀疏",
            Self::Blooming => "快速增长",
            Self::Drifting => "周期/漂移",
            Self::Stabilizing => "稳定形成",
            Self::Turbulent => "边界竞争",
            Self::Dense => "密集饱和",
            Self::Fading => "稳定/衰退",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Sparse => "场质量很低，只有少量区域还可能组织起来。",
            Self::Blooming => "质量或活力正在上升，结构正在形成。",
            Self::Drifting => "结构已经存在，但仍在移动或缓慢变化。",
            Self::Stabilizing => "连续帧接近，运动正在稳定。",
            Self::Turbulent => "熵和变化量较高，边界正在竞争。",
            Self::Dense => "场质量很高，系统接近饱和。",
            Self::Fading => "质量和活力下降，结构正在衰退。",
        }
    }
}

impl VariantParameter {
    const ALL: [Self; 4] = [
        Self::KernelRadius,
        Self::GrowthCenter,
        Self::GrowthWidth,
        Self::Damping,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::KernelRadius => "卷积半径",
            Self::GrowthCenter => "增长中心",
            Self::GrowthWidth => "增长宽度",
            Self::Damping => "阻尼",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::KernelRadius => "kernel_radius",
            Self::GrowthCenter => "growth_center",
            Self::GrowthWidth => "growth_width",
            Self::Damping => "damping",
        }
    }

    fn current_value(self, lenia: &LeniaSim) -> f32 {
        match self {
            Self::KernelRadius => lenia.radius as f32,
            Self::GrowthCenter => lenia.growth_center,
            Self::GrowthWidth => lenia.growth_width,
            Self::Damping => lenia.decay,
        }
    }

    fn range(self) -> std::ops::RangeInclusive<f32> {
        match self {
            Self::KernelRadius => 3.0..=32.0,
            Self::GrowthCenter => 0.05..=0.95,
            Self::GrowthWidth => 0.005..=0.18,
            Self::Damping => 0.0..=0.04,
        }
    }

    fn apply(self, lenia: &mut LeniaSim, value: f32) {
        match self {
            Self::KernelRadius => lenia.set_radius(value.round() as usize),
            Self::GrowthCenter => lenia.growth_center = value,
            Self::GrowthWidth => lenia.growth_width = value,
            Self::Damping => lenia.decay = value,
        }
    }
}

impl PerformanceStats {
    fn record_frame_delta(&mut self, delta: Duration) {
        let frame_ms = duration_ms(delta);
        self.latest.frame_ms = frame_ms;
        if frame_ms > 0.0 {
            let instant_fps = 1000.0 / frame_ms;
            self.fps_estimate = if self.frame_samples == 0 {
                instant_fps
            } else {
                self.fps_estimate * 0.88 + instant_fps * 0.12
            };
        }
        self.frame_samples = self.frame_samples.saturating_add(1);
    }

    fn set_timings(&mut self, update: Duration, render: Duration, cpu_sync: Duration) {
        self.latest.update_ms = duration_ms(update);
        self.latest.render_ms = duration_ms(render);
        self.latest.cpu_sync_ms = duration_ms(cpu_sync);
    }
}

impl PeterMathApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let width = 192;
        let render_style = RenderStyle::Artistic;
        let lenia = LeniaSim::new(width, width, 1001);
        let overview_lenia = LeniaSim::new(96, 96, 1001);
        let overview_reaction = ReactionDiffusionSim::new(128, 128, 2001);
        let overview_life = LifeSim::new(64, 64, 3001);
        let inspected_lenia = Some(lenia.inspect_point(width / 2, width / 2));
        let metric_history = vec![MetricHistorySample::from_metrics(0, lenia.metrics())];
        let gpu_lenia = cc.wgpu_render_state.as_ref().and_then(|render_state| {
            GpuLeniaArt::new(
                render_state,
                lenia.field(),
                width,
                width,
                lenia.kernel_entries(),
                lenia_params(&lenia),
                render_style,
            )
            .ok()
        });
        let gpu_ready = gpu_lenia.is_some();
        let performance = PerformanceStats {
            source_grid: lenia.size(),
            gpu_grid: if gpu_ready { Some(512) } else { None },
            cpu_sync_interval: GPU_CPU_REFERENCE_SYNC_INTERVAL,
            ..Default::default()
        };
        Self {
            screen: AppScreen::Overview,
            mode: SimMode::Lenia,
            overview_focus: SimMode::GameOfLife,
            render_style,
            lenia,
            reaction: ReactionDiffusionSim::new(width, width, 2001),
            life: LifeSim::new(96, 96, 3001),
            overview_lenia,
            overview_reaction,
            overview_life,
            overview_step: 0,
            overview_lenia_texture: None,
            overview_reaction_texture: None,
            overview_life_texture: None,
            gpu_lenia,
            prefer_gpu_lenia: gpu_ready,
            running: true,
            judge_mode: false,
            tool: InteractionTool::Draw,
            active_preset: LeniaPreset::OrbitalField,
            active_reaction_preset: ReactionPreset::Labyrinth,
            active_life_preset: LifePreset::StructureShowcase,
            active_stamp: LeniaStamp::SoftCell,
            grid_profile: GridProfile::Reference192,
            random_density: 0.24,
            brush_radius: 9.0,
            brush_strength: 0.42,
            life_brush_radius: 1.3,
            reaction_brush_radius: 8.0,
            reaction_brush_strength: 0.75,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pointer_edit_active: false,
            inspected_lenia,
            show_kernel_overlay: false,
            metric_history,
            dev_diagnostics: false,
            performance,
            cpu_texture_dirty: true,
            tick_accumulator: Duration::ZERO,
            gpu_cpu_sync_interval: GPU_CPU_REFERENCE_SYNC_INTERVAL,
            gpu_cpu_sync_counter: 0,
            steps_per_frame: 1,
            step_count: 0,
            pixels: vec![0; width * width * 4],
            texture: None,
            life_rle_input: "x = 3, y = 3, rule = B3/S23\nbob$2bo$3o!\n".to_owned(),
            life_rle_output: String::new(),
            active_region_history: Vec::new(),
            life_hash_history: Vec::new(),
            last_glider_centroid: None,
            show_active_region_overlay: false,
            comparison_baseline: None,
            comparison_parameter: VariantParameter::GrowthCenter,
            comparison_value: 0.36,
            comparison_steps: 80,
            comparison_result: None,
            comparison_baseline_texture: None,
            comparison_variant_texture: None,
            status: if gpu_ready {
                "GPU Lenia 已启用。总览中可以先比较三种数学系统。".to_owned()
            } else {
                "当前使用 CPU 参考模式。程序仍可运行并展示数学结构。".to_owned()
            },
            last_interaction: "打开程序，显示三系统中文导览。".to_owned(),
            last_tick: Instant::now(),
        }
    }

    fn active_size(&self) -> (usize, usize) {
        match self.mode {
            SimMode::Lenia if self.gpu_lenia_active() => {
                let size = self.gpu_lenia.as_ref().map(|gpu| gpu.size()).unwrap_or(192) as usize;
                (size, size)
            }
            SimMode::Lenia => self.lenia.size(),
            SimMode::ReactionDiffusion => self.reaction.size(),
            SimMode::GameOfLife => self.life.size(),
        }
    }

    fn active_seed(&self) -> u64 {
        match self.mode {
            SimMode::Lenia => self.lenia.seed,
            SimMode::ReactionDiffusion => self.reaction.seed,
            SimMode::GameOfLife => self.life.seed,
        }
    }

    fn active_metrics(&self) -> Metrics {
        match self.mode {
            SimMode::Lenia => self.lenia.metrics(),
            SimMode::ReactionDiffusion => self.reaction.metrics(),
            SimMode::GameOfLife => self.life.metrics(),
        }
    }

    fn recommended_steps_for(mode: SimMode) -> usize {
        match mode {
            SimMode::Lenia => 1,
            SimMode::ReactionDiffusion => 8,
            SimMode::GameOfLife => 3,
        }
    }

    fn active_stage_zh(&self) -> &'static str {
        let metrics = self.active_metrics();
        let phase = self.population_phase_analysis();
        match self.mode {
            SimMode::Lenia => self.lenia_phase().label_zh(),
            SimMode::ReactionDiffusion => {
                if self.step_count < 60 {
                    "初始扰动"
                } else if phase.mass_trend.abs() > 0.004 || phase.entropy_trend.abs() > 0.004 {
                    "结构形成"
                } else if metrics.stability > 0.982 {
                    "稳定/衰退"
                } else {
                    "快速增长"
                }
            }
            SimMode::GameOfLife => {
                if metrics.active == 0 {
                    "稳定/衰退"
                } else if phase.centroid_drift.0.abs() + phase.centroid_drift.1.abs() > 0.15 {
                    "周期/漂移"
                } else if metrics.stability > 0.97 {
                    "稳定形成"
                } else {
                    "快速增长"
                }
            }
        }
    }

    fn record_metric_history(&mut self) {
        let sample = MetricHistorySample::from_metrics(self.step_count, self.active_metrics());
        if let Some(last) = self.metric_history.last_mut() {
            if last.step_count == sample.step_count {
                *last = sample;
                self.record_interpretability_history();
                return;
            }
        }
        self.metric_history.push(sample);
        if self.metric_history.len() > METRIC_HISTORY_LIMIT {
            self.metric_history.remove(0);
        }
        self.record_interpretability_history();
    }

    fn reset_metric_history(&mut self) {
        self.metric_history.clear();
        self.active_region_history.clear();
        self.life_hash_history.clear();
        self.last_glider_centroid = None;
        self.record_metric_history();
    }

    fn refresh_lenia_inspection(&mut self) {
        if let Some(inspection) = self.inspected_lenia {
            self.inspected_lenia = Some(self.lenia.inspect_point(inspection.x, inspection.y));
        }
    }

    fn lenia_phase(&self) -> LeniaPhase {
        let metrics = self.lenia.metrics();
        let reference_mass = self
            .metric_history
            .len()
            .checked_sub(2)
            .and_then(|idx| self.metric_history.get(idx))
            .or_else(|| self.metric_history.last())
            .map(|sample| sample.mass)
            .unwrap_or(metrics.mass);
        let mass_trend = metrics.mass - reference_mass;
        LeniaPhase::from_metrics(metrics, mass_trend)
    }

    fn previous_centroid(&self) -> Option<(f32, f32)> {
        self.active_region_history.last().copied()
    }

    fn active_region(&self) -> ActiveRegionAnalysis {
        match self.mode {
            SimMode::Lenia => analysis::active_region_from_scalar_grid(
                self.lenia.field(),
                self.lenia.size().0,
                self.lenia.size().1,
                0.08,
                self.previous_centroid(),
            ),
            SimMode::ReactionDiffusion => analysis::active_region_from_scalar_grid(
                self.reaction.field(),
                self.reaction.size().0,
                self.reaction.size().1,
                0.08,
                self.previous_centroid(),
            ),
            SimMode::GameOfLife => {
                let (w, h) = self.life.size();
                analysis::active_region_from_points(
                    w,
                    h,
                    self.life.live_points().into_iter(),
                    self.previous_centroid(),
                )
            }
        }
    }

    fn population_phase_analysis(&self) -> PopulationPhaseAnalysis {
        let current = self.active_metrics();
        let previous = self.previous_metric_history(current);
        let label = match self.mode {
            SimMode::Lenia => self.lenia_phase().label(),
            SimMode::ReactionDiffusion => {
                let mass_trend = previous
                    .map(|previous| current.mass - previous.mass)
                    .unwrap_or_default();
                LeniaPhase::from_metrics(current, mass_trend).label()
            }
            SimMode::GameOfLife => "discrete",
        };
        self.population_phase_from(label, current, self.active_region())
    }

    fn previous_metric_history(&self, current: Metrics) -> Option<Metrics> {
        self.metric_history
            .iter()
            .rev()
            .nth(1)
            .map(|sample| Metrics {
                mass: sample.mass,
                entropy: sample.entropy,
                symmetry: current.symmetry,
                stability: sample.stability,
                vitality: sample.vitality,
                active: current.active,
            })
    }

    fn population_phase_from(
        &self,
        label: &'static str,
        current: Metrics,
        active_region: ActiveRegionAnalysis,
    ) -> PopulationPhaseAnalysis {
        let previous = self.previous_metric_history(current);
        analysis::population_phase_analysis(label, current, previous, active_region)
    }

    fn life_pattern_report(&self) -> PatternDetectionReport {
        self.life.detect_known_patterns(
            &self.life_hash_history,
            self.step_count,
            self.last_glider_centroid,
        )
    }

    fn record_interpretability_history(&mut self) {
        if let Some(centroid) = self.active_region().centroid {
            if self
                .active_region_history
                .last()
                .map(|last| {
                    (last.0 - centroid.0).abs() > 0.001 || (last.1 - centroid.1).abs() > 0.001
                })
                .unwrap_or(true)
            {
                self.active_region_history.push(centroid);
                if self.active_region_history.len() > 64 {
                    self.active_region_history.remove(0);
                }
            }
        }

        if self.mode == SimMode::GameOfLife {
            let report = self.life_pattern_report();
            self.last_glider_centroid = report.glider_track.and_then(|track| track.centroid);
            let hash = self.life.state_hash();
            if let Some(last) = self.life_hash_history.last_mut() {
                if last.0 == self.step_count {
                    *last = (self.step_count, hash);
                    return;
                }
            }
            self.life_hash_history.push((self.step_count, hash));
            if self.life_hash_history.len() > 64 {
                self.life_hash_history.remove(0);
            }
        }
    }

    fn reset_active(&mut self) {
        self.step_count = 0;
        match self.mode {
            SimMode::Lenia => {
                self.lenia.reset_preset(self.active_preset.id());
                self.sync_gpu_lenia_from_cpu();
            }
            SimMode::ReactionDiffusion => {
                self.reaction.reset_preset(self.active_reaction_preset.id())
            }
            SimMode::GameOfLife => self.life.reset_preset(self.active_life_preset.id()),
        }
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.last_interaction = format!("{}已重置到当前预设。", self.mode.label_zh());
        self.status = "已重新计算当前规则：重置生效。".to_owned();
    }

    fn step_active(&mut self) {
        match self.mode {
            SimMode::Lenia => self.lenia.step(),
            SimMode::ReactionDiffusion => self.reaction.step(),
            SimMode::GameOfLife => self.life.step(),
        }
        self.step_count += 1;
        self.mark_cpu_texture_dirty();
    }

    fn render_active(&mut self) -> (usize, usize) {
        let (w, h) = self.active_size();
        let required = w * h * 4;
        if self.pixels.len() != required {
            self.pixels.resize(required, 0);
        }
        match self.mode {
            SimMode::Lenia => self.lenia.render_rgba(self.render_style, &mut self.pixels),
            SimMode::ReactionDiffusion => self
                .reaction
                .render_rgba(self.render_style, &mut self.pixels),
            SimMode::GameOfLife => self.life.render_rgba(self.render_style, &mut self.pixels),
        }
        (w, h)
    }

    fn export_snapshot(&mut self) {
        self.update_performance_metadata();
        if self.gpu_lenia_active() {
            self.export_gpu_lenia_snapshot();
            return;
        }

        let (w, h) = self.render_active();
        let stem = format!(
            "peterMath_{:?}_seed{}_step{}",
            self.mode,
            self.active_seed(),
            self.step_count
        );
        let png_path = format!("{}_snapshot.png", stem);
        let json_path = format!("{}_parameters.json", stem);
        let metrics = self.active_metrics();
        let result = (|| -> anyhow::Result<()> {
            export::save_png(&png_path, w, h, &self.pixels)?;
            export::save_json(
                &json_path,
                export::SnapshotExport {
                    mode: self.mode.label(),
                    render_style: self.render_style.label(),
                    backend: self.backend_label(),
                    seed: self.active_seed(),
                    step_count: self.step_count,
                    grid_width: w,
                    grid_height: h,
                    parameters: self.parameter_json(),
                    metrics,
                },
            )?;
            Ok(())
        })();
        self.status = match result {
            Ok(()) => format!("已导出 {} 和 {}", png_path, json_path),
            Err(err) => format!("导出失败：{err}"),
        };
    }

    fn gpu_lenia_export_state(&self, gpu: &GpuLeniaArt) -> anyhow::Result<GpuLeniaExportState> {
        let (size, field, previous) = gpu.read_fields_blocking()?;
        let mut pixels = vec![0; size * size * 4];
        gpu::colorize_fields(&field, &previous, size, self.render_style, &mut pixels);
        let metrics = Metrics::from_scalar_grid(&field, Some(&previous), size, size);
        let active_region = analysis::active_region_from_scalar_grid(
            &field,
            size,
            size,
            0.08,
            self.previous_centroid(),
        );
        let mass_trend = self
            .previous_metric_history(metrics)
            .map(|previous| metrics.mass - previous.mass)
            .unwrap_or_default();
        let phase = self.population_phase_from(
            LeniaPhase::from_metrics(metrics, mass_trend).label(),
            metrics,
            active_region,
        );
        let parameters = self.lenia_parameter_json(active_region, phase);

        Ok(GpuLeniaExportState {
            size,
            pixels,
            metrics,
            parameters,
        })
    }

    fn export_gpu_lenia_snapshot(&mut self) {
        self.update_performance_metadata();
        let Some(gpu) = &self.gpu_lenia else {
            self.status = "GPU 导出失败：GPU Lenia 不可用。".to_owned();
            return;
        };

        let stem = format!(
            "peterMath_GpuLenia_seed{}_step{}",
            self.active_seed(),
            self.step_count
        );
        let png_path = format!("{}_snapshot.png", stem);
        let json_path = format!("{}_parameters.json", stem);
        let result = (|| -> anyhow::Result<()> {
            let export_state = self.gpu_lenia_export_state(gpu)?;
            export::save_png(
                &png_path,
                export_state.size,
                export_state.size,
                &export_state.pixels,
            )?;
            export::save_json(
                &json_path,
                export::SnapshotExport {
                    mode: self.mode.label(),
                    render_style: self.render_style.label(),
                    backend: self.backend_label(),
                    seed: self.active_seed(),
                    step_count: self.step_count,
                    grid_width: export_state.size,
                    grid_height: export_state.size,
                    parameters: export_state.parameters,
                    metrics: export_state.metrics,
                },
            )?;
            Ok(())
        })();
        self.status = match result {
            Ok(()) => format!("已导出 {} 和 {}", png_path, json_path),
            Err(err) => format!("GPU 导出失败：{err}"),
        };
    }

    fn export_share_state(&mut self) {
        self.update_performance_metadata();
        let result = (|| -> anyhow::Result<()> {
            let (w, h, metrics, parameters) = if self.gpu_lenia_active() {
                let gpu = self
                    .gpu_lenia
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GPU Lenia is unavailable"))?;
                let export_state = self.gpu_lenia_export_state(gpu)?;
                (
                    export_state.size,
                    export_state.size,
                    export_state.metrics,
                    export_state.parameters,
                )
            } else {
                let (w, h) = self.active_size();
                (w, h, self.active_metrics(), self.parameter_json())
            };
            export::save_share_state(
                "peterMath_share_state.json",
                export::ShareStateExport {
                    mode: self.mode.label(),
                    render_style: self.render_style.label(),
                    backend: self.backend_label(),
                    seed: self.active_seed(),
                    step_count: self.step_count,
                    grid_width: w,
                    grid_height: h,
                    parameters,
                    metrics,
                },
            )
        })();
        self.status = match result {
            Ok(()) => "已导出 peterMath_share_state.json。".to_owned(),
            Err(err) => format!("可复现状态导出失败：{err}"),
        };
    }

    fn export_evidence_pack(&mut self) {
        self.update_performance_metadata();
        let stem = format!(
            "peterMath_{}_seed{}_step{}",
            self.mode.label().replace([' ', '-'], "_"),
            self.active_seed(),
            self.step_count
        );
        let dir = format!(
            "peterMath_exports/evidence_seed{}_step{}",
            self.active_seed(),
            self.step_count
        );

        let result = (|| -> anyhow::Result<export::EvidencePack> {
            let (w, h, pixels, metrics, parameters) = if self.gpu_lenia_active() {
                let gpu = self
                    .gpu_lenia
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GPU Lenia is unavailable"))?;
                let export_state = self.gpu_lenia_export_state(gpu)?;
                (
                    export_state.size,
                    export_state.size,
                    export_state.pixels,
                    export_state.metrics,
                    export_state.parameters,
                )
            } else {
                let (w, h) = self.render_active();
                (
                    w,
                    h,
                    self.pixels.clone(),
                    self.active_metrics(),
                    self.parameter_json(),
                )
            };

            export::create_evidence_pack(
                &dir,
                &stem,
                w,
                h,
                &pixels,
                export::ShareStateExport {
                    mode: self.mode.label(),
                    render_style: self.render_style.label(),
                    backend: self.backend_label(),
                    seed: self.active_seed(),
                    step_count: self.step_count,
                    grid_width: w,
                    grid_height: h,
                    parameters,
                    metrics,
                },
            )
        })();

        self.status = match result {
            Ok(pack) => format!(
                "已导出证据包：{}（PNG {}，JSON {}，状态 {}，摘要 {}）",
                pack.dir.display(),
                pack.snapshot_png.display(),
                pack.parameters_json.display(),
                pack.share_state_json.display(),
                pack.summary_md.display()
            ),
            Err(err) => format!("证据包导出失败：{err}"),
        };
    }

    fn gpu_lenia_active(&self) -> bool {
        self.mode == SimMode::Lenia && self.prefer_gpu_lenia && self.gpu_lenia.is_some()
    }

    fn backend_label(&self) -> &'static str {
        if self.gpu_lenia_active() {
            "GPU Lenia"
        } else {
            "CPU 参考"
        }
    }

    fn mark_cpu_texture_dirty(&mut self) {
        self.cpu_texture_dirty = true;
    }

    fn update_performance_metadata(&mut self) {
        self.performance.source_grid = match self.mode {
            SimMode::Lenia => self.lenia.size(),
            SimMode::ReactionDiffusion => self.reaction.size(),
            SimMode::GameOfLife => self.life.size(),
        };
        self.performance.gpu_grid = self.gpu_lenia.as_ref().map(|gpu| gpu.size() as usize);
        self.performance.pending_steps = self
            .gpu_lenia
            .as_ref()
            .map(|gpu| gpu.pending_steps())
            .unwrap_or_default();
        self.performance.cpu_sync_interval = self.gpu_cpu_sync_interval;
    }

    fn sync_gpu_lenia_from_cpu(&self) {
        if let Some(gpu) = &self.gpu_lenia {
            let (w, h) = self.lenia.size();
            gpu.reset_from_cpu(
                self.lenia.field(),
                self.lenia.previous_field(),
                (w, h),
                self.lenia.kernel_entries(),
                lenia_params(&self.lenia),
                self.render_style,
            );
        }
    }

    fn capture_lenia_history(&self) -> LeniaHistorySnapshot {
        LeniaHistorySnapshot {
            state: self.lenia.snapshot(),
            step_count: self.step_count,
            active_preset: self.active_preset,
            grid_profile: self.grid_profile,
            random_density: self.random_density,
        }
    }

    fn push_lenia_history(&mut self) {
        if self.mode != SimMode::Lenia {
            return;
        }
        self.undo_stack.push(self.capture_lenia_history());
        if self.undo_stack.len() > HISTORY_LIMIT {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    fn restore_lenia_history(&mut self, snapshot: LeniaHistorySnapshot) {
        self.lenia.restore(&snapshot.state);
        self.step_count = snapshot.step_count;
        self.active_preset = snapshot.active_preset;
        self.grid_profile = snapshot.grid_profile;
        self.random_density = snapshot.random_density;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
    }

    fn undo_lenia(&mut self) {
        let Some(snapshot) = self.undo_stack.pop() else {
            self.status = "没有可撤销的 Lenia 状态。".to_owned();
            return;
        };
        self.redo_stack.push(self.capture_lenia_history());
        if self.redo_stack.len() > HISTORY_LIMIT {
            self.redo_stack.remove(0);
        }
        self.restore_lenia_history(snapshot);
        self.status = "已恢复上一个 Lenia 场状态。".to_owned();
    }

    fn redo_lenia(&mut self) {
        let Some(snapshot) = self.redo_stack.pop() else {
            self.status = "没有可重做的 Lenia 状态。".to_owned();
            return;
        };
        self.undo_stack.push(self.capture_lenia_history());
        if self.undo_stack.len() > HISTORY_LIMIT {
            self.undo_stack.remove(0);
        }
        self.restore_lenia_history(snapshot);
        self.status = "已重做 Lenia 场状态。".to_owned();
    }

    fn load_lenia_preset(&mut self, preset: LeniaPreset) {
        if self.active_preset == preset {
            return;
        }
        self.push_lenia_history();
        self.active_preset = preset;
        self.lenia.reset_preset(preset.id());
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = format!("已载入 Lenia 预设：{}。", preset.label());
        self.last_interaction = format!("Lenia 预设切换为{}。", preset.label());
    }

    fn load_reaction_preset(&mut self, preset: ReactionPreset) {
        if self.active_reaction_preset == preset {
            return;
        }
        self.active_reaction_preset = preset;
        self.reaction.reset_preset(preset.id());
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.status = format!("已载入反应扩散预设：{}。", preset.label());
        self.last_interaction = format!("反应扩散预设切换为{}。", preset.label());
    }

    fn load_life_preset(&mut self, preset: LifePreset) {
        if self.active_life_preset == preset {
            return;
        }
        self.active_life_preset = preset;
        self.life.reset_preset(preset.id());
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.status = format!("已载入生命游戏预设：{}。", preset.label());
        self.last_interaction = format!("生命游戏预设切换为{}。", preset.label());
    }

    fn apply_grid_profile(&mut self, profile: GridProfile) {
        if self.grid_profile == profile {
            return;
        }
        self.push_lenia_history();
        self.grid_profile = profile;
        let size = profile.size();
        self.lenia.resize(size, size);
        self.lenia.reset_preset(self.active_preset.id());
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = format!("网格精度已切换为 {}。", profile.label());
    }

    fn randomize_lenia_field(&mut self) {
        self.push_lenia_history();
        let seed = self
            .lenia
            .seed
            .wrapping_mul(2_862_933_555_777_941_757)
            .wrapping_add(3_037_000_493);
        self.lenia.randomize_density(seed, self.random_density);
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = format!(
            "已按密度 {:.2} 随机化 Lenia 场，种子 {seed}。",
            self.random_density
        );
    }

    fn reset_lenia_with_history(&mut self) {
        self.push_lenia_history();
        self.reset_active();
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.status = format!("已重置 Lenia 预设：{}。", self.active_preset.label());
    }

    fn step_once(&mut self) {
        if self.gpu_lenia_active() {
            if let Some(gpu) = &self.gpu_lenia {
                gpu.update_params(lenia_params(&self.lenia), self.render_style);
                gpu.queue_steps(1);
            }
            self.lenia.step();
            self.step_count += 1;
            self.mark_cpu_texture_dirty();
        } else {
            self.step_active();
        }
        self.refresh_lenia_inspection();
        self.record_metric_history();
    }

    fn change_brush_radius(&mut self, delta: f32) {
        self.brush_radius = (self.brush_radius + delta).clamp(1.0, 32.0);
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }

        let shortcuts = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::Space),
                input.key_pressed(egui::Key::Period),
                input.key_pressed(egui::Key::R),
                input.key_pressed(egui::Key::C),
                input.key_pressed(egui::Key::N),
                input.key_pressed(egui::Key::Z) && !input.modifiers.shift,
                input.key_pressed(egui::Key::Z) && input.modifiers.shift,
                input.key_pressed(egui::Key::D),
                input.key_pressed(egui::Key::E),
                input.key_pressed(egui::Key::S),
                input.key_pressed(egui::Key::OpenBracket),
                input.key_pressed(egui::Key::CloseBracket),
            )
        });

        if shortcuts.0 {
            self.running = !self.running;
        }
        if shortcuts.1 {
            self.step_once();
        }
        if shortcuts.2 {
            if self.mode == SimMode::Lenia {
                self.reset_lenia_with_history();
            } else {
                self.reset_active();
            }
        }
        if shortcuts.4 {
            self.new_active_seed();
        }
        if shortcuts.7 {
            self.tool = InteractionTool::Draw;
        }
        if shortcuts.8 {
            self.tool = InteractionTool::Erase;
        }
        if self.mode == SimMode::Lenia {
            if shortcuts.3 {
                self.clear_lenia_field();
            }
            if shortcuts.5 {
                self.undo_lenia();
            }
            if shortcuts.6 {
                self.redo_lenia();
            }
            if shortcuts.9 {
                self.tool = InteractionTool::Stamp;
            }
            if shortcuts.10 {
                self.change_brush_radius(-1.0);
            }
            if shortcuts.11 {
                self.change_brush_radius(1.0);
            }
        }
    }

    fn clear_lenia_field(&mut self) {
        self.push_lenia_history();
        self.lenia.clear();
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = "已清空 Lenia 场；可绘制或选择新种子继续。".to_owned();
    }

    fn new_lenia_seed(&mut self) {
        self.push_lenia_history();
        let next_seed = self
            .lenia
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.lenia.reseed(next_seed);
        self.lenia.reset_preset(self.active_preset.id());
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = format!("已载入确定性 Lenia 种子 {next_seed}。");
    }

    fn new_active_seed(&mut self) {
        match self.mode {
            SimMode::Lenia => self.new_lenia_seed(),
            SimMode::ReactionDiffusion => {
                let next_seed = next_seed(self.reaction.seed);
                self.reaction.seed = next_seed;
                self.reaction.reset_preset(self.active_reaction_preset.id());
                self.step_count = 0;
                self.texture = None;
                self.mark_cpu_texture_dirty();
                self.reset_metric_history();
                self.last_interaction = format!("反应扩散载入新确定性种子 {next_seed}。");
                self.status = "已重新计算当前规则：反应扩散新种子生效。".to_owned();
            }
            SimMode::GameOfLife => {
                let next_seed = next_seed(self.life.seed);
                self.life.seed = next_seed;
                self.active_life_preset = LifePreset::RandomSoup;
                self.life.reset_random();
                self.step_count = 0;
                self.texture = None;
                self.mark_cpu_texture_dirty();
                self.reset_metric_history();
                self.last_interaction = format!("生命游戏载入随机汤种子 {next_seed}。");
                self.status = "已重新计算当前规则：生命游戏新种子生效。".to_owned();
            }
        }
    }

    fn add_life_glider(&mut self) {
        let (w, h) = self.life.size();
        self.life.stamp_glider(w as f32 * 0.50, h as f32 * 0.50);
        self.step_count = 0;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.last_interaction = "现场添加滑翔机；它会按 B3/S23 斜向移动。".to_owned();
        self.status = "已重新计算当前规则：滑翔机已加入。".to_owned();
    }

    fn add_life_oscillator(&mut self) {
        let (w, h) = self.life.size();
        self.life.stamp_oscillator(w as f32 * 0.50, h as f32 * 0.50);
        self.step_count = 0;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.last_interaction = "现场添加振荡器；它会在几个状态之间来回变化。".to_owned();
        self.status = "已重新计算当前规则：振荡器已加入。".to_owned();
    }

    fn add_reaction_perturbation(&mut self) {
        let (w, h) = self.reaction.size();
        self.reaction.paint_brush(
            w as f32 * 0.50,
            h as f32 * 0.50,
            self.reaction_brush_radius * 1.4,
            self.reaction_brush_strength,
            true,
        );
        self.step_count = 0;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.last_interaction = "现场注入 B 物质扰动；迷宫/波纹会继续生长。".to_owned();
        self.status = "已重新计算当前规则：反应扩散扰动已加入。".to_owned();
    }

    fn import_life_rle(&mut self) {
        match LifeRlePattern::parse(&self.life_rle_input) {
            Ok(pattern) => {
                self.life.apply_rle_centered(&pattern);
                self.step_count = 0;
                self.texture = None;
                self.mark_cpu_texture_dirty();
                self.reset_metric_history();
                self.status = format!(
                    "已导入生命游戏 RLE 图案：{}x{}，活细胞 {} 个。",
                    pattern.width,
                    pattern.height,
                    pattern.cells.len()
                );
            }
            Err(err) => {
                self.status = format!("RLE 导入失败：{err}");
            }
        }
    }

    fn export_life_rle(&mut self) {
        self.life_rle_output = self.life.export_rle();
        self.status = "已将当前生命游戏活跃边界导出为 RLE。".to_owned();
    }

    fn clear_comparison_result(&mut self) {
        self.comparison_result = None;
        self.comparison_baseline_texture = None;
        self.comparison_variant_texture = None;
    }

    fn capture_comparison_baseline(&mut self) {
        self.comparison_baseline = Some(self.lenia.snapshot());
        self.comparison_value = self.comparison_parameter.current_value(&self.lenia);
        self.clear_comparison_result();
        self.status = "已记录 Lenia 规则对照基线。".to_owned();
    }

    fn apply_variant_to_current_lenia(&mut self) {
        self.push_lenia_history();
        self.comparison_parameter
            .apply(&mut self.lenia, self.comparison_value);
        self.step_count = 0;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = format!(
            "已应用 Lenia 参数变量：{} = {:.4}。",
            self.comparison_parameter.label(),
            self.comparison_value
        );
    }

    fn run_rule_variant_comparison(&mut self) {
        let Some(baseline_state) = &self.comparison_baseline else {
            self.status = "请先记录 Lenia 基线，再运行对照。".to_owned();
            return;
        };

        let mut baseline = LeniaSim::from_state(baseline_state);
        let mut variant = LeniaSim::from_state(baseline_state);
        self.comparison_parameter
            .apply(&mut variant, self.comparison_value);
        for _ in 0..self.comparison_steps {
            baseline.step();
            variant.step();
        }
        let baseline_metrics = baseline.metrics();
        let variant_metrics = variant.metrics();
        let (w, h) = baseline.size();
        let mut baseline_pixels = vec![0; w * h * 4];
        let mut variant_pixels = vec![0; w * h * 4];
        baseline.render_rgba(RenderStyle::Artistic, &mut baseline_pixels);
        variant.render_rgba(RenderStyle::Artistic, &mut variant_pixels);
        self.comparison_result = Some(RuleVariantComparison {
            parameter: self.comparison_parameter,
            value: self.comparison_value,
            steps: self.comparison_steps,
            baseline_metrics,
            variant_metrics,
            width: w,
            height: h,
            baseline_pixels,
            variant_pixels,
        });
        self.comparison_baseline_texture = None;
        self.comparison_variant_texture = None;
        self.status = "已运行 CPU Lenia 规则变量对照。".to_owned();
    }

    fn apply_lenia_brush(&mut self, rect: egui::Rect, response: &egui::Response) {
        if self.mode != SimMode::Lenia {
            return;
        }
        if self.tool == InteractionTool::Pan {
            return;
        }
        if !(response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary))
        {
            return;
        }
        if self.tool == InteractionTool::Stamp && !response.clicked_by(egui::PointerButton::Primary)
        {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if !rect.contains(pos) {
            return;
        }

        let (w, h) = self.lenia.size();
        let x = ((pos.x - rect.min.x) / rect.width() * w as f32).clamp(0.0, w as f32 - 1.0);
        let y = ((pos.y - rect.min.y) / rect.height() * h as f32).clamp(0.0, h as f32 - 1.0);
        if !self.pointer_edit_active {
            self.push_lenia_history();
            self.pointer_edit_active = true;
        }

        match self.tool {
            InteractionTool::Draw => {
                self.lenia
                    .paint_brush(x, y, self.brush_radius, self.brush_strength);
            }
            InteractionTool::Erase => {
                self.lenia
                    .erase_brush(x, y, self.brush_radius, self.brush_strength);
            }
            InteractionTool::Stamp => {
                if response.clicked_by(egui::PointerButton::Primary) {
                    self.lenia.apply_stamp(
                        x,
                        y,
                        self.active_stamp.id(),
                        self.brush_radius,
                        self.brush_strength,
                    );
                }
            }
            InteractionTool::Pan => {}
        }
        self.sync_gpu_lenia_from_cpu();
        self.mark_cpu_texture_dirty();
        self.refresh_lenia_inspection();
        self.record_metric_history();
        self.last_interaction = "Lenia 画布已被现场编辑，GPU/CPU 状态已重新同步。".to_owned();
        self.status = "已重新计算当前规则：Lenia 场编辑生效。".to_owned();
    }

    fn apply_life_brush(&mut self, rect: egui::Rect, response: &egui::Response) {
        if self.mode != SimMode::GameOfLife || self.tool == InteractionTool::Pan {
            return;
        }
        if !(response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary))
        {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let Some((x, y)) = Self::canvas_grid_pos(rect, pos, self.life.size()) else {
            return;
        };

        let started_edit = !self.pointer_edit_active;
        self.pointer_edit_active = true;
        let alive = self.tool != InteractionTool::Erase;
        self.life.paint_brush(x, y, self.life_brush_radius, alive);
        self.mark_cpu_texture_dirty();
        if started_edit {
            self.reset_metric_history();
        } else {
            self.record_metric_history();
        }
        self.last_interaction = if alive {
            "生命游戏细胞笔添加了活细胞，下一步会按 B3/S23 重新演化。".to_owned()
        } else {
            "生命游戏细胞笔擦除了细胞，下一步会按 B3/S23 重新演化。".to_owned()
        };
        self.status = "已重新计算当前规则：生命游戏细胞编辑生效。".to_owned();
    }

    fn apply_reaction_brush(&mut self, rect: egui::Rect, response: &egui::Response) {
        if self.mode != SimMode::ReactionDiffusion || self.tool == InteractionTool::Pan {
            return;
        }
        if !(response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary))
        {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let Some((x, y)) = Self::canvas_grid_pos(rect, pos, self.reaction.size()) else {
            return;
        };

        let started_edit = !self.pointer_edit_active;
        self.pointer_edit_active = true;
        let inject_b = self.tool != InteractionTool::Erase;
        self.reaction.paint_brush(
            x,
            y,
            self.reaction_brush_radius,
            self.reaction_brush_strength,
            inject_b,
        );
        self.mark_cpu_texture_dirty();
        if started_edit {
            self.reset_metric_history();
        } else {
            self.record_metric_history();
        }
        self.last_interaction = if inject_b {
            "反应扩散画笔注入了 B 物质，纹理会按当前 feed/kill 继续生长。".to_owned()
        } else {
            "反应扩散画笔擦除了 B 物质，纹理会按当前 feed/kill 继续生长。".to_owned()
        };
        self.status = "已重新计算当前规则：反应扩散扰动生效。".to_owned();
    }

    fn apply_canvas_interaction(&mut self, rect: egui::Rect, response: &egui::Response) {
        self.update_lenia_inspection_from_canvas(rect, response);
        self.apply_lenia_brush(rect, response);
        self.apply_life_brush(rect, response);
        self.apply_reaction_brush(rect, response);
    }

    fn canvas_grid_pos(
        rect: egui::Rect,
        pos: egui::Pos2,
        size: (usize, usize),
    ) -> Option<(f32, f32)> {
        if !rect.contains(pos) {
            return None;
        }
        let (w, h) = size;
        let x = ((pos.x - rect.min.x) / rect.width() * w as f32).clamp(0.0, w as f32 - 1.0);
        let y = ((pos.y - rect.min.y) / rect.height() * h as f32).clamp(0.0, h as f32 - 1.0);
        Some((x, y))
    }

    fn update_lenia_inspection_from_canvas(&mut self, rect: egui::Rect, response: &egui::Response) {
        if self.mode != SimMode::Lenia {
            return;
        }
        let Some(pos) = response
            .hover_pos()
            .or_else(|| response.interact_pointer_pos())
        else {
            return;
        };
        if !rect.contains(pos) {
            return;
        }
        let (w, h) = self.lenia.size();
        let x = ((pos.x - rect.min.x) / rect.width() * w as f32).clamp(0.0, w as f32 - 1.0);
        let y = ((pos.y - rect.min.y) / rect.height() * h as f32).clamp(0.0, h as f32 - 1.0);
        self.inspected_lenia = Some(self.lenia.inspect_point(x as usize, y as usize));
    }

    fn draw_lenia_inspection_overlay(&self, painter: &egui::Painter, rect: egui::Rect) {
        if self.mode != SimMode::Lenia || !(self.show_kernel_overlay || self.judge_mode) {
            return;
        }
        let Some(inspection) = self.inspected_lenia else {
            return;
        };
        let (w, h) = self.lenia.size();
        let center = egui::pos2(
            rect.min.x + (inspection.x as f32 + 0.5) / w as f32 * rect.width(),
            rect.min.y + (inspection.y as f32 + 0.5) / h as f32 * rect.height(),
        );
        let radius = self.lenia.radius as f32 / w as f32 * rect.width();
        let stroke = egui::Stroke::new(1.2, Color32::from_rgba_unmultiplied(120, 238, 224, 170));
        painter.circle_stroke(center, radius, stroke);
        painter.circle_filled(center, 3.0, Color32::from_rgb(255, 118, 168));
        painter.line_segment(
            [
                egui::pos2(center.x - 8.0, center.y),
                egui::pos2(center.x + 8.0, center.y),
            ],
            egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 118, 168, 150)),
        );
        painter.line_segment(
            [
                egui::pos2(center.x, center.y - 8.0),
                egui::pos2(center.x, center.y + 8.0),
            ],
            egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 118, 168, 150)),
        );
    }

    fn draw_life_grid_overlay(&self, painter: &egui::Painter, rect: egui::Rect) {
        if self.mode != SimMode::GameOfLife {
            return;
        }
        let (w, h) = self.life.size();
        if w > 128 || h > 128 {
            return;
        }
        let stroke = egui::Stroke::new(0.35, Color32::from_rgba_unmultiplied(80, 108, 116, 70));
        for x in 1..w {
            let px = egui::lerp(rect.left()..=rect.right(), x as f32 / w as f32);
            painter.line_segment(
                [egui::pos2(px, rect.top()), egui::pos2(px, rect.bottom())],
                stroke,
            );
        }
        for y in 1..h {
            let py = egui::lerp(rect.top()..=rect.bottom(), y as f32 / h as f32);
            painter.line_segment(
                [egui::pos2(rect.left(), py), egui::pos2(rect.right(), py)],
                stroke,
            );
        }
    }

    fn draw_active_region_overlay(&self, painter: &egui::Painter, rect: egui::Rect) {
        if !(self.show_active_region_overlay || self.judge_mode) {
            return;
        }
        let region = self.active_region();
        let Some((min_x, min_y, max_x, max_y)) = region.bounds else {
            return;
        };
        let (w, h) = match self.mode {
            SimMode::Lenia => self.lenia.size(),
            SimMode::ReactionDiffusion => self.reaction.size(),
            SimMode::GameOfLife => self.life.size(),
        };
        let left = rect.min.x + min_x as f32 / w as f32 * rect.width();
        let right = rect.min.x + (max_x + 1) as f32 / w as f32 * rect.width();
        let top = rect.min.y + min_y as f32 / h as f32 * rect.height();
        let bottom = rect.min.y + (max_y + 1) as f32 / h as f32 * rect.height();
        let bounds_rect =
            egui::Rect::from_min_max(egui::pos2(left, top), egui::pos2(right, bottom));
        painter.rect_stroke(
            bounds_rect,
            0.0,
            egui::Stroke::new(1.2, Color32::from_rgba_unmultiplied(216, 240, 139, 170)),
            egui::StrokeKind::Outside,
        );
        if let Some((cx, cy)) = region.centroid {
            let center = egui::pos2(
                rect.min.x + (cx + 0.5) / w as f32 * rect.width(),
                rect.min.y + (cy + 0.5) / h as f32 * rect.height(),
            );
            painter.circle_filled(center, 3.2, Color32::from_rgb(216, 240, 139));
        }
    }

    fn lenia_inspection_json(&self) -> serde_json::Value {
        let Some(inspection) = self.inspected_lenia else {
            return serde_json::Value::Null;
        };
        json!({
            "x": inspection.x,
            "y": inspection.y,
            "field_value": inspection.value,
            "previous_value": inspection.previous,
            "delta": inspection.delta,
            "gradient": inspection.gradient,
            "kernel_convolution": inspection.convolution,
            "growth_response": inspection.growth,
            "estimated_next": inspection.estimated_next,
        })
    }

    fn metric_history_summary_json(&self) -> serde_json::Value {
        let Some(latest) = self.metric_history.last() else {
            return serde_json::Value::Null;
        };

        let mut min_mass = f32::INFINITY;
        let mut max_mass = f32::NEG_INFINITY;
        let mut min_entropy = f32::INFINITY;
        let mut max_entropy = f32::NEG_INFINITY;
        let mut min_stability = f32::INFINITY;
        let mut max_stability = f32::NEG_INFINITY;
        let mut min_vitality = f32::INFINITY;
        let mut max_vitality = f32::NEG_INFINITY;

        for sample in &self.metric_history {
            min_mass = min_mass.min(sample.mass);
            max_mass = max_mass.max(sample.mass);
            min_entropy = min_entropy.min(sample.entropy);
            max_entropy = max_entropy.max(sample.entropy);
            min_stability = min_stability.min(sample.stability);
            max_stability = max_stability.max(sample.stability);
            min_vitality = min_vitality.min(sample.vitality);
            max_vitality = max_vitality.max(sample.vitality);
        }

        json!({
            "samples": self.metric_history.len(),
            "latest_step": latest.step_count,
            "mass": {"latest": latest.mass, "min": min_mass, "max": max_mass},
            "entropy": {"latest": latest.entropy, "min": min_entropy, "max": max_entropy},
            "stability": {"latest": latest.stability, "min": min_stability, "max": max_stability},
            "vitality": {"latest": latest.vitality, "min": min_vitality, "max": max_vitality},
        })
    }

    fn performance_json(&self) -> serde_json::Value {
        json!({
            "fps_estimate": self.performance.fps_estimate,
            "frame_ms": self.performance.latest.frame_ms,
            "update_ms": self.performance.latest.update_ms,
            "render_upload_ms": self.performance.latest.render_ms,
            "cpu_sync_ms": self.performance.latest.cpu_sync_ms,
            "backend": self.backend_label(),
            "source_grid": {
                "width": self.performance.source_grid.0,
                "height": self.performance.source_grid.1,
            },
            "gpu_grid": self.performance.gpu_grid,
            "pending_gpu_steps": self.performance.pending_steps,
            "cpu_sync_interval": self.performance.cpu_sync_interval,
            "frame_samples": self.performance.frame_samples,
        })
    }

    fn active_region_json(&self) -> serde_json::Value {
        Self::active_region_value(self.active_region())
    }

    fn active_region_value(region: ActiveRegionAnalysis) -> serde_json::Value {
        json!({
            "active_count": region.active_count,
            "bounds": region.bounds.map(|(min_x, min_y, max_x, max_y)| {
                json!({"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y})
            }),
            "centroid": region.centroid.map(|(x, y)| json!({"x": x, "y": y})),
            "area_ratio": region.area_ratio,
            "drift": {"x": region.drift.0, "y": region.drift.1},
        })
    }

    fn population_phase_json(&self) -> serde_json::Value {
        Self::population_phase_value(self.population_phase_analysis())
    }

    fn population_phase_value(phase: PopulationPhaseAnalysis) -> serde_json::Value {
        json!({
            "label": phase.label,
            "mass_trend": phase.mass_trend,
            "entropy_trend": phase.entropy_trend,
            "stability_trend": phase.stability_trend,
            "vitality_trend": phase.vitality_trend,
            "centroid_drift": {"x": phase.centroid_drift.0, "y": phase.centroid_drift.1},
        })
    }

    fn lenia_parameter_json(
        &self,
        active_region: ActiveRegionAnalysis,
        phase_analysis: PopulationPhaseAnalysis,
    ) -> serde_json::Value {
        json!({
            "schema_version": export::SCHEMA_VERSION,
            "system_id": SimMode::Lenia.id(),
            "overview_focus": self.overview_focus.id(),
            "display_name_zh": SimMode::Lenia.label_zh(),
            "explanation_zh": self.mode_significance(),
            "child_explanation_zh": mode_child_explanation(SimMode::Lenia),
            "formula_ascii": formula_rows_json(SimMode::Lenia),
            "last_interaction_zh": self.last_interaction,
            "stage_zh": phase_label_zh(phase_analysis.label),
            "render_style_id": self.render_style.id(),
            "render_style_zh": self.render_style.label_zh(),
            "kernel_radius": self.lenia.radius,
            "growth_center": self.lenia.growth_center,
            "growth_width": self.lenia.growth_width,
            "time_step": self.lenia.dt,
            "damping": self.lenia.decay,
            "backend": self.backend_label(),
            "active_tool": self.tool.id(),
            "active_preset": self.active_preset.id(),
            "active_stamp": self.active_stamp.id(),
            "brush_radius": self.brush_radius,
            "brush_strength": self.brush_strength,
            "random_density": self.random_density,
            "grid_profile": self.grid_profile.label(),
            "phase_label": phase_analysis.label,
            "inspected_point": self.lenia_inspection_json(),
            "metric_history": self.metric_history_summary_json(),
            "performance": self.performance_json(),
            "active_region": Self::active_region_value(active_region),
            "phase_analysis": Self::population_phase_value(phase_analysis),
            "rule_variant_comparison": self.comparison_json(),
            "source_grid": {
                "width": self.lenia.size().0,
                "height": self.lenia.size().1,
            },
        })
    }

    fn pattern_detection_json(&self) -> serde_json::Value {
        if self.mode != SimMode::GameOfLife {
            return serde_json::Value::Null;
        }
        let report = self.life_pattern_report();
        json!({
            "oscillator_period": report.oscillator_period,
            "detections": report.detections.iter().map(|detection| json!({
                "pattern": detection.pattern.label(),
                "kind": detection.pattern.kind(),
                "x": detection.x,
                "y": detection.y,
                "width": detection.width,
                "height": detection.height,
            })).collect::<Vec<_>>(),
            "glider_track": report.glider_track.map(|track| json!({
                "count": track.count,
                "centroid": track.centroid.map(|(x, y)| json!({"x": x, "y": y})),
                "direction": track.direction.map(|(x, y)| json!({"x": x, "y": y})),
            })),
        })
    }

    fn comparison_json(&self) -> serde_json::Value {
        let Some(comparison) = &self.comparison_result else {
            return serde_json::Value::Null;
        };
        json!({
            "parameter": comparison.parameter.id(),
            "value": comparison.value,
            "steps": comparison.steps,
            "baseline": metrics_json(comparison.baseline_metrics),
            "variant": metrics_json(comparison.variant_metrics),
            "delta": {
                "mass": comparison.variant_metrics.mass - comparison.baseline_metrics.mass,
                "entropy": comparison.variant_metrics.entropy - comparison.baseline_metrics.entropy,
                "stability": comparison.variant_metrics.stability - comparison.baseline_metrics.stability,
                "vitality": comparison.variant_metrics.vitality - comparison.baseline_metrics.vitality,
            },
        })
    }

    fn parameter_json(&self) -> serde_json::Value {
        match self.mode {
            SimMode::Lenia => {
                self.lenia_parameter_json(self.active_region(), self.population_phase_analysis())
            }
            SimMode::ReactionDiffusion => json!({
                "schema_version": export::SCHEMA_VERSION,
                "system_id": SimMode::ReactionDiffusion.id(),
                "overview_focus": self.overview_focus.id(),
                "display_name_zh": SimMode::ReactionDiffusion.label_zh(),
                "explanation_zh": self.mode_significance(),
                "child_explanation_zh": mode_child_explanation(SimMode::ReactionDiffusion),
                "formula_ascii": formula_rows_json(SimMode::ReactionDiffusion),
                "last_interaction_zh": self.last_interaction,
                "stage_zh": self.active_stage_zh(),
                "render_style_id": self.render_style.id(),
                "render_style_zh": self.render_style.label_zh(),
                "active_preset": self.active_reaction_preset.id(),
                "feed": self.reaction.feed,
                "kill": self.reaction.kill,
                "diffusion_a": self.reaction.diff_a,
                "diffusion_b": self.reaction.diff_b,
                "time_step": self.reaction.dt,
                "performance": self.performance_json(),
                "active_region": self.active_region_json(),
                "phase_analysis": self.population_phase_json(),
            }),
            SimMode::GameOfLife => json!({
                "schema_version": export::SCHEMA_VERSION,
                "system_id": SimMode::GameOfLife.id(),
                "overview_focus": self.overview_focus.id(),
                "display_name_zh": SimMode::GameOfLife.label_zh(),
                "explanation_zh": self.mode_significance(),
                "child_explanation_zh": mode_child_explanation(SimMode::GameOfLife),
                "formula_ascii": formula_rows_json(SimMode::GameOfLife),
                "last_interaction_zh": self.last_interaction,
                "stage_zh": self.active_stage_zh(),
                "render_style_id": self.render_style.id(),
                "render_style_zh": self.render_style.label_zh(),
                "rule": "B3/S23",
                "active_preset": self.active_life_preset.id(),
                "seed_density": self.life.random_density,
                "rle_export": self.life.export_rle(),
                "performance": self.performance_json(),
                "active_region": self.active_region_json(),
                "phase_analysis": self.population_phase_json(),
                "pattern_detection": self.pattern_detection_json(),
            }),
        }
    }

    fn enter_experiment(&mut self, mode: SimMode) {
        self.screen = AppScreen::Experiment;
        self.mode = mode;
        self.steps_per_frame = Self::recommended_steps_for(mode);
        self.step_count = 0;
        self.texture = None;
        self.tick_accumulator = Duration::ZERO;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.refresh_lenia_inspection();
        self.overview_focus = mode;
        self.last_interaction = format!("从中文总览进入{}实时实验。", mode.label_zh());
        self.status = format!(
            "进入{}。右侧只保留常用预设，高级参数可展开。",
            mode.label_zh()
        );
    }

    fn update_overview_systems(&mut self) {
        if self.overview_step.is_multiple_of(4) {
            self.overview_life.step();
        }
        for _ in 0..8 {
            self.overview_reaction.step();
        }
        if self.overview_step.is_multiple_of(2) {
            self.overview_lenia.step();
        }
        self.overview_step = self.overview_step.saturating_add(1);
    }

    fn update_texture_from_pixels(
        ctx: &egui::Context,
        texture: &mut Option<TextureHandle>,
        name: &str,
        w: usize,
        h: usize,
        pixels: &[u8],
        options: TextureOptions,
    ) {
        let image = ColorImage::from_rgba_unmultiplied([w, h], pixels);
        if let Some(texture) = texture {
            texture.set(image, options);
        } else {
            *texture = Some(ctx.load_texture(name, image, options));
        }
    }

    fn refresh_overview_textures(&mut self, ctx: &egui::Context) {
        let mut pixels = Vec::new();

        let (w, h) = self.overview_life.size();
        pixels.resize(w * h * 4, 0);
        self.overview_life
            .render_rgba(RenderStyle::Artistic, &mut pixels);
        Self::update_texture_from_pixels(
            ctx,
            &mut self.overview_life_texture,
            "peterMath-overview-life",
            w,
            h,
            &pixels,
            TextureOptions::NEAREST,
        );

        let (w, h) = self.overview_reaction.size();
        pixels.resize(w * h * 4, 0);
        self.overview_reaction
            .render_rgba(RenderStyle::Artistic, &mut pixels);
        Self::update_texture_from_pixels(
            ctx,
            &mut self.overview_reaction_texture,
            "peterMath-overview-reaction",
            w,
            h,
            &pixels,
            TextureOptions::LINEAR,
        );

        let (w, h) = self.overview_lenia.size();
        pixels.resize(w * h * 4, 0);
        self.overview_lenia
            .render_rgba(RenderStyle::Artistic, &mut pixels);
        Self::update_texture_from_pixels(
            ctx,
            &mut self.overview_lenia_texture,
            "peterMath-overview-lenia",
            w,
            h,
            &pixels,
            TextureOptions::LINEAR,
        );
    }

    fn draw_overview(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.refresh_overview_textures(ctx);
        ui.painter()
            .rect_filled(ui.max_rect(), 0.0, Color32::from_rgb(7, 10, 12));
        ui.vertical_centered(|ui| {
            ui.add_space(12.0);
            ui.heading("三种数学生命系统导览");
            ui.label(
                "先看一个大画面，再切换系统。每一次变化都来自正在运行的规则，不是预渲染视频。",
            );
            ui.add_space(8.0);
        });

        ui.horizontal_centered(|ui| {
            for mode in [
                SimMode::GameOfLife,
                SimMode::ReactionDiffusion,
                SimMode::Lenia,
            ] {
                if ui
                    .selectable_label(self.overview_focus == mode, mode.label_zh())
                    .clicked()
                {
                    self.overview_focus = mode;
                    self.status = format!("总览切换到{}。", mode.label_zh());
                }
            }
        });

        ui.add_space(10.0);
        let focused_card = self.overview_card(self.overview_focus);
        let focused_texture_id = self
            .overview_texture(self.overview_focus)
            .map(|texture| texture.id());
        let body_height = (ui.available_height() - 210.0).clamp(430.0, 660.0);
        let (body_rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), body_height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(body_rect);
        painter.rect_filled(body_rect, 8.0, Color32::from_rgb(9, 13, 15));
        painter.rect_stroke(
            body_rect,
            8.0,
            egui::Stroke::new(1.0, Color32::from_rgb(34, 48, 54)),
            egui::StrokeKind::Inside,
        );

        let graph_side = (body_rect.height() - 28.0)
            .min(body_rect.width() * 0.58)
            .clamp(300.0, 620.0);
        let graph_rect = egui::Rect::from_min_size(
            body_rect.min + egui::vec2(14.0, 14.0),
            egui::vec2(graph_side, graph_side),
        );
        painter.rect_filled(graph_rect, 4.0, Color32::from_rgb(3, 6, 8));
        if let Some(texture_id) = focused_texture_id {
            painter.image(
                texture_id,
                graph_rect.shrink(4.0),
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            painter.text(
                graph_rect.center(),
                egui::Align2::CENTER_CENTER,
                "图像纹理正在加载",
                egui::FontId::proportional(18.0),
                Color32::from_rgb(220, 235, 235),
            );
        }
        painter.rect_stroke(
            graph_rect,
            4.0,
            egui::Stroke::new(2.0, Color32::from_rgb(94, 197, 255)),
            egui::StrokeKind::Inside,
        );

        let card_rect = egui::Rect::from_min_max(
            egui::pos2(graph_rect.right() + 18.0, graph_rect.top()),
            egui::pos2(body_rect.right() - 14.0, graph_rect.bottom()),
        );
        let mut enter = None;
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(card_rect)
                .layout(egui::Layout::top_down(egui::Align::Min)),
            |ui| {
                ui.set_width(card_rect.width());
                ui.set_max_width(card_rect.width());
                ui.set_min_height(card_rect.height());
                if Self::draw_overview_card(ui, focused_card, self.overview_focus) {
                    enter = Some(self.overview_focus);
                }
            },
        );

        ui.add_space(10.0);
        ui.label("切换对照：");
        ui.columns(3, |columns| {
            for (column, mode) in [
                SimMode::GameOfLife,
                SimMode::ReactionDiffusion,
                SimMode::Lenia,
            ]
            .into_iter()
            .enumerate()
            {
                if self.draw_overview_thumbnail(&mut columns[column], mode) {
                    self.overview_focus = mode;
                    self.status = format!("总览切换到{}。", mode.label_zh());
                }
            }
        });

        if let Some(mode) = enter {
            self.enter_experiment(mode);
        }
    }

    fn draw_overview_card(ui: &mut egui::Ui, card: OverviewCard<'_>, mode: SimMode) -> bool {
        let mut clicked = false;
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::same(14))
            .show(ui, |ui| {
                ui.set_min_width(360.0);
                ui.heading(card.title);
                Self::explanation_block(ui, "这是什么", mode_child_explanation(mode));
                Self::explanation_block(ui, "看什么", card.goal);
                Self::explanation_block(ui, "现在发生了什么", card.conclusion);
                Self::explanation_block(ui, "可以试一下", card.try_action);
                egui::CollapsingHeader::new("数学公式")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.small(card.plain_rule);
                        Self::draw_formula_card_for(ui, mode);
                    });
                ui.label(format!("阶段：{}", card.stage));
                ui.label(format!(
                    "活跃度 {:.3} · 最近变化 {:.3}",
                    card.metrics.mass,
                    1.0 - card.metrics.stability
                ));
                if ui.button("进入实验").clicked() {
                    clicked = true;
                }
            });
        clicked
    }

    fn draw_overview_thumbnail(&self, ui: &mut egui::Ui, mode: SimMode) -> bool {
        let Some(texture) = self.overview_texture(mode) else {
            return false;
        };
        let selected = self.overview_focus == mode;
        let side = ui.available_width().clamp(120.0, 168.0);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(side, side + 34.0), egui::Sense::click());
        let painter = ui.painter_at(rect);
        let image_rect = egui::Rect::from_min_size(rect.min, egui::vec2(side, side));
        painter.rect_filled(rect, 4.0, Color32::from_rgb(10, 14, 16));
        painter.image(
            texture.id(),
            image_rect.shrink(4.0),
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        let stroke_color = if selected {
            Color32::from_rgb(109, 235, 222)
        } else {
            Color32::from_rgb(46, 61, 66)
        };
        painter.rect_stroke(
            image_rect,
            4.0,
            egui::Stroke::new(if selected { 2.2 } else { 1.0 }, stroke_color),
            egui::StrokeKind::Inside,
        );
        painter.text(
            egui::pos2(rect.center().x, image_rect.bottom() + 18.0),
            egui::Align2::CENTER_CENTER,
            mode.label_zh(),
            egui::FontId::proportional(13.0),
            Color32::from_rgb(214, 226, 226),
        );
        response.clicked()
    }

    fn overview_texture(&self, mode: SimMode) -> Option<&TextureHandle> {
        match mode {
            SimMode::Lenia => self.overview_lenia_texture.as_ref(),
            SimMode::ReactionDiffusion => self.overview_reaction_texture.as_ref(),
            SimMode::GameOfLife => self.overview_life_texture.as_ref(),
        }
    }

    fn overview_seed(&self, mode: SimMode) -> u64 {
        match mode {
            SimMode::Lenia => self.overview_lenia.seed,
            SimMode::ReactionDiffusion => self.overview_reaction.seed,
            SimMode::GameOfLife => self.overview_life.seed,
        }
    }

    fn overview_card(&self, mode: SimMode) -> OverviewCard<'static> {
        match mode {
            SimMode::GameOfLife => {
                let metrics = self.overview_life.metrics();
                let report =
                    self.overview_life
                        .detect_known_patterns(&[], self.overview_step, None);
                let conclusion = if report.glider_track.is_some() {
                    "这里有会斜着移动的滑翔机，也有保持不变或来回闪烁的小结构。"
                } else if !report.detections.is_empty() {
                    "这里已经识别出稳定或周期结构：简单规则正在筛选能留下来的形状。"
                } else {
                    "细胞正在互相影响，随机区域会逐渐淘汰，只留下少数结构。"
                };
                OverviewCard {
                    title: "生命游戏 Game of Life",
                    plain_rule: "每个格子只看周围 8 个邻居。",
                    goal: "找出三种结果：不动的结构、闪烁的结构、会移动的滑翔机。",
                    try_action: "进入实验后暂停画面，用细胞笔画几格，再单步观察它们出生或消失。",
                    stage: if metrics.stability > 0.97 {
                        "周期/漂移"
                    } else {
                        "快速增长"
                    },
                    metrics,
                    conclusion,
                }
            }
            SimMode::ReactionDiffusion => {
                let metrics = self.overview_reaction.metrics();
                OverviewCard {
                    title: "反应扩散 Reaction-Diffusion",
                    plain_rule: "两种颜料会扩散，也会互相反应。",
                    goal: "观察斑点、波纹和迷宫边界怎样从小扰动长出来。",
                    try_action:
                        "进入实验后调 feed 或 kill，或者用画笔注入 B 物质，看纹理立即改变。",
                    stage: self.overview_reaction_stage(metrics),
                    metrics,
                    conclusion: "扩散速度差会把小点放大成空间纹理，像自然界的斑纹和波前。",
                }
            }
            SimMode::Lenia => {
                let metrics = self.overview_lenia.metrics();
                let phase = LeniaPhase::from_metrics(metrics, 0.0);
                OverviewCard {
                    title: "连续生命场 Lenia",
                    plain_rule: "每个点听周围一圈邻居的平均值，条件刚好时增长。",
                    goal: "观察连续场、卷积核和增长函数怎样形成柔性的生命形态。",
                    try_action: "进入实验后用画笔加一点质量，再改增长中心，比较形态是否继续活跃。",
                    stage: phase.label_zh(),
                    metrics,
                    conclusion: "同一份数值场既能显示规则，也能转成更有美感的颜色和轮廓。",
                }
            }
        }
    }

    fn overview_reaction_stage(&self, metrics: Metrics) -> &'static str {
        if self.overview_step < 20 {
            "初始扰动"
        } else if metrics.stability < 0.975 {
            "结构形成"
        } else {
            "稳定/衰退"
        }
    }

    fn explanation_block(ui: &mut egui::Ui, title: &str, body: &str) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new(title).strong());
        ui.label(body);
    }

    fn draw_formula_card_for(ui: &mut egui::Ui, mode: SimMode) {
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                for (label, formula) in formula_rows_for(mode) {
                    Self::formula_row(ui, label, formula);
                }
            });
    }

    fn formula_row(ui: &mut egui::Ui, label: &str, formula: &str) {
        ui.horizontal_wrapped(|ui| {
            ui.label(label);
            ui.monospace(formula);
        });
    }

    fn draw_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("peterMath");
        ui.label("数学规则生成的数字生命实验室");
        ui.small("美感来自细胞规则、扩散方程、连续场和可复现实验。");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("当前：");
            ui.strong(self.screen.label_zh());
        });
        if ui.button("返回三系统总览").clicked() {
            self.screen = AppScreen::Overview;
            self.texture = None;
            self.status = "回到总览：比较三种系统的数学差异。".to_owned();
        }

        let mut selected_mode = self.mode;
        egui::ComboBox::from_label("数学系统")
            .selected_text(selected_mode.label_zh())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut selected_mode,
                    SimMode::Lenia,
                    SimMode::Lenia.label_zh(),
                );
                ui.selectable_value(
                    &mut selected_mode,
                    SimMode::ReactionDiffusion,
                    SimMode::ReactionDiffusion.label_zh(),
                );
                ui.selectable_value(
                    &mut selected_mode,
                    SimMode::GameOfLife,
                    SimMode::GameOfLife.label_zh(),
                );
            });
        if selected_mode != self.mode {
            self.mode = selected_mode;
            self.steps_per_frame = Self::recommended_steps_for(selected_mode);
            self.step_count = 0;
            self.texture = None;
            self.mark_cpu_texture_dirty();
            self.reset_metric_history();
            self.refresh_lenia_inspection();
            self.last_interaction =
                format!("切换到{}，使用推荐速度重新开始。", selected_mode.label_zh());
            self.status = "已重新计算当前规则：系统切换生效。".to_owned();
        }

        let mut selected_render_style = self.render_style;
        egui::ComboBox::from_label("显示方式")
            .selected_text(selected_render_style.label_zh())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut selected_render_style,
                    RenderStyle::RawMath,
                    RenderStyle::RawMath.label_zh(),
                );
                ui.selectable_value(
                    &mut selected_render_style,
                    RenderStyle::Artistic,
                    RenderStyle::Artistic.label_zh(),
                );
            });
        ui.small(self.render_style.explanation_zh());
        if selected_render_style != self.render_style {
            self.render_style = selected_render_style;
            self.mark_cpu_texture_dirty();
            self.last_interaction = format!("显示方式切换为{}。", selected_render_style.label_zh());
            self.status = "已重新计算当前规则：显示方式已刷新。".to_owned();
        }

        ui.checkbox(&mut self.judge_mode, "评审讲解模式");
        ui.checkbox(&mut self.dev_diagnostics, "开发诊断");
        ui.checkbox(&mut self.show_active_region_overlay, "显示活跃区域")
            .on_hover_text("显示自动检测的活跃边界和中心点。");
        if self.gpu_lenia.is_some() {
            let previous = self.prefer_gpu_lenia;
            ui.checkbox(&mut self.prefer_gpu_lenia, "GPU 高质量 Lenia");
            if previous != self.prefer_gpu_lenia {
                self.mark_cpu_texture_dirty();
                self.tick_accumulator = Duration::ZERO;
            }
        } else {
            ui.label("GPU 高质量 Lenia：不可用");
        }
        ui.separator();
        ui.heading("实时实验控制");
        ui.small("改参数、画画布、单步运行都会立刻重新计算。");
        if ui
            .add(egui::Slider::new(&mut self.steps_per_frame, 1..=20).text("演化速度"))
            .changed()
        {
            self.last_interaction = format!("演化速度改为每次 {} 步。", self.steps_per_frame);
            self.status = "已重新计算当前规则：速度设置已更新。".to_owned();
        }

        ui.horizontal(|ui| {
            if ui
                .button(if self.running { "暂停" } else { "运行" })
                .clicked()
            {
                self.running = !self.running;
                self.last_interaction = if self.running {
                    "继续实时运行。".to_owned()
                } else {
                    "暂停画面，方便单步和现场编辑。".to_owned()
                };
                self.status = "实时控制已更新。".to_owned();
            }
            if ui.button("单步").clicked() {
                self.step_once();
                self.last_interaction = "手动单步推进，画面来自当前规则即时计算。".to_owned();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("重置").clicked() {
                if self.mode == SimMode::Lenia {
                    self.reset_lenia_with_history();
                } else {
                    self.reset_active();
                }
                self.last_interaction = format!("重置{}到当前预设。", self.mode.label_zh());
            }
            if ui.button("新种子").clicked() {
                self.new_active_seed();
            }
        });

        if self.mode == SimMode::Lenia {
            ui.separator();
            ui.heading("创作画布");
            ui.horizontal(|ui| {
                for tool in InteractionTool::ALL {
                    let shortcut = match tool {
                        InteractionTool::Draw => "D",
                        InteractionTool::Erase => "E",
                        InteractionTool::Stamp => "S",
                        InteractionTool::Pan => "安全光标",
                    };
                    ui.selectable_value(&mut self.tool, tool, tool.label())
                        .on_hover_text(shortcut);
                }
            });

            egui::ComboBox::from_label("Lenia 预设")
                .selected_text(self.active_preset.label())
                .show_ui(ui, |ui| {
                    let mut selected = self.active_preset;
                    for preset in LeniaPreset::ALL {
                        ui.selectable_value(&mut selected, preset, preset.label());
                    }
                    if selected != self.active_preset {
                        self.load_lenia_preset(selected);
                    }
                });
            ui.small(self.active_preset.description());

            egui::ComboBox::from_label("盖章形状")
                .selected_text(self.active_stamp.label())
                .show_ui(ui, |ui| {
                    for stamp in LeniaStamp::ALL {
                        ui.selectable_value(&mut self.active_stamp, stamp, stamp.label());
                    }
                });
            ui.small(self.active_stamp.description());

            ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=32.0).text("画笔半径"));
            ui.add(egui::Slider::new(&mut self.brush_strength, 0.05..=1.0).text("画笔强度"));
            ui.add(egui::Slider::new(&mut self.random_density, 0.02..=0.85).text("随机密度"));

            egui::ComboBox::from_label("网格精度")
                .selected_text(self.grid_profile.label())
                .show_ui(ui, |ui| {
                    let mut selected = self.grid_profile;
                    for profile in GridProfile::ALL {
                        ui.selectable_value(&mut selected, profile, profile.label());
                    }
                    if selected != self.grid_profile {
                        self.apply_grid_profile(selected);
                    }
                });

            ui.horizontal(|ui| {
                if ui.button("清空场").clicked() {
                    self.clear_lenia_field();
                }
                if ui.button("新种子").clicked() {
                    self.new_lenia_seed();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("随机场").clicked() {
                    self.randomize_lenia_field();
                }
                if ui
                    .add_enabled(!self.undo_stack.is_empty(), egui::Button::new("撤销"))
                    .on_hover_text("Z")
                    .clicked()
                {
                    self.undo_lenia();
                }
                if ui
                    .add_enabled(!self.redo_stack.is_empty(), egui::Button::new("重做"))
                    .on_hover_text("Shift+Z")
                    .clicked()
                {
                    self.redo_lenia();
                }
            });
            ui.small("Space 运行/暂停 · . 单步 · R 重置 · C 清空 · N 新种子 · [ ] 画笔");
        }

        if ui.button("导出截图 + 参数").clicked() {
            self.export_snapshot();
        }
        ui.horizontal(|ui| {
            if ui.button("导出可复现状态").clicked() {
                self.export_share_state();
            }
            if ui.button("证据包").clicked() {
                self.export_evidence_pack();
            }
        });

        ui.separator();
        ui.label(format!("系统：{}", self.mode.label_zh()));
        ui.label(format!("后端：{}", self.backend_label()));
        let (grid_w, grid_h) = self.active_size();
        ui.label(format!("网格：{}x{}", grid_w, grid_h));
        let (source_w, source_h) = self.lenia.size();
        if self.mode == SimMode::Lenia {
            ui.label(format!(
                "源场：{}x{} · {}",
                source_w,
                source_h,
                self.grid_profile.label()
            ));
        }
        ui.label(format!("种子：{}", self.active_seed()));
        ui.label(format!("步数：{}", self.step_count));
        ui.label(format!("观察目标：{}", self.mode_statement()));
        if self.mode == SimMode::Lenia {
            let phase = self.lenia_phase();
            ui.label(format!("阶段：{}", phase.label_zh()));
            ui.small(phase.description());
        } else {
            ui.label(format!("阶段：{}", self.active_stage_zh()));
        }
        let m = self.active_metrics();
        ui.label(format!("活跃像素/细胞：{}", m.active));
        ui.label(format!("活跃度 {:.3} · 熵 {:.3}", m.mass, m.entropy));
        ui.label(format!(
            "稳定度 {:.3} · 生命力 {:.3}",
            m.stability, m.vitality
        ));
        ui.label(format!("最近操作：{}", self.last_interaction));
        ui.label(&self.status);
    }

    fn draw_right_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("参数与解释");
        match self.mode {
            SimMode::Lenia => {
                ui.label("观察目标：连续数值场如何通过卷积核和增长函数形成生命感。");
                egui::ComboBox::from_label("安全预设")
                    .selected_text(self.active_preset.label())
                    .show_ui(ui, |ui| {
                        let mut selected = self.active_preset;
                        for preset in LeniaPreset::ALL {
                            ui.selectable_value(&mut selected, preset, preset.label());
                        }
                        if selected != self.active_preset {
                            self.load_lenia_preset(selected);
                        }
                    });
                ui.small(self.active_preset.description());
                ui.horizontal(|ui| {
                    for tool in [
                        InteractionTool::Draw,
                        InteractionTool::Erase,
                        InteractionTool::Stamp,
                    ] {
                        ui.selectable_value(&mut self.tool, tool, tool.label());
                    }
                });
                ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=32.0).text("画笔半径"));
                ui.add(egui::Slider::new(&mut self.brush_strength, 0.05..=1.0).text("画笔强度"));
                if ui
                    .add(
                        egui::Slider::new(&mut self.lenia.growth_center, 0.05..=0.95)
                            .text("增长中心"),
                    )
                    .on_hover_text("卷积结果接近这个值时增长最强。")
                    .changed()
                {
                    self.sync_gpu_lenia_from_cpu();
                    self.step_count = 0;
                    self.mark_cpu_texture_dirty();
                    self.refresh_lenia_inspection();
                    self.reset_metric_history();
                    self.last_interaction = "Lenia 增长中心已现场调整。".to_owned();
                    self.status = "已重新计算当前规则：Lenia 参数生效。".to_owned();
                }
                ui.checkbox(&mut self.show_kernel_overlay, "显示卷积半径")
                    .on_hover_text("在画面上显示被检查点的邻域范围。");
                egui::CollapsingHeader::new("高级参数")
                    .default_open(false)
                    .show(ui, |ui| {
                        let mut lenia_changed = false;
                        let mut radius = self.lenia.radius as u32;
                        if ui
                            .add(egui::Slider::new(&mut radius, 3..=32).text("卷积半径"))
                            .on_hover_text("邻域越大，每个点受更远区域影响。")
                            .changed()
                        {
                            self.lenia.set_radius(radius as usize);
                            lenia_changed = true;
                        }
                        lenia_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.lenia.growth_center, 0.05..=0.95)
                                    .text("增长中心"),
                            )
                            .on_hover_text("卷积结果接近这个值时增长最强。")
                            .changed();
                        lenia_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.lenia.growth_width, 0.005..=0.18)
                                    .text("增长宽度"),
                            )
                            .on_hover_text("越窄越挑剔，越宽越容易增长。")
                            .changed();
                        lenia_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.lenia.dt, 0.005..=0.25)
                                    .text("时间步长"),
                            )
                            .on_hover_text("每次更新推进的幅度。")
                            .changed();
                        lenia_changed |= ui
                            .add(egui::Slider::new(&mut self.lenia.decay, 0.0..=0.04).text("阻尼"))
                            .on_hover_text("已有质量的自然衰减。")
                            .changed();
                        if lenia_changed {
                            self.sync_gpu_lenia_from_cpu();
                            self.step_count = 0;
                            self.mark_cpu_texture_dirty();
                            self.refresh_lenia_inspection();
                            self.reset_metric_history();
                            self.last_interaction = "Lenia 高级参数已现场调整。".to_owned();
                            self.status = "已重新计算当前规则：Lenia 高级参数生效。".to_owned();
                        }
                    });
            }
            SimMode::ReactionDiffusion => {
                ui.label("观察目标：两种物质的扩散速度差如何放大扰动，形成斑点、波纹和迷宫。");
                egui::ComboBox::from_label("安全预设")
                    .selected_text(self.active_reaction_preset.label())
                    .show_ui(ui, |ui| {
                        let mut selected = self.active_reaction_preset;
                        for preset in ReactionPreset::ALL {
                            ui.selectable_value(&mut selected, preset, preset.label());
                        }
                        if selected != self.active_reaction_preset {
                            self.load_reaction_preset(selected);
                        }
                    });
                ui.small(self.active_reaction_preset.description());
                let mut primary_changed = false;
                primary_changed |= ui
                    .add(egui::Slider::new(&mut self.reaction.feed, 0.005..=0.09).text("补给 feed"))
                    .on_hover_text("A 物质补充速度，影响图案是否持续生长。")
                    .changed();
                primary_changed |= ui
                    .add(egui::Slider::new(&mut self.reaction.kill, 0.02..=0.09).text("消耗 kill"))
                    .on_hover_text("B 物质消耗速度，过高会让图案衰退。")
                    .changed();
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tool, InteractionTool::Draw, "注入 B");
                    ui.selectable_value(&mut self.tool, InteractionTool::Erase, "擦除 B");
                });
                ui.add(
                    egui::Slider::new(&mut self.reaction_brush_radius, 2.0..=28.0).text("扰动半径"),
                );
                ui.add(
                    egui::Slider::new(&mut self.reaction_brush_strength, 0.05..=1.0)
                        .text("扰动强度"),
                );
                if ui.button("中心注入扰动").clicked() {
                    self.add_reaction_perturbation();
                }
                if primary_changed {
                    self.step_count = 0;
                    self.mark_cpu_texture_dirty();
                    self.reset_metric_history();
                    self.last_interaction = "反应扩散 feed/kill 已现场调整。".to_owned();
                    self.status = "已重新计算当前规则：反应扩散参数生效。".to_owned();
                }
                egui::CollapsingHeader::new("高级参数")
                    .default_open(false)
                    .show(ui, |ui| {
                        let mut changed = false;
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.reaction.feed, 0.005..=0.09)
                                    .text("补给 feed"),
                            )
                            .on_hover_text("A 物质补充速度，影响图案是否持续生长。")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.reaction.kill, 0.02..=0.09)
                                    .text("消耗 kill"),
                            )
                            .on_hover_text("B 物质消耗速度，过高会让图案衰退。")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.reaction.diff_a, 0.02..=0.30)
                                    .text("A 扩散"),
                            )
                            .on_hover_text("A 物质向周围扩散的速度。")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.reaction.diff_b, 0.005..=0.20)
                                    .text("B 扩散"),
                            )
                            .on_hover_text("B 物质扩散速度，与 A 的差异制造纹理。")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.reaction.dt, 0.2..=1.5)
                                    .text("时间步长"),
                            )
                            .on_hover_text("每次模拟推进的幅度。")
                            .changed();
                        if changed {
                            self.step_count = 0;
                            self.mark_cpu_texture_dirty();
                            self.reset_metric_history();
                            self.last_interaction = "反应扩散高级参数已现场调整。".to_owned();
                            self.status = "已重新计算当前规则：反应扩散高级参数生效。".to_owned();
                        }
                    });
            }
            SimMode::GameOfLife => {
                ui.label("观察目标：只靠邻居数量，产生静止、周期振荡和滑翔移动。");
                egui::ComboBox::from_label("安全预设")
                    .selected_text(self.active_life_preset.label())
                    .show_ui(ui, |ui| {
                        let mut selected = self.active_life_preset;
                        for preset in LifePreset::ALL {
                            ui.selectable_value(&mut selected, preset, preset.label());
                        }
                        if selected != self.active_life_preset {
                            self.load_life_preset(selected);
                        }
                    });
                ui.small(self.active_life_preset.description());
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tool, InteractionTool::Draw, "绘制活细胞");
                    ui.selectable_value(&mut self.tool, InteractionTool::Erase, "擦除细胞");
                });
                ui.add(
                    egui::Slider::new(&mut self.life_brush_radius, 0.5..=4.5).text("细胞笔半径"),
                );
                if ui
                    .add(
                        egui::Slider::new(&mut self.life.random_density, 0.02..=0.55)
                            .text("随机密度"),
                    )
                    .changed()
                {
                    self.last_interaction =
                        format!("生命游戏随机密度调整为 {:.2}。", self.life.random_density);
                    self.status = "随机密度已更新；点击随机确定性种子可立即应用。".to_owned();
                }
                ui.horizontal(|ui| {
                    if ui.button("添加滑翔机").clicked() {
                        self.add_life_glider();
                    }
                    if ui.button("添加振荡器").clicked() {
                        self.add_life_oscillator();
                    }
                });
                if ui.button("随机确定性种子").clicked() {
                    self.active_life_preset = LifePreset::RandomSoup;
                    self.life.reset_random();
                    self.step_count = 0;
                    self.mark_cpu_texture_dirty();
                    self.reset_metric_history();
                    self.last_interaction = "生命游戏按当前密度生成随机汤。".to_owned();
                    self.status = "已重新计算当前规则：随机汤生效。".to_owned();
                }
                ui.label("规则 B3/S23：3 个邻居出生，2 或 3 个邻居存活。");
                ui.separator();
                ui.heading("RLE 图案");
                ui.small("RLE 只适用于离散生命游戏。");
                ui.label("导入 RLE");
                ui.add(
                    egui::TextEdit::multiline(&mut self.life_rle_input)
                        .desired_rows(5)
                        .code_editor(),
                );
                ui.horizontal(|ui| {
                    if ui.button("导入 RLE").clicked() {
                        self.import_life_rle();
                    }
                    if ui.button("导出 RLE").clicked() {
                        self.export_life_rle();
                    }
                });
                if !self.life_rle_output.is_empty() {
                    ui.label("已导出 RLE");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.life_rle_output)
                            .desired_rows(5)
                            .code_editor(),
                    );
                }
            }
        }

        ui.separator();
        ui.heading("诊断指标");
        if self.gpu_lenia_active() {
            ui.small("GPU 负责实时画面；指标使用同步的 CPU 参考场。");
        }
        let m = self.active_metrics();
        metric_bar(ui, "活跃度", m.mass);
        metric_bar(ui, "熵", m.entropy);
        metric_bar(ui, "对称性", m.symmetry);
        metric_bar(ui, "稳定度", m.stability);
        metric_bar(ui, "生命力", m.vitality);
        ui.label(format!("活跃细胞/像素：{}", m.active));
        ui.label(format!("最近变化量：{:.3}", 1.0 - m.stability));
        ui.label(format!("阶段结论：{}", self.active_stage_zh()));
        if self.mode == SimMode::Lenia {
            let phase = self.lenia_phase();
            ui.label(format!("Lenia 阶段：{}", phase.label_zh()));
            ui.small(phase.description());
            self.draw_metric_history(ui);
        }

        ui.separator();
        self.draw_interpretability_panel(ui);

        if self.judge_mode || self.dev_diagnostics {
            ui.separator();
            self.draw_performance_diagnostics(ui);
        }

        ui.separator();
        ui.heading("数学框架");
        ui.label(mode_child_explanation(self.mode));
        egui::CollapsingHeader::new("数学公式")
            .default_open(false)
            .show(ui, |ui| {
                Self::draw_formula_card_for(ui, self.mode);
                ui.small(self.mode_significance());
            });
        if self.mode == SimMode::Lenia {
            self.draw_lenia_inspector(ui);
            ui.separator();
            self.draw_kernel_lens(ui);
        }

        if self.judge_mode {
            ui.separator();
            ui.heading("评审讲解");
            if self.mode == SimMode::Lenia {
                ui.label("1. 先看连续场的颜色和轮廓。");
                ui.label("2. 用画笔加一点质量。");
                ui.label("3. 改增长中心，看形态是否继续活跃。");
                ui.label("4. 导出 PNG + JSON 作为证据。");
            } else {
                ui.label("1. 先看规则产生的当前图案。");
                ui.label("2. 暂停后改一个参数或画一下。");
                ui.label("3. 单步运行，看变化量。");
                ui.label("4. 导出证据，证明画面来自实时规则。");
            }
        }
    }

    fn draw_interpretability_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("结构解释");
        let region = self.active_region();
        if let Some((min_x, min_y, max_x, max_y)) = region.bounds {
            ui.label(format!(
                "活跃边界：({min_x}, {min_y}) 到 ({max_x}, {max_y})"
            ));
        } else {
            ui.label("活跃边界：无");
        }
        if let Some((x, y)) = region.centroid {
            ui.label(format!(
                "质心 {:.1}, {:.1} · 漂移 {:+.2}, {:+.2}",
                x, y, region.drift.0, region.drift.1
            ));
        }
        ui.label(format!("活跃区域比例 {:.3}", region.area_ratio));
        let phase = self.population_phase_analysis();
        ui.small(format!(
            "阶段 {} · 活跃度 {:+.3} · 熵 {:+.3} · 生命力 {:+.3}",
            phase.label, phase.mass_trend, phase.entropy_trend, phase.vitality_trend
        ));

        match self.mode {
            SimMode::GameOfLife => self.draw_life_pattern_report(ui),
            SimMode::Lenia => self.draw_rule_variant_explorer(ui),
            _ => {}
        }
    }

    fn draw_life_pattern_report(&self, ui: &mut egui::Ui) {
        let report = self.life_pattern_report();
        ui.separator();
        ui.heading("已知图案识别");
        if report.detections.is_empty() {
            ui.small("暂未识别到静物、振荡器或滑翔机。");
        } else {
            for detection in &report.detections {
                ui.label(format!(
                    "{} ({}) 位置 {}, {}",
                    detection.pattern.label(),
                    detection.pattern.kind(),
                    detection.x,
                    detection.y
                ));
            }
        }
        if let Some(period) = report.oscillator_period {
            ui.label(format!("振荡周期：{period}"));
        }
        if let Some(track) = report.glider_track {
            ui.label(format!("追踪到滑翔机：{}", track.count));
            if let Some((dx, dy)) = track.direction {
                ui.small(format!("方向 {:+.2}, {:+.2}", dx, dy));
            }
        }
    }

    fn draw_rule_variant_explorer(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("规则变量对照");
        ui.horizontal(|ui| {
            if ui.button("记录基线").clicked() {
                self.capture_comparison_baseline();
            }
            if ui.button("运行对照").clicked() {
                self.run_rule_variant_comparison();
            }
        });
        egui::ComboBox::from_label("改变一个参数")
            .selected_text(self.comparison_parameter.label())
            .show_ui(ui, |ui| {
                let mut selected = self.comparison_parameter;
                for parameter in VariantParameter::ALL {
                    ui.selectable_value(&mut selected, parameter, parameter.label());
                }
                if selected != self.comparison_parameter {
                    self.comparison_parameter = selected;
                    self.comparison_value = selected.current_value(&self.lenia);
                    self.clear_comparison_result();
                }
            });
        let value_changed = ui
            .add(
                egui::Slider::new(
                    &mut self.comparison_value,
                    self.comparison_parameter.range(),
                )
                .text(self.comparison_parameter.label()),
            )
            .changed();
        let steps_changed = ui
            .add(egui::Slider::new(&mut self.comparison_steps, 8..=240).text("对照步数"))
            .changed();
        if value_changed || steps_changed {
            self.clear_comparison_result();
        }
        if ui.button("应用到当前场").clicked() {
            self.apply_variant_to_current_lenia();
        }

        if self.comparison_baseline.is_some() {
            ui.small("已从当前 Lenia 场记录基线。");
        }
        if self.comparison_result.is_some() {
            self.draw_rule_variant_result(ui);
        }
    }

    fn draw_rule_variant_result(&mut self, ui: &mut egui::Ui) {
        let Some(result) = &self.comparison_result else {
            return;
        };
        ui.separator();
        ui.label(format!(
            "{} = {:.4} · {} steps",
            result.parameter.label(),
            result.value,
            result.steps
        ));
        ui.label(format!(
            "Δ mass {:+.3} · Δ entropy {:+.3}",
            result.variant_metrics.mass - result.baseline_metrics.mass,
            result.variant_metrics.entropy - result.baseline_metrics.entropy
        ));
        ui.label(format!(
            "Δ stability {:+.3} · Δ vitality {:+.3}",
            result.variant_metrics.stability - result.baseline_metrics.stability,
            result.variant_metrics.vitality - result.baseline_metrics.vitality
        ));

        let baseline_image = ColorImage::from_rgba_unmultiplied(
            [result.width, result.height],
            &result.baseline_pixels,
        );
        let variant_image = ColorImage::from_rgba_unmultiplied(
            [result.width, result.height],
            &result.variant_pixels,
        );
        if self.comparison_baseline_texture.is_none() {
            self.comparison_baseline_texture = Some(ui.ctx().load_texture(
                "peterMath-comparison-baseline",
                baseline_image,
                TextureOptions::LINEAR,
            ));
        }
        if self.comparison_variant_texture.is_none() {
            self.comparison_variant_texture = Some(ui.ctx().load_texture(
                "peterMath-comparison-variant",
                variant_image,
                TextureOptions::LINEAR,
            ));
        }
        ui.horizontal(|ui| {
            if let Some(texture) = &self.comparison_baseline_texture {
                ui.vertical(|ui| {
                    ui.small("基线");
                    ui.add(egui::Image::new((texture.id(), egui::vec2(112.0, 112.0))));
                });
            }
            if let Some(texture) = &self.comparison_variant_texture {
                ui.vertical(|ui| {
                    ui.small("变量");
                    ui.add(egui::Image::new((texture.id(), egui::vec2(112.0, 112.0))));
                });
            }
        });
    }

    fn draw_performance_diagnostics(&self, ui: &mut egui::Ui) {
        ui.heading("性能诊断");
        ui.label(format!("FPS 估计 {:.1}", self.performance.fps_estimate));
        ui.label(format!(
            "帧 {:.2} ms · 更新 {:.2} ms",
            self.performance.latest.frame_ms, self.performance.latest.update_ms
        ));
        ui.label(format!(
            "渲染/上传 {:.2} ms · CPU 同步 {:.2} ms",
            self.performance.latest.render_ms, self.performance.latest.cpu_sync_ms
        ));
        let source = self.performance.source_grid;
        let gpu = self
            .performance
            .gpu_grid
            .map(|size| format!("{size}x{size}"))
            .unwrap_or_else(|| "unavailable".to_owned());
        ui.small(format!(
            "{} · 源网格 {}x{} · GPU {}",
            self.backend_label(),
            source.0,
            source.1,
            gpu
        ));
        ui.small(format!(
            "CPU 每 {} 个 GPU 批次同步 · 待处理 GPU 步数 {} · 指标样本 {}",
            self.performance.cpu_sync_interval,
            self.performance.pending_steps,
            self.metric_history.len()
        ));
    }

    fn draw_lenia_inspector(&self, ui: &mut egui::Ui) {
        ui.heading("场检查器");
        let Some(inspection) = self.inspected_lenia else {
            ui.small("鼠标悬停在画布上，可检查局部 Lenia 数学。");
            return;
        };
        ui.label(format!("点：{}, {}", inspection.x, inspection.y));
        ui.label(format!(
            "u[t] {:.4} · 上一步 {:.4}",
            inspection.value, inspection.previous
        ));
        ui.label(format!(
            "变化量 {:+.4} · 梯度 {:.4}",
            inspection.delta, inspection.gradient
        ));
        ui.label(format!(
            "K * u {:.4} · G {:.4}",
            inspection.convolution, inspection.growth
        ));
        ui.label(format!("估计 u[t+1] {:.4}", inspection.estimated_next));
    }

    fn draw_kernel_lens(&self, ui: &mut egui::Ui) {
        ui.heading("卷积核透镜");
        ui.label(format!(
            "半径 {} · 中心 {:.3} · 宽度 {:.3} · 阻尼 {:.4}",
            self.lenia.radius, self.lenia.growth_center, self.lenia.growth_width, self.lenia.decay
        ));

        let profile = self.lenia.kernel_profile(56);
        let desired = egui::vec2(ui.available_width(), 54.0);
        let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_rgb(10, 14, 16));
        painter.line_segment(
            [
                egui::pos2(rect.left(), rect.bottom() - 8.0),
                egui::pos2(rect.right(), rect.bottom() - 8.0),
            ],
            egui::Stroke::new(1.0, Color32::from_rgb(42, 58, 62)),
        );

        let mut points = Vec::with_capacity(profile.len());
        for (i, value) in profile.iter().enumerate() {
            let t = if profile.len() > 1 {
                i as f32 / (profile.len() - 1) as f32
            } else {
                0.0
            };
            let x = egui::lerp(rect.left()..=rect.right(), t);
            let y = egui::lerp(
                (rect.bottom() - 8.0)..=rect.top() + 6.0,
                value.clamp(0.0, 1.0),
            );
            points.push(egui::pos2(x, y));
        }
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(1.6, Color32::from_rgb(108, 232, 218)),
        ));
    }

    fn draw_metric_history(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("指标曲线");
        if self.metric_history.len() < 2 {
            ui.small("运行一段时间后会形成指标轨迹。");
            return;
        }
        self.metric_history_chart(ui, "活跃度", Color32::from_rgb(103, 222, 209), |s| {
            s.mass
        });
        self.metric_history_chart(ui, "熵", Color32::from_rgb(255, 157, 102), |s| s.entropy);
        self.metric_history_chart(ui, "稳定度", Color32::from_rgb(154, 185, 255), |s| {
            s.stability
        });
        self.metric_history_chart(ui, "生命力", Color32::from_rgb(255, 111, 167), |s| {
            s.vitality
        });
    }

    fn metric_history_chart(
        &self,
        ui: &mut egui::Ui,
        label: &str,
        color: Color32,
        value: impl Fn(&MetricHistorySample) -> f32,
    ) {
        let latest = self.metric_history.last().map(&value).unwrap_or_default();
        ui.small(format!("{label} {:.3}", latest));
        let desired = egui::vec2(ui.available_width(), 28.0);
        let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_rgb(8, 12, 14));
        let count = self.metric_history.len();
        let mut points = Vec::with_capacity(count);
        for (i, sample) in self.metric_history.iter().enumerate() {
            let t = if count > 1 {
                i as f32 / (count - 1) as f32
            } else {
                0.0
            };
            let x = egui::lerp(rect.left()..=rect.right(), t);
            let y = egui::lerp(rect.bottom()..=rect.top(), value(sample).clamp(0.0, 1.0));
            points.push(egui::pos2(x, y));
        }
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.3, color)));
    }

    fn mode_statement(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "连续场如何通过邻域卷积产生柔性生命形态",
            SimMode::ReactionDiffusion => "两种物质的扩散和反应如何形成空间纹理",
            SimMode::GameOfLife => "离散细胞如何从局部邻居规则产生结构",
        }
    }

    fn mode_significance(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "柔性的邻域卷积核把细小数值变化转化为类似生命的运动。",
            SimMode::ReactionDiffusion => "扩散速度和反应速率的竞争会产生斑点、膜、波和迷宫。",
            SimMode::GameOfLife => "最简单的网格规则展示了离散细胞如何产生稳定、周期和移动结构。",
        }
    }
}

impl eframe::App for PeterMathApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        let frame_delta = frame_start
            .checked_duration_since(self.last_tick)
            .unwrap_or(Duration::ZERO)
            .min(MAX_FRAME_DELTA);
        self.last_tick = frame_start;
        self.performance.record_frame_delta(frame_delta);

        self.handle_shortcuts(ctx);
        if !ctx.input(|input| input.pointer.primary_down()) {
            self.pointer_edit_active = false;
        }

        let mut update_duration = Duration::ZERO;
        let mut render_duration = Duration::ZERO;
        let mut cpu_sync_duration = Duration::ZERO;

        if self.running {
            self.tick_accumulator += frame_delta;
            let update_start = Instant::now();
            let mut batches = 0;
            while self.tick_accumulator >= TARGET_TICK && batches < MAX_UPDATE_BATCHES {
                if self.screen == AppScreen::Overview {
                    self.update_overview_systems();
                } else if self.gpu_lenia_active() {
                    if let Some(gpu) = &self.gpu_lenia {
                        gpu.update_params(lenia_params(&self.lenia), self.render_style);
                        gpu.queue_steps(self.steps_per_frame);
                    }
                    self.gpu_cpu_sync_counter += 1;
                    if self.gpu_cpu_sync_counter >= self.gpu_cpu_sync_interval {
                        let cpu_sync_start = Instant::now();
                        self.lenia.step();
                        cpu_sync_duration += cpu_sync_start.elapsed();
                        self.gpu_cpu_sync_counter = 0;
                        self.mark_cpu_texture_dirty();
                    }
                    self.step_count += self.steps_per_frame as u64;
                } else {
                    for _ in 0..self.steps_per_frame {
                        self.step_active();
                    }
                }
                self.tick_accumulator -= TARGET_TICK;
                batches += 1;
            }
            if batches == MAX_UPDATE_BATCHES && self.tick_accumulator > TARGET_TICK {
                self.tick_accumulator = TARGET_TICK;
            }
            if batches > 0 && self.screen == AppScreen::Experiment {
                self.refresh_lenia_inspection();
                self.record_metric_history();
            }
            update_duration = update_start.elapsed();
            ctx.request_repaint_after(TARGET_TICK);
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let (grid_w, grid_h) = self.active_size();
                ui.strong("peterMath");
                ui.separator();
                ui.label(self.screen.label_zh());
                ui.separator();
                ui.label(if self.screen == AppScreen::Overview {
                    "三系统并列"
                } else {
                    self.backend_label()
                });
                ui.separator();
                if self.screen == AppScreen::Overview {
                    ui.label("生命游戏 + 反应扩散 + Lenia");
                } else {
                    ui.label(format!("{}x{}", grid_w, grid_h));
                }
                ui.separator();
                let seed = if self.screen == AppScreen::Overview {
                    self.overview_seed(self.overview_focus)
                } else {
                    self.active_seed()
                };
                ui.label(format!("种子 {}", seed));
                ui.separator();
                ui.label(format!(
                    "步数 {}",
                    if self.screen == AppScreen::Overview {
                        self.overview_step
                    } else {
                        self.step_count
                    }
                ));
                if self.screen == AppScreen::Experiment && self.mode == SimMode::Lenia {
                    ui.separator();
                    ui.label(self.lenia_phase().label_zh());
                }
            });
        });

        if self.screen == AppScreen::Overview {
            egui::CentralPanel::default().show(ctx, |ui| self.draw_overview(ctx, ui));
            self.performance
                .set_timings(update_duration, render_duration, cpu_sync_duration);
            self.update_performance_metadata();
            return;
        }

        egui::SidePanel::left("left_controls")
            .resizable(false)
            .default_width(260.0)
            .show(ctx, |ui| self.draw_left_panel(ui));

        egui::SidePanel::right("right_parameters")
            .resizable(false)
            .default_width(330.0)
            .show(ctx, |ui| self.draw_right_panel(ui));

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let square = (available.x.min(available.y) - 28.0).max(320.0);
            let size = egui::vec2(square, square);
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                if self.gpu_lenia_active() {
                    let render_start = Instant::now();
                    egui::Frame::canvas(ui.style()).show(ui, |ui| {
                        let (rect, response) =
                            ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                        if let Some(gpu) = self.gpu_lenia.as_ref() {
                            gpu.update_params(lenia_params(&self.lenia), self.render_style);
                            ui.painter()
                                .add(egui::Shape::Callback(gpu.paint_callback(rect)));
                        }
                        self.apply_canvas_interaction(rect, &response);
                        self.draw_lenia_inspection_overlay(ui.painter(), rect);
                        self.draw_active_region_overlay(ui.painter(), rect);
                    });
                    render_duration += render_start.elapsed();
                } else {
                    if self.cpu_texture_dirty || self.texture.is_none() {
                        let render_start = Instant::now();
                        let (w, h) = self.render_active();
                        let image = ColorImage::from_rgba_unmultiplied([w, h], &self.pixels);
                        let texture_options = if self.mode == SimMode::GameOfLife {
                            TextureOptions::NEAREST
                        } else {
                            TextureOptions::LINEAR
                        };
                        if let Some(texture) = &mut self.texture {
                            texture.set(image, texture_options);
                        } else {
                            self.texture =
                                Some(ctx.load_texture("peterMath-field", image, texture_options));
                        }
                        self.cpu_texture_dirty = false;
                        render_duration += render_start.elapsed();
                    }
                    if let Some(texture) = &self.texture {
                        let texture_id = texture.id();
                        egui::Frame::canvas(ui.style()).show(ui, |ui| {
                            let (rect, response) =
                                ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                            ui.painter().image(
                                texture_id,
                                rect,
                                egui::Rect::from_min_max(
                                    egui::Pos2::ZERO,
                                    egui::Pos2::new(1.0, 1.0),
                                ),
                                Color32::WHITE,
                            );
                            self.apply_canvas_interaction(rect, &response);
                            self.draw_life_grid_overlay(ui.painter(), rect);
                            self.draw_lenia_inspection_overlay(ui.painter(), rect);
                            self.draw_active_region_overlay(ui.painter(), rect);
                        });
                    }
                }
                ui.add_space(8.0);
                ui.small(format!(
                    "{} | {} | {} | seed {} | step {}",
                    self.mode.label_zh(),
                    self.render_style.label_zh(),
                    self.backend_label(),
                    self.active_seed(),
                    self.step_count
                ));
            });
        });

        self.performance
            .set_timings(update_duration, render_duration, cpu_sync_duration);
        self.update_performance_metadata();
    }
}

fn lenia_params(lenia: &LeniaSim) -> GpuLeniaParams {
    GpuLeniaParams {
        growth_center: lenia.growth_center,
        growth_width: lenia.growth_width,
        dt: lenia.dt,
        decay: lenia.decay,
    }
}

fn configure_style(ctx: &egui::Context) {
    configure_chinese_fonts(ctx);

    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = Color32::from_rgb(12, 16, 18);
    style.visuals.window_fill = Color32::from_rgb(9, 12, 14);
    style.visuals.extreme_bg_color = Color32::from_rgb(4, 6, 7);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(18, 24, 27);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(24, 32, 36);
    style.visuals.selection.bg_fill = Color32::from_rgb(60, 116, 106);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.slider_width = 170.0;
    ctx.set_style(style);
}

fn configure_chinese_fonts(ctx: &egui::Context) {
    let Some((font_name, font_bytes)) = load_chinese_font() else {
        eprintln!(
            "peterMath warning: no Chinese-capable system font found; GUI CJK text may not render."
        );
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        font_name.clone(),
        Arc::new(egui::FontData::from_owned(font_bytes)),
    );

    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let entry = fonts.families.entry(family).or_default();
        if !entry.iter().any(|name| name == &font_name) {
            let insert_at = entry.len().min(1);
            entry.insert(insert_at, font_name.clone());
        }
    }

    ctx.set_fonts(fonts);
}

fn load_chinese_font() -> Option<(String, Vec<u8>)> {
    let candidates = [
        // Windows
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/Deng.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        // macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/NISC18030.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        // Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ];

    for path in candidates {
        if let Ok(bytes) = fs::read(path) {
            return Some((
                format!("peterMath-cjk-{}", path.replace(['/', '\\', ':'], "_")),
                bytes,
            ));
        }
    }
    None
}

fn metric_bar(ui: &mut egui::Ui, label: &str, value: f32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::ProgressBar::new(value.clamp(0.0, 1.0)).show_percentage());
    });
}

fn mode_child_explanation(mode: SimMode) -> &'static str {
    match mode {
        SimMode::GameOfLife => {
            "每个格子只看周围 8 个邻居。太孤单会消失，太拥挤也会消失，刚好 3 个邻居会出生。"
        }
        SimMode::ReactionDiffusion => {
            "把两种颜料放在纸上：它们一边扩散，一边互相反应，于是长出斑点、波纹和迷宫。"
        }
        SimMode::Lenia => {
            "每个点会听周围一圈邻居的平均值。周围条件刚好时它会增长，不合适时会慢慢变弱。"
        }
    }
}

fn formula_rows_for(mode: SimMode) -> &'static [(&'static str, &'static str)] {
    match mode {
        SimMode::GameOfLife => &[
            ("出生", "dead + n=3 -> alive"),
            ("存活", "alive + n=2 or 3 -> alive"),
            ("其他", "else -> dead"),
        ],
        SimMode::ReactionDiffusion => &[
            (
                "A 物质",
                "a_next = a + dt*(Da*laplace(a) - a*b*b + feed*(1-a))",
            ),
            (
                "B 物质",
                "b_next = b + dt*(Db*laplace(b) + a*b*b - (kill+feed)*b)",
            ),
        ],
        SimMode::Lenia => &[
            ("邻域", "neighbor = kernel*u"),
            ("增长", "growth = bell(neighbor)"),
            (
                "下一帧",
                "u_next = clamp(u + dt*(growth - damping*u), 0, 1)",
            ),
        ],
    }
}

fn formula_rows_json(mode: SimMode) -> Vec<serde_json::Value> {
    formula_rows_for(mode)
        .iter()
        .map(|(label, formula)| json!({"label_zh": label, "formula": formula}))
        .collect()
}

fn metrics_json(metrics: Metrics) -> serde_json::Value {
    json!({
        "mass": metrics.mass,
        "entropy": metrics.entropy,
        "symmetry": metrics.symmetry,
        "stability": metrics.stability,
        "vitality": metrics.vitality,
        "active": metrics.active,
    })
}

fn duration_ms(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

fn next_seed(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

fn phase_label_zh(label: &str) -> &'static str {
    match label {
        "sparse" => "稀疏",
        "blooming" => "快速增长",
        "drifting" => "周期/漂移",
        "stabilizing" => "稳定形成",
        "turbulent" => "边界竞争",
        "dense" => "密集饱和",
        "fading" => "稳定/衰退",
        "discrete" => "离散结构",
        _ => "结构形成",
    }
}
