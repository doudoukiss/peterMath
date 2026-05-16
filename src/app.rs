use crate::analysis::{self, ActiveRegionAnalysis, PopulationPhaseAnalysis};
use crate::export;
use crate::gpu::{self, GpuLeniaArt, GpuLeniaParams};
use crate::metrics::Metrics;
use crate::simulation::lenia::{LeniaInspection, LeniaSim, LeniaState};
use crate::simulation::RenderStyle;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use serde_json::json;
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
    render_style: RenderStyle,
    lenia: LeniaSim,
    gpu_lenia: Option<GpuLeniaArt>,
    prefer_gpu_lenia: bool,
    running: bool,
    judge_mode: bool,
    show_mode: ShowModeState,
    info_tab: MainInfoTab,
    active_major_case: Option<MajorCaseId>,
    tool: InteractionTool,
    active_preset: LeniaPreset,
    active_stamp: LeniaStamp,
    grid_profile: GridProfile,
    random_density: f32,
    brush_radius: f32,
    brush_strength: f32,
    undo_stack: Vec<LeniaHistorySnapshot>,
    redo_stack: Vec<LeniaHistorySnapshot>,
    pointer_edit_active: bool,
    inspected_lenia: Option<LeniaInspection>,
    show_kernel_overlay: bool,
    metric_history: Vec<MetricHistorySample>,
    active_region_history: Vec<(f32, f32)>,
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
    show_active_region_overlay: bool,
    comparison_baseline: Option<LeniaState>,
    comparison_parameter: VariantParameter,
    comparison_value: f32,
    comparison_steps: usize,
    comparison_result: Option<RuleVariantComparison>,
    comparison_baseline_texture: Option<TextureHandle>,
    comparison_variant_texture: Option<TextureHandle>,
    status: String,
    last_tick: Instant,
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
    parameters: serde_json::Value,
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

#[derive(Clone, Copy, Default)]
struct ShowModeState {
    enabled: bool,
    playing: bool,
    finished: bool,
    scene_index: usize,
    scene_elapsed: f32,
    total_elapsed: f32,
    applied_scene_index: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ShowSceneId {
    Opening,
    OrbitalField,
    TwinOrganisms,
    KernelRing,
    DenseBloom,
    CoralFading,
    EvidenceSummary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MainInfoTab {
    ShowNarration,
    MajorCases,
    ParametersDiagnostics,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MajorCaseId {
    OrbitalField,
    TwinOrganisms,
    KernelRing,
    SparseSoup,
    DenseBloom,
    CoralFading,
}

#[derive(Clone, Copy)]
struct ShowNarration {
    core_question_zh: &'static str,
    initial_zh: &'static str,
    parameters_zh: &'static str,
    formula_ascii: &'static str,
    variables_zh: &'static str,
    algorithm_zh: &'static str,
    why_zh: &'static str,
    conclusion_zh: &'static str,
}

#[derive(Clone, Copy)]
struct ShowScene {
    id: ShowSceneId,
    chapter: &'static str,
    title_zh: &'static str,
    duration_secs: f32,
    preset: LeniaPreset,
    render_style: RenderStyle,
    step_rate: usize,
    narration: ShowNarration,
    case_id: Option<MajorCaseId>,
    hold_on_finish: bool,
}

#[derive(Clone, Copy)]
struct MajorCase {
    id: MajorCaseId,
    title_zh: &'static str,
    behavior_label_zh: &'static str,
    preset: LeniaPreset,
    render_style: RenderStyle,
    step_rate: usize,
    explanation_zh: &'static str,
    expected_outcome_zh: &'static str,
}

fn show_scenes() -> [ShowScene; 7] {
    [
        ShowScene {
            id: ShowSceneId::Opening,
            chapter: "开场",
            title_zh: "Lenia 是连续生命场",
            duration_secs: 15.0,
            preset: LeniaPreset::OrbitalField,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "一个连续数值场，能不能像生命一样形成、移动和衰退？",
                initial_zh: "初始图形：一组可复现的连续质量种子，全部由 seed 决定。",
                parameters_zh: "参数条件：打开后使用 GPU Lenia；CPU 参考场同步指标和检查器。",
                formula_ascii: "u_next = clamp(u + dt*(G(K*u) - damping*u), 0, 1)",
                variables_zh: "u 是生命量；K*u 是邻域卷积；G 是增长函数；damping 是阻尼。",
                algorithm_zh: "算法步骤：卷积邻域 -> 计算增长 -> 更新场 -> 用梯度和等值线渲染。",
                why_zh: "每个点只听周围一圈邻居的平均值，局部规则叠加后出现整体形体。",
                conclusion_zh: "本作品只保留 Lenia：把数学场本身做成可解释的计算艺术。",
            },
            case_id: Some(MajorCaseId::OrbitalField),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::OrbitalField,
            chapter: "漂移",
            title_zh: "轨道场：局部卷积生成漂移",
            duration_secs: 30.0,
            preset: LeniaPreset::OrbitalField,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "为什么一个固定公式会产生持续漂移，而不是静止图案？",
                initial_zh: "初始图形：螺旋分布的软团块，边界有轻微不对称。",
                parameters_zh:
                    "参数条件：radius=9，growth center=0.31，growth width=0.052，damping=0.003。",
                formula_ascii: "drift emerges when G(K*u) differs across the boundary",
                variables_zh: "边界两侧的 K*u 不同，增长值也不同，于是轮廓被推着移动。",
                algorithm_zh: "算法步骤：每帧重新计算全部点的邻域平均，不使用预制路径。",
                why_zh: "不对称边界会让一侧增长、一侧衰减，形体中心因此慢慢漂移。",
                conclusion_zh: "观察结论：运动不是动画脚本，而是卷积核和增长函数的结果。",
            },
            case_id: Some(MajorCaseId::OrbitalField),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::TwinOrganisms,
            chapter: "相互作用",
            title_zh: "双生命体：同一场中的相互影响",
            duration_secs: 30.0,
            preset: LeniaPreset::TwinOrganisms,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "两个结构靠近时，为什么会互相影响而不是各自独立？",
                initial_zh: "初始图形：两个相近但不完全相同的连续团块。",
                parameters_zh: "参数条件：两个结构共享同一张场、同一个卷积核和同一个增长函数。",
                formula_ascii: "K*u contains self influence plus neighbor influence",
                variables_zh: "K*u 同时看见自己和附近结构，因此邻近质量会改变增长响应。",
                algorithm_zh: "算法步骤：整张场同时更新，两个结构的邻域在空间中自然重叠。",
                why_zh: "连续场没有硬边界，影响通过核半径范围内的平均值传播。",
                conclusion_zh: "观察结论：Lenia 的互动来自场耦合，不来自碰撞规则。",
            },
            case_id: Some(MajorCaseId::TwinOrganisms),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::KernelRing,
            chapter: "尺度",
            title_zh: "卷积核环：半径决定形体尺度",
            duration_secs: 30.0,
            preset: LeniaPreset::KernelRing,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "为什么改变邻域半径就会改变作品的形体尺度？",
                initial_zh: "初始图形：环状质量靠近卷积核的有效采样范围。",
                parameters_zh: "参数条件：较大 radius=14，让核半径和画面结构关系更明显。",
                formula_ascii: "K*u = weighted average inside radius",
                variables_zh: "K 是权重；radius 决定每个点能听到多远的邻居。",
                algorithm_zh: "算法步骤：对半径内不同距离赋权，再把加权平均送入 G。",
                why_zh: "半径太小会破碎，半径太大又会抹平细节；尺度由核决定。",
                conclusion_zh: "观察结论：数学核不只是速度参数，它直接塑造视觉尺度。",
            },
            case_id: Some(MajorCaseId::KernelRing),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::DenseBloom,
            chapter: "湍动",
            title_zh: "密集开花：增长、饱和与湍动",
            duration_secs: 30.0,
            preset: LeniaPreset::DenseBloom,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "当生命量太多时，系统会稳定、爆发还是衰退？",
                initial_zh: "初始图形：高密度随机质量加环形扰动。",
                parameters_zh: "参数条件：较宽增长窗口让许多区域同时接近增长条件。",
                formula_ascii: "too much mass changes K*u, then G(K*u) turns negative",
                variables_zh: "mass 是总生命量；entropy 是复杂度；stability 衡量连续帧差异。",
                algorithm_zh: "算法步骤：先快速增长，再由局部竞争导致饱和、湍动或局部消退。",
                why_zh: "过密区域的邻域平均离开增长中心，增长函数会从正变负。",
                conclusion_zh: "观察结论：美来自接近平衡和失衡之间的张力。",
            },
            case_id: Some(MajorCaseId::DenseBloom),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::CoralFading,
            chapter: "边界",
            title_zh: "珊瑚衰退：阻尼与边界条件",
            duration_secs: 25.0,
            preset: LeniaPreset::CoralDrift,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "为什么有些结构会像珊瑚一样生长，又逐渐变薄？",
                initial_zh: "初始图形：分枝状软种子，局部厚度不均匀。",
                parameters_zh: "参数条件：较低 growth center 和适中 damping，让边缘竞争更明显。",
                formula_ascii: "damping*u removes mass unless growth is strong enough",
                variables_zh: "damping 是持续衰减；只有合适的邻域平均才能抵消它。",
                algorithm_zh: "算法步骤：边缘增长、内部平均、阻尼消耗同时作用。",
                why_zh: "薄弱区域无法长期维持增长，边界会漂移、断裂或衰退。",
                conclusion_zh: "观察结论：Lenia 同时展示生成和消亡，而不是无尽循环。",
            },
            case_id: Some(MajorCaseId::CoralFading),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::EvidenceSummary,
            chapter: "证据",
            title_zh: "证据总结：参数、指标、导出",
            duration_secs: 20.0,
            preset: LeniaPreset::CoralDrift,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            narration: ShowNarration {
                core_question_zh: "评委如何确认这不是预渲染视频？",
                initial_zh: "初始图形：停在可导出证据的 Lenia 状态。",
                parameters_zh:
                    "参数条件：seed、step、kernel、growth、metrics 和 inspector 全部写入 JSON。",
                formula_ascii: "evidence = PNG + parameters JSON + share state JSON + summary",
                variables_zh: "PNG 是画面；JSON 是规则和状态；metrics 是同一帧的数值证据。",
                algorithm_zh: "算法步骤：运行实时场 -> 读取当前状态 -> 同步保存图像、参数和指标。",
                why_zh: "任何参数变化都会暂停演示并实时改变场，导出的证据能复现实验条件。",
                conclusion_zh: "总结结论：peterMath 是一个 Lenia 计算艺术实验，不是视频播放器。",
            },
            case_id: Some(MajorCaseId::CoralFading),
            hold_on_finish: true,
        },
    ]
}

fn major_cases() -> [MajorCase; 6] {
    [
        MajorCase {
            id: MajorCaseId::OrbitalField,
            title_zh: "轨道场",
            behavior_label_zh: "漂移",
            preset: LeniaPreset::OrbitalField,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            explanation_zh: "螺旋种子让边界两侧的卷积平均不同，形体会被增长函数推着移动。",
            expected_outcome_zh: "柔性轮廓、亮色脊线和质心漂移共同形成连续生命感。",
        },
        MajorCase {
            id: MajorCaseId::TwinOrganisms,
            title_zh: "双生命体",
            behavior_label_zh: "相互作用",
            preset: LeniaPreset::TwinOrganisms,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            explanation_zh: "两个团块共享同一张场，卷积核会把彼此的质量纳入邻域平均。",
            expected_outcome_zh: "两个结构会靠近、变形或避让，显示连续场的耦合。",
        },
        MajorCase {
            id: MajorCaseId::KernelRing,
            title_zh: "卷积核环",
            behavior_label_zh: "尺度",
            preset: LeniaPreset::KernelRing,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            explanation_zh: "环形质量让卷积半径和形体尺度的关系变得清楚。",
            expected_outcome_zh: "调节半径会直接改变边界尺度、细节密度和稳定性。",
        },
        MajorCase {
            id: MajorCaseId::SparseSoup,
            title_zh: "稀疏汤",
            behavior_label_zh: "自组织",
            preset: LeniaPreset::SparseSoup,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            explanation_zh: "低密度随机质量测试少量岛屿能否在同一规则下组织成结构。",
            expected_outcome_zh: "局部岛屿会合并、消失或形成小尺度漂移。",
        },
        MajorCase {
            id: MajorCaseId::DenseBloom,
            title_zh: "密集开花",
            behavior_label_zh: "湍动",
            preset: LeniaPreset::DenseBloom,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            explanation_zh: "高密度质量让许多区域同时接近增长条件，随后出现竞争和饱和。",
            expected_outcome_zh: "质量、熵、稳定度的变化会显示从增长到湍动的过程。",
        },
        MajorCase {
            id: MajorCaseId::CoralFading,
            title_zh: "珊瑚衰退",
            behavior_label_zh: "衰退",
            preset: LeniaPreset::CoralDrift,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            explanation_zh: "分枝结构在阻尼和增长窗口之间竞争，薄弱边界会逐渐变淡。",
            expected_outcome_zh: "局部纹理漂移、变薄或断裂，展示生成和消亡的边界条件。",
        },
    ]
}

impl PeterMathApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let width = 192;
        let render_style = RenderStyle::Artistic;
        let lenia = LeniaSim::new(width, width, 1001);
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
            render_style,
            lenia,
            gpu_lenia,
            prefer_gpu_lenia: gpu_ready,
            running: true,
            judge_mode: true,
            show_mode: ShowModeState::enabled_default(),
            info_tab: MainInfoTab::ShowNarration,
            active_major_case: Some(MajorCaseId::OrbitalField),
            tool: InteractionTool::Pan,
            active_preset: LeniaPreset::OrbitalField,
            active_stamp: LeniaStamp::SoftCell,
            grid_profile: GridProfile::Reference192,
            random_density: 0.24,
            brush_radius: 9.0,
            brush_strength: 0.42,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pointer_edit_active: false,
            inspected_lenia,
            show_kernel_overlay: true,
            metric_history,
            active_region_history: Vec::new(),
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
            show_active_region_overlay: true,
            comparison_baseline: None,
            comparison_parameter: VariantParameter::GrowthCenter,
            comparison_value: 0.36,
            comparison_steps: 80,
            comparison_result: None,
            comparison_baseline_texture: None,
            comparison_variant_texture: None,
            status: if gpu_ready {
                "GPU Lenia 已启用。演示会自动播放，手动改参数会暂停并实时重算。".to_owned()
            } else {
                "当前使用 CPU 参考模式。GPU Lenia 不可用，但作品仍可运行。".to_owned()
            },
            last_tick: Instant::now(),
        }
    }

    fn active_size(&self) -> (usize, usize) {
        if self.gpu_lenia_active() {
            let size = self.gpu_lenia.as_ref().map(|gpu| gpu.size()).unwrap_or(192) as usize;
            (size, size)
        } else {
            self.lenia.size()
        }
    }

    fn active_seed(&self) -> u64 {
        self.lenia.seed
    }

    fn active_metrics(&self) -> Metrics {
        self.lenia.metrics()
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
            .iter()
            .rev()
            .nth(1)
            .or_else(|| self.metric_history.last())
            .map(|sample| sample.mass)
            .unwrap_or(metrics.mass);
        LeniaPhase::from_metrics(metrics, metrics.mass - reference_mass)
    }

    fn previous_centroid(&self) -> Option<(f32, f32)> {
        self.active_region_history.last().copied()
    }

    fn active_region(&self) -> ActiveRegionAnalysis {
        let (w, h) = self.lenia.size();
        analysis::active_region_from_scalar_grid(
            self.lenia.field(),
            w,
            h,
            0.08,
            self.previous_centroid(),
        )
    }

    fn population_phase_analysis(&self) -> PopulationPhaseAnalysis {
        let current = self.active_metrics();
        let previous = self.previous_metric_history(current);
        analysis::population_phase_analysis(
            self.lenia_phase().label(),
            current,
            previous,
            self.active_region(),
        )
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

    fn record_interpretability_history(&mut self) {
        if let Some(centroid) = self.active_region().centroid {
            let changed = self
                .active_region_history
                .last()
                .map(|last| {
                    (last.0 - centroid.0).abs() > 0.001 || (last.1 - centroid.1).abs() > 0.001
                })
                .unwrap_or(true);
            if changed {
                self.active_region_history.push(centroid);
                if self.active_region_history.len() > 64 {
                    self.active_region_history.remove(0);
                }
            }
        }
    }

    fn reset_active(&mut self) {
        self.step_count = 0;
        self.lenia.reset_preset(self.active_preset.id());
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
    }

    fn current_show_scene(&self) -> ShowScene {
        let scenes = show_scenes();
        scenes[self.show_mode.scene_index.min(scenes.len() - 1)]
    }

    fn update_show_total_elapsed(&mut self) {
        self.show_mode.total_elapsed =
            show_elapsed_before_scene(self.show_mode.scene_index) + self.show_mode.scene_elapsed;
    }

    fn start_show_mode(&mut self) {
        self.show_mode = ShowModeState::enabled_default();
        self.judge_mode = true;
        self.info_tab = MainInfoTab::ShowNarration;
        self.show_active_region_overlay = true;
        self.show_kernel_overlay = true;
        self.apply_show_scene(0);
        self.running = true;
        self.status = "Lenia-only 评审演示已开始：约 3 分钟后停在证据总结页。".to_owned();
    }

    fn exit_show_mode(&mut self) {
        self.show_mode.enabled = false;
        self.show_mode.playing = false;
        self.show_mode.finished = false;
        self.running = false;
        self.status = "已退出演示模式；当前 Lenia 状态保留为手动实验起点。".to_owned();
    }

    fn toggle_show_playing(&mut self) {
        if !self.show_mode.enabled {
            self.start_show_mode();
            return;
        }
        self.show_mode.playing = !self.show_mode.playing;
        self.show_mode.finished = false;
        self.running = self.show_mode.playing;
        self.status = if self.show_mode.playing {
            "演示继续播放。".to_owned()
        } else {
            "演示已暂停，可继续、跳段或退出手动实验。".to_owned()
        };
    }

    fn set_show_scene(&mut self, index: usize) {
        let scenes = show_scenes();
        self.show_mode.enabled = true;
        self.show_mode.finished = false;
        self.show_mode.scene_index = index.min(scenes.len() - 1);
        self.show_mode.scene_elapsed = 0.0;
        self.update_show_total_elapsed();
        self.apply_show_scene(self.show_mode.scene_index);
    }

    fn jump_show_scene(&mut self, delta: isize) {
        let scenes = show_scenes();
        let next = (self.show_mode.scene_index as isize + delta).clamp(0, scenes.len() as isize - 1)
            as usize;
        self.set_show_scene(next);
        self.show_mode.playing = true;
        self.running = true;
    }

    fn restart_show_mode(&mut self) {
        self.show_mode = ShowModeState::enabled_default();
        self.info_tab = MainInfoTab::ShowNarration;
        self.apply_show_scene(0);
        self.running = true;
        self.status = "演示已从第一段重新开始。".to_owned();
    }

    fn pause_show_for_manual_interaction(&mut self) {
        if self.show_mode.enabled && self.show_mode.playing {
            self.show_mode.playing = false;
            self.running = false;
            self.show_mode.finished = false;
            self.status =
                "演示已暂停：检测到手动参数或画布操作，可继续演示或退出手动实验。".to_owned();
        }
    }

    fn ensure_show_scene_applied(&mut self) {
        if self.show_mode.enabled
            && self.show_mode.applied_scene_index != Some(self.show_mode.scene_index)
        {
            self.apply_show_scene(self.show_mode.scene_index);
        }
    }

    fn apply_show_scene(&mut self, index: usize) {
        let scenes = show_scenes();
        let index = index.min(scenes.len() - 1);
        let scene = scenes[index];

        self.render_style = scene.render_style;
        self.steps_per_frame = scene.step_rate;
        self.active_preset = scene.preset;
        self.active_major_case = scene.case_id;
        self.judge_mode = true;
        self.show_active_region_overlay = true;
        self.show_kernel_overlay = true;
        self.tool = InteractionTool::Pan;
        self.step_count = 0;
        self.tick_accumulator = Duration::ZERO;
        self.gpu_cpu_sync_counter = 0;
        self.clear_comparison_result();

        let size = self.grid_profile.size();
        if self.lenia.size() != (size, size) {
            self.lenia.resize(size, size);
        }
        self.lenia.reset_preset(scene.preset.id());
        self.prefer_gpu_lenia = self.gpu_lenia.is_some();
        let (w, h) = self.lenia.size();
        self.inspected_lenia = Some(self.lenia.inspect_point(w / 2, h / 2));

        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.reset_metric_history();
        self.show_mode.applied_scene_index = Some(index);
        self.update_show_total_elapsed();
        self.status = format!("演示场景：{}", scene.title_zh);
    }

    fn load_major_case(&mut self, case: MajorCase) {
        self.show_mode.enabled = false;
        self.show_mode.playing = false;
        self.show_mode.finished = false;
        self.render_style = case.render_style;
        self.steps_per_frame = case.step_rate;
        self.active_preset = case.preset;
        self.active_major_case = Some(case.id);
        self.judge_mode = true;
        self.show_active_region_overlay = true;
        self.show_kernel_overlay = true;
        self.tool = InteractionTool::Pan;
        self.step_count = 0;
        self.tick_accumulator = Duration::ZERO;
        self.gpu_cpu_sync_counter = 0;
        self.clear_comparison_result();

        let size = self.grid_profile.size();
        if self.lenia.size() != (size, size) {
            self.lenia.resize(size, size);
        }
        self.lenia.reset_preset(case.preset.id());
        self.prefer_gpu_lenia = self.gpu_lenia.is_some();
        let (w, h) = self.lenia.size();
        self.inspected_lenia = Some(self.lenia.inspect_point(w / 2, h / 2));

        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.reset_metric_history();
        self.running = true;
        self.info_tab = MainInfoTab::ShowNarration;
        self.status = format!(
            "已载入主要情况：{}。这是实时 Lenia 模拟，可暂停、单步或改参数。",
            case.title_zh
        );
    }

    fn advance_show_mode(&mut self, frame_delta: Duration) {
        if !self.show_mode.enabled || !self.show_mode.playing {
            return;
        }

        let scenes = show_scenes();
        let mut remaining = frame_delta.as_secs_f32();
        while remaining > 0.0 && self.show_mode.playing {
            let scene = scenes[self.show_mode.scene_index.min(scenes.len() - 1)];
            let remaining_in_scene = (scene.duration_secs - self.show_mode.scene_elapsed).max(0.0);
            if remaining < remaining_in_scene {
                self.show_mode.scene_elapsed += remaining;
                remaining = 0.0;
            } else {
                self.show_mode.scene_elapsed = scene.duration_secs;
                remaining -= remaining_in_scene;
                if self.show_mode.scene_index + 1 >= scenes.len() {
                    self.show_mode.playing = false;
                    self.show_mode.finished = true;
                    self.running = false;
                    self.status = "演示已完成；总结页会保留，直到评委点击结束演示。".to_owned();
                    break;
                }
                self.show_mode.scene_index += 1;
                self.show_mode.scene_elapsed = 0.0;
                self.apply_show_scene(self.show_mode.scene_index);
            }
        }

        self.update_show_total_elapsed();
        if self.show_mode.playing {
            self.running = true;
        }
    }

    fn step_active(&mut self) {
        self.lenia.step();
        self.step_count += 1;
        self.mark_cpu_texture_dirty();
    }

    fn render_active(&mut self) -> (usize, usize) {
        let (w, h) = self.lenia.size();
        let required = w * h * 4;
        if self.pixels.len() != required {
            self.pixels.resize(required, 0);
        }
        self.lenia.render_rgba(self.render_style, &mut self.pixels);
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
            "peterMath_Lenia_seed{}_step{}",
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
                    mode: lenia_mode_label(),
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
        let previous = self.previous_metric_history(metrics);
        let mass_trend = previous
            .map(|previous| metrics.mass - previous.mass)
            .unwrap_or_default();
        let phase = analysis::population_phase_analysis(
            LeniaPhase::from_metrics(metrics, mass_trend).label(),
            metrics,
            previous,
            active_region,
        );
        let parameters =
            self.attach_show_mode_json(self.lenia_parameter_json(active_region, phase));

        Ok(GpuLeniaExportState {
            size,
            pixels,
            metrics,
            parameters,
        })
    }

    fn export_gpu_lenia_snapshot(&mut self) {
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
                    mode: lenia_mode_label(),
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
                let (w, h) = self.lenia.size();
                (w, h, self.active_metrics(), self.parameter_json())
            };
            export::save_share_state(
                "peterMath_share_state.json",
                export::ShareStateExport {
                    mode: lenia_mode_label(),
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
            "peterMath_Lenia_seed{}_step{}",
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
                    mode: lenia_mode_label(),
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
                "已导出证据包：{}（PNG {}，JSON {}，可复现状态 {}，摘要 {}）",
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
        self.prefer_gpu_lenia && self.gpu_lenia.is_some()
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
        self.performance.source_grid = self.lenia.size();
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
            self.status = "没有可撤销的状态。".to_owned();
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
            self.status = "没有可重做的状态。".to_owned();
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
        self.active_major_case = MajorCaseId::from_preset(preset);
        self.lenia.reset_preset(preset.id());
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = format!("已载入 Lenia 预设：{}。", preset.label());
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
        self.active_major_case = Some(MajorCaseId::SparseSoup);
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

        if self.show_mode.enabled {
            if shortcuts.0 {
                self.toggle_show_playing();
                return;
            }
            if shortcuts.1
                || shortcuts.2
                || shortcuts.3
                || shortcuts.4
                || shortcuts.5
                || shortcuts.6
                || shortcuts.7
                || shortcuts.8
                || shortcuts.9
                || shortcuts.10
                || shortcuts.11
            {
                self.pause_show_for_manual_interaction();
                self.show_mode.playing = false;
            }
        }

        if shortcuts.0 {
            self.running = !self.running;
        }
        if shortcuts.1 {
            self.step_once();
        }
        if shortcuts.2 {
            self.reset_lenia_with_history();
        }
        if shortcuts.3 {
            self.clear_lenia_field();
        }
        if shortcuts.4 {
            self.new_lenia_seed();
        }
        if shortcuts.5 {
            self.undo_lenia();
        }
        if shortcuts.6 {
            self.redo_lenia();
        }
        if shortcuts.7 {
            self.tool = InteractionTool::Draw;
        }
        if shortcuts.8 {
            self.tool = InteractionTool::Erase;
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

    fn clear_lenia_field(&mut self) {
        self.push_lenia_history();
        self.lenia.clear();
        self.step_count = 0;
        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.sync_gpu_lenia_from_cpu();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.status = "已清空 Lenia 场；可绘制、盖章或选择新种子继续。".to_owned();
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

    fn clear_comparison_result(&mut self) {
        self.comparison_result = None;
        self.comparison_baseline_texture = None;
        self.comparison_variant_texture = None;
    }

    fn capture_comparison_baseline(&mut self) {
        self.comparison_baseline = Some(self.lenia.snapshot());
        self.comparison_value = self.comparison_parameter.current_value(&self.lenia);
        self.clear_comparison_result();
        self.status = "已记录 Lenia 规则变量对照基线。".to_owned();
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
            "已应用 Lenia 变量：{} = {:.4}。",
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

        self.pause_show_for_manual_interaction();
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
                    .paint_brush(x, y, self.brush_radius, self.brush_strength)
            }
            InteractionTool::Erase => {
                self.lenia
                    .erase_brush(x, y, self.brush_radius, self.brush_strength)
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
    }

    fn update_lenia_inspection_from_canvas(&mut self, rect: egui::Rect, response: &egui::Response) {
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
        if !(self.show_kernel_overlay || self.judge_mode) {
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
        painter.circle_stroke(
            center,
            radius,
            egui::Stroke::new(1.2, Color32::from_rgba_unmultiplied(120, 238, 224, 170)),
        );
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

    fn draw_active_region_overlay(&self, painter: &egui::Painter, rect: egui::Rect) {
        if !(self.show_active_region_overlay || self.judge_mode) {
            return;
        }
        let region = self.active_region();
        let Some((min_x, min_y, max_x, max_y)) = region.bounds else {
            return;
        };
        let (w, h) = self.lenia.size();
        let bounds_rect = egui::Rect::from_min_max(
            egui::pos2(
                rect.min.x + min_x as f32 / w as f32 * rect.width(),
                rect.min.y + min_y as f32 / h as f32 * rect.height(),
            ),
            egui::pos2(
                rect.min.x + (max_x + 1) as f32 / w as f32 * rect.width(),
                rect.min.y + (max_y + 1) as f32 / h as f32 * rect.height(),
            ),
        );
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
            "source_grid": {"width": self.performance.source_grid.0, "height": self.performance.source_grid.1},
            "gpu_grid": self.performance.gpu_grid,
            "pending_gpu_steps": self.performance.pending_steps,
            "cpu_sync_interval": self.performance.cpu_sync_interval,
            "frame_samples": self.performance.frame_samples,
        })
    }

    fn major_case_json(&self) -> serde_json::Value {
        let Some(case_id) = self.active_major_case else {
            return serde_json::Value::Null;
        };
        let Some(case) = major_cases().into_iter().find(|case| case.id == case_id) else {
            return serde_json::Value::Null;
        };
        json!({
            "id": case.id.id(),
            "title_zh": case.title_zh,
            "behavior_label_zh": case.behavior_label_zh,
            "render_style": case.render_style.label(),
            "step_rate": case.step_rate,
            "preset": case.preset.id(),
            "explanation_zh": case.explanation_zh,
            "expected_outcome_zh": case.expected_outcome_zh,
        })
    }

    fn attach_show_mode_json(&self, mut parameters: serde_json::Value) -> serde_json::Value {
        if let Some(object) = parameters.as_object_mut() {
            object.insert("major_case".to_owned(), self.major_case_json());
            if self.show_mode.enabled {
                object.insert(
                    "show_mode".to_owned(),
                    show_mode_json_from_state(&self.show_mode),
                );
            }
        }
        parameters
    }

    fn active_region_value(region: ActiveRegionAnalysis) -> serde_json::Value {
        json!({
            "active_count": region.active_count,
            "bounds": region.bounds.map(|(min_x, min_y, max_x, max_y)| json!({"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y})),
            "centroid": region.centroid.map(|(x, y)| json!({"x": x, "y": y})),
            "area_ratio": region.area_ratio,
            "drift": {"x": region.drift.0, "y": region.drift.1},
        })
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
            "stable_id": "lenia_only",
            "display_name_zh": lenia_mode_label(),
            "explanation_zh": "连续场通过邻域卷积、增长函数和阻尼生成柔性结构。",
            "formula_ascii": "u_next = clamp(u + dt*(G(K*u) - damping*u), 0, 1)",
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
            "source_grid": {"width": self.lenia.size().0, "height": self.lenia.size().1},
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
        self.attach_show_mode_json(
            self.lenia_parameter_json(self.active_region(), self.population_phase_analysis()),
        )
    }

    fn draw_show_mode_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("评审演示模式");
        if !self.show_mode.enabled {
            if ui.button("开始 Lenia 演示").clicked() {
                self.start_show_mode();
            }
            ui.small("约 3 分钟：只讲 Lenia 的连续场、卷积核、增长和证据导出。");
            return;
        }

        let scene = self.current_show_scene();
        let total_duration = show_total_duration_secs();
        let scene_progress = (self.show_mode.scene_elapsed / scene.duration_secs).clamp(0.0, 1.0);
        let total_progress = (self.show_mode.total_elapsed / total_duration).clamp(0.0, 1.0);
        if self.show_mode.finished {
            ui.colored_label(
                Color32::from_rgb(255, 219, 128),
                "演示已完成，停留在总结页。",
            );
        }
        ui.label(format!(
            "{} · 第 {}/{} 段",
            scene.chapter,
            self.show_mode.scene_index + 1,
            show_scenes().len()
        ));
        ui.strong(scene.title_zh);
        ui.add(egui::ProgressBar::new(scene_progress).text(format!(
            "{:.0}/{:.0}s",
            self.show_mode.scene_elapsed, scene.duration_secs
        )));
        ui.add(
            egui::ProgressBar::new(total_progress)
                .text(format!("总进度 {:.0}%", total_progress * 100.0)),
        );
        ui.horizontal(|ui| {
            if ui
                .button(if self.show_mode.playing {
                    "暂停演示"
                } else if self.show_mode.finished {
                    "重新播放"
                } else {
                    "继续演示"
                })
                .clicked()
            {
                if self.show_mode.finished {
                    self.restart_show_mode();
                } else {
                    self.toggle_show_playing();
                }
            }
            if ui.button("上一段").clicked() {
                self.jump_show_scene(-1);
            }
            if ui.button("下一段").clicked() {
                self.jump_show_scene(1);
            }
        });
        ui.horizontal(|ui| {
            if ui.button("重新开始").clicked() {
                self.restart_show_mode();
            }
            if ui.button("结束演示进入手动实验").clicked() {
                self.exit_show_mode();
            }
        });
        egui::ComboBox::from_label("跳到章节")
            .selected_text(scene.chapter)
            .show_ui(ui, |ui| {
                for (index, candidate) in show_scenes().iter().enumerate() {
                    if ui
                        .selectable_label(
                            self.show_mode.scene_index == index,
                            format!("{} · {}", candidate.chapter, candidate.title_zh),
                        )
                        .clicked()
                    {
                        self.set_show_scene(index);
                        self.show_mode.playing = true;
                        self.running = true;
                    }
                }
            });
        ui.small("手动修改参数或画布时，演示会自动暂停。");
    }

    fn draw_show_mode_narration(&self, ui: &mut egui::Ui) {
        if !self.show_mode.enabled {
            if let Some(case_id) = self.active_major_case {
                if let Some(case) = major_cases().into_iter().find(|case| case.id == case_id) {
                    ui.heading("当前主要情况");
                    ui.strong(case.title_zh);
                    ui.label(format!("行为标签：{}", case.behavior_label_zh));
                    ui.label(case.explanation_zh);
                    ui.label(case.expected_outcome_zh);
                    ui.separator();
                }
            }
            return;
        }
        let scene = self.current_show_scene();
        ui.heading("科学解释卡片");
        ui.small(scene.chapter);
        ui.strong(scene.title_zh);
        ui.separator();
        explanation_row(ui, "核心问题", scene.narration.core_question_zh);
        explanation_row(ui, "初始条件", scene.narration.initial_zh);
        explanation_row(ui, "参数条件", scene.narration.parameters_zh);
        formula_card(
            ui,
            scene.narration.formula_ascii,
            scene.narration.variables_zh,
            scene.narration.algorithm_zh,
        );
        explanation_row(ui, "为什么会这样", scene.narration.why_zh);
        explanation_row(ui, "观察结论", scene.narration.conclusion_zh);
        ui.separator();
    }

    fn draw_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("peterMath");
        ui.label("Lenia 连续生命场计算艺术");
        ui.small("美感来自场、卷积核、增长函数、阻尼和可复现种子。");
        ui.separator();
        self.draw_show_mode_controls(ui);
        ui.separator();

        let mut selected_render_style = self.render_style;
        egui::ComboBox::from_label("显示方式")
            .selected_text(selected_render_style.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut selected_render_style,
                    RenderStyle::RawMath,
                    RenderStyle::RawMath.label(),
                );
                ui.selectable_value(
                    &mut selected_render_style,
                    RenderStyle::Artistic,
                    RenderStyle::Artistic.label(),
                );
            });
        if selected_render_style != self.render_style {
            self.pause_show_for_manual_interaction();
            self.render_style = selected_render_style;
            self.mark_cpu_texture_dirty();
            self.sync_gpu_lenia_from_cpu();
        }

        ui.checkbox(&mut self.judge_mode, "评审讲解模式");
        ui.checkbox(&mut self.dev_diagnostics, "开发诊断");
        ui.checkbox(&mut self.show_active_region_overlay, "显示活跃区域")
            .on_hover_text("显示自动检测的活跃边界和中心点。");
        if self.gpu_lenia.is_some() {
            let previous = self.prefer_gpu_lenia;
            ui.checkbox(&mut self.prefer_gpu_lenia, "GPU 高质量 Lenia");
            if previous != self.prefer_gpu_lenia {
                self.pause_show_for_manual_interaction();
                self.mark_cpu_texture_dirty();
                self.tick_accumulator = Duration::ZERO;
                self.sync_gpu_lenia_from_cpu();
            }
        } else {
            ui.label("GPU 高质量 Lenia：不可用");
        }
        if ui
            .add(egui::Slider::new(&mut self.steps_per_frame, 1..=8).text("演化速度"))
            .changed()
        {
            self.pause_show_for_manual_interaction();
        }

        ui.horizontal(|ui| {
            if ui
                .button(if self.running { "暂停" } else { "运行" })
                .clicked()
            {
                let was_running = self.running;
                self.pause_show_for_manual_interaction();
                self.running = !was_running;
                self.show_mode.playing = false;
            }
            if ui.button("单步").clicked() {
                self.pause_show_for_manual_interaction();
                self.show_mode.playing = false;
                self.running = false;
                self.step_once();
            }
            if ui.button("重置").clicked() {
                self.pause_show_for_manual_interaction();
                self.reset_lenia_with_history();
            }
        });

        ui.separator();
        ui.heading("交互实验室");
        ui.horizontal_wrapped(|ui| {
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
                    self.pause_show_for_manual_interaction();
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

        let brush_changed = ui
            .add(egui::Slider::new(&mut self.brush_radius, 1.0..=32.0).text("画笔半径"))
            .changed()
            | ui.add(egui::Slider::new(&mut self.brush_strength, 0.05..=1.0).text("画笔强度"))
                .changed()
            | ui.add(egui::Slider::new(&mut self.random_density, 0.02..=0.85).text("随机密度"))
                .changed();
        if brush_changed {
            self.pause_show_for_manual_interaction();
        }

        egui::ComboBox::from_label("网格精度")
            .selected_text(self.grid_profile.label())
            .show_ui(ui, |ui| {
                let mut selected = self.grid_profile;
                for profile in GridProfile::ALL {
                    ui.selectable_value(&mut selected, profile, profile.label());
                }
                if selected != self.grid_profile {
                    self.pause_show_for_manual_interaction();
                    self.apply_grid_profile(selected);
                }
            });

        ui.horizontal(|ui| {
            if ui.button("清空场").clicked() {
                self.pause_show_for_manual_interaction();
                self.clear_lenia_field();
            }
            if ui.button("新种子").clicked() {
                self.pause_show_for_manual_interaction();
                self.new_lenia_seed();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("随机场").clicked() {
                self.pause_show_for_manual_interaction();
                self.randomize_lenia_field();
            }
            if ui
                .add_enabled(!self.undo_stack.is_empty(), egui::Button::new("撤销"))
                .on_hover_text("Z")
                .clicked()
            {
                self.pause_show_for_manual_interaction();
                self.undo_lenia();
            }
            if ui
                .add_enabled(!self.redo_stack.is_empty(), egui::Button::new("重做"))
                .on_hover_text("Shift+Z")
                .clicked()
            {
                self.pause_show_for_manual_interaction();
                self.redo_lenia();
            }
        });
        ui.small("Space 运行/暂停 · . 单步 · R 重置 · C 清空 · N 新种子 · [ ] 画笔");

        ui.separator();
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
        ui.label(format!("系统：{}", lenia_mode_label()));
        ui.label(format!("后端：{}", self.backend_label()));
        let (grid_w, grid_h) = self.active_size();
        ui.label(format!("显示网格：{}x{}", grid_w, grid_h));
        let (source_w, source_h) = self.lenia.size();
        ui.label(format!(
            "源场：{}x{} · {}",
            source_w,
            source_h,
            self.grid_profile.label()
        ));
        ui.label(format!("种子：{}", self.active_seed()));
        ui.label(format!("步数：{}", self.step_count));
        let phase = self.lenia_phase();
        ui.label(format!("阶段：{}", phase.label()));
        ui.small(phase.description());
        let m = self.active_metrics();
        ui.label(format!("活跃像素：{}", m.active));
        ui.label(format!("质量 {:.3} · 熵 {:.3}", m.mass, m.entropy));
        ui.label(format!(
            "稳定度 {:.3} · 生命力 {:.3}",
            m.stability, m.vitality
        ));
        ui.label(&self.status);
    }

    fn draw_right_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for tab in [
                MainInfoTab::ShowNarration,
                MainInfoTab::MajorCases,
                MainInfoTab::ParametersDiagnostics,
            ] {
                ui.selectable_value(&mut self.info_tab, tab, tab.label());
            }
        });
        ui.separator();
        match self.info_tab {
            MainInfoTab::ShowNarration => {
                self.draw_show_mode_narration(ui);
                self.draw_compact_live_diagnostics(ui);
                return;
            }
            MainInfoTab::MajorCases => {
                self.draw_major_cases_panel(ui);
                return;
            }
            MainInfoTab::ParametersDiagnostics => {}
        }

        ui.heading("Lenia 参数");
        let mut lenia_changed = false;
        let mut radius = self.lenia.radius as u32;
        if ui
            .add(egui::Slider::new(&mut radius, 3..=32).text("卷积半径"))
            .changed()
        {
            self.lenia.set_radius(radius as usize);
            lenia_changed = true;
        }
        lenia_changed |= ui
            .add(egui::Slider::new(&mut self.lenia.growth_center, 0.05..=0.95).text("增长中心"))
            .changed();
        lenia_changed |= ui
            .add(egui::Slider::new(&mut self.lenia.growth_width, 0.005..=0.18).text("增长宽度"))
            .changed();
        lenia_changed |= ui
            .add(egui::Slider::new(&mut self.lenia.dt, 0.005..=0.25).text("时间步长"))
            .changed();
        lenia_changed |= ui
            .add(egui::Slider::new(&mut self.lenia.decay, 0.0..=0.04).text("阻尼"))
            .changed();
        ui.checkbox(&mut self.show_kernel_overlay, "显示卷积半径")
            .on_hover_text("在画面上显示被检查点的邻域范围。");
        if lenia_changed {
            self.pause_show_for_manual_interaction();
            self.sync_gpu_lenia_from_cpu();
            self.step_count = 0;
            self.mark_cpu_texture_dirty();
            self.refresh_lenia_inspection();
            self.reset_metric_history();
        }

        ui.separator();
        ui.heading("指标");
        if self.gpu_lenia_active() {
            ui.small("GPU 负责实时画面；指标使用同步的 CPU 参考场。");
        }
        let m = self.active_metrics();
        metric_bar(ui, "质量/活跃度", m.mass);
        metric_bar(ui, "熵", m.entropy);
        metric_bar(ui, "对称性", m.symmetry);
        metric_bar(ui, "稳定度", m.stability);
        metric_bar(ui, "生命力", m.vitality);
        ui.label(format!("活跃像素：{}", m.active));
        let phase = self.lenia_phase();
        ui.label(format!("阶段：{}", phase.label()));
        ui.small(phase.description());
        self.draw_metric_history(ui);

        ui.separator();
        self.draw_interpretability_panel(ui);

        if self.judge_mode || self.dev_diagnostics {
            ui.separator();
            self.draw_performance_diagnostics(ui);
        }

        ui.separator();
        ui.heading("数学框架");
        formula_card(
            ui,
            "u_next = clamp(u + dt*(G(K*u) - damping*u), 0, 1)",
            "u: 当前生命量；K*u: 卷积邻域；G: 增长函数；damping: 阻尼",
            "读取同一张场，计算邻域平均和增长响应，再生成下一帧。",
        );
        ui.small("艺术表达图只改变颜色映射，不改变底层数值场。");

        ui.separator();
        self.draw_rule_variant_panel(ui);
    }

    fn draw_central_explanation_bar(&self, ui: &mut egui::Ui) {
        let phase = self.lenia_phase();
        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(9, 13, 15))
            .show(ui, |ui| {
                if self.show_mode.enabled {
                    let scene = self.current_show_scene();
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(Color32::from_rgb(100, 232, 218), scene.chapter);
                        ui.strong(scene.title_zh);
                        ui.separator();
                        ui.label(scene.narration.core_question_zh);
                    });
                    ui.small(scene.narration.conclusion_zh);
                } else {
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(Color32::from_rgb(100, 232, 218), "手动实验");
                        ui.strong(self.active_preset.label());
                        ui.separator();
                        ui.label("每个点根据邻域卷积和增长函数更新，画面由同一数值场实时生成。");
                    });
                }
                ui.small(format!(
                    "当前阶段：{} · {}",
                    phase.label(),
                    phase.description()
                ));
            });
    }

    fn draw_compact_live_diagnostics(&self, ui: &mut egui::Ui) {
        let m = self.active_metrics();
        let region = self.active_region();
        ui.heading("当前证据");
        ui.label(format!(
            "阶段：{} · {}",
            self.lenia_phase().label(),
            self.lenia_phase().description()
        ));
        ui.label(format!(
            "质量 {:.3} · 熵 {:.3} · 稳定度 {:.3}",
            m.mass, m.entropy, m.stability
        ));
        if let Some((cx, cy)) = region.centroid {
            ui.label(format!(
                "活跃中心：({cx:.1}, {cy:.1}) · 漂移 ({:.2}, {:.2})",
                region.drift.0, region.drift.1
            ));
        } else {
            ui.label("活跃中心：暂无足够质量");
        }
        if let Some(inspection) = self.inspected_lenia {
            ui.label(format!(
                "检查点：u={:.3}, K*u={:.3}, G={:.3}, next={:.3}",
                inspection.value,
                inspection.convolution,
                inspection.growth,
                inspection.estimated_next
            ));
        }
    }

    fn draw_major_cases_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("主要情况");
        ui.small("这些案例覆盖 Lenia 的稳定、漂移、尺度、湍动和衰退。评委无需调参即可载入。");
        ui.separator();
        for case in major_cases() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    draw_case_swatch(ui, case);
                    ui.vertical(|ui| {
                        ui.strong(case.title_zh);
                        ui.colored_label(Color32::from_rgb(255, 219, 128), case.behavior_label_zh);
                        ui.label(case.explanation_zh);
                        ui.small(case.expected_outcome_zh);
                        if ui.button("载入并演示").clicked() {
                            self.load_major_case(case);
                        }
                    });
                });
            });
        }
    }

    fn draw_metric_history(&self, ui: &mut egui::Ui) {
        let desired = egui::vec2(ui.available_width(), 96.0);
        let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_rgb(6, 9, 11));
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, Color32::from_rgb(32, 48, 54)),
            egui::StrokeKind::Inside,
        );
        if self.metric_history.len() < 2 {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "等待指标样本",
                egui::TextStyle::Small.resolve(ui.style()),
                Color32::GRAY,
            );
            return;
        }
        draw_history_line(
            &painter,
            rect,
            &self.metric_history,
            |s| s.mass,
            Color32::from_rgb(100, 232, 218),
        );
        draw_history_line(
            &painter,
            rect,
            &self.metric_history,
            |s| s.entropy,
            Color32::from_rgb(255, 118, 168),
        );
        draw_history_line(
            &painter,
            rect,
            &self.metric_history,
            |s| s.stability,
            Color32::from_rgb(255, 219, 128),
        );
        painter.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            "质量 / 熵 / 稳定度",
            egui::TextStyle::Small.resolve(ui.style()),
            Color32::from_rgb(190, 205, 210),
        );
    }

    fn draw_interpretability_panel(&self, ui: &mut egui::Ui) {
        ui.heading("检查器");
        if let Some(inspection) = self.inspected_lenia {
            egui::Grid::new("lenia_inspection_grid")
                .num_columns(2)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label("坐标");
                    ui.label(format!("({}, {})", inspection.x, inspection.y));
                    ui.end_row();
                    ui.label("u 当前值");
                    ui.label(format!("{:.4}", inspection.value));
                    ui.end_row();
                    ui.label("上一帧");
                    ui.label(format!("{:.4}", inspection.previous));
                    ui.end_row();
                    ui.label("delta");
                    ui.label(format!("{:.4}", inspection.delta));
                    ui.end_row();
                    ui.label("梯度");
                    ui.label(format!("{:.4}", inspection.gradient));
                    ui.end_row();
                    ui.label("K*u");
                    ui.label(format!("{:.4}", inspection.convolution));
                    ui.end_row();
                    ui.label("G(K*u)");
                    ui.label(format!("{:.4}", inspection.growth));
                    ui.end_row();
                    ui.label("估计下一帧");
                    ui.label(format!("{:.4}", inspection.estimated_next));
                    ui.end_row();
                });
        } else {
            ui.label("把鼠标移到画面上可检查某一点的数学状态。");
        }

        ui.separator();
        ui.heading("卷积核");
        ui.label(format!(
            "半径 {} · 增长中心 {:.3} · 增长宽度 {:.3} · 阻尼 {:.4}",
            self.lenia.radius, self.lenia.growth_center, self.lenia.growth_width, self.lenia.decay
        ));
        let profile = self.lenia.kernel_profile(48);
        let desired = egui::vec2(ui.available_width(), 54.0);
        let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_rgb(6, 9, 11));
        let mut points = Vec::with_capacity(profile.len());
        for (i, value) in profile.iter().enumerate() {
            let t = i as f32 / (profile.len() - 1).max(1) as f32;
            points.push(egui::pos2(
                egui::lerp(rect.left() + 6.0..=rect.right() - 6.0, t),
                egui::lerp(
                    rect.bottom() - 6.0..=rect.top() + 6.0,
                    value.clamp(0.0, 1.0),
                ),
            ));
        }
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(1.6, Color32::from_rgb(100, 232, 218)),
        ));
    }

    fn draw_performance_diagnostics(&self, ui: &mut egui::Ui) {
        ui.heading("性能诊断");
        ui.label(format!("FPS {:.1}", self.performance.fps_estimate));
        ui.label(format!(
            "frame {:.2}ms · update {:.2}ms · render {:.2}ms · CPU sync {:.2}ms",
            self.performance.latest.frame_ms,
            self.performance.latest.update_ms,
            self.performance.latest.render_ms,
            self.performance.latest.cpu_sync_ms
        ));
        ui.label(format!(
            "后端：{} · 源场 {}x{}",
            self.backend_label(),
            self.lenia.size().0,
            self.lenia.size().1
        ));
        if let Some(gpu) = self.performance.gpu_grid {
            ui.label(format!(
                "GPU 网格：{}x{} · pending {}",
                gpu, gpu, self.performance.pending_steps
            ));
        }
        ui.label(format!(
            "CPU 同步间隔：{} · 指标样本 {}",
            self.gpu_cpu_sync_interval,
            self.metric_history.len()
        ));
    }

    fn draw_rule_variant_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("规则变量对照");
        ui.small("固定同一基线，只改变一个变量，比较数值指标差异。");
        egui::ComboBox::from_label("变量")
            .selected_text(self.comparison_parameter.label())
            .show_ui(ui, |ui| {
                for parameter in VariantParameter::ALL {
                    ui.selectable_value(
                        &mut self.comparison_parameter,
                        parameter,
                        parameter.label(),
                    );
                }
            });
        ui.add(
            egui::Slider::new(
                &mut self.comparison_value,
                self.comparison_parameter.range(),
            )
            .text("对照值"),
        );
        ui.add(egui::Slider::new(&mut self.comparison_steps, 8..=240).text("对照步数"));
        ui.horizontal(|ui| {
            if ui.button("记录基线").clicked() {
                self.capture_comparison_baseline();
            }
            if ui.button("应用变量").clicked() {
                self.pause_show_for_manual_interaction();
                self.apply_variant_to_current_lenia();
            }
            if ui.button("运行对照").clicked() {
                self.run_rule_variant_comparison();
            }
        });

        let Some(comparison) = &self.comparison_result else {
            return;
        };
        ui.separator();
        ui.label(format!(
            "{} = {:.4} · {} 步",
            comparison.parameter.label(),
            comparison.value,
            comparison.steps
        ));
        ui.label(format!(
            "质量差 {:.4} · 熵差 {:.4} · 稳定度差 {:.4}",
            comparison.variant_metrics.mass - comparison.baseline_metrics.mass,
            comparison.variant_metrics.entropy - comparison.baseline_metrics.entropy,
            comparison.variant_metrics.stability - comparison.baseline_metrics.stability
        ));

        let image_a = ColorImage::from_rgba_unmultiplied(
            [comparison.width, comparison.height],
            &comparison.baseline_pixels,
        );
        let image_b = ColorImage::from_rgba_unmultiplied(
            [comparison.width, comparison.height],
            &comparison.variant_pixels,
        );
        if self.comparison_baseline_texture.is_none() {
            self.comparison_baseline_texture = Some(ui.ctx().load_texture(
                "lenia-comparison-baseline",
                image_a,
                TextureOptions::LINEAR,
            ));
        }
        if self.comparison_variant_texture.is_none() {
            self.comparison_variant_texture = Some(ui.ctx().load_texture(
                "lenia-comparison-variant",
                image_b,
                TextureOptions::LINEAR,
            ));
        }
        ui.horizontal(|ui| {
            if let Some(texture) = &self.comparison_baseline_texture {
                ui.add(egui::Image::new((texture.id(), egui::vec2(110.0, 110.0))));
            }
            if let Some(texture) = &self.comparison_variant_texture {
                ui.add(egui::Image::new((texture.id(), egui::vec2(110.0, 110.0))));
            }
        });
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
        self.ensure_show_scene_applied();
        self.advance_show_mode(frame_delta);

        let mut update_duration = Duration::ZERO;
        let mut render_duration = Duration::ZERO;
        let mut cpu_sync_duration = Duration::ZERO;

        if self.running {
            self.tick_accumulator += frame_delta;
            let update_start = Instant::now();
            let mut batches = 0;
            while self.tick_accumulator >= TARGET_TICK && batches < MAX_UPDATE_BATCHES {
                if self.gpu_lenia_active() {
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
            if batches > 0 {
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
                ui.label(lenia_mode_label());
                ui.separator();
                ui.label(self.backend_label());
                ui.separator();
                ui.label(format!("{}x{}", grid_w, grid_h));
                ui.separator();
                ui.label(format!("种子 {}", self.active_seed()));
                ui.separator();
                ui.label(format!("步数 {}", self.step_count));
                ui.separator();
                ui.label(self.lenia_phase().label());
                if self.show_mode.enabled {
                    ui.separator();
                    ui.label(format!("演示 {}", self.current_show_scene().title_zh));
                }
            });
        });

        egui::SidePanel::left("left_controls")
            .resizable(false)
            .default_width(270.0)
            .show(ctx, |ui| self.draw_left_panel(ui));

        egui::SidePanel::right("right_parameters")
            .resizable(false)
            .default_width(360.0)
            .show(ctx, |ui| self.draw_right_panel(ui));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                self.draw_central_explanation_bar(ui);
                let available = ui.available_size();
                let square = (available.x.min(available.y) - 28.0).max(320.0);
                let size = egui::vec2(square, square);
                if self.gpu_lenia_active() {
                    let render_start = Instant::now();
                    egui::Frame::canvas(ui.style()).show(ui, |ui| {
                        let (rect, response) =
                            ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                        if let Some(gpu) = self.gpu_lenia.as_ref() {
                            gpu.update_params(lenia_params(&self.lenia), self.render_style);
                            let callback = gpu.paint_callback(rect);
                            ui.painter().add(egui::Shape::Callback(callback));
                        }
                        self.update_lenia_inspection_from_canvas(rect, &response);
                        self.apply_lenia_brush(rect, &response);
                        self.draw_lenia_inspection_overlay(ui.painter(), rect);
                        self.draw_active_region_overlay(ui.painter(), rect);
                    });
                    render_duration += render_start.elapsed();
                } else {
                    if self.cpu_texture_dirty || self.texture.is_none() {
                        let render_start = Instant::now();
                        let (w, h) = self.render_active();
                        let image = ColorImage::from_rgba_unmultiplied([w, h], &self.pixels);
                        if let Some(texture) = &mut self.texture {
                            texture.set(image, TextureOptions::LINEAR);
                        } else {
                            self.texture = Some(ctx.load_texture(
                                "peterMath-lenia-field",
                                image,
                                TextureOptions::LINEAR,
                            ));
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
                            self.update_lenia_inspection_from_canvas(rect, &response);
                            self.apply_lenia_brush(rect, &response);
                            self.draw_lenia_inspection_overlay(ui.painter(), rect);
                            self.draw_active_region_overlay(ui.painter(), rect);
                        });
                    }
                }
                ui.add_space(8.0);
                ui.small(format!(
                    "{} | {} | {} | 种子 {} | 步数 {}",
                    lenia_mode_label(),
                    self.render_style.label(),
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

impl ShowModeState {
    fn enabled_default() -> Self {
        Self {
            enabled: true,
            playing: true,
            finished: false,
            scene_index: 0,
            scene_elapsed: 0.0,
            total_elapsed: 0.0,
            applied_scene_index: None,
        }
    }
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
            Self::CoralDrift => "珊瑚衰退",
            Self::KernelRing => "卷积核环",
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
            Self::OrbitalField => "螺旋种子展示旋转梯度和柔性的卷积传递。",
            Self::TwinOrganisms => "两个团块展示同一连续场中的相互影响。",
            Self::CoralDrift => "分枝种子强调脊线生长、阻尼和边界衰退。",
            Self::KernelRing => "环形质量让径向邻域卷积核更容易看懂。",
            Self::SparseSoup => "低密度随机质量测试少量岛屿能否自组织。",
            Self::DenseBloom => "高密度质量会把场推向饱和、湍动或崩塌。",
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
            Self::RingSeed => "径向盖章，对应卷积核的圆形采样结构。",
            Self::TwinSeed => "成对质量会在同一规则下合并、排斥或绕行。",
            Self::ArcSeed => "局部弧线，用来观察不对称梯度流。",
            Self::NoisePatch => "带种子的微结构，用来激发局部不稳定和纹理。",
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
            Self::Sparse => "稀疏",
            Self::Blooming => "快速增长",
            Self::Drifting => "漂移",
            Self::Stabilizing => "稳定",
            Self::Turbulent => "湍动",
            Self::Dense => "密集",
            Self::Fading => "衰退",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Sparse => "场质量很低，只有少量区域还可能组织起来",
            Self::Blooming => "质量或生命力正在上升，结构正在形成",
            Self::Drifting => "结构已经存在，同时仍在缓慢运动",
            Self::Stabilizing => "连续帧很接近，运动正在稳定",
            Self::Turbulent => "熵和变化量较高，边界正在竞争",
            Self::Dense => "场质量较高，增长接近饱和",
            Self::Fading => "质量和生命力下降，结构正在衰退",
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

impl MainInfoTab {
    fn label(self) -> &'static str {
        match self {
            Self::ShowNarration => "演示讲解",
            Self::MajorCases => "主要情况",
            Self::ParametersDiagnostics => "参数/诊断",
        }
    }
}

impl MajorCaseId {
    fn id(self) -> &'static str {
        match self {
            Self::OrbitalField => "orbital_field",
            Self::TwinOrganisms => "twin_organisms",
            Self::KernelRing => "kernel_ring",
            Self::SparseSoup => "sparse_soup",
            Self::DenseBloom => "dense_bloom",
            Self::CoralFading => "coral_fading",
        }
    }

    fn from_preset(preset: LeniaPreset) -> Option<Self> {
        Some(match preset {
            LeniaPreset::OrbitalField => Self::OrbitalField,
            LeniaPreset::TwinOrganisms => Self::TwinOrganisms,
            LeniaPreset::KernelRing => Self::KernelRing,
            LeniaPreset::SparseSoup => Self::SparseSoup,
            LeniaPreset::DenseBloom => Self::DenseBloom,
            LeniaPreset::CoralDrift => Self::CoralFading,
        })
    }
}

impl ShowSceneId {
    fn id(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::OrbitalField => "orbital_field",
            Self::TwinOrganisms => "twin_organisms",
            Self::KernelRing => "kernel_ring",
            Self::DenseBloom => "dense_bloom",
            Self::CoralFading => "coral_fading",
            Self::EvidenceSummary => "evidence_summary",
        }
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

fn lenia_mode_label() -> &'static str {
    "连续生命场 Lenia"
}

fn show_total_duration_secs() -> f32 {
    show_scenes().iter().map(|scene| scene.duration_secs).sum()
}

fn show_elapsed_before_scene(scene_index: usize) -> f32 {
    show_scenes()
        .iter()
        .take(scene_index)
        .map(|scene| scene.duration_secs)
        .sum()
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
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/Deng.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/NISC18030.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
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

fn explanation_row(ui: &mut egui::Ui, label: &str, text: &str) {
    ui.vertical(|ui| {
        ui.colored_label(Color32::from_rgb(100, 232, 218), label);
        ui.label(text);
    });
}

fn formula_card(ui: &mut egui::Ui, formula: &str, variables: &str, algorithm: &str) {
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(8, 12, 14))
        .show(ui, |ui| {
            ui.colored_label(Color32::from_rgb(255, 219, 128), "规则公式");
            ui.monospace(formula);
            ui.separator();
            ui.small(format!("变量：{variables}"));
            ui.small(format!("算法：{algorithm}"));
        });
}

fn draw_case_swatch(ui: &mut egui::Ui, case: MajorCase) {
    let desired = egui::vec2(52.0, 40.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(5, 8, 10));
    let accent = match case.id {
        MajorCaseId::DenseBloom => Color32::from_rgb(255, 118, 168),
        MajorCaseId::CoralFading => Color32::from_rgb(255, 219, 128),
        _ => Color32::from_rgb(100, 232, 218),
    };
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, accent),
        egui::StrokeKind::Inside,
    );
    painter.circle_stroke(rect.center(), 13.0, egui::Stroke::new(2.0, accent));
    painter.circle_filled(
        egui::pos2(rect.center().x + 7.0, rect.center().y - 4.0),
        4.0,
        Color32::from_rgb(255, 118, 168),
    );
}

fn show_mode_json_from_state(show_mode: &ShowModeState) -> serde_json::Value {
    if !show_mode.enabled {
        return serde_json::Value::Null;
    }
    let scenes = show_scenes();
    let scene = scenes[show_mode.scene_index.min(scenes.len() - 1)];
    let total_duration = show_total_duration_secs();
    json!({
        "enabled": show_mode.enabled,
        "playing": show_mode.playing,
        "finished": show_mode.finished,
        "scene_id": scene.id.id(),
        "chapter": scene.chapter,
        "scene_title_zh": scene.title_zh,
        "case_id": scene.case_id.map(|id| id.id()),
        "hold_on_finish": scene.hold_on_finish,
        "scene_index": show_mode.scene_index,
        "scene_elapsed_seconds": show_mode.scene_elapsed,
        "scene_duration_seconds": scene.duration_secs,
        "scene_progress": (show_mode.scene_elapsed / scene.duration_secs).clamp(0.0, 1.0),
        "total_elapsed_seconds": show_mode.total_elapsed,
        "total_duration_seconds": total_duration,
        "total_progress": (show_mode.total_elapsed / total_duration).clamp(0.0, 1.0),
        "core_question_zh": scene.narration.core_question_zh,
        "initial_zh": scene.narration.initial_zh,
        "parameters_zh": scene.narration.parameters_zh,
        "formula_ascii": scene.narration.formula_ascii,
        "variables_zh": scene.narration.variables_zh,
        "algorithm_zh": scene.narration.algorithm_zh,
        "why_zh": scene.narration.why_zh,
        "conclusion_zh": scene.narration.conclusion_zh,
    })
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

fn draw_history_line(
    painter: &egui::Painter,
    rect: egui::Rect,
    samples: &[MetricHistorySample],
    value: impl Fn(&MetricHistorySample) -> f32,
    color: Color32,
) {
    let mut points = Vec::with_capacity(samples.len());
    for (i, sample) in samples.iter().enumerate() {
        let t = if samples.len() > 1 {
            i as f32 / (samples.len() - 1) as f32
        } else {
            0.0
        };
        let x = egui::lerp(rect.left() + 5.0..=rect.right() - 5.0, t);
        let y = egui::lerp(
            rect.bottom() - 8.0..=rect.top() + 18.0,
            value(sample).clamp(0.0, 1.0),
        );
        points.push(egui::pos2(x, y));
    }
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.3, color)));
}

fn duration_ms(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn show_scenes_have_complete_three_minute_script() {
        let scenes = show_scenes();
        let total: f32 = scenes.iter().map(|scene| scene.duration_secs).sum();
        assert!(
            (170.0..=190.0).contains(&total),
            "unexpected total duration {total}"
        );

        let mut ids = HashSet::new();
        for scene in scenes {
            assert!(ids.insert(scene.id.id()));
            assert!(!scene.chapter.is_empty());
            assert!(!scene.title_zh.is_empty());
            assert!(!scene.narration.core_question_zh.is_empty());
            assert!(!scene.narration.initial_zh.is_empty());
            assert!(!scene.narration.parameters_zh.is_empty());
            assert!(!scene.narration.formula_ascii.is_empty());
            assert!(scene.narration.formula_ascii.is_ascii());
            assert!(!scene.narration.variables_zh.is_empty());
            assert!(!scene.narration.algorithm_zh.is_empty());
            assert!(!scene.narration.why_zh.is_empty());
            assert!(!scene.narration.conclusion_zh.is_empty());
            assert!((1..=8).contains(&scene.step_rate));
        }
        assert!(scenes.last().is_some_and(|scene| scene.hold_on_finish));
    }

    #[test]
    fn major_cases_are_lenia_presets_with_complete_metadata() {
        let mut ids = HashSet::new();
        for case in major_cases() {
            assert!(ids.insert(case.id.id()));
            assert!(!case.title_zh.is_empty());
            assert!(!case.behavior_label_zh.is_empty());
            assert!(!case.explanation_zh.is_empty());
            assert!(!case.expected_outcome_zh.is_empty());
            assert!((1..=8).contains(&case.step_rate));
        }
        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn show_mode_export_json_describes_lenia_scene() {
        let mut state = ShowModeState::enabled_default();
        state.scene_index = 3;
        state.scene_elapsed = 5.0;
        state.total_elapsed = show_elapsed_before_scene(3) + 5.0;
        let value = show_mode_json_from_state(&state);

        assert_eq!(value["scene_id"], ShowSceneId::KernelRing.id());
        assert_eq!(value["scene_title_zh"], "卷积核环：半径决定形体尺度");
        assert!(value["formula_ascii"].as_str().unwrap().is_ascii());
        assert!(value["variables_zh"].as_str().unwrap().contains("radius"));
        assert!(value["total_progress"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn show_mode_completion_keeps_enabled_until_exit() {
        let mut state = ShowModeState::enabled_default();
        state.scene_index = show_scenes().len() - 1;
        state.scene_elapsed = show_scenes().last().unwrap().duration_secs;
        state.total_elapsed = show_total_duration_secs();
        state.playing = false;
        state.finished = true;
        let value = show_mode_json_from_state(&state);
        assert_eq!(value["enabled"], true);
        assert_eq!(value["playing"], false);
        assert_eq!(value["finished"], true);
        assert_eq!(value["hold_on_finish"], true);
    }
}
