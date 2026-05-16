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
    mode: SimMode,
    render_style: RenderStyle,
    lenia: LeniaSim,
    reaction: ReactionDiffusionSim,
    life: LifeSim,
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
    parameters: Value,
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
    QuickOpening,
    QuickLife,
    QuickReaction,
    QuickLenia,
    QuickComparison,
    LifeStillLifes,
    LifeOscillators,
    LifeGlider,
    ReactionSpots,
    ReactionLabyrinth,
    ReactionWaves,
    ReactionMitosis,
    LeniaOrbital,
    LeniaTwin,
    LeniaKernel,
    LeniaDense,
    FinalSummary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MainInfoTab {
    ShowNarration,
    MajorCases,
    ParametersDiagnostics,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MajorCaseId {
    LifeStillLifes,
    LifeOscillators,
    LifeGlider,
    ReactionSpots,
    ReactionLabyrinth,
    ReactionWaves,
    ReactionMitosis,
    LeniaOrbital,
    LeniaTwin,
    LeniaKernel,
    LeniaDense,
    LeniaFading,
}

#[derive(Clone, Copy)]
enum ShowSetup {
    Lenia(LeniaPreset),
    Reaction(&'static str),
    Life(&'static str),
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
    mode: SimMode,
    render_style: RenderStyle,
    step_rate: usize,
    setup: ShowSetup,
    narration: ShowNarration,
    case_id: Option<MajorCaseId>,
    hold_on_finish: bool,
}

#[derive(Clone, Copy)]
struct MajorCase {
    id: MajorCaseId,
    title_zh: &'static str,
    behavior_label_zh: &'static str,
    mode: SimMode,
    render_style: RenderStyle,
    step_rate: usize,
    setup: ShowSetup,
    explanation_zh: &'static str,
    expected_outcome_zh: &'static str,
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

impl ShowSceneId {
    fn id(self) -> &'static str {
        match self {
            Self::QuickOpening => "quick_opening",
            Self::QuickLife => "quick_life",
            Self::QuickReaction => "quick_reaction",
            Self::QuickLenia => "quick_lenia",
            Self::QuickComparison => "quick_comparison",
            Self::LifeStillLifes => "life_still_lifes",
            Self::LifeOscillators => "life_oscillators",
            Self::LifeGlider => "life_glider",
            Self::ReactionSpots => "reaction_spots",
            Self::ReactionLabyrinth => "reaction_labyrinth",
            Self::ReactionWaves => "reaction_waves",
            Self::ReactionMitosis => "reaction_mitosis",
            Self::LeniaOrbital => "lenia_orbital_field",
            Self::LeniaTwin => "lenia_twin_organisms",
            Self::LeniaKernel => "lenia_kernel_ring",
            Self::LeniaDense => "lenia_dense_bloom",
            Self::FinalSummary => "final_summary",
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
            Self::LifeStillLifes => "life_still_lifes",
            Self::LifeOscillators => "life_oscillators",
            Self::LifeGlider => "life_glider",
            Self::ReactionSpots => "reaction_spots",
            Self::ReactionLabyrinth => "reaction_labyrinth",
            Self::ReactionWaves => "reaction_waves",
            Self::ReactionMitosis => "reaction_mitosis",
            Self::LeniaOrbital => "lenia_orbital",
            Self::LeniaTwin => "lenia_twin",
            Self::LeniaKernel => "lenia_kernel_ring",
            Self::LeniaDense => "lenia_dense_bloom",
            Self::LeniaFading => "lenia_fading",
        }
    }
}

fn show_scenes() -> [ShowScene; 17] {
    [
        ShowScene {
            id: ShowSceneId::QuickOpening,
            chapter: "快速总览",
            title_zh: "开场：局部规则生成整体结构",
            duration_secs: 15.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::OrbitalField),
            narration: ShowNarration {
                core_question_zh: "这个项目到底在证明什么？",
                initial_zh: "初始图形：一组柔和的连续场种子，作为三种系统的共同入口。",
                parameters_zh: "参数条件：自动演示会依次切换系统、预设和速度，评委无需先调参。",
                formula_ascii: "local rule + repeated update -> global structure",
                variables_zh: "local rule 是每个点的局部规则；update 是重复迭代；global structure 是最后出现的整体形态。",
                algorithm_zh: "算法步骤：初始化一个场；每一帧按同一规则更新；记录指标；用颜色解释同一份数据。",
                why_zh: "每个点只根据附近邻居更新，但重复很多次后会出现整体形态。",
                conclusion_zh: "观察目标：把项目理解为一组可运行的数学实验，而不是预渲染动画。",
            },
            case_id: None,
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::QuickLife,
            chapter: "快速总览",
            title_zh: "生命游戏：离散结构一眼看懂",
            duration_secs: 18.0,
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("structure_showcase"),
            narration: ShowNarration {
                core_question_zh: "只有活/死两种格子，为什么会出现稳定、闪烁和移动？",
                initial_zh: "初始图形：方块、蜂巢、面包、闪烁器、蟾蜍、信标和滑翔机分区摆放。",
                parameters_zh: "参数条件：96x96 教学网格，B3/S23 固定规则，演化速度 1 步/批次。",
                formula_ascii: "dead+n=3->alive; alive+n=2 or 3->alive; else->dead",
                variables_zh: "n 是周围 8 个邻居里的活细胞数量；alive/dead 是当前格子的状态。",
                algorithm_zh: "算法步骤：数邻居；同时决定所有格子的下一步；用尾迹显示移动方向。",
                why_zh: "每个格子只看周围 8 个邻居，孤立、拥挤和刚好平衡会产生不同命运。",
                conclusion_zh: "预期结果：同时看到静止结构、周期结构和会移动的滑翔机。",
            },
            case_id: Some(MajorCaseId::LifeGlider),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::QuickReaction,
            chapter: "快速总览",
            title_zh: "反应扩散：化学竞争生成纹理",
            duration_secs: 18.0,
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("labyrinth"),
            narration: ShowNarration {
                core_question_zh: "两种虚拟化学物质为什么会长出斑点和迷宫？",
                initial_zh: "初始图形：许多 B 物质扰动点散布在 A 物质背景中。",
                parameters_zh: "参数条件：feed=0.029，kill=0.057，较密集种子，演化速度 8 步/批次。",
                formula_ascii: "b_next = b + dt*(Db*laplace(b) + a*b*b - (kill+feed)*b)",
                variables_zh: "a/b 是两种物质浓度；laplace 表示向周围扩散；feed/kill 表示补给和消耗。",
                algorithm_zh: "算法步骤：扩散 A 和 B；计算反应 a*b*b；补给 A；消耗 B；重复后形成边界。",
                why_zh: "B 会扩散也会消耗 A；反应和扩散速度接近时，边界会互相追逐。",
                conclusion_zh: "预期结果：斑点逐渐连成弯曲通道，形成迷宫式不稳定边界。",
            },
            case_id: Some(MajorCaseId::ReactionLabyrinth),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::QuickLenia,
            chapter: "快速总览",
            title_zh: "Lenia：连续生命场",
            duration_secs: 22.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::OrbitalField),
            narration: ShowNarration {
                core_question_zh: "如果格子不再只有活/死，而是 0 到 1 的连续生命量，会发生什么？",
                initial_zh: "初始图形：螺旋分布的连续质量种子，形成可运动的柔性场。",
                parameters_zh: "参数条件：GPU 优先，radius=9，growth center=0.31，growth width=0.052。",
                formula_ascii: "u_next = clamp(u + dt*(G(K*u) - damping*u), 0, 1)",
                variables_zh: "u 是当前生命场；K*u 是邻域平均；G 是增长函数；damping 是衰减。",
                algorithm_zh: "算法步骤：卷积邻域；计算增长；扣除阻尼；限制到 0..1；用梯度和等值线渲染。",
                why_zh: "每个点听取一圈邻域平均值，刚好接近增长中心时增强，否则衰减。",
                conclusion_zh: "预期结果：边界、脊线和质心漂移共同呈现类似生命的连续运动。",
            },
            case_id: Some(MajorCaseId::LeniaOrbital),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::QuickComparison,
            chapter: "快速总览",
            title_zh: "三系统对比：同一个思想的三种形式",
            duration_secs: 17.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::TwinOrganisms),
            narration: ShowNarration {
                core_question_zh: "为什么这不是三段普通动画，而是三种数学生命实验？",
                initial_zh: "初始图形：从离散细胞、化学浓度到连续生命场，数据类型逐层变丰富。",
                parameters_zh: "参数条件：每个系统都有确定性种子和可调规则，导出 JSON 可复现实验。",
                formula_ascii: "state[t+1] = rule(state[t], neighbors, parameters)",
                variables_zh: "state 是当前状态；neighbors 是局部邻域；parameters 是规则旋钮。",
                algorithm_zh: "算法步骤：选择系统；载入代表性初始状态；运行规则；比较指标和图像。",
                why_zh: "三者都说明复杂性不一定来自复杂代码，而可能来自简单规则被反复执行。",
                conclusion_zh: "预期结果：评委先记住共同核心，再进入每个系统的深度章节。",
            },
            case_id: Some(MajorCaseId::LeniaTwin),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LifeStillLifes,
            chapter: "生命游戏深度",
            title_zh: "生命游戏：稳定结构",
            duration_secs: 35.0,
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("still_lifes"),
            narration: ShowNarration {
                core_question_zh: "什么样的局部结构会保持不变？",
                initial_zh: "初始图形：多个方块、蜂巢和面包分散排列，彼此距离足够远。",
                parameters_zh: "参数条件：B3/S23；低速演化；教学渲染显示活细胞边界。",
                formula_ascii: "alive+n=2 or 3->alive; dead+n=3->alive",
                variables_zh: "n 是邻居数量；稳定结构的每个活细胞都刚好有 2 或 3 个邻居。",
                algorithm_zh: "算法步骤：每步重新数邻居；静物因为下一步仍等于当前状态而停住。",
                why_zh: "方块、蜂巢、面包里的每个细胞都处在局部平衡，不会出生也不会死亡。",
                conclusion_zh: "观察结论：简单规则可以产生稳定的局部对象，像数学里的定点。",
            },
            case_id: Some(MajorCaseId::LifeStillLifes),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LifeOscillators,
            chapter: "生命游戏深度",
            title_zh: "生命游戏：周期振荡",
            duration_secs: 35.0,
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("oscillators"),
            narration: ShowNarration {
                core_question_zh: "为什么有些结构不是静止，而是来回切换？",
                initial_zh: "初始图形：闪烁器、蟾蜍和信标重复摆放，方便看到不同周期。",
                parameters_zh: "参数条件：同一 B3/S23 规则；每 250ms 左右推进一次，保留尾迹。",
                formula_ascii: "state[t+period] = state[t]",
                variables_zh: "period 是周期；state[t] 是第 t 步的整张网格。",
                algorithm_zh: "算法步骤：记录状态哈希；如果若干步后哈希重复，就检测到周期。",
                why_zh: "局部出生和死亡互相接力，结构在有限状态之间循环。",
                conclusion_zh: "观察结论：离散系统不只会稳定，也会形成清晰的周期时间结构。",
            },
            case_id: Some(MajorCaseId::LifeOscillators),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LifeGlider,
            chapter: "生命游戏深度",
            title_zh: "生命游戏：滑翔机漂移",
            duration_secs: 35.0,
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("glider_lane"),
            narration: ShowNarration {
                core_question_zh: "没有任何速度变量，图案为什么会移动？",
                initial_zh: "初始图形：多组滑翔机沿对角线排列，尾迹显示运动路径。",
                parameters_zh: "参数条件：B3/S23；教学网格；自动检测质心漂移。",
                formula_ascii: "glider[t+4] = shift(glider[t], +1, +1)",
                variables_zh: "shift 表示整个图案平移；+1,+1 表示向右下移动一个格子。",
                algorithm_zh: "算法步骤：运行 4 步；检测滑翔机组件；比较质心位置变化。",
                why_zh: "滑翔机的出生/死亡模式每 4 步复制自己，但整体位置移动一个格子。",
                conclusion_zh: "观察结论：移动可以从局部规则中涌现出来，不需要写运动方程。",
            },
            case_id: Some(MajorCaseId::LifeGlider),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::ReactionSpots,
            chapter: "反应扩散深度",
            title_zh: "反应扩散：斑点膜",
            duration_secs: 45.0,
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("spots"),
            narration: ShowNarration {
                core_question_zh: "为什么局部扰动会长成斑点，而不是均匀扩散掉？",
                initial_zh: "初始图形：许多小圆形 B 物质种子嵌入 A 物质背景。",
                parameters_zh: "参数条件：feed=0.042，kill=0.060，B 物质较容易形成孤立斑点。",
                formula_ascii: "A + 2B -> 3B; B decays by (kill+feed)*B",
                variables_zh: "A 是背景物质；B 是激活物质；feed/kill 决定补给和消耗的平衡。",
                algorithm_zh: "算法步骤：先局部反应放大 B；再扩散抹平边界；重复后斑点稳定。",
                why_zh: "反应让斑点中心变强，扩散让边缘变宽，消耗阻止它无限长大。",
                conclusion_zh: "观察结论：斑点是增长、扩散、消耗三者平衡后的形态。",
            },
            case_id: Some(MajorCaseId::ReactionSpots),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::ReactionLabyrinth,
            chapter: "反应扩散深度",
            title_zh: "反应扩散：迷宫边界",
            duration_secs: 45.0,
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("labyrinth"),
            narration: ShowNarration {
                core_question_zh: "为什么同一公式会从斑点变成迷宫？",
                initial_zh: "初始图形：密集扰动让多个生长前沿互相碰撞。",
                parameters_zh: "参数条件：feed=0.029，kill=0.057，前沿更容易连接成曲线。",
                formula_ascii: "b_next = b + dt*(Db*laplace(b) + a*b*b - (kill+feed)*b)",
                variables_zh: "Db 控制 B 扩散；a*b*b 控制自催化增长；kill+feed 控制衰减。",
                algorithm_zh: "算法步骤：扩散推动前沿；反应强化边界；相邻前沿相遇后形成通道。",
                why_zh: "参数把系统推向边界竞争区，孤立斑点会连成连续的迷宫墙。",
                conclusion_zh: "观察结论：微小参数变化能导致完全不同的宏观图案。",
            },
            case_id: Some(MajorCaseId::ReactionLabyrinth),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::ReactionWaves,
            chapter: "反应扩散深度",
            title_zh: "反应扩散：波纹传播",
            duration_secs: 45.0,
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("waves"),
            narration: ShowNarration {
                core_question_zh: "反应扩散如何表现出像水波一样的传播？",
                initial_zh: "初始图形：几条环形和斜向 B 物质带，制造可见波前。",
                parameters_zh: "参数条件：feed=0.026，kill=0.051，扩散前沿更平滑。",
                formula_ascii: "laplace(B) spreads; reaction sharpens the wave front",
                variables_zh: "laplace(B) 是扩散项；reaction 是自催化项；wave front 是浓度快速变化的边界。",
                algorithm_zh: "算法步骤：把条带作为扰动；每步扩散到邻域；反应项保留亮边。",
                why_zh: "扩散把信号向外推，反应又让局部前沿变清晰，于是出现波纹。",
                conclusion_zh: "观察结论：同一数值网格可以生成斑点、迷宫，也可以生成传播波。",
            },
            case_id: Some(MajorCaseId::ReactionWaves),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::ReactionMitosis,
            chapter: "反应扩散深度",
            title_zh: "反应扩散：细胞分裂",
            duration_secs: 45.0,
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("mitosis"),
            narration: ShowNarration {
                core_question_zh: "为什么一个斑块会像细胞一样伸长、分裂或破碎？",
                initial_zh: "初始图形：少量圆形扰动点，像培养皿里的化学斑块。",
                parameters_zh: "参数条件：feed=0.0367，kill=0.0649，补给和消耗更适合边界断裂。",
                formula_ascii: "a_next = a + dt*(Da*laplace(a) - a*b*b + feed*(1-a))",
                variables_zh: "Da 是 A 的扩散率；a*b*b 会消耗 A 并制造更多 B。",
                algorithm_zh: "算法步骤：圆形斑块增长；边界被扩散拉长；局部消耗导致分裂。",
                why_zh: "当中心和边缘的补给不平衡，斑块会从圆形变成分裂形态。",
                conclusion_zh: "观察结论：参数不是装饰旋钮，而是控制形态命运的数学条件。",
            },
            case_id: Some(MajorCaseId::ReactionMitosis),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LeniaOrbital,
            chapter: "Lenia 深度",
            title_zh: "Lenia：轨道生命场",
            duration_secs: 45.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::OrbitalField),
            narration: ShowNarration {
                core_question_zh: "连续场如何产生看起来像柔性生命的运动？",
                initial_zh: "初始图形：螺旋分布的连续质量种子。",
                parameters_zh: "参数条件：radius=9，growth center=0.31，growth width=0.052，damping 较低。",
                formula_ascii: "u_next = clamp(u + dt*(G(K*u) - damping*u), 0, 1)",
                variables_zh: "u 是生命量；K 是卷积核；G 是钟形增长函数；damping 防止质量无限积累。",
                algorithm_zh: "算法步骤：计算邻域平均 K*u；映射为增长 G；更新 u；提取梯度和轮廓绘图。",
                why_zh: "当局部邻域接近增长中心，边缘会被推着移动；过高或过低都会衰减。",
                conclusion_zh: "观察结论：形体的美来自卷积、增长函数和阻尼之间的平衡。",
            },
            case_id: Some(MajorCaseId::LeniaOrbital),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LeniaTwin,
            chapter: "Lenia 深度",
            title_zh: "Lenia：双生命体相互影响",
            duration_secs: 45.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::TwinOrganisms),
            narration: ShowNarration {
                core_question_zh: "两个连续生命结构靠近时，会互相吸引、避让还是瓦解？",
                initial_zh: "初始图形：两个相近的连续场种子，质量分布并不完全相同。",
                parameters_zh: "参数条件：双生命体预设，使用相同卷积核和增长中心。",
                formula_ascii: "neighbor influence = K*u",
                variables_zh: "K*u 同时包含自己和邻居的质量，因此两个结构会通过场相互影响。",
                algorithm_zh: "算法步骤：同时卷积整张场；两个结构的邻域重叠后改变增长响应。",
                why_zh: "连续场没有硬边界，两个生命体的影响会在邻域核里混合。",
                conclusion_zh: "观察结论：Lenia 的交互不是碰撞脚本，而是同一场方程的自然结果。",
            },
            case_id: Some(MajorCaseId::LeniaTwin),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LeniaKernel,
            chapter: "Lenia 深度",
            title_zh: "Lenia：卷积核环",
            duration_secs: 45.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::KernelRing),
            narration: ShowNarration {
                core_question_zh: "为什么卷积核半径会决定形体尺度？",
                initial_zh: "初始图形：环状生命场，边界接近卷积核的有效尺度。",
                parameters_zh: "参数条件：kernel ring 预设，观察半径圈和场边界的关系。",
                formula_ascii: "K*u = weighted average inside a radius",
                variables_zh: "K 是权重；radius 决定每个点能听到多远的邻居。",
                algorithm_zh: "算法步骤：给邻域不同距离分配权重；用结果判断增长或衰减。",
                why_zh: "如果半径太小，形体破碎；半径太大，局部细节被平均掉。",
                conclusion_zh: "观察结论：数学核不仅影响速度，也直接塑造作品的视觉尺度。",
            },
            case_id: Some(MajorCaseId::LeniaKernel),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::LeniaDense,
            chapter: "Lenia 深度",
            title_zh: "Lenia：密集开花与湍动",
            duration_secs: 45.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::DenseBloom),
            narration: ShowNarration {
                core_question_zh: "当生命场太满时，美会变成湍动还是衰退？",
                initial_zh: "初始图形：高密度连续场，让许多区域同时接近增长条件。",
                parameters_zh: "参数条件：dense bloom 预设，较宽增长窗口和较高初始质量。",
                formula_ascii: "growth = G(K*u); too much mass -> saturation or decay",
                variables_zh: "growth 是增长响应；mass 是场中总生命量；saturation 是过密后的饱和。",
                algorithm_zh: "算法步骤：快速增长；局部竞争；指标显示质量、熵、稳定度变化。",
                why_zh: "当太多点同时增长，局部竞争会增强，系统可能饱和、湍动或衰退。",
                conclusion_zh: "观察结论：美不只是漂亮画面，也包括系统接近失衡时的数学张力。",
            },
            case_id: Some(MajorCaseId::LeniaDense),
            hold_on_finish: false,
        },
        ShowScene {
            id: ShowSceneId::FinalSummary,
            chapter: "证据与总结",
            title_zh: "总结：本项目证明了什么",
            duration_secs: 45.0,
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::CoralDrift),
            narration: ShowNarration {
                core_question_zh: "评委应从这个作品带走什么结论？",
                initial_zh: "初始图形：三类系统已经依次展示，最后停在可导出证据的状态。",
                parameters_zh: "参数条件：所有实验都有确定性种子、公式、指标和导出 JSON。",
                formula_ascii: "visible artwork = reproducible simulation + interpretation layer",
                variables_zh: "reproducible 表示可复现；simulation 是数值实验；interpretation 是解释层。",
                algorithm_zh: "算法步骤：载入案例；运行规则；解释变量；比较指标；导出证据包。",
                why_zh: "代码不是只画图，而是在同一界面中连接规则、形态、指标和解释。",
                conclusion_zh: "总结结论：数学规则可以产生可观察、可解释、可复现的计算艺术。",
            },
            case_id: Some(MajorCaseId::LeniaFading),
            hold_on_finish: true,
        },
    ]
}

fn show_total_duration_secs() -> f32 {
    show_scenes().iter().map(|scene| scene.duration_secs).sum()
}

fn major_cases() -> [MajorCase; 12] {
    [
        MajorCase {
            id: MajorCaseId::LifeStillLifes,
            title_zh: "生命游戏：静物",
            behavior_label_zh: "稳定",
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("still_lifes"),
            explanation_zh: "每个活细胞都有 2 或 3 个邻居，下一步仍保持原状。",
            expected_outcome_zh: "方块、蜂巢和面包会停住，说明规则存在稳定定点。",
        },
        MajorCase {
            id: MajorCaseId::LifeOscillators,
            title_zh: "生命游戏：振荡器",
            behavior_label_zh: "周期",
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("oscillators"),
            explanation_zh: "出生和死亡在几个状态间循环，形成可检测的周期。",
            expected_outcome_zh: "闪烁器、蟾蜍和信标会反复切换，尾迹显示周期变化。",
        },
        MajorCase {
            id: MajorCaseId::LifeGlider,
            title_zh: "生命游戏：滑翔机通道",
            behavior_label_zh: "漂移",
            mode: SimMode::GameOfLife,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Life("glider_lane"),
            explanation_zh: "同一图形每 4 步复制一次，但整体平移一个格子。",
            expected_outcome_zh: "滑翔机会沿对角线移动，质心漂移指标同步变化。",
        },
        MajorCase {
            id: MajorCaseId::ReactionSpots,
            title_zh: "反应扩散：斑点膜",
            behavior_label_zh: "稳定/分岔",
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("spots"),
            explanation_zh: "反应放大局部 B，扩散拉宽边界，消耗限制斑点大小。",
            expected_outcome_zh: "许多亮斑逐渐形成，边界清晰但不会无限扩张。",
        },
        MajorCase {
            id: MajorCaseId::ReactionLabyrinth,
            title_zh: "反应扩散：迷宫",
            behavior_label_zh: "不稳定边界",
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("labyrinth"),
            explanation_zh: "低 feed 与中等 kill 让前沿相互连接，斑点转成通道。",
            expected_outcome_zh: "几秒内出现弯曲迷宫边界，变化量和活跃面积上升。",
        },
        MajorCase {
            id: MajorCaseId::ReactionWaves,
            title_zh: "反应扩散：波纹",
            behavior_label_zh: "传播",
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("waves"),
            explanation_zh: "扩散把浓度向外推，反应项保持前沿明亮。",
            expected_outcome_zh: "条带和圆环会展开成波纹，显示局部浓度传播。",
        },
        MajorCase {
            id: MajorCaseId::ReactionMitosis,
            title_zh: "反应扩散：分裂",
            behavior_label_zh: "分裂",
            mode: SimMode::ReactionDiffusion,
            render_style: RenderStyle::Artistic,
            step_rate: 8,
            setup: ShowSetup::Reaction("mitosis"),
            explanation_zh: "补给和消耗的平衡让圆形斑块伸长并局部分裂。",
            expected_outcome_zh: "少量斑块会膨胀、拉伸或断裂，形成另一类纹理。",
        },
        MajorCase {
            id: MajorCaseId::LeniaOrbital,
            title_zh: "Lenia：轨道场",
            behavior_label_zh: "漂移",
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::OrbitalField),
            explanation_zh: "卷积核把局部平均转成增长响应，边界不断被推移。",
            expected_outcome_zh: "柔性轮廓、脊线和质心漂移形成连续生命感。",
        },
        MajorCase {
            id: MajorCaseId::LeniaTwin,
            title_zh: "Lenia：双生命体",
            behavior_label_zh: "相互作用",
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::TwinOrganisms),
            explanation_zh: "两个结构共享同一连续场，邻域影响会互相叠加。",
            expected_outcome_zh: "两个柔性形体会靠近、变形或避让，显示场的耦合。",
        },
        MajorCase {
            id: MajorCaseId::LeniaKernel,
            title_zh: "Lenia：核环",
            behavior_label_zh: "尺度",
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::KernelRing),
            explanation_zh: "卷积核半径决定每个点能听到多远的邻居。",
            expected_outcome_zh: "环状结构展示半径、邻域平均和视觉尺度之间的关系。",
        },
        MajorCase {
            id: MajorCaseId::LeniaDense,
            title_zh: "Lenia：密集开花",
            behavior_label_zh: "湍动",
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::DenseBloom),
            explanation_zh: "大量区域同时接近增长条件，局部竞争增强。",
            expected_outcome_zh: "从快速增长到饱和或湍动，指标显示稳定度下降。",
        },
        MajorCase {
            id: MajorCaseId::LeniaFading,
            title_zh: "Lenia：珊瑚衰退",
            behavior_label_zh: "衰退",
            mode: SimMode::Lenia,
            render_style: RenderStyle::Artistic,
            step_rate: 1,
            setup: ShowSetup::Lenia(LeniaPreset::CoralDrift),
            explanation_zh: "增长窗口和阻尼让部分结构无法长期维持质量。",
            expected_outcome_zh: "局部纹理会漂移、变薄或衰退，展示生命场的边界条件。",
        },
    ]
}

fn show_elapsed_before_scene(scene_index: usize) -> f32 {
    show_scenes()
        .iter()
        .take(scene_index)
        .map(|scene| scene.duration_secs)
        .sum()
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
            Self::OrbitalField => "螺旋种子展示旋转梯度和柔性的卷积传递。",
            Self::TwinOrganisms => "两个镜像团块展示同一规则如何分化出不同生命形态。",
            Self::CoralDrift => "分枝种子强调脊线生长、衰减和边界竞争。",
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
            mode: SimMode::Lenia,
            render_style,
            lenia,
            reaction: ReactionDiffusionSim::new(width, width, 2001),
            life: LifeSim::new(160, 160, 3001),
            gpu_lenia,
            prefer_gpu_lenia: gpu_ready,
            running: true,
            judge_mode: true,
            show_mode: ShowModeState::enabled_default(),
            info_tab: MainInfoTab::ShowNarration,
            active_major_case: None,
            tool: InteractionTool::Draw,
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
                "GPU Lenia 已启用。调整一条规则，观察形态、运动和指标如何同步变化。".to_owned()
            } else {
                "当前使用 CPU 参考模式。GPU Lenia 不可用，但作品仍可运行。".to_owned()
            },
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
            SimMode::ReactionDiffusion => self.reaction.reset_preset("labyrinth"),
            SimMode::GameOfLife => self.life.reset_preset("structure_showcase"),
        }
        self.texture = None;
        self.mark_cpu_texture_dirty();
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
        self.apply_show_scene(0);
        self.running = true;
        self.status = "10 分钟评审演示已开始：先快速总览，再进入深度章节。".to_owned();
    }

    fn exit_show_mode(&mut self) {
        self.show_mode.enabled = false;
        self.show_mode.playing = false;
        self.show_mode.finished = false;
        self.running = false;
        self.status = "已退出演示模式；当前画面保留为手动实验起点。".to_owned();
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

        self.mode = scene.mode;
        self.render_style = scene.render_style;
        self.steps_per_frame = scene.step_rate;
        self.judge_mode = true;
        self.show_active_region_overlay = true;
        self.show_kernel_overlay = matches!(scene.mode, SimMode::Lenia);
        self.tool = InteractionTool::Pan;
        self.active_major_case = scene.case_id;
        self.step_count = 0;
        self.tick_accumulator = Duration::ZERO;
        self.gpu_cpu_sync_counter = 0;
        self.clear_comparison_result();

        match scene.setup {
            ShowSetup::Lenia(preset) => {
                self.active_preset = preset;
                self.grid_profile = GridProfile::Reference192;
                let size = self.grid_profile.size();
                if self.lenia.size() != (size, size) {
                    self.lenia.resize(size, size);
                }
                self.lenia.reset_preset(preset.id());
                self.prefer_gpu_lenia = self.gpu_lenia.is_some();
                let (w, h) = self.lenia.size();
                self.inspected_lenia = Some(self.lenia.inspect_point(w / 2, h / 2));
                self.sync_gpu_lenia_from_cpu();
            }
            ShowSetup::Reaction(preset) => {
                self.reaction.reset_preset(preset);
            }
            ShowSetup::Life(preset) => {
                if self.life.size() != (96, 96) {
                    self.life.resize(96, 96);
                }
                self.life.reset_preset(preset);
            }
        }

        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.refresh_lenia_inspection();
        self.reset_metric_history();
        self.show_mode.applied_scene_index = Some(index);
        self.update_show_total_elapsed();
        self.status = format!("演示场景：{}", scene.title_zh);
    }

    fn load_major_case(&mut self, case: MajorCase) {
        self.show_mode.enabled = false;
        self.show_mode.playing = false;
        self.show_mode.finished = false;
        self.mode = case.mode;
        self.render_style = case.render_style;
        self.steps_per_frame = case.step_rate;
        self.judge_mode = true;
        self.show_active_region_overlay = true;
        self.show_kernel_overlay = matches!(case.mode, SimMode::Lenia);
        self.tool = InteractionTool::Pan;
        self.active_major_case = Some(case.id);
        self.step_count = 0;
        self.tick_accumulator = Duration::ZERO;
        self.gpu_cpu_sync_counter = 0;
        self.clear_comparison_result();

        match case.setup {
            ShowSetup::Lenia(preset) => {
                self.active_preset = preset;
                self.grid_profile = GridProfile::Reference192;
                let size = self.grid_profile.size();
                if self.lenia.size() != (size, size) {
                    self.lenia.resize(size, size);
                }
                self.lenia.reset_preset(preset.id());
                self.prefer_gpu_lenia = self.gpu_lenia.is_some();
                let (w, h) = self.lenia.size();
                self.inspected_lenia = Some(self.lenia.inspect_point(w / 2, h / 2));
                self.sync_gpu_lenia_from_cpu();
            }
            ShowSetup::Reaction(preset) => {
                self.reaction.reset_preset(preset);
            }
            ShowSetup::Life(preset) => {
                if self.life.size() != (96, 96) {
                    self.life.resize(96, 96);
                }
                self.life.reset_preset(preset);
            }
        }

        self.texture = None;
        self.mark_cpu_texture_dirty();
        self.reset_metric_history();
        self.running = true;
        self.info_tab = MainInfoTab::ShowNarration;
        self.status = format!(
            "已载入主要情况：{}。这是实时模拟，可暂停、单步或改参数。",
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

    fn active_texture_options(&self) -> TextureOptions {
        if self.mode == SimMode::GameOfLife {
            TextureOptions::NEAREST
        } else {
            TextureOptions::LINEAR
        }
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
            if self.mode == SimMode::Lenia {
                self.reset_lenia_with_history();
            } else {
                self.reset_active();
            }
        }
        if self.mode == SimMode::Lenia {
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

    fn show_mode_json(&self) -> serde_json::Value {
        if !self.show_mode.enabled {
            return serde_json::Value::Null;
        }
        show_mode_json_from_state(&self.show_mode)
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
            "system": case.mode.label(),
            "render_style": case.render_style.label(),
            "step_rate": case.step_rate,
            "explanation_zh": case.explanation_zh,
            "expected_outcome_zh": case.expected_outcome_zh,
        })
    }

    fn attach_show_mode_json(&self, mut parameters: serde_json::Value) -> serde_json::Value {
        if let Some(object) = parameters.as_object_mut() {
            object.insert("major_case".to_owned(), self.major_case_json());
        }
        if self.show_mode.enabled {
            if let Some(object) = parameters.as_object_mut() {
                object.insert("show_mode".to_owned(), self.show_mode_json());
            }
        }
        parameters
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
        let parameters = match self.mode {
            SimMode::Lenia => {
                self.lenia_parameter_json(self.active_region(), self.population_phase_analysis())
            }
            SimMode::ReactionDiffusion => json!({
                "schema_version": export::SCHEMA_VERSION,
                "feed": self.reaction.feed,
                "kill": self.reaction.kill,
                "diffusion_a": self.reaction.diff_a,
                "diffusion_b": self.reaction.diff_b,
                "time_step": self.reaction.dt,
                "formula_ascii": "b_next = b + dt*(Db*laplace(b) + a*b*b - (kill+feed)*b)",
                "diagnostics": {
                    "field_stats": {
                        "min": self.reaction.field_stats().min,
                        "max": self.reaction.field_stats().max,
                        "mean": self.reaction.field_stats().mean,
                        "delta_mean": self.reaction.field_stats().delta_mean,
                        "activity": self.reaction.field_stats().activity,
                    }
                },
                "performance": self.performance_json(),
                "active_region": self.active_region_json(),
                "phase_analysis": self.population_phase_json(),
            }),
            SimMode::GameOfLife => json!({
                "schema_version": export::SCHEMA_VERSION,
                "rule": "B3/S23",
                "formula_ascii": "dead+n=3->alive; alive+n=2 or 3->alive; else->dead",
                "seed_density": self.life.random_density,
                "rle_export": self.life.export_rle(),
                "performance": self.performance_json(),
                "active_region": self.active_region_json(),
                "phase_analysis": self.population_phase_json(),
                "pattern_detection": self.pattern_detection_json(),
            }),
        };
        self.attach_show_mode_json(parameters)
    }

    fn draw_show_mode_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("演示模式 Show Mode");
        if !self.show_mode.enabled {
            if ui.button("开始评审演示").clicked() {
                self.start_show_mode();
            }
            ui.small("约 10 分钟：先 90 秒快速总览，再进入深度章节。");
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
        ui.label("由数学规则生成的计算艺术");
        ui.small("美感来自场、卷积核、扩散和可复现的确定性种子。");
        ui.separator();
        self.draw_show_mode_controls(ui);
        ui.separator();

        let mut selected_mode = self.mode;
        egui::ComboBox::from_label("数学系统")
            .selected_text(selected_mode.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut selected_mode, SimMode::Lenia, SimMode::Lenia.label());
                ui.selectable_value(
                    &mut selected_mode,
                    SimMode::ReactionDiffusion,
                    SimMode::ReactionDiffusion.label(),
                );
                ui.selectable_value(
                    &mut selected_mode,
                    SimMode::GameOfLife,
                    SimMode::GameOfLife.label(),
                );
            });
        if selected_mode != self.mode {
            self.pause_show_for_manual_interaction();
            self.mode = selected_mode;
            self.step_count = 0;
            self.texture = None;
            self.mark_cpu_texture_dirty();
            self.reset_metric_history();
            self.refresh_lenia_inspection();
        }

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
            }
        } else {
            ui.label("GPU 高质量 Lenia：不可用");
        }
        if ui
            .add(egui::Slider::new(&mut self.steps_per_frame, 1..=20).text("演化速度"))
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
                if self.mode == SimMode::Lenia {
                    self.reset_lenia_with_history();
                } else {
                    self.reset_active();
                }
            }
        });

        if self.mode == SimMode::Lenia {
            ui.separator();
            ui.heading("交互实验室");
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
        ui.label(format!("系统：{}", self.mode.label()));
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
        ui.label(format!("观察框架：{}", self.mode_statement()));
        if self.mode == SimMode::Lenia {
            let phase = self.lenia_phase();
            ui.label(format!("阶段：{}", phase.label()));
            ui.small(phase.description());
        }
        let m = self.active_metrics();
        ui.label(format!("活跃像素/细胞：{}", m.active));
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

        ui.heading("参数");
        match self.mode {
            SimMode::Lenia => {
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
                    .add(
                        egui::Slider::new(&mut self.lenia.growth_center, 0.05..=0.95)
                            .text("增长中心"),
                    )
                    .changed();
                lenia_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.lenia.growth_width, 0.005..=0.18)
                            .text("增长宽度"),
                    )
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
            }
            SimMode::ReactionDiffusion => {
                let reaction_changed = ui
                    .add(egui::Slider::new(&mut self.reaction.feed, 0.005..=0.09).text("feed"))
                    .changed()
                    | ui.add(egui::Slider::new(&mut self.reaction.kill, 0.02..=0.09).text("kill"))
                        .changed()
                    | ui.add(
                        egui::Slider::new(&mut self.reaction.diff_a, 0.02..=0.30).text("A 扩散"),
                    )
                    .changed()
                    | ui.add(
                        egui::Slider::new(&mut self.reaction.diff_b, 0.005..=0.20).text("B 扩散"),
                    )
                    .changed()
                    | ui.add(egui::Slider::new(&mut self.reaction.dt, 0.2..=1.5).text("时间步长"))
                        .changed();
                if reaction_changed {
                    self.pause_show_for_manual_interaction();
                    self.step_count = 0;
                    self.mark_cpu_texture_dirty();
                    self.reset_metric_history();
                }
                ui.label("规则：两种虚拟化学物质扩散并反应。feed/kill 控制斑点、波纹和迷宫结构。");
            }
            SimMode::GameOfLife => {
                if ui
                    .add(
                        egui::Slider::new(&mut self.life.random_density, 0.02..=0.55)
                            .text("种子密度"),
                    )
                    .changed()
                {
                    self.pause_show_for_manual_interaction();
                }
                if ui.button("随机确定性种子").clicked() {
                    self.pause_show_for_manual_interaction();
                    self.life.reset_random();
                    self.step_count = 0;
                    self.mark_cpu_texture_dirty();
                    self.reset_metric_history();
                }
                ui.label("规则 B3/S23：3 个邻居时出生，2 或 3 个邻居时存活。");
                ui.separator();
                ui.heading("RLE 图案");
                ui.small("导入/导出只适用于离散生命游戏模式。");
                ui.label("导入 RLE");
                ui.add(
                    egui::TextEdit::multiline(&mut self.life_rle_input)
                        .desired_rows(5)
                        .code_editor(),
                );
                ui.horizontal(|ui| {
                    if ui.button("导入 RLE").clicked() {
                        self.pause_show_for_manual_interaction();
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
        ui.label(format!("活跃细胞/像素：{}", m.active));
        if self.mode == SimMode::Lenia {
            let phase = self.lenia_phase();
            ui.label(format!("阶段：{}", phase.label()));
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
        if self.mode == SimMode::Lenia {
            self.draw_lenia_mathematical_frame(ui);
            ui.separator();
            self.draw_lenia_inspector(ui);
            ui.separator();
            self.draw_kernel_lens(ui);
        } else {
            ui.label(self.mode_formula());
            ui.small(self.mode_significance());
        }

        if self.judge_mode {
            ui.separator();
            ui.heading("评审讲解");
            if self.mode == SimMode::Lenia {
                ui.label("1. 数学原始图显示连续数值场。");
                ui.label("2. 艺术表达图用颜色解释同一份数据。");
                ui.label("3. 检查一个点，查看 K * u 和 G(K * u)。");
                ui.label("4. 改一个参数后比较指标历史。");
                ui.label("5. 从当前状态导出 PNG + JSON 证据。");
            } else {
                ui.label("1. 先用数学原始图展示数据场。");
                ui.label("2. 运行约 100 步，观察指标变化。");
                ui.label("3. 一次只改变一个参数。");
                ui.label("4. 比较新图案并导出证据。");
            }
        }
    }

    fn draw_compact_live_diagnostics(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("实时证据");
        let metrics = self.active_metrics();
        ui.label(format!(
            "质量 {:.3} · 熵 {:.3} · 稳定度 {:.3} · 生命力 {:.3}",
            metrics.mass, metrics.entropy, metrics.stability, metrics.vitality
        ));
        let region = self.active_region();
        if let Some((x, y)) = region.centroid {
            ui.small(format!(
                "活跃区域 {:.1}% · 质心 {:.1},{:.1} · 漂移 {:+.2},{:+.2}",
                region.area_ratio * 100.0,
                x,
                y,
                region.drift.0,
                region.drift.1
            ));
        } else {
            ui.small("活跃区域：暂无可检测结构。");
        }
        if self.mode == SimMode::ReactionDiffusion {
            let stats = self.reaction.field_stats();
            ui.small(format!(
                "B 浓度 min {:.3} / max {:.3} / mean {:.3} · 最近变化 {:.4}",
                stats.min, stats.max, stats.mean, stats.delta_mean
            ));
        }
    }

    fn draw_major_cases_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("主要情况");
        ui.small("这些案例覆盖稳定、周期、漂移、波纹、分裂、湍动和衰退。评委可一键载入，不需要先懂参数。");
        ui.separator();

        for (title, mode) in [
            ("生命游戏 Game of Life", SimMode::GameOfLife),
            ("反应扩散 Reaction-Diffusion", SimMode::ReactionDiffusion),
            ("连续生命场 Lenia", SimMode::Lenia),
        ] {
            ui.collapsing(title, |ui| {
                for case in major_cases().into_iter().filter(|case| case.mode == mode) {
                    self.draw_major_case_card(ui, case);
                }
            });
        }
    }

    fn draw_major_case_card(&mut self, ui: &mut egui::Ui, case: MajorCase) {
        let active = self.active_major_case == Some(case.id);
        egui::Frame::group(ui.style())
            .fill(if active {
                Color32::from_rgb(24, 44, 42)
            } else {
                Color32::from_rgb(14, 20, 23)
            })
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    draw_case_swatch(ui, case);
                    ui.vertical(|ui| {
                        ui.strong(case.title_zh);
                        ui.colored_label(Color32::from_rgb(255, 219, 128), case.behavior_label_zh);
                    });
                });
                ui.small(case.explanation_zh);
                ui.small(case.expected_outcome_zh);
                if ui.button("载入并演示").clicked() {
                    self.load_major_case(case);
                }
            });
        ui.add_space(6.0);
    }

    fn draw_central_explanation_bar(&self, ui: &mut egui::Ui) {
        if self.show_mode.enabled {
            let scene = self.current_show_scene();
            let total_progress =
                (self.show_mode.total_elapsed / show_total_duration_secs()).clamp(0.0, 1.0);
            egui::Frame::group(ui.style())
                .fill(Color32::from_rgb(13, 20, 23))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(Color32::from_rgb(100, 232, 218), scene.chapter);
                        ui.strong(scene.title_zh);
                        if self.show_mode.finished {
                            ui.colored_label(Color32::from_rgb(255, 219, 128), "总结页");
                        }
                    });
                    ui.label(scene.narration.core_question_zh);
                    ui.small(scene.narration.conclusion_zh);
                    ui.add(
                        egui::ProgressBar::new(total_progress)
                            .text(format!("演示总进度 {:.0}%", total_progress * 100.0)),
                    );
                });
            ui.add_space(8.0);
        } else if let Some(case_id) = self.active_major_case {
            if let Some(case) = major_cases().into_iter().find(|case| case.id == case_id) {
                egui::Frame::group(ui.style())
                    .fill(Color32::from_rgb(13, 20, 23))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.colored_label(
                                Color32::from_rgb(255, 219, 128),
                                case.behavior_label_zh,
                            );
                            ui.strong(case.title_zh);
                        });
                        ui.small(case.expected_outcome_zh);
                    });
                ui.add_space(8.0);
            }
        }
    }

    fn draw_interpretability_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("可解释分析");
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
        ui.label(format!("活跃面积比例 {:.3}", region.area_ratio));
        let phase = self.population_phase_analysis();
        ui.small(format!(
            "阶段 {} · 质量 {:+.3} · 熵 {:+.3} · 生命力 {:+.3}",
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
        ui.heading("结构识别");
        if report.detections.is_empty() {
            ui.small("暂未检测到已知静物、振荡器或滑翔机。");
        } else {
            for detection in &report.detections {
                ui.label(format!(
                    "{}（{}）位置 {}, {}",
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
        egui::ComboBox::from_label("变量参数")
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
            "{} = {:.4} · {} 步",
            result.parameter.label(),
            result.value,
            result.steps
        ));
        ui.label(format!(
            "质量变化 {:+.3} · 熵变化 {:+.3}",
            result.variant_metrics.mass - result.baseline_metrics.mass,
            result.variant_metrics.entropy - result.baseline_metrics.entropy
        ));
        ui.label(format!(
            "稳定度变化 {:+.3} · 生命力变化 {:+.3}",
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
            .unwrap_or_else(|| "不可用".to_owned());
        ui.small(format!(
            "{} · 源场 {}x{} · GPU {}",
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

    fn draw_lenia_mathematical_frame(&self, ui: &mut egui::Ui) {
        ui.monospace("u[t]       当前标量场");
        ui.monospace("K * u      加权邻域");
        ui.monospace("G(K * u)   钟形增长响应");
        ui.monospace("damping    对已有质量的衰减");
        ui.monospace("u[t+1]     clamp(u[t] + dt * G - damping * u[t])");
        ui.small(self.mode_significance());
    }

    fn draw_lenia_inspector(&self, ui: &mut egui::Ui) {
        ui.heading("场检查器");
        let Some(inspection) = self.inspected_lenia else {
            ui.small("悬停在画布上，查看局部 Lenia 数学量。");
            return;
        };
        ui.label(format!("坐标：{}, {}", inspection.x, inspection.y));
        ui.label(format!(
            "u[t] {:.4} · 前一帧 {:.4}",
            inspection.value, inspection.previous
        ));
        ui.label(format!(
            "变化 {:+.4} · 梯度 {:.4}",
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
        ui.heading("指标历史");
        if self.metric_history.len() < 2 {
            ui.small("运行场以后会形成指标轨迹。");
            return;
        }
        self.metric_history_chart(
            ui,
            "质量/活跃度",
            Color32::from_rgb(103, 222, 209),
            |s| s.mass,
        );
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
            SimMode::Lenia => "连续场生命",
            SimMode::ReactionDiffusion => "化学图案形成",
            SimMode::GameOfLife => "离散局部规则",
        }
    }

    fn mode_formula(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "u[t+1] = clamp(u[t] + dt * G(K * u[t]) - damping * u[t])",
            SimMode::ReactionDiffusion => "A,B 局部扩散，同时发生 A + 2B -> 3B 反应。",
            SimMode::GameOfLife => "B3/S23：3 个邻居出生，2 或 3 个邻居存活。",
        }
    }

    fn mode_significance(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "柔性的邻域卷积核把微小数值变化转化成类似生命体的运动。",
            SimMode::ReactionDiffusion => "扩散速度和反应速率的竞争会显现斑点、膜、波纹和迷宫。",
            SimMode::GameOfLife => "一个简单网格规则展示离散细胞如何通向连续场的想象。",
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
                ui.label("数学生命场");
                ui.separator();
                ui.label(self.backend_label());
                ui.separator();
                ui.label(format!("{}x{}", grid_w, grid_h));
                ui.separator();
                ui.label(format!("种子 {}", self.active_seed()));
                ui.separator();
                ui.label(format!("步数 {}", self.step_count));
                if self.show_mode.enabled {
                    ui.separator();
                    ui.label(format!("演示 {}", self.current_show_scene().title_zh));
                }
                if self.mode == SimMode::Lenia {
                    ui.separator();
                    ui.label(self.lenia_phase().label());
                }
            });
        });

        egui::SidePanel::left("left_controls")
            .resizable(false)
            .default_width(260.0)
            .show(ctx, |ui| self.draw_left_panel(ui));

        egui::SidePanel::right("right_parameters")
            .resizable(false)
            .default_width(330.0)
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
                            ui.painter()
                                .add(egui::Shape::Callback(gpu.paint_callback(rect)));
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
                        let texture_options = self.active_texture_options();
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
                    self.mode.label(),
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
    let desired = egui::vec2(48.0, 38.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(5, 8, 10));
    let accent = match case.mode {
        SimMode::GameOfLife => Color32::from_rgb(216, 240, 139),
        SimMode::ReactionDiffusion => Color32::from_rgb(255, 157, 102),
        SimMode::Lenia => Color32::from_rgb(100, 232, 218),
    };
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, accent),
        egui::StrokeKind::Inside,
    );
    match case.mode {
        SimMode::GameOfLife => {
            for y in 0..4 {
                for x in 0..5 {
                    if (x + y + case.id.id().len()).is_multiple_of(3) {
                        let r = egui::Rect::from_min_size(
                            egui::pos2(
                                rect.left() + 6.0 + x as f32 * 8.0,
                                rect.top() + 5.0 + y as f32 * 7.0,
                            ),
                            egui::vec2(5.0, 5.0),
                        );
                        painter.rect_filled(r, 1.0, accent);
                    }
                }
            }
        }
        SimMode::ReactionDiffusion => {
            for i in 0..5 {
                let x = rect.left() + 7.0 + i as f32 * 8.0;
                painter.circle_filled(
                    egui::pos2(x, rect.center().y + ((i % 2) as f32 - 0.5) * 10.0),
                    4.0 + (i % 3) as f32,
                    accent,
                );
            }
        }
        SimMode::Lenia => {
            painter.circle_stroke(rect.center(), 13.0, egui::Stroke::new(2.0, accent));
            painter.circle_filled(
                egui::pos2(rect.center().x + 7.0, rect.center().y - 4.0),
                4.0,
                Color32::from_rgb(255, 118, 168),
            );
        }
    }
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

fn duration_ms(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn show_scenes_have_complete_ten_minute_script() {
        let scenes = show_scenes();
        let total: f32 = scenes.iter().map(|scene| scene.duration_secs).sum();
        assert!(
            (590.0..=610.0).contains(&total),
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
            assert!((1..=20).contains(&scene.step_rate));
        }
        assert!(scenes.last().is_some_and(|scene| scene.hold_on_finish));
    }

    #[test]
    fn major_cases_have_complete_metadata_and_unique_ids() {
        let mut ids = HashSet::new();
        for case in major_cases() {
            assert!(ids.insert(case.id.id()));
            assert!(!case.title_zh.is_empty());
            assert!(!case.behavior_label_zh.is_empty());
            assert!(!case.explanation_zh.is_empty());
            assert!(!case.expected_outcome_zh.is_empty());
            assert!((1..=20).contains(&case.step_rate));
        }
    }

    #[test]
    fn show_mode_export_json_describes_current_scene() {
        let mut state = ShowModeState::enabled_default();
        state.scene_index = 9;
        state.scene_elapsed = 5.0;
        state.total_elapsed = show_elapsed_before_scene(9) + 5.0;
        let value = show_mode_json_from_state(&state);

        assert_eq!(value["scene_id"], ShowSceneId::ReactionLabyrinth.id());
        assert_eq!(value["scene_title_zh"], "反应扩散：迷宫边界");
        assert!(value["formula_ascii"].as_str().unwrap().is_ascii());
        assert!(value["variables_zh"].as_str().unwrap().contains("B"));
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
