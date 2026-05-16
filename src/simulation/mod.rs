pub mod lenia;
pub mod life;
pub mod reaction_diffusion;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimMode {
    Lenia,
    ReactionDiffusion,
    GameOfLife,
}

impl SimMode {
    pub fn label(self) -> &'static str {
        match self {
            SimMode::Lenia => "Lenia-like field",
            SimMode::ReactionDiffusion => "Reaction-Diffusion",
            SimMode::GameOfLife => "Game of Life",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            SimMode::Lenia => "lenia",
            SimMode::ReactionDiffusion => "reaction_diffusion",
            SimMode::GameOfLife => "game_of_life",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            SimMode::Lenia => "连续生命场 Lenia",
            SimMode::ReactionDiffusion => "反应扩散 Reaction-Diffusion",
            SimMode::GameOfLife => "生命游戏 Game of Life",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    RawMath,
    Artistic,
}

impl RenderStyle {
    pub fn label(self) -> &'static str {
        match self {
            RenderStyle::RawMath => "Raw Math View",
            RenderStyle::Artistic => "Artistic View",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            RenderStyle::RawMath => "raw_math",
            RenderStyle::Artistic => "artistic",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            RenderStyle::RawMath => "数学原始图",
            RenderStyle::Artistic => "艺术表达图",
        }
    }

    pub fn explanation_zh(self) -> &'static str {
        match self {
            RenderStyle::RawMath => "直接显示数值场或细胞状态，方便看规则本身。",
            RenderStyle::Artistic => "用同一份数据生成颜色、轮廓和亮度，方便看美感。",
        }
    }
}

pub fn wrap_index(x: isize, y: isize, w: usize, h: usize) -> usize {
    let xx = x.rem_euclid(w as isize) as usize;
    let yy = y.rem_euclid(h as isize) as usize;
    yy * w + xx
}
