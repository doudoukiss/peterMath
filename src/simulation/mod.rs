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
            RenderStyle::RawMath => "数学原始图",
            RenderStyle::Artistic => "艺术表达图",
        }
    }
}

pub fn wrap_index(x: isize, y: isize, w: usize, h: usize) -> usize {
    let xx = x.rem_euclid(w as isize) as usize;
    let yy = y.rem_euclid(h as isize) as usize;
    yy * w + xx
}
