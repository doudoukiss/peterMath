pub mod lenia;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    RawMath,
    Artistic,
    LifeHighlight,
}

impl RenderStyle {
    pub fn label(self) -> &'static str {
        match self {
            RenderStyle::RawMath => "数学原始图",
            RenderStyle::Artistic => "艺术表达图",
            RenderStyle::LifeHighlight => "生命高光图",
        }
    }
}

pub fn wrap_index(x: isize, y: isize, w: usize, h: usize) -> usize {
    let xx = x.rem_euclid(w as isize) as usize;
    let yy = y.rem_euclid(h as isize) as usize;
    yy * w + xx
}
