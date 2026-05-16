pub fn raw_gray(v: f32) -> [u8; 4] {
    let c = to_u8(v);
    [c, c, c, 255]
}

pub fn scientific(v: f32) -> [u8; 4] {
    let x = v.clamp(0.0, 1.0);
    let r = smooth(0.35, 1.00, x);
    let g = smooth(0.10, 0.85, x) * (1.0 - smooth(0.88, 1.00, x) * 0.35);
    let b = 1.0 - smooth(0.15, 0.75, x) * 0.85;
    [to_u8(r), to_u8(g), to_u8(b), 255]
}

pub fn life_field(v: f32, edge: f32, contour_phase: f32) -> [u8; 4] {
    let x = v.clamp(0.0, 1.0);
    let ridge = smooth(0.015, 0.18, edge);
    let contour_distance = ((contour_phase * 19.0).fract() - 0.5).abs();
    let contour = 1.0 - smooth(0.025, 0.17, contour_distance);
    let glow = smooth(0.03, 0.82, x);
    let core = smooth(0.58, 1.00, x);

    [
        to_u8(0.020 + 0.18 * glow + 0.72 * core + 0.22 * contour),
        to_u8(0.040 + 0.45 * glow + 0.18 * core + 0.38 * contour + 0.12 * ridge),
        to_u8(0.060 + 0.42 * glow + 0.10 * core + 0.18 * contour + 0.34 * ridge),
        255,
    ]
}

pub fn reaction_field(v: f32, edge: f32) -> [u8; 4] {
    let x = v.clamp(0.0, 1.0);
    let line = ((x * 21.0).fract() - 0.5).abs();
    let contour = 1.0 - smooth(0.02, 0.13, line);
    let ridge = smooth(0.025, 0.22, edge);
    let bloom = smooth(0.08, 0.80, x);

    [
        to_u8(0.035 + 0.42 * bloom + 0.22 * contour),
        to_u8(0.040 + 0.35 * bloom + 0.55 * contour + 0.16 * ridge),
        to_u8(0.055 + 0.20 * bloom + 0.70 * ridge),
        255,
    ]
}

fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
