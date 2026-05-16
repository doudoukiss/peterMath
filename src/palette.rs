pub fn raw_gray(v: f32) -> [u8; 4] {
    let c = to_u8(v);
    [c, c, c, 255]
}

pub fn lenia_field_delta(v: f32, edge: f32, contour_phase: f32, delta: f32) -> [u8; 4] {
    let x = v.clamp(0.0, 1.0);
    let ridge = smooth(0.015, 0.18, edge);
    let contour_distance = ((contour_phase * 19.0).fract() - 0.5).abs();
    let contour = 1.0 - smooth(0.025, 0.17, contour_distance);
    let glow = smooth(0.03, 0.82, x);
    let core = smooth(0.58, 1.00, x);
    let birth = smooth(0.002, 0.060, delta.max(0.0));
    let decay = smooth(0.002, 0.060, (-delta).max(0.0));

    [
        to_u8(0.018 + 0.14 * glow + 0.70 * core + 0.24 * contour + 0.46 * decay),
        to_u8(0.034 + 0.48 * glow + 0.16 * core + 0.42 * contour + 0.16 * ridge + 0.30 * birth),
        to_u8(
            0.054
                + 0.48 * glow
                + 0.10 * core
                + 0.22 * contour
                + 0.36 * ridge
                + 0.34 * birth
                + 0.18 * decay,
        ),
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
