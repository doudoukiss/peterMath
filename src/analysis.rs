use crate::metrics::Metrics;

#[derive(Debug, Clone, Copy, Default)]
pub struct ActiveRegionAnalysis {
    pub active_count: usize,
    pub bounds: Option<(usize, usize, usize, usize)>,
    pub centroid: Option<(f32, f32)>,
    pub area_ratio: f32,
    pub drift: (f32, f32),
}

#[derive(Debug, Clone, Copy)]
pub struct PopulationPhaseAnalysis {
    pub label: &'static str,
    pub mass_trend: f32,
    pub entropy_trend: f32,
    pub stability_trend: f32,
    pub vitality_trend: f32,
    pub centroid_drift: (f32, f32),
}

pub fn active_region_from_scalar_grid(
    grid: &[f32],
    w: usize,
    h: usize,
    threshold: f32,
    previous_centroid: Option<(f32, f32)>,
) -> ActiveRegionAnalysis {
    let points = grid
        .iter()
        .enumerate()
        .filter_map(|(i, &value)| (value > threshold).then_some((i % w, i / w)));
    active_region_from_points(w, h, points, previous_centroid)
}

pub fn active_region_from_points(
    w: usize,
    h: usize,
    points: impl Iterator<Item = (usize, usize)>,
    previous_centroid: Option<(f32, f32)>,
) -> ActiveRegionAnalysis {
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut active_count = 0usize;

    for (x, y) in points {
        if x >= w || y >= h {
            continue;
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        sum_x += x as f32;
        sum_y += y as f32;
        active_count += 1;
    }

    if active_count == 0 {
        return ActiveRegionAnalysis::default();
    }

    let centroid = (sum_x / active_count as f32, sum_y / active_count as f32);
    let bounds = (min_x, min_y, max_x, max_y);
    let area = (max_x - min_x + 1) * (max_y - min_y + 1);
    let area_ratio = area as f32 / (w * h).max(1) as f32;
    let drift = previous_centroid
        .map(|previous| (centroid.0 - previous.0, centroid.1 - previous.1))
        .unwrap_or_default();

    ActiveRegionAnalysis {
        active_count,
        bounds: Some(bounds),
        centroid: Some(centroid),
        area_ratio,
        drift,
    }
}

pub fn population_phase_analysis(
    label: &'static str,
    current: Metrics,
    previous: Option<Metrics>,
    active_region: ActiveRegionAnalysis,
) -> PopulationPhaseAnalysis {
    let previous = previous.unwrap_or(current);
    PopulationPhaseAnalysis {
        label,
        mass_trend: current.mass - previous.mass,
        entropy_trend: current.entropy - previous.entropy,
        stability_trend: current.stability - previous.stability,
        vitality_trend: current.vitality - previous.vitality,
        centroid_drift: active_region.drift,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_region_reports_bounds_and_centroid() {
        let points = [(2, 3), (4, 3), (4, 5)].into_iter();
        let region = active_region_from_points(10, 10, points, Some((2.0, 2.0)));
        assert_eq!(region.active_count, 3);
        assert_eq!(region.bounds, Some((2, 3, 4, 5)));
        let centroid = region.centroid.expect("centroid");
        assert!((centroid.0 - 3.333).abs() < 0.01);
        assert!((centroid.1 - 3.666).abs() < 0.01);
        assert!((region.drift.0 - 1.333).abs() < 0.01);
    }
}
