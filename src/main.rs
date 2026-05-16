#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod export;
mod gpu;
mod metrics;
mod palette;
mod simulation;

use app::PeterMathApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("peterMath")
            .with_inner_size([1360.0, 820.0])
            .with_min_inner_size([1000.0, 680.0]),
        ..Default::default()
    };

    eframe::run_native(
        "peterMath",
        options,
        Box::new(|cc| Ok(Box::new(PeterMathApp::new(cc)))),
    )
}
