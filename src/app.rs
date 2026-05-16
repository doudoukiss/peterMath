use crate::export;
use crate::gpu::{self, GpuLeniaArt, GpuLeniaParams};
use crate::metrics::Metrics;
use crate::simulation::lenia::LeniaSim;
use crate::simulation::life::LifeSim;
use crate::simulation::reaction_diffusion::ReactionDiffusionSim;
use crate::simulation::{RenderStyle, SimMode};
use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use serde_json::json;
use std::time::{Duration, Instant};

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
    brush_mode: BrushMode,
    brush_radius: f32,
    brush_strength: f32,
    steps_per_frame: usize,
    step_count: u64,
    pixels: Vec<u8>,
    texture: Option<TextureHandle>,
    status: String,
    last_tick: Instant,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BrushMode {
    Draw,
    Erase,
}

impl PeterMathApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let width = 192;
        let render_style = RenderStyle::Artistic;
        let lenia = LeniaSim::new(width, width, 1001);
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
        Self {
            mode: SimMode::Lenia,
            render_style,
            lenia,
            reaction: ReactionDiffusionSim::new(width, width, 2001),
            life: LifeSim::new(160, 160, 3001),
            gpu_lenia,
            prefer_gpu_lenia: gpu_ready,
            running: true,
            judge_mode: false,
            brush_mode: BrushMode::Draw,
            brush_radius: 9.0,
            brush_strength: 0.42,
            steps_per_frame: 1,
            step_count: 0,
            pixels: vec![0; width * width * 4],
            texture: None,
            status: if gpu_ready {
                "GPU Lenia is active. Tune one rule and watch form, motion, and metrics agree."
                    .to_owned()
            } else {
                "CPU reference mode. GPU Lenia was unavailable, but the artwork remains runnable."
                    .to_owned()
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

    fn reset_active(&mut self) {
        self.step_count = 0;
        match self.mode {
            SimMode::Lenia => {
                self.lenia.reset_preset("orbital_field");
                self.sync_gpu_lenia_from_cpu();
            }
            SimMode::ReactionDiffusion => self.reaction.reset_preset("mitosis"),
            SimMode::GameOfLife => self.life.reset_preset("symmetric_seed"),
        }
    }

    fn step_active(&mut self) {
        match self.mode {
            SimMode::Lenia => self.lenia.step(),
            SimMode::ReactionDiffusion => self.reaction.step(),
            SimMode::GameOfLife => self.life.step(),
        }
        self.step_count += 1;
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
            Ok(()) => format!("Exported {} and {}", png_path, json_path),
            Err(err) => format!("Export failed: {err}"),
        };
    }

    fn export_gpu_lenia_snapshot(&mut self) {
        let Some(gpu) = &self.gpu_lenia else {
            self.status = "GPU export failed: GPU Lenia is unavailable.".to_owned();
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
            let (size, field, previous) = gpu.read_fields_blocking()?;
            let mut pixels = vec![0; size * size * 4];
            gpu::colorize_fields(&field, &previous, size, self.render_style, &mut pixels);
            let metrics = Metrics::from_scalar_grid(&field, Some(&previous), size, size);
            export::save_png(&png_path, size, size, &pixels)?;
            export::save_json(
                &json_path,
                export::SnapshotExport {
                    mode: self.mode.label(),
                    render_style: self.render_style.label(),
                    backend: self.backend_label(),
                    seed: self.active_seed(),
                    step_count: self.step_count,
                    grid_width: size,
                    grid_height: size,
                    parameters: self.parameter_json(),
                    metrics,
                },
            )?;
            Ok(())
        })();
        self.status = match result {
            Ok(()) => format!("Exported {} and {}", png_path, json_path),
            Err(err) => format!("GPU export failed: {err}"),
        };
    }

    fn gpu_lenia_active(&self) -> bool {
        self.mode == SimMode::Lenia && self.prefer_gpu_lenia && self.gpu_lenia.is_some()
    }

    fn backend_label(&self) -> &'static str {
        if self.gpu_lenia_active() {
            "GPU Lenia"
        } else {
            "CPU Reference"
        }
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

    fn clear_lenia_field(&mut self) {
        self.lenia.clear();
        self.step_count = 0;
        self.sync_gpu_lenia_from_cpu();
        self.status = "Cleared the Lenia field; draw or choose New seed to continue.".to_owned();
    }

    fn new_lenia_seed(&mut self) {
        let next_seed = self
            .lenia
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.lenia.reseed(next_seed);
        self.step_count = 0;
        self.sync_gpu_lenia_from_cpu();
        self.status = format!("Loaded deterministic Lenia seed {next_seed}.");
    }

    fn apply_lenia_brush(&mut self, rect: egui::Rect, response: &egui::Response) {
        if self.mode != SimMode::Lenia {
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
        if !rect.contains(pos) {
            return;
        }

        let (w, h) = self.lenia.size();
        let x = ((pos.x - rect.min.x) / rect.width() * w as f32).clamp(0.0, w as f32 - 1.0);
        let y = ((pos.y - rect.min.y) / rect.height() * h as f32).clamp(0.0, h as f32 - 1.0);
        match self.brush_mode {
            BrushMode::Draw => {
                self.lenia
                    .paint_brush(x, y, self.brush_radius, self.brush_strength);
            }
            BrushMode::Erase => {
                self.lenia
                    .erase_brush(x, y, self.brush_radius, self.brush_strength);
            }
        }
        self.sync_gpu_lenia_from_cpu();
    }

    fn parameter_json(&self) -> serde_json::Value {
        match self.mode {
            SimMode::Lenia => json!({
                "kernel_radius": self.lenia.radius,
                "growth_center": self.lenia.growth_center,
                "growth_width": self.lenia.growth_width,
                "time_step": self.lenia.dt,
                "damping": self.lenia.decay,
                "backend": self.backend_label(),
            }),
            SimMode::ReactionDiffusion => json!({
                "feed": self.reaction.feed,
                "kill": self.reaction.kill,
                "diffusion_a": self.reaction.diff_a,
                "diffusion_b": self.reaction.diff_b,
                "time_step": self.reaction.dt,
            }),
            SimMode::GameOfLife => json!({
                "rule": "B3/S23",
                "seed_density": self.life.random_density,
            }),
        }
    }

    fn draw_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("peterMath");
        ui.label("Computational artwork from mathematical rules");
        ui.small("Beauty is generated by fields, kernels, diffusion, and deterministic seeds.");
        ui.separator();

        egui::ComboBox::from_label("System")
            .selected_text(self.mode.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.mode, SimMode::Lenia, "Lenia-like field");
                ui.selectable_value(
                    &mut self.mode,
                    SimMode::ReactionDiffusion,
                    "Reaction-Diffusion",
                );
                ui.selectable_value(&mut self.mode, SimMode::GameOfLife, "Game of Life");
            });

        egui::ComboBox::from_label("View")
            .selected_text(self.render_style.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.render_style,
                    RenderStyle::RawMath,
                    "Raw Math View",
                );
                ui.selectable_value(
                    &mut self.render_style,
                    RenderStyle::Artistic,
                    "Artistic View",
                );
            });

        ui.checkbox(&mut self.judge_mode, "Judge Mode");
        if self.gpu_lenia.is_some() {
            ui.checkbox(&mut self.prefer_gpu_lenia, "GPU high-quality Lenia");
        } else {
            ui.label("GPU high-quality Lenia: unavailable");
        }
        ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=20).text("evolution rate"));

        ui.horizontal(|ui| {
            if ui
                .button(if self.running { "Pause" } else { "Run" })
                .clicked()
            {
                self.running = !self.running;
            }
            if ui.button("Step").clicked() {
                if self.gpu_lenia_active() {
                    if let Some(gpu) = &self.gpu_lenia {
                        gpu.update_params(lenia_params(&self.lenia), self.render_style);
                        gpu.queue_steps(1);
                    }
                    self.lenia.step();
                    self.step_count += 1;
                } else {
                    self.step_active();
                }
            }
            if ui.button("Reset").clicked() {
                self.reset_active();
            }
        });

        if self.mode == SimMode::Lenia {
            ui.separator();
            ui.heading("Field Brush");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.brush_mode, BrushMode::Draw, "Draw");
                ui.selectable_value(&mut self.brush_mode, BrushMode::Erase, "Erase");
            });
            ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=32.0).text("brush radius"));
            ui.add(egui::Slider::new(&mut self.brush_strength, 0.05..=1.0).text("brush strength"));
            ui.horizontal(|ui| {
                if ui.button("Clear field").clicked() {
                    self.clear_lenia_field();
                }
                if ui.button("New seed").clicked() {
                    self.new_lenia_seed();
                }
            });
        }

        if ui.button("Export snapshot + parameters").clicked() {
            self.export_snapshot();
        }

        ui.separator();
        ui.label(format!("Mode: {}", self.mode.label()));
        ui.label(format!("Backend: {}", self.backend_label()));
        let (grid_w, grid_h) = self.active_size();
        ui.label(format!("Grid: {}x{}", grid_w, grid_h));
        ui.label(format!("Seed: {}", self.active_seed()));
        ui.label(format!("Step: {}", self.step_count));
        ui.label(format!("Frame: {}", self.mode_statement()));
        let m = self.active_metrics();
        ui.label(format!("Active pixels: {}", m.active));
        ui.label(format!("Mass {:.3} · Entropy {:.3}", m.mass, m.entropy));
        ui.label(format!(
            "Stability {:.3} · Vitality {:.3}",
            m.stability, m.vitality
        ));
        ui.label(&self.status);
    }

    fn draw_right_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Parameters");
        match self.mode {
            SimMode::Lenia => {
                let mut lenia_changed = false;
                let mut radius = self.lenia.radius as u32;
                if ui
                    .add(egui::Slider::new(&mut radius, 3..=32).text("kernel radius"))
                    .changed()
                {
                    self.lenia.set_radius(radius as usize);
                    lenia_changed = true;
                }
                lenia_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.lenia.growth_center, 0.05..=0.95)
                            .text("growth center"),
                    )
                    .changed();
                lenia_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.lenia.growth_width, 0.005..=0.18)
                            .text("growth width"),
                    )
                    .changed();
                lenia_changed |= ui
                    .add(egui::Slider::new(&mut self.lenia.dt, 0.005..=0.25).text("time step"))
                    .changed();
                lenia_changed |= ui
                    .add(egui::Slider::new(&mut self.lenia.decay, 0.0..=0.04).text("damping"))
                    .changed();
                ui.label("Rule: a continuous field grows according to a weighted neighborhood kernel and a bell-shaped growth curve.");
                if lenia_changed {
                    self.sync_gpu_lenia_from_cpu();
                    self.step_count = 0;
                }
            }
            SimMode::ReactionDiffusion => {
                ui.add(egui::Slider::new(&mut self.reaction.feed, 0.005..=0.09).text("feed"));
                ui.add(egui::Slider::new(&mut self.reaction.kill, 0.02..=0.09).text("kill"));
                ui.add(
                    egui::Slider::new(&mut self.reaction.diff_a, 0.02..=0.30).text("diffusion A"),
                );
                ui.add(
                    egui::Slider::new(&mut self.reaction.diff_b, 0.005..=0.20).text("diffusion B"),
                );
                ui.add(egui::Slider::new(&mut self.reaction.dt, 0.2..=1.5).text("time step"));
                ui.label("Rule: two virtual chemicals diffuse and react. Feed/kill parameters control spots, waves, and labyrinths.");
            }
            SimMode::GameOfLife => {
                ui.add(
                    egui::Slider::new(&mut self.life.random_density, 0.02..=0.55)
                        .text("seed density"),
                );
                if ui.button("Random deterministic seed").clicked() {
                    self.life.reset_random();
                    self.step_count = 0;
                }
                ui.label("Rule B3/S23: birth with 3 neighbors; survival with 2 or 3 neighbors.");
            }
        }

        ui.separator();
        ui.heading("Metrics");
        if self.gpu_lenia_active() {
            ui.small("Live GPU field; metrics use the synchronized CPU reference.");
        }
        let m = self.active_metrics();
        metric_bar(ui, "mass/activity", m.mass);
        metric_bar(ui, "entropy", m.entropy);
        metric_bar(ui, "symmetry", m.symmetry);
        metric_bar(ui, "stability", m.stability);
        metric_bar(ui, "vitality", m.vitality);
        ui.label(format!("active cells/pixels: {}", m.active));

        ui.separator();
        ui.heading("Mathematical Frame");
        ui.label(self.mode_formula());
        ui.small(self.mode_significance());

        if self.judge_mode {
            ui.separator();
            ui.heading("Judge Mode Guide");
            ui.label("1. Start with Raw Math View to show the data field.");
            ui.label("2. Run 100 steps and watch metrics change.");
            ui.label("3. Change one parameter only.");
            ui.label("4. Compare the new pattern and export evidence.");
        }
    }

    fn mode_statement(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "continuous field life",
            SimMode::ReactionDiffusion => "chemical pattern formation",
            SimMode::GameOfLife => "discrete local rule",
        }
    }

    fn mode_formula(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "u[t+1] = clamp(u[t] + dt * G(K * u[t]) - damping * u[t])",
            SimMode::ReactionDiffusion => "A,B diffuse locally while A + 2B -> 3B reacts.",
            SimMode::GameOfLife => "B3/S23: birth at 3 neighbors; survive at 2 or 3.",
        }
    }

    fn mode_significance(&self) -> &'static str {
        match self.mode {
            SimMode::Lenia => "A soft neighborhood kernel turns small numeric changes into organism-like motion.",
            SimMode::ReactionDiffusion => "Competing diffusion and reaction rates reveal spots, membranes, waves, and labyrinths.",
            SimMode::GameOfLife => "A simple grid rule explains the bridge from discrete cells to continuous fields.",
        }
    }
}

impl eframe::App for PeterMathApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.running {
            let elapsed = self.last_tick.elapsed();
            if elapsed >= Duration::from_millis(66) {
                if self.gpu_lenia_active() {
                    if let Some(gpu) = &self.gpu_lenia {
                        gpu.update_params(lenia_params(&self.lenia), self.render_style);
                        gpu.queue_steps(self.steps_per_frame);
                    }
                    if self.step_count.is_multiple_of(4) {
                        self.lenia.step();
                    }
                    self.step_count += self.steps_per_frame as u64;
                } else {
                    for _ in 0..self.steps_per_frame {
                        self.step_active();
                    }
                }
                self.last_tick = Instant::now();
                ctx.request_repaint();
            } else {
                ctx.request_repaint_after(Duration::from_millis(66) - elapsed);
            }
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let (grid_w, grid_h) = self.active_size();
                ui.strong("peterMath");
                ui.separator();
                ui.label("Lenia living field");
                ui.separator();
                ui.label(self.backend_label());
                ui.separator();
                ui.label(format!("{}x{}", grid_w, grid_h));
                ui.separator();
                ui.label(format!("seed {}", self.active_seed()));
                ui.separator();
                ui.label(format!("step {}", self.step_count));
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
            let available = ui.available_size();
            let square = (available.x.min(available.y) - 28.0).max(320.0);
            let size = egui::vec2(square, square);
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                if self.gpu_lenia_active() {
                    egui::Frame::canvas(ui.style()).show(ui, |ui| {
                        let (rect, response) =
                            ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                        if let Some(gpu) = self.gpu_lenia.as_ref() {
                            gpu.update_params(lenia_params(&self.lenia), self.render_style);
                            ui.painter()
                                .add(egui::Shape::Callback(gpu.paint_callback(rect)));
                        }
                        self.apply_lenia_brush(rect, &response);
                    });
                } else {
                    let (w, h) = self.render_active();
                    let image = ColorImage::from_rgba_unmultiplied([w, h], &self.pixels);
                    if let Some(texture) = &mut self.texture {
                        texture.set(image, TextureOptions::LINEAR);
                    } else {
                        self.texture = Some(ctx.load_texture(
                            "peterMath-field",
                            image,
                            TextureOptions::LINEAR,
                        ));
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
                            self.apply_lenia_brush(rect, &response);
                        });
                    }
                }
                ui.add_space(8.0);
                ui.small(format!(
                    "{} | {} | {} | seed {} | step {}",
                    self.mode.label(),
                    self.render_style.label(),
                    self.backend_label(),
                    self.active_seed(),
                    self.step_count
                ));
            });
        });
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

fn metric_bar(ui: &mut egui::Ui, label: &str, value: f32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::ProgressBar::new(value.clamp(0.0, 1.0)).show_percentage());
    });
}
