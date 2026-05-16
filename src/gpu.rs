use crate::palette;
use crate::simulation::RenderStyle;
use eframe::{egui_wgpu, wgpu};
use std::borrow::Cow;
use std::sync::{mpsc, Arc, Mutex};

const GPU_SIZE: u32 = 512;
const MAX_KERNEL: usize = 4_225;
const PARAMS_SIZE: u64 = 32;
const WORKGROUP_SIZE: u32 = 8;

#[derive(Clone, Copy)]
pub struct GpuLeniaParams {
    pub growth_center: f32,
    pub growth_width: f32,
    pub dt: f32,
    pub decay: f32,
}

pub struct GpuLeniaArt {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shared: Arc<Mutex<GpuLeniaShared>>,
}

struct GpuLeniaShared {
    size: u32,
    field_a: wgpu::Buffer,
    field_b: wgpu::Buffer,
    kernel: wgpu::Buffer,
    params: wgpu::Buffer,
    readback: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    compute_a_to_b: wgpu::BindGroup,
    compute_b_to_a: wgpu::BindGroup,
    render_a: wgpu::BindGroup,
    render_b: wgpu::BindGroup,
    current_is_a: bool,
    kernel_len: u32,
    params_value: GpuLeniaParams,
    render_style: RenderStyle,
    pending_steps: u32,
    pending_field: Option<Vec<f32>>,
    pending_kernel: Option<Vec<KernelEntry>>,
}

#[derive(Clone, Copy)]
struct KernelEntry {
    dx: i32,
    dy: i32,
    weight: f32,
}

impl GpuLeniaArt {
    pub fn new(
        render_state: &egui_wgpu::RenderState,
        source_field: &[f32],
        source_width: usize,
        source_height: usize,
        kernel: &[(isize, isize, f32)],
        params: GpuLeniaParams,
        render_style: RenderStyle,
    ) -> Result<Self, String> {
        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let size = GPU_SIZE;
        let field = upscale_field(source_field, source_width, source_height, size as usize);
        let field_bytes = f32_bytes(&field);
        let field_size = field_bytes.len() as u64;

        let field_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("peterMath GPU Lenia field A"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let field_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("peterMath GPU Lenia field B"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&field_a, 0, &field_bytes);
        queue.write_buffer(&field_b, 0, &field_bytes);

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("peterMath GPU Lenia readback"),
            size: field_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let kernel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("peterMath GPU Lenia kernel"),
            size: (MAX_KERNEL * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kernel_entries = pack_kernel(kernel);
        if kernel_entries.len() > MAX_KERNEL {
            return Err("Lenia kernel is larger than the GPU buffer".to_owned());
        }
        queue.write_buffer(&kernel_buffer, 0, &kernel_bytes(&kernel_entries));

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("peterMath GPU Lenia params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            &params_bytes(size, kernel_entries.len() as u32, params, render_style),
        );

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("peterMath GPU Lenia compute layout"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
                uniform_entry(3, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("peterMath GPU Lenia render layout"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("peterMath Lenia compute WGSL"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LENIA_COMPUTE_WGSL)),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("peterMath Lenia render WGSL"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LENIA_RENDER_WGSL)),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("peterMath GPU Lenia compute pipeline layout"),
                bind_group_layouts: &[&compute_layout],
                push_constant_ranges: &[],
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("peterMath GPU Lenia compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("peterMath GPU Lenia render pipeline layout"),
                bind_group_layouts: &[&render_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("peterMath GPU Lenia render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_state.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        let compute_a_to_b = compute_bind_group(
            &device,
            &compute_layout,
            &field_a,
            &field_b,
            &kernel_buffer,
            &params_buffer,
            "peterMath GPU Lenia compute A to B",
        );
        let compute_b_to_a = compute_bind_group(
            &device,
            &compute_layout,
            &field_b,
            &field_a,
            &kernel_buffer,
            &params_buffer,
            "peterMath GPU Lenia compute B to A",
        );
        let render_a = render_bind_group(
            &device,
            &render_layout,
            &field_a,
            &params_buffer,
            "peterMath GPU Lenia render A",
        );
        let render_b = render_bind_group(
            &device,
            &render_layout,
            &field_b,
            &params_buffer,
            "peterMath GPU Lenia render B",
        );

        Ok(Self {
            device,
            queue,
            shared: Arc::new(Mutex::new(GpuLeniaShared {
                size,
                field_a,
                field_b,
                kernel: kernel_buffer,
                params: params_buffer,
                readback,
                compute_pipeline,
                render_pipeline,
                compute_a_to_b,
                compute_b_to_a,
                render_a,
                render_b,
                current_is_a: true,
                kernel_len: kernel_entries.len() as u32,
                params_value: params,
                render_style,
                pending_steps: 0,
                pending_field: None,
                pending_kernel: None,
            })),
        })
    }

    pub fn size(&self) -> u32 {
        self.shared
            .lock()
            .map(|shared| shared.size)
            .unwrap_or(GPU_SIZE)
    }

    pub fn queue_steps(&self, steps: usize) {
        if let Ok(mut shared) = self.shared.lock() {
            shared.pending_steps = shared.pending_steps.saturating_add(steps as u32);
        }
    }

    pub fn update_params(&self, params: GpuLeniaParams, render_style: RenderStyle) {
        if let Ok(mut shared) = self.shared.lock() {
            shared.params_value = params;
            shared.render_style = render_style;
        }
    }

    pub fn reset_from_cpu(
        &self,
        field: &[f32],
        source_width: usize,
        source_height: usize,
        kernel: &[(isize, isize, f32)],
        params: GpuLeniaParams,
        render_style: RenderStyle,
    ) {
        if let Ok(mut shared) = self.shared.lock() {
            shared.pending_field = Some(upscale_field(
                field,
                source_width,
                source_height,
                shared.size as usize,
            ));
            shared.pending_kernel = Some(pack_kernel(kernel));
            shared.params_value = params;
            shared.render_style = render_style;
            shared.pending_steps = 0;
        }
    }

    pub fn paint_callback(&self, rect: egui::Rect) -> egui::PaintCallback {
        egui_wgpu::Callback::new_paint_callback(
            rect,
            LeniaPaintCallback {
                shared: Arc::clone(&self.shared),
            },
        )
    }

    pub fn read_field_blocking(&self) -> anyhow::Result<(usize, Vec<f32>)> {
        let (size, readback) = {
            let mut shared = self
                .shared
                .lock()
                .map_err(|_| anyhow::anyhow!("GPU Lenia state lock failed"))?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("peterMath GPU Lenia readback encoder"),
                });

            apply_pending_updates(&mut shared, &self.queue);
            dispatch_pending_steps(&mut shared, &mut encoder, u32::MAX);

            let source = if shared.current_is_a {
                shared.field_a.clone()
            } else {
                shared.field_b.clone()
            };
            let size = shared.size as usize;
            let readback = shared.readback.clone();
            encoder.copy_buffer_to_buffer(&source, 0, &readback, 0, (size * size * 4) as u64);
            self.queue.submit(Some(encoder.finish()));

            (size, readback)
        };

        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow::anyhow!("GPU Lenia readback callback failed"))?
            .map_err(|err| anyhow::anyhow!("GPU Lenia readback failed: {err:?}"))?;

        let mapped = slice.get_mapped_range();
        let mut values = Vec::with_capacity(size * size);
        for chunk in mapped.chunks_exact(4) {
            values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        drop(mapped);
        readback.unmap();
        Ok((size, values))
    }
}

struct LeniaPaintCallback {
    shared: Arc<Mutex<GpuLeniaShared>>,
}

impl egui_wgpu::CallbackTrait for LeniaPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Ok(mut shared) = self.shared.lock() else {
            return Vec::new();
        };

        apply_pending_updates(&mut shared, queue);
        dispatch_pending_steps(&mut shared, encoder, 16);

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Ok(shared) = self.shared.lock() else {
            return;
        };
        render_pass.set_pipeline(&shared.render_pipeline);
        if shared.current_is_a {
            render_pass.set_bind_group(0, &shared.render_a, &[]);
        } else {
            render_pass.set_bind_group(0, &shared.render_b, &[]);
        }
        render_pass.draw(0..3, 0..1);
    }
}

fn apply_pending_updates(shared: &mut GpuLeniaShared, queue: &wgpu::Queue) {
    if let Some(field) = shared.pending_field.take() {
        let bytes = f32_bytes(&field);
        queue.write_buffer(&shared.field_a, 0, &bytes);
        queue.write_buffer(&shared.field_b, 0, &bytes);
        shared.current_is_a = true;
    }

    if let Some(kernel) = shared.pending_kernel.take() {
        let kernel_len = kernel.len().min(MAX_KERNEL);
        queue.write_buffer(&shared.kernel, 0, &kernel_bytes(&kernel[..kernel_len]));
        shared.kernel_len = kernel_len as u32;
    }

    queue.write_buffer(
        &shared.params,
        0,
        &params_bytes(
            shared.size,
            shared.kernel_len,
            shared.params_value,
            shared.render_style,
        ),
    );
}

fn dispatch_pending_steps(
    shared: &mut GpuLeniaShared,
    encoder: &mut wgpu::CommandEncoder,
    max_steps: u32,
) {
    let steps = shared.pending_steps.min(max_steps);
    shared.pending_steps -= steps;

    for _ in 0..steps {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("peterMath GPU Lenia compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.compute_pipeline);
            if shared.current_is_a {
                pass.set_bind_group(0, &shared.compute_a_to_b, &[]);
            } else {
                pass.set_bind_group(0, &shared.compute_b_to_a, &[]);
            }
            let groups = shared.size.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, groups, 1);
        }
        shared.current_is_a = !shared.current_is_a;
    }
}

pub fn colorize_field(field: &[f32], size: usize, render_style: RenderStyle, out: &mut [u8]) {
    for y in 0..size {
        for x in 0..size {
            let i = y * size + x;
            let value = field[i].clamp(0.0, 1.0);
            let rgba = match render_style {
                RenderStyle::RawMath => palette::raw_gray(value),
                RenderStyle::Artistic => {
                    let gx = field[y * size + ((x + 1) % size)]
                        - field[y * size + ((x + size - 1) % size)];
                    let gy = field[((y + 1) % size) * size + x]
                        - field[((y + size - 1) % size) * size + x];
                    let edge = (gx * gx + gy * gy).sqrt() * 3.0;
                    palette::life_field((value * 1.30).clamp(0.0, 1.0), edge, value)
                }
            };
            out[i * 4..i * 4 + 4].copy_from_slice(&rgba);
        }
    }
}

fn upscale_field(source: &[f32], source_w: usize, source_h: usize, target_size: usize) -> Vec<f32> {
    let mut out = vec![0.0; target_size * target_size];
    if source.is_empty() || source_w == 0 || source_h == 0 {
        return out;
    }

    for y in 0..target_size {
        for x in 0..target_size {
            let sx = x * source_w / target_size;
            let sy = y * source_h / target_size;
            out[y * target_size + x] = source[sy * source_w + sx];
        }
    }
    out
}

fn pack_kernel(kernel: &[(isize, isize, f32)]) -> Vec<KernelEntry> {
    kernel
        .iter()
        .take(MAX_KERNEL)
        .map(|&(dx, dy, weight)| KernelEntry {
            dx: dx as i32,
            dy: dy as i32,
            weight,
        })
        .collect()
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
    kernel: &wgpu::Buffer,
    params: &wgpu::Buffer,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            buffer_entry(0, source),
            buffer_entry(1, destination),
            buffer_entry(2, kernel),
            buffer_entry(3, params),
        ],
    })
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    field: &wgpu::Buffer,
    params: &wgpu::Buffer,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[buffer_entry(0, field), buffer_entry(1, params)],
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn kernel_bytes(entries: &[KernelEntry]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(entries.len() * 16);
    for entry in entries {
        bytes.extend_from_slice(&entry.dx.to_ne_bytes());
        bytes.extend_from_slice(&entry.dy.to_ne_bytes());
        bytes.extend_from_slice(&entry.weight.to_ne_bytes());
        bytes.extend_from_slice(&0.0f32.to_ne_bytes());
    }
    bytes
}

fn params_bytes(
    size: u32,
    kernel_len: u32,
    params: GpuLeniaParams,
    render_style: RenderStyle,
) -> [u8; PARAMS_SIZE as usize] {
    let mut bytes = [0; PARAMS_SIZE as usize];
    write_u32(&mut bytes, 0, size);
    write_u32(&mut bytes, 4, kernel_len);
    write_u32(
        &mut bytes,
        8,
        match render_style {
            RenderStyle::RawMath => 0,
            RenderStyle::Artistic => 1,
        },
    );
    write_u32(&mut bytes, 12, 0);
    write_f32(&mut bytes, 16, params.growth_center);
    write_f32(&mut bytes, 20, params.growth_width);
    write_f32(&mut bytes, 24, params.dt);
    write_f32(&mut bytes, 28, params.decay);
    bytes
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

const LENIA_COMPUTE_WGSL: &str = r#"
struct KernelEntry {
    offset: vec2<i32>,
    weight: f32,
    pad: f32,
}

struct Params {
    size: u32,
    kernel_len: u32,
    render_style: u32,
    pad0: u32,
    growth_center: f32,
    growth_width: f32,
    dt: f32,
    decay: f32,
}

@group(0) @binding(0) var<storage, read> source_field: array<f32>;
@group(0) @binding(1) var<storage, read_write> dest_field: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<KernelEntry>;
@group(0) @binding(3) var<uniform> params: Params;

fn wrap_coord(value: i32, size: i32) -> i32 {
    return ((value % size) + size) % size;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.size || id.y >= params.size) {
        return;
    }

    let size_i = i32(params.size);
    var neighborhood = 0.0;
    for (var i = 0u; i < params.kernel_len; i = i + 1u) {
        let entry = kernel[i];
        let sx = wrap_coord(i32(id.x) + entry.offset.x, size_i);
        let sy = wrap_coord(i32(id.y) + entry.offset.y, size_i);
        let source_index = u32(sy) * params.size + u32(sx);
        neighborhood = neighborhood + source_field[source_index] * entry.weight;
    }

    let sigma2 = 2.0 * params.growth_width * params.growth_width;
    let growth = 2.0 * exp(-pow(neighborhood - params.growth_center, 2.0) / sigma2) - 1.0;
    let index = id.y * params.size + id.x;
    let value = source_field[index] + params.dt * growth - params.decay * source_field[index];
    dest_field[index] = clamp(value, 0.0, 1.0);
}
"#;

const LENIA_RENDER_WGSL: &str = r#"
struct Params {
    size: u32,
    kernel_len: u32,
    render_style: u32,
    pad0: u32,
    growth_center: f32,
    growth_width: f32,
    dt: f32,
    decay: f32,
}

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = positions[index];
    var out: VertexOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

fn sample_field(x: i32, y: i32) -> f32 {
    let size_i = i32(params.size);
    let sx = u32(((x % size_i) + size_i) % size_i);
    let sy = u32(((y % size_i) + size_i) % size_i);
    return field[sy * params.size + sx];
}

fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn life_palette(value: f32, edge: f32) -> vec3<f32> {
    let x = clamp(value, 0.0, 1.0);
    let ridge = smooth_step(0.015, 0.18, edge);
    let contour_distance = abs(fract(x * 19.0) - 0.5);
    let contour = 1.0 - smooth_step(0.025, 0.17, contour_distance);
    let glow = smooth_step(0.03, 0.82, x);
    let core = smooth_step(0.58, 1.0, x);
    return vec3<f32>(
        0.020 + 0.18 * glow + 0.72 * core + 0.22 * contour,
        0.040 + 0.45 * glow + 0.18 * core + 0.38 * contour + 0.12 * ridge,
        0.060 + 0.42 * glow + 0.10 * core + 0.18 * contour + 0.34 * ridge
    );
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(0.9999, 0.9999));
    let x = i32(uv.x * f32(params.size));
    let y = i32(uv.y * f32(params.size));
    let value = sample_field(x, y);

    if (params.render_style == 0u) {
        return vec4<f32>(vec3<f32>(value), 1.0);
    }

    let gx = sample_field(x + 1, y) - sample_field(x - 1, y);
    let gy = sample_field(x, y + 1) - sample_field(x, y - 1);
    let edge = sqrt(gx * gx + gy * gy) * 3.0;
    let color = clamp(life_palette(value * 1.3, edge), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(color, 1.0);
}
"#;
