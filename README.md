# peterMath

`peterMath` is a native Rust **Lenia 数学生命教学游戏** for competition submission. Judges should be able to open the app, start a short mission, touch the field, see feedback, reveal the math card, and export evidence without reading external instructions first.

This folder is the correct Git repository root.

## What Judges See

- The app opens into `任务模式`, a playable Lenia mission shell.
- Five missions teach the core ideas: wake the field, shape life, tune a rule, prove Raw/Artistic views share one field, and export an evidence report.
- The central canvas is a real-time continuous field, not a video.
- `数学原始图` and `艺术表达图` show the same Lenia data in two visual languages.
- Brush, stamp, seed, presets, and sliders become teaching actions with immediate mission feedback.
- The math card unlocks after play: `u`, `K*u`, growth response, damping, metrics, and inspector values explain what happened.
- Expert/evidence panels still provide metrics, active region, rule-variant comparison, GPU/CPU diagnostics, PNG/JSON snapshots, share-state JSON, and evidence packs.
- `web_html/index.html` is an offline teaching-game fallback if the executable cannot run.

The final judge-facing artifact is a Windows folder containing `peterMath.exe`, generated Lenia preview images, explanation templates, and the static `web_html/` fallback. Judges should not need Rust, Node, npm, Python, Visual Studio, internet access, or a local server.

## Run on macOS

```bash
cargo run
```

Useful commands:

```bash
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo build --release
cargo run --bin render_preview
cargo run --bin perf_probe
```

Preview images are written to:

```text
peterMath_exports/previews/
```

Local app exports are written as root-level snapshot/parameter files or into:

```text
peterMath_exports/evidence_seed<seed>_step<step>/
```

## Build Windows Artifact

Push this folder as the root of the GitHub repository. The workflow:

```text
.github/workflows/windows-release.yml
```

builds the release executable, runs checks, generates Lenia preview images, and uploads a `peterMath-windows-x64` artifact containing the executable, docs, assets, previews, templates, and web fallback.

## Product Constraints

- Name remains `peterMath`.
- Native executable is primary; web fallback is backup only.
- The first experience must be playable missions, not a research-style control panel.
- Visual beauty must come from the Lenia field, not decorative effects or pre-rendered animation.
- Preserve deterministic seeds, parameters, metrics, exports, and secondary automatic explanation mode.
- Keep exported JSON versioned with `schema_version: 1`.

See `docs/TEACHING_GAME_SPEC.md` and `docs/PROJECT_PLAN.md` for the current competition plan.
