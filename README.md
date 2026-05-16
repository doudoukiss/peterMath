# peterMath

`peterMath` is a native Rust Lenia computational artwork for competition submission. The project now focuses on one identity: a continuous living field where local convolution, growth response, damping, seed, and interaction create visible mathematical beauty.

This folder is the correct Git repository root.

## What Judges See

- The app opens directly into a Chinese `评审演示模式` for Lenia.
- The central canvas is a real-time continuous field, not a video.
- `数学原始图` and `艺术表达图` show the same Lenia data in two visual languages.
- Main cases load representative Lenia behaviors: 轨道场、双生命体、卷积核环、稀疏汤、密集开花、珊瑚衰退.
- Brush, erase, stamp, random field, seed, grid profile, and sliders prove the field is interactive.
- The inspector shows `u`, previous `u`, delta, gradient, `K*u`, growth response, and estimated next value.
- Metrics track mass, entropy, symmetry, stability, vitality, active region, phase, and drift.
- Rule-variant comparison changes one Lenia parameter from the same baseline and reports metric deltas.
- Exports produce PNG snapshots, parameter JSON, share-state JSON, and evidence packs.
- `web_html/index.html` is an offline Lenia fallback/public page if the executable cannot run.

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
- Visual beauty must come from the Lenia field, not decorative effects.
- Preserve deterministic seeds, parameters, metrics, exports, and Judge Mode.
- Keep exported JSON versioned with `schema_version: 1`.

See `docs/PROJECT_PLAN.md` for the current competition plan.
