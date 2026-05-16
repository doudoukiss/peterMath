# peterMath

`peterMath` is a native Rust mathematical life-system artwork for competition submission. It is built around one primary identity: Lenia as a living mathematical field.

This folder is the correct Git repository root.

## What Judges See

- GPU Lenia opens first as the main artwork, with CPU fallback if GPU setup fails.
- Raw Math View and Artistic View show the same data field in two visual languages.
- The inspector explains field value, delta, gradient, kernel convolution, growth response, and estimated next value.
- Metrics track mass/activity, entropy, symmetry, stability, and vitality.
- Judge Mode gives a concise path for comparing math, visual expression, parameters, and evidence.
- Exports produce PNG snapshots, parameter JSON, share-state JSON, and evidence packs.
- `web_html/index.html` is an offline browser fallback if the executable cannot run.

The final judge-facing artifact is a Windows folder containing `peterMath.exe`, explanation/evidence files, generated preview images, and the static `web_html/` fallback. Judges should not need Rust, Node, npm, Python, Visual Studio, or a local server.

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

builds the release executable, runs checks, generates preview images, and uploads a `peterMath-windows-x64` artifact containing the executable, docs, assets, previews, templates, and web fallback.

## Product Constraints

- Name remains `peterMath`.
- Native executable is primary; web fallback is backup only.
- Visual beauty must come from mathematical state fields, not decorative effects.
- Preserve deterministic seeds, parameters, metrics, exports, and Judge Mode.
- Keep exported JSON versioned with `schema_version: 1`.

See `docs/PROJECT_PLAN.md` for the current competition plan.
