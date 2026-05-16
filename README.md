# peterMath

`peterMath` is a native Rust mathematical life-system laboratory for competition submission. It is designed for macOS development and Windows judge delivery.

This folder is the correct Git repository root.

## Product goal

The app should demonstrate how simple mathematical rules create complex life-like structures through:

- deterministic seeds
- adjustable parameters
- Raw Math View and Artistic View
- rule/formula explanations
- metrics and experiment export
- Judge Mode for a three-minute evaluation session

The final judge-facing artifact is a Windows folder containing `peterMath.exe`, explanation/evidence files, and a static `web_html/` fallback. Judges should not need Rust, Node, npm, Python, Visual Studio, or a local server.

## Current status

- CPU reference modes: Lenia-like field, Reaction-Diffusion, and Game of Life.
- UI: side controls, parameter panel, metrics, Raw/Artistic view toggle, early Judge Mode.
- Export: PNG snapshot plus parameter JSON.
- Web fallback: `web_html/index.html`, a standalone reaction-diffusion browser demo for emergency presentation if the executable cannot run on a judge machine.
- Planned next: stronger Lenia presets, experiment CSV/JSON history, guided Judge Mode, Physarum mode, and wgpu/WGSL simulation path.

See `docs/PROJECT_PLAN.md` for the competition execution plan.

## Run on macOS

```bash
rustup update
cargo fmt
cargo test
cargo run --release
```

For stricter checks:

```bash
cargo clippy --all-targets -- -D warnings
```

To generate still previews for visual review without relying on native window capture:

```bash
cargo run --bin render_preview
```

The images are written to `peterMath_exports/previews/`.

## Build Windows artifact from Mac

Initialize and push this folder as the root of a GitHub repository named `peterMath`:

```bash
git init
git branch -M main
git add .
git commit -m "Initialize peterMath native Rust app"
git remote add origin <your-github-repo-url>
git push -u origin main
```

Then run the GitHub Actions workflow:

```text
.github/workflows/windows-release.yml
```

The workflow should produce a downloadable `peterMath-windows-x64` artifact.

## Product constraints

- Name must remain `peterMath`.
- Do not add large decorative particles.
- All visuals should come from mathematical state fields.
- Preserve Raw Math View, Artistic View, and Judge Mode.
- Preserve deterministic seeds and exportable parameters.
- Keep the native executable as the primary product; keep `web_html/` as a fallback demonstration only.

## Development note

The scaffold is intentionally small. It is easier for Codex to upgrade a clean CPU reference model into a GPU model than to debug an over-complex shader pipeline from the first commit.

Use the prompts in `../docs/CODEX_PROMPTS_EN.md` while this folder is still inside the upgrade pack. After publishing this folder as its own repository, keep the repo-local `docs/PROJECT_PLAN.md` as the main plan and copy only the few docs you still need.
