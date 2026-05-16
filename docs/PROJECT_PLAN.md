# peterMath Project Plan

## Repository Structure

This folder, `peterMath/`, is the Git repository root for the competition app. The outer `peterMath_pack/` folder is only a planning/archive workspace.

```text
peterMath/
├─ Cargo.toml
├─ README.md
├─ .github/workflows/windows-release.yml
├─ assets/
├─ docs/
├─ judge_submission_template/
├─ screenshots/
├─ web_html/
└─ src/
```

## Competition Objective

Build `peterMath` into a Windows-runnable Lenia artwork and evidence lab:

```text
continuous living field + deterministic seed + adjustable rule + inspector + metrics + exportable evidence
```

The judging story is native-first and Chinese-first. The app should persuade through mathematical beauty: a scalar field, a convolution kernel, a growth function, damping, and human interaction.

## Current Implemented Direction

- Opening experience: a Lenia-only Chinese guided show mode, about three minutes long.
- Main identity: Lenia as a continuous living mathematical field.
- Rendering: GPU Lenia is primary; CPU Lenia remains fallback/reference for metrics, inspector, exports, and manual validation.
- Interaction: draw, erase, stamps, presets, density randomization, undo/redo, keyboard shortcuts.
- Explanation: Chinese labels, preset-first controls, field inspector, kernel lens, phase labels, metric history, and central narration.
- Interpretability: active-region analysis, population phase trends, point inspection, and Lenia rule variant comparison.
- Performance: diagnostics, bounded scheduler, CPU texture dirty tracking, and `perf_probe`.
- Shareability: PNG/JSON snapshot export, share-state JSON, evidence packs, generated Lenia previews, and offline web fallback.

## Final Submission Checklist

1. Run all validation commands from the repository root.
2. Generate previews with `cargo run --bin render_preview`.
3. Confirm the app opens directly into Lenia show mode.
4. Create at least one evidence pack from the native app.
5. Confirm `web_html/index.html` opens directly from disk and hash sharing restores Lenia parameters.
6. Push to GitHub and run the Windows release workflow.
7. Download `peterMath-windows-x64` and confirm it contains:
   - `peterMath.exe`
   - judge templates
   - web fallback
   - assets
   - generated Lenia previews
   - screenshot guidance
8. Prepare a three-minute demonstration using `评审演示模式`.

## Working Rule

Do not add features that weaken the judging story. Every visible feature should support one of these claims:

- The pattern comes from a Lenia mathematical rule.
- The rule has adjustable parameters.
- The same seed and parameters reproduce the same result.
- Raw and Artistic views share the same data.
- Metrics and exports provide evidence.
- Active-region, inspector, and rule-variant tools make emergence legible.
- If the executable fails on a judge machine, the static web fallback still demonstrates Lenia-style rule-driven field formation.
