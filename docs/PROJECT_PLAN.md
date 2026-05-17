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

Build `peterMath` into a Windows-runnable Lenia teaching game:

```text
playable missions + continuous living field + deterministic seed + adjustable rule + math card + exportable evidence
```

The judging story is native-first and Chinese-first. The app should first feel like something judges can play immediately, then reveal the mathematics behind the play: scalar field, convolution kernel, growth function, damping, metrics, and reproducible evidence.

## Current Implemented Direction

- Opening experience: a Chinese `任务模式` with short playable missions.
- Main identity: Lenia as a mathematical life teaching game.
- Rendering: GPU Lenia is primary; CPU Lenia remains fallback/reference for metrics, inspector, exports, and manual validation.
- Interaction: run, draw, erase, stamp, switch view, tune rule, inspect a point, and export evidence as mission actions.
- Explanation: mission objective, feedback pulse, hint, unlocked math card, field inspector, kernel lens, phase labels, and metric history.
- Interpretability: active-region analysis, population phase trends, point inspection, Lenia rule variant comparison, and mission progress.
- Performance: diagnostics, bounded scheduler, CPU texture dirty tracking, and `perf_probe`.
- Shareability: PNG/JSON snapshot export, share-state JSON, evidence packs, generated Lenia previews, and offline web fallback.
- Secondary presentation: the older automatic Lenia show mode remains available as `自动讲解`, but it is no longer the primary product direction.

## Final Submission Checklist

1. Run all validation commands from the repository root.
2. Generate previews with `cargo run --bin render_preview`.
3. Confirm the app opens directly into `任务模式`.
4. Complete the five teaching missions in a fresh run.
5. Create at least one evidence pack from the final mission.
6. Confirm `web_html/index.html` opens directly from disk and presents the same teaching-game positioning.
7. Push to GitHub and run the Windows release workflow.
8. Download `peterMath-windows-x64` and confirm it contains:
   - `peterMath.exe`
   - judge templates
   - web fallback
   - assets
   - generated Lenia previews
   - screenshot guidance
9. Prepare a three-minute demonstration by playing missions first, then optionally using `自动讲解`.

## Working Rule

Do not add features that weaken the judging story. Every visible feature should support one of these claims:

- The judge can play before reading a long explanation.
- The pattern comes from a Lenia mathematical rule.
- The rule has adjustable parameters.
- The same seed and parameters reproduce the same result.
- Raw and Artistic views share the same data.
- Mission feedback, metrics, and exports provide evidence after play.
- Active-region, inspector, and rule-variant tools make emergence legible.
- If the executable fails on a judge machine, the static web fallback still demonstrates Lenia-style rule-driven field formation.
