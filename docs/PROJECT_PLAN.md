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

Build `peterMath` into a Windows-runnable mathematical artwork and evidence lab:

```text
living field + deterministic seed + adjustable rule + inspector + metrics + exportable evidence
```

The judging story is native-first and Chinese-first. The app should persuade through mathematical beauty, not decorative animation.

## Current Implemented Direction

- Opening experience: Chinese three-system overview comparing Game of Life, Reaction-Diffusion, and Lenia.
- Systems: Game of Life, Reaction-Diffusion, and GPU Lenia are presented as equal mathematical examples; Lenia remains the richest visual artwork.
- Interaction: draw, erase, stamps, presets, density randomization, undo/redo, keyboard shortcuts.
- Explanation: Chinese labels, preset-first parameters, field inspector, kernel lens, phase labels, metric history, and judge guide.
- Advanced interpretability: active-region analysis, population phase trends, Game of Life pattern detection, glider tracking, oscillator periods, and Lenia rule variant comparison.
- Performance: diagnostics, bounded scheduler, CPU texture dirty tracking, `perf_probe`.
- Shareability: PNG/JSON snapshot export, share-state JSON, evidence packs, Game of Life RLE import/export, and offline web fallback.

## Final Submission Checklist

1. Run all validation commands from the repository root.
2. Generate previews with `cargo run --bin render_preview`.
3. Confirm the first screen shows the three-system overview.
4. Create at least one evidence pack from the native app.
5. Confirm `web_html/index.html` opens directly from disk and hash sharing works.
6. Push to GitHub and run the Windows release workflow.
7. Download `peterMath-windows-x64` and confirm it contains:
   - `peterMath.exe`
   - judge templates
   - web fallback
   - assets
   - generated previews
   - screenshot guidance
8. Record or prepare a three-minute demonstration using `评审讲解模式`.

## Working Rule

Do not add features that weaken the judging story. Every visible feature should support one of these claims:

- The pattern comes from a mathematical rule.
- The rule has adjustable parameters.
- The same seed and parameters reproduce the same result.
- Raw and Artistic views share the same data.
- Metrics and exports provide evidence.
- Active-region, pattern, and rule-variant tools make emergence legible.
- If the executable fails on a judge machine, the static web fallback still demonstrates rule-driven pattern formation.
