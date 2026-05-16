# peterMath Project Plan

## Repository Structure

This folder, `peterMath/`, is the Git repository root for the competition app.

```text
peterMath/
├─ Cargo.toml
├─ Cargo.lock
├─ README.md
├─ .gitignore
├─ .github/workflows/windows-release.yml
├─ assets/
├─ docs/
├─ judge_submission_template/
├─ web_html/
└─ src/
```

The outer `peterMath_pack/` folder is a planning/archive workspace. It should not be the main repository root.

## Competition Objective

Build `peterMath` into a Windows-runnable mathematical life-system laboratory:

```text
rules + fields + parameters + deterministic seeds + metrics + exportable evidence
```

The app should be judged as mathematical engineering software, not as decorative animation.

## Immediate Next Steps

1. Run `cargo fmt` and commit a formatted baseline.
2. Run `cargo test` and `cargo run --release`; fix any build or runtime issues.
3. Push this folder to a GitHub repo named `peterMath`.
4. Run the Windows GitHub Actions workflow and confirm `peterMath-windows-x64` contains `peterMath.exe` and `web_html/index.html`.
5. Improve Lenia first: stronger presets, finer visual field, clearer kernel/growth explanation.
6. Improve Reaction-Diffusion second: feed/kill presets, Raw Math View, contour-like Artistic View.
7. Turn Judge Mode into a guided 3-minute flow with before/after metric comparison.
8. Add experiment logging and submission snapshot export.
9. Add Physarum only after the first two systems and Judge Mode are solid.
10. Record screenshots and a 3-minute video from the final Windows build.

Use `cargo run --bin render_preview` after visual changes to produce still images for review.

## Working Rule

Do not add features that weaken the judging story. Every visible feature should support one of these claims:

- The pattern comes from a mathematical rule.
- The rule has adjustable parameters.
- The same seed and parameters reproduce the same result.
- Metrics and exports provide evidence.
- The final program runs on Windows without developer tools.
- If the executable cannot run on a judge machine, the static web fallback can still demonstrate the rule-driven animation.
