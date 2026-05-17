# peterMath — Codex Prompts For Teaching Game Direction

Use these prompts in order. Do not paste all prompts at once. After each prompt, review the diff, build, and run the app.

Run prompts from the real app repository root:

```text
peterMath/
```

The current product direction is **Lenia mathematical life teaching game**. Do not expand into other simulation systems in the first reconstruction pass.

---

## Prompt 0 — Repository audit and build verification

You are working on a Rust desktop project named `peterMath`. The final deliverable is a Windows executable that judges can run without installing developer tools.

Tasks:

1. Inspect repository structure.
2. Verify package metadata, binary name, window title, artifact names, and app-root folder use `peterMath`.
3. Verify `.gitignore` excludes `target/`, exports, and OS/editor noise.
4. Verify `web_html/index.html` is a static fallback and does not require a local server.
5. Run or prepare:
   - `cargo fmt`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test`
   - `cargo run --release`
6. Do not redesign the app in this prompt.

Acceptance:

- Build status and exact blockers are known.
- No simulation logic is removed.
- The next implementation step is clear.

---

## Prompt 1 — Teaching game shell

Transform the opening experience from a lab/demo into a mission-based teaching game.

Tasks:

1. Default to `任务模式`.
2. Add five missions: wake field, shape life, tune rule, prove same data, export evidence.
3. Left panel: mission list, run/pause/reset, tools, simple controls.
4. Center: large simulation canvas with compact mission objective bar.
5. Right panel: feedback, hint, math card, live metrics.
6. Move advanced sliders and diagnostics into expert/evidence sections.

Acceptance:

- A judge can start playing without reading external instructions.
- The simulation remains visually central.
- Mission progress and completion are visible.

---

## Prompt 2 — Mission progress and math cards

Make each mission teach one mathematical idea.

Tasks:

1. Track mission baseline step, active region, metrics, style views, inspection, parameter changes, and exports.
2. Show one goal, one hint, one success message, and one math takeaway per mission.
3. Keep copy Chinese-first and concise.
4. Add tests for mission metadata and completion logic.

Acceptance:

- The five missions can be completed in under three minutes by a prepared presenter.
- Mission logic is deterministic and testable.

---

## Prompt 3 — Lenia visual quality

Keep Lenia as the only main system and make it visually strong.

Tasks:

1. Preserve Raw Math and Artistic views from the same scalar field.
2. Keep deterministic presets: orbital field, twin organisms, kernel ring, sparse soup, dense bloom, coral fading.
3. Ensure the default mission looks alive within 60 steps.
4. Keep brush, stamp, undo/redo, seed, and rule sliders.

Acceptance:

- Visual beauty comes from Lenia field rules.
- No decorative unrelated animation layer is added.

---

## Prompt 4 — Evidence and export

Make evidence an outcome of play.

Tasks:

1. Keep PNG snapshot, parameter JSON, share-state JSON, and evidence pack exports.
2. Add optional top-level `teaching_mission` to exported JSON.
3. Include mission id, title, status, progress, completed missions, and takeaway.
4. Update judge/student docs to describe evidence after play.

Acceptance:

- Existing export fields remain stable.
- Evidence files are readable without the app.

---

## Prompt 5 — Web fallback alignment

Update `web_html/index.html` as a smaller teaching-game fallback.

Tasks:

1. Keep direct file-open behavior.
2. Present the same mission names.
3. Show mission objective, progress, feedback, metrics, Raw/Artistic toggle, share link, and PNG export.
4. Do not require Node, npm, Rust, Python, internet, or a server.

Acceptance:

- The fallback page reflects the native teaching-game direction.
- It remains secondary to `peterMath.exe`.

---

## Prompt 6 — Automatic explanation mode

Keep the current guided show mode as a secondary presentation path.

Tasks:

1. Rename or frame it as `自动讲解`.
2. Do not make it the default first-run experience.
3. Keep the three-minute Lenia story available for student presentation.
4. Ensure manual mission play and automatic explanation do not fight each other.

Acceptance:

- Mission mode is primary.
- Automatic explanation remains useful for presentation.

---

## Prompt 7 — Windows submission polish

Prepare the final competition artifact.

Tasks:

1. Inspect `.github/workflows/windows-release.yml`.
2. Ensure it builds on `windows-latest`.
3. Ensure it uploads `peterMath-windows-x64`.
4. Run `python3 scripts/package_submission.py --out dist/peterMath_windows_submission`; use `--official-name`, `--school`, `--group`, and `--student` for final metadata when known.
5. Include `peterMath.exe`, `START_WINDOWS.bat`, `双击运行-评委版.bat`, `打开备用网页.bat`, judge README, student explanation, screenshots/previews, assets, and `web_html/`.
6. Note SmartScreen warning if code signing is not configured.

Acceptance:

- No judge needs developer tools.
- The artifact tells judges to play missions first.
- The artifact includes the native-first launchers and direct-open web fallback.
