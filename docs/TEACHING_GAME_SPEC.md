# peterMath Teaching Game Spec

## Product Identity

`peterMath` is a Lenia 数学生命教学游戏. It should feel like a playable learning experience for students and judges, not a technical visualization that only works after reading instructions.

The first-run path is:

```text
Open app -> 1 choose tool -> 2 choose mission -> 3 touch field -> see feedback -> reveal math card -> export evidence
```

The native Rust executable is primary. The static web page is a smaller offline fallback with the same teaching-game story.

## Game Loop

- The app opens in `任务模式`.
- Each mission has one visible goal, one hint, one progress bar, one success message, and one short math takeaway.
- The simulation canvas remains the dominant screen element, with `生命高光图` as the default mission-play rendering style.
- The mission panel shows an actionable recommended-tool chip so `1 选工具` is a real control, not just instructional copy.
- The first canvas view shows a small coachmark; field edits, tool changes, parameter changes, mission completion, and exports create canvas-local feedback pulses.
- Canvas feedback pulses render above the coachmark so success/export messages are never hidden by first-run guidance.
- A compact `看懂演化` strip explains `K*u`, `G(K*u)`, damping, same-field views, and seed/evidence after the canvas.
- Expert controls exist, but they are secondary to mission play.
- `自动讲解` remains available for presentation, but it is not the default path.

## Missions

1. `wake_field` / 唤醒生命场
   - Action: run the default Lenia field for at least 60 steps.
   - Success: visible active mass remains in the field.
   - Teaches: Lenia is a live rule-driven field, not a video.

2. `shape_life` / 塑造生命
   - Action: use brush or stamp to add structure.
   - Success: active region increases meaningfully from the mission baseline.
   - Teaches: local edits change later evolution.

3. `tune_rule` / 半径挑战
   - Action: change one Lenia rule parameter and run at least 80 steps.
   - Success: metrics differ from the mission baseline.
   - Teaches: small parameter changes produce different behavior.

4. `same_field` / 证明同一数据
   - Action: view both Raw Math and Artistic styles and inspect a point.
   - Success: both views have been seen and the inspector has a point.
   - Teaches: artistic view is a color mapping of the same scalar field.

5. `evidence_report` / 生成证据报告
   - Action: export share state or evidence pack.
   - Success: JSON/PNG evidence is written.
   - Teaches: the visible state is reproducible by seed, parameters, metrics, and step count.

## Interface

- Left panel: title, current mission, recommended tool, mission list, run/pause/reset, tool buttons, simple seed/display controls, collapsed expert/evidence controls.
- Center: objective bar, large simulation canvas, compact status line, and `看懂演化` strip.
- Right panel: mission feedback, hint, unlocked math card, live metrics, and collapsed expert diagnostics.
- Canvas: draw/erase/stamp show a cursor-radius preview so field edits feel tactile.
- Onboarding copy: the first screen should visibly say `1 选工具 / 2 选任务 / 3 点生命场`.
- Top bar: project name, `任务模式`, current mission, backend, seed, step count, and phase.

## Copy Tone

Chinese is primary. Use short action-oriented labels:

- `任务模式`
- `目标`
- `提示`
- `成功`
- `数学卡片`
- `证据`
- `专家设置`

Avoid making the primary UI sound like a technical demo, research lab, or pure artwork. Expert panels may still use precise terms such as field, kernel, growth response, damping, metrics, and inspector.

## Arcade Science Feel

- Mission rows are cards with active/completed states.
- The central HUD shows current mission, next action, progress, and short feedback pulses.
- Success, warning, and info states use distinct accents.
- `生命高光图` emphasizes birth/decay changes while staying derived from the same scalar field.

## Export Behavior

Snapshot JSON, share-state JSON, and evidence packs keep existing fields and add an optional top-level `teaching_mission` object:

```json
{
  "mission_id": "wake_field",
  "title_zh": "唤醒生命场",
  "status": "completed",
  "progress": 1.0,
  "completed_missions": ["wake_field"],
  "takeaway_zh": "Lenia 每一步都由同一个局部规则实时计算。"
}
```

## Acceptance

- A fresh judge can start playing within seconds.
- The native package includes `START_WINDOWS.bat`, `双击运行-评委版.bat`, `打开备用网页.bat`, `评委入口.html`, `web_html/`, screenshots/video/data placeholders, and a package manifest.
- Package metadata can be supplied with `--official-name`, `--school`, `--group`, and `--student`; placeholder defaults are acceptable until final submission details are known.
- The five missions can be completed in under three minutes by a prepared student presenter.
- Existing Lenia simulation, deterministic presets, GPU/CPU paths, metrics, inspector, rule comparison, exports, and Windows artifact naming remain intact.
- The fallback page opens directly from disk and uses the same mission names.
