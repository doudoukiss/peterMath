# Screenshot And Preview Guide

Generated preview images are created with:

```bash
cargo run --bin render_preview
```

They are written to:

```text
peterMath_exports/previews/
```

Use these files for judge-facing screenshots:

- `lenia_hero.png`: main Lenia teaching-game field.
- `lenia_showcase.png`: representative Lenia mission scenarios.
- `judge_mode_reference.png`: mission cards, recommended-tool chip, `1 选工具 / 2 选任务 / 3 点生命场`, Raw Math View, Artistic View, and mission/evidence panel.
- `show_mode_storyboard.png`: secondary automatic explanation sequence.
- `major_cases_gallery.png`: one-click scenario gallery.
- `lenia_explanation_reference.png`: artistic field plus math card reference.

Also capture one manual screenshot showing the canvas coachmark or feedback pulse after a field edit. Native app exports and evidence packs should be created during final review and placed in the final submission folder, not committed to the repository.
