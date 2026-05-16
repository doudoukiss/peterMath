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

- `lenia_hero.png`: main Lenia artwork.
- `lenia_showcase.png`: six representative Lenia cases.
- `judge_mode_reference.png`: Raw Math View, Artistic View, and explanation/evidence panel.
- `show_mode_storyboard.png`: the Lenia-only guided show sequence.
- `major_cases_gallery.png`: one-click Lenia case gallery.
- `lenia_explanation_reference.png`: artistic field plus explanation panel.

Native app exports and evidence packs should be created during final review and placed in the final submission folder, not committed to the repository.
