# Final Submission Structure

Recommended final folder:

```text
学校-数学工程创意实践类-小学组-学生姓名/
├─ peterMath.exe
├─ README_给评委.txt
├─ 作品说明_学生版.docx 或 .md
├─ 学生作品原创性与AI规范使用声明.docx
├─ 参数实验记录表.csv
├─ 3分钟演示视频.mp4
├─ web_html/
│  ├─ index.html
│  └─ README.md
├─ screenshots/
│  ├─ 01_lenia_raw.png
│  ├─ 02_lenia_artistic.png
│  ├─ 03_reaction_diffusion.png
│  └─ 04_judge_mode.png
├─ previews/
│  ├─ lenia_hero.png
│  ├─ reaction_diffusion_texture.png
│  └─ judge_mode_reference.png
└─ data_exports/
   ├─ experiment_001_parameters.json
   ├─ experiment_001_snapshot.png
   ├─ peterMath_share_state.json
   └─ SUMMARY.md
```

Do not submit only the source code. Judges should receive the executable plus explanation, preview images, and evidence exports. Include `web_html/` as a fallback only; the native executable remains the primary work.

The final explanation should describe only features that are actually implemented in the submitted `peterMath.exe`: GPU Lenia, CPU fallback, Raw/Artistic views, inspector, kernel lens, metrics, active-region analysis, Game of Life pattern detection, Lenia rule variant comparison, performance diagnostics, evidence exports, Reaction-Diffusion, Game of Life, and RLE import/export.
