# Final Submission Structure

Recommended final folder:

```text
学校-数学工程创意实践类-小学组-学生姓名/
├─ peterMath.exe
├─ START_WINDOWS.bat
├─ 双击运行-评委版.bat
├─ 打开备用网页.bat
├─ 评委入口.html
├─ README_给评委.txt
├─ 提交前最后检查.md
├─ PACKAGE_MANIFEST.json
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
│  ├─ 03_lenia_inspector.png
│  └─ 04_judge_mode.png
├─ video/
│  └─ 3分钟演示视频.mp4
├─ previews/
│  ├─ lenia_hero.png
│  ├─ lenia_showcase.png
│  ├─ judge_mode_reference.png
│  ├─ show_mode_storyboard.png
│  ├─ major_cases_gallery.png
│  └─ lenia_explanation_reference.png
└─ data_exports/
   ├─ experiment_001_parameters.json
   ├─ experiment_001_snapshot.png
   ├─ peterMath_share_state.json
   └─ SUMMARY.md
```

Do not submit only the source code. Judges should receive the executable plus explanation, preview images, and evidence exports. Include `web_html/` as a fallback only; the native executable remains the primary work.

Generate this folder with:

```bash
python3 scripts/package_submission.py --out dist/peterMath_windows_submission
```

Use `--zip --official-name "学校-数学工程创意实践类-小学组-学生姓名" --school "学校名称" --group "小学组" --student "学生姓名"` when preparing the final archive. The final explanation should describe only features that are actually implemented in the submitted `peterMath.exe`: mission-based teaching game, Lenia scenarios, GPU Lenia, CPU fallback, 数学原始图/艺术表达图/生命高光图, math cards, inspector, kernel lens, metrics, active-region analysis, Lenia rule variant comparison, performance diagnostics, and evidence exports.
