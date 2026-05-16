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
└─ data_exports/
   ├─ experiment_001_parameters.json
   └─ experiment_001_snapshot.png
```

Do not submit only the source code. Judges should receive the executable plus explanation and evidence. Include `web_html/` as a fallback only; the native executable remains the primary work.

The final explanation should describe only features that are actually implemented in the submitted `peterMath.exe`. If Physarum or GPU compute are still incomplete, keep them out of the student-facing claims and focus on the strongest finished Lenia, Reaction-Diffusion, Game of Life, metrics, and export evidence.
