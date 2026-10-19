# 综合报告生成与检查

Word 文件位于 `reports/SegFormer_B0_课程设计与实验综合报告.docx`。它以课程报告为主线，合并基线分析、消融、分辨率、类别加权、多种子、效率及基础验收记录，去掉重复内容与过时运行进度。文档保留 12 张原生表格、4 个原生数学公式、6 个编号图组和 12 张嵌入图片。

`build_word_report.py` 使用 python-docx，从原始结果 JSON/CSV 核对全部 10 次运行的 50 轮历史、验证选模、测试混淆矩阵、一次测试记录和实际权重 SHA256。只读取已完成结果，不重新训练或执行测试集推理。课程报告来源和指标文件哈希记入 `reports/word_report_evidence.json`。

文档制作使用 Codex 提供的独立文档运行环境，不修改已冻结的训练依赖。使用该环境的 Python 运行 `delivery/build_word_report.py` 会重建 DOCX；中间源稿保存在 `.cache/word_report/combined_source.md`。随后，在 Windows 上使用新建的独立 Microsoft Word 实例执行：

```powershell
./delivery/export_word_pdf.ps1 -InputDocx reports/SegFormer_B0_课程设计与实验综合报告.docx -OutputPdf .cache/word_report/preview.pdf
```

该步骤更新目录和页码，导出本地检查用 PDF，然后只关闭自己新建的实例，不终止现有 Word 进程或修改其他打开的文档。本机未提供捆绑的 Windows LibreOffice，因而采用 Microsoft Word 原生导出，再通过 documents 技能提供的 `render_docx.py` 栅格化预览；`render_word_report.py` 的 `--renderer` 参数指定该脚本绝对路径。

最终文件已经过 20 页逐页图片检查。PDF 与页面图片只用于内部排版验收，不作为本次交付文件。综合 Word 是独立新增报告，之前的训练提交包和 release.json 保留原有身份。修改 Word 后需要重新更新目录、渲染并逐页检查。

`python -m delivery.verify_word_report` 核对 Word 中的总体、逐类、配对运行及三种子逐类统计表，并验证 12 张嵌入图片与原 PNG 字节一致、4 个原生公式和已渲染页数；同时清除 Word 自动写入的最后编辑者个人字段。它要求本地检查 PDF 已位于 `.cache/word_report/report_final.pdf`。最终文档 SHA256、来源和检查记录保存在 word_report_evidence.json。
