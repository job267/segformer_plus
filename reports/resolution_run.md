# E4 执行记录

日期：2026-10-18。用户授权继续推进此前建议的优化顺序，先执行 512×512 分辨率实验，完整口径见 docs/resolution_protocol.md。

E0–E3 已全部完成；E3 为当前验证集选择方案：Val mIoU=71.00%，Test mIoU=61.42%。E4 独立从固定预训练编码器初始化，seed=42、batch=4、50 轮，增强、CE+0.5 Dice、AdamW 和固定学习率均与 E3 相同。训练图像/标签为 512×512，验证/测试标签为共同的 384×384 网格；不改变原 E0–E3 结果。

启动前检查已通过：

- 40 项测试通过，新增不同训练/评价网格、直接原标注缩放、配置差异约束与验证选模汇总测试。
- 真实 CUDA AMP、batch=4、512×512 前向/反向/更新通过：loss=2.859554，梯度范数=4.953426，optimizer_updated=true，峰值 allocated VRAM=1401.21 MiB。
- 101 张验证、233 张测试样本的标签均与 E3 逐像素一致，11 类像素总数同原始归档。
- 真实前四张验证样本的 FP32 512 输入/384 标签检查通过。该诊断为未训练分割头检查，其指标不作为正式成绩；检查期间没有测试模型推理。

诊断位于 results/stage5/E4_smoke/。正式执行命令为 `python scripts/train_resolution.py --offline`，中断后加 `--resume`，恢复要求配置、代码和审计哈希完全相同。正式状态位于 results/E4_resolution/status.json，逐轮日志为 train_log.csv；完成后自动生成独立分辨率比较表及 reports/resolution_ablation.md。E4 结果尚未产生，不预设提高分辨率会提升精度。
