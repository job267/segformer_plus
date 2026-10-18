# E5 类别加权实验协议

冻结日期：2026-10-18，正式训练前记录。用户授权以 E4 为对照开展 E5，同时完成 E4 推理与效率评测。E0–E4 原有代码、配置、权重及结果不修改。

对照 E4：512×512 输入/训练标签、384×384 验证/测试标签，第 50 轮最佳验证 mIoU=72.877041%，对应 Test mIoU=64.287681%。E5 独立从同一固定 ImageNet MiT-B0 编码器和新建分割头开始，不接续 E4 微调权重。

唯一计划因素为 CE 的类别权重。对 367 张未增强训练 mask 直接最近邻缩放到 512×512，排除 ignore=255，计算各类频率 `f_c`；`raw_c = f_c^(-1/2)`，`w_c = clip(raw_c/mean(raw),0.5,3.0)`。11 类必须均有正的训练像素。权重规则运行前固定，不能根据验证/测试类别频率或结果改权重。逐类计数、权重、全部训练 ID、train 清单及数据审计哈希保存在 results/stage6/class_weights.json。

加权 CE 为 `sum(w[y] * CE_pixel) / sum(w[y])`，忽略像素不参与分子/分母；全 ignore 返回可反传零值。Dice 定义及系数 0.5 保持不变。损失均在 FP32 计算，logits 先插值到标签网格。逐轮 loss 保留按批次有效像素数加权的原日志定义，因此加权 CE 的日志不是全数据权重和归一化的精确 CE，也不与 E4 原始 loss 数值直接比较。

相同数据、预处理、增强、初始化来源、seed=42、batch=4、50 轮、AdamW lr=6e-5、wd=0.01、固定学习率和 CUDA AMP。每轮用 384×384 共同标签网格的 FP32 验证 mIoU 选择最高轮次；完整训练后只对该权重执行一次最终测试，归档中断后复用同一权重已完成的测试结果。

原 E4 训练引擎不改动。E5 独立适配器显式接收加权 criterion，复用原训练 epoch、评价和原子 checkpoint/RNG 工具。在全 1 权重的原生小模型测试中，适配器与原引擎训练权重和 loss/指标逐项一致；非均匀权重的中断恢复与连续训练一致。checkpoint 保存实际权重张量，恢复时同时核对配置、源文件清单、数据审计、权重归档和 E4 对照 checkpoint 身份。

结果保存 results/weighted_comparison.csv、weighted_per_class.csv（Test）、weighted_validation_per_class.csv（Val）和 reports/weighted_ablation.md。预先关注验证集 Pole、SignSymbol、Pedestrian，同时报告全部 11 类，按整体验证 mIoU 选择 E4/E5，同分保留 E4。收益可能为负；测试集已查看过，不能宣称新增盲测；单 seed 不证明稳定性。多 seed、长预算、scheduler 和新架构未自动排入队列，待 E5 结果后决定。

推理侧使用 E4 已完成的最佳模型。FP32/FP16 只在验证集检查精度；测速单独预热并同步 CUDA，模型 batch=1、输入 512×512、输出评价网格 384×384，与包含读图/预处理/传输/原尺寸着色的图片流程分开报告。FP16 为 CUDA autocast、权重仍 FP32，默认精度依据本机实测决定，不预设它更快。效率评测在 E5 训练前执行。
