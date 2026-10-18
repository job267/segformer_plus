# E0–E3 消融实验

已完成 4/4 组。所有数字来自各组原始 metrics.json 和逐轮日志。
空白表示尚未完成，不是零分；测试集不参与选模。

| 实验 | 状态 | 最佳 Val mIoU | Test mIoU | 相对 E0 的 Test 变化 |
|---|---|---:|---:|---:|
| E0_baseline | completed | 67.76% | 59.25% | +0.00 个百分点 |
| E1_augmentation | completed | 68.92% | 60.25% | +1.00 个百分点 |
| E2_dice | completed | 69.34% | 60.73% | +1.48 个百分点 |
| E3_combined | completed | 71.00% | 61.42% | +2.17 个百分点 |

按验证集选择的方案为 **E3_combined**，最佳 Val mIoU=71.00%；该方案的最终 Test mIoU=61.42%。方案选择依据为验证集，不按 Test 排名改选。

- E1_augmentation：Test mIoU 高于 E0 1.00 个百分点；最佳 Val 变化 +1.16 个百分点。
- E2_dice：Test mIoU 高于 E0 1.48 个百分点；最佳 Val 变化 +1.57 个百分点。
- E3_combined：Test mIoU 高于 E0 2.17 个百分点；最佳 Val 变化 +3.24 个百分点。

单 seed 对照不证明统计显著性或因果机制。训练损失的 CE 与 CE+Dice 数值不能直接作跨组优劣比较。输入、预算、优化器、种子、数据及原有训练代码一致，唯一计划差异是增强开关与 Dice 项。每类指标见 results/comparison_per_class.csv。
