# E4/E5 多种子执行记录

日期：2026-10-18。用户授权继续对 E4/E5 的小幅差异做多种子复核。协议见 docs/multiseed_protocol.md。

原 seed=42 已完成并核验：E4 Val/Test mIoU=72.877041%/64.287681%，E5=73.061277%/64.916999%。保留原始归档，既不重复训练也不重新测试。

新增顺序为 E4_seed2024 → E5_seed2024 → E4_seed3407 → E5_seed3407，每次 50 轮。继续复用原 E4/E5 训练引擎，只有种子和输出身份改变；同一种子下预训练构造后的全部初始参数及 buffers 必须具有相同 SHA256。E5 权重仍为 results/stage6/class_weights.json 中的固定训练频率权重。

64 项测试通过，包含固定种子矩阵、初始模型哈希不消耗 RNG、原始验证混淆矩阵复算、样本标准差、排除未配对运行、无提前选方案、只按验证均值选择、同分保留 E4 及 seed=42 固定部署代表。真实预训练初始化预检使用 scripts/check_multiseed.py，正式入口为 scripts/run_multiseed.py；恢复加 --resume，汇总入口为 scripts/compare_multiseed.py。

队列状态 results/multiseed/queue/status.json；各组逐轮状态与 checkpoint 位于 results/multiseed/E4_seed2024/ 等目录。完成一组后自动更新 runs.csv、summary.csv、paired_differences.csv、per_class_summary.csv 与 reports/multiseed.md。中途只展示完整配对种子的统计，全部完成后才生成 selection.json；三种子结果不证明统计显著性。

实际预训练初始化预检已通过，未进行测试模型推理。seed=2024 下 E4/E5 全部初始参数与 buffers 哈希均为 `32c718d916d8cbc3055eda36adcb66cd84c2ba8764c524ec70fbee48275f110d`；seed=3407 下均为 `214fad2219848b25517f91e05c6df3331d1fa8b1a509cc1cd41f3c16df078aba`。同种子两组完全一致，两个新种子不同。完整源文件、数据、参考 checkpoint 和权重哈希已冻结到 results/multiseed/preflight.json。

## 已核验的正式启动

北京时间 2026-10-18 23:44:24 后台启动队列，PID=31020；首组 E4_seed2024 子进程 PID=21420。训练源码提交 `498eaa07c7fd8f1d69436fd3a99c83dad59e39fb` 已上传 GitHub main。实际模型初始哈希同预检一致，独立预训练初始化，resuming=false，原代码/数据与环境复核通过。

首组第 1/50 轮于 23:45:49 完成：train CE+Dice=1.541598，val CE+Dice=1.086091，val mIoU=34.21%，耗时 67.15 秒，峰值 allocated VRAM=1460.80 MiB。best_model 与 last_checkpoint 已保存，stderr 为空。该数值为启动验收，不代表最终种子成绩或稳定性结论；后续三组由队列自动依次执行。

## 完成记录

2026-10-19 北京时间 03:20:37 队列正常完成；四次新增训练均为 50/50 轮，stderr 为空。原 seed=42 两次归档保持不变，因此完整统计为 3 个配对种子、6 次实验。2026-10-19 再次通过 compare_multiseed.py 核验各组原代码/数据/预检身份、初始化哈希、50 行历史、最佳验证轮次、最佳权重 SHA256、共同标签像素及一次已完成测试；未重新执行测试推理。

| 方案 | Val mIoU 均值±样本 std | Test mIoU 均值±样本 std | 测试像素准确率均值 |
|---|---:|---:|---:|
| E4 | 72.47% ± 0.47 pp | 63.88% ± 0.79 pp | 90.83% |
| E5 | 72.70% ± 0.40 pp | 64.39% ± 0.55 pp | 90.49% |

E5 相对 E4 的配对 Val/Test 差值均值为 +0.231/+0.508 个百分点；对应差值样本 std=0.558/1.017 个百分点。seed=42 和 3407 下 E5 更高，seed=2024 下 Val/Test 分别低 0.303/0.564 个百分点。可以描述为平均小幅改善且存在随机种子波动，不能宣称各次均改善或统计显著。

按预先协议的平均最佳 Val mIoU 选择 E5；部署代表固定使用原有 seed=42 最佳模型（其单次 Test mIoU=64.92%），没有按测试成绩挑种子或集成。完整统计、逐类均值/std 与选择身份见 reports/multiseed.md 及 results/multiseed/selection.json。训练与稳定性描述阶段完成，后续为课程报告和最终复现归档，不自动追加训练。
