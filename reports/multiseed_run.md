# E4/E5 多种子执行记录

日期：2026-10-18。用户授权继续对 E4/E5 的小幅差异做多种子复核。协议见 docs/multiseed_protocol.md。

原 seed=42 已完成并核验：E4 Val/Test mIoU=72.877041%/64.287681%，E5=73.061277%/64.916999%。保留原始归档，既不重复训练也不重新测试。

新增顺序为 E4_seed2024 → E5_seed2024 → E4_seed3407 → E5_seed3407，每次 50 轮。继续复用原 E4/E5 训练引擎，只有种子和输出身份改变；同一种子下预训练构造后的全部初始参数及 buffers 必须具有相同 SHA256。E5 权重仍为 results/stage6/class_weights.json 中的固定训练频率权重。

64 项测试通过，包含固定种子矩阵、初始模型哈希不消耗 RNG、原始验证混淆矩阵复算、样本标准差、排除未配对运行、无提前选方案、只按验证均值选择、同分保留 E4 及 seed=42 固定部署代表。真实预训练初始化预检使用 scripts/check_multiseed.py，正式入口为 scripts/run_multiseed.py；恢复加 --resume，汇总入口为 scripts/compare_multiseed.py。

队列状态 results/multiseed/queue/status.json；各组逐轮状态与 checkpoint 位于 results/multiseed/E4_seed2024/ 等目录。完成一组后自动更新 runs.csv、summary.csv、paired_differences.csv、per_class_summary.csv 与 reports/multiseed.md。中途只展示完整配对种子的统计，全部完成后才生成 selection.json；三种子结果不证明统计显著性。

实际预训练初始化预检已通过，未进行测试模型推理。seed=2024 下 E4/E5 全部初始参数与 buffers 哈希均为 `32c718d916d8cbc3055eda36adcb66cd84c2ba8764c524ec70fbee48275f110d`；seed=3407 下均为 `214fad2219848b25517f91e05c6df3331d1fa8b1a509cc1cd41f3c16df078aba`。同种子两组完全一致，两个新种子不同。完整源文件、数据、参考 checkpoint 和权重哈希已冻结到 results/multiseed/preflight.json。
