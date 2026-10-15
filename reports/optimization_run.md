# E1–E3 优化执行记录

日期：2026-10-15。用户授权按原方案继续优化；沿用已授权的 GitHub main 代码同步。

E0 已完成 50 轮，最佳验证轮次 34，Val mIoU=67.76%，对应 Test mIoU=59.25%。已核验完整逐轮日志、最佳权重 SHA256 和测试混淆矩阵，错误分析见 baseline_analysis.md。

| 顺序 | 实验 | 增强 | 损失 | 轮数 |
|---|---|---|---|---:|
| 1 | E1_augmentation | 水平翻转 + brightness/contrast/saturation 0.2 | CE | 50 |
| 2 | E2_dice | 关闭 | CE + 0.5 Dice | 50 |
| 3 | E3_combined | 与 E1 相同 | CE + 0.5 Dice | 50 |

每组独立从固定 ImageNet MiT-B0 编码器初始化，seed=42，输入 384×384，batch=4，AdamW，lr=6e-5，weight decay=0.01，CUDA AMP。E0 原有训练实现保持一致；不从 E0 微调权重接续。验证集最高 mIoU 选权重，训练结束后各组执行一次最终测试。

启动前检查：31 项测试通过；E1 真实批次 CE=2.406290、梯度范数=4.493629，E3 真实批次 CE+Dice=2.868187、梯度范数=4.662997，两者 optimizer_updated=true。诊断保存在 results/stage4/，不计入正式训练。

scripts/run_experiments.py 持有系统文件锁，依次运行 E1→E2→E3；单组失败就停止队列。每次启动前核验源文件、配置和依赖哈希，逐组归档后重新核验数据来源、原训练代码、权重与指标，再更新 comparison.csv 和 reports/ablation.md。运行中不修改 src/、scripts/、configs/ 或 requirements.txt。中断后使用同一代码执行 `python scripts/run_experiments.py --offline --resume`。

实时状态以 results/ablation_queue/status.json 和各组 status.json 为准。三组结束后才判断收益，按最高 Val mIoU 选择后续演示方案；当前没有新增实验成绩。

## 已验证的正式启动

北京时间 2026-10-15 09:40 启动后台顺序队列，PID=24284；E1 子进程 PID=50488。训练源码提交 `9bbabf5511cdb31487c559206db3d741c0c3ed33` 已推送至 GitHub main。E1 的来源与初始化日志确认使用固定 ImageNet MiT-B0 编码器及新建 11 类分割头，未启用续训。

E1 首轮训练与验证于 09:41:30 完成：train CE=1.199229，val CE=0.806523，val mIoU=34.35%，耗时 47.29 秒，峰值 allocated VRAM=820.70 MiB。best_model 与 last_checkpoint 已保存，stderr 为空。这是第 1/50 轮的启动检查，不代表最终成绩。队列将在 E1 完整训练、测试和归档后自动启动 E2，再启动 E3；本记录不会随每轮变化，实时轮数以状态文件为准。
