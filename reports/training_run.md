# E0 正式基线训练

日期：2026-10-14。用户已授权开始训练，并将项目代码提交到 `job267/segformer_plus` 的 main 分支。

当前配置：SegFormer-B0 / CamVid 11 类，输入 384×384，batch=4，CE，AdamW，学习率 6e-5，weight decay=0.01，50 epoch，seed=42，CUDA AMP。正式模型独立从固定 MiT-B0 编码器初始化，按 val mIoU 保存最佳模型。

已于北京时间 2026-10-14 19:03 在后台启动，进程 PID=40612，源码提交为 `acf99ee136416497c59c390f74342b3624fd7b12`，已上传 GitHub main。首轮启动检查通过：train loss=1.196523，val loss=0.788057，val mIoU=0.344552，epoch 耗时约 96.36 秒，峰值 allocated VRAM 820.70 MiB；best 与 last checkpoint 已保存。这是第 1/50 轮的验证记录，尚非最终成绩；后续进度以实时状态和 CSV 为准。

```powershell
python scripts/train.py --config configs/baseline.yaml --offline
```

本次新增 `src/experiment.py`、`src/artifacts.py`、`scripts/train.py` 和 `tests/test_experiment.py`；补充了原子日志写入、操作系统运行锁、最佳模型导出、完整轮次断点状态、源文件和数据哈希核对。25 项测试通过，其中新增测试验证中断恢复与连续训练完全一致、最佳验证模型选择和混淆矩阵 CSV 数值布局。

实时状态以 `results/E0_baseline/status.json` 为准，逐轮日志为 `train_log.csv`。启动信息、源码哈希、环境与实际初始化记录分别位于同目录的 `run_info.json`、`environment.json`、`model_loading.json`；checkpoint 保存在 `last_checkpoint/` 与 `best_model/`。

只有 50 轮训练和验证选模结束后才进行 test 评价；随后自动输出真实指标、曲线、混淆矩阵和固定样本预测。未完成阶段不填写成绩，也不自动启动 E1–E3。
