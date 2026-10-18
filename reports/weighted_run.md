# E5 与推理交付记录

日期：2026-10-18。用户授权继续，以 E4 为对照只改变 CE 类别权重；同时交付图片推理与 FP32/FP16 评测。协议见 docs/weighted_protocol.md。

E4 已完成且核验：Val mIoU=72.88%，Test mIoU=64.29%。E5 仍为独立固定预训练初始化、512×512 输入/训练标签、384×384 评价标签、seed=42、batch=4、50 轮、CE+0.5 Dice、AdamW 固定 lr=6e-5；不从 E4 权重接续。

权重只统计 367 张训练 mask，采用逆平方根频率、均值归一化后截断 [0.5,3.0]。实际权重：Sky=0.5000、Building=0.5000、Pole=1.4318、Road=0.5000、Sidewalk=0.6484、Tree=0.5000、SignSymbol=1.3163、Fence=1.3430、Car=0.5689、Pedestrian=1.6985、Bicyclist=2.6397。完整数据与哈希见 results/stage6/class_weights.json。

启动前检查：54 项测试通过；包括加权 CE 与 PyTorch 定义一致、ignore、全 1 还原原 CE+Dice、训练适配器与原引擎一致、非均匀权重断点恢复、拒绝权重变化、验证选模及推理输出。真实 CUDA 加权训练批次更新通过：loss=2.892969，梯度范数=4.164286，optimizer_updated=true，峰值 allocated VRAM=1401.21 MiB。首四张验证样本的检查通过，不作为正式成绩，未进行测试集模型推理。

E4 图片推理：scripts/predict.py 支持单图或目录，输出原尺寸索引/彩色/叠加图及 HTML 图例，样例使用首张验证图片 0016E5_07959。已目视检查样例，样例不代表全数据表现。预览位于 results/inference_demo/index.html。

E4 实测效率：FP32/FP16 的完整 101 张验证 mIoU 均约 72.88%，FP32 混淆矩阵逐项等于归档；FP16 变化 +0.0038 个百分点。batch=1 模型均时分别为 21.14/25.25 ms、allocated 峰值 325.09/228.52 MiB；当前实现 FP16 更省显存，但未加速，默认保留 FP32。完整图片流程均时分别为 92.61/94.89 ms，不含模型加载和保存文件。计时范围、预热和样本详见 inference_benchmark.md；这次评测完成后才启动 E5。

正式命令为 `python scripts/train_weighted.py --offline`，恢复时加 `--resume`。进度为 results/E5_weighted/status.json；结束后自动评价和输出 weighted_comparison.csv、验证/测试逐类对照及 weighted_ablation.md。E5 尚无正式结果，不提前判断加权收益。

## 正式启动检查

北京时间 2026-10-18 20:49:19 后台启动 E5，PID=34932。训练源码为已上传 GitHub main 的 `2b008ae`，独立预训练初始化，resuming=false；权重归档、E4 原始源文件和 checkpoint、数据审计与环境复核均通过。

第 1/50 轮于 20:50:51 完成：train 加权 CE+Dice=1.682864，val 加权 CE+Dice=1.280476，val mIoU=46.18%，耗时 75.52 秒，峰值 allocated VRAM=1460.80 MiB。best_model 和 last_checkpoint 已保存，stderr 为空。该记录为启动验收，最终成绩待全部训练及验证选模结束后产生；实际进度以状态和逐轮 CSV 为准。

## 完成记录

2026-10-18 北京时间 21:45:08 完成 50 轮与归档，最佳验证轮次 50，Val mIoU=73.061277%，Test mIoU=64.916999%，测试像素准确率=90.705805%。相对 E4 的 Val/Test mIoU 分别提升 0.184237/0.629318 个百分点，像素准确率下降 0.235674 个百分点。完整历史、权重 SHA256、原训练集权重、共同评价网格与一次已完成测试均已核验，stderr 为空。

验证集 Pole/SignSymbol 分别提升 2.24/2.57 个百分点，Pedestrian 下降 0.48 个百分点；测试集 Bicyclist/Pole 分别提升 4.95/2.81 个百分点，Fence 下降 2.22 个百分点。当前按验证集选择 E5，收益较小且存在类别取舍；用户授权继续，以固定种子 42、2024、3407 复核 E4/E5，方案见 docs/multiseed_protocol.md，原 E5 结果不覆盖。
