# SegFormer-B0 / CamVid 课程设计

依据根目录的项目方案，Stage 0–1 基础构建和验收已完成，当前进入 **Stage 2：E0 正式基线训练**。E1–E3 消融、演示和课程报告按后续阶段开展。基础检查记录见 `reports/stage0_stage1_status.md`；所有数值必须来自实际生成的文件。

**当前验证：25 项测试通过，包含基础验收、断点续训一致性、最佳验证模型选择与指标导出检查。CamVid 完整数据与预训练编码器已就绪。**

## 1. 环境与安装

已核验本机 Python 3.12.3、PyTorch 2.3.1、CUDA 构建版本 11.8、Transformers 4.44.1，RTX 4060 Laptop GPU 约 8GB；驱动 576.02。详细记录位于 `environment.json`。当前 Python 位于 `D:/anaconda/777/python.exe`，已有依赖，可直接运行以下命令。

新环境建议使用 Python 3.12，在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

CPU 环境把 PyTorch 安装来源改为 `https://download.pytorch.org/whl/cpu`。这里固定已验证的版本，安装命令参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。后续命令里的 `python` 可以替换成 `.\.venv\Scripts\python.exe`。

## 2. 按顺序完成基础验收

```powershell
# 环境与四组 YAML 配置检查
python scripts/check_environment.py

# 从同一固定版本下载划分、图像和 11 类标签，可重复运行续传
python scripts/download_camvid.py --workers 4

# 核验全部文件，输出标签统计、划分隔离检查和训练样本叠加图
python scripts/audit_dataset.py

# 载入指定预训练编码器，保存 missing/unexpected keys
python scripts/check_model.py

# 核心逻辑测试，不下载外部数据或权重
python -m pytest -q --basetemp=.cache/pytest

# 真实训练集的一批数据：前向、CE、反向、AdamW 更新
python scripts/smoke_test.py --offline --batch-size 4

# 20 张固定训练样本的学习诊断，单独归档，不作为正式实验
python scripts/check_overfit.py --offline --samples 20 --epochs 20
```

任何命令失败都会返回非零退出码；数据审计、模型载入、冒烟和过拟合脚本会保留诊断 JSON。先排查错误，再进入 E0。首次下载权重需要联网；后续 `--offline` 仅访问本项目缓存。也可先手动按 [数据说明](data/README.md) 放置完整数据，跳过下载命令。

只有需要验证 CPU 或暂时没有真实数据时，运行显式的合成检查：

```powershell
python scripts/smoke_test.py --synthetic --offline --device cpu --size 64 --output-dir results/stage1/cpu_logic_smoke
# 无权重时显式测试同一 B0 架构；日志标记随机初始化
python scripts/smoke_test.py --synthetic --random-init --device cpu --size 64 --output-dir results/stage1/random_logic_smoke
```

合成检查不能替代真实数据验收。真实数据检查不接受 `--random-init`。默认数据缺失时会报错，不会自动生成数据或替换模型。

## 3. 项目结构

```text
configs/        E0–E3 配置，后三组继承 baseline.yaml
src/            配置、标签、Dataset、成对变换、模型、损失、指标、epoch 函数
scripts/        环境自检、下载、审计、模型载入、冒烟、小样本学习检查
tests/          标签、划分隔离、增强同步、ignore、Dice、指标与配置测试
docs/           冻结的实验协议
data/           获取与许可说明；实际数据被 Git 忽略
results/        检查日志、配置快照和可视化；后续正式实验独立归档
reports/        当前阶段验收记录
environment.json  真实环境记录
```

## 4. 模型、数据和配置约定

- 主模型采用 [Transformers 4.44.1 的 SegFormer 实现](https://huggingface.co/docs/transformers/v4.44.1/model_doc/segformer)。载入 [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0) 的 ImageNet 编码器，然后严格复制编码器权重，并新建 11 类解码头。分类层的两项多余权重可以解释，其他缺失或多余键会直接报错。权重版本固定为 `80983a413c30d36a39c20203974ae7807835e2b4`。
- 数据来自 [lih627/CamVid](https://github.com/lih627/CamVid)，版本固定为 `79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6`；使用该来源的三个清单及 `CamVidGray` 标签，不混入其他划分。367/101/233 是参考数量，验收以审计后的文件为准。
- 配置中路径相对于项目根目录解析；YAML 的 `extends` 相对于当前配置文件解析。实际修改训练预算、输入尺寸或 batch 时，修改 `baseline.yaml`，保证四组继承同一设置。
- 标签 0 为 Sky，有效类为 0–10，255 为 ignore。默认显式选择 `index` 编码；RGB 11 类标签可设 `rgb`，未知颜色会报错。不会自动把原始 32 类 RGB 标签当成 11 类数据。
- 使用自有成对变换实现 384×384 拉伸缩放和 ImageNet Normalize；图像用双线性、标签用最近邻。验证、测试不采用随机增强。标签 ID 从不平移。
- 模型缓存位于 `.cache/huggingface/`，数据和大型权重不会纳入 Git。Windows 缺少符号链接权限时缓存会使用普通文件，无需管理员权限。

详细指标定义、消融公平性、随机性限制和下一阶段门槛见 [实验协议](docs/experiment_protocol.md)。

## 5. 检查产物

| 文件/目录 | 用途 |
|---|---|
| `environment.json` | 依赖、Python、GPU、驱动、CUDA 实际信息 |
| `data/CamVid/source_metadata.json` | 数据来源版本、下载状态和清单摘要 |
| `results/stage1/data_audit.json` | 实际配对、标签、像素统计、文件哈希和错误 |
| `results/stage1/previews/` | 原图、索引标签、彩色标签、叠加图 |
| `results/stage1/model_loading.json` | 编码器载入情况及新建分割头参数 |
| `results/stage1/real_smoke/` | 真实批次前向/反向/更新、配置和日志 |
| `results/stage1/cpu_logic_smoke/` | 单独标记的 CPU 合成逻辑检查 |
| `results/stage1/overfit/` | 训练子集 ID、可重载的实际配置、原配置副本、逐轮日志和随机样本叠加图 |

## 6. 正式 E0 训练与断点恢复

```powershell
python scripts/train.py --config configs/baseline.yaml --offline
# 进程中断后，使用完全相同的配置与代码从上一个完整 epoch 恢复
python scripts/train.py --config configs/baseline.yaml --offline --resume
```

默认使用 384×384、batch=4、CE、AdamW、学习率 6e-5、50 epoch 和 CUDA AMP，独立从固定的 MiT-B0 编码器开始。启动前重新核对审计清单、全部数据文件哈希和 Stage 1 验收。模型不会从小样本诊断权重接续训练。

训练目录为 `results/E0_baseline/`。`status.json` 显示当前状态和完成轮数；`train_log.csv` 逐轮记录 train/val loss、val mIoU、学习率、耗时和峰值 allocated VRAM。`best_model/` 是按 val mIoU 选择的 Hugging Face 权重；`last_checkpoint/state.pt` 保存最新完整轮次的模型、优化器、AMP scaler、随机状态、数据加载器状态与历史，支持断点恢复。运行时记录配置、真实环境、代码 SHA256 和 Git 提交号；续训会拒绝配置或代码发生变化的运行，已有实验不会被静默覆盖。

50 轮结束后，自动用最佳验证权重进行一次完整 test 评价，保存 `metrics.json`、每类 IoU、混淆矩阵 CSV/PNG、训练曲线，以及六个预先确定位置的测试样本可视化。测试只用于最终评价。若归档中断，已完成的同一权重测试结果会复用，不重新选择模型。

实际启动与进度记录见 `reports/training_run.md`。E1–E3 不会自动启动；FP32/FP16 效率测试仍在后续阶段。
