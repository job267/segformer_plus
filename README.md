# SegFormer-B0 / CamVid 课程设计

依据根目录的项目方案，E0–E5 训练、消融及 E4/E5 三种子复核均已完成。按平均最佳 Val mIoU 选择 E5：三种子 Val=72.70%±0.40 pp，Test=64.39%±0.55 pp；E4 分别为 72.47%±0.47 pp、63.88%±0.79 pp，详见 [多种子结果](reports/multiseed.md)。部署代表按预先协议使用 E5 原 seed=42 最佳模型，其单次 Test mIoU=64.92%。图片推理和 FP32/FP16 评测已完成，剩余为课程报告与最终复现归档。

**当前验证：64 项测试通过，新增固定种子矩阵、配对统计、样本标准差、无提前选模及初始化哈希检查。E4/E5 原训练实现和已完成结果保持冻结。**

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
- 配置中路径相对于项目根目录解析；YAML 的 `extends` 相对于当前配置文件解析。E0–E3 配置和实现已冻结；扩展实验使用独立配置覆盖，不修改 `baseline.yaml`。
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

实际启动与进度记录见 `reports/training_run.md`。单组入口不会自动启动其他实验；FP32/FP16 效率测试仍在后续阶段。

## 7. E1–E3 顺序训练与比较

```powershell
# 验证 E0 归档和冻结配置，依次训练 E1、E2、E3
python scripts/run_experiments.py --offline
# 中断后跳过已完整归档的组，恢复未完成的组
python scripts/run_experiments.py --offline --resume
# 四组全部完成后重新核验并生成比较；查看中途记录加 --allow-incomplete
python scripts/compare_experiments.py
```

E1 仅加水平翻转和轻量颜色增强；E2 仅改为 CE+0.5 Dice；E3 同时启用两项。各组独立从固定预训练编码器、相同 seed=42 和新建分割头开始，均为 50 轮，其余预算、数据、预处理和优化器与 E0 相同。运行前校验 E0 原有训练源码哈希；队列运行期间冻结源码、配置和依赖文件。出现失败时停止后续实验，保留诊断与恢复入口。

队列进度位于 `results/ablation_queue/status.json`；各组 `status.json` 与 `train_log.csv` 保存实际轮数。每完成一组便更新 `results/comparison.csv`、`results/comparison_per_class.csv` 和 [消融记录](reports/ablation.md)，未完成组留空。最终依据最高 Val mIoU 选择方案并保存 `results/validation_selected_experiment.json`，Test 仅用于最终评价。增强或 Dice 的收益以实际结果为准。权重、原图预测和运行日志保留在本地，不纳入 Git。

## 8. E4 分辨率扩展实验

```powershell
# 独立诊断目录：真实 512×512、batch=4 的一次参数更新
python scripts/smoke_test.py --config configs/resolution.yaml --offline --batch-size 4 --output-dir results/stage5/E4_smoke
# 核对全部 val/test 标签与 E3 完全一致，并检查真实 FP32 验证批次
python scripts/check_resolution.py --offline
# 正式 E4：独立预训练初始化，50 轮；中断后加 --resume
python scripts/train_resolution.py --offline
# 全部结束后重建独立比较表；中途查看加 --allow-incomplete
python scripts/compare_resolution.py
```

E4 沿用 E3 的数据、增强、CE+0.5 Dice、seed=42、batch=4、50 轮和 AdamW 设置；输入及训练标签为 512×512。验证、测试标签直接从原始标注最近邻缩放至 384×384，模型 logits 双线性插值到这一共同网格后计算 loss 与指标。以相同网格的最高验证 mIoU 选模，最终测试一次，不从 E3 微调权重接续。训练引擎与 E0–E3 原有源文件保持一致，仅增加独立数据适配器和执行入口。

进度位于 `results/E4_resolution/status.json`；完成后自动生成 `results/resolution_comparison.csv`、`results/resolution_per_class.csv`、`results/resolution_validation_selected_experiment.json` 和 [分辨率比较](reports/resolution_ablation.md)。原 E0–E3 比较表与选模记录保留。该实验固定训练轮数和 batch，增加了每步像素计算量，不能称为等算力实验；当前测试集已查看过，不将新增结果视为未见盲测。执行记录见 [E4 运行说明](reports/resolution_run.md)。

## 9. E5 类别加权实验

```powershell
# 训练 mask 统计、权重归档、真实 512×512 参数更新和验证批次检查
python scripts/check_weighted.py --offline
# 正式独立训练 50 轮；中断后加 --resume
python scripts/train_weighted.py --offline
# 完成后重建 E4/E5 比较；中途查看加 --allow-incomplete
python scripts/compare_weighted.py
```

E5 继承 E4，仅给 CE 增加类别权重，Dice 系数仍为 0.5。权重仅用 367 张未增强训练 mask 在 512×512 网格上的有效像素计算：`raw_c = 1/sqrt(frequency_c)`，再除以 raw 的类别均值，截断至 `[0.5,3.0]`；不使用 val/test 频率。权重、样本 ID、计数和哈希位于 `results/stage6/class_weights.json`。加权 CE 以有效像素权重之和归一化；逐轮 loss 沿用原批次有效像素加权记录方式，不直接与 E4 loss 比较。

输入/训练标签 512×512，验证/测试标签 384×384，独立预训练初始化、增强、seed=42、batch=4、50 轮及优化器不变。原 E0–E4 源码保持冻结；E5 使用单独的训练适配器，复用原 epoch、评价与 checkpoint 工具，不修改原训练全局函数。除完整训练状态外，checkpoint 同时保存类别权重，恢复时核对配置、代码、数据、权重与 E4 对照身份。

进度为 `results/E5_weighted/status.json`。完成后自动生成 `results/weighted_comparison.csv`、`weighted_per_class.csv`、`weighted_validation_per_class.csv`、`weighted_validation_selected_experiment.json` 和 [类别加权报告](reports/weighted_ablation.md)。按最高验证 mIoU 选择 E4/E5，同分保留 E4；不预设加权有效。完整协议见 [E5 协议](docs/weighted_protocol.md)。

## 10. 图片推理与效率评测

```powershell
# --input 支持单张图片或图片目录；使用空输出目录保留历史结果
python scripts/predict.py --input data/CamVid/CamVid_RGB/0016E5_07959.png --output-dir results/inference_demo_new
# 在 CUDA 上使用 FP16 autocast；模型权重仍为 FP32
python scripts/predict.py --input path/to/image.jpg --precision fp16 --output-dir results/inference_fp16
# 默认已验证的 E4；可用 --run-dir 指定其他已完成且归档一致的实验
python scripts/benchmark_inference.py --output-dir results/benchmark_E4_new
```

推理输出原图分辨率的 `index.png`（0–10 类 ID）、`segmentation.png`、`overlay.png`、原图及可本地打开的 HTML 图例预览。只加载本地完整归档且权重哈希相符的最佳模型，无额外下载；FP16 需要 CUDA。样例见本地 `results/inference_demo/index.html`，由验证图像生成，不代表整套数据的精度。

完整 101 张验证样本的 FP32/FP16 评价及 batch=1 性能见 [推理评测](reports/inference_benchmark.md)。本机此次模型均时 FP32=21.14 ms、FP16 autocast=25.25 ms；后者 allocated 峰值更低，但未获得加速，默认保留 FP32。模型测速不含读图与传输；另外单独记录含预处理、传输、原尺寸输出及叠加着色的图片流程耗时，二者不能混称端到端 FPS。评测在 E5 启动前完成，未重新评价测试集。

## 11. E4/E5 三种子配对复核

```powershell
# 核验原归档、数据和两组在新种子下完全相同的实际预训练初始化
python scripts/check_multiseed.py --offline
# 顺序执行四次新增训练；中断后加 --resume，已完成的归档会核验后跳过
python scripts/run_multiseed.py --offline
# 完成后重建均值、样本标准差和配对差值；中途查看加 --allow-incomplete
python scripts/compare_multiseed.py
```

固定种子为 `[42,2024,3407]`。seed=42 使用已有 E4/E5 归档，不重复训练或测试。新增顺序为 E4_seed2024 → E5_seed2024 → E4_seed3407 → E5_seed3407，各 50 轮；两组继续复用原训练引擎，除种子、实验 ID 和独立输出目录外不改变配置。E5 使用原训练频率权重，不按种子重新调权重；每次从固定预训练编码器和新建头独立开始。正式训练前、每个新种子下 E4/E5 全部初始参数与 buffers 的 SHA256 必须一致，原数据与所有已有源文件也须一致。

队列状态位于 `results/multiseed/queue/status.json`，各训练位于 `results/multiseed/E4_seed2024/` 等目录。每完成一组便更新 `runs.csv`、`summary.csv`、`paired_differences.csv`、`per_class_summary.csv` 和 [多种子报告](reports/multiseed.md)。汇总仅使用 E4/E5 都完整完成的相同种子，标准差为 ddof=1，只有一对时留空。全部三对结束后，按平均最佳 Val mIoU 选择方案，同分保留 E4；部署代表固定使用胜出方案已有 seed=42 模型，不按 Test 挑种子或集成。最终选择保存 `results/multiseed/selection.json`。三种子为描述性复核，不证明统计显著性或场景泛化。协议见 [多种子协议](docs/multiseed_protocol.md)。
