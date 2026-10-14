# 基于 SegFormer-B0 的轻量级语义分割模型复现、性能优化与应用实现

> **文档用途**：交接给 Codex，作为项目初始化、代码实现、实验复现、验收以及课程报告撰写的统一依据。  
> **项目性质**：《图像处理与分析》本科课程设计；以成熟模型复现、规范实验和本地演示为主，**不追求论文级算法创新**。  
> **文档版本**：v1.0（2026-10-13）  
> **状态**：实施方案已确定；**尚未执行训练，不存在可引用的真实实验成绩**。

---

## 0. 一页式决策摘要

| 项目要素 | 已确定的选择 |
|---|---|
| 课程设计题目 | **基于 SegFormer-B0 的轻量级语义分割模型复现、性能优化与应用实现** |
| 主模型 | SegFormer-B0（MiT-B0 编码器 + All-MLP 分割解码头） |
| 模型来源 | 优先采用 Hugging Face Transformers 中受维护的 SegFormer 实现；参考 NVIDIA 原始论文与仓库 |
| 初始化 | 使用预训练的 MiT-B0 编码器权重；**CamVid 11 类分割头需初始化并在 CamVid 上训练** |
| 主数据集 | CamVid 11 类语义分割，使用固定、明确、经过核验的 train/val/test 划分 |
| 硬件 | 本地 RTX 4060 Laptop，约 8GB 显存；资源紧张时优先降 batch，不更换模型 |
| 固定主对照 | E0：CE；E1：CE+数据增强；E2：CE+Dice；E3：CE+Dice+数据增强 |
| 主要指标 | mIoU、每类 IoU、混淆矩阵、训练/验证损失、推理延迟、峰值显存 |
| 部署 | **必须**完成本地 PyTorch 图片分割演示和 FP32/FP16 推理对比；ONNX 为选做 |
| 工程交付 | 完整源码、配置、环境记录、训练日志、最优权重、图表、可视化、课程报告、运行指南 |
| 不做的事 | 不新增复杂 Attention/网络结构，不切换基础骨干，不训练大型视觉模型，不强制 TensorRT/ONNX |

### 0.1 成功的定义

本项目成功**不以达到论文原始指标或 E3 必然优于 E0 为条件**。成功条件为：

1. 环境、数据与标签检查通过，可重复运行 SegFormer-B0 训练、验证和测试流程。
2. E0 基线收敛，有可信的 mIoU、每类 IoU、混淆矩阵与可视化结果。
3. E1/E2/E3 在统一协议下完成，能够得出数据增强和 Dice Loss 是否有效的实验结论。
4. 完成单图本地推理展示和 FP32/FP16 延迟、显存记录。
5. 交付可复现工程与规范的课程设计报告；对于失败或负收益实验如实分析。

---

## 1. 背景、动机与约束

### 1.1 项目缘起

本课程允许选择视觉领域的深度学习论文进行复现与优化，并不强制以传统数字图像处理算法为主要实现方式。此前讨论过 U-Net、SegFormer、YOLO 小目标检测、MobileSAM、低光照增强、PatchCore 等方向。最终为兼顾**成熟可靠、复现成本可控、内容完整、容易验收**，选择 SegFormer-B0。

用户对视觉领域没有计划投入大量科研精力；项目定位是高质量完成本科课程设计，而不是构建一个持续扩张的视觉研究项目。机器学习课程的推荐系统实验属于**另一独立项目**，此仓库无需引入推荐系统代码。

### 1.2 技术约束

- 系统：Windows + 可选 WSL2/Linux；以最终能够稳定访问 CUDA 的环境为准。
- GPU：RTX 4060 Laptop（8GB 显存）；模型和输入规模应适合本地训练/微调。
- 框架：PyTorch + Transformers；不强制使用旧版 MMSegmentation。
- 训练：可使用 AMP 混合精度；从可用的已预训练 MiT-B0 权重开始，而非全随机初始化整个编码器。
- 时间：预计 3–4 周；优先级排序为 **正确性 > 可重复性 > 完整性 > 指标 > 工程花样**。
- 数据：使用公开数据；确认来源、划分、标签编码与许可；数据和大型权重不必上传 Git 仓库。

### 1.3 必须避免的范围膨胀

- 不自行设计新 Transformer、复杂边界分支或多尺度架构。
- 不因单次结果不理想频繁更换数据集或网络。
- 不把数据增强、CE+Dice 表述为算法原创。
- 不把图片视觉效果当作唯一评价依据。
- 不把一次测试集结果用于调参；模型和超参数应根据验证集选择。

---

## 2. 模型与数据集设计

### 2.1 SegFormer-B0 简介

SegFormer 以层次化 MiT（Mix Transformer）提取多尺度视觉特征，通过轻量的 All-MLP Decoder 融合并输出像素类别预测。B0 为较小规模配置，适合 8GB 显存设备进行任务微调。

建议阅读：

1. 论文：<https://arxiv.org/abs/2105.15203>
2. 官方源码：<https://github.com/NVlabs/SegFormer>
3. Transformers 文档：<https://huggingface.co/docs/transformers/model_doc/segformer>

**实现要求**：由 Codex 核对所安装 Transformers 版本的实际模型 API 和 checkpoint 结构，再实现加载。预训练 backbone 和随机初始化的 11 类 segmentation head 必须在日志中说明；不得将 `nvidia/mit-b0` 与 ADE20K 分割任务的微调权重当作等价模型。不得凭空假定 checkpoint 加载成功；要记录 missing/unexpected keys 并核实其合理性。

### 2.2 CamVid（11 类）

选用 CamVid 的常见 11 类语义分割版本，建议沿用提供 train/val/test 清单的同一个整理来源。常见分割数量为 **367/101/233**，但必须按**实际获得的文件及标签清单核验**，不能因为与记忆中的数字相同就认为数据正确。

参考数据整理入口（第三方，需验证）：<https://github.com/lih627/CamVid>

主要数据检查项：

- 每张图像必须有对应 Mask，文件名、分辨率与划分清单对应。
- 明确 Mask 是**单通道类别索引图**还是**RGB 颜色编码标签图**；若是 RGB，需显式建立 palette→class_id 映射，不能直接把 RGB 值当类别。
- 计划使用 11 个有效类（0–10）；忽略区域常使用 255，**具体以所选数据版本为准**。
- 对于 0 是真实类别而非背景忽略区域的版本，**不得将标签 0 自动平移为 ignore**；应设置/检查 `do_reduce_labels=False`。
- Train/Val/Test 严格隔离。禁止训练时使用测试标签或基于测试集选超参数。
- 图像可双线性插值，Mask 几何变换必须最近邻插值；随机翻转时图像与 Mask 必须同步。

### 2.3 输入、输出与分割指标

- 输入建议固定为 384×384（起始配置；可根据显存调 batch size）。这个尺寸可能改变原图长宽比，需在报告中说明所采用的 resize 策略及其限制。
- 输出类别数 11，Ignore Index 以标签检查的结果为准（预期 255）。
- 对模型低分辨率 logits，先以双线性插值恢复到标签尺寸（`align_corners=False`），再进行 `argmax` 和指标计算。
- 基础评价：

\[
\mathrm{IoU}_k=\frac{TP_k}{TP_k+FP_k+FN_k},\qquad
\mathrm{mIoU}=\frac{1}{K}\sum_{k=1}^K\mathrm{IoU}_k.
\]

- 统计混淆矩阵时跳过忽略像素。对分母为零的缺失类别预先定义并记录处理方式，避免 NaN 导致跨实验不可比。
- 需要保存每类像素数和 IoU，便于分析小类别或易混淆类别。

---

## 3. 固定基线与四组实验

### 3.1 初始训练配置（建议值，不是论文官方设置）

| 配置项 | 初始值 | 说明 |
|---|---|---|
| 模型 | SegFormer-B0 | 统一模型架构 |
| 预训练 | ImageNet 预训练的 MiT-B0 | 精确 checkpoint 由运行时确认 |
| 类别数 | 11 | CamVid 11 类 |
| 输入尺寸 | 384×384 | 全部 E0–E3 保持一致 |
| Batch Size | 4 | OOM 时统一改为 2；必要时梯度累积 |
| 训练轮数 | 50 | 固定最大预算；根据 val mIoU 选择最佳权重 |
| Optimizer | AdamW | 全部组一致 |
| 学习率 | 6e-5 | 初始建议；若有调参，仅用 val 并记录 |
| Weight Decay | 0.01 | 全部组一致 |
| 主损失 | Cross Entropy | `ignore_index` 正确设置 |
| Dice 系数 | 0.5 | 初始建议；禁止看 test 后调参 |
| AMP | FP16 autocast | 保持训练对比一致 |
| Seed | 42 | 初轮固定；可扩展多个 seed |
| 选模标准 | 最高 Val mIoU | 保存 best 与 last checkpoints |

备注：模型 `num_labels`、`id2label/label2id`、数据预处理均需与实际模型 API 适配。训练集 loss、验证集 loss、mIoU 的计算口径需要统一并写入 README。

### 3.2 实验矩阵（必须完成）

| 编号 | 训练数据增强 | 损失函数 | 问题 |
|---|---|---|---|
| **E0** | 基础 Resize/Normalize | CE | 基线能否正确收敛？ |
| **E1** | E0 + 随机水平翻转 + 轻量 ColorJitter | CE | 轻量增强是否提升泛化？ |
| **E2** | 基础 Resize/Normalize | CE + Dice | 区域重叠监督是否带来收益？ |
| **E3** | 与 E1 相同 | CE + Dice | 两项改进有无互补效果？ |

- 四组实验必须从**相同预训练初始化**独立训练，不得将 E1/E2/E3 直接从 E0 已微调完的 checkpoint 接着训练后声称为平行消融。
- `E1`/`E3` 的随机增强只作用于训练集；验证与测试必须确定性处理。
- CE+Dice 要对有效像素做 one-hot/softmax 相关处理，正确屏蔽 ignore 像素；说明针对类别缺失时的数值稳定处理。
- 固定划分、训练预算、评价脚本、种子及其他参数，避免变量混杂。
- 如果 E3 表现不如 E0，也保留真实结果，分析可能原因。

### 3.3 建议补充（时间允许）

- 对 E0 和由验证集选定的最佳改进方案，再使用 2–3 个随机种子，报告均值和标准差。
- 在统一硬件、batch=1、固定输入分辨率下测试 PyTorch FP32 和 AMP FP16：warmup、CUDA synchronize、重复多次并记录均值/P50/P95。
- ONNX Runtime 导出和对比属于 **P2 选做**，不应阻塞课程报告交付。

---

## 4. 分阶段实施计划（按验收门槛推进）

以下阶段不得无条件自动跳过。**每一阶段交付物与验收标准满足后，再进入下一阶段。**

### Stage 0：仓库初始化和实施方案冻结（约 0.5–1 天）

**Codex 任务**

1. 建立清晰的仓库目录、依赖安装说明、Git ignore、配置系统和基本命令行入口。
2. 检查 CUDA/PyTorch/Transformers 兼容性；真实记录显卡、驱动、Python、PyTorch、CUDA、Transformers 版本。
3. 固定随机种子和设备选择逻辑；确保 CPU 环境至少能进行模型加载与小规模逻辑测试。
4. 写入 `docs/experiment_protocol.md`：冻结训练/验证/测试流程及 E0–E3 的唯一差异。

**交付**：`README.md`、`requirements.txt` 或 `pyproject.toml`、`configs/*.yaml`、`docs/experiment_protocol.md`、`environment.json`。

**验收**：环境自检命令能返回必要版本信息；配置文件可被正确读取；未引入不必要的旧框架。

### Stage 1：数据核验和冒烟测试（约 1–2 天）

**Codex 任务**

1. 编写数据目录检查脚本；核验划分数量、图像/Mask 匹配、类别 ID 分布、ignore 比例、非法标签。
2. 编写 Dataset 和图像/Mask 成对变换；随机展示若干样本的 image/mask/overlay。
3. 正确实例化 SegFormer-B0 11 类模型，并写下加载权重情况。
4. 单批次前向、计算 CE、反向传播、optimizer.step，确认 loss 有限且梯度正常。
5. 在 20–50 张训练样本上尝试过拟合，以排查损失、标签映射和变换问题。

**交付**：`data_audit.json`、示例标签可视化、`smoke_test.log`、`overfit_check.log`。

**验收**：数据检查无未解释错误，输出通道=11，loss 不为 NaN/Inf，梯度可更新，Mask 颜色/类别映射正确。小样本过拟合测试应显示学习能力；若未出现，先排查问题再进行完整基线。

### Stage 2：E0 基线复现（约 3–5 天）

**Codex 任务**

1. 基于冻结配置完整运行 E0 训练（上限 50 epoch），保存 best 和 last。
2. 逐 epoch 记录 train_loss、val_loss、val_mIoU、learning_rate、epoch_time、peak_allocated_vram_mb。
3. 保存训练配置副本、权重、环境信息、运行命令、Git commit hash（可获取时）。
4. 生成训练/验证 loss 曲线、验证集 mIoU 曲线。
5. 用最佳验证权重完成一次受控测试集评价，输出详细指标和代表性可视化。

**交付**：`results/E0_baseline/` 完整归档。

**验收**：模型训练和评估可重复，日志齐全；报告基线性能和错误类型，不假定达到任何未实测阈值。

### Stage 3：基线错误分析（约 1 天）

**Codex 任务**

1. 生成每类 IoU 排名与混淆矩阵。
2. 导出具有代表性的正确案例和失败案例：原图、GT、预测 Mask、半透明 Overlay。
3. 简述边界、小类别、类间混淆等潜在问题，禁止在未分析前直接断言根因。

**交付**：`reports/baseline_analysis.md`、per-class 表格、混淆矩阵和预测案例。

**验收**：至少能够解释哪些类别表现较差、错误主要体现在哪些位置，以及这些结论对应的可观测证据。

### Stage 4：E1–E3 简单优化实验（约 3–6 天）

**Codex 任务**

1. 实现训练增强开关，并确保验证/测试无随机增强。
2. 实现可选 CE+Dice 损失，添加 ignore 和数值稳定性单测。
3. 为 E1、E2、E3 配置独立运行目录和相同随机初始化逻辑。
4. 使用统一训练与评估协议运行三组实验。
5. 合并四组结果，输出 mIoU、每类 IoU、运行开销以及与 E0 的变化量。

**交付**：`results/E1_augmentation/`、`E2_dice/`、`E3_combined/`、`results/comparison.csv`、`reports/ablation.md`。

**验收**：四组结果具有可比性；变化量可由原始 `metrics.json` 重新计算；实验结果为真实数据，不事先编造“提升 X%”。

### Stage 5：本地推理、速度测试及演示（约 1–2 天）

**Codex 任务**

1. 编写加载指定权重的单图推理脚本，返回 index mask、color mask、overlay。
2. 同一 checkpoint 下比较 FP32 与 FP16 autocast 推理；统一 batch=1 和图像尺寸。
3. 计时前进行预热并同步 CUDA；记录延迟分布及峰值显存，区分模型推理时间与含预处理、后处理的端到端时间。
4. 用 Gradio（可选 Streamlit）实现简洁页面：图片上传、权重选择、分割预测、叠加图、延迟显示。
5. 将 ONNX 导出列为非阻塞性选做项。

**交付**：`demo/app.py`、`scripts/benchmark_inference.py`、`results/inference_benchmark.json`、演示截图或录屏、使用说明。

**验收**：使用本地权重即可离线完成预测；显示的 Mask 与输入图像空间对应；性能指标可重复测得。

### Stage 6：归档与课程报告（约 2–3 天）

**Codex 任务**

1. 自动生成报告需要的图片/表格，核对所有数字来自实验文件。
2. 形成 `reports/final_report.md` 初稿和 `reports/results_summary.md`。
3. 检查可复现性：全新工作目录是否能依据 README 安装依赖、放置数据、加载权重并执行评估与演示。
4. 清理临时大文件；保留必要日志、配置、模型路径、数据许可与权重获取说明。
5. 输出一份 `DELIVERY_CHECKLIST.md`，逐项说明已完成、选做未完成、已知问题。

**交付**：最终报告草稿、提交包、明确的复现实验说明。

**验收**：报告中的每个指标均可追溯到日志或结果文件；无把示意图/预估值混作真实结果的情况。

---

## 5. 推荐仓库结构

```text
segformer-course-project/
├── README.md
├── requirements.txt                 # 实际验证后锁定版本
├── .gitignore
├── configs/
│   ├── baseline.yaml               # E0
│   ├── augmentation.yaml           # E1
│   ├── dice_loss.yaml              # E2
│   └── combined.yaml               # E3
├── data/
│   └── README.md                   # 数据获取方式和目录规范；不提交大数据
├── docs/
│   └── experiment_protocol.md
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── losses.py
│   ├── inference.py
│   └── utils.py
├── scripts/
│   ├── check_environment.py
│   ├── audit_dataset.py
│   ├── smoke_test.py
│   ├── check_overfit.py
│   ├── run_experiments.py
│   ├── make_plots.py
│   └── benchmark_inference.py
├── tests/
│   ├── test_dataset_labels.py
│   ├── test_augment_sync.py
│   ├── test_metrics_ignore.py
│   └── test_dice_loss.py
├── results/
│   ├── E0_baseline/
│   ├── E1_augmentation/
│   ├── E2_dice/
│   ├── E3_combined/
│   ├── comparison.csv
│   └── inference_benchmark.json
├── demo/
│   └── app.py
├── reports/
│   ├── baseline_analysis.md
│   ├── ablation.md
│   ├── results_summary.md
│   └── final_report.md
└── DELIVERY_CHECKLIST.md
```

**设计原则**：目录是目标结构，不要求第一次提交就放入大量空文件。先保证 Stage 0–1 运行正常，再按需增长。

---

## 6. 实验产物与日志规范

建议每一实验目录具有以下基本文件：

```text
results/E0_baseline/
├── config.yaml                  # 当次执行的冻结配置副本
├── environment.json             # 运行版本、设备、CUDA信息
├── run_info.json                # seed、时间、命令、Git commit、checkpoint来源
├── train_log.csv                # epoch级历史指标
├── best_model/                  # 最优权重与模型配置
├── last_checkpoint/             # 训练最后状态；可选 optimizer/scheduler 便于恢复
├── metrics.json                 # 测试结果，另存最优val指标
├── per_class_iou.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── training_curves.png
└── predictions/
    ├── sample_001_input.png
    ├── sample_001_gt.png
    ├── sample_001_pred.png
    └── sample_001_overlay.png
```

### 6.1 推荐 `train_log.csv` 字段

```csv
epoch,train_loss,val_loss,val_miou,learning_rate,epoch_time_sec,peak_vram_mb
```

### 6.2 推荐 `metrics.json` 字段

```json
{
  "experiment_id": "E0_baseline",
  "dataset": "CamVid_11",
  "split": "test",
  "seed": 42,
  "best_checkpoint_selected_by": "val_miou",
  "best_val_epoch": null,
  "best_val_miou": null,
  "test_miou": null,
  "pixel_accuracy": null,
  "per_class_iou": {},
  "ignore_index": 255,
  "notes": "All null values must be replaced by measured values after evaluation."
}
```

**说明**：JSON 为字段模板，null 不是成绩。其他类别名、耗时等字段应按实际记录补充。若配置和指标存在不同定义，必须通过 `metric_definition` 明确说明。

### 6.3 优先保留的证明材料

- 数据清单及类别统计（训练/验证/测试）。
- 安装版本、预训练权重来源、模型初始化输出。
- 每次训练调用命令及配置快照。
- 逐轮 CSV 日志、best checkpoint 的选取依据。
- 统一评估代码及真实结果 JSON。
- 效果图（包括失败案例）、混淆矩阵和曲线。
- 推理 benchmark 运行条件与原始计时统计。

---

## 7. 核心正确性与评估风险（Codex 必须检查）

| 风险 | 预防措施 |
|---|---|
| 标签类别映射错误 | 数据审计、唯一值统计、RGB→ID palette 显式映射、Mask overlay 人工核查 |
| 错误减少标签 ID | 11 个有效类中 0 若有效，禁用 `reduce_labels` |
| 图片与 Mask 增强不同步 | 共享同一翻转/几何随机参数；Mask 最近邻插值 |
| 模型预训练权重加载失败却未发现 | 输出 checkpoint 来源、missing/unexpected keys、分割头初始化情况 |
| Logits 与 Mask 尺寸不同 | 在 loss/metric 之前检查并正确 resize logits；统一 `align_corners` 策略 |
| Ignore 像素污染指标 | CE、Dice、confusion matrix、mIoU 均显式屏蔽 ignore |
| 使用 test 调参 | 仅 val 选模、选超参；test 用于最终结果呈现 |
| 消融起点不公平 | 四组从相同预训练权重独立初始化，使用一致训练预算 |
| OOM、数值不稳定 | 降 batch、AMP 合理使用、检查 NaN/Inf，记录实际配置变化 |
| 小数据集指标波动 | 固定 seed，可能时重复种子；如实报告统计不确定性 |
| 不科学的速度比较 | 同设备、同分辨率、同批量、同权重、预热、同步、记录环境 |
| 虚构实验结论 | 代码输出原始结果，不允许填充假指标；先有日志再写结论 |

---

## 8. 课程报告提纲（可直接成为 Markdown 初稿）

1. **绪论**：语义分割的意义、经典 CNN 到 Transformer 的演进、课设目标。
2. **技术基础**：图像语义分割、Attention 基本机制、MiT、评价指标（mIoU 等）。
3. **SegFormer-B0 方法**：模型结构、MiT 分层特征、All-MLP Decoder、模型训练机制。
4. **数据与复现方案**：CamVid 划分、标签、数据预处理、训练环境、基线配置。
5. **两项简单训练优化**：水平翻转/ColorJitter 数据增强、CE+Dice 损失及其动机。
6. **实验设计与结果分析**：E0–E3 指标、消融对比、类别 IoU、混淆矩阵、可视化与失败案例。
7. **本地部署和效率测试**：单图推理流程、FP32/FP16 对比、端到端演示。
8. **总结与局限**：哪些优化有效/无效、限制条件、后续工作（可简单提及 ONNX）。

报告中必须清楚区分“原论文贡献”“本课设实现工作”“本课设的训练策略对照”。**不得声称数据增强或 Dice Loss 是新的算法发明。**

---

## 9. 四周推进建议与优先级

| 时间 | 工作中心 | 阶段成果 |
|---|---|---|
| 第 1 周 | Stage 0–1：环境、数据、冒烟、过拟合检查 | 安全可靠的数据及训练底座 |
| 第 2 周 | Stage 2–3：E0 完整训练与基线分析 | 权重、曲线、mIoU、错误分析 |
| 第 3 周 | Stage 4：E1/E2/E3 与比较 | 四组实验结果和消融结论 |
| 第 4 周 | Stage 5–6：推理演示、效率对比、报告与归档 | 完整可交付课程项目 |

### 时间不够时的降级策略

- **P0（不可省）**：Stage 0、Stage 1、E0、数据与标签正确性、mIoU 与可视化。
- **P1（应完成）**：E1/E2/E3、结果对比、FP32/FP16 基础 benchmark、单图演示、课程报告。
- **P2（选做）**：多随机种子统计、ONNX Runtime、视频流分割、漂亮的 Web 界面。

若出现环境/硬件/数据问题，应先解决正确性阻塞，**不要为了凑满实验表而运行不可信实验**。

---

## 10. 给 Codex 的执行指令（可直接复制）

> 请阅读此 Markdown 全文，以其为当前《图像处理与分析》课程设计的**冻结实施方案**。我们已确定使用 SegFormer-B0 + CamVid 11 类开展复现，完成 E0/E1/E2/E3 四组训练策略实验，并进行本地单图分割与 FP32/FP16 推理对比。不需要设计复杂创新模块，也不需要升级到更大的视觉模型。
>
> **请先实施 Stage 0 和 Stage 1，不要直接启动 50 epoch 完整训练。** 先检查当前机器的 Python/PyTorch/CUDA/Transformers 环境、数据集是否已存在、训练/验证/测试目录与标签编码、预训练权重是否可以正确加载；建立仓库、配置、数据审计、基本测试与单批次冒烟脚本。
>
> 如果缺少外部数据或权重、网络不可达或存在版本不兼容，请先提供可重现的错误信息与最小解决步骤，不要静默替换数据、权重或模型架构。若需要作未在方案中指定的工程选择，选择最简单可靠的实现，并写入 README 与配置。
>
> 完成每个 Stage 后，请提交：①本阶段修改文件清单；②执行命令；③自动测试/验收结果；④日志及产物路径；⑤发现的问题及尚未完成事项。**Stage 验收通过后再推进下一阶段**，不应编造指标或提前撰写实验结论。
>
> 代码尽可能遵循模块化设计，配置驱动四组实验，避免重复实现。数据不要直接提交到 Git；日志和重要配置必须可追溯。等 E0 基线稳定后，再开展 E1/E2/E3。

---

## 11. 最终交付验收表

- [ ] 代码能够安装、运行，并有清晰 README 和完整依赖版本。
- [ ] CamVid 图像与 Mask 检查通过，类别 ID/ignore 配置正确。
- [ ] SegFormer-B0 11 类模型能加载、训练、验证、测试。
- [ ] 小样本过拟合检查通过，E0 产生完整日志与最佳权重。
- [ ] E0–E3 有各自配置、日志、评估指标与可视化结果。
- [ ] mIoU/各类别 IoU/混淆矩阵与图像可视化均可追溯。
- [ ] 推理 benchmark 保留 FP32/FP16 的测量条件及结果。
- [ ] 本地演示能够选择权重并对上传图片输出分割结果。
- [ ] 课程报告覆盖原理、方法、实验、分析与局限。
- [ ] 归档中没有虚构结果，且可区分必须完成和选做内容。

---

## 12. 参考资料

1. Xie et al. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** (NeurIPS 2021): <https://arxiv.org/abs/2105.15203>
2. NVIDIA SegFormer 参考实现：<https://github.com/NVlabs/SegFormer>
3. Hugging Face Transformers SegFormer 文档：<https://huggingface.co/docs/transformers/model_doc/segformer>
4. CamVid 数据整理（第三方；需核验）：<https://github.com/lih627/CamVid>
5. SegFormer MiT-B0 权重参考：<https://huggingface.co/nvidia/mit-b0>
6. PyTorch AMP 文档：<https://docs.pytorch.org/docs/stable/amp.html>

**文档结论**：本项目以规范复现与可信实验为第一目标。先做 E0，分析结果，再做 E1–E3，最后交付演示与报告。不要扩张到复杂视觉研究项目。
