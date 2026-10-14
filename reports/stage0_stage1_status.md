# Stage 0–1 基础构建验收记录

日期：2026-10-13。**Stage 0 与 Stage 1 均已通过**。本记录覆盖项目基础构建，不是正式 E0 基线或最终实验报告。

## 1. 实际验收结果

| 检查 | 实际结果 | 原始记录 |
|---|---|---|
| 环境与四组配置 | 通过；Python 3.12.3、PyTorch 2.3.1、CUDA 11.8、Transformers 4.44.1 | `environment.json` |
| GPU 与驱动 | RTX 4060 Laptop，约 8GB；驱动 576.02；CUDA 计算探针通过 | `environment.json` |
| 核心测试 | **22 passed**；覆盖标签、路径与划分、增强同步、ignore、手算指标、CE/Dice、配置差异 | `results/stage0/pytest.xml` |
| 数据审计 | **367 / 101 / 233** 张分别属于 train / val / test；全部实际读取核验；错误数 0 | `results/stage1/data_audit.json` |
| 模型载入 | MiT-B0 编码器完整载入，11 类分割头新建；模型参数总数 3,716,971 | `results/stage1/model_loading.json` |
| CPU 逻辑检查 | 预训练编码器、合成输入 2×3×64×64；损失有限，梯度和更新通过 | `results/stage1/cpu_logic_smoke/smoke_test.json` |
| CUDA CE+Dice 逻辑检查 | 合成输入 batch=4、384×384、AMP；前向、反向和更新通过 | `results/stage1/cuda_logic_smoke/smoke_test.json` |
| 真实数据冒烟 | train 前 4 张、384×384、AMP；CE=2.403231859，梯度范数 4.958008289，参数成功更新 | `results/stage1/real_smoke/smoke_test.json` |
| 小样本学习诊断 | 20 张固定随机训练样本、20 轮、诊断学习率 1e-3；预先声明的学习判定通过 | `results/stage1/overfit/overfit_check.json` |

数据全部为 RGB 图像 / L 模式索引标签，尺寸 960×720；类别 0–10 和 ignore=255 已核验，Sky=0 没有被忽略或平移。原始分辨率 ignore 比例分别约为 train 3.2604%、val 0.8549%、test 3.5423%。清单及全部图像/Mask 的 SHA256 已保留，未发现清单、路径或图像文件内容的跨划分复用。

预训练来源固定为 `nvidia/mit-b0@80983a413c30d36a39c20203974ae7807835e2b4`。加载时 missing/mismatched/error 均为空；仅有可解释的 ImageNet `classifier.weight` 和 `classifier.bias` 被丢弃；复制到分割模型编码器的 missing/unexpected 均为空。数据来源固定为 `lih627/CamVid@79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6`。

真实批次 logits 为 4×11×96×96，在计算损失前恢复为 4×11×384×384。此单步检查测得峰值 allocated VRAM 为 759.45 MiB；它不是推理效率测试或完整训练的显存结论。

## 2. 小样本诊断的范围

固定 seed=42，从 train 清单抽取 20 张，关闭增强，独立从预训练编码器初始化。在同一训练子集上作确定性评估：

| 诊断量 | 初始化 | 第 20 轮 |
|---|---:|---:|
| 子集评估 CE | 2.398108368 | 0.242806817 |
| 子集 mIoU | 0.025663023 | 0.572667158 |

loss 相对下降 **89.8751%**，mIoU 上升，满足事先声明的“下降至少 20% 且 mIoU 上升”的学习能力检查。小类别仍未充分学会，例如该子集末轮 Bicyclist IoU=0；这项诊断只证明训练流程有学习能力，不证明最终泛化或所有类别已充分拟合。没有把这些数值当作 val/test 成绩，也没有用诊断权重接续正式消融。

样本 ID、初始/末轮每类指标和混淆矩阵在 `overfit_check.json`，逐轮数据在 `overfit_log.csv`。`config.yaml` 保存可重新读取的实际诊断配置，`source_config.yaml` 保存原始基线配置。配置重载已验证，原始 E0–E3 设置没有被修改为诊断学习率。

已人工查看审计样本及三个固定随机训练样本的叠加图：建筑、道路、天空、车辆、行人和骑行者区域与原图空间对应。可视化分别位于 `results/stage1/previews/` 和 `results/stage1/overfit/previews/`。

## 3. 本次创建的工程文件

- 根目录：`.gitignore`、`README.md`、`requirements.txt`、`requirements-dev.txt`、`pyproject.toml`、`environment.json`、`DELIVERY_CHECKLIST.md`；初始化本地 Git 的 main 分支。
- 配置：`configs/baseline.yaml`、`augmentation.yaml`、`dice_loss.yaml`、`combined.yaml`。
- 文档：`data/README.md`、`docs/experiment_protocol.md`、本验收记录。
- 基础代码：`src/__init__.py`、`config.py`、`utils.py`、`labels.py`、`transforms.py`、`dataset.py`、`audit.py`、`model.py`、`losses.py`、`metrics.py`、`train.py`。
- 命令行：`scripts/_bootstrap.py`、`check_environment.py`、`download_camvid.py`、`audit_dataset.py`、`check_model.py`、`smoke_test.py`、`check_overfit.py`。
- 测试：`tests/conftest.py`、`test_config.py`、`test_dataset_labels.py`、`test_augment_sync.py`、`test_metrics_ignore.py`、`test_ce_loss.py`、`test_dice_loss.py`。
- 检查输出：`results/stage0/pytest.xml` 与 `results/stage1/` 下的实际 JSON、CSV、配置副本、日志和标签可视化。数据与模型缓存被 Git 忽略，完整数据保留在 `data/CamVid/`，预训练缓存保留在 `.cache/huggingface/`。

## 4. 已执行的主要命令

以下命令在项目根目录执行，最终均通过。下载曾中断，重试后完整成功。

```powershell
git init --initial-branch=main
python scripts/check_environment.py
python scripts/download_camvid.py --workers 4
python scripts/audit_dataset.py
python scripts/check_model.py
python -m pytest -q --basetemp=.cache/pytest --junitxml=results/stage0/pytest.xml
python scripts/smoke_test.py --synthetic --offline --device cpu --size 64 --output-dir results/stage1/cpu_logic_smoke
python scripts/smoke_test.py --config configs/combined.yaml --synthetic --offline --batch-size 4 --output-dir results/stage1/cuda_logic_smoke
python scripts/smoke_test.py --offline --batch-size 4
python scripts/check_overfit.py --offline --samples 20 --epochs 20
```

## 5. 处理过的问题与下一阶段

- 下载中发生 SSL 连接中断。增加持久连接、重新连接、间隔重试及已有文件续传，以 4 个连接完成全部文件；未关闭 SSL 校验或更换数据源。
- 初次并发创建目录时出现合法路径校验误报。下载目标现在线程启动前统一解析并验证，后续完整下载通过；路径越界拒绝仍有测试覆盖。
- PyTorch 2.3 的 CUDA NLL 空间汇总发出确定性警告。改为逐像素 CE 加显式有效像素归约，手算一致性测试通过，重新运行 CUDA 检查未再出现该警告。仍采用固定种子和 warn_only 的确定性策略，不承诺跨硬件逐位一致。
- Windows 缺少符号链接权限时 Hugging Face 缓存使用普通文件；加载已通过，无需修改系统设置或管理员权限。

当前没有未解决的 Stage 0–1 阻塞。下一步为 Stage 2：实现正式 E0 运行与归档入口，再按冻结协议训练、以 val mIoU 选模并进行受控 test 评价。E0–E3 完整训练、每类错误分析、推理演示、FP32/FP16 基准及课程报告尚未开展；本次不提供这些阶段的成绩或结论。
