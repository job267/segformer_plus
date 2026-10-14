# 冻结的实验协议（Stage 0）

版本：2026-10-13。此文件实现根目录项目方案中的基础实验约定。后续改变须在运行前记录，测试集结果不能驱动调参。

## 阶段门槛

1. Stage 0：真实环境自检通过，E0–E3 配置可读，核心测试通过。
2. Stage 1：三个划分的完整数据审计通过；固定预训练 MiT-B0 编码器完整加载；真实批次前向、反向和更新通过；20–50 张训练样本显示学习能力。
3. Stage 2：仅在上述条件满足后实现并运行 E0 正式训练。后续按原方案推进错误分析、E1–E3、推理与报告。

合成数据或全随机初始化的冒烟输出只证明逻辑可运行，不能代替 Stage 1。下载失败或不存在真实数据时，保留错误与解决命令，不补造验收结果。

## 数据与预处理

采用 `lih627/CamVid` 固定版本 `79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6` 的三个清单与 `CamVidGray`。Sky=0，不减标签；有效类 0–10，ignore=255。逐文件核对配对、尺寸和编码；保留清单及图像/Mask SHA256。每个 split 的类别像素数和 ignore 比例均以原始分辨率统计。

全组输入拉伸到 384×384，可能改变原图长宽比，应在报告说明这一限制。图像双线性缩放，Mask 最近邻；像素除以 255 后按 ImageNet mean/std 标准化。训练翻转图像和标签共享同一个随机决定。ColorJitter 只作用图像，brightness/contrast/saturation 幅度各 0.2；三种操作随机排序。验证、测试只 Resize/Normalize。

## 初始化与公平性

使用 `nvidia/mit-b0` 固定版本 `80983a413c30d36a39c20203974ae7807835e2b4`，明确记录它是 ImageNet 编码器。加载 `SegformerModel` 后核验 B0 配置、missing/unexpected/mismatched keys。仅允许丢弃 ImageNet `classifier.weight` 和 `classifier.bias`；新建 All-MLP 解码头与 11 类分类层，然后严格复制编码器权重。

每组独立设置相同 seed=42，再从同一个预训练编码器构造相同初始分割头；不从另一组微调权重接续训练。保存来源版本和实际载入日志。

| 实验 | 增强 | 损失 |
|---|---|---|
| E0 | 关闭 | CE |
| E1 | 同步水平翻转 + 轻量 ColorJitter | CE |
| E2 | 关闭 | CE + 0.5 Dice |
| E3 | 与 E1 相同 | CE + 0.5 Dice |

其余配置相同：batch=4，最多 50 epoch，AdamW，学习率 6e-5，weight decay=0.01，全部模型参数参与更新，默认固定学习率，不引入额外 scheduler；CUDA 上使用 FP16 autocast 与 GradScaler，CPU 使用 FP32。显存不足时统一降低各组 batch 并记录变更。

## 损失与评价口径

低分辨率 logits 在 loss 和 argmax 前双线性插值至标签尺寸，`align_corners=False`。损失在 FP32 计算。CE 先计算逐像素损失，再对有效像素求均值，避免当前 PyTorch CUDA NLL 空间汇总实现的确定性警告；Dice 在 batch 和空间维度累计 softmax 与 one-hot 重叠，屏蔽 ignore，并仅对该批 GT 出现的类别取平均。GT 缺失类仍受 CE 约束，Dice 的 epsilon=1e-6。全 ignore 批次返回可反传的零损失，训练循环跳过其更新；整套数据没有有效像素则报错。

混淆矩阵行=GT、列=预测，在整个划分累计后计算指标，不平均逐张 mIoU。忽略像素不进入矩阵。每类 IoU=TP/(TP+FP+FN)。当分母为 0 时该类 IoU 写为 JSON null，并从此次 mIoU 排除；如果仅存在该类的误预测，分母非零，IoU=0，仍计入 mIoU。保存参与平均的类别数，防止不同实验暗中改变口径。没有有效像素时 mIoU/accuracy 为 null。

epoch loss 为各 batch loss 按有效像素数加权的均值。CE 由此等于划分级像素均值；组合损失的 Dice 部分保留批次口径，随 batch 的实际组成变化，应维持各组相同 batch 和预算。验证以确定性 FP32 推理计算 loss/mIoU。依据最高 val mIoU 选 best，保存 last；最终 test 只在固定最佳权重后执行，不参与选模。

## 可重复性与实际限制

固定 Python、NumPy、PyTorch 和 CUDA seed，关闭 cuDNN benchmark，启用 deterministic algorithms（warn_only）。当前 PyTorch 的 CUDA 双线性插值反向传播可能没有确定性实现，因此同硬件同配置也不承诺逐位完全一致；原始警告和环境应保留。DataLoader 默认 num_workers=0；多进程时固定 worker seed 和独立 Generator。CPU 冒烟限制线程数以控制开销。

## 小样本诊断

`check_overfit.py` 从 train 清单用 seed=42 固定抽取 20–50 张，关闭增强，在相同子集上作确定性评估，绝不读取 val/test 标签。默认运行 20 轮，诊断学习率 1e-3，与正式实验配置分开记录。默认通过条件为子集评估 loss 相对初始下降至少 20%，且子集 mIoU 上升；这只是预先声明的学习能力检查，不是要求正式实验达到某个成绩。失败时先排查，不自动进入 E0。

## 归档与后续工作

基础检查输出写入 `results/stage0/`、`results/stage1/`。正式实验将分别使用 E0_baseline、E1_augmentation、E2_dice、E3_combined 目录。Stage 0–1 不生成 test_miou 或虚构比较表，不提前写实验结论。训练日志、最佳权重、统一评估、推理和效率测试、报告在后续阶段实现并验收。

### 2026-10-14：Stage 2 执行入口

`scripts/train.py` 已实现单组正式训练及归档，冻结的模型、数据、损失和训练超参数不变。最后完整轮次的 checkpoint 为恢复依据，CSV 与最佳模型在续训时从该状态修复；同一目录使用操作系统文件锁防止重复训练。每次运行保留配置、环境、数据审计和代码哈希；配置、代码或审计内容改变后不允许沿用原训练状态。测试在全部训练和验证选模完成后执行，已完成的同一最佳权重测试结果不重复评价。
