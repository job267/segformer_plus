# 课程交付与复现指南

完成日期：2026-10-19。正式实验已完成，不需要为查看报告再次训练。最终方案按三种子平均最佳验证 mIoU 选为 E5；部署代表固定为原 seed=42 模型。

## 1. 交付内容与边界

课程报告为 `reports/final_report.md`，离线阅读版为同目录 `final_report.html`；简表见 `results_summary.md`。HTML 与 figures 文件夹应保持相对位置。报告生成器核对所有 10 次运行的配置、50 轮历史、验证选模、测试混淆矩阵和实际最佳权重身份，然后生成表格，不重新执行测试推理。

GitHub main 保存全部源代码、配置、数值归档、曲线、报告、许可和测试。大型权重、完整数据和照片样例被 Git 忽略。本地 `delivery/artifacts/` 下 ZIP 包另行包含 10 次运行的 Hugging Face 最佳权重、两组固定测试图的报告插图、一张验证示例和逐文件 SHA256 清单；不包含完整 CamVid、预训练缓存或最后训练状态。它支持离线推理及结果核验，不是完整数据备份，也不能独立恢复被排除的训练中间状态。请一并阅读根目录 THIRD_PARTY_NOTICES.md。

## 2. 使用提交包离线演示

将 ZIP 完整解压到一个新目录，在该目录运行。使用 Python 3.12 与已固定的 requirements.txt；开发验证另装 requirements-dev.txt。本次新目录验证使用现有的 Python 3.12.3 环境，没有声称在新系统中重新安装过依赖。

```powershell
# 新环境：先安装适合设备的固定 PyTorch，再安装其余依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# 以下 python 可替换为上述虚拟环境解释器；CPU 安装来源见 README
python delivery/verify.py --output ../bundle_verification.json
python demo/app.py --device auto --port 8765
```

打开 `http://127.0.0.1:8765`，上传图片或点击“载入示例”，选择 E5/E4/E0，点击“开始分割”。默认 E5、FP32；输出原尺寸类别索引、彩色图、叠加图，可下载 PNG。FP16 需要 CUDA。界面耗时含预处理、推理及原尺寸着色叠加，不含加载权重、PNG 编码和浏览器传输。上传图片只在内存处理。

`verify.py` 先核验清单中的每个文件，再强制 Hugging Face 离线、指定独立空缓存，重新核验 10 次归档和配对统计，最后用 CPU 对随包验证示例预测一次。它不需要完整数据集，也不执行 test 数据集推理。输出应为 status=passed、verified_runs=10、selected_family=E5，预测大小与原图相同、类别为 0–10。清单核验用于发现字节变化；清单自身没有外部数字签名，ZIP SHA256 另记于 `delivery/release.json`。

命令行推理也可使用：

```powershell
python scripts/predict.py --run-dir results/E5_weighted --input demo/example.png --device cpu --output-dir results/my_prediction
```

输出目录需为空。Git 克隆不包含权重或示例照片，仅克隆代码不能直接运行已训练模型；请使用本地提交包，或从零复现训练。

## 3. 从零训练：创建没有历史结果的新工作目录

项目拒绝覆盖已完成实验。不要在包含交付结果的目录中直接重跑训练，也不要改写历史审计和训练记录。打包工具可复制源码与协议，创建独立、没有 results/data 权重缓存的新工作目录：

```powershell
# 在原项目或已解压的提交包中运行；目标目录必须不存在
python delivery/package.py --training-workspace ../segformer_retrain
cd ../segformer_retrain
python -m pytest -q --basetemp=.cache/pytest
```

新目录不复制历史实验结果、照片示例或模型权重。先按 README 安装依赖，再按固定来源获取数据与初始化模型，并在新根目录重新生成审计：

```powershell
python scripts/check_environment.py
python scripts/download_camvid.py --workers 4
python scripts/audit_dataset.py
python scripts/check_model.py
python scripts/smoke_test.py --offline --batch-size 4
python scripts/check_overfit.py --offline --samples 20 --epochs 20
```

下载器固定 CamVid 整理版本 `79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6`，数据目录为 data/CamVid，实际划分为 367/101/233。`check_model.py` 首次联网下载固定 nvidia/mit-b0 revision `80983a413c30d36a39c20203974ae7807835e2b4` 到项目缓存；后续 --offline 使用该缓存。也可按 data/README.md 手工取得同一版本，仍需完整审计。

历史 data_audit.json 中的绝对根路径属于原实验身份，移动后不能充当新训练的数据验收。报告和离线推理核验会保留它作为历史记录；新训练必须在新目录重做验收。所有数据与模型须遵守第三方来源要求。

## 4. 按冻结顺序复现完整实验

完成真实数据与模型验收后，依次运行，上一项通过后才执行下一项：

```powershell
python scripts/train.py --config configs/baseline.yaml --offline
python scripts/run_experiments.py --offline
python scripts/compare_experiments.py
python scripts/smoke_test.py --config configs/resolution.yaml --offline --batch-size 4 --output-dir results/stage5/E4_smoke
python scripts/check_resolution.py --offline
python scripts/train_resolution.py --offline
python scripts/compare_resolution.py
python scripts/check_weighted.py --offline
python scripts/train_weighted.py --offline
python scripts/compare_weighted.py
python scripts/check_multiseed.py --offline
python scripts/run_multiseed.py --offline
python scripts/compare_multiseed.py
```

这是 6 次 seed=42 和 4 次新增种子运行，每次 50 轮；多种子会复用已有 seed=42，不能理解为每次复现只训练一轮。中断后在同一目录使用对应训练命令加 --resume，核验配置、源代码及数据不变后从最后完整 epoch 恢复。已完成的同一最佳权重测试记录会复用，避免重新评价来挑选结果。每次测试只在该次训练结束、验证选模以后执行。

评价格式：0–10 为有效类，255 忽略；无 reduce_labels。E0–E3 输入/训练标签384×384，E4/E5输入/训练标签512×512；所有正式 Val/Test 标签均从原始标注直接缩放至384×384，再在该网格上累计混淆矩阵。E5 的类别权重只统计未增强训练标签。最终按固定三个种子的平均最佳 Val 选择方案，代表 seed=42，不按 Test 挑种子。

训练会记录 src/*.py、scripts/*.py、configs/*.yaml 与 requirements.txt 的字节哈希；严格恢复要求这些文件与原训练相同。Git 的文本换行规范化可能改变既有 Windows 字节哈希，因此本地 ZIP 保留训练现场字节；新克隆应创建全新运行，不把它当作旧训练的严格恢复目录。新增 demo/ 与 delivery/ 不改变冻结训练源码清单。CUDA 非确定性、设备及软件差异意味着跨机器数值不承诺逐位一致。

## 5. 报告重建、速度基准与打包

完成训练后，按 README 第10节取得图片样例和 E4 基准，再生成报告：

```powershell
python scripts/predict.py --run-dir results/E4_resolution --input data/CamVid/CamVid_RGB/0016E5_07959.png --output-dir results/inference_demo
python scripts/benchmark_inference.py --output-dir results/benchmark_E4
python delivery/build_report.py
python delivery/verify.py --working-tree
# 先提交代码和报告；--output 必须是尚不存在的 ZIP
python delivery/package.py --output delivery/artifacts/segformer_plus_delivery.zip
```

已有归档中直接使用提交包核验，无需重跑 benchmark。report_evidence.json 记录图像与指标来源哈希。提交包复制少量原预测文件供报告重建，但不复制全部预测。默认打包要求 Git 工作区干净；仅校验候选包可用 --allow-dirty，并在清单标出状态。从没有 .git 的解压目录创建新训练工作目录不依赖 Git；重新生成正式包则需要有效 Git 仓库和已提交版本。

本次交付另外在独立目录解压 ZIP，以现有固定环境验证清单、十次归档和 CPU 离线预测；并在没有数据、模型与 results 的源码目录执行测试。它证明文件迁移和离线推理可用，不替代一次新硬件上的完整训练复现。核验详情与本地 ZIP SHA256 见 delivery/release.json。
