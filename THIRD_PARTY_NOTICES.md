# 第三方来源与许可

本交付面向课程研究与评价。保留依赖及预训练来源各自的许可；本文件不替代原许可，也不为第三方数据或权重另行授予使用权。

- **SegFormer / MiT-B0**：架构来自 Xie 等人的 [SegFormer（NeurIPS 2021）](https://arxiv.org/abs/2105.15203)。编码器来自 [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0)，固定 revision `80983a413c30d36a39c20203974ae7807835e2b4`。模型卡标注 License 为 other，并指向 [NVIDIA SegFormer 许可](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)。随附完整副本 [SegFormer_LICENSE.txt](docs/licenses/SegFormer_LICENSE.txt)，包括其研究、评价及非商业用途限制和再分发要求。本项目训练后的分割权重保留其预训练来源与许可说明。
- **CamVid**：原始数据来自 [University of Cambridge CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)，请遵循作者页面要求并引用 Brostow 等人的 ECCV 2008 与 Pattern Recognition Letters 数据论文。11 类整理来自 [lih627/CamVid 固定提交](https://github.com/lih627/CamVid/tree/79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6)。其仓库 MIT 许可副本见 [CamVid_repository_LICENSE.txt](docs/licenses/CamVid_repository_LICENSE.txt)；这不自动覆盖原始图像与标注。提交包仅保留有来源记录的少量课程展示图和一张验证示例，不包含完整数据集。
- **软件依赖**：Transformers、PyTorch、NumPy、Pillow、PyYAML、requests、matplotlib、huggingface-hub、safetensors 使用各自项目的许可；版本见 requirements.txt。提交包不打包这些第三方软件的安装文件。

许可副本的来源和 SHA256 记录于 [sources.json](docs/licenses/sources.json)。项目报告的数字来自本地真实实验，预训练权重、原始数据与第三方方法不作为本项目原创成果。
