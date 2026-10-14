# CamVid 数据放置与来源

使用同一个来源的 11 类数据和 train/val/test 清单：

- 整理来源：[lih627/CamVid](https://github.com/lih627/CamVid)
- 固定版本：`79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6`
- 原始来源：[Cambridge CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

下载工具只获取所需图像、`CamVidGray` 标签、三个清单及来源说明，不下载原视频和重复的 32 类标签。

```powershell
python scripts/download_camvid.py --workers 4
python scripts/audit_dataset.py
```

需要先查看清单时可使用 `--metadata-only`；这只下载元数据，无法通过完整数据验收。网络失败后重新运行相同命令会复用已有文件，并检查已有图像是否可读。下载完成后仍必须运行数据审计。

## 目录约定

```text
data/CamVid/
  CamVid_RGB/0001TP_006690.png
  CamVidGray/0001TP_006690_L.png
  camvid_train.txt
  camvid_val.txt
  camvid_test.txt
  LICENSE
  README.md
  camvid_data.py
  source_metadata.json
```

每个清单行是两条相对路径，例如：

```text
CamVid_RGB/0001TP_006690.png CamVidGray/0001TP_006690_L.png
```

也可以手动获取固定版本并按上述目录放置。清单不可包含越出数据根目录的路径；同一张图像或标签不可跨划分复用。审计同时核对清单数量、图像/Mask 名称与尺寸、有效 ID、ignore 比例，以及图像内容哈希是否跨划分重复。

## 类别编码

类别顺序和颜色已对照固定版本的 `camvid_data.py`；不是按另一数据集的标签顺序猜测。

| ID | 类别 | RGB |
|---|---|---|
| 0 | Sky | 128,128,128 |
| 1 | Building | 128,0,0 |
| 2 | Pole | 192,192,128 |
| 3 | Road | 128,64,128 |
| 4 | Sidewalk | 0,0,192 |
| 5 | Tree | 128,128,0 |
| 6 | SignSymbol | 192,128,128 |
| 7 | Fence | 64,64,128 |
| 8 | Car | 64,0,128 |
| 9 | Pedestrian | 64,64,0 |
| 10 | Bicyclist | 0,128,192 |
| 255 | Void / ignore | 0,0,0 |

默认 `label_encoding: index`。单通道 PNG 和 P 模式 PNG 都保留其原始 ID，禁止先转换成灰度亮度。RGB 模式只接受表中 11 类颜色和黑色 ignore；非法颜色会报告，不能静默吞并为 ignore。

## 许可与引用记录

工具保存第三方仓库的 `LICENSE` 和 `README.md`。仓库代码的 MIT 许可不能自动视为原始图像和标注的完整授权；保留原始 CamVid 来源及使用要求，并在课程报告引用 Brostow 等人的数据集论文。原始页面要求引用 ECCV 2008 的 *Segmentation and Recognition Using Structure from Motion Point Clouds* 及 *Semantic Object Classes in Video: A High-Definition Ground Truth Database*。实际数据和大型权重已加入 Git ignore。
