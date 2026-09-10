# DINOv2 多类别人脸瑕疵热力图分割

一个完整的 PyTorch + Hugging Face DINOv2 多标签分割项目。输入为人像图片，每个类别对应一张独立热力图，因此同一像素可同时属于多个类别：

- `255`：该类别的监督正区域，目标为 1；
- `0`：该类别的监督负区域，目标为 0；
- `128`：该类别的不监督区域，仅对当前类别忽略。

数据路径、模型类型、Dataloader 类型、Loss、优化器、训练步数、验证间隔和 Accelerator 参数全部从 YAML 读取。命令行只需传入 YAML 文件。

## 目录

```text
detect_project/
├── acne_dinov2/
│   ├── config.py       # YAML 加载与校验
│   ├── data.py         # Dataset、配对逻辑、Dataloader registry
│   ├── model.py        # DINOv2 分割器与 smoke-test 小模型
│   ├── loss.py         # 忽略 128 的 BCE + Dice / Focal + Dice
│   ├── metrics.py      # IoU、Dice/F1、Precision、Recall、Accuracy
│   └── engine.py       # Accelerator 分布式验证
├── configs/
│   ├── train.yaml      # 正式 DINOv2 配置模板
│   └── smoke.yaml      # 无需下载 DINOv2 的流水线自检配置
├── tools/
│   └── create_smoke_data.py
├── train.py
├── test.py
├── requirements.txt
└── pyproject.toml
```

## 数据格式

类别由 YAML 的 `data.classes` 定义。图片和每个类别 mask 通过“相对于根目录的无扩展名路径”配对，mask 可统一使用 PNG：

```text
dataset/train/
├── images/
│   ├── person_001.jpg
│   └── sub/person_002.jpeg
└── labels/
    ├── acne/
    │   ├── person_001.png
    │   └── sub/person_002.png
    ├── pigmentation/
    │   ├── person_001.png
    │   └── sub/person_002.png
    └── scar/
        ├── person_001.png
        └── sub/person_002.png
```

每张输入图在每个类别目录下都必须存在一张灰度 mask。尺寸可以与输入不同，读取后使用最近邻插值。开启 `strict_labels: true` 时，出现 0、128、255 以外的像素将立即报错，避免 JPEG 标签或双线性缩放污染类别值。

## 安装

```powershell
cd D:\detect_project
python -m venv .venv
.venv\Scripts\Activate.ps1

# 有 NVIDIA GPU 时，建议先从 PyTorch 官网选择对应 CUDA 版本
pip install -e .
```

首次运行 DINOv2 会从 Hugging Face 下载 `facebook/dinov2-small` 权重。输入高宽最好为 patch size 14 的整数倍，例如 448×448。

## 配置

复制并编辑 `configs/train.yaml`，至少修改：

```yaml
data:
  classes: [acne, pigmentation, scar]
  train:
    input_dir: D:/your_dataset/train/images
    label_dir: D:/your_dataset/train/labels
  val:
    input_dir: D:/your_dataset/val/images
    label_dir: D:/your_dataset/val/labels
```

模型、Dataloader 和 Loss 均由 `type` 选择：

```yaml
model:
  type: dinov2_segmenter
dataloader:
  type: standard            # 或 weighted
loss:
  type: masked_bce_dice     # 或 masked_focal_dice
  positive_weight: [4.0, 2.0, 3.0]  # 顺序与 data.classes 相同
```

## 训练

单卡/CPU：

```powershell
python train.py --config configs/train.yaml
```

多卡：

```powershell
accelerate config
accelerate launch train.py --config configs/train.yaml
```

`train.val_every_steps` 控制每隔多少个**优化器 step**运行验证；梯度累积中的 micro-step 不会重复计数。`train.save_every_steps` 控制完整训练状态的保存间隔。最佳模型保存在：

```text
outputs/dinov2_acne/best/model.pt
```

中断恢复：将 YAML 中 `train.resume_from` 设置为某个 `checkpoint-step-XXXXXX` 目录。它会通过 `Accelerator.load_state()` 恢复模型、优化器、学习率调度器和随机状态。

## 测试

在 YAML 中设置：

```yaml
test:
  checkpoint: outputs/dinov2_acne/best/model.pt
  save_heatmaps: true
  output_dir: outputs/dinov2_acne/test_heatmaps
```

然后运行：

```powershell
python test.py --config configs/train.yaml
```

测试结果包含每类指标以及 macro/micro 汇总，并写入 `test_metrics.json`。预测热力图取值为 0—255，保存为 `输出目录/<类别>/<相对文件名>.png`。

## 不下载权重的完整流水线自检

```powershell
python tools/create_smoke_data.py --output-dir smoke_data
python train.py --config configs/smoke.yaml
python test.py --config configs/smoke.yaml
```

`smoke.yaml` 使用 acne、pigmentation、scar 三类独立 mask 和 `tiny_segmenter`，仅验证多通道读取、逐类别 128 ignore、Loss、反向传播、定期验证、checkpoint 和分类别热力图；正式训练必须使用 `configs/train.yaml` 中的 `dinov2_segmenter`。

## 设计说明

- DINOv2 输出多层 patch token，分割头将指定层投影、拼接并恢复为像素级 logits。
- 不确定像素被转换成 `ignore_index=-1`，不会进入 BCE、Dice 或指标统计。
- 验证指标通过 `accelerator.reduce` 在所有进程间汇总。
- `weighted` Dataloader 会提高含正样本标签图片的抽样概率，但不会替代像素级正类权重。
- 默认按 Dice 选择最佳 checkpoint。痘痘像素通常稀少，建议同时关注 PR-AUC；本项目的阈值型指标用于训练监控。

## 注意事项

本项目只做图像分割研究，不是医疗诊断系统。人脸图像属于敏感数据，应取得授权并按人物身份切分 train/val/test，避免同一人的不同照片跨集合造成数据泄漏。
