# ActionRecognition (ByteTrack + MMAction2 + MMPose)

![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)
![Status](https://img.shields.io/badge/Status-Working-success)

一个基于 **ByteTrack + MMAction2 + MMPose** 的多人动作识别项目，支持 RGB 动作识别与 Skeleton 动作识别两条支线。

---

## 项目架构

当前支持 4 种动作模式：
- `rgb_x3d`
- `rgb_tsm`
- `skeleton_ctrgcn`
- `skeleton_stgcn`

### 1) RGB 支线（X3D / TSM）

```text
Video -> ByteTrack(人框+ID) -> ClipBuffer(按track缓存裁剪帧)
      -> MMAction2 RGB Recognizer(X3D/TSM) -> Action Label -> Visualization
```

### 2) Skeleton 支线（CTR-GCN / ST-GCN）

```text
Video -> ByteTrack(人框+ID) -> MMPose(逐track关键点)
      -> SkeletonBuffer(按track缓存关键点序列)
      -> SkeletonFormatter(整理为MMACTION输入)
      -> MMAction2 Skeleton Recognizer(CTR-GCN/ST-GCN)
      -> Action Label -> Visualization
```

说明：
- 当前默认按 `track_id` 进行单人序列推理（`M=1`）。
- Skeleton 分支可选骨架可视化叠加（关键点+连线）。

---

## 环境要求（简要）

推荐：
```text
Python 3.8
CUDA 11+
PyTorch
```

需要的开源模型/框架环境：
- **ByteTrack / YOLOX**（目标检测+多目标跟踪）
- **MMAction2**（X3D、TSM、CTR-GCN、ST-GCN 推理）
- **MMPose**（人体关键点估计，Skeleton 支线需要）


---

## 关键模型文件位置

```text
ByteTrack checkpoint:
  ByteTrack/pretrained/bytetrack_x_mot17.pth.tar

RGB checkpoints:
  mmaction2/checkpoints/x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth
  mmaction2/checkpoints/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth

Skeleton checkpoints:
  mmaction2/checkpoints/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230308-7aba454e.pth
  mmaction2/checkpoints/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth

Pose checkpoint:
  mmpose/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth
```

---

## 命令行参数说明

主命令：
```bash
python core/pipeline_demo.py --video video/demo.mp4 --mode <mode>
```

常用参数：
- `--video`：输入视频路径。
- `--mode`：运行模式，取值：
  - `rgb_x3d`
  - `rgb_tsm`
  - `skeleton_ctrgcn`
  - `skeleton_stgcn`
- `--action-model`：仅 RGB 模式生效，用于覆盖默认模型（如 `x3d_s`、`tsm_r50_1x1x8`）。
- `--suffix`：输出文件后缀；不填时自动包含 mode/model，避免覆盖。
- `--max-frames`：只跑前 N 帧用于调试。

Skeleton 相关参数：
- `--show-skeleton`：在输出视频中绘制骨架。
- `--skeleton-score-thr`：骨架可视化时关键点置信度阈值。
- `--pose-conf-thr`：判定 pose 有效的阈值（覆盖 config 默认值）。
- `--skeleton-smooth-window`：Skeleton 标签平滑窗口长度。
- `--skeleton-min-update-score`：低于该分数不更新显示标签。
- `--dry-run-skeleton`：只跑 pose+buffer，不跑 skeleton 分类器（快速排查链路）。

---

## 四种模型跑 Demo 命令

```bash
# 1) RGB + X3D
python core/pipeline_demo.py --video video/demo.mp4 --mode rgb_x3d

# 2) RGB + TSM
python core/pipeline_demo.py --video video/demo.mp4 --mode rgb_tsm --action-model tsm_r50_1x1x8

# 3) Skeleton + CTR-GCN
python core/pipeline_demo.py --video video/demo.mp4 --mode skeleton_ctrgcn --show-skeleton

# 4) Skeleton + ST-GCN
python core/pipeline_demo.py --video video/demo.mp4 --mode skeleton_stgcn --show-skeleton
```

---

## 输出文件命名策略

当 `--suffix` 为空时：
```text
output/<视频名>_<mode>_<effective_model>.mp4
```
示例：
```text
output/demo_rgb_x3d_x3d_s.mp4
output/demo_rgb_tsm_tsm_r50_1x1x8.mp4
output/demo_skeleton_ctrgcn_ctrgcn.mp4
output/demo_skeleton_stgcn_stgcn.mp4
```

---

## 目录结构（核心）

```text
core/
  config.py                        # 全局配置（路径、模式、模型）
  pipeline_demo.py                 # 编排入口（参数解析、主循环）
  pipeline_runners.py              # RGB/Skeleton 两个 runner
  tracker_adapter.py               # ByteTrack 适配
  action_recognizer.py             # RGB 识别器（X3D/TSM）
  clip_buffer.py                   # RGB clip 缓冲
  pose_estimator.py                # MMPose 封装
  skeleton_buffer.py               # 骨架时序缓冲
  skeleton_formatter.py            # 骨架输入格式整理
  skeleton_action_recognizer.py    # Skeleton 识别器（CTR-GCN/ST-GCN）
  visualizer.py                    # 画框/标签/骨架
```

---

## 说明

- 当前重点是在线推理闭环，不包含训练脚本。
- Skeleton 分支类别来自 NTU 标签体系，和日常视频语义可能存在域差异。
- 本项目遵循所依赖开源项目的许可证约束。
