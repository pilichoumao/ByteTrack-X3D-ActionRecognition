import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch

from config import ACTION_MODELS, DEFAULT_ACTION_MODEL

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MMACTION_DIR = os.path.join(ROOT_DIR, "mmaction2")

if MMACTION_DIR not in sys.path:
    sys.path.insert(0, MMACTION_DIR)

from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.dataset import Compose


class ActionRecognizer:
    def __init__(self, model_name=DEFAULT_ACTION_MODEL):
        if model_name not in ACTION_MODELS:
            raise ValueError(
                f"Unknown action model '{model_name}'. "
                f"Available: {sorted(ACTION_MODELS.keys())}"
            )

        model_cfg = ACTION_MODELS[model_name]
        self.model_name = model_name
        self.model_config_path = model_cfg["config"]
        self.model_ckpt_path = model_cfg["ckpt"]
        self.label_map_path = model_cfg["label_map"]

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Action device =", self.device)
        print("Action model =", self.model_name)

        self.model = init_recognizer(
            self.model_config_path,
            self.model_ckpt_path,
            device=self.device
        )

        self.labels = []
        with open(self.label_map_path, "r", encoding="utf-8") as f:
            for line in f:
                self.labels.append(line.strip())

        # 基于原 test_pipeline 构建“内存 clip 推理”专用 pipeline
        self.test_pipeline, self.color_format = self._build_array_test_pipeline()

        print("Action array inference pipeline ready")
        print("color format =", self.color_format)

    def _build_array_test_pipeline(self):
        """
        保留原 config 的测试流程，只把“视频初始化/视频解码”替换为 ArrayDecode，
        从而直接接受内存里的 4D array: [T, H, W, C]
        """
        pipeline_cfg = deepcopy(self.model.cfg.test_pipeline)

        init_ops = {
            "DecordInit", "OpenCVInit", "PyAVInit", "PIMSInit"
        }
        decode_ops = {
            "DecordDecode", "OpenCVDecode", "PyAVDecode", "PIMSDecode"
        }

        # 根据原来的 decode 类型决定是否需要 BGR->RGB
        # 一般 Decord/PyAV/PIMS 路线更接近 RGB；OpenCVDecode 更接近 BGR
        color_format = "RGB"
        for step in pipeline_cfg:
            step_type = step.get("type", "")
            if step_type == "OpenCVDecode":
                color_format = "BGR"
                break
            if step_type in {"DecordDecode", "PyAVDecode", "PIMSDecode"}:
                color_format = "RGB"
                break

        new_pipeline = []
        array_decode_inserted = False

        for step in pipeline_cfg:
            step = deepcopy(step)
            step_type = step.get("type", "")

            # 去掉视频 reader 初始化
            if step_type in init_ops:
                continue

            # 原本的视频解码改成 ArrayDecode
            if step_type in decode_ops:
                if not array_decode_inserted:
                    new_pipeline.append(dict(type="ArrayDecode"))
                    array_decode_inserted = True
                continue

            new_pipeline.append(step)

        # 如果原 pipeline 里没出现 decode 步骤，就尝试补一个 ArrayDecode
        if not array_decode_inserted:
            inserted = False
            patched = []
            for step in new_pipeline:
                patched.append(step)
                if step.get("type", "") in {
                    "SampleFrames", "UniformSample", "DenseSampleFrames"
                } and not inserted:
                    patched.append(dict(type="ArrayDecode"))
                    inserted = True

            if inserted:
                new_pipeline = patched
            else:
                # 极端情况下，直接放最前面
                new_pipeline = [dict(type="ArrayDecode")] + new_pipeline

        return Compose(new_pipeline), color_format

    def _prepare_array(self, clip):
        """
        clip: list[np.ndarray], 每帧是 HWC, OpenCV 读出来通常是 BGR
        返回: np.ndarray [T, H, W, C]
        """
        frames = []
        for img in clip:
            if self.color_format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        array = np.stack(frames, axis=0)  # [T, H, W, C]
        return array

    def infer_clip(self, clip):
        if not clip:
            return {
                "label": "unknown",
                "score": 0.0
            }

        array = self._prepare_array(clip)

        # 这里传入的是“结果字典”，不是视频路径
        # frame_inds 会由 pipeline 中的 SampleFrames 等步骤生成
        data = {
            "array": array,
            "total_frames": array.shape[0],
            "start_index": 0,
            "modality": "RGB",
            "label": -1
        }

        with torch.no_grad():
            result = inference_recognizer(
                self.model,
                data,
                test_pipeline=self.test_pipeline
            )

        pred_label = int(result.pred_label.item())
        pred_score = float(result.pred_score[pred_label].item())
        pred_action = self.labels[pred_label]

        return {
            "label": pred_action,
            "score": pred_score
        }
