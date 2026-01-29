# -*- coding: utf-8 -*-
"""
ALL-IN-ONE Pipeline
===================
1) YOLO(带proc预处理) 推理 -> detections.csv + diff_base.mp4
2) IoU Tracker -> tracks.csv + tracks_summary.csv + tracked_overlay.mp4
3) Action Inference (EfficientNetV2-S + GRU) -> pred_tracks.csv (+可选 ROI clips)
4) ✅ Action Overlay Video：在 tracked_overlay.mp4 的基础上，把每个track的“动作/物体类别”写到框旁边 -> action_overlay.mp4
5) ✅ Efficient额外输出两个视频：
   A) ROI 预览视频：preprocess后裁ROI（把每条track采样到的ROI按顺序拼成一个总览视频） -> roi_preview.mp4
   B) proc底图识别视频：逐帧对原视频做 preprocess 得到 proc，再在proc上画轨迹框+类别文字 -> proc_action_overlay.mp4

核心原则：统一坐标系 = “原视频坐标”
- detections.csv: 原视频坐标（整合版推荐 RESIZE_TO=None）
- tracks.csv: 原视频坐标
- action裁ROI：对原视频帧先做 preprocess_frame 得到 proc_frame，然后用 tracks 的 bbox 直接裁（因为 preprocess 不改变尺寸）
"""

# ============================================================
# =========================== CONFIG ==========================
# ============================================================

import os
import csv
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ------------------ 总开关：你可以只跑某一步 ------------------
# 是否执行 YOLO 检测阶段（生成 detections.csv / diff/overlay）
RUN_YOLO = True
# 是否执行 Tracking 阶段（生成 tracks.csv / tracked_overlay.mp4）
RUN_TRACK = True
# 是否执行 Action 阶段（EfficientNet+GRU 分类每条轨迹）
RUN_ACTION = True
# 是否输出 action_overlay.mp4（在 tracked_overlay 上叠动作标签）
RUN_ACTION_OVERLAY_VIDEO = True   # ✅ 在 tracked_overlay.mp4 上叠类别字 -> action_overlay.mp4

# ✅ Efficient 额外输出：ROI预览 + proc底图识别视频
# 是否输出 ROI 预览视频（把每条轨迹采样的 ROI 拼成总览）
RUN_ACTION_ROI_PREVIEW_VIDEO = True
# 是否输出 proc 底图识别视频（逐帧 preprocess 再画框+标签）
RUN_ACTION_PROC_OVERLAY_VIDEO = True

# ------------------ 输入输出路径 ------------------
# 原始视频路径（作为默认兜底）
VIDEO_PATH = r"C:\Users\Administrator\Desktop\sonar_track\data\test_video\1.mp4"

# 分阶段输入视频（留空则回退到 VIDEO_PATH）
# - YOLO/Track: 建议填 proc.mp4
# - Action: 建议填 diff.mp4
YOLO_INPUT_VIDEO_PATH = ""   # e.g. r"..\data\processed_sonar_video\001\proc.mp4"
TRACK_INPUT_VIDEO_PATH = ""  # e.g. r"..\data\processed_sonar_video\001\proc.mp4"
ACTION_INPUT_VIDEO_PATH = "" # e.g. r"..\data\processed_sonar_video\001\diff.mp4"

# 输出总目录（下面自动生成 01_yolo / 02_track / 03_action）
OUT_ROOT = os.path.join(ROOT_DIR, "runs_allinone")
YOLO_OUT_DIR = os.path.join(OUT_ROOT, "01_yolo")
TRACK_OUT_DIR = os.path.join(OUT_ROOT, "02_track")
ACTION_OUT_DIR = os.path.join(OUT_ROOT, "03_action")

# ------------------ YOLO 参数 ------------------
# YOLO 权重路径（best.pt/last.pt）
WEIGHTS_PATH = r"C:\Users\Administrator\Desktop\sonar_track\YOLO_training\runs_yolo11_from_scratch\y11_from_images_img1280\weights\last.pt"
# YOLOv11 配置 YAML（state_dict 兼容加载时用）
YOLO11_YAML = "ultralytics/cfg/models/11/yolo11.yaml"
# 类别数（必须和训练一致）
NC = 1

# 推理输入尺寸（YOLO 内部会做 letterbox）
IMGSZ = 1280
# 置信度阈值（越低召回高但噪声多）
CONF_THRES = 0.20
# NMS IoU 阈值（越低抑制更强）
IOU_THRES = 0.50
# 单帧最大检测框数上限
MAX_DET = 300
# 设备：cuda / cpu
DEVICE = "cuda"  # cuda/cpu

# YOLO 输入类型：
# - "raw": 原始视频帧，需执行 preprocess_frame
# - "proc": 已预处理好的视频帧，直接用于 YOLO（不再重复 preprocess）
YOLO_INPUT_MODE = "proc"

# 视频帧范围（原视频帧号）
START_FRAME = 0
END_FRAME = -1
# 帧步长：=1 全帧；>1 表示跳帧（注意会影响 diff/track 对齐）
FRAME_STRIDE = 1

# ------------------ 预处理参数（与你现有保持一致） ------------------
# UI/字幕/小窗遮罩（x1,y1,x2,y2），会被置黑
MASK_RECTS = [
    # (0, 620, 1280, 720),
    # (860, 470, 1280, 720),
]

# 扇形 mask：手动模式（更可控）
USE_MANUAL_SECTOR = False
APEX_X = 640
APEX_Y = 710
RADIUS_MIN = 0
RADIUS_MAX = 720
ANGLE_LEFT_DEG = -35.0
ANGLE_RIGHT_DEG = 35.0
SECTOR_MASK_STEP_DEG = 0.2

# 扇形 mask：自动模式（阈值+形态学）
USE_AUTO_SECTOR_IF_MANUAL_FALSE = True
AUTO_BLACK_THRESH = 8
AUTO_MORPH_KERNEL = 9
AUTO_KEEP_LARGEST_CONTOUR = True

# CLAHE：提升局部对比度（可选，可能放大噪声）
USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# 百分位裁剪（稳健归一化，去离群亮点）
PCT_CLIP_LOW = 0.5
PCT_CLIP_HIGH = 99.5
EPS = 1e-6

# 输出格式（给 YOLO）：灰度 or 3 通道
OUTPUT_GRAYSCALE = False
OUTPUT_3CH = True

# 关键：为了坐标一致，整合版建议不 resize（保持和原视频同尺寸）
RESIZE_TO = None  # None 表示不resize（推荐）

# 降噪开关（声呐 speckle 噪声）
DENOISE_ENABLE = False
# 降噪方法：bilateral/nlm/median/gaussian/none
DENOISE_METHOD = "bilateral"
# bilateral 参数（保边平滑）
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 10
# NLM 参数（强但慢）
NLM_H = 10
NLM_TEMPLATE_W = 7
NLM_SEARCH_W = 21
# median/gaussian 参数（便宜但易糊边）
MEDIAN_KSIZE = 3
GAUSSIAN_KSIZE = 3
GAUSSIAN_SIGMA = 0

# YOLO 输出
YOLO_SAVE_JSON = True
# ✅ 若 Action 使用 diff 作为输入，必须输出 diff_base.mp4
YOLO_SAVE_DIFF_BASE = True   # 纯diff底图（无框）
YOLO_SAVE_OVERLAY = True     # 输出 yolo_overlay_diff（diff底图+框）

# ------------------ Tracking 参数（与你现有一致） ------------------
# IoU 匹配阈值（越大越严格）
IOU_MATCH_THRES = 0.30
# 轨迹最大“失配”帧数（超出即删除）
MAX_AGE = 20

# 轨迹有效性门槛（summary/clip 用）
MIN_HITS = 5
# 平均置信度门槛
MIN_MEAN_CONF = 0.25
# 是否先过滤低置信检测框
USE_CONF_FILTER = True
# 低置信过滤阈值（检测层）
CONF_FILTER = 0.20

# 显示/写出轨迹的闸门（决定 tracks.csv 是否写入）
MIN_HITS_TO_SHOW = 2
# 最近窗口长度（单位：帧；<=0 表示关闭连续性过滤）
RECENT_WINDOW = 5
# 最近窗口内最少命中次数
RECENT_MIN_HITS = 0
# 新轨迹孵化期：允许短轨先展示
WARMUP_ALLOW = True
WARMUP_MAX_FRAMES = 8

# EMA 平滑（让框更稳）
USE_EMA = True
EMA_ALPHA = 0.7

# 画框/ID/置信度显示开关
TRACK_DRAW_BOX = True
TRACK_DRAW_ID = True
TRACK_DRAW_CONF = True

# 是否在 tracking 阶段把“底图”预处理成 proc 再画框
# True: 即使输入是原视频，也会显示为 proc 背景
TRACK_PREPROCESS_BACKGROUND = True

# ✅ 对齐 track_and_export_clips：默认用原视频做底图
# 若想用 diff 视频，可手动填路径（例如 yolo 输出的 overlay/diff）
TRACK_DIFF_VIDEO_PATH = ""

# ------------------ Action 模型参数 ------------------
# 动作模型权重路径
ACTION_CKPT = os.path.join(ROOT_DIR, "EfficientNet_training", "runs_action", "effnetv2s_gru_binary", "best.pt")  # 你的动作模型 ckpt
# Action 输入类型：
# - "raw": 原始视频帧（内部会做 preprocess_frame）
# - "diff": 已生成的 diff 视频帧（不再做 preprocess_frame）
ACTION_INPUT_MODE = "diff"
# 每条轨迹采样多少帧做动作分类
ACTION_NUM_FRAMES = 16
# ROI 输入尺寸（模型输入）
ACTION_FRAME_SIZE = 224
# 轨迹框外扩倍数（包含更多上下文）
ACTION_BBOX_EXPAND = 1.5
# 二分类阈值（sigmoid >= 阈值判为正类）
ACTION_THRESH = 0.5

# 是否导出每条轨迹的 ROI clip
EXPORT_ROI_CLIPS = True
# ROI clip 输出帧率
ROI_CLIPS_FPS = 7.0

# ✅ 类别名称（你想在视频里显示的“物体种类/动作种类”）
# 二分类时：0=非蛙人，1=蛙人（你可以改成你自己的命名）
ACTION_LABEL_NAMES = {
    0: "非蛙人",
    1: "蛙人",
}

# ✅ 输出“带类别文字”的视频参数（基于 tracked_overlay.mp4）
# 叠字大小
ACTION_OVERLAY_TEXT_SCALE = 0.65
# 叠字粗细
ACTION_OVERLAY_TEXT_THICK = 2
# 叠字颜色（BGR）
ACTION_OVERLAY_TEXT_COLOR = (0, 255, 255)  # 黄字更醒目
# 是否给文字加半透明黑底（增强可读性）
ACTION_OVERLAY_BG = True                   # 是否画一个黑底让字更清楚
ACTION_OVERLAY_BG_ALPHA = 0.55             # 黑底透明度
# 输出文件名
ACTION_OVERLAY_OUT_NAME = "action_overlay.mp4"
# ✅ 额外输出：diff / raw 两个底图版本
ACTION_OVERLAY_DIFF_OUT_NAME = "action_overlay_diff.mp4"
ACTION_OVERLAY_RAW_OUT_NAME = "action_overlay_raw.mp4"

# 中文字体（PIL 绘字用）
TEXT_FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
TEXT_FONT_SIZE_BASE = 24

# ✅ Efficient 额外输出（你刚刚要的两个视频）
ACTION_ROI_PREVIEW_OUT_NAME = "roi_preview.mp4"            # preprocess后裁ROI，总览拼接视频
ACTION_PROC_OVERLAY_OUT_NAME = "proc_action_overlay.mp4"   # proc底图 + 轨迹框 + 类别文字

# ROI预览视频：是否在每个ROI上把 tid/label 写上去（更容易对齐）
ACTION_ROI_PREVIEW_DRAW_TEXT = True


# ============================================================
# ======================= Common Utils =======================
# ============================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    n = 0
    for _, v in obj.items():
        if torch.is_tensor(v):
            n += 1
        if n >= 5:
            return True
    return False

def strip_bom(s: str) -> str:
    """去掉字符串开头的 BOM（\ufeff）"""
    if not isinstance(s, str):
        return s
    return s.lstrip("\ufeff")


# ============================================================
# ===================== Preprocess (YOLO/Action) =============
# ============================================================

def _apply_rect_masks(frame_bgr: np.ndarray, rects: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2) in rects:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c > x1c and y2c > y1c:
            out[y1c:y2c, x1c:x2c] = 0
    return out

def _build_manual_sector_mask(
    shape_hw: Tuple[int, int],
    apex_xy: Tuple[int, int],
    radius_min: int,
    radius_max: int,
    angle_left_deg: float,
    angle_right_deg: float,
    step_deg: float = 0.2,
) -> np.ndarray:
    h, w = shape_hw
    ax, ay = apex_xy
    a0 = min(angle_left_deg, angle_right_deg)
    a1 = max(angle_left_deg, angle_right_deg)

    angles = np.arange(a0, a1 + step_deg, step_deg, dtype=np.float32)
    thetas = np.deg2rad(angles)

    outer_pts = []
    for th in thetas:
        x = ax + radius_max * np.sin(th)
        y = ay - radius_max * np.cos(th)
        outer_pts.append([int(round(x)), int(round(y))])

    if radius_min > 0:
        inner_pts = []
        for th in thetas[::-1]:
            x = ax + radius_min * np.sin(th)
            y = ay - radius_min * np.cos(th)
            inner_pts.append([int(round(x)), int(round(y))])
        poly = np.array(outer_pts + inner_pts, dtype=np.int32)
    else:
        poly = np.array([[ax, ay]] + outer_pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

def _build_auto_sector_mask(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, AUTO_BLACK_THRESH, 255, cv2.THRESH_BINARY)

    k = AUTO_MORPH_KERNEL
    if k and k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    if AUTO_KEEP_LARGEST_CONTOUR:
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bw
        largest = max(contours, key=cv2.contourArea)
        out = np.zeros_like(bw)
        cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
        return out
    return bw

def _normalize_intensity(gray_u8: np.ndarray) -> np.ndarray:
    g = gray_u8.astype(np.float32)
    lo = np.percentile(g, PCT_CLIP_LOW)
    hi = np.percentile(g, PCT_CLIP_HIGH)
    hi = max(hi, lo + EPS)

    g = np.clip(g, lo, hi)
    g = (g - lo) / (hi - lo + EPS) * 255.0
    g = g.astype(np.uint8)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        g = clahe.apply(g)
    return g

def _denoise_gray(gray_u8: np.ndarray) -> np.ndarray:
    if (not DENOISE_ENABLE) or (DENOISE_METHOD is None):
        return gray_u8
    method = str(DENOISE_METHOD).lower().strip()
    g = gray_u8.astype(np.uint8) if gray_u8.dtype != np.uint8 else gray_u8

    if method == "none":
        return g
    if method == "bilateral":
        return cv2.bilateralFilter(g, d=int(BILATERAL_D),
                                  sigmaColor=float(BILATERAL_SIGMA_COLOR),
                                  sigmaSpace=float(BILATERAL_SIGMA_SPACE))
    if method == "nlm":
        return cv2.fastNlMeansDenoising(g, None,
                                       h=float(NLM_H),
                                       templateWindowSize=int(NLM_TEMPLATE_W),
                                       searchWindowSize=int(NLM_SEARCH_W))
    if method == "median":
        k = int(MEDIAN_KSIZE)
        k = k if (k % 2 == 1) else (k + 1)
        k = max(3, k)
        return cv2.medianBlur(g, k)
    if method == "gaussian":
        k = int(GAUSSIAN_KSIZE)
        k = k if (k % 2 == 1) else (k + 1)
        k = max(3, k)
        return cv2.GaussianBlur(g, (k, k), float(GAUSSIAN_SIGMA))
    return g

def preprocess_frame(frame_bgr: np.ndarray, sector_mask: np.ndarray) -> np.ndarray:
    """
    输入：原帧（或resize后的帧）
    输出：proc帧（mask + 灰度归一化逻辑保持一致）
    注意：整合版里不改变尺寸，所以bbox坐标能保持一致
    """
    masked = frame_bgr.copy()
    masked[sector_mask == 0] = 0

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray = _denoise_gray(gray)
    gray = _normalize_intensity(gray)

    if OUTPUT_GRAYSCALE:
        if OUTPUT_3CH:
            return cv2.merge([gray, gray, gray])
        return gray

    return masked

def build_sector_mask_from_video_first_frame(cap: cv2.VideoCapture) -> np.ndarray:
    cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("无法读取视频第一帧，无法构建 sector mask。")

    if RESIZE_TO is not None:
        first = cv2.resize(first, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

    first = _apply_rect_masks(first, MASK_RECTS)

    if USE_MANUAL_SECTOR:
        h, w = first.shape[:2]
        sector_mask = _build_manual_sector_mask(
            (h, w),
            (APEX_X, APEX_Y),
            RADIUS_MIN,
            RADIUS_MAX,
            ANGLE_LEFT_DEG,
            ANGLE_RIGHT_DEG,
            step_deg=SECTOR_MASK_STEP_DEG,
        )
    else:
        if not USE_AUTO_SECTOR_IF_MANUAL_FALSE:
            raise RuntimeError("手动扇形mask关闭且自动mask关闭，请至少开启一个。")
        sector_mask = _build_auto_sector_mask(first)

    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
    return sector_mask


# ============================================================
# =========================== YOLO ============================
# ============================================================

def load_yolo_model_auto(weights_path: str, yaml_path: str, device: str):
    """
    兼容：
    A) 标准 Ultralytics 权重：YOLO(weights)
    B) 纯 state_dict：YOLO(yaml) + DetectionModel(nc=NC) 替换 y.model 再 load_state_dict
    """
    device_t = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    from ultralytics import YOLO

    try:
        y = YOLO(weights_path)
        return ("ultralytics_weights", y, device_t)
    except Exception as e:
        print(f"[Warn] YOLO(weights) 直接加载失败，尝试 state_dict。原因：{e}")

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif is_state_dict_like(ckpt):
        sd = ckpt
    else:
        raise RuntimeError("无法识别权重格式。")

    y = YOLO(yaml_path)
    from ultralytics.nn.tasks import DetectionModel
    y.model = DetectionModel(cfg=yaml_path, ch=3, nc=NC, verbose=False)
    missing, unexpected = y.model.load_state_dict(sd, strict=False)
    print(f"[Info] state_dict loaded. missing={len(missing)} unexpected={len(unexpected)}")
    y.model.to(device_t).eval()
    return ("ultralytics_state_dict", y, device_t)

def yolo_predict_one_frame(yolo_model, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    results = yolo_model.predict(
        source=frame_bgr,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False,
    )
    r = results[0]
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets

    xyxy = r.boxes.xyxy.detach().cpu().numpy()
    conf = r.boxes.conf.detach().cpu().numpy()
    cls = r.boxes.cls.detach().cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        dets.append({
            "cls": int(cls[i]),
            "conf": float(conf[i]),
            "xyxy": [float(x) for x in xyxy[i].tolist()],
        })
    return dets

def proc_to_gray_u8(proc_frame: np.ndarray) -> np.ndarray:
    if proc_frame.ndim == 2:
        g = proc_frame
    else:
        g = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = g.astype(np.uint8)
    return g

def gray_to_3ch_bgr(gray_u8: np.ndarray) -> np.ndarray:
    if gray_u8.ndim == 2:
        return cv2.merge([gray_u8, gray_u8, gray_u8])
    if gray_u8.ndim == 3 and gray_u8.shape[2] == 3:
        return gray_u8
    raise ValueError(f"invalid gray shape: {gray_u8.shape}")

def map_dets_to_original(dets: List[Dict[str, Any]], sx: float, sy: float) -> List[Dict[str, Any]]:
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return dets
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        out.append({
            "cls": d["cls"],
            "conf": d["conf"],
            "xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
        })
    return out

def draw_boxes(img_bgr: np.ndarray, dets: List[Dict[str, Any]]):
    h, w = img_bgr.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{d['cls']} {d['conf']:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

def yolo_stage(video_path: str, out_dir: str, input_mode: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    输出：
      - detections_csv
      - diff_base_video（可选，给 tracker 当底图）
    """
    ensure_dir(out_dir)
    detections_csv = os.path.join(out_dir, "detections.csv")
    detections_json = os.path.join(out_dir, "detections.json")
    diff_base_video = os.path.join(out_dir, "diff_base.mp4")
    overlay_video = os.path.join(out_dir, "yolo_overlay_diff.mp4")

    mode, yolo, device_t = load_yolo_model_auto(WEIGHTS_PATH, YOLO11_YAML, DEVICE)
    print(f"[YOLO] model mode={mode} device={device_t}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[YOLO] 无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 7.0
    if w0 <= 0 or h0 <= 0:
        raise RuntimeError("[YOLO] 视频分辨率读取失败。")

    mode = (input_mode or YOLO_INPUT_MODE)
    mode = str(mode).lower().strip()
    sector_mask = None
    if mode != "proc":
        sector_mask = build_sector_mask_from_video_first_frame(cap)

    # 输出视频尺寸：与 yolo_infer_video.py 一致
    if RESIZE_TO is not None:
        out_w, out_h = int(RESIZE_TO[0]), int(RESIZE_TO[1])
    else:
        out_w, out_h = w0, h0
    out_fps = fps / max(1, FRAME_STRIDE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    vw_diff = None
    vw_overlay = None
    if YOLO_SAVE_DIFF_BASE:
        vw_diff = cv2.VideoWriter(diff_base_video, fourcc, float(out_fps), (out_w, out_h))
        if not vw_diff.isOpened():
            raise RuntimeError(f"[YOLO] VideoWriter 打开失败：{diff_base_video}")
    if YOLO_SAVE_OVERLAY:
        vw_overlay = cv2.VideoWriter(overlay_video, fourcc, float(out_fps), (out_w, out_h))
        if not vw_overlay.isOpened():
            raise RuntimeError(f"[YOLO] VideoWriter 打开失败：{overlay_video}")

    end_frame = (total_frames - 1) if (END_FRAME < 0) else END_FRAME
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frame_id = START_FRAME

    output_json = {
        "video_path": video_path,
        "weights_path": WEIGHTS_PATH,
        "mode": mode,
        "imgsz": IMGSZ,
        "conf_thres": CONF_THRES,
        "iou_thres": IOU_THRES,
        "max_det": MAX_DET,
        "start_frame": START_FRAME,
        "end_frame": int(end_frame),
        "frame_stride": FRAME_STRIDE,
        "video_meta": {"fps": fps, "width": w0, "height": h0, "total_frames": total_frames},
        "frames": []
    }

    prev_proc_gray = None

    # ✅ 对齐 yolo_infer_video.py 的 CSV 输出格式（utf-8，无 BOM）
    with open(detections_csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("frame_id,time_sec,cls,conf,x1,y1,x2,y2\n")

        if total_frames > 0:
            n_proc = max(0, (min(end_frame, total_frames - 1) - START_FRAME) // max(1, FRAME_STRIDE) + 1)
        else:
            n_proc = None

        pbar = tqdm(total=n_proc, ncols=110, desc="YOLO")
        t0 = time.time()

        while True:
            if frame_id > end_frame:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if (frame_id - START_FRAME) % FRAME_STRIDE != 0:
                frame_id += 1
                continue

            frame_work = frame

            if RESIZE_TO is not None:
                frame_work = cv2.resize(frame_work, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

            # 输入模式：
            # - raw: 需要 mask+normalize -> preprocess_frame
            # - proc: 已预处理，直接作为 YOLO 输入
            if mode == "proc":
                frame_proc = frame_work
            else:
                frame_work = _apply_rect_masks(frame_work, MASK_RECTS)
                frame_proc = preprocess_frame(frame_work, sector_mask)  # proc 输入 YOLO

            dets_proc = yolo_predict_one_frame(yolo, frame_proc)

            proc_gray = proc_to_gray_u8(frame_proc)
            if prev_proc_gray is None:
                diff_gray = np.zeros_like(proc_gray)
            else:
                diff_gray = cv2.absdiff(proc_gray, prev_proc_gray)
            prev_proc_gray = proc_gray

            diff_bgr = gray_to_3ch_bgr(diff_gray)

            # 确保尺寸匹配 writer
            if diff_bgr.shape[1] != out_w or diff_bgr.shape[0] != out_h:
                diff_bgr = cv2.resize(diff_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            if vw_diff is not None:
                vw_diff.write(diff_bgr)

            if vw_overlay is not None:
                vis = diff_bgr.copy()
                draw_boxes(vis, dets_proc)
                vw_overlay.write(vis)

            # ✅ 保存检测结果（默认保存为“原视频坐标”）
            if RESIZE_TO is not None:
                sx = float(w0) / float(RESIZE_TO[0])
                sy = float(h0) / float(RESIZE_TO[1])
            else:
                sx = 1.0
                sy = 1.0
            dets_orig = map_dets_to_original(dets_proc, sx, sy)

            t_sec = frame_id / fps
            output_json["frames"].append({"frame_id": int(frame_id), "time_sec": float(t_sec), "dets": dets_orig})

            for d in dets_orig:
                x1, y1, x2, y2 = d["xyxy"]
                fcsv.write(f"{frame_id},{t_sec:.6f},{d['cls']},{d['conf']:.6f},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")

            dt = time.time() - t0
            proc_frames = len(output_json["frames"])
            fps_now = proc_frames / max(1e-6, dt)
            pbar.set_postfix(proc_fps=f"{fps_now:.2f}", dets=len(dets_proc), frame=frame_id)
            pbar.update(1)

            frame_id += 1

        pbar.close()

    cap.release()
    if vw_diff is not None:
        vw_diff.release()
    if vw_overlay is not None:
        vw_overlay.release()

    if YOLO_SAVE_JSON:
        with open(detections_json, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

    print("\n[YOLO] done")
    print("  detections.csv:", os.path.abspath(detections_csv))
    if YOLO_SAVE_DIFF_BASE:
        print("  diff_base.mp4 :", os.path.abspath(diff_base_video))
    if YOLO_SAVE_OVERLAY:
        print("  yolo_overlay  :", os.path.abspath(overlay_video))

    return detections_csv, (diff_base_video if YOLO_SAVE_DIFF_BASE else None)


# ============================================================
# =========================== Tracking ========================
# ============================================================

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def clamp_xyxy(xyxy, w, h):
    x1,y1,x2,y2 = xyxy
    x1 = float(np.clip(x1, 0, w-1))
    x2 = float(np.clip(x2, 0, w-1))
    y1 = float(np.clip(y1, 0, h-1))
    y2 = float(np.clip(y2, 0, h-1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1,y1,x2,y2]

class Track:
    def __init__(self, tid, frame_id, det):
        self.tid = tid
        self.last_frame = frame_id
        self.age = 0
        self.hits = 1
        self.conf_sum = float(det["conf"])
        self.cls = int(det["cls"])
        self.xyxy = det["xyxy"][:]
        self.history = [(frame_id, det["xyxy"][:], float(det["conf"]))]

    def mean_conf(self):
        return self.conf_sum / max(1, self.hits)

    def recent_hits(self, window: int, cur_frame: int) -> int:
        if window <= 0:
            return self.hits
        start = cur_frame - window + 1
        cnt = 0
        for fid, _, _ in reversed(self.history):
            if fid < start:
                break
            cnt += 1
        return cnt

    def life_frames(self, cur_frame: int) -> int:
        first = self.history[0][0] if self.history else cur_frame
        return int(cur_frame - first + 1)

    def update(self, frame_id, det):
        self.last_frame = frame_id
        self.age = 0
        self.hits += 1
        self.conf_sum += float(det["conf"])
        nb = np.array(det["xyxy"], dtype=np.float32)

        if USE_EMA:
            bb = np.array(self.xyxy, dtype=np.float32)
            sm = (EMA_ALPHA * bb + (1.0 - EMA_ALPHA) * nb).tolist()
        else:
            sm = nb.tolist()

        self.xyxy = sm
        self.history.append((frame_id, sm, float(det["conf"])))

    def step(self):
        self.age += 1

def should_show_track(t: Track, cur_frame: int) -> bool:
    if not t.history or t.history[-1][0] != cur_frame:
        return False

    if t.mean_conf() < MIN_MEAN_CONF:
        if not (WARMUP_ALLOW and t.life_frames(cur_frame) <= WARMUP_MAX_FRAMES and t.hits >= 2):
            return False

    if t.hits < MIN_HITS_TO_SHOW:
        if not (WARMUP_ALLOW and t.life_frames(cur_frame) <= WARMUP_MAX_FRAMES and t.hits >= 2):
            return False

    if RECENT_WINDOW > 0:
        rh = t.recent_hits(RECENT_WINDOW, cur_frame)
        if rh < RECENT_MIN_HITS:
            return False

    return True

def draw_track(img, det, tid=None):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = clamp_xyxy(det["xyxy"], w, h)
    x1i,y1i,x2i,y2i = int(x1),int(y1),int(x2),int(y2)
    if TRACK_DRAW_BOX:
        cv2.rectangle(img, (x1i,y1i), (x2i,y2i), (0,255,0), 2)
    parts = []
    if TRACK_DRAW_ID and tid is not None:
        parts.append(f"id={tid}")
    if TRACK_DRAW_CONF:
        parts.append(f"{det['conf']:.2f}")
    text = " ".join(parts)
    if text:
        cv2.putText(img, text, (x1i, max(0,y1i-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

def tracking_stage(detections_csv: str, base_video_path: str, out_dir: str) -> Tuple[str, str, str]:
    ensure_dir(out_dir)

    tracks_csv = os.path.join(out_dir, "tracks.csv")
    tracks_summary_csv = os.path.join(out_dir, "tracks_summary.csv")
    tracked_video = os.path.join(out_dir, "tracked_overlay.mp4")

    # ✅ 对齐 track_and_export_clips：直接读取 CSV
    df = pd.read_csv(detections_csv)

    df = df.sort_values(["frame_id", "conf"], ascending=[True, False]).reset_index(drop=True)
    if USE_CONF_FILTER:
        df = df[df["conf"] >= CONF_FILTER].reset_index(drop=True)

    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Track] 无法打开视频: {base_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 7.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(tracked_video, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError("[Track] VideoWriter 打开失败")

    # 如果要求底图使用 proc，则在这里构建 sector_mask
    sector_mask = None
    if TRACK_PREPROCESS_BACKGROUND:
        sector_mask = build_sector_mask_from_video_first_frame(cap)

    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for _, r in df.iterrows():
        fid = int(r["frame_id"])
        det = {
            "cls": int(r["cls"]),
            "conf": float(r["conf"]),
            "xyxy": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
        }
        by_frame.setdefault(fid, []).append(det)

    tracks: List[Track] = []
    next_id = 1
    rows_out = []

    pbar = tqdm(total=total_frames if total_frames > 0 else None, ncols=110, desc="Track")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        dets = by_frame.get(frame_id, [])

        for t in tracks:
            t.step()

        used_det = [False] * len(dets)
        for t in tracks:
            best_j = -1
            best_iou = 0.0
            for j, d in enumerate(dets):
                if used_det[j]:
                    continue
                if int(d["cls"]) != t.cls:
                    continue
                i = iou_xyxy(t.xyxy, d["xyxy"])
                if i > best_iou:
                    best_iou = i
                    best_j = j
            if best_j >= 0 and best_iou >= IOU_MATCH_THRES:
                used_det[best_j] = True
                t.update(frame_id, dets[best_j])

        for j, d in enumerate(dets):
            if not used_det[j]:
                tracks.append(Track(next_id, frame_id, d))
                next_id += 1

        tracks = [t for t in tracks if t.age <= MAX_AGE]

        if TRACK_PREPROCESS_BACKGROUND and sector_mask is not None:
            frame_work = frame
            if RESIZE_TO is not None:
                frame_work = cv2.resize(frame_work, RESIZE_TO, interpolation=cv2.INTER_LINEAR)
            frame_work = _apply_rect_masks(frame_work, MASK_RECTS)
            proc = preprocess_frame(frame_work, sector_mask)
            vis = _ensure_3ch(proc).copy()
            if vis.shape[1] != W or vis.shape[0] != H:
                vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            vis = frame.copy()
        for t in tracks:
            if should_show_track(t, frame_id):
                det_vis = {"cls": t.cls, "conf": t.history[-1][2], "xyxy": t.xyxy}
                draw_track(vis, det_vis, tid=t.tid)

                rows_out.append({
                    "frame_id": frame_id,
                    "track_id": t.tid,
                    "cls": t.cls,
                    "conf": float(det_vis["conf"]),
                    "x1": det_vis["xyxy"][0], "y1": det_vis["xyxy"][1],
                    "x2": det_vis["xyxy"][2], "y2": det_vis["xyxy"][3],
                })

        writer.write(vis)
        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    tracks_df = pd.DataFrame(rows_out)
    if len(tracks_df) == 0:
        tracks_df.to_csv(tracks_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["track_id","cls","len","start_frame","end_frame","mean_conf"]).to_csv(
            tracks_summary_csv, index=False, encoding="utf-8-sig"
        )
        print("[Track] ⚠️ 没有轨迹通过过滤门槛")
        return tracks_csv, tracks_summary_csv, tracked_video

    summary = []
    for tid, g in tracks_df.groupby("track_id"):
        summary.append({
            "track_id": int(tid),
            "cls": int(g["cls"].iloc[0]),
            "len": int(len(g)),
            "start_frame": int(g["frame_id"].min()),
            "end_frame": int(g["frame_id"].max()),
            "mean_conf": float(g["conf"].mean()),
        })
    summary_df = pd.DataFrame(summary).sort_values(["len","mean_conf"], ascending=[False, False])

    tracks_df.to_csv(tracks_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(tracks_summary_csv, index=False, encoding="utf-8-sig")

    print("\n[Track] done")
    print("  tracked_overlay:", os.path.abspath(tracked_video))
    print("  tracks.csv     :", os.path.abspath(tracks_csv))
    print("  tracks_summary :", os.path.abspath(tracks_summary_csv))
    return tracks_csv, tracks_summary_csv, tracked_video


# ============================================================
# =========================== Action ==========================
# ============================================================

@dataclass
class Box:
    x1: int; y1: int; x2: int; y2: int
    def clip(self, w: int, h: int) -> "Box":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 <= x1: x2 = min(w - 1, x1 + 1)
        if y2 <= y1: y2 = min(h - 1, y1 + 1)
        return Box(x1, y1, x2, y2)
    def w(self): return self.x2 - self.x1
    def h(self): return self.y2 - self.y1

def expand_box(box: Box, scale: float, w: int, h: int) -> Box:
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    bw = max(2.0, box.w() * scale)
    bh = max(2.0, box.h() * scale)
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))
    return Box(x1, y1, x2, y2).clip(w, h)

def resize_with_letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / w, size / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def uniform_indices(num_total: int, num_samples: int) -> List[int]:
    if num_total <= 0:
        return list(range(num_samples))
    if num_total <= num_samples:
        return list(range(num_total)) + [num_total - 1] * (num_samples - num_total)
    xs = np.linspace(0, num_total - 1, num_samples)
    return [int(round(x)) for x in xs]

class EffNetV2S_GRU(nn.Module):
    def __init__(self, num_classes: int, gru_hidden: int, gru_layers: int, bidir: bool, dropout: float):
        super().__init__()
        self.num_classes = int(num_classes)
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        base = torchvision.models.efficientnet_v2_s(weights=weights)

        self.encoder = base.features
        self.avgpool = base.avgpool
        feat_dim = 1280

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=int(gru_hidden),
            num_layers=int(gru_layers),
            batch_first=True,
            bidirectional=bool(bidir),
        )
        out_dim = int(gru_hidden) * (2 if bidir else 1)
        self.drop = nn.Dropout(float(dropout))
        self.fc = nn.Linear(out_dim, self.num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = feat.reshape(b, t, -1)
        out, _ = self.gru(feat)
        last = out[:, -1, :]
        last = self.drop(last)
        return self.fc(last)

def load_action_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到动作模型 ckpt：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    num_classes = int(cfg.get("NUM_CLASSES", 1))
    gru_hidden = int(cfg.get("GRU_HIDDEN", 256))
    gru_layers = int(cfg.get("GRU_LAYERS", 1))
    bidir = bool(cfg.get("GRU_BIDIR", False))
    dropout = float(cfg.get("DROPOUT", 0.2))

    model = EffNetV2S_GRU(num_classes, gru_hidden, gru_layers, bidir, dropout)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, cfg

def load_tracks_csv(tracks_csv: str) -> Dict[int, Dict[int, Box]]:
    """
    读取 tracks.csv，返回：
      tracks[track_id][frame_id] = Box(...)
    ✅ 这里做了 BOM/utf-8-sig 兼容：不会再出现 '\ufeffframe_id' 这种列名问题
    """
    tracks: Dict[int, Dict[int, Box]] = {}

    with open(tracks_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [strip_bom(x) for x in reader.fieldnames]
        cols = reader.fieldnames or []

        need = ["frame_id", "track_id", "x1", "y1", "x2", "y2"]
        for k in need:
            if k not in cols:
                raise RuntimeError(f"tracks.csv 缺少列：{k}，当前列={cols}")

        for row in reader:
            row = {strip_bom(k): v for k, v in row.items()}
            fi = int(float(row["frame_id"]))
            tid = int(float(row["track_id"]))
            x1 = int(float(row["x1"]))
            y1 = int(float(row["y1"]))
            x2 = int(float(row["x2"]))
            y2 = int(float(row["y2"]))
            tracks.setdefault(tid, {})[fi] = Box(x1, y1, x2, y2)

    return tracks

def load_summary_csv(summary_csv: str) -> Dict[int, Tuple[int, int]]:
    if not summary_csv or (not os.path.isfile(summary_csv)):
        return {}
    out = {}
    with open(summary_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [strip_bom(x) for x in reader.fieldnames]
        for row in reader:
            row = {strip_bom(k): v for k, v in row.items()}
            tid = int(float(row["track_id"]))
            s = int(float(row["start_frame"]))
            e = int(float(row["end_frame"]))
            out[tid] = (s, e)
    return out

def rois_to_tensor(rois_bgr: List[np.ndarray], num_frames: int, frame_size: int) -> torch.Tensor:
    idxs = uniform_indices(len(rois_bgr), num_frames)
    picked = [rois_bgr[i] if i < len(rois_bgr) else rois_bgr[-1] for i in idxs]
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in picked]
    rgb = [cv2.resize(im, (frame_size, frame_size), interpolation=cv2.INTER_AREA) for im in rgb]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frames = []
    for im in rgb:
        x = im.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        frames.append(torch.from_numpy(x))
    return torch.stack(frames, dim=0).unsqueeze(0)  # (1,T,3,H,W)

@torch.no_grad()
def infer_action(model: nn.Module, x: torch.Tensor, num_classes: int, device: torch.device) -> Tuple[List[float], int]:
    x = x.to(device, non_blocking=True)
    logits = model(x)  # (1,C)
    if num_classes == 1:
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= ACTION_THRESH else 0
        return [float(prob)], int(pred)
    prob = torch.softmax(logits.squeeze(0), dim=0).detach().cpu().numpy().tolist()
    pred = int(np.argmax(prob))
    return [float(p) for p in prob], pred


# ==========================
# Action 附加：视频输出工具
# ==========================

def _text_for_pred(num_classes: int, pred: int, probs: List[float]) -> Tuple[str, str]:
    """
    返回 (label_name, prob_txt)
    - num_classes==1: probs=[prob_1]，pred为0/1
    - num_classes>1: probs为softmax列表
    """
    label_name = ACTION_LABEL_NAMES.get(int(pred), f"class_{int(pred)}")

    prob_txt = ""
    if num_classes == 1 and len(probs) >= 1:
        prob_txt = f"{float(probs[0]):.2f}"
    elif num_classes > 1 and 0 <= int(pred) < len(probs):
        prob_txt = f"{float(probs[int(pred)]):.2f}"

    return label_name, prob_txt

def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.merge([img, img, img])
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    return img

def _draw_simple_text(img: np.ndarray, text: str, org: Tuple[int, int],
                      scale: float = 0.6, thick: int = 2,
                      color: Tuple[int, int, int] = (0, 255, 255)):
    put_text_with_bg(img, text, org, float(scale), int(thick),
                     text_color=color, bg=False, bg_alpha=0.0)

def _build_frame_to_tracks(tracks: Dict[int, Dict[int, Box]]) -> Dict[int, List[Tuple[int, Box]]]:
    out: Dict[int, List[Tuple[int, Box]]] = {}
    for tid, fmap in tracks.items():
        for fi, b in fmap.items():
            out.setdefault(int(fi), []).append((int(tid), b))
    return out

def make_proc_action_overlay_video(video_path: str,
                                   out_video_path: str,
                                   sector_mask: Optional[np.ndarray],
                                   tracks: Dict[int, Dict[int, Box]],
                                   pred_text_by_tid: Dict[int, str]) -> str:
    """
    输出：proc底图（原视频逐帧 preprocess） + 轨迹框 + 类别文字
    注意：这里的 bbox 来自 tracks.csv，坐标系是原视频坐标（RESIZE_TO=None 时严格一致）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ProcOverlay] 无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 7.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vw = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"[ProcOverlay] VideoWriter 打开失败：{out_video_path}")

    frame2tracks = _build_frame_to_tracks(tracks)

    pbar = tqdm(total=total_frames if total_frames > 0 else None, ncols=110, desc="ProcOverlay")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_work = frame
        if RESIZE_TO is not None:
            frame_work = cv2.resize(frame_work, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

        if sector_mask is None:
            # diff 输入：不再做 preprocess，直接用输入帧作为底图
            proc_vis = _ensure_3ch(frame_work).copy()
        else:
            frame_work = _apply_rect_masks(frame_work, MASK_RECTS)
            proc = preprocess_frame(frame_work, sector_mask)
            proc_vis = _ensure_3ch(proc).copy()

        # 如果 preprocess 产生的尺寸与原视频不一致，强制拉回（一般 RESIZE_TO=None 不会发生）
        if proc_vis.shape[1] != W or proc_vis.shape[0] != H:
            proc_vis = cv2.resize(proc_vis, (W, H), interpolation=cv2.INTER_LINEAR)

        items = frame2tracks.get(frame_id, [])
        for tid, b in items:
            bb = b.clip(W, H)
            x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
            cv2.rectangle(proc_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = pred_text_by_tid.get(int(tid), "")
            if text:
                tx = int(x1)
                ty = int(y1) - 8
                if ty < 18:
                    ty = int(y1) + 18
                # 这里复用你 Stage4 的“黑底半透明文字”风格
                put_text_with_bg(
                    proc_vis, text, (tx, ty),
                    font_scale=float(ACTION_OVERLAY_TEXT_SCALE),
                    thickness=int(ACTION_OVERLAY_TEXT_THICK),
                    text_color=ACTION_OVERLAY_TEXT_COLOR,
                    bg=bool(ACTION_OVERLAY_BG),
                    bg_alpha=float(ACTION_OVERLAY_BG_ALPHA),
                )

        vw.write(proc_vis)
        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    vw.release()

    print("\n[ProcOverlay] done")
    print("  proc_action_overlay.mp4:", os.path.abspath(out_video_path))
    return out_video_path


def action_stage(video_path: str, tracks_csv: str, summary_csv: str, out_dir: str,
                 input_mode: Optional[str] = None) -> str:
    """
    输出：
      - pred_tracks.csv（每条 track 的分类结果）
      - 可选 roi_clips_pred/
      - ✅ roi_preview.mp4（preprocess后裁ROI，总览视频）   [RUN_ACTION_ROI_PREVIEW_VIDEO=True]
      - ✅ proc_action_overlay.mp4（proc底图识别视频）      [RUN_ACTION_PROC_OVERLAY_VIDEO=True]
    返回：pred_tracks.csv 的路径（给下一步“视频叠字”使用）
    """
    ensure_dir(out_dir)
    out_pred_csv = os.path.join(out_dir, "pred_tracks.csv")
    roi_dir = os.path.join(out_dir, "roi_clips_pred")
    if EXPORT_ROI_CLIPS:
        ensure_dir(roi_dir)

    device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    model, cfg = load_action_model(ACTION_CKPT, device)
    num_classes = int(cfg.get("NUM_CLASSES", 1))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Action] 无法打开视频：{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 7.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    mode = (input_mode or ACTION_INPUT_MODE)
    mode = str(mode).lower().strip()

    # sector mask（只有 raw 输入才需要）
    sector_mask = None
    if mode != "diff":
        cap0 = cv2.VideoCapture(video_path)
        sector_mask = build_sector_mask_from_video_first_frame(cap0)
        cap0.release()

    tracks = load_tracks_csv(tracks_csv)
    summary = load_summary_csv(summary_csv)

    # ✅ 记录每条track的“显示文本”，供 proc_overlay / tracked_overlay_overlay 使用
    pred_text_by_tid: Dict[int, str] = {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # ✅ ROI preview 总览视频 writer
    roi_preview_path = os.path.join(out_dir, ACTION_ROI_PREVIEW_OUT_NAME)
    vw_roi_preview = None
    if RUN_ACTION_ROI_PREVIEW_VIDEO:
        vw_roi_preview = cv2.VideoWriter(
            roi_preview_path,
            fourcc,
            float(ROI_CLIPS_FPS),
            (int(ACTION_FRAME_SIZE), int(ACTION_FRAME_SIZE))
        )
        if not vw_roi_preview.isOpened():
            vw_roi_preview = None

    with open(out_pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["track_id", "start_frame", "end_frame", "len_frames", "pred_label"]
        if num_classes == 1:
            header += ["prob_1"]
        else:
            header += [f"prob_{i}" for i in range(num_classes)]
        w.writerow(header)

        for tid in sorted(tracks.keys()):
            tf = tracks[tid]
            if tid in summary:
                start_f, end_f = summary[tid]
            else:
                fs = sorted(tf.keys())
                start_f, end_f = fs[0], fs[-1]

            start_f = max(0, int(start_f))
            end_f = max(start_f, int(end_f))
            if total_frames > 0:
                end_f = min(end_f, total_frames - 1)

            available = [fi for fi in range(start_f, end_f + 1) if fi in tf]
            if len(available) == 0:
                continue

            sample_idx = uniform_indices(len(available), int(ACTION_NUM_FRAMES))
            sample_frames = [available[i] if i < len(available) else available[-1] for i in sample_idx]

            cap2 = cv2.VideoCapture(video_path)
            if not cap2.isOpened():
                continue

            rois = []
            for fi in sample_frames:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap2.read()
                if not ok or frame is None:
                    rois.append(np.zeros((ACTION_FRAME_SIZE, ACTION_FRAME_SIZE, 3), dtype=np.uint8))
                    continue

                frame_work = frame
                if RESIZE_TO is not None:
                    frame_work = cv2.resize(frame_work, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

                # 输入模式：
                # - raw: 需要 preprocess_frame
                # - diff: 直接使用 diff 帧（不再做 preprocess）
                if mode == "diff":
                    proc = _ensure_3ch(frame_work)
                else:
                    frame_work = _apply_rect_masks(frame_work, MASK_RECTS)
                    proc = preprocess_frame(frame_work, sector_mask)

                b = tf.get(fi, None)
                if b is None:
                    rois.append(np.zeros((ACTION_FRAME_SIZE, ACTION_FRAME_SIZE, 3), dtype=np.uint8))
                    continue

                eb = expand_box(b, float(ACTION_BBOX_EXPAND), proc.shape[1], proc.shape[0])
                crop = proc[eb.y1:eb.y2, eb.x1:eb.x2].copy()
                roi = resize_with_letterbox(crop, int(ACTION_FRAME_SIZE))
                rois.append(roi)

            cap2.release()

            if len(rois) == 0:
                continue

            x = rois_to_tensor(rois, int(ACTION_NUM_FRAMES), int(ACTION_FRAME_SIZE))
            probs, pred = infer_action(model, x, num_classes, device)

            w.writerow([tid, start_f, end_f, len(available), pred] + probs)

            # ✅ 生成用于视频叠字的文本（label + prob）
            label_name, prob_txt = _text_for_pred(num_classes, int(pred), probs)
            if prob_txt != "":
                text = f"{label_name} ({prob_txt})"
            else:
                text = f"{label_name}"
            pred_text_by_tid[int(tid)] = text

            # ✅ per-track ROI clip（保留你的原逻辑）
            if EXPORT_ROI_CLIPS:
                out_clip = os.path.join(roi_dir, f"track_{tid:06d}_pred{pred}.mp4")
                vw = cv2.VideoWriter(out_clip, fourcc, float(ROI_CLIPS_FPS),
                                     (int(ACTION_FRAME_SIZE), int(ACTION_FRAME_SIZE)))
                if vw.isOpened():
                    for im in rois:
                        vw.write(im)
                    vw.release()

            # ✅ ROI preview 总览：把这条轨迹的采样ROI顺序写进去
            if vw_roi_preview is not None:
                for im in rois:
                    frame_roi = im.copy()
                    if ACTION_ROI_PREVIEW_DRAW_TEXT:
                        _draw_simple_text(frame_roi, f"tid={tid} {text}", (6, 20),
                                          scale=0.55, thick=2, color=(0, 255, 255))
                    vw_roi_preview.write(frame_roi)

            if num_classes == 1:
                print(f"[Action] track={tid} pred={pred} prob_1={probs[0]:.4f} frames={len(available)}")
            else:
                print(f"[Action] track={tid} pred={pred} probs={probs} frames={len(available)}")

    if vw_roi_preview is not None:
        vw_roi_preview.release()
        print("[Action] roi_preview.mp4:", os.path.abspath(roi_preview_path))

    # ✅ 生成 proc 底图识别视频（逐帧 preprocess + 画框 + 文字）
    if RUN_ACTION_PROC_OVERLAY_VIDEO:
        proc_overlay_path = os.path.join(out_dir, ACTION_PROC_OVERLAY_OUT_NAME)
        make_proc_action_overlay_video(
            video_path=video_path,
            out_video_path=proc_overlay_path,
            sector_mask=sector_mask,
            tracks=tracks,
            pred_text_by_tid=pred_text_by_tid
        )

    print("\n[Action] done")
    print("  pred_tracks.csv:", os.path.abspath(out_pred_csv))
    if EXPORT_ROI_CLIPS:
        print("  roi_clips_pred:", os.path.abspath(roi_dir))

    return out_pred_csv


# ============================================================
# =========== Stage4: 在 tracked_overlay 上叠加 Action 标签 ============
# ============================================================

def load_pred_tracks_csv(pred_csv: str) -> Dict[int, Dict[str, Any]]:
    """
    读取 pred_tracks.csv，返回：
      pred_by_tid[track_id] = {
        "pred_label": int,
        "probs": [..],
      }
    """
    df = pd.read_csv(pred_csv, encoding="utf-8-sig")
    df.columns = [strip_bom(c) for c in df.columns]
    out: Dict[int, Dict[str, Any]] = {}

    # 找出 prob_* 列
    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    for _, r in df.iterrows():
        tid = int(r["track_id"])
        pred = int(r["pred_label"])
        probs = [float(r[c]) for c in prob_cols] if len(prob_cols) > 0 else []
        out[tid] = {"pred_label": pred, "probs": probs, "prob_cols": prob_cols}
    return out

def put_text_with_bg(img: np.ndarray, text: str, org: Tuple[int, int],
                     font_scale: float, thickness: int,
                     text_color=(0, 255, 255),
                     bg=True, bg_alpha=0.55):
    """
    在图片上写字，并可选画一个半透明黑底提升可读性
    org: (x,y) 为文字左下角
    """
    # 优先用 PIL 绘字（支持中文）
    if PIL_AVAILABLE:
        x, y = int(org[0]), int(org[1])
        font_size = max(10, int(TEXT_FONT_SIZE_BASE * float(font_scale)))
        try:
            font = ImageFont.truetype(TEXT_FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        x1 = max(0, x - 2)
        y1 = max(0, y - th - 2)
        x2 = min(img.shape[1] - 1, x + tw + 2)
        y2 = min(img.shape[0] - 1, y + 2)

        if bg:
            overlay = img_pil.convert("RGBA")
            ov_draw = ImageDraw.Draw(overlay)
            ov_draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, int(255 * bg_alpha)))
            img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

        draw.text((x, y - th), text, font=font,
                  fill=(int(text_color[2]), int(text_color[1]), int(text_color[0])))

        img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return

    # 回退：OpenCV 字体（不支持中文）
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if bg:
        x1 = max(0, x - 2)
        y1 = max(0, y - th - baseline - 2)
        x2 = min(img.shape[1] - 1, x + tw + 2)
        y2 = min(img.shape[0] - 1, y + baseline + 2)

        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            overlay = roi.copy()
            overlay[:] = (0, 0, 0)
            cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, roi)

    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def action_overlay_stage(tracked_video_path: str,
                         tracks_csv: str,
                         pred_tracks_csv: str,
                         out_dir: str) -> str:
    """
    输入：
      - tracked_overlay.mp4（绿框轨迹视频）
      - tracks.csv（每帧 bbox）
      - pred_tracks.csv（每条 track 的分类结果）
    输出：
      - action_overlay.mp4（在绿框基础上写“类别/概率”）
    """
    ensure_dir(out_dir)
    out_video = os.path.join(out_dir, ACTION_OVERLAY_OUT_NAME)

    # 1) 读 tracks（按 frame 聚合，便于逐帧画字）
    df = pd.read_csv(tracks_csv, encoding="utf-8-sig")
    df.columns = [strip_bom(c) for c in df.columns]

    need_cols = ["frame_id", "track_id", "x1", "y1", "x2", "y2"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"[ActionOverlay] tracks.csv 缺少列 {c}，当前列={list(df.columns)}")

    # 建索引：frame_id -> rows(list)
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for _, r in df.iterrows():
        fid = int(r["frame_id"])
        tid = int(r["track_id"])
        by_frame.setdefault(fid, []).append({
            "track_id": tid,
            "x1": int(float(r["x1"])),
            "y1": int(float(r["y1"])),
            "x2": int(float(r["x2"])),
            "y2": int(float(r["y2"])),
        })

    # 2) 读 pred
    pred_by_tid = load_pred_tracks_csv(pred_tracks_csv)

    # 3) 打开 tracked_overlay.mp4，逐帧画字
    cap = cv2.VideoCapture(tracked_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ActionOverlay] 无法打开视频：{tracked_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 7.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"[ActionOverlay] VideoWriter 打开失败：{out_video}")

    pbar = tqdm(total=total_frames if total_frames > 0 else None, ncols=110, desc="ActionOverlay")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        rows = by_frame.get(frame_id, [])

        for rr in rows:
            tid = rr["track_id"]
            x1, y1, x2, y2 = rr["x1"], rr["y1"], rr["x2"], rr["y2"]

            info = pred_by_tid.get(tid, None)
            if info is None:
                continue

            pred = int(info["pred_label"])
            label_name = ACTION_LABEL_NAMES.get(pred, f"class_{pred}")

            probs = info.get("probs", [])
            prob_cols = info.get("prob_cols", [])

            prob_txt = ""
            if len(prob_cols) > 0 and len(probs) > 0:
                if "prob_1" in prob_cols:
                    idx = prob_cols.index("prob_1")
                    p1 = float(probs[idx])
                    prob_txt = f"{p1:.2f}"
                else:
                    if 0 <= pred < len(probs):
                        prob_txt = f"{float(probs[pred]):.2f}"

            text = f"{label_name}"
            if prob_txt != "":
                text += f" ({prob_txt})"

            tx = int(x1)
            ty = int(y1) - 8
            if ty < 18:
                ty = int(y1) + 18

            put_text_with_bg(
                frame, text, (tx, ty),
                font_scale=float(ACTION_OVERLAY_TEXT_SCALE),
                thickness=int(ACTION_OVERLAY_TEXT_THICK),
                text_color=ACTION_OVERLAY_TEXT_COLOR,
                bg=bool(ACTION_OVERLAY_BG),
                bg_alpha=float(ACTION_OVERLAY_BG_ALPHA),
            )

        vw.write(frame)
        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    vw.release()

    print("\n[ActionOverlay] done")
    print("  action_overlay.mp4:", os.path.abspath(out_video))
    return out_video


def action_overlay_on_video(base_video_path: str,
                            tracks_csv: str,
                            pred_tracks_csv: str,
                            out_dir: str,
                            out_name: str,
                            draw_boxes: bool = True) -> str:
    """
    在任意底图视频上叠加轨迹框+标签（支持中文字体）
    """
    ensure_dir(out_dir)
    out_video = os.path.join(out_dir, out_name)

    df = pd.read_csv(tracks_csv, encoding="utf-8-sig")
    df.columns = [strip_bom(c) for c in df.columns]

    need_cols = ["frame_id", "track_id", "x1", "y1", "x2", "y2"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"[ActionOverlay] tracks.csv 缺少列 {c}，当前列={list(df.columns)}")

    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for _, r in df.iterrows():
        fid = int(r["frame_id"])
        tid = int(r["track_id"])
        by_frame.setdefault(fid, []).append({
            "track_id": tid,
            "x1": int(float(r["x1"])),
            "y1": int(float(r["y1"])),
            "x2": int(float(r["x2"])),
            "y2": int(float(r["y2"])),
        })

    pred_by_tid = load_pred_tracks_csv(pred_tracks_csv)

    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ActionOverlay] 无法打开视频：{base_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 7.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"[ActionOverlay] VideoWriter 打开失败：{out_video}")

    pbar = tqdm(total=total_frames if total_frames > 0 else None, ncols=110, desc="ActionOverlay")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        vis = frame.copy()
        rows = by_frame.get(frame_id, [])
        for rr in rows:
            tid = rr["track_id"]
            x1, y1, x2, y2 = rr["x1"], rr["y1"], rr["x2"], rr["y2"]

            if draw_boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            info = pred_by_tid.get(tid, None)
            if info is None:
                continue

            pred = int(info["pred_label"])
            label_name = ACTION_LABEL_NAMES.get(pred, f"class_{pred}")
            probs = info.get("probs", [])
            prob_cols = info.get("prob_cols", [])

            prob_txt = ""
            if len(prob_cols) > 0 and len(probs) > 0:
                if "prob_1" in prob_cols:
                    idx = prob_cols.index("prob_1")
                    p1 = float(probs[idx])
                    prob_txt = f"{p1:.2f}"
                else:
                    p = float(probs[pred]) if pred < len(probs) else 0.0
                    prob_txt = f"{p:.2f}"

            text = label_name if prob_txt == "" else f"{label_name} ({prob_txt})"
            put_text_with_bg(vis, text, (x1, max(0, y1 - 6)),
                             font_scale=float(ACTION_OVERLAY_TEXT_SCALE),
                             thickness=int(ACTION_OVERLAY_TEXT_THICK),
                             text_color=ACTION_OVERLAY_TEXT_COLOR,
                             bg=bool(ACTION_OVERLAY_BG),
                             bg_alpha=float(ACTION_OVERLAY_BG_ALPHA))

        vw.write(vis)
        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    vw.release()

    print("\n[ActionOverlay] done")
    print("  action_overlay.mp4:", os.path.abspath(out_video))
    return out_video


# ============================================================
# =============================== MAIN ========================
# ============================================================

def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(YOLO_OUT_DIR)
    ensure_dir(TRACK_OUT_DIR)
    ensure_dir(ACTION_OUT_DIR)

    detections_csv = os.path.join(YOLO_OUT_DIR, "detections.csv")
    diff_base = os.path.join(YOLO_OUT_DIR, "diff_base.mp4")

    tracked_video = os.path.join(TRACK_OUT_DIR, "tracked_overlay.mp4")
    tracks_csv = os.path.join(TRACK_OUT_DIR, "tracks.csv")
    tracks_summary_csv = os.path.join(TRACK_OUT_DIR, "tracks_summary.csv")

    pred_tracks_csv = os.path.join(ACTION_OUT_DIR, "pred_tracks.csv")

    yolo_input = (YOLO_INPUT_VIDEO_PATH.strip() if isinstance(YOLO_INPUT_VIDEO_PATH, str) else "")
    yolo_input_mode = str(YOLO_INPUT_MODE).lower().strip()
    if not yolo_input:
        # 没有提供 proc 视频时，回退到原视频并用 raw 模式预处理
        yolo_input = VIDEO_PATH
        if yolo_input_mode == "proc":
            yolo_input_mode = "raw"
    track_input = (TRACK_INPUT_VIDEO_PATH.strip() if isinstance(TRACK_INPUT_VIDEO_PATH, str) else "")
    if not track_input:
        track_input = yolo_input
    action_input = (ACTION_INPUT_VIDEO_PATH.strip() if isinstance(ACTION_INPUT_VIDEO_PATH, str) else "")
    action_input_mode = str(ACTION_INPUT_MODE).lower().strip()
    if not action_input:
        action_input = VIDEO_PATH
        if action_input_mode == "diff":
            # 若未提供 diff 视频，则尝试使用 YOLO 阶段生成的 diff_base
            if RUN_YOLO and os.path.isfile(diff_base):
                action_input = diff_base
            else:
                # 没有 diff 时回退到 raw
                action_input_mode = "raw"

    if RUN_YOLO:
        detections_csv, diff_base_path = yolo_stage(yolo_input, YOLO_OUT_DIR, input_mode=yolo_input_mode)
        if diff_base_path is not None:
            diff_base = diff_base_path

    if RUN_TRACK:
        base_video = TRACK_DIFF_VIDEO_PATH.strip() if isinstance(TRACK_DIFF_VIDEO_PATH, str) else ""
        if base_video:
            if not os.path.isfile(base_video):
                raise RuntimeError(f"[Track] 指定的 TRACK_DIFF_VIDEO_PATH 不存在：{base_video}")
        else:
            base_video = track_input
        tracks_csv, tracks_summary_csv, tracked_video = tracking_stage(detections_csv, base_video, TRACK_OUT_DIR)

    if RUN_ACTION:
        pred_tracks_csv = action_stage(action_input, tracks_csv, tracks_summary_csv,
                                       ACTION_OUT_DIR, input_mode=action_input_mode)

    if RUN_ACTION_OVERLAY_VIDEO:
        if not os.path.isfile(pred_tracks_csv):
            raise RuntimeError(f"[ActionOverlay] 找不到 pred_tracks.csv：{pred_tracks_csv}")
        # 1) diff 底图 + 框 + 标签
        diff_video = diff_base if os.path.isfile(diff_base) else ""
        if not diff_video and action_input_mode == "diff" and os.path.isfile(action_input):
            diff_video = action_input
        if diff_video:
            action_overlay_on_video(diff_video, tracks_csv, pred_tracks_csv,
                                    ACTION_OUT_DIR, ACTION_OVERLAY_DIFF_OUT_NAME, draw_boxes=True)
        else:
            print("[ActionOverlay] ⚠️ 找不到 diff 视频，跳过 diff overlay 输出")

        # 2) 原视频 + 框 + 标签
        if os.path.isfile(VIDEO_PATH):
            action_overlay_on_video(VIDEO_PATH, tracks_csv, pred_tracks_csv,
                                    ACTION_OUT_DIR, ACTION_OVERLAY_RAW_OUT_NAME, draw_boxes=True)
        else:
            print("[ActionOverlay] ⚠️ 找不到原视频，跳过 raw overlay 输出")

    print("\n✅ ALL DONE.")
    print("OUT_ROOT:", os.path.abspath(OUT_ROOT))


if __name__ == "__main__":
    main()
