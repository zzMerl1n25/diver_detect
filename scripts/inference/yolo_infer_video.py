# -*- coding: utf-8 -*-
"""
YOLO 视频推理脚本（推理前做proc预处理，输出diff视频并画BBOX）- 完整版
===========================================================

✅ 输入：原视频 + 权重
✅ 推理输入：proc帧（扇形mask + 去UI + 可选降噪 + 强度归一化逻辑保持一致）
✅ 输出底图：diff帧（|proc_gray(t) - proc_gray(t-1)|）
✅ 输出：
  - overlay_diff.mp4（diff底图+框）
  - detections.json / detections.csv（默认保存为“原视频坐标”）

✅ 兼容两类权重：
A) Ultralytics 标准权重：YOLO(best.pt).predict(...)
B) 纯 state_dict：YOLO(yaml) 外壳 + DetectionModel(cfg,nc=NC) 覆盖 y.model 再 load_state_dict
   -> 仍然用 Ultralytics 的 predict 做 decode+NMS（避免框都跑左上角）

依赖：
  pip install ultralytics opencv-python tqdm numpy
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================
# ---------------------- 超参数（全部集中在开头） --------------
# ============================================================

VIDEO_PATH = r"C:\Users\Administrator\Desktop\sonar_track\data\test_video\1.mp4"

WEIGHTS_PATH = r"C:\Users\Administrator\Desktop\sonar_track\YOLO_training\runs_yolo11_from_scratch\y11_from_images_img1280\weights\last.pt"
YOLO11_YAML = "ultralytics/cfg/models/11/yolo11.yaml"
NC = 1  # 你的类别数（一定要和训练一致）

IMGSZ = 1280
CONF_THRES = 0.20
IOU_THRES = 0.50
MAX_DET = 300

START_FRAME = 0
END_FRAME = -1
FRAME_STRIDE = 1

DEVICE = "cuda"  # "cuda" or "cpu"

OUTPUT_DIR = os.path.join(ROOT_DIR, "runs_infer")
OUTPUT_NAME = "infer_diff_overlay"

DRAW_LABEL = True
DRAW_CONF = True
LINE_THICKNESS = 2
FONT_SCALE = 0.6

# ============================================================
# 预处理参数（与你提供的预处理脚本保持一致）
# ============================================================

MASK_RECTS = [
    # (0, 620, 1280, 720),
    # (860, 470, 1280, 720),
]

USE_MANUAL_SECTOR = False
APEX_X = 640
APEX_Y = 710
RADIUS_MIN = 0
RADIUS_MAX = 720
ANGLE_LEFT_DEG = -35.0
ANGLE_RIGHT_DEG = 35.0
SECTOR_MASK_STEP_DEG = 0.2

USE_AUTO_SECTOR_IF_MANUAL_FALSE = True
AUTO_BLACK_THRESH = 8
AUTO_MORPH_KERNEL = 9
AUTO_KEEP_LARGEST_CONTOUR = True

USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

PCT_CLIP_LOW = 0.5
PCT_CLIP_HIGH = 99.5
EPS = 1e-6

OUTPUT_GRAYSCALE = False
OUTPUT_3CH = True

RESIZE_TO = None  # 例如 (1280, 720)；None 表示不resize（要和训练一致）

# 降噪
DENOISE_ENABLE = False
DENOISE_METHOD = "bilateral"  # "bilateral" | "nlm" | "median" | "gaussian" | "none"
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 10
NLM_H = 10
NLM_TEMPLATE_W = 7
NLM_SEARCH_W = 21
MEDIAN_KSIZE = 3
GAUSSIAN_KSIZE = 3
GAUSSIAN_SIGMA = 0


# ============================================================
# ---------------------- 实现代码 ------------------------------
# ============================================================

import json
import time
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm


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


# ------------------ 预处理：与你脚本一致 ------------------

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
        return cv2.bilateralFilter(
            g, d=int(BILATERAL_D),
            sigmaColor=float(BILATERAL_SIGMA_COLOR),
            sigmaSpace=float(BILATERAL_SIGMA_SPACE),
        )
    if method == "nlm":
        return cv2.fastNlMeansDenoising(
            g, None,
            h=float(NLM_H),
            templateWindowSize=int(NLM_TEMPLATE_W),
            searchWindowSize=int(NLM_SEARCH_W),
        )
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


# ------------------ 模型加载（关键：修复 nc=80 mismatch） ------------------

def load_model_auto(weights_path: str, yaml_path: str, device: str):
    """
    1) 尝试 YOLO(weights_path) 直接加载（标准 best.pt/last.pt）
    2) 如果失败：YOLO(yaml) 外壳 + DetectionModel(cfg,nc=NC) 覆盖 y.model 再 load_state_dict
       -> 仍然用 Ultralytics 的 predict 做 decode+NMS（避免框坐标乱）
    """
    device_t = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    from ultralytics import YOLO

    # A) 标准权重
    try:
        y = YOLO(weights_path)
        return ("ultralytics_yolo_weights", y, device_t)
    except Exception as e:
        print(f"[Warn] YOLO(weights) 直接加载失败，尝试 state_dict 方式。原因：{e}")

    # B) state_dict
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif is_state_dict_like(ckpt):
        sd = ckpt
    else:
        raise RuntimeError("无法识别权重格式：既不是标准 YOLO 权重，也不像 state_dict。")

    # YOLO 外壳（用于 predict 的 decode+NMS）
    y = YOLO(yaml_path)

    # 用 nc=NC 重建 DetectionModel 覆盖默认 nc=80 的模型
    from ultralytics.nn.tasks import DetectionModel
    y.model = DetectionModel(cfg=yaml_path, ch=3, nc=NC, verbose=False)

    missing, unexpected = y.model.load_state_dict(sd, strict=False)
    print(f"[Info] state_dict loaded. missing={len(missing)} unexpected={len(unexpected)}")

    y.model.to(device_t).eval()
    return ("ultralytics_yolo_state_dict", y, device_t)


# ------------------ 推理（统一用 Ultralytics 的 predict 返回 boxes） ------------------

def infer_one_frame_with_ultralytics_predict(yolo_model, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    不管权重来自哪条路径，统一用 Ultralytics 的 predict:
    - 内部会做 letterbox
    - 会解码输出
    - 会做 NMS
    - 返回 boxes.xyxy / conf / cls
    """
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
    if r.boxes is None:
        return dets

    xyxy = r.boxes.xyxy.detach().cpu().numpy() if r.boxes.xyxy is not None else np.zeros((0, 4))
    conf = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.zeros((0,))
    cls = r.boxes.cls.detach().cpu().numpy() if r.boxes.cls is not None else np.zeros((0,))

    for i in range(len(xyxy)):
        dets.append({
            "cls": int(cls[i]),
            "conf": float(conf[i]),
            "xyxy": [float(x) for x in xyxy[i].tolist()]
        })
    return dets


# ------------------ 画框 ------------------

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

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)

        if DRAW_LABEL or DRAW_CONF:
            label = ""
            if DRAW_LABEL:
                label += f"{d['cls']}"
            if DRAW_CONF:
                label += ("" if label == "" else " ")
                label += f"{d['conf']:.2f}"
            if label:
                cv2.putText(
                    img_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 2, cv2.LINE_AA
                )


# ------------------ 坐标还原（如果 RESIZE_TO 改过尺寸） ------------------

def map_dets_to_original(dets: List[Dict[str, Any]], sx: float, sy: float) -> List[Dict[str, Any]]:
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return dets
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        out.append({
            "cls": d["cls"],
            "conf": d["conf"],
            "xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        })
    return out


# ------------------ diff 底图（基于 proc_gray） ------------------

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


# ============================================================
# main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    out_video_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_overlay_diff.mp4")
    out_json_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_detections.json")
    out_csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_detections.csv")

    mode, yolo, device_t = load_model_auto(WEIGHTS_PATH, YOLO11_YAML, DEVICE)
    print(f"[Info] Model load mode = {mode} | device = {device_t}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        fps = 7.0
    if w0 <= 0 or h0 <= 0:
        raise RuntimeError("视频分辨率读取失败。")

    # 构建扇形mask（按你预处理脚本：用第0帧构建）
    sector_mask = build_sector_mask_from_video_first_frame(cap)

    # 输出视频尺寸：diff 的尺寸 =（resize后）proc_gray 尺寸
    if RESIZE_TO is not None:
        out_w, out_h = int(RESIZE_TO[0]), int(RESIZE_TO[1])
    else:
        out_w, out_h = w0, h0

    out_fps = fps / max(1, FRAME_STRIDE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, float(out_fps), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter 打开失败：{out_video_path}")

    if END_FRAME < 0:
        end_frame = total_frames - 1 if total_frames > 0 else 10**18
    else:
        end_frame = END_FRAME

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frame_id = START_FRAME

    output = {
        "video_path": VIDEO_PATH,
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

    prev_proc_gray: Optional[np.ndarray] = None

    with open(out_csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("frame_id,time_sec,cls,conf,x1,y1,x2,y2\n")

        if total_frames > 0:
            n_proc = max(0, (min(end_frame, total_frames - 1) - START_FRAME) // max(1, FRAME_STRIDE) + 1)
        else:
            n_proc = None

        t0 = time.time()
        pbar = tqdm(total=n_proc, ncols=110, desc="Infer")

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

            # 1) resize（如需，与训练一致）
            if RESIZE_TO is not None:
                frame_work = cv2.resize(frame_work, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

            # 2) 去UI
            frame_work = _apply_rect_masks(frame_work, MASK_RECTS)

            # 3) proc（模型输入）
            frame_proc = preprocess_frame(frame_work, sector_mask)

            # 4) YOLO推理（基于proc），统一用 Ultralytics predict（解码+NMS）
            dets_proc = infer_one_frame_with_ultralytics_predict(yolo, frame_proc)

            # 5) diff 底图（基于 proc_gray）
            proc_gray = proc_to_gray_u8(frame_proc)
            if prev_proc_gray is None:
                diff_gray = np.zeros_like(proc_gray)
            else:
                diff_gray = cv2.absdiff(proc_gray, prev_proc_gray)
            prev_proc_gray = proc_gray

            diff_bgr = gray_to_3ch_bgr(diff_gray)

            # 6) 在 diff 上画框：用 dets_proc（坐标系一致：都是 resize 后的proc尺寸）
            vis = diff_bgr.copy()
            draw_boxes(vis, dets_proc)

            # 确保尺寸匹配 writer
            if vis.shape[1] != out_w or vis.shape[0] != out_h:
                vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            writer.write(vis)

            # 7) 保存检测结果（默认保存为“原视频坐标”）
            if RESIZE_TO is not None:
                sx = float(w0) / float(RESIZE_TO[0])
                sy = float(h0) / float(RESIZE_TO[1])
            else:
                sx = 1.0
                sy = 1.0

            dets_orig = map_dets_to_original(dets_proc, sx, sy)

            t_sec = frame_id / fps
            output["frames"].append({"frame_id": int(frame_id), "time_sec": float(t_sec), "dets": dets_orig})

            for d in dets_orig:
                x1, y1, x2, y2 = d["xyxy"]
                fcsv.write(f"{frame_id},{t_sec:.6f},{d['cls']},{d['conf']:.6f},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")

            # 进度
            dt = time.time() - t0
            proc_frames = len(output["frames"])
            fps_now = proc_frames / max(1e-6, dt)
            pbar.set_postfix(proc_fps=f"{fps_now:.2f}", dets=len(dets_proc), frame=frame_id)
            pbar.update(1)

            frame_id += 1

        pbar.close()

    cap.release()
    writer.release()

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n✅ 推理完成（输出diff底图+框）")
    print("Overlay diff video:", os.path.abspath(out_video_path))
    print("Detections JSON    :", os.path.abspath(out_json_path))
    print("Detections CSV     :", os.path.abspath(out_csv_path))


if __name__ == "__main__":
    main()
