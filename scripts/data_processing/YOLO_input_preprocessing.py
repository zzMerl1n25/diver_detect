# -*- coding: utf-8 -*-
"""
输入预处理模块（声呐视频）- 批量处理 + 输出到文件夹版
=====================================================

目标
----
对“输入文件夹内所有声呐视频”逐个做统一预处理，并把结果写入“输出文件夹”：
1）proc：用于 YOLO 检测的“干净”图像（扇形区域 + 去掉 UI/字幕/小窗 + 强度归一化）
2）diff：（可选）帧差图 |I_t - I_{t-1}|，用于后续“动作/时序”模型输入
3）mask：每个视频保存一张 sector_mask.png，方便你核对扇形参数是否正确

说明（很重要）
--------------
✅ 不改你原来的预处理功能/逻辑（扇形mask、矩形遮罩、归一化、帧差都不动）
✅ 只新增一个“写盘层”：把生成器 yield 出来的结果写成视频文件/图片

流水线位置：
[视频帧] -> [预处理] -> (YOLO检测) -> (跟踪) -> (时序动作判别)

依赖
----
- Python 3.9+
- opencv-python (cv2)
- numpy
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================
# ---------------------- 超参数（集中在开头） ------------------
# ============================================================

# ---------- 输入输出（批量） ----------
VIDEO_DIR = os.path.join(ROOT_DIR, "data", "sonar_video")              # ✅ 输入：包含多个视频的文件夹（可递归）
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed_sonar_video")   # ✅ 输出：所有处理结果写到这里

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v")  # ✅ 支持的视频后缀（按需增删）

OUTPUT_FPS = 7                         # 下游处理的目标帧率（None 表示保持原视频帧率）
RESIZE_TO = None                       # 例如 (1280, 720)；None 表示保持原分辨率

# ---------- 输出格式（写盘） ----------
# 输出为视频文件时，一般建议统一用 mp4（兼容性好）
OUTPUT_VIDEO_EXT = ".mp4"              # 输出视频后缀（建议 .mp4）
OUTPUT_FOURCC = "mp4v"                 # mp4 常用 fourcc：mp4v（更通用）；也可试 "avc1"/"H264"（依赖环境）

# 是否同时输出“处理后视频(proc)”和“帧差视频(diff)”
WRITE_PROC_VIDEO = True
WRITE_DIFF_VIDEO = True                # 只有 DIFF_OUTPUT=True 且 WRITE_DIFF_VIDEO=True 才会写 diff

# 是否保存扇形 mask 图片（每个视频保存一张 sector_mask.png）
WRITE_SECTOR_MASK_PNG = True

# ---------- UI / 叠加层遮罩（把字幕、小窗等遮掉） ----------
MASK_RECTS = [
    # (0, 620, 1280, 720),
    # (860, 470, 1280, 720),
]

# ---------- 扇形有效区域（声呐扇形区域） ----------
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

# ---------- 强度归一化 ----------
USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

PCT_CLIP_LOW = 0.5
PCT_CLIP_HIGH = 99.5
EPS = 1e-6

# ---------- 输出给模型的帧格式 ----------
OUTPUT_GRAYSCALE = False
OUTPUT_3CH = True

DIFF_OUTPUT = True
DIFF_USE_PROCESSED_FRAME = True


# ============================================================
# ✅ 新增：降噪（声呐散斑噪声）超参数（集中在开头，方便你调）
# ============================================================
# 说明：
# - 声呐常见“散斑噪声(speckle)”会被 CLAHE 放大，因此建议：
#   先降噪 -> 再做强度归一化/CLAHE
# - 这里默认用 bilateral（保边平滑，效果/速度折中）
DENOISE_ENABLE = False                 # 是否启用降噪（建议先开着对比）
DENOISE_METHOD = "bilateral"           # "bilateral" | "nlm" | "median" | "gaussian" | "none"

# --- bilateral 参数（推荐优先调它） ---
# d: 邻域直径（越大越平滑，通常 5~11）
# sigmaColor: 颜色/强度域的平滑强度（越大越平滑，通常 15~60）
# sigmaSpace: 空间域的平滑强度（越大越平滑，通常 5~30）
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 10

# --- NLM 参数（更强但慢，离线做数据集可以） ---
# h: 降噪强度（越大越平滑，通常 6~20）
# templateWindowSize/searchWindowSize: 搜索窗口，越大越慢
NLM_H = 10
NLM_TEMPLATE_W = 7
NLM_SEARCH_W = 21

# --- median / gaussian（便宜但更容易糊边，仅做轻度） ---
MEDIAN_KSIZE = 3                       # 必须是奇数：3/5/7...
GAUSSIAN_KSIZE = 3                     # 必须是奇数：3/5/7...
GAUSSIAN_SIGMA = 0                     # 0 表示让 OpenCV 自动根据核大小估计


# ============================================================
# ---------------------- 代码实现（原功能不动） -----------------
# ============================================================

from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Iterable
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """一次输出结果：包括处理后的帧、可选帧差、以及扇形mask"""
    frame_idx: int
    timestamp_sec: float
    frame_proc: np.ndarray
    diff_proc: Optional[np.ndarray]
    sector_mask: np.ndarray


def _apply_rect_masks(frame_bgr: np.ndarray, rects: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    用矩形区域把 UI/字幕/小窗等叠加层直接遮掉（设为黑色）
    """
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
    """
    手动生成“扇形区域mask”（最推荐）
    """
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
    """
    自动估计扇形区域mask（备用方案）
    """
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
    """
    灰度强度归一化（百分位裁剪 + 归一化 + 可选CLAHE）
    """
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


# ============================================================
# ✅ 新增：降噪函数（不改你原有函数，只额外提供一个可选步骤）
# ============================================================
def _denoise_gray(gray_u8: np.ndarray) -> np.ndarray:
    """
    声呐图像降噪（主要针对散斑/颗粒噪声）

    设计原则：
    - 尽量“保边缘”：避免把蛙人轮廓/边界抹掉
    - 放在强度归一化/CLAHE 之前：因为 CLAHE 会把噪声一起增强

    支持方法：
    - bilateral：保边平滑，速度/效果折中（推荐默认）
    - nlm：Non-Local Means，效果更强但更慢（离线生成训练集可用）
    - median：中值滤波，便宜但容易糊边（仅轻度）
    - gaussian：高斯滤波，最便宜但最容易糊边（仅轻度）
    - none：不做任何处理
    """
    if (not DENOISE_ENABLE) or (DENOISE_METHOD is None):
        return gray_u8

    method = str(DENOISE_METHOD).lower().strip()

    # 统一保证输入是 uint8 灰度
    g = gray_u8
    if g.dtype != np.uint8:
        g = g.astype(np.uint8)

    if method == "none":
        return g

    if method == "bilateral":
        # bilateralFilter：在空间域 + 强度域同时做加权平均
        # - 空间近、强度相近的像素才会相互平滑 -> 能较好保留边缘
        return cv2.bilateralFilter(
            g,
            d=int(BILATERAL_D),
            sigmaColor=float(BILATERAL_SIGMA_COLOR),
            sigmaSpace=float(BILATERAL_SIGMA_SPACE),
        )

    if method == "nlm":
        # fastNlMeansDenoising：利用“相似块”做加权平均
        # - 对纹理型噪声往往比简单滤波更强
        # - 但速度慢，适合离线预处理生成训练数据
        return cv2.fastNlMeansDenoising(
            g,
            None,
            h=float(NLM_H),
            templateWindowSize=int(NLM_TEMPLATE_W),
            searchWindowSize=int(NLM_SEARCH_W),
        )

    if method == "median":
        # medianBlur：对椒盐类噪声强，对散斑只能缓解一点
        # ksize 必须是奇数
        k = int(MEDIAN_KSIZE)
        k = k if (k % 2 == 1) else (k + 1)
        k = max(3, k)
        return cv2.medianBlur(g, k)

    if method == "gaussian":
        # GaussianBlur：最便宜的平滑，但最容易糊边
        k = int(GAUSSIAN_KSIZE)
        k = k if (k % 2 == 1) else (k + 1)
        k = max(3, k)
        return cv2.GaussianBlur(g, (k, k), float(GAUSSIAN_SIGMA))

    # 如果 method 写错：直接原样返回，避免程序崩掉
    return g


def preprocess_frame(frame_bgr: np.ndarray, sector_mask: np.ndarray) -> np.ndarray:
    """
    单帧预处理核心函数（不改动功能）

    流程：
    1）扇形外置黑
    2）转灰度
    3）强度归一化
    4）按需输出灰度/三通道
    """
    masked = frame_bgr.copy()
    masked[sector_mask == 0] = 0

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # ✅ 新增：降噪（放在强度归一化之前，避免 CLAHE/拉伸放大散斑）
    # ============================================================
    gray = _denoise_gray(gray)

    gray = _normalize_intensity(gray)

    if OUTPUT_GRAYSCALE:
        if OUTPUT_3CH:
            return cv2.merge([gray, gray, gray])
        return gray

    return masked


def iter_preprocessed_video(video_path: str) -> Generator[PreprocessResult, None, None]:
    """
    单视频：逐帧读取并输出预处理结果（生成器）

    ✅ 不改动功能：仍然只负责“读->预处理->yield”
    ✅ 写盘不在这里做（写盘在外层 batch writer 做）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    if OUTPUT_FPS is None or OUTPUT_FPS <= 0 or src_fps <= 0:
        step = 1
        out_fps = src_fps if src_fps > 0 else 0.0
    else:
        step = max(1, int(round(src_fps / float(OUTPUT_FPS))))
        out_fps = src_fps / step if src_fps > 0 else float(OUTPUT_FPS)

    prev_proc_gray_for_diff: Optional[np.ndarray] = None
    read_idx = 0
    emit_idx = 0

    ok, first = cap.read()
    if not ok:
        cap.release()
        return

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

    proc = preprocess_frame(first, sector_mask)

    if DIFF_OUTPUT and DIFF_USE_PROCESSED_FRAME:
        if proc.ndim == 3:
            prev_proc_gray_for_diff = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        else:
            prev_proc_gray_for_diff = proc.copy()

    yield PreprocessResult(
        frame_idx=emit_idx,
        timestamp_sec=0.0,
        frame_proc=proc,
        diff_proc=None,
        sector_mask=sector_mask,
    )
    emit_idx += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_idx += 1

        if (read_idx % step) != 0:
            continue

        if RESIZE_TO is not None:
            frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

        frame = _apply_rect_masks(frame, MASK_RECTS)
        proc = preprocess_frame(frame, sector_mask)

        diff_img = None
        if DIFF_OUTPUT and DIFF_USE_PROCESSED_FRAME:
            if proc.ndim == 3:
                proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            else:
                proc_gray = proc

            if prev_proc_gray_for_diff is not None:
                diff_img = cv2.absdiff(proc_gray, prev_proc_gray_for_diff)

            prev_proc_gray_for_diff = proc_gray.copy()

        t_sec = emit_idx / (out_fps + EPS) if out_fps > 0 else 0.0

        yield PreprocessResult(
            frame_idx=emit_idx,
            timestamp_sec=t_sec,
            frame_proc=proc,
            diff_proc=diff_img,
            sector_mask=sector_mask,
        )
        emit_idx += 1

    cap.release()


# ============================================================
# ---------------------- 批量读视频：只负责枚举文件 -------------
# ============================================================

def iter_video_files(video_dir: str, exts: Tuple[str, ...] = VIDEO_EXTS) -> Iterable[str]:
    """
    遍历文件夹内所有视频文件（递归）

    参数：
        video_dir: 输入视频文件夹
        exts:      允许的视频后缀（小写比较）

    产出：
        每个视频的完整路径
    """
    for root, _, files in os.walk(video_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                yield os.path.join(root, fn)


# ============================================================
# ---------------------- ✅ 写盘层：把结果输出到输出文件夹 --------
# ============================================================

def _ensure_dir(p: str) -> None:
    """确保目录存在：不存在就创建"""
    os.makedirs(p, exist_ok=True)


def _make_video_writer(out_path: str, fps: float, frame_wh: Tuple[int, int]) -> cv2.VideoWriter:
    """
    创建 OpenCV VideoWriter

    参数：
        out_path: 输出视频路径
        fps:      输出帧率（写文件时必须给一个数）
        frame_wh: (W, H) 注意顺序：VideoWriter 需要 (width, height)

    返回：
        writer: 可写入 BGR 3通道帧 的 VideoWriter

    注意：
        - 大多数编码器/容器对“单通道灰度视频”支持不好
        - 所以这里默认写 3通道（BGR），灰度图要先转成3通道再写
    """
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)
    w, h = frame_wh
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter 打开失败：{out_path}\n"
            f"可能原因：fourcc={OUTPUT_FOURCC} 不支持。可以尝试：mp4v / avc1 / XVID"
        )
    return writer


def _to_3ch_bgr(img: np.ndarray) -> np.ndarray:
    """
    把输入图像变成“可写视频的 3通道 BGR uint8”

    情况处理：
    - 如果已经是 (H,W,3) uint8：直接返回
    - 如果是 (H,W) 灰度：复制成3通道
    - 如果不是 uint8：这里强制转 uint8（理论上你上游已经是uint8）
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        return cv2.merge([img, img, img])

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    # 理论上不会走到这里（除非上游输出了奇怪的通道数）
    raise ValueError(f"无法转换为3通道：shape={img.shape}")


def batch_process_folder_to_output(input_dir: str, output_dir: str) -> None:
    """
    批量处理入口：把 input_dir 内所有视频逐个预处理，并输出到 output_dir

    输出目录结构（默认）：
    output_dir/
      video_001_name/
        proc.mp4              （预处理后图像序列写成视频）
        diff.mp4              （可选：帧差序列写成视频）
        sector_mask.png       （扇形mask，方便核对参数）
      video_002_name/
        ...

    设计说明：
    - 每个视频一个子文件夹，避免不同视频同名覆盖，也便于管理
    - proc/diff 都写成 mp4：便于快速播放检查和后续抽帧/训练
    """
    _ensure_dir(output_dir)

    # 遍历所有输入视频
    for video_path in iter_video_files(input_dir, VIDEO_EXTS):
        vp = Path(video_path)
        video_stem = vp.stem  # 文件名（不含后缀）
        out_subdir = os.path.join(output_dir, video_stem)
        _ensure_dir(out_subdir)

        # 约定输出文件名
        proc_out_path = os.path.join(out_subdir, f"proc{OUTPUT_VIDEO_EXT}")
        diff_out_path = os.path.join(out_subdir, f"diff{OUTPUT_VIDEO_EXT}")
        mask_out_path = os.path.join(out_subdir, "sector_mask.png")

        # -----------------------
        # writer 延迟初始化（拿到第一帧后才知道尺寸）
        # -----------------------
        proc_writer: Optional[cv2.VideoWriter] = None
        diff_writer: Optional[cv2.VideoWriter] = None

        # 由于 iter_preprocessed_video 内部计算 out_fps，但没直接返回
        # 这里我们用一个“可接受”的策略：
        # - 如果 OUTPUT_FPS 有设置：写盘用 OUTPUT_FPS
        # - 否则：尝试从原视频读取 src_fps；若失败则用 7
        if OUTPUT_FPS is not None and OUTPUT_FPS > 0:
            write_fps = float(OUTPUT_FPS)
        else:
            cap_tmp = cv2.VideoCapture(video_path)
            src_fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 0.0
            cap_tmp.release()
            write_fps = float(src_fps) if src_fps > 0 else 7.0

        # -----------------------
        # 开始逐帧处理并写盘
        # -----------------------
        print(f"\n[Batch] Processing video: {video_path}")
        first_item: Optional[PreprocessResult] = None

        for item in iter_preprocessed_video(video_path):
            # 记录第一帧（用于保存mask、初始化writer等）
            if first_item is None:
                first_item = item

                # 1) 保存扇形mask（每个视频一张，调参必备）
                if WRITE_SECTOR_MASK_PNG:
                    cv2.imwrite(mask_out_path, item.sector_mask)

                # 2) 初始化 proc_writer（如果需要写 proc）
                if WRITE_PROC_VIDEO:
                    proc_bgr = _to_3ch_bgr(item.frame_proc)
                    h, w = proc_bgr.shape[:2]
                    proc_writer = _make_video_writer(proc_out_path, write_fps, (w, h))

                # 3) 初始化 diff_writer（如果需要写 diff）
                # 注意：第一帧 diff_proc=None，所以 writer 可以先不创建，
                # 但为了统一，我们用“帧尺寸跟proc一致”的策略创建。
                if WRITE_DIFF_VIDEO and DIFF_OUTPUT:
                    # diff 也按 proc 的尺寸写（最稳），帧差图会被转成3通道写入
                    proc_bgr = _to_3ch_bgr(item.frame_proc)
                    h, w = proc_bgr.shape[:2]
                    diff_writer = _make_video_writer(diff_out_path, write_fps, (w, h))

            # ---------- 写 proc ----------
            if WRITE_PROC_VIDEO and proc_writer is not None:
                proc_bgr = _to_3ch_bgr(item.frame_proc)
                proc_writer.write(proc_bgr)

            # ---------- 写 diff ----------
            # diff_proc 第一帧为 None：为了保持帧数对齐，这里写一张全黑diff帧
            if WRITE_DIFF_VIDEO and DIFF_OUTPUT and diff_writer is not None:
                if item.diff_proc is None:
                    # 用全黑图占位，确保 diff 视频帧数与 proc 一致
                    # 尺寸从 proc 推导（proc 已保证存在）
                    proc_bgr = _to_3ch_bgr(item.frame_proc)
                    black = np.zeros(proc_bgr.shape[:2], dtype=np.uint8)
                    diff_bgr = _to_3ch_bgr(black)
                else:
                    diff_bgr = _to_3ch_bgr(item.diff_proc)

                    # 理论上 diff 尺寸应与 proc 一致；若不一致，强制 resize 保证 writer 不崩
                    proc_bgr = _to_3ch_bgr(item.frame_proc)
                    ph, pw = proc_bgr.shape[:2]
                    dh, dw = diff_bgr.shape[:2]
                    if (dh != ph) or (dw != pw):
                        diff_bgr = cv2.resize(diff_bgr, (pw, ph), interpolation=cv2.INTER_NEAREST)

                diff_writer.write(diff_bgr)

            # 控制台进度（不影响功能）
            if item.frame_idx % 30 == 0:
                print(
                    f"  frame={item.frame_idx:6d}  t={item.timestamp_sec:8.2f}s",
                    end="\r"
                )

        # -----------------------
        # 收尾：释放 writer
        # -----------------------
        if proc_writer is not None:
            proc_writer.release()
        if diff_writer is not None:
            diff_writer.release()

        # 小结打印（清晰告诉你输出在哪）
        print(
            f"\n[Done] {video_path}\n"
            f"  -> output dir: {out_subdir}\n"
            f"  -> proc video : {proc_out_path if WRITE_PROC_VIDEO else '(disabled)'}\n"
            f"  -> diff video : {diff_out_path if (WRITE_DIFF_VIDEO and DIFF_OUTPUT) else '(disabled)'}\n"
            f"  -> mask png   : {mask_out_path if WRITE_SECTOR_MASK_PNG else '(disabled)'}"
        )


# ============================================================
# ---------------------- 主入口：批量处理并输出 -----------------
# ============================================================

if __name__ == "__main__":
    """
    运行说明：
    - 把 VIDEO_DIR 改成你的输入视频文件夹
    - 把 OUTPUT_DIR 改成你想输出到的文件夹
    - 运行后会在 OUTPUT_DIR 下生成“每个视频一个子文件夹”的处理结果
    """
    batch_process_folder_to_output(VIDEO_DIR, OUTPUT_DIR)
