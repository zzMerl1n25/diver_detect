# -*- coding: utf-8 -*-
"""
批量版（按子文件夹处理）：
YOLO推理 -> 人工逐帧检查/修改bbox + 逐帧打标签 -> 保存CSV -> 导出ROI全视频 + overlay -> 按标签导出训练clips
==========================================================================================================

输入结构（你的情况）：
processed_sonar_video/
  01/diff.mp4
  02/diff.mp4
  xxx/diff.mp4
  ...

输出：
OUT_CSV_DIR/
  01.csv
OUT_CROPPED_ALL_DIR/
  01_cropped_all.mp4
OUT_OVERLAY_ALL_DIR/
  01_overlay_all.mp4
OUT_LABELED_CLIPS_DIR/
  label_0/01_clip_000000_f000000-000034.mp4
  label_1/01_clip_000001_f000035-000069.mp4
  ...

交互说明：
- 空格/回车：下一帧
- b：上一帧
- r：重新跑YOLO并接受其框（覆盖当前框）
- n：当前帧设为“无目标”(清空bbox)
- s：保存进度
- u：撤销（Undo）当前帧最近一次修改（框/清空/标签）
- 数字键 0~9：设置当前帧 label（可扩展到更多标签，先用0~9足够）
- q / ESC：结束当前视频并保存 -> 继续下一个
- x：结束当前视频并保存 -> 停止后续批量

鼠标：
- 左键拖拽空白区域：画新框（替换当前框）
- 左键按住框内部拖动：移动框
- 右键：清空框（无目标）
"""

import os
import csv
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================
# ✅ 全部参数集中在这里
# ============================================================
CONFIG = {
    # -------- 输入根目录（包含很多子文件夹）--------
    # 每个子文件夹里应有 diff.mp4（或 DIFF_FILENAME 指定的名字）
    "IN_ROOT": os.path.join(PROJECT_ROOT, "data", "processed_sonar_video"),
    "DIFF_FILENAME": "diff.mp4",  # 在每个子文件夹里寻找的 diff 视频名
    "SKIP_IF_NO_DIFF": True,  # 若找不到 diff 视频是否跳过该子文件夹

    # 训练 clips 的视频来源（通常用 proc 而不是 diff）
    "PROC_FILENAME": "proc.mp4",  # 每个子文件夹里的 proc 视频名
    "CLIP_SOURCE": "proc",  # "proc" | "diff"

    # 从第几个子文件夹开始（两种任选一种）
    "START_INDEX_0BASED": 0,  # 0 表示从第一个子文件夹开始
    "START_INDEX_1BASED": None,  # 设为 None 表示不用（若填写将覆盖 0based）

    # -------- 输出目录（分开）--------
    "OUT_CSV_DIR": os.path.join(PROJECT_ROOT, "video_dataset_process", "csv"),  # 每个视频的逐帧标注 CSV
    "OUT_CROPPED_ALL_DIR": os.path.join(PROJECT_ROOT, "video_dataset_process", "cropped_all"),  # 全部 ROI 预览视频
    "OUT_OVERLAY_ALL_DIR": os.path.join(PROJECT_ROOT, "video_dataset_process", "overlay_all"),  # 全部 overlay 视频

    # ✅ 按标签导出训练 clips（推荐直接用它训练 EfficientNet+GRU）
    "EXPORT_LABELED_CLIPS": True,  # 是否导出分段 clips
    "OUT_LABELED_CLIPS_DIR": os.path.join(PROJECT_ROOT, "video_dataset_process", "labeled_clips"),
    "CLIP_SECONDS": 5.0,  # 每个 clip 的时长（秒）
    "STRIDE_SECONDS": 5.0,  # 相邻 clips 间隔（=5 表示不重叠；=1 表示滑窗）
    "DROP_LAST_SHORT": True,  # 不足一个 clip 时长的尾段是否丢弃
    "CLIP_LABEL_RULE": "majority",  # "majority" | "all_same"：如何决定 clip 标签
    "SKIP_UNLABELED_CLIPS": True,  # True: clip里label全是-1或多数为-1就跳过

    # -------- YOLO 权重 --------
    "YOLO_WEIGHTS": os.path.join(PROJECT_ROOT, "YOLO_training", "runs_yolo11_from_scratch", "y11_from_images_img1280", "weights", "best.pt"),
    "YOLO_MODEL_YAML": "ultralytics/cfg/models/11/yolo11.yaml",  # state_dict 加载时用
    "NC": 1,  # 类别数（必须与训练一致）

    # -------- YOLO推理参数 --------
    "IMGSZ": 1280,  # 推理输入尺寸（与训练一致更稳）
    "CONF_THRES": 0.25,  # 置信度阈值（越低召回高但噪声多）
    "IOU_THRES": 0.50,  # NMS IoU 阈值（越低抑制越强）
    "MAX_DET": 50,  # 单帧最大检测框数
    "DEVICE": "cuda",  # "cuda" or "cpu"

    # -------- 导出ROI参数 --------
    "ROI_SIZE": 224,  # ROI 输出尺寸（与动作模型输入一致）
    "BBOX_EXPAND": 1.5,  # bbox 扩张倍率（给目标留边）
    "EXPORT_OVERLAY": True,  # 是否输出带框 overlay 视频

    # -------- UI/操作 --------
    "MOVE_STEP": 2,  # 键盘移动框时每步像素
    "WINDOW_NAME": "YOLO Review (space next / b prev / r rerun / n none / s save / u undo / 0-9 label / q next / x stop_all)",

    # -------- 标签显示（可扩展）--------
    # 实际保存的是数字label；这里仅用于UI显示
    "LABEL_NAMES": {
        -1: "UNLABELED",
         0: "NOT_DIVER",
         1: "DIVER",
         2: "LABEL_2",
         3: "LABEL_3",
         4: "LABEL_4",
         5: "LABEL_5",
         6: "LABEL_6",
         7: "LABEL_7",
         8: "LABEL_8",
         9: "LABEL_9",
    },
}


# ============================================================
# 工具：文件夹扫描
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_subfolders(root: str) -> List[str]:
    subs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            subs.append(p)
    return sorted(subs)


def safe_id_from_folder(folder_path: str) -> str:
    name = os.path.basename(folder_path.rstrip("/\\"))
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip()
    return name if name else "unnamed"


# ============================================================
# 权重判别 + 加载（支持纯 state_dict）
# ============================================================
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


def load_model_auto(weights_path: str, yaml_path: str, nc: int, device: str):
    device_t = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # A) 标准 Ultralytics checkpoint
    try:
        y = YOLO(weights_path)
        try:
            y.model.to(device_t).eval()
        except Exception:
            pass
        print("[OK] Loaded as Ultralytics checkpoint:", weights_path)
        return y
    except Exception as e:
        print(f"[Warn] YOLO(weights) load failed -> try state_dict. reason: {type(e).__name__}: {e}")

    ckpt = torch.load(weights_path, map_location="cpu")

    if is_state_dict_like(ckpt):
        sd = ckpt
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    else:
        raise RuntimeError("无法识别权重格式：既不是标准 YOLO 权重，也不像 state_dict。")

    y = YOLO(yaml_path)

    from ultralytics.nn.tasks import DetectionModel
    y.model = DetectionModel(cfg=yaml_path, ch=3, nc=nc, verbose=False)

    missing, unexpected = y.model.load_state_dict(sd, strict=True)
    print("[OK] Loaded state_dict into DetectionModel.")
    print(f"     missing={len(missing)} unexpected={len(unexpected)}")

    y.model.to(device_t).eval()
    return y


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0
    cls: int = 0

    def clip(self, w: int, h: int) -> "Box":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return Box(x1, y1, x2, y2, self.conf, self.cls)

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1


def expand_box(box: Box, scale: float, w: int, h: int) -> Box:
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    bw = box.width() * scale
    bh = box.height() * scale
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))
    return Box(x1, y1, x2, y2, box.conf, box.cls).clip(w, h)


def resize_with_letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
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


def draw_box(img: np.ndarray, box: Box, color=(0, 255, 0), thickness=2, label: str = "") -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
    if label:
        cv2.putText(out, label, (box.x1, max(0, box.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


# ============================================================
# CSV：保存 bbox + label（逐帧）
# ============================================================
def load_progress_csv(path: str) -> Tuple[Dict[int, Optional[Box]], Dict[int, int], int]:
    """
    返回：
      boxes_by_frame[fi] = Box or None
      labels_by_frame[fi] = int label（默认 -1）
      last_frame = csv里最大frame_idx
    """
    if not os.path.exists(path):
        return {}, {}, -1

    boxes_by_frame: Dict[int, Optional[Box]] = {}
    labels_by_frame: Dict[int, int] = {}
    last_frame = -1

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_idx"])
            last_frame = max(last_frame, fi)

            label = int(row.get("label", "-1"))
            labels_by_frame[fi] = label

            has = int(row.get("has_box", "1"))
            if has == 0:
                boxes_by_frame[fi] = None
                continue

            boxes_by_frame[fi] = Box(
                x1=int(row["x1"]), y1=int(row["y1"]),
                x2=int(row["x2"]), y2=int(row["y2"]),
                conf=float(row.get("conf", "1.0")),
                cls=int(row.get("cls", "0")),
            )

    return boxes_by_frame, labels_by_frame, last_frame


def save_progress_csv(path: str, total_frames: int, boxes_by_frame: Dict[int, Optional[Box]], labels_by_frame: Dict[int, int]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_idx", "label", "has_box", "x1", "y1", "x2", "y2", "conf", "cls"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fi in range(total_frames):
            label = int(labels_by_frame.get(fi, -1))
            box = boxes_by_frame.get(fi, None)
            if box is None:
                writer.writerow({"frame_idx": fi, "label": label, "has_box": 0, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "conf": 0.0, "cls": 0})
            else:
                writer.writerow({"frame_idx": fi, "label": label, "has_box": 1, "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf, "cls": box.cls})


# ============================================================
# 鼠标编辑框
# ============================================================
class BBoxEditor:
    def __init__(self):
        self.dragging_new = False
        self.dragging_move = False
        self.start_pt = (0, 0)
        self.last_pt = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        # param: {"box": Box|None, "img_shape": (h,w)}
        box: Optional[Box] = param["box"]
        h, w = param["img_shape"]

        if event == cv2.EVENT_RBUTTONDOWN:
            param["box"] = None
            self.dragging_new = False
            self.dragging_move = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pt = (x, y)
            self.last_pt = (x, y)
            if box is not None and (box.x1 <= x <= box.x2) and (box.y1 <= y <= box.y2):
                self.dragging_move = True
            else:
                self.dragging_new = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_new:
                x1, y1 = self.start_pt
                x2, y2 = x, y
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    param["box"] = Box(x1, y1, x2, y2, conf=1.0, cls=0).clip(w, h)

            elif self.dragging_move and box is not None:
                dx = x - self.last_pt[0]
                dy = y - self.last_pt[1]
                param["box"] = Box(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy, box.conf, box.cls).clip(w, h)
                self.last_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_new = False
            self.dragging_move = False


def yolo_best_box(model: YOLO, frame: np.ndarray) -> Optional[Box]:
    results = model.predict(
        source=frame,
        imgsz=CONFIG["IMGSZ"],
        conf=CONFIG["CONF_THRES"],
        iou=CONFIG["IOU_THRES"],
        max_det=CONFIG["MAX_DET"],
        device=CONFIG["DEVICE"],
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)

    best = int(np.argmax(conf))
    x1, y1, x2, y2 = xyxy[best].tolist()
    return Box(int(x1), int(y1), int(x2), int(y2), float(conf[best]), int(cls[best]))


# ============================================================
# 导出：按标签生成训练 clips
# ============================================================
def export_labeled_clips(
    folder_id: str,
    video_path: str,
    boxes_by_frame: Dict[int, Optional[Box]],
    labels_by_frame: Dict[int, int],
):
    if not CONFIG["EXPORT_LABELED_CLIPS"]:
        return

    out_root = CONFIG["OUT_LABELED_CLIPS_DIR"]
    ensure_dir(out_root)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SkipClips] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 7.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roi_size = int(CONFIG["ROI_SIZE"])

    frames_per_clip = max(1, int(round(float(fps) * float(CONFIG["CLIP_SECONDS"]))))
    stride_frames = max(1, int(round(float(fps) * float(CONFIG["STRIDE_SECONDS"]))))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def decide_clip_label(labs: List[int]) -> int:
        # labs length = written frames
        if CONFIG["CLIP_LABEL_RULE"] == "all_same":
            uniq = set(labs)
            if len(uniq) == 1:
                return int(next(iter(uniq)))
            return -9999  # invalid
        # majority
        counts = {}
        for x in labs:
            counts[x] = counts.get(x, 0) + 1
        # 多数票（若平票，取更大的count后按label值排序）
        best = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
        return int(best)

    clip_idx = 0
    start_frame = 0

    while True:
        if total_frames > 0:
            if CONFIG["DROP_LAST_SHORT"] and (start_frame + frames_per_clip > total_frames):
                break
            if (not CONFIG["DROP_LAST_SHORT"]) and (start_frame >= total_frames):
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, _ = cap.read()
        if not ok:
            break
        # 回退一帧（因为我们要从start_frame开始统一写）
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 先收集这段的 label 列表
        labs = []
        end_frame = start_frame + frames_per_clip - 1
        max_read = frames_per_clip

        # 读帧并写ROI到内存（避免重复解码太复杂，这里直接一次写）
        out_frames = []
        written = 0

        for fi in range(start_frame, start_frame + max_read):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # 这帧 label / bbox
            lab = int(labels_by_frame.get(fi, -1))
            labs.append(lab)

            box = boxes_by_frame.get(fi, None)
            if box is None:
                roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            else:
                h, w = frame.shape[:2]
                eb = expand_box(box, float(CONFIG["BBOX_EXPAND"]), w, h)
                crop = frame[eb.y1:eb.y2, eb.x1:eb.x2].copy()
                roi = resize_with_letterbox(crop, roi_size)
            out_frames.append(roi)
            written += 1

        if written <= 0:
            break

        if written < frames_per_clip and CONFIG["DROP_LAST_SHORT"]:
            break

        clip_label = decide_clip_label(labs[:written])

        # 跳过无效clip
        if clip_label == -9999:
            start_frame += stride_frames
            continue

        # 跳过未标注为主的clip
        if CONFIG["SKIP_UNLABELED_CLIPS"]:
            if clip_label == -1:
                start_frame += stride_frames
                continue

        label_dir = os.path.join(out_root, f"label_{clip_label}")
        ensure_dir(label_dir)

        out_name = f"{folder_id}_clip_{clip_idx:06d}_f{start_frame:06d}-{(start_frame+written-1):06d}.mp4"
        out_path = os.path.join(label_dir, out_name)

        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (roi_size, roi_size))
        if not writer.isOpened():
            print(f"[SkipClips] VideoWriter open failed: {out_path}")
            start_frame += stride_frames
            continue

        for fr in out_frames:
            writer.write(fr)
        writer.release()

        clip_idx += 1
        start_frame += stride_frames

    cap.release()
    print(f"[Clips] {folder_id} -> wrote clips to: {os.path.abspath(out_root)}")


# ============================================================
# 单视频处理：review + export
# 返回 stop_all(bool)
# ============================================================
def process_one_video(model: YOLO, folder_id: str, diff_path: str, proc_path: Optional[str]) -> bool:
    # 输出路径
    ensure_dir(CONFIG["OUT_CSV_DIR"])
    ensure_dir(CONFIG["OUT_CROPPED_ALL_DIR"])
    ensure_dir(CONFIG["OUT_OVERLAY_ALL_DIR"])

    csv_path = os.path.join(CONFIG["OUT_CSV_DIR"], f"{folder_id}.csv")
    out_crop_all = os.path.join(CONFIG["OUT_CROPPED_ALL_DIR"], f"{folder_id}_cropped_all.mp4")
    out_overlay_all = os.path.join(CONFIG["OUT_OVERLAY_ALL_DIR"], f"{folder_id}_overlay_all.mp4")

    cap = cv2.VideoCapture(diff_path)
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {diff_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total_frames <= 0 or vw <= 0 or vh <= 0:
        cap.release()
        print(f"[Skip] Bad meta: frames={total_frames} size=({vw},{vh}) video={diff_path}")
        return False

    boxes_by_frame, labels_by_frame, last_done = load_progress_csv(csv_path)
    start_frame = min(max(last_done + 1, 0), max(total_frames - 1, 0))

    editor = BBoxEditor()
    window_name = f"{CONFIG['WINDOW_NAME']} | {folder_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def read_frame(fi: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        return fr if ok else None

    frame_idx = start_frame
    goto_export = False
    stop_all = False

    # ✅ Undo：当前帧的历史栈（box,label）
    undo_stack: List[Tuple[Optional[Box], int]] = []

    def push_undo(cur_box: Optional[Box], cur_label: int):
        # 只记录少量，避免内存无限涨
        undo_stack.append((cur_box, cur_label))
        if len(undo_stack) > 50:
            undo_stack.pop(0)

    while True:
        frame = read_frame(frame_idx)
        if frame is None:
            break
        h, w = frame.shape[:2]

        cur_box = boxes_by_frame.get(frame_idx, None)
        if frame_idx not in boxes_by_frame:
            cur_box = yolo_best_box(model, frame)

        cur_label = int(labels_by_frame.get(frame_idx, -1))

        param = {"box": cur_box, "img_shape": (h, w)}
        cv2.setMouseCallback(window_name, editor.on_mouse, param)

        # 每进入一帧，清空undo（只做“当前帧撤销”）
        undo_stack = []

        while True:
            display = frame.copy()
            box = param["box"]

            lab_name = CONFIG["LABEL_NAMES"].get(cur_label, f"label={cur_label}")
            header = f"{folder_id}  fi={frame_idx}/{total_frames-1}  label={cur_label}({lab_name})"

            if box is not None:
                label_txt = f"{header}  conf={box.conf:.2f} cls={box.cls}"
                display = draw_box(display, box, color=(0, 255, 0), thickness=2, label=label_txt)
            else:
                cv2.putText(display, f"{header}  (NO BOX)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            # q/ESC：结束当前视频并保存 -> 继续下一个
            if key in [27, ord("q")]:
                boxes_by_frame[frame_idx] = param["box"]
                labels_by_frame[frame_idx] = cur_label
                save_progress_csv(csv_path, total_frames, boxes_by_frame, labels_by_frame)
                cv2.destroyWindow(window_name)
                cap.release()
                goto_export = True
                break

            # x：结束当前视频并保存 -> 停止后续批量
            if key == ord("x"):
                boxes_by_frame[frame_idx] = param["box"]
                labels_by_frame[frame_idx] = cur_label
                save_progress_csv(csv_path, total_frames, boxes_by_frame, labels_by_frame)
                cv2.destroyWindow(window_name)
                cap.release()
                goto_export = True
                stop_all = True
                break

            # 保存
            if key == ord("s"):
                boxes_by_frame[frame_idx] = param["box"]
                labels_by_frame[frame_idx] = cur_label
                save_progress_csv(csv_path, total_frames, boxes_by_frame, labels_by_frame)
                print(f"[Saved] {os.path.abspath(csv_path)}")
                continue

            # Undo
            if key == ord("u"):
                if len(undo_stack) > 0:
                    prev_box, prev_label = undo_stack.pop()
                    param["box"] = prev_box
                    cur_label = int(prev_label)
                continue

            # 重新跑YOLO
            if key == ord("r"):
                push_undo(param["box"], cur_label)
                param["box"] = yolo_best_box(model, frame)
                continue

            # 清空框
            if key == ord("n"):
                push_undo(param["box"], cur_label)
                param["box"] = None
                continue

            # 数字键打标签 0~9
            if ord("0") <= key <= ord("9"):
                push_undo(param["box"], cur_label)
                cur_label = int(chr(key))  # 0..9
                continue

            # 下一帧
            if key == ord(" ") or key == 13:
                boxes_by_frame[frame_idx] = param["box"]
                labels_by_frame[frame_idx] = cur_label
                frame_idx = min(frame_idx + 1, total_frames - 1)
                break

            # 上一帧
            if key == ord("b"):
                boxes_by_frame[frame_idx] = param["box"]
                labels_by_frame[frame_idx] = cur_label
                frame_idx = max(frame_idx - 1, 0)
                break

            # 微调框
            box = param["box"]
            if box is None:
                continue

            ms = int(CONFIG["MOVE_STEP"])

            if key in [ord("w"), ord("a"), ord("d"), ord("s"), ord("["), ord("]")]:
                push_undo(param["box"], cur_label)

            if key == ord("w"):
                param["box"] = Box(box.x1, box.y1 - ms, box.x2, box.y2 - ms, box.conf, box.cls).clip(w, h)
            elif key == ord("s"):
                param["box"] = Box(box.x1, box.y1 + ms, box.x2, box.y2 + ms, box.conf, box.cls).clip(w, h)
            elif key == ord("a"):
                param["box"] = Box(box.x1 - ms, box.y1, box.x2 - ms, box.y2, box.conf, box.cls).clip(w, h)
            elif key == ord("d"):
                param["box"] = Box(box.x1 + ms, box.y1, box.x2 + ms, box.y2, box.conf, box.cls).clip(w, h)
            elif key == ord("["):
                cx = (box.x1 + box.x2) / 2.0
                cy = (box.y1 + box.y2) / 2.0
                bw = max(2, int(box.width() * 0.95))
                bh = max(2, int(box.height() * 0.95))
                param["box"] = Box(int(cx - bw/2), int(cy - bh/2), int(cx + bw/2), int(cy + bh/2), box.conf, box.cls).clip(w, h)
            elif key == ord("]"):
                cx = (box.x1 + box.x2) / 2.0
                cy = (box.y1 + box.y2) / 2.0
                bw = int(box.width() * 1.05)
                bh = int(box.height() * 1.05)
                param["box"] = Box(int(cx - bw/2), int(cy - bh/2), int(cx + bw/2), int(cy + bh/2), box.conf, box.cls).clip(w, h)

        if goto_export:
            break

        if frame_idx == total_frames - 1:
            boxes_by_frame[frame_idx] = boxes_by_frame.get(frame_idx, param.get("box", None))
            labels_by_frame[frame_idx] = int(labels_by_frame.get(frame_idx, cur_label))
            save_progress_csv(csv_path, total_frames, boxes_by_frame, labels_by_frame)
            print(f"[Review] {folder_id} reached last frame. Auto-saved -> {os.path.abspath(csv_path)}")
            break

    # ==========================
    # 导出 cropped_all / overlay_all
    # ==========================
    cap = cv2.VideoCapture(diff_path)
    if not cap.isOpened():
        print(f"[SkipExport] Cannot reopen video: {diff_path}")
        return stop_all

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    used_fps = fps if fps and fps > 0 else 7.0

    roi_size = int(CONFIG["ROI_SIZE"])
    crop_writer = cv2.VideoWriter(out_crop_all, fourcc, float(used_fps), (roi_size, roi_size))
    overlay_writer = None
    if CONFIG["EXPORT_OVERLAY"]:
        overlay_writer = cv2.VideoWriter(out_overlay_all, fourcc, float(used_fps), (vw, vh))

    for fi in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        box = boxes_by_frame.get(fi, None)
        if box is None:
            roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            crop_writer.write(roi)
            if overlay_writer is not None:
                overlay_writer.write(frame)
            continue

        h, w = frame.shape[:2]
        eb = expand_box(box, float(CONFIG["BBOX_EXPAND"]), w, h)
        crop = frame[eb.y1:eb.y2, eb.x1:eb.x2].copy()
        roi = resize_with_letterbox(crop, roi_size)
        crop_writer.write(roi)

        if overlay_writer is not None:
            lab = int(labels_by_frame.get(fi, -1))
            lab_name = CONFIG["LABEL_NAMES"].get(lab, str(lab))
            ov = draw_box(frame, box, color=(0, 255, 0), thickness=2,
                          label=f"{folder_id} fi={fi} lab={lab}({lab_name}) conf={box.conf:.2f}")
            overlay_writer.write(ov)

    crop_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()
    cap.release()

    print(f"[Done] {folder_id}")
    print("  Diff        :", os.path.abspath(diff_path))
    print("  CSV         :", os.path.abspath(csv_path))
    print("  Cropped ALL :", os.path.abspath(out_crop_all))
    if CONFIG["EXPORT_OVERLAY"]:
        print("  Overlay ALL :", os.path.abspath(out_overlay_all))

    # ✅ 按标签导出训练 clips（直接可用于 EfficientNet+GRU）
    clip_source = str(CONFIG.get("CLIP_SOURCE", "proc")).lower().strip()
    clip_video = proc_path if (clip_source == "proc") else diff_path
    if clip_source == "proc" and (not proc_path or (not os.path.isfile(proc_path))):
        print(f"[Warn] proc 视频不存在，回退到 diff 导出 clips: {proc_path}")
        clip_video = diff_path
    export_labeled_clips(folder_id, clip_video, boxes_by_frame, labels_by_frame)
    print("  Clips Src   :", os.path.abspath(clip_video))

    return stop_all


# ============================================================
# main：按子文件夹批量
# ============================================================
def main():
    in_root = CONFIG["IN_ROOT"]
    if not os.path.isdir(in_root):
        raise RuntimeError(f"IN_ROOT 不是文件夹：{in_root}")

    subfolders = list_subfolders(in_root)
    if not subfolders:
        raise RuntimeError(f"IN_ROOT 下没有子文件夹：{in_root}")

    tasks = []
    for sf in subfolders:
        diff_path = os.path.join(sf, CONFIG["DIFF_FILENAME"])
        proc_path = os.path.join(sf, CONFIG["PROC_FILENAME"])
        if os.path.exists(diff_path):
            tasks.append((safe_id_from_folder(sf), diff_path, proc_path))
        else:
            if CONFIG["SKIP_IF_NO_DIFF"]:
                continue
            raise RuntimeError(f"子文件夹缺少 {CONFIG['DIFF_FILENAME']}：{sf}")

    if not tasks:
        raise RuntimeError(f"没有找到任何 {CONFIG['DIFF_FILENAME']}：{in_root}")

    start_idx = int(CONFIG["START_INDEX_0BASED"] or 0)
    if CONFIG["START_INDEX_1BASED"] is not None:
        start_idx = max(0, int(CONFIG["START_INDEX_1BASED"]) - 1)
    start_idx = max(0, min(start_idx, len(tasks) - 1))

    print("Found folders(with diff):", len(tasks))
    print("Start index            :", start_idx, "(0-based)")
    print("Start task             :", tasks[start_idx][0], os.path.abspath(tasks[start_idx][1]))
    print()

    model = load_model_auto(
        weights_path=CONFIG["YOLO_WEIGHTS"],
        yaml_path=CONFIG["YOLO_MODEL_YAML"],
        nc=CONFIG["NC"],
        device=CONFIG["DEVICE"],
    )

    for i in range(start_idx, len(tasks)):
        folder_id, diff_video, proc_video = tasks[i]
        print("\n====================================================")
        print(f"[Task {i}/{len(tasks)-1}] {folder_id}")
        print(os.path.abspath(diff_video))
        print("====================================================")

        stop_all = process_one_video(model, folder_id, diff_video, proc_video)
        if stop_all:
            print("\n[StopAll] user pressed 'x' -> stop processing remaining folders.")
            break

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
