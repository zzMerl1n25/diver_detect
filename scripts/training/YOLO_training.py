    # -*- coding: utf-8 -*-
"""
YOLOv11 从零训练（手写 PyTorch 训练循环 + 图片帧数据集）
========================================================

✅ 适配你的 split 后图片格式：
yolo_dataset_split/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

✅ 修复 Ultralytics 内部 loss 依赖：为 DetectionModel 注入 model.args
✅ 修复 loss 非标量导致 backward 报错：强制把 loss 转为标量（mean）
✅ 训练中 tqdm 进度条 + 中间打印
✅ 结果输出到“模型文件夹”：
  run_dir/
    weights/best.pt / last.pt / epoch_xxx.pt
    checkpoint_last.pth   (包含 optimizer/scaler/epoch 等，可断点续训)
    metrics.csv
    loss_curve.png
    lr_curve.png
    config.json

✅ 中途保存 + 断点继续训练：
- 自动每个 epoch 保存 checkpoint_last.pth
- 可选每 N step 保存一次 checkpoint（更安全）
- 启动时若 RESUME=True，会从 checkpoint_last.pth 或指定路径加载继续训练
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================
# ---------------------- 超参数（全部集中在开头） --------------
# ============================================================

# ---------- 路径配置 ----------
DATASET_ROOT = os.path.join(PROJECT_ROOT, "YOLO_training", "yolo_dataset_split")
TRAIN_IMAGES_DIR = f"{DATASET_ROOT}/images/train"
VAL_IMAGES_DIR   = f"{DATASET_ROOT}/images/val"
TRAIN_LABELS_DIR = f"{DATASET_ROOT}/labels/train"
VAL_LABELS_DIR   = f"{DATASET_ROOT}/labels/val"

# ---------- 类别数 ----------
NC = 1

# ---------- 输入尺寸 ----------
IMGSZ = 1280

# ---------- 训练基础参数 ----------
EPOCHS = 120
BATCH_SIZE = 4
NUM_WORKERS = 4
DEVICE = "cuda"           # "cuda" / "cpu"
SEED = 42

# ---------- 优化器 ----------
LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
OPTIM = "adamw"           # "sgd" or "adamw"

# ---------- AMP ----------
USE_AMP = True

# ---------- 梯度累积 ----------
GRAD_ACCUM_STEPS = 2

# ---------- EMA（建议先关，跑通再开） ----------
USE_EMA = False
EMA_DECAY = 0.9998

# ---------- 轻量增强 ----------
AUG_HFLIP_P = 0.5
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST = 0.15

# ---------- 数据过滤 ----------
ONLY_LABELED = False   # True: 只保留有label的图片；False: 允许空label

# ---------- 日志与保存 ----------
RUN_DIR = os.path.join(PROJECT_ROOT, "YOLO_training", "runs_yolo11_from_scratch")
EXP_NAME = "y11_from_images_img1280"

SAVE_EVERY_EPOCH = 5         # 每隔多少 epoch 额外保存 epoch_xxx.pt
PRINT_EVERY = 20             # 每隔多少 step 输出一条 log（tqdm 同时也在显示）
SAVE_CHECKPOINT_EVERY_STEPS = 0  # 0=关闭；>0 表示每 N step 存一次 checkpoint_last.pth（更安全）

# ---------- 早停（防过拟合） ----------
EARLY_STOP = True
EARLY_STOP_PATIENCE = 10      # 连续多少个 epoch 无提升就停止
EARLY_STOP_MIN_DELTA = 0.0    # 认为“有提升”的最小 val_loss 下降
EARLY_STOP_WARMUP = 0         # 训练前 N 个 epoch 不启用早停

# ---------- YOLOv11 模型 YAML ----------
YOLO11_YAML = "ultralytics/cfg/models/11/yolo11.yaml"

# ---------- Loss 超参（给 Ultralytics loss 用） ----------
LOSS_BOX = 7.5
LOSS_CLS = 0.5
LOSS_DFL = 1.5
LABEL_SMOOTHING = 0.0
FL_GAMMA = 0.0

# ---------- 断点续训 ----------
RESUME = True
RESUME_PATH = ""     # 留空表示从“输出目录/checkpoint_last.pth”恢复；填路径表示从指定 checkpoint 恢复


# ============================================================
# ---------------------- 实现代码 ------------------------------
# ============================================================

import glob
import json
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ultralytics.nn.tasks import DetectionModel


# -----------------------------
# 基础工具
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")


def get_gpu_mem_gb_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


# -----------------------------
# 图像预处理：letterbox
# -----------------------------
def letterbox(
    img: np.ndarray,
    new_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw = new_size - nw
    dh = new_size - nh
    dw //= 2
    dh //= 2

    img_lb = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    img_lb[dh:dh + nh, dw:dw + nw] = img_resized
    return img_lb, r, (dw, dh)


# -----------------------------
# 读取 YOLO txt
# -----------------------------
def yolo_txt_load(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((0, 5), dtype=np.float32)
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return np.zeros((0, 5), dtype=np.float32)

    data = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        data.append([float(cls), float(x), float(y), float(w), float(h)])
    if not data:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


# -----------------------------
# 轻量增强（返回是否 flip 用于同步 bbox）
# -----------------------------
def apply_simple_aug(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    out = img
    do_flip = (random.random() < AUG_HFLIP_P)
    if do_flip:
        out = cv2.flip(out, 1)

    if AUG_BRIGHTNESS > 0 or AUG_CONTRAST > 0:
        c = 1.0 + random.uniform(-AUG_CONTRAST, AUG_CONTRAST)
        b = 255.0 * random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
        out = np.clip(out.astype(np.float32) * c + b, 0, 255).astype(np.uint8)

    return out, do_flip


# ============================================================
# 数据集：读图片 + label
# ============================================================

@dataclass
class ImageItem:
    img_path: str
    label_path: str


class SonarImageDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, imgsz: int, only_labeled: bool):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.imgsz = imgsz
        self.only_labeled = only_labeled

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_paths = []
        for e in exts:
            img_paths += glob.glob(os.path.join(images_dir, f"*{e}"))
            img_paths += glob.glob(os.path.join(images_dir, f"*{e.upper()}"))
        img_paths = sorted(set(img_paths))

        items: List[ImageItem] = []
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            lp = os.path.join(labels_dir, f"{stem}.txt")
            if only_labeled and (not os.path.exists(lp)):
                continue
            items.append(ImageItem(img_path=p, label_path=lp))

        if len(items) == 0:
            raise RuntimeError(
                f"没有找到任何可用样本：\n"
                f"  images_dir={images_dir}\n"
                f"  labels_dir={labels_dir}\n"
                f"  only_labeled={only_labeled}"
            )

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]

        frame = cv2.imread(it.img_path, cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1) 增强
        frame, do_flip = apply_simple_aug(frame)

        # 2) letterbox
        frame_lb, r, (dw, dh) = letterbox(frame, new_size=self.imgsz)

        # 3) 读 label：相对原图归一化
        labels = yolo_txt_load(it.label_path)

        h0, w0 = frame.shape[:2]
        targets = []
        for row in labels:
            cls, x, y, w, h = row.tolist()

            # flip 同步
            if do_flip:
                x = 1.0 - x

            # 原图像素
            xc = x * w0
            yc = y * h0
            bw = w * w0
            bh = h * h0

            # 映射到 letterbox 后坐标
            xc = xc * r + dw
            yc = yc * r + dh
            bw = bw * r
            bh = bh * r

            # 归一化到 IMGSZ
            x_new = float(np.clip(xc / self.imgsz, 0.0, 1.0))
            y_new = float(np.clip(yc / self.imgsz, 0.0, 1.0))
            w_new = float(np.clip(bw / self.imgsz, 0.0, 1.0))
            h_new = float(np.clip(bh / self.imgsz, 0.0, 1.0))

            if w_new <= 0.0 or h_new <= 0.0:
                continue

            targets.append([int(cls), x_new, y_new, w_new, h_new])

        targets = np.asarray(targets, dtype=np.float32)

        # 4) to tensor
        img_rgb = cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0

        return img, torch.from_numpy(targets)


def collate_fn(batch):
    imgs, tars = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    all_targets = []
    for bi, t in enumerate(tars):
        if t.numel() == 0:
            continue
        batch_idx = torch.full((t.shape[0], 1), bi, dtype=torch.float32)
        all_targets.append(torch.cat([batch_idx, t], dim=1))

    if len(all_targets) == 0:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    else:
        targets = torch.cat(all_targets, dim=0)

    return imgs, targets


# ============================================================
# EMA（可选）
# ============================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if k in msd:
                esd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)


# ============================================================
# 关键：给 Ultralytics loss 注入 args
# ============================================================

def attach_ultralytics_args(model: DetectionModel, imgsz: int, nc: int):
    from types import SimpleNamespace
    model.args = SimpleNamespace(
        imgsz=imgsz,
        nc=nc,
        box=LOSS_BOX,
        cls=LOSS_CLS,
        dfl=LOSS_DFL,
        label_smoothing=LABEL_SMOOTHING,
        fl_gamma=FL_GAMMA,
    )
    return model


def build_model_yolo11_from_yaml(yaml_path: str, nc: int, imgsz: int) -> DetectionModel:
    model = DetectionModel(cfg=yaml_path, ch=3, nc=nc, verbose=True)
    model = attach_ultralytics_args(model, imgsz=imgsz, nc=nc)
    return model


# ============================================================
# 关键：把 model(batch) 的输出稳健地解析成“标量 loss”
# ============================================================

def extract_scalar_loss(out: Any) -> torch.Tensor:
    """
    兼容 Ultralytics 不同版本返回结构：
    - (loss, loss_items)
    - {"loss": loss, ...}
    - 直接返回 Tensor
    并强制 loss 为标量（0-d tensor），否则 backward 会炸。
    """
    if isinstance(out, (tuple, list)):
        loss = out[0]
    elif isinstance(out, dict) and "loss" in out:
        loss = out["loss"]
    elif torch.is_tensor(out):
        loss = out
    else:
        raise RuntimeError(f"无法解析 model(batch) 返回值：type={type(out)}")

    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss)

    # 关键修复：如果不是标量，变成标量（mean 最稳）
    if loss.ndim != 0:
        loss = loss.mean()

    return loss


# ============================================================
# checkpoint：保存/加载（支持断点续训）
# ============================================================

def save_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    global_step: int,
    best_val: float,
    out_dir: str,
    ema: Optional[ModelEMA] = None
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "ema": ema.ema.state_dict() if (ema is not None) else None,
        "out_dir": out_dir,
    }
    torch.save(ckpt, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    ema: Optional[ModelEMA] = None
):
    # PyTorch 2.6 defaults weights_only=True; our checkpoints include optimizer/scaler/RNG, so load fully.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't support weights_only.
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    # 恢复随机状态（保证可复现/一致继续）
    rng = ckpt.get("rng", {})
    try:
        random.setstate(rng.get("python"))
    except Exception:
        pass
    try:
        np.random.set_state(rng.get("numpy"))
    except Exception:
        pass
    try:
        torch.set_rng_state(rng.get("torch"))
    except Exception:
        pass
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(rng.get("torch_cuda"))
        except Exception:
            pass

    if ema is not None and ckpt.get("ema") is not None:
        try:
            ema.ema.load_state_dict(ckpt["ema"], strict=True)
        except Exception:
            pass

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    out_dir = ckpt.get("out_dir", "")

    return start_epoch, global_step, best_val, out_dir


# ============================================================
# 训练/验证
# ============================================================

def train_one_epoch(
    model: DetectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    global_step: int,
    ckpt_path: str,
    out_dir: str,
    best_val: float,
    ema: Optional[ModelEMA],
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    t0 = time.time()

    pbar = tqdm(total=len(loader), desc=f"Train E{epoch}/{EPOCHS}", ncols=110)
    for step, (imgs, targets) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 构造 Ultralytics batch dict
        if targets.numel() == 0:
            batch = {
                "img": imgs,
                "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                "cls": torch.zeros((0, 1), device=device, dtype=torch.float32),
                "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
            }
        else:
            batch_idx = targets[:, 0].long()
            cls = targets[:, 1:2].float()
            bboxes = targets[:, 2:6].float()
            batch = {"img": imgs, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

        # forward + loss
        # 注意：只有 cuda 才启用 autocast
        use_autocast = (scaler is not None) and (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
            out = model(batch)
            loss = extract_scalar_loss(out)

        # 梯度累积
        loss_scaled = loss / float(GRAD_ACCUM_STEPS)

        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        running_loss += float(loss.detach().item())
        global_step += 1

        # tqdm 显示
        lr = optimizer.param_groups[0]["lr"]
        avg = running_loss / step
        mem = get_gpu_mem_gb_peak()
        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", mem=f"{mem:.2f}GB")
        pbar.update(1)

        # 中间输出
        if step % PRINT_EVERY == 0:
            tqdm.write(f"[Train] epoch={epoch} step={step}/{len(loader)} global_step={global_step} avg_loss={avg:.4f}")

        # 中途保存 checkpoint（可选）
        if SAVE_CHECKPOINT_EVERY_STEPS > 0 and (global_step % SAVE_CHECKPOINT_EVERY_STEPS == 0):
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, global_step, best_val, out_dir, ema=ema)
            tqdm.write(f"[CKPT] saved -> {ckpt_path}")

    pbar.close()
    epoch_time = time.time() - t0
    return running_loss / max(1, len(loader)), epoch_time, global_step


@torch.no_grad()
def validate(model: DetectionModel, loader: DataLoader, device: torch.device, epoch: int):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(total=len(loader), desc=f"Val   E{epoch}/{EPOCHS}", ncols=110)
    for step, (imgs, targets) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if targets.numel() == 0:
            batch = {
                "img": imgs,
                "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                "cls": torch.zeros((0, 1), device=device, dtype=torch.float32),
                "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
            }
        else:
            batch_idx = targets[:, 0].long()
            cls = targets[:, 1:2].float()
            bboxes = targets[:, 2:6].float()
            batch = {"img": imgs, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

        out = model(batch)
        loss = extract_scalar_loss(out)

        running_loss += float(loss.detach().item())
        avg = running_loss / step
        pbar.set_postfix(val_loss=f"{avg:.4f}")
        pbar.update(1)

    pbar.close()
    return running_loss / max(1, len(loader))


# ============================================================
# 结果输出：CSV + 曲线图
# ============================================================

def save_config(out_dir: str):
    cfg = {k: v for k, v in globals().items() if k.isupper()}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def append_csv(csv_path: str, row: dict):
    header_needed = not os.path.exists(csv_path)
    import csv
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header_needed:
            w.writeheader()
        w.writerow(row)


def plot_curves(out_dir: str, csv_path: str):
    import matplotlib.pyplot as plt
    import csv

    epochs = []
    tr = []
    va = []
    lrs = []

    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
            lrs.append(float(row["lr"]))

    if not epochs:
        return

    # loss curve
    plt.figure()
    plt.plot(epochs, tr, label="train_loss")
    plt.plot(epochs, va, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # lr curve
    plt.figure()
    plt.plot(epochs, lrs, label="lr")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_curve.png"), dpi=200)
    plt.close()


# ============================================================
# main
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device =", device)

    # 输出目录（固定一个“模型文件夹”，便于断点续训）
    # 注意：为了能“继续训练写同一个目录”，这里不再每次启动都新建时间戳目录
    out_dir = os.path.join(RUN_DIR, EXP_NAME)
    weights_dir = os.path.join(out_dir, "weights")
    ensure_dir(weights_dir)
    save_config(out_dir)

    print("Output dir:", os.path.abspath(out_dir))

    csv_path = os.path.join(out_dir, "metrics.csv")
    ckpt_path_default = os.path.join(out_dir, "checkpoint_last.pth")
    ckpt_path = RESUME_PATH.strip() if RESUME_PATH.strip() else ckpt_path_default

    # Dataset / Loader
    train_ds = SonarImageDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, IMGSZ, ONLY_LABELED)
    val_ds   = SonarImageDataset(VAL_IMAGES_DIR,   VAL_LABELS_DIR,   IMGSZ, ONLY_LABELED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"Train samples = {len(train_ds)} | Val samples = {len(val_ds)}")
    print(f"Train steps/epoch = {len(train_loader)} | Val steps/epoch = {len(val_loader)}")

    # Model
    model = build_model_yolo11_from_yaml(YOLO11_YAML, NC, IMGSZ).to(device)

    # Optim
    if OPTIM.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # AMP scaler（仅 CUDA 启用）
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    # EMA（可选）
    ema = ModelEMA(model, EMA_DECAY) if USE_EMA else None

    # Resume
    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    bad_epochs = 0

    if RESUME and os.path.exists(ckpt_path):
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        start_epoch, global_step, best_val, _ = load_checkpoint(
            ckpt_path, model, optimizer, scaler if (USE_AMP and device.type == "cuda") else None, device, ema=ema
        )
        print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, best_val={best_val:.6f}")
    else:
        print("[Resume] Not resuming (checkpoint not found or RESUME=False).")

    # Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        tr_loss, ep_time, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler if (USE_AMP and device.type == "cuda") else None,
            device, epoch, global_step,
            ckpt_path_default, out_dir, best_val, ema=ema
        )
        va_loss = validate(model, val_loader, device, epoch)

        lr = optimizer.param_groups[0]["lr"]
        mem_peak = get_gpu_mem_gb_peak()

        # 保存 best/last/epoch_xxx
        improved = va_loss < (best_val - EARLY_STOP_MIN_DELTA)
        if improved:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            bad_epochs = 0
        else:
            bad_epochs += 1

        torch.save(model.state_dict(), os.path.join(weights_dir, "last.pt"))

        if (epoch % SAVE_EVERY_EPOCH) == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch:03d}.pt"))

        # 保存 checkpoint_last（用于断点续训）
        save_checkpoint(ckpt_path_default, model, optimizer,
                        scaler if (USE_AMP and device.type == "cuda") else None,
                        epoch, global_step, best_val, out_dir, ema=ema)

        # 写 CSV
        append_csv(csv_path, {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "best_val": best_val,
            "lr": lr,
            "epoch_time_sec": ep_time,
            "gpu_mem_gb_peak": mem_peak,
            "global_step": global_step,
        })

        # 更新图表
        plot_curves(out_dir, csv_path)

        print(f"[Epoch {epoch}/{EPOCHS}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} best_val={best_val:.4f} "
              f"time={ep_time:.1f}s mem_peak={mem_peak:.2f}GB global_step={global_step}")

        if EARLY_STOP and epoch >= EARLY_STOP_WARMUP and bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"[EarlyStop] no val_loss improvement for {bad_epochs} epochs (best={best_val:.4f}). Stop.")
            break

    print("\n训练完成 ✅")
    print("结果目录：", os.path.abspath(out_dir))
    print("权重：")
    print("  best.pt =", os.path.join(weights_dir, "best.pt"))
    print("  last.pt =", os.path.join(weights_dir, "last.pt"))
    print("断点：")
    print("  checkpoint_last.pth =", ckpt_path_default)
    print("图表：")
    print("  loss_curve.png =", os.path.join(out_dir, "loss_curve.png"))
    print("  lr_curve.png   =", os.path.join(out_dir, "lr_curve.png"))
    print("日志：")
    print("  metrics.csv    =", os.path.join(out_dir, "metrics.csv"))


if __name__ == "__main__":
    main()
