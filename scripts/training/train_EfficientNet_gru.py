# -*- coding: utf-8 -*-
"""
EfficientNetV2-S（逐帧特征提取 / frame encoder） + GRU（时序建模） + FC（二分类/多类别）
==========================================================================================

✅ 本版：读取你已经切好的 train/val/test（三种输入方式任选其一）
------------------------------------------------------------
方式A（推荐）：目录结构
OUT_ROOT/
  train/label_0/*.mp4
  train/label_1/*.mp4
  val/label_0/*.mp4
  test/label_1/*.mp4

方式B：split 列表文件（每行 "abs_path,label"）
OUT_ROOT/splits/split_train.txt
OUT_ROOT/splits/split_val.txt
OUT_ROOT/splits/split_test.txt

方式C：你仍然用一个 DATA_ROOT（label_x/）时，可回退为内部 val_ratio 划分
（但你现在已经有 train/val/test，就别用C了）

✅ 其他保留：
- TensorBoard + CSV + 曲线PNG
- 断点续训（step_last.pt / last.pt）
- 训练中 step 级保存（抗崩溃）

依赖：
pip install torch torchvision opencv-python numpy matplotlib tensorboard
"""

import os
import time
import json
import random
import csv
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================
# ✅ CONFIG：你一般只需要改这里
# ============================================================
CONFIG = {
    # ---------------- 数据读取模式 ----------------
    # "dir_splits"   : 从 OUT_ROOT/train|val|test/label_x 扫描
    # "list_splits"  : 从 split_train.txt / split_val.txt / split_test.txt 读取
    # "single_root"  : 只从 DATA_ROOT(label_x/) 扫描 + val_ratio 划分（兼容旧用法）
    "DATA_MODE": "dir_splits",  # 数据集组织方式（建议与 split 脚本输出一致）

    # 方式A：目录结构（train/val/test）
    "SPLIT_ROOT": os.path.join(PROJECT_ROOT, "EfficientNet_training", "splitted_dataset"),  # train/val/test 根目录
    "LABEL_PREFIX": "label_",  # 类别子目录前缀（label_0/label_1/...）

    # 固定随机种子，保证每次采样/打乱一致
    "RANDOM_SEED": 42,

    # ---------------- 视频采样 ----------------
    "NUM_FRAMES": 16,  # 每个视频采样多少帧（越多时序信息越足，但更慢）
    "FRAME_SIZE": 224,  # 送入 EfficientNet 的图像尺寸（建议与模型默认相同）
    "READ_RGB": True,  # True 读取 RGB（cv2 默认 BGR，此处会转）

    # ---------------- 模型结构 ----------------
    "NUM_CLASSES": 1,  # 二分类=1（输出logits维度=1）；多类=K（one-hot）
    "GRU_HIDDEN": 256,  # GRU 隐状态维度（越大容量越强，但更慢）
    "GRU_LAYERS": 1,  # GRU 层数（多层更强但更难训）
    "GRU_BIDIR": False,  # 是否双向 GRU（True 会增加计算/参数）
    "DROPOUT": 0.2,  # dropout 比例（防过拟合；过大会欠拟合）

    # ---------------- 训练超参 ----------------
    "EPOCHS": 30,  # 训练轮数
    "BATCH_SIZE": 2,  # 批大小（受显存限制）
    "NUM_WORKERS": 4,  # DataLoader 线程数（越大越快，但更吃 CPU）
    "LR": 3e-4,  # 学习率（过大会发散，过小收敛慢）
    "WEIGHT_DECAY": 1e-4,  # 权重衰减（L2 正则）
    "GRAD_CLIP_NORM": 1.0,  # 梯度裁剪阈值（防止梯度爆炸）

    # ---------------- 运行配置 ----------------
    "DEVICE": "cuda",  # 训练设备：cuda/cpu
    "AMP": True,  # 自动混合精度（省显存/加速）
    "LOG_EVERY": 20,  # 每多少 step 打印一次日志

    # ---------------- checkpoint / resume ----------------
    "SAVE_DIR": os.path.join(PROJECT_ROOT, "EfficientNet_training", "runs_action"),  # 输出目录
    "RUN_NAME": "effnetv2s_gru_binary",  # 本次实验名（用于区分日志/权重）
    "AUTO_RESUME": True,  # 自动从 last/step_last 恢复
    "RESUME_CKPT": None,  # 指定恢复 checkpoint（None 表示自动）
    "SAVE_EVERY_STEPS": 200,  # 每 N step 保存一次 step_last（防崩溃）

    # ---------------- 可视化与记录 ----------------
    "ENABLE_TENSORBOARD": True,  # 是否写 TensorBoard 日志
    "ENABLE_CSV_LOG": True,  # 是否写 CSV 训练曲线
    "PLOT_EVERY_EPOCH": True,  # 每个 epoch 画一次 loss/metric 曲线

    # ---------------- 测试集评估 ----------------
    "EVAL_TEST_EVERY_EPOCH": False,  # True：每个epoch都跑一次test（慢）
    "EVAL_TEST_AT_END": True,  # True：训练结束跑一次test

    # ---------------- 早停（防过拟合） ----------------
    "EARLY_STOP": True,  # 是否启用早停
    "EARLY_STOP_PATIENCE": 10,  # 连续多少个 epoch 无提升就停止
    "EARLY_STOP_MIN_DELTA": 0.0,  # 认为“有提升”的最小 val_loss 下降
    "EARLY_STOP_WARMUP": 0,  # 前 N 个 epoch 不启用早停
}


# ============================================================
# 工具函数：目录/随机种子/扫描/读取split list/采样/自动resume
# ============================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_video_file(fn: str) -> bool:
    return fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def scan_dir_split(split_dir: str, label_prefix: str) -> List[Tuple[str, int]]:
    """
    扫描一个 split 目录：
      split_dir/label_0/*.mp4
      split_dir/label_1/*.mp4
    返回 [(path,label), ...]
    """
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Split dir not found: {split_dir}")

    items: List[Tuple[str, int]] = []
    for name in sorted(os.listdir(split_dir)):
        p = os.path.join(split_dir, name)
        if not os.path.isdir(p):
            continue
        if not name.startswith(label_prefix):
            continue
        try:
            lab = int(name[len(label_prefix):])
        except Exception:
            continue

        for fn in sorted(os.listdir(p)):
            if is_video_file(fn):
                items.append((os.path.abspath(os.path.join(p, fn)), lab))
    return items


def load_split_list(list_path: str) -> List[Tuple[str, int]]:
    """
    读取 split_*.txt，每行：
      abs_path,label
    """
    if not os.path.isfile(list_path):
        raise RuntimeError(f"Split list not found: {list_path}")

    items: List[Tuple[str, int]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 兼容 path 里包含逗号的极端情况：从最后一个逗号切
            if "," not in s:
                continue
            path, lab_s = s.rsplit(",", 1)
            path = path.strip().strip('"')
            try:
                lab = int(lab_s.strip())
            except Exception:
                continue
            if os.path.isfile(path):
                items.append((os.path.abspath(path), lab))
    return items


def list_videos_by_label(data_root: str, label_prefix: str) -> List[Tuple[str, int]]:
    """旧用法：扫描 data_root/label_x/*.mp4"""
    items: List[Tuple[str, int]] = []
    if not os.path.isdir(data_root):
        raise RuntimeError(f"DATA_ROOT not a folder: {data_root}")

    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if not os.path.isdir(p):
            continue
        if not name.startswith(label_prefix):
            continue
        try:
            lab = int(name[len(label_prefix):])
        except Exception:
            continue

        for fn in sorted(os.listdir(p)):
            if is_video_file(fn):
                items.append((os.path.abspath(os.path.join(p, fn)), lab))
    return items


def stratified_split(items: List[Tuple[str, int]], val_ratio: float, seed: int):
    """仅用于 single_root 模式：分层划分 train/val。"""
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for path, lab in items:
        by_label.setdefault(lab, []).append((path, lab))

    train, val = [], []
    for _, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_val = int(round(n * val_ratio))
        n_val = max(1, n_val) if n >= 5 else max(0, n_val)
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def safe_get_total_frames(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


def uniform_indices(num_total: int, num_samples: int) -> List[int]:
    if num_total <= 0:
        return list(range(num_samples))
    if num_total <= num_samples:
        return list(range(num_total)) + [num_total - 1] * (num_samples - num_total)
    xs = np.linspace(0, num_total - 1, num_samples)
    return [int(round(x)) for x in xs]


def find_auto_resume_ckpt(run_dir: str) -> Optional[str]:
    step_path = os.path.join(run_dir, "step_last.pt")
    last_path = os.path.join(run_dir, "last.pt")
    if os.path.isfile(step_path):
        return step_path
    if os.path.isfile(last_path):
        return last_path
    return None


# ============================================================
# 训练曲线与CSV记录工具
# ============================================================

class MetricLogger:
    def __init__(self, run_dir: str, enable_csv: bool = True, enable_plot: bool = True):
        self.run_dir = run_dir
        self.enable_csv = enable_csv
        self.enable_plot = enable_plot

        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self.plot_path = os.path.join(run_dir, "curves.png")

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
            # 可选：test（如果你启用）
            "test_loss": [],
            "test_acc": [],
            "test_f1": [],
        }

        if self.enable_csv and (not os.path.exists(self.csv_path)):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss", "val_acc", "val_f1",
                    "test_loss", "test_acc", "test_f1",
                    "lr",
                    "global_step"
                ])

        if os.path.exists(self.csv_path):
            self._load_csv_history()

    def _load_csv_history(self):
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.history["epoch"].append(float(row["epoch"]))
                    self.history["train_loss"].append(float(row["train_loss"]))
                    self.history["val_loss"].append(float(row.get("val_loss", 0.0) or 0.0))
                    self.history["val_acc"].append(float(row.get("val_acc", 0.0) or 0.0))
                    self.history["val_f1"].append(float(row.get("val_f1", 0.0) or 0.0))
                    self.history["test_loss"].append(float(row.get("test_loss", 0.0) or 0.0))
                    self.history["test_acc"].append(float(row.get("test_acc", 0.0) or 0.0))
                    self.history["test_f1"].append(float(row.get("test_f1", 0.0) or 0.0))
                    self.history["lr"].append(float(row.get("lr", 0.0) or 0.0))
        except Exception:
            pass

    def append(self,
               epoch: int,
               train_loss: float,
               val_metrics: Dict[str, float],
               lr: float,
               global_step: int,
               test_metrics: Optional[Dict[str, float]] = None):
        val_loss = float(val_metrics.get("loss", 0.0))
        val_acc = float(val_metrics.get("acc", 0.0))
        val_f1 = float(val_metrics.get("f1", 0.0))

        if test_metrics is None:
            test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
        else:
            test_loss = float(test_metrics.get("loss", 0.0))
            test_acc = float(test_metrics.get("acc", 0.0))
            test_f1 = float(test_metrics.get("f1", 0.0))

        self.history["epoch"].append(float(epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_f1"].append(val_f1)
        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc)
        self.history["test_f1"].append(test_f1)
        self.history["lr"].append(float(lr))

        if self.enable_csv:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch,
                    train_loss,
                    val_loss, val_acc, val_f1,
                    test_loss, test_acc, test_f1,
                    lr,
                    global_step
                ])

        if self.enable_plot:
            self.save_plot()

    def save_plot(self):
        epochs = self.history["epoch"]
        if len(epochs) == 0:
            return

        fig = plt.figure(figsize=(14, 5))

        # loss
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(epochs, self.history["train_loss"], label="train_loss")
        ax1.plot(epochs, self.history["val_loss"], label="val_loss")
        if any(v > 0 for v in self.history["test_loss"]):
            ax1.plot(epochs, self.history["test_loss"], label="test_loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # val metrics
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(epochs, self.history["val_acc"], label="val_acc")
        ax2.plot(epochs, self.history["val_f1"], label="val_f1")
        ax2.set_title("Val metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # test metrics
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(epochs, self.history["test_acc"], label="test_acc")
        ax3.plot(epochs, self.history["test_f1"], label="test_f1")
        ax3.set_title("Test metrics")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Metric")
        ax3.set_ylim(0.0, 1.0)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=160)
        plt.close(fig)


# ============================================================
# Dataset：视频clip -> (T,3,H,W) + label
# ============================================================

class VideoClipDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], num_frames: int, frame_size: int, num_classes: int):
        self.items = items
        self.num_frames = int(num_frames)
        self.frame_size = int(frame_size)
        self.num_classes = int(num_classes)

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def _read_frames(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((self.num_frames, self.frame_size, self.frame_size, 3), dtype=np.uint8)

        total = safe_get_total_frames(cap)
        idxs = uniform_indices(total, self.num_frames)
        want = set(idxs)
        got: Dict[int, np.ndarray] = {}

        fi = 0
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if fi in want:
                got[fi] = fr
                if len(got) >= len(want):
                    break
            fi += 1

        cap.release()

        frames = []
        last = None
        for idx in idxs:
            fr = got.get(idx, None)
            if fr is None:
                fr = last
            if fr is None:
                fr = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            last = fr

            if CONFIG["READ_RGB"]:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

            if fr.shape[0] != self.frame_size or fr.shape[1] != self.frame_size:
                fr = cv2.resize(fr, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)

            frames.append(fr)

        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int):
        video_path, lab = self.items[idx]
        frames_u8 = self._read_frames(video_path)

        frames = [self.tf(frames_u8[t]) for t in range(frames_u8.shape[0])]
        x = torch.stack(frames, dim=0)  # (T,3,H,W)

        if self.num_classes == 1:
            y = torch.tensor([float(lab)], dtype=torch.float32)
        else:
            y = torch.zeros((self.num_classes,), dtype=torch.float32)
            if 0 <= lab < self.num_classes:
                y[lab] = 1.0

        return x, y, video_path


# ============================================================
# Model：EfficientNetV2-S + GRU + FC
# ============================================================

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
        feat = torch.flatten(feat, 1)   # (B*T, 1280)
        feat = feat.reshape(b, t, -1)   # (B,T,1280)

        out, _ = self.gru(feat)
        last = out[:, -1, :]

        last = self.drop(last)
        logits = self.fc(last)
        return logits


# ============================================================
# Checkpoint：保存/加载（“记忆点”）
# ============================================================

def save_ckpt(path: str,
              model, optimizer, scaler,
              epoch: int, global_step: int,
              best_val: float, config: dict):
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    torch.save(state, path)


def load_ckpt(path: str, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", 1e9))
    return epoch, global_step, best_val


# ============================================================
# Evaluate：验证/测试集
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total = 0

    tp = fp = tn = fn = 0
    correct = 0

    crit = nn.BCEWithLogitsLoss()

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = crit(logits, y)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        if num_classes == 1:
            prob = torch.sigmoid(logits).squeeze(1)
            pred = (prob >= 0.5).long()
            gt = (y.squeeze(1) >= 0.5).long()

            correct += int((pred == gt).sum().item())
            tp += int(((pred == 1) & (gt == 1)).sum().item())
            fp += int(((pred == 1) & (gt == 0)).sum().item())
            tn += int(((pred == 0) & (gt == 0)).sum().item())
            fn += int(((pred == 0) & (gt == 1)).sum().item())

    avg_loss = total_loss / max(1, total)

    if num_classes == 1:
        acc = correct / max(1, total)
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(1e-9, (precision + recall))
        return {"loss": avg_loss, "acc": acc, "precision": precision, "recall": recall, "f1": f1}

    return {"loss": avg_loss}


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, device, scaler,
                    num_classes: int, grad_clip: float,
                    log_every: int, amp: bool,
                    run_dir: str,
                    epoch: int,
                    global_step: int,
                    best_val: float,
                    tb: Optional[SummaryWriter] = None):
    model.train()
    crit = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0
    t0 = time.time()

    save_every = int(CONFIG["SAVE_EVERY_STEPS"] or 0)
    step_ckpt_path = os.path.join(run_dir, "step_last.pt")

    for step, (x, y, _) in enumerate(loader, start=1):
        global_step += 1

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = crit(logits, y)

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.size(0)
        loss_val = float(loss.item())
        total_loss += loss_val * bs
        total += bs

        if tb is not None:
            tb.add_scalar("train/loss_step", loss_val, global_step)

        if save_every > 0 and (global_step % save_every == 0):
            save_ckpt(step_ckpt_path, model, optimizer, scaler, epoch, global_step, best_val, CONFIG)
            print(f"[StepSave] global_step={global_step} -> {step_ckpt_path}")

        if log_every > 0 and (step % log_every == 0):
            dt = time.time() - t0
            cur = total_loss / max(1, total)
            sps = total / max(1e-6, dt)
            print(f"  step {step:5d}/{len(loader):5d}  loss={cur:.4f}  samples/s={sps:.2f}  global_step={global_step}")

    avg_loss = total_loss / max(1, total)
    return avg_loss, global_step


# ============================================================
# 核心：根据 DATA_MODE 读取 train/val/test
# ============================================================

def load_splits_from_config() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    返回：train_items, val_items, test_items（test_items 允许为空）
    """
    mode = str(CONFIG["DATA_MODE"]).lower().strip()
    label_prefix = str(CONFIG["LABEL_PREFIX"])

    if mode == "dir_splits":
        root = CONFIG["SPLIT_ROOT"]
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        test_dir = os.path.join(root, "test")

        train_items = scan_dir_split(train_dir, label_prefix)
        val_items = scan_dir_split(val_dir, label_prefix)
        test_items = scan_dir_split(test_dir, label_prefix) if os.path.isdir(test_dir) else []

        return train_items, val_items, test_items

    if mode == "list_splits":
        d = CONFIG["SPLIT_LIST_DIR"]
        train_list = os.path.join(d, CONFIG["TRAIN_LIST"])
        val_list = os.path.join(d, CONFIG["VAL_LIST"])
        test_list = os.path.join(d, CONFIG["TEST_LIST"])

        train_items = load_split_list(train_list)
        val_items = load_split_list(val_list)
        test_items = load_split_list(test_list) if os.path.isfile(test_list) else []

        return train_items, val_items, test_items

    if mode == "single_root":
        items = list_videos_by_label(CONFIG["DATA_ROOT"], label_prefix)
        if not items:
            raise RuntimeError(f"No videos found under: {CONFIG['DATA_ROOT']}")
        train_items, val_items = stratified_split(items, float(CONFIG["VAL_RATIO"]), int(CONFIG["RANDOM_SEED"]))
        return train_items, val_items, []

    raise ValueError("DATA_MODE must be one of: dir_splits / list_splits / single_root")


def print_dist(name: str, items: List[Tuple[str, int]]):
    dist: Dict[int, int] = {}
    for _, lab in items:
        dist[lab] = dist.get(lab, 0) + 1
    print(f"{name}: {len(items)} | dist: {dist}")


# ============================================================
# main
# ============================================================

def main():
    set_seed(int(CONFIG["RANDOM_SEED"]))

    device = torch.device(CONFIG["DEVICE"] if (CONFIG["DEVICE"] == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    train_items, val_items, test_items = load_splits_from_config()
    if not train_items or not val_items:
        raise RuntimeError("train 或 val 为空。请检查 DATA_MODE 与路径。")

    print_dist("Train", train_items)
    print_dist("Val  ", val_items)
    if test_items:
        print_dist("Test ", test_items)
    else:
        print("Test : (empty)")

    num_classes = int(CONFIG["NUM_CLASSES"])

    train_ds = VideoClipDataset(train_items, CONFIG["NUM_FRAMES"], CONFIG["FRAME_SIZE"], num_classes)
    val_ds = VideoClipDataset(val_items, CONFIG["NUM_FRAMES"], CONFIG["FRAME_SIZE"], num_classes)
    test_ds = VideoClipDataset(test_items, CONFIG["NUM_FRAMES"], CONFIG["FRAME_SIZE"], num_classes) if test_items else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(CONFIG["NUM_WORKERS"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(CONFIG["NUM_WORKERS"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(CONFIG["BATCH_SIZE"]),
            shuffle=False,
            num_workers=int(CONFIG["NUM_WORKERS"]),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = EffNetV2S_GRU(
        num_classes=num_classes,
        gru_hidden=CONFIG["GRU_HIDDEN"],
        gru_layers=CONFIG["GRU_LAYERS"],
        bidir=CONFIG["GRU_BIDIR"],
        dropout=CONFIG["DROPOUT"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["LR"]),
        weight_decay=float(CONFIG["WEIGHT_DECAY"]),
    )

    amp = bool(CONFIG["AMP"]) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    run_dir = os.path.join(CONFIG["SAVE_DIR"], CONFIG["RUN_NAME"])
    ensure_dir(run_dir)

    # 保存 config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    # TensorBoard
    tb = None
    if CONFIG["ENABLE_TENSORBOARD"]:
        tb_dir = os.path.join(run_dir, "tb")
        ensure_dir(tb_dir)
        tb = SummaryWriter(log_dir=tb_dir)
        print("TensorBoard logdir:", os.path.abspath(tb_dir))
        print("Run: tensorboard --logdir", os.path.abspath(tb_dir))

    metric_logger = MetricLogger(
        run_dir=run_dir,
        enable_csv=bool(CONFIG["ENABLE_CSV_LOG"]),
        enable_plot=bool(CONFIG["PLOT_EVERY_EPOCH"]),
    )

    # resume
    start_epoch = 0
    global_step = 0
    best_val = 1e9
    bad_epochs = 0

    resume_path = None
    if CONFIG["RESUME_CKPT"]:
        resume_path = CONFIG["RESUME_CKPT"]
    elif CONFIG["AUTO_RESUME"]:
        resume_path = find_auto_resume_ckpt(run_dir)

    if resume_path:
        start_epoch, global_step, best_val = load_ckpt(resume_path, model, optimizer, scaler)
        print(f"[Resume] {resume_path}")
        print(f"         start_epoch={start_epoch} global_step={global_step} best_val={best_val:.4f}")

    # checkpoints
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    step_path = os.path.join(run_dir, "step_last.pt")

    print("Run dir:", os.path.abspath(run_dir))

    try:
        for epoch in range(start_epoch + 1, int(CONFIG["EPOCHS"]) + 1):
            print(f"\nEpoch {epoch}/{CONFIG['EPOCHS']}")
            cur_lr = float(optimizer.param_groups[0]["lr"])

            if tb is not None:
                tb.add_scalar("train/lr", cur_lr, epoch)

            train_loss, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                num_classes=num_classes,
                grad_clip=float(CONFIG["GRAD_CLIP_NORM"]),
                log_every=int(CONFIG["LOG_EVERY"]),
                amp=amp,
                run_dir=run_dir,
                epoch=epoch,
                global_step=global_step,
                best_val=best_val,
                tb=tb,
            )

            val_metrics = evaluate(model, val_loader, device, num_classes)
            val_loss = float(val_metrics["loss"])

            # 可选：每epoch评估test（默认关）
            test_metrics = None
            if test_loader is not None and bool(CONFIG["EVAL_TEST_EVERY_EPOCH"]):
                test_metrics = evaluate(model, test_loader, device, num_classes)

            # 打印
            if num_classes == 1:
                msg = (f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                       f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} "
                       f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f}")
                if test_metrics is not None:
                    msg += (f" | Test loss: {test_metrics['loss']:.4f} "
                            f"test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1']:.4f}")
                print(msg)
            else:
                print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            # TensorBoard：epoch级
            if tb is not None:
                tb.add_scalar("train/loss_epoch", float(train_loss), epoch)
                tb.add_scalar("val/loss", float(val_loss), epoch)
                if num_classes == 1:
                    tb.add_scalar("val/acc", float(val_metrics["acc"]), epoch)
                    tb.add_scalar("val/f1", float(val_metrics["f1"]), epoch)
                    tb.add_scalar("val/precision", float(val_metrics["precision"]), epoch)
                    tb.add_scalar("val/recall", float(val_metrics["recall"]), epoch)

                if test_metrics is not None:
                    tb.add_scalar("test/loss", float(test_metrics["loss"]), epoch)
                    if num_classes == 1:
                        tb.add_scalar("test/acc", float(test_metrics["acc"]), epoch)
                        tb.add_scalar("test/f1", float(test_metrics["f1"]), epoch)

            # CSV + 曲线
            metric_logger.append(
                epoch=int(epoch),
                train_loss=float(train_loss),
                val_metrics=val_metrics,
                lr=cur_lr,
                global_step=int(global_step),
                test_metrics=test_metrics
            )

            improved = val_loss < (best_val - float(CONFIG["EARLY_STOP_MIN_DELTA"]))
            if improved:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            # 保存 last
            save_ckpt(last_path, model, optimizer, scaler, epoch, global_step, best_val, CONFIG)

            # 保存 best（按 val_loss）
            if improved:
                save_ckpt(best_path, model, optimizer, scaler, epoch, global_step, best_val, CONFIG)
                print(f"[Best] saved: {best_path} (val_loss={best_val:.4f})")

            if CONFIG["EARLY_STOP"] and epoch >= int(CONFIG["EARLY_STOP_WARMUP"]) and \
               bad_epochs >= int(CONFIG["EARLY_STOP_PATIENCE"]):
                print(f"[EarlyStop] no val_loss improvement for {bad_epochs} epochs (best={best_val:.4f}). Stop.")
                break

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Caught KeyboardInterrupt. Saving step checkpoint then exit...")
        cur_epoch = locals().get("epoch", start_epoch)
        save_ckpt(step_path, model, optimizer, scaler, cur_epoch, global_step, best_val, CONFIG)
        print(f"[Saved] {step_path}")

    # 训练结束：跑一次 test（默认开）
    if test_loader is not None and bool(CONFIG["EVAL_TEST_AT_END"]):
        test_metrics = evaluate(model, test_loader, device, num_classes)
        print("\n[Test @ End]")
        if num_classes == 1:
            print(f"loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f} "
                  f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f}")
        else:
            print(f"loss={test_metrics['loss']:.4f}")
        if tb is not None:
            tb.add_scalar("test_end/loss", float(test_metrics["loss"]), 0)
            if num_classes == 1:
                tb.add_scalar("test_end/acc", float(test_metrics["acc"]), 0)
                tb.add_scalar("test_end/f1", float(test_metrics["f1"]), 0)

    if tb is not None:
        tb.flush()
        tb.close()

    metric_logger.save_plot()

    print("\n✅ Done.")
    print("Run dir:", os.path.abspath(run_dir))
    print("Best:", os.path.abspath(best_path) if os.path.isfile(best_path) else "(none)")
    print("Last:", os.path.abspath(last_path) if os.path.isfile(last_path) else "(none)")
    print("Step:", os.path.abspath(step_path) if os.path.isfile(step_path) else "(none)")
    print("Metrics CSV:", os.path.abspath(metric_logger.csv_path))
    print("Curves PNG :", os.path.abspath(metric_logger.plot_path))


if __name__ == "__main__":
    main()
