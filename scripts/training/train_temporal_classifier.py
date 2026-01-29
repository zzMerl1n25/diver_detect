# -*- coding: utf-8 -*-
"""
train_temporal_classifier.py
============================

【任务】
训练一个“轨迹级动作判别器”：
输入：某条 track 的 5 秒 ROI 序列（frames + 可选 diffs）
输出：该序列是否为“蛙人/游泳人”（二分类）

【数据来源】
build_track_dataset.py 输出：
- index.csv：样本索引（npz文件名 + label）
- samples/*.npz：每个轨迹片段样本

【为什么用 frames + diffs】
- frames：目标外形/轮廓信息
- diffs：突出“动作变化”（游泳推进周期），声呐纹理弱时 diff 非常有用

【模型结构（建议第一版）】
EfficientNetV2-S（逐帧特征提取） + GRU（时序编码） + 分类头
- backbone 对每帧 ROI 提特征（共享权重）
- GRU 学 T 帧的时间模式
- 最后一帧隐藏状态作为整个序列表示进行二分类

【输出】
OUT_DIR/EXP_NAME/
  best.pt   # 验证集 loss 最低的权重
  last.pt   # 最后一轮权重
  epoch_XXX.pt（定期保存）
"""

import os

# ============================================================
# ---------------------- 超参数（集中在开头） ------------------
# ============================================================

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------- 数据路径 ----------
NPZ_DIR_TRAIN = os.path.join(PROJECT_ROOT, "data", "track_dataset_out", "samples")
INDEX_CSV_TRAIN = os.path.join(PROJECT_ROOT, "data", "track_dataset_out", "index.csv")

NPZ_DIR_VAL = os.path.join(PROJECT_ROOT, "track_dataset_out_val", "samples")
INDEX_CSV_VAL = os.path.join(PROJECT_ROOT, "track_dataset_out_val", "index.csv")

# ---------- 输入序列配置 ----------
USE_DIFF = True                     # True: 输入通道=2(frame+diff)；False: 输入通道=1
SEQ_LEN = 35                        # 序列长度（需与你生成npz一致）
ROI_SIZE = 224                      # ROI尺寸（需与你生成npz一致）

# ---------- 训练超参 ----------
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"
NUM_WORKERS = 4
SEED = 42
USE_AMP = True                      # 混合精度（GPU推荐开）

# ---------- 模型结构超参 ----------
EMBED_DIM = 256                     # 每帧特征降维后的embedding维度
GRU_HIDDEN = 256
GRU_LAYERS = 1
DROPOUT = 0.2

# ---------- 标签定义 ----------
POS_LABEL = "positive"              # 正样本（人/蛙人）
NEG_LABEL = "negative"              # 负样本（非人）

# ---------- 输出 ----------
OUT_DIR = os.path.join(PROJECT_ROOT, "runs_temporal")
EXP_NAME = "effv2s_gru_roi224_t35"
SAVE_EVERY = 5
PRINT_EVERY = 50

# ---------- 早停（防过拟合） ----------
EARLY_STOP = True
EARLY_STOP_PATIENCE = 8       # 连续多少个 epoch 无提升就停止
EARLY_STOP_MIN_DELTA = 0.0    # 认为“有提升”的最小 val_loss 下降
EARLY_STOP_WARMUP = 0         # 前 N 个 epoch 不启用早停


# ============================================================
# ---------------------- 导入依赖 -----------------------------
# ============================================================

import csv
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm


# ============================================================
# ---------------------- 工具：固定随机种子 --------------------
# ============================================================

def set_seed(seed: int) -> None:
    """
    固定随机种子，便于复现实验

    参数：
        seed: 随机种子整数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# ---------------------- 数据集：读取轨迹 npz ------------------
# ============================================================

class TrackNPZDataset(Dataset):
    """
    轨迹样本数据集：按 index.csv 加载 npz 序列样本

    index.csv 需要至少包含两列：
        npz,label
    其中 label 只有以下值会被采纳：
        positive / negative
    unknown 会被跳过（减少噪声）

    每个样本返回：
        x: (T,C,H,W) float32, 0~1
        y: float32 标量（positive=1, negative=0）
    """

    def __init__(self, index_csv: str, npz_dir: str, use_diff: bool = True):
        """
        参数：
            index_csv: index.csv 路径
            npz_dir:   npz 文件所在目录
            use_diff:  是否读取并使用 diffs 通道
        """
        self.items = []          # [(npz_name, label_str), ...]
        self.npz_dir = npz_dir
        self.use_diff = use_diff

        # 读取索引文件
        with open(index_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lab = row["label"]
                # 只用正负样本
                if lab not in (POS_LABEL, NEG_LABEL):
                    continue
                self.items.append((row["npz"], lab))

        if len(self.items) == 0:
            raise RuntimeError(f"没有可用样本：{index_csv}（可能label都是unknown或路径不对）")

    def __len__(self) -> int:
        """样本数量"""
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        读取一个样本

        返回：
            x: torch.FloatTensor, shape=(T,C,H,W), 值域0~1
            y: torch.FloatTensor, shape=(), 0或1
        """
        npz_name, lab = self.items[idx]
        p = os.path.join(self.npz_dir, npz_name)

        # allow_pickle=True 是为了兼容 npz 内存了字符串等元信息
        data = np.load(p, allow_pickle=True)

        # frames: uint8 -> float32 归一化到 0..1
        frames = data["frames"].astype(np.float32) / 255.0  # (T,H,W)

        if self.use_diff:
            diffs = data["diffs"].astype(np.float32) / 255.0  # (T,H,W)
        else:
            diffs = None

        # 组装通道： (T,C,H,W)
        if self.use_diff:
            # frame + diff 组成 2 通道
            x = np.stack([frames, diffs], axis=1)   # (T,2,H,W)
        else:
            # 只有 frame -> 1 通道
            x = frames[:, None, :, :]               # (T,1,H,W)

        # 标签：positive=1, negative=0
        y = 1.0 if lab == POS_LABEL else 0.0

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def collate_fn(batch):
    """
    DataLoader 拼接函数

    batch 是一个 list，元素是 (x,y)
    其中每个 x 的形状为 (T,C,H,W)

    拼接后：
        x: (B,T,C,H,W)
        y: (B,)
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y


# ============================================================
# ---------------------- 模型：EffV2-S + GRU ------------------
# ============================================================

class EffV2sGRU(nn.Module):
    """
    轨迹序列二分类模型

    输入：
        x: (B,T,C,H,W)
    输出：
        logit: (B,)  # 未过 sigmoid 的 logits

    结构分解：
    1) backbone：对每帧 ROI 提取特征向量（共享权重）
       - 先把 (B,T,...) reshape 成 (B*T,...)
       - EfficientNetV2-S 输出维度通常是 1280

    2) proj：把 1280 -> EMBED_DIM
       - 降维减少GRU负担
       - 加 Dropout 提升泛化

    3) GRU：输入 (B,T,EMBED_DIM) 输出 (B,T,GRU_HIDDEN)

    4) head：用最后时刻 hidden 做二分类
    """

    def __init__(self, in_ch: int):
        """
        参数：
            in_ch: 输入通道数
                - use_diff=True -> in_ch=2
                - use_diff=False -> in_ch=1
        """
        super().__init__()

        # ---------- 1) backbone ----------
        # weights=None 表示从头训练（随机初始化）
        backbone = tvm.efficientnet_v2_s(weights=None)

        # EfficientNetV2-S 默认第一层卷积输入通道为3（RGB）
        # 我们输入可能是 1 或 2 通道，所以要替换第一层卷积
        first_conv = backbone.features[0][0]

        if in_ch != 3:
            # 构建新的 conv，保持输出通道与kernel/stride/pad一致，只改 in_channels
            new_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
            # 初始化：Kaiming 对 ReLU 网络常用
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.features[0][0] = new_conv

        # 去掉分类头，只输出特征向量
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # EfficientNetV2-S 输出维度通常是 1280（torchvision实现）
        backbone_out_dim = 1280

        # ---------- 2) 特征投影 ----------
        self.proj = nn.Sequential(
            nn.Linear(backbone_out_dim, EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
        )

        # ---------- 3) GRU ----------
        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            batch_first=True,
            bidirectional=False,
        )

        # ---------- 4) 分类头 ----------
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(GRU_HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: (B,T,C,H,W)
        返回：
            logit: (B,)
        """
        B, T, C, H, W = x.shape

        # 把时间维展平：逐帧过 backbone
        x = x.view(B * T, C, H, W)

        # backbone 输出每帧特征： (B*T, 1280)
        feat = self.backbone(x)

        # 投影到更小维度： (B*T, EMBED_DIM)
        feat = self.proj(feat)

        # 恢复时间序列： (B, T, EMBED_DIM)
        feat = feat.view(B, T, -1)

        # GRU 编码：out (B,T,GRU_HIDDEN)
        out, _ = self.gru(feat)

        # 取最后一个时刻的 hidden（代表整个序列）
        last = out[:, -1, :]  # (B, GRU_HIDDEN)

        # 输出 logits（未过 sigmoid）
        logit = self.head(last).squeeze(1)  # (B,)
        return logit


# ============================================================
# ---------------------- 训练/验证循环 -------------------------
# ============================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    device: torch.device,
                    epoch: int) -> float:
    """
    训练一个 epoch

    参数：
        model:      模型
        loader:     训练 DataLoader
        optimizer:  优化器
        scaler:     AMP scaler（若USE_AMP=False，可传 None）
        device:     训练设备
        epoch:      当前 epoch 编号
    返回：
        avg_loss:   平均训练损失
    """
    model.train()

    # BCEWithLogitsLoss 内部包含 sigmoid，数值稳定
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    steps = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)  # (B,T,C,H,W)
        y = y.to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        # AMP 混合精度：在 GPU 上加速并省显存
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logit = model(x)                 # (B,)
            loss = criterion(logit, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item())
        steps += 1

        if (step + 1) % PRINT_EVERY == 0:
            print(f"[Train] epoch={epoch} step={step+1}/{len(loader)} loss={total_loss/steps:.4f}")

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             epoch: int) -> Tuple[float, float]:
    """
    在验证集评估

    参数：
        model:  模型
        loader: 验证 DataLoader
        device: 设备
        epoch:  当前epoch编号
    返回：
        avg_loss: 平均验证loss
        acc:      简单准确率（阈值0.5）
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    steps = 0

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model(x)
        loss = criterion(logit, y)

        # 概率
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).float()

        correct += int((pred == y).sum().item())
        total += int(y.numel())

        total_loss += float(loss.detach().item())
        steps += 1

    avg_loss = total_loss / max(1, steps)
    acc = correct / max(1, total)

    print(f"[Val]   epoch={epoch} loss={avg_loss:.4f} acc={acc:.4f}")
    return avg_loss, acc


# ============================================================
# ---------------------- 主入口 -------------------------------
# ============================================================

def main():
    """
    主训练入口：
    1) 固定随机种子
    2) 构建数据集与 DataLoader
    3) 构建模型
    4) 训练循环：train + val + 保存 best/last
    """
    set_seed(SEED)

    os.makedirs(os.path.join(OUT_DIR, EXP_NAME), exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Device =", device)

    # 1) 数据集
    train_ds = TrackNPZDataset(INDEX_CSV_TRAIN, NPZ_DIR_TRAIN, use_diff=USE_DIFF)
    val_ds = TrackNPZDataset(INDEX_CSV_VAL, NPZ_DIR_VAL, use_diff=USE_DIFF)

    # 2) DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 3) 模型
    in_ch = 2 if USE_DIFF else 1
    model = EffV2sGRU(in_ch=in_ch).to(device)

    # 4) 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 5) AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    best_val = 1e18
    bad_epochs = 0

    # 6) 训练循环
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        va_loss, va_acc = evaluate(model, val_loader, device, epoch)

        # 保存 best（以 val loss 为准）
        improved = va_loss < (best_val - EARLY_STOP_MIN_DELTA)
        if improved:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, EXP_NAME, "best.pt"))
            bad_epochs = 0
        else:
            bad_epochs += 1

        # 定期保存
        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(OUT_DIR, EXP_NAME, f"epoch_{epoch:03d}.pt"))

        print(f"[Epoch {epoch}] train={tr_loss:.4f} val={va_loss:.4f} acc={va_acc:.4f} best_val={best_val:.4f}")

        if EARLY_STOP and epoch >= EARLY_STOP_WARMUP and bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"[EarlyStop] no val_loss improvement for {bad_epochs} epochs (best={best_val:.4f}). Stop.")
            break

    # 保存 last
    torch.save(model.state_dict(), os.path.join(OUT_DIR, EXP_NAME, "last.pt"))
    print("完成 ✅ 输出：", os.path.join(OUT_DIR, EXP_NAME))


if __name__ == "__main__":
    main()
