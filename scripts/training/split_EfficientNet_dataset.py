# -*- coding: utf-8 -*-
"""
把“按 label_x/ 子文件夹存放的视频数据集”划分为：train / val / test
=====================================================================

✅ 支持两种输出方式（二选一）：
A) 复制/移动文件到新目录（默认 COPY，不破坏原数据）
B) 只生成 split_{train,val,test}.txt 列表文件（不复制）

✅ 分层划分（Stratified）：每个类别在 train/val/test 的比例尽量一致
✅ 可复现：固定 RANDOM_SEED

输入结构（例）：
DATA_ROOT/
  label_0/
    a.mp4
    b.mp4
  label_1/
    c.mp4
    d.mp4
  label_2/
    ...

输出结构（如果用 COPY/MOVE）：
OUT_ROOT/
  train/label_0/...
  train/label_1/...
  val/label_0/...
  test/label_1/...
同时会生成：
OUT_ROOT/splits/split_train.txt
OUT_ROOT/splits/split_val.txt
OUT_ROOT/splits/split_test.txt
以及 split_summary.json（数量统计）

依赖：Python 标准库（不需要额外 pip）
"""

import os
import json
import random
import shutil
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================
# ✅ 配置：只改这里
# ============================================================
CONFIG = {
    # 输入数据集根目录（里面有 label_0/ label_1/ ...）
    "DATA_ROOT": os.path.join(PROJECT_ROOT, "video_dataset_process", "labeled_clips"),
    "LABEL_PREFIX": "label_",

    # 输出根目录
    "OUT_ROOT": os.path.join(PROJECT_ROOT, "EfficientNet_training", "splitted_dataset"),

    # 划分比例（必须加起来 = 1.0）
    "RATIO_TRAIN": 0.7,
    "RATIO_VAL": 0.2,
    "RATIO_TEST": 0.1,

    # 随机种子（保证可复现）
    "RANDOM_SEED": 42,

    # 支持的视频扩展名
    "VIDEO_EXTS": [".mp4", ".avi", ".mov", ".mkv"],

    # 输出方式：
    # "copy"：复制到 OUT_ROOT/train|val|test 下
    # "move"：移动（会破坏原数据，慎用）
    # "list_only"：不复制不移动，只输出 split_*.txt 列表
    "MODE": "copy",

    # 如果目标文件已存在怎么处理：
    # True：跳过；False：覆盖
    "SKIP_IF_EXISTS": True,
}


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def is_video_file(fn: str, exts: List[str]) -> bool:
    ext = os.path.splitext(fn)[1].lower()
    return ext in set(e.lower() for e in exts)


def scan_items(data_root: str, label_prefix: str, exts: List[str]) -> List[Tuple[str, int]]:
    """
    扫描 DATA_ROOT 下 label_x 子目录，返回 [(abs_path, label_id), ...]
    """
    if not os.path.isdir(data_root):
        raise RuntimeError(f"DATA_ROOT 不是文件夹：{data_root}")

    items: List[Tuple[str, int]] = []
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
            if is_video_file(fn, exts):
                items.append((os.path.abspath(os.path.join(p, fn)), lab))

    return items


def stratified_split(
    items: List[Tuple[str, int]],
    ratio_train: float,
    ratio_val: float,
    ratio_test: float,
    seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    分层划分，保证每个 label 的比例尽量一致。
    """
    s = ratio_train + ratio_val + ratio_test
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须=1.0，目前={s}")

    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for path, lab in items:
        by_label.setdefault(lab, []).append((path, lab))

    train, val, test = [], [], []

    for lab, group in by_label.items():
        rng.shuffle(group)
        n = len(group)

        n_train = int(round(n * ratio_train))
        n_val = int(round(n * ratio_val))
        # test 兜底：确保总数不变
        n_test = n - n_train - n_val

        # 极小数据量保护：保证每类不至于全被分空（你也可以按需求删掉这段）
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            # 再次校正总数
            while n_train + n_val + n_test > n:
                # 优先从 train 扣
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            while n_train + n_val + n_test < n:
                n_train += 1

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_split_list(out_dir: str, name: str, split: List[Tuple[str, int]]):
    """
    生成列表文件：每行 "abs_path,label"
    """
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"split_{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for p, lab in split:
            f.write(f"{p},{lab}\n")
    return path


def copy_or_move_split(
    out_root: str,
    split_name: str,
    split: List[Tuple[str, int]],
    label_prefix: str,
    mode: str,
    skip_if_exists: bool
):
    """
    把 split 里的视频复制/移动到：
    OUT_ROOT/split_name/label_x/xxx.mp4
    """
    assert mode in ("copy", "move")

    for src, lab in split:
        label_dir = os.path.join(out_root, split_name, f"{label_prefix}{lab}")
        ensure_dir(label_dir)

        dst = os.path.join(label_dir, os.path.basename(src))

        if os.path.exists(dst) and skip_if_exists:
            continue
        if os.path.exists(dst) and (not skip_if_exists):
            try:
                os.remove(dst)
            except Exception:
                pass

        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)


def summarize(split: List[Tuple[str, int]]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for _, lab in split:
        k = str(lab)
        d[k] = d.get(k, 0) + 1
    return d


# ============================================================
# main
# ============================================================

def main():
    data_root = CONFIG["DATA_ROOT"]
    out_root = CONFIG["OUT_ROOT"]
    label_prefix = CONFIG["LABEL_PREFIX"]
    exts = CONFIG["VIDEO_EXTS"]
    seed = int(CONFIG["RANDOM_SEED"])
    mode = str(CONFIG["MODE"]).lower().strip()

    items = scan_items(data_root, label_prefix, exts)
    if not items:
        raise RuntimeError(f"在 {data_root} 下没有找到任何视频（label_x 子目录）")

    train, val, test = stratified_split(
        items,
        float(CONFIG["RATIO_TRAIN"]),
        float(CONFIG["RATIO_VAL"]),
        float(CONFIG["RATIO_TEST"]),
        seed
    )

    ensure_dir(out_root)
    splits_dir = os.path.join(out_root, "splits")
    ensure_dir(splits_dir)

    # 1) 写列表文件（不管 mode 是啥都写，方便你训练时直接读）
    p_train = write_split_list(splits_dir, "train", train)
    p_val = write_split_list(splits_dir, "val", val)
    p_test = write_split_list(splits_dir, "test", test)

    # 2) 复制/移动文件（可选）
    if mode in ("copy", "move"):
        copy_or_move_split(out_root, "train", train, label_prefix, mode, bool(CONFIG["SKIP_IF_EXISTS"]))
        copy_or_move_split(out_root, "val", val, label_prefix, mode, bool(CONFIG["SKIP_IF_EXISTS"]))
        copy_or_move_split(out_root, "test", test, label_prefix, mode, bool(CONFIG["SKIP_IF_EXISTS"]))
    elif mode == "list_only":
        pass
    else:
        raise ValueError(f"MODE 只能是 copy/move/list_only，当前={mode}")

    # 3) 输出统计
    summary = {
        "data_root": os.path.abspath(data_root),
        "out_root": os.path.abspath(out_root),
        "mode": mode,
        "seed": seed,
        "ratios": {
            "train": CONFIG["RATIO_TRAIN"],
            "val": CONFIG["RATIO_VAL"],
            "test": CONFIG["RATIO_TEST"],
        },
        "counts": {
            "all": len(items),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "by_label": {
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
        },
        "split_lists": {
            "train": os.path.abspath(p_train),
            "val": os.path.abspath(p_val),
            "test": os.path.abspath(p_test),
        }
    }

    with open(os.path.join(out_root, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Split done.")
    print("DATA_ROOT:", os.path.abspath(data_root))
    print("OUT_ROOT :", os.path.abspath(out_root))
    print("MODE     :", mode)
    print("Counts   :", summary["counts"])
    print("Train by label:", summary["by_label"]["train"])
    print("Val   by label:", summary["by_label"]["val"])
    print("Test  by label:", summary["by_label"]["test"])
    print("\nList files:")
    print(" ", p_train)
    print(" ", p_val)
    print(" ", p_test)
    print("Summary:", os.path.abspath(os.path.join(out_root, "split_summary.json")))


if __name__ == "__main__":
    main()
