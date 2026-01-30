# -*- coding: utf-8 -*-
"""
把 YOLO 数据集拆成 train/val/test（支持同一视频内 temporal 切分）

输入：
  input_dir/
    images/*.jpg
    labels/*.txt

输出：
  output_dir/
    images/train|val|test
    labels/train|val|test
    data.yaml

关键点：
1) temporal 模式：同一个视频（group）内也会按时间顺序划分 train/val/test
2) 可选 gap：在 train->val、val->test 之间留出若干帧“隔离带”，不参与任何集合，减少相邻帧泄漏
3) 默认 group 解析：文件名 stem 的最后一个 '_' 前为 group
   例：sample_0001_000123.jpg -> group = sample_0001, frame = 000123
"""

import os
import re
import glob
import shutil
import random
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# -----------------------------
# 默认超参（你也可以用命令行覆盖）
# -----------------------------
# 输入数据集目录（images/ + labels/）
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "YOLO_training", "yolo_dataset")
# 输出目录（会生成 images/train|val|test + labels/train|val|test）
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "YOLO_training", "yolo_dataset_split")
# 训练/验证/测试比例（三者之和应为 1.0）
DEFAULT_TRAIN = 0.8
DEFAULT_VAL = 0.1
DEFAULT_TEST = 0.1
# 随机种子（random 模式下可复现）
DEFAULT_SEED = 42
# temporal 分割时的“隔离带”（防止相邻帧泄漏）
DEFAULT_GAP = 0  # 建议：如果帧很多，gap=10~50 会更靠谱
# 支持的图片扩展名（其他后缀会被忽略）
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 用于从文件名里提取帧号：默认抓最后一段纯数字
# e.g. sample_0001_000123 -> frame_id = 123
FRAME_RE = re.compile(r"(\d+)$")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_group_and_frame(stem: str):
    """
    从文件名（不含后缀）解析 (group, frame_id)

    默认规则：
    - frame_id：抓最后一段数字（stem 末尾的数字）
    - group：去掉最后一个 '_' + frame数字 的前缀

    例：
      stem = "sample_0001_000123"
      -> frame_id = 123
      -> group = "sample_0001"
    """
    m = FRAME_RE.search(stem)
    if not m:
        # 没有帧号，就当 frame_id=0，group=stem
        return stem, 0

    frame_str = m.group(1)
    frame_id = int(frame_str)

    # 去掉末尾数字；如果末尾还有下划线，则去掉一个下划线
    prefix = stem[: m.start(1)]
    if prefix.endswith("_"):
        prefix = prefix[:-1]

    group = prefix if prefix else stem
    return group, frame_id


def collect_items(images_dir: Path, labels_dir: Path, only_labeled: bool, exts=DEFAULT_IMAGE_EXTS):
    """
    收集样本：(img_path, lbl_path_or_None, group, frame_id)
    only_labeled=True：只保留有 label 的样本（推荐）
    """
    imgs = []
    for ext in exts:
        imgs.extend(images_dir.glob(f"*{ext}"))
        imgs.extend(images_dir.glob(f"*{ext.upper()}"))
    imgs = sorted(set(imgs))

    items = []
    for img in imgs:
        stem = img.stem
        lbl = labels_dir / f"{stem}.txt"
        if only_labeled and (not lbl.exists()):
            continue
        group, frame_id = parse_group_and_frame(stem)
        items.append((img, lbl if lbl.exists() else None, group, frame_id))
    return items


def split_temporal_per_group(items, train_ratio, val_ratio, test_ratio, gap):
    """
    temporal 切分：每个 group 内按 frame_id 排序，再切 train/val/test
    gap：在分割边界处跳过 gap 帧（不进入任何集合）

    返回：train_items, val_items, test_items
    """
    groups = {}
    for it in items:
        groups.setdefault(it[2], []).append(it)

    train, val, test = [], [], []

    for g, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: x[3])  # 按 frame_id 排序
        n = len(lst_sorted)
        if n == 0:
            continue

        # 基础切分点（不含 gap）
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        # 边界位置
        a = n_train
        b = n_train + n_val

        # 应用 gap：从 train 末尾往后跳 gap，从 val 末尾往后跳 gap
        # 也就是把 [a, a+gap) 和 [b, b+gap) 这两段丢弃
        a2 = min(n, a + gap)
        b2 = min(n, b + gap)

        train.extend(lst_sorted[:a])
        val.extend(lst_sorted[a2:b])
        test.extend(lst_sorted[b2:])

    return train, val, test, groups


def split_random(items, train_ratio, val_ratio, test_ratio, seed):
    """
    完全随机切分（不推荐用于视频帧，会泄漏）
    """
    items = items[:]
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def copy_split(items, out_dir: Path, split_name: str):
    """
    拷贝到输出目录：images/split_name 和 labels/split_name
    对于没有 label 的样本，会写一个空 txt（YOLO 允许无目标帧）
    """
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    for img, lbl, _group, _frame_id in items:
        shutil.copy2(img, img_out / img.name)
        if lbl is not None and lbl.exists():
            shutil.copy2(lbl, lbl_out / lbl.name)
        else:
            (lbl_out / f"{img.stem}.txt").write_text("", encoding="utf-8")


def write_data_yaml(out_dir: Path, class_names):
    yaml_path = out_dir / "data.yaml"
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
    content = f"""# Auto-generated
path: {out_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_lines}
"""
    yaml_path.write_text(content, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR, help="输入数据集目录（含 images/ labels/）")
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    ap.add_argument("--train", type=float, default=DEFAULT_TRAIN, help="训练集比例")
    ap.add_argument("--val", type=float, default=DEFAULT_VAL, help="验证集比例")
    ap.add_argument("--test", type=float, default=DEFAULT_TEST, help="测试集比例")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（random模式用）")
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP, help="temporal 模式下的隔离帧数（建议 10~50）")
    ap.add_argument("--mode", type=str, default="temporal", choices=["temporal", "random"],
                    help="temporal=按时间切分（推荐视频帧）；random=随机切分（不推荐）")
    ap.add_argument("--only-labeled", action="store_true", help="只保留有 label 的图片（推荐）")
    ap.add_argument("--names", type=str, default="frogman", help="类别名，逗号分隔，如: frogman,boat")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不拷贝文件")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+val+test 必须等于 1.0，现在是 {total}")

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    images_dir = in_dir / "images"
    labels_dir = in_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "输入目录必须包含 images/ 和 labels/。\n"
            f"  input={in_dir.resolve()}\n"
            f"  cwd={Path.cwd().resolve()}\n"
            "  你可以用 --input 指定目录，或把当前工作目录切到项目根目录。"
        )

    class_names = [x.strip() for x in args.names.split(",") if x.strip()]
    if not class_names:
        class_names = ["class0"]

    items = collect_items(images_dir, labels_dir, only_labeled=args.only_labeled)
    if not items:
        raise RuntimeError("没有找到任何可用样本（可能 --only-labeled 过滤掉了全部，或 images/ 为空）")

    if args.mode == "temporal":
        train_items, val_items, test_items, groups = split_temporal_per_group(
            items, args.train, args.val, args.test, gap=max(0, args.gap)
        )
        group_count = len(groups)
        mode_info = f"temporal per-group | groups={group_count} | gap={args.gap}"
    else:
        train_items, val_items, test_items = split_random(items, args.train, args.val, args.test, args.seed)
        mode_info = f"random | seed={args.seed}"

    print("====== Split Summary ======")
    print(f"Mode:   {mode_info}")
    print(f"Input:  {in_dir.resolve()}")
    print(f"Output: {out_dir.resolve()}")
    print(f"Total items: {len(items)}")
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")
    print(f"Ratios: train={args.train}, val={args.val}, test={args.test}")
    print(f"only_labeled={args.only_labeled}")

    if args.dry_run:
        print("DRY RUN: 未拷贝文件。")
        return

    ensure_dir(out_dir)
    copy_split(train_items, out_dir, "train")
    copy_split(val_items, out_dir, "val")
    copy_split(test_items, out_dir, "test")
    write_data_yaml(out_dir, class_names)

    print("\nDone.")
    print(f"data.yaml -> {(out_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
