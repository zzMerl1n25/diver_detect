# -*- coding: utf-8 -*-
"""
========================================================
Diff-only YOLO 手工标注器（逐帧画框 -> 导出 YOLO txt）
========================================================

你的数据结构（示例）：
ROOT_DIR/
  sample_0001/
    xxx_diff.mp4   (文件名包含 "diff")
    ...            (其它文件无所谓)
  sample_0002/
    yyy_proc.mp4
  ...

输出结构：
OUT_DIR/
  images/
    sample_0001_000123.jpg
  labels/
    sample_0001_000123.txt   (YOLO 格式：class xc yc w h，归一化)

交互操作：
- 鼠标左键拖拽：添加一个框（可多个）
- d：删除最后一个框
- n：清空当前帧所有框（无目标）
- c：复制上一帧框到当前帧（省时神器）
- s：保存当前帧（不跳帧）
- 空格/回车：保存并跳到 idx + STRIDE（按步长走）
- b：保存并跳到 idx - STRIDE（按步长走）
- ]：不保存，跳到 idx + STRIDE
- [：不保存，跳到 idx - STRIDE
- .：不保存，前进 1 帧
- ,：不保存，后退 1 帧
- g：跳转到指定帧（在控制台输入帧号）
- q / ESC：保存并结束当前视频（继续下一个）
- x：保存并停止全部

注意：
1) 本脚本只用 DIFF 视频作为训练图像来源（不处理 proc）
2) 若你在无桌面GUI环境运行，OpenCV窗口可能无法创建，会直接报错提醒
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import glob
import cv2
import numpy as np


# ========================================================
# 0) 所有“超参/可调参数”集中在这里（你只需要改这一块）
# ========================================================

# -------- 数据路径相关 --------
# 样本文件夹根目录（里面有很多子文件夹，每个子文件夹一个视频）
ROOT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_sonar_video")
# 输出目录（会自动创建 images/labels；用于 YOLO 训练）
OUT_DIR  = os.path.join(PROJECT_ROOT, "YOLO_training")

# -------- 如何在每个样本文件夹中找到 diff 视频 --------
# 规则：在该文件夹内寻找“文件名包含 PROC_KEY”的视频文件（mp4/avi/mov/mkv/m4v）
# 例：diff.mp4 / xxx_diff.mp4
PROC_KEY = "diff"

# -------- 导出设置 --------
# 导出图片格式：jpg / png 都可以（一般 jpg 更省空间）
IMG_EXT = "jpg"
# 目标类别 ID（只有一个类别就用 0；多类时要匹配训练）
CLASS_ID = 0

# -------- 标注流程设置（你要的功能：从第几帧开始 / 隔几帧标一帧）--------
# 从第几帧开始标（例如 200 表示跳过前 200 帧）
START_FRAME = 0
# 每次“步进/跳过/保存后跳转”的步长：例如 5 表示只标 0,5,10...
STRIDE = 1
# 若该帧已存在 images+labels，则自动跳过（按 STRIDE 跳）
SKIP_LABELED = True

# -------- OpenCV 显示窗口设置 --------
# True：可调整窗口大小；False：固定大小
WINDOW_NORMAL = True
# 画面左上角信息文字大小
FONT_SCALE = 0.85
# 文字粗细
TEXT_THICKNESS = 2

# -------- 画框显示设置（仅影响显示，不影响导出）--------
# 已确认框的颜色（BGR）
BOX_COLOR = (0, 255, 0)
# 已确认框线粗
BOX_THICKNESS = 2
# 鼠标拖拽中框颜色（BGR）
DRAG_COLOR = (0, 255, 255)
# 拖拽框线粗
DRAG_THICKNESS = 2

# -------- 安全保护 --------
# 鼠标拖出来的框，宽/高至少多少像素才算有效（防止误点）
MIN_BOX_SIZE = 2


# ========================================================
# 1) 通用工具函数（不要改，除非你知道在做什么）
# ========================================================

def ensure_dir(path: str):
    """确保目录存在，不存在就创建"""
    os.makedirs(path, exist_ok=True)


def clamp(v: int, lo: int, hi: int) -> int:
    """把 v 限制在 [lo, hi] 区间"""
    return max(lo, min(v, hi))


def find_video_by_key(folder: str, key: str):
    """
    在 folder 里找一个“文件名包含 key”的视频文件。
    返回：视频路径 或 None
    """
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")
    candidates = []
    for e in exts:
        candidates += glob.glob(os.path.join(folder, e))
    candidates = sorted(candidates)

    key_lower = key.lower()
    for p in candidates:
        if key_lower in os.path.basename(p).lower():
            return p
    return None


def draw_boxes(img, boxes, color, thickness):
    """在 img 上把 boxes（像素坐标）画出来，仅用于显示"""
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


def yolo_line_from_box(box, img_w: int, img_h: int, class_id: int):
    """
    把像素坐标框 (x1,y1,x2,y2) 转成 YOLO 格式：
    class xc yc w h （全部归一化到 [0,1]）

    注意：YOLO 使用的是“中心点 + 宽高”形式，并且都按图像宽高归一化。
    """
    x1, y1, x2, y2 = box

    # 防止越界
    x1 = clamp(int(x1), 0, img_w - 1)
    x2 = clamp(int(x2), 0, img_w - 1)
    y1 = clamp(int(y1), 0, img_h - 1)
    y2 = clamp(int(y2), 0, img_h - 1)

    # 无效框直接返回 None
    if x2 <= x1 or y2 <= y1:
        return None

    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


# ========================================================
# 2) 标注器主体（交互 + 保存）
# ========================================================

class Annotator:
    """
    一个样本（一个 diff 视频）对应一个 Annotator 实例
    负责：
    - 打开视频
    - 逐帧读取
    - 鼠标画框 / 键盘控制
    - 保存 images + labels（YOLO 格式）
    """

    def __init__(self, proc_path: str, out_img_dir: str, out_lbl_dir: str, sample_name: str):
        # -------- 基本信息 --------
        self.proc_path = proc_path
        self.out_img_dir = out_img_dir
        self.out_lbl_dir = out_lbl_dir
        self.sample_name = sample_name

        # -------- 打开视频 --------
        self.cap = cv2.VideoCapture(proc_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open proc video: {proc_path}")

        # -------- 读取视频元信息 --------
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # -------- 当前帧索引（你要求可从 START_FRAME 开始）--------
        self.idx = clamp(int(START_FRAME), 0, max(0, self.frame_count - 1))

        # -------- 当前帧的标注框列表（可以多个框）--------
        self.boxes = []

        # -------- 上一帧框（按 c 复制）--------
        self.prev_boxes = []

        # -------- 鼠标拖拽状态 --------
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_end = (0, 0)

        # -------- 窗口创建（重点：确保窗口句柄真的存在）--------
        self.win = f"YOLO Annotator | {sample_name}"
        flags = cv2.WINDOW_NORMAL if WINDOW_NORMAL else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self.win, flags)

        # 有些 OpenCV 构建会“惰性创建窗口句柄”，导致 setMouseCallback 句柄为空
        # 所以我们先 show 一张 dummy 图再 waitKey(1)，强制窗口真正创建出来。
        dummy = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.imshow(self.win, dummy)
        cv2.waitKey(1)

        # 检测窗口是否创建成功（失败通常发生在无GUI环境）
        try:
            prop = cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE)
        except Exception:
            prop = -1
        if prop < 0:
            raise RuntimeError(
                "OpenCV GUI window was not created.\n"
                "你可能在无桌面GUI环境/远程无窗口环境运行。\n"
                "建议：换到本机桌面运行，或用 CVAT/Label Studio 做标注。"
            )

        # 绑定鼠标回调（左键拖拽画框）
        cv2.setMouseCallback(self.win, self.on_mouse)

    # ---------------------------
    # 鼠标回调：左键按下/移动/抬起
    # ---------------------------
    def on_mouse(self, event, x, y, flags, param):
        """
        鼠标交互规则：
        - 左键按下：开始拖拽
        - 移动：更新拖拽框终点
        - 左键松开：形成一个 bbox，加入 boxes
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False

            x1, y1 = self.drag_start
            x2, y2 = self.drag_end

            # 确保 x1<x2, y1<y2
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            # 过滤太小的误触框
            if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
                self.boxes.append((x1, y1, x2, y2))

    # ---------------------------
    # 读取指定帧
    # ---------------------------
    def _read_frame(self, idx: int):
        """
        从视频读取 idx 帧（随机访问）。
        注意：随机 seek 在部分编码下可能稍慢，但对标注足够用。
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    # ---------------------------
    # 输出文件路径
    # ---------------------------
    def image_path(self, idx: int) -> str:
        """导出图像的路径：images/sample_0001_000123.jpg"""
        return os.path.join(self.out_img_dir, f"{self.sample_name}_{idx:06d}.{IMG_EXT}")

    def label_path(self, idx: int) -> str:
        """导出标签的路径：labels/sample_0001_000123.txt"""
        return os.path.join(self.out_lbl_dir, f"{self.sample_name}_{idx:06d}.txt")

    def is_labeled(self, idx: int) -> bool:
        """
        判断某一帧是否已标注：
        - images 和 labels 都存在才算完成
        """
        return os.path.exists(self.image_path(idx)) and os.path.exists(self.label_path(idx))

    # ---------------------------
    # 保存当前帧（图像 + YOLO 标签）
    # ---------------------------
    def save_current(self, frame):
        """
        保存逻辑：
        1) 把当前帧 frame 写到 images/
        2) 把当前 boxes 转成 YOLO 行，写到 labels/
        """
        # 1) 保存图像
        cv2.imwrite(self.image_path(self.idx), frame)

        # 2) 保存标签
        lines = []
        for b in self.boxes:
            line = yolo_line_from_box(b, self.w, self.h, CLASS_ID)
            if line:
                lines.append(line)

        with open(self.label_path(self.idx), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ---------------------------
    # 跳转到某一帧（你要求“选择下一个帧”/“跳转帧”）
    # ---------------------------
    def jump_to(self, target_idx: int):
        """
        跳转到目标帧：
        - 会自动 clamp 到合法范围
        - 默认清空 boxes（防止把上一帧框带到新帧）
        """
        self.idx = clamp(int(target_idx), 0, max(0, self.frame_count - 1))
        self.boxes = []

    # ---------------------------
    # 主循环：显示 + 键盘处理
    # ---------------------------
    def run(self):
        print(f"\n=== Annotating: {self.sample_name} ===")
        print(f"diff: {self.proc_path}")
        print(f"frames: {self.frame_count}, fps: {self.fps:.2f}, size: {self.w}x{self.h}")
        print(f"START_FRAME={START_FRAME}, STRIDE={STRIDE}, SKIP_LABELED={SKIP_LABELED}")
        print("\n快捷键：")
        print("  鼠标左键拖拽：画框（可多个）")
        print("  d：删除最后一个框 | n：清空本帧（无目标） | c：复制上一帧框")
        print("  s：仅保存")
        print("  空格/回车：保存并跳到 idx + STRIDE | b：保存并跳到 idx - STRIDE")
        print("  ]：不保存跳 idx + STRIDE | [：不保存跳 idx - STRIDE")
        print("  .：前进 1 帧 | ,：后退 1 帧（不保存）")
        print("  g：跳转到指定帧（控制台输入帧号）")
        print("  q / ESC：保存并结束当前视频（继续下一个）")
        print("  x：保存并停止全部\n")

        stop_all = False
        while True:
            # 到末尾就结束
            if self.idx >= self.frame_count:
                print("到视频末尾了。")
                break

            # 自动跳过已标注帧（按 STRIDE 跳）
            if SKIP_LABELED and self.is_labeled(self.idx):
                self.idx += STRIDE
                continue

            # 读取当前帧
            frame = self._read_frame(self.idx)
            if frame is None:
                print("读帧失败，结束。")
                break

            # 用于显示的副本（避免污染原始 frame）
            show = frame.copy()

            # 1) 画已确认框
            draw_boxes(show, self.boxes, BOX_COLOR, BOX_THICKNESS)

            # 2) 如果正在拖拽，画拖拽框
            if self.dragging:
                x1, y1 = self.drag_start
                x2, y2 = self.drag_end
                cv2.rectangle(show, (x1, y1), (x2, y2), DRAG_COLOR, DRAG_THICKNESS)

            # 3) 叠加状态信息（左上角）
            info = f"{self.sample_name} | frame {self.idx}/{self.frame_count-1} | boxes {len(self.boxes)} | stride {STRIDE}"
            cv2.putText(
                show,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (255, 255, 255),
                TEXT_THICKNESS,
                cv2.LINE_AA
            )

            # 显示
            cv2.imshow(self.win, show)

            # 等待键盘（20ms 刷新，足够流畅）
            key = cv2.waitKey(20) & 0xFF

            # ========== 退出/停止 ==========
            if key == 27 or key == ord('q'):
                self.save_current(frame)
                self.prev_boxes = list(self.boxes)
                print(f"Saved frame {self.idx} (exit current)")
                break
            elif key == ord('x'):
                self.save_current(frame)
                self.prev_boxes = list(self.boxes)
                print(f"Saved frame {self.idx} (stop all)")
                stop_all = True
                break

            # ========== 编辑框 ==========
            elif key == ord('d'):     # 删除最后一个框
                if self.boxes:
                    self.boxes.pop()

            elif key == ord('n'):     # 清空所有框（无目标）
                self.boxes = []

            elif key == ord('c'):     # 复制上一帧框
                self.boxes = list(self.prev_boxes)

            # ========== 保存 ==========
            elif key == ord('s'):     # 仅保存，不跳
                self.save_current(frame)
                self.prev_boxes = list(self.boxes)
                print(f"Saved frame {self.idx}")

            elif key == ord(' ') or key == 13:  # 保存并跳到 idx + STRIDE
                self.save_current(frame)
                self.prev_boxes = list(self.boxes)
                self.jump_to(self.idx + STRIDE)

            elif key == ord('b'):     # 保存并跳到 idx - STRIDE
                self.save_current(frame)
                self.prev_boxes = list(self.boxes)
                self.jump_to(self.idx - STRIDE)

            # ========== 跳帧（不保存）==========
            elif key == ord(']'):     # 跳到 idx + STRIDE
                self.jump_to(self.idx + STRIDE)

            elif key == ord('['):     # 跳到 idx - STRIDE
                self.jump_to(self.idx - STRIDE)

            elif key == ord('.'):     # 前进 1 帧
                self.jump_to(self.idx + 1)

            elif key == ord(','):     # 后退 1 帧
                self.jump_to(self.idx - 1)

            # ========== 跳转到指定帧 ==========
            elif key == ord('g'):
                # 注意：OpenCV 窗口里输入不方便，所以用控制台 input（阻塞但稳定）
                try:
                    target = input(
                        f"\n[{self.sample_name}] 当前 {self.idx}，输入要跳转到的帧号(0~{self.frame_count-1})："
                    ).strip()
                    if target != "":
                        self.jump_to(int(target))
                        print(f"Jump -> {self.idx}\n")
                except Exception as e:
                    print("跳转失败：", e)

        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        return stop_all


# ========================================================
# 3) 程序入口：遍历 ROOT_DIR 下每个样本文件夹
# ========================================================

def main():
    # 输出目录准备
    ensure_dir(OUT_DIR)
    img_dir = os.path.join(OUT_DIR, "yolo_dataset/images")
    lbl_dir = os.path.join(OUT_DIR, "yolo_dataset/labels")
    ensure_dir(img_dir)
    ensure_dir(lbl_dir)

    # 获取 ROOT_DIR 下所有子文件夹（每个子文件夹视为一个样本）
    folders = [p for p in glob.glob(os.path.join(ROOT_DIR, "*")) if os.path.isdir(p)]
    folders = sorted(folders)

    if not folders:
        print(f"没找到任何样本文件夹：{ROOT_DIR}")
        return

    print(f"Found {len(folders)} sample folders under {ROOT_DIR}")
    print(f"Output -> {os.path.abspath(OUT_DIR)}")
    print(f"PROC_KEY={PROC_KEY}, START_FRAME={START_FRAME}, STRIDE={STRIDE}, SKIP_LABELED={SKIP_LABELED}\n")

    # 逐样本处理
    for folder in folders:
        sample_name = os.path.basename(folder)

        # 找 diff 视频
        proc_path = find_video_by_key(folder, PROC_KEY)
        if not proc_path:
            print(f"[SKIP] {sample_name}: 找不到 diff 视频（文件名需包含 '{PROC_KEY}'）")
            continue

        print(f"\n[OK] {sample_name}")
        print("DIFF =", proc_path)

        # 打开标注器
        ann = Annotator(proc_path, img_dir, lbl_dir, sample_name)
        stop_all = ann.run()
        if stop_all:
            print("[Stop] user requested to stop all.")
            break

    print("\n全部处理完成。输出目录：")
    print(os.path.abspath(OUT_DIR))
    print("\n下一步：你需要把 images/labels 划分 train/val，并写 data.yaml 给 YOLO。")


if __name__ == "__main__":
    main()
