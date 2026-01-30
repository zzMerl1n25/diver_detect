# -*- coding: utf-8 -*-
"""
检测结果 -> 跟踪(Tracking) -> 输出带TrackID视频 + 导出轨迹clips
==============================================================

核心改动：把“短轨迹噪声”在画框/写出 rows_out 前就过滤掉
- 轨迹未成熟（hits < MIN_HITS_TO_SHOW）不画、不写、不导出
- 最近窗口内命中不足也不画（抑制断断续续的噪声轨）
- 平均置信度不足也不画

"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===================== 配置区 =====================

DETECTIONS_CSV = os.path.join(ROOT_DIR, "runs_infer", "infer_diff_overlay_detections.csv")  # YOLO 推理输出 CSV
VIDEO_PATH     = r"C:\Users\Administrator\Desktop\sonar_track\data\test_video\1.mp4"  # 原视频路径（兜底）
DIFF_VIDEO_PATH = ""  # 填 diff 视频路径就用 diff 底图，否则用 VIDEO_PATH

OUTPUT_DIR = os.path.join(ROOT_DIR, "runs_track")  # 跟踪输出目录
OUTPUT_VIDEO_NAME = "tracked_overlay.mp4"  # 叠框视频名

# 跟踪参数
IOU_MATCH_THRES = 0.10  # IoU 匹配阈值（越大越严格）
MAX_AGE = 20  # 轨迹最大“失配”帧数（超过即删除）

# ✅ 这些是“最终保留轨迹”的过滤（summary/clip 用）
MIN_HITS = 5  # 轨迹最少命中帧数
MIN_MEAN_CONF = 0.25  # 轨迹平均置信度门槛

USE_CONF_FILTER = True  # 是否先过滤低置信检测框
CONF_FILTER = 0.20  # 低置信过滤阈值（检测层）

# ===================== ✅ 去除短轨迹噪声：显示/写出闸门 =====================
# 轨迹至少命中多少帧才“允许出现在画面/写tracks.csv”
MIN_HITS_TO_SHOW = 5  # 建议与 MIN_HITS 一致，或略小(3~5)

# 最近窗口内至少命中多少次才显示（抑制断续噪声）
RECENT_WINDOW = 10  # 看最近10帧
RECENT_MIN_HITS = 2  # 这10帧里至少命中4次才显示（可调 3~7）

# 若轨迹刚创建，允许一点“孵化期”
WARMUP_ALLOW = True  # 是否开启孵化期
WARMUP_MAX_FRAMES = 8  # 轨迹生命前8帧内，只要 hits>=2 就允许显示（可关）

# EMA 平滑（让框更稳）
USE_EMA = True  # 是否用 EMA 平滑框坐标
EMA_ALPHA = 0.7  # 越大越平滑(更粘旧框)；0.6~0.85

# 导出 clip
EXPORT_CLIPS = True  # 是否按轨迹导出 ROI clips
CLIP_LEN = 64  # 每个 clip 的帧数
CLIP_MARGIN = 1.5  # bbox 扩张倍率（避免裁剪过紧）
CLIP_FPS = 7  # 输出 clip 帧率

# 可视化
DRAW_BOX = True  # 是否画框
DRAW_ID = True  # 是否画 track id
DRAW_CONF = True  # 是否画置信度

# ===================== 实现区 =====================

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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
        """
        轨迹状态：
        - tid: 轨迹ID
        - hits/conf_sum: 累计命中与置信度统计
        - history: 每帧的框/置信度历史（用于连续性过滤）
        """
        self.tid = tid
        self.last_frame = frame_id
        self.age = 0
        self.hits = 1
        self.conf_sum = float(det["conf"])
        self.cls = int(det["cls"])
        self.xyxy = det["xyxy"][:]  # last bbox (smoothed)
        # history: (frame_id, xyxy_smoothed, conf)
        self.history = [(frame_id, det["xyxy"][:], float(det["conf"]))]

    def mean_conf(self):
        return self.conf_sum / max(1, self.hits)

    def recent_hits(self, window: int, cur_frame: int) -> int:
        """最近 window 帧内命中次数（history里有记录就算命中）"""
        if window <= 0:
            return self.hits
        start = cur_frame - window + 1
        cnt = 0
        # history按时间递增，倒序更快
        for fid, _, _ in reversed(self.history):
            if fid < start:
                break
            cnt += 1
        return cnt

    def life_frames(self, cur_frame: int) -> int:
        """轨迹存活帧数（从第一次出现到当前）"""
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

def draw_det(img, det, tid=None):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = clamp_xyxy(det["xyxy"], w, h)
    x1i,y1i,x2i,y2i = int(x1),int(y1),int(x2),int(y2)
    if DRAW_BOX:
        cv2.rectangle(img, (x1i,y1i), (x2i,y2i), (0,255,0), 2)
    if DRAW_ID or DRAW_CONF:
        parts = []
        if DRAW_ID and tid is not None:
            parts.append(f"id={tid}")
        if DRAW_CONF:
            parts.append(f"{det['conf']:.2f}")
        text = " ".join(parts)
        if text:
            cv2.putText(img, text, (x1i, max(0,y1i-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

def should_show_track(t: Track, cur_frame: int) -> bool:
    """
    ✅ 关键：短轨迹噪声过滤门槛
    返回 True 才允许画框/写rows_out
    """
    # 轨迹当前帧没命中就不显示
    if not t.history or t.history[-1][0] != cur_frame:
        return False

    # 平均置信度门槛
    if t.mean_conf() < MIN_MEAN_CONF:
        # warmup阶段允许略放宽
        if not (WARMUP_ALLOW and t.life_frames(cur_frame) <= WARMUP_MAX_FRAMES and t.hits >= 2):
            return False

    # 命中数门槛
    if t.hits < MIN_HITS_TO_SHOW:
        if not (WARMUP_ALLOW and t.life_frames(cur_frame) <= WARMUP_MAX_FRAMES and t.hits >= 2):
            return False

    # 最近窗口连续性门槛（防止断断续续的噪声轨）
    if RECENT_WINDOW > 0:
        rh = t.recent_hits(RECENT_WINDOW, cur_frame)
        if rh < RECENT_MIN_HITS:
            return False

    return True

def main():
    ensure_dir(OUTPUT_DIR)
    clips_dir = os.path.join(OUTPUT_DIR, "clips")
    if EXPORT_CLIPS:
        ensure_dir(clips_dir)

    df = pd.read_csv(DETECTIONS_CSV)
    df = df.sort_values(["frame_id", "conf"], ascending=[True, False]).reset_index(drop=True)
    if USE_CONF_FILTER:
        df = df[df["conf"] >= CONF_FILTER].reset_index(drop=True)

    base_video = DIFF_VIDEO_PATH if DIFF_VIDEO_PATH.strip() else VIDEO_PATH
    cap = cv2.VideoCapture(base_video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {base_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 7.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_video_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    # 按帧组织检测
    by_frame = {}
    for _, r in df.iterrows():
        fid = int(r["frame_id"])
        det = {
            "cls": int(r["cls"]),
            "conf": float(r["conf"]),
            "xyxy": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
        }
        by_frame.setdefault(fid, []).append(det)

    tracks = []
    next_id = 1
    rows_out = []  # ✅ 只写“通过 should_show_track 的轨迹”

    pbar = tqdm(total=total_frames if total_frames > 0 else None, ncols=110, desc="Track")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        dets = by_frame.get(frame_id, [])

        # 1) age++（所有轨迹先老化）
        for t in tracks:
            t.step()

        # 2) 匹配：贪心 IoU
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

        # 3) 未匹配 det -> 新建轨迹
        for j, d in enumerate(dets):
            if not used_det[j]:
                tracks.append(Track(next_id, frame_id, d))
                next_id += 1

        # 4) 清理超龄轨迹
        tracks = [t for t in tracks if t.age <= MAX_AGE]

        # 5) 可视化：只画“通过闸门”的轨迹
        vis = frame.copy()
        for t in tracks:
            if should_show_track(t, frame_id):
                det_vis = {"cls": t.cls, "conf": t.history[-1][2], "xyxy": t.xyxy}
                draw_det(vis, det_vis, tid=t.tid)

                # ✅ 只写出成熟轨迹
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

    # ===== 汇总（基于 rows_out，所以天然去掉短轨迹噪声）=====
    tracks_df = pd.DataFrame(rows_out)
    tracks_csv = os.path.join(OUTPUT_DIR, "tracks.csv")
    tracks_summary_csv = os.path.join(OUTPUT_DIR, "tracks_summary.csv")

    if len(tracks_df) == 0:
        tracks_df.to_csv(tracks_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["track_id","cls","len","start_frame","end_frame","mean_conf"]).to_csv(
            tracks_summary_csv, index=False, encoding="utf-8-sig"
        )
        print("\n⚠️ 没有任何轨迹通过过滤门槛（可能门槛太严或检测太弱）")
        print("tracked video   :", os.path.abspath(out_video_path))
        print("tracks.csv      :", os.path.abspath(tracks_csv))
        print("tracks_summary  :", os.path.abspath(tracks_summary_csv))
        return

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

    keep_ids = set(summary_df[
        (summary_df["len"] >= MIN_HITS) & (summary_df["mean_conf"] >= MIN_MEAN_CONF)
    ]["track_id"].tolist())

    print("\n✅ 跟踪完成（短轨迹噪声已在显示/写出阶段过滤）")
    print("tracked video   :", os.path.abspath(out_video_path))
    print("tracks.csv      :", os.path.abspath(tracks_csv))
    print("tracks_summary  :", os.path.abspath(tracks_summary_csv))
    print(f"kept tracks(summary filter): {len(keep_ids)}/{len(summary_df)}")

    # ===== clips 导出：只导出 keep_ids（更稳）=====
    if EXPORT_CLIPS and len(keep_ids) > 0:
        print("\n[Clips] exporting...")
        cap2 = cv2.VideoCapture(base_video)
        if not cap2.isOpened():
            print("[Clips] 无法打开视频，跳过导出")
            return

        idx = {}
        for _, r in tracks_df.iterrows():
            tid = int(r["track_id"])
            if tid not in keep_ids:
                continue
            idx.setdefault(int(r["frame_id"]), []).append(r)

        all_tracks = {}
        for tid, g in tracks_df.groupby("track_id"):
            if int(tid) in keep_ids:
                all_tracks[int(tid)] = g.sort_values("frame_id")

        for tid in sorted(list(keep_ids)):
            g = all_tracks[tid]
            fids = g["frame_id"].tolist()
            mid = fids[len(fids)//2]
            start = max(0, int(mid - CLIP_LEN//2))
            end = start + CLIP_LEN - 1

            outp = os.path.join(clips_dir, f"track_{tid:04d}.mp4")
            vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), float(CLIP_FPS), (W, H))
            if not vw.isOpened():
                continue

            cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
            fid = start
            while fid <= end:
                ok, fr = cap2.read()
                if not ok or fr is None:
                    break

                dets_here = idx.get(fid, [])
                for rr in dets_here:
                    if int(rr["track_id"]) != tid:
                        continue
                    det = {
                        "cls": int(rr["cls"]),
                        "conf": float(rr["conf"]),
                        "xyxy": [float(rr["x1"]),float(rr["y1"]),float(rr["x2"]),float(rr["y2"])]
                    }
                    draw_det(fr, det, tid=tid)

                vw.write(fr)
                fid += 1
            vw.release()

        cap2.release()
        print("[Clips] done ->", os.path.abspath(clips_dir))


if __name__ == "__main__":
    main()
