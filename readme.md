# sonar_track 声呐目标检测 / 跟踪 / 动作识别

本项目面向声呐视频中的“蛙人/游泳者”等目标，提供从**数据预处理 → 目标检测 → 跟踪 → 动作识别**的一整套流程。

核心思路：
- **几何与强度预处理**：去 UI、扇形区域裁剪、强度归一化、帧差。
- **YOLO 检测**：单帧检测候选目标。
- **跨帧跟踪**：把“同一亮斑”串成轨迹。
- **动作识别**：对 ROI clip / 序列做时序分类（EfficientNetV2-S + GRU）。

---

## 目录结构（核心）
```
/data
  /sonar_video                原始声呐视频
  /processed_sonar_video       预处理输出（proc/diff/mask）

/scripts
  /data_processing            批量预处理
  /labeling                   标注与复核
  /training                   训练与数据划分
  /inference                  推理/跟踪/一体化

/YOLO_training
  /yolo_dataset               YOLO 标注数据（images/labels）
  /yolo_dataset_split         YOLO 划分后的 train/val/test
  /runs_yolo11_from_scratch   YOLO 训练输出

/EfficientNet_training
  /splitted_dataset           动作分类数据集（train/val/test）
  /runs_action                动作模型训练输出

/video_dataset_process        标注复核输出（CSV/overlay/ROI/clip）
/runs_infer                   YOLO 推理输出
/runs_track                   跟踪输出
/runs_allinone                一体化推理输出
```

---

## 环境依赖
建议 Python 3.9+。常用依赖：
```
pip install torch torchvision ultralytics opencv-python numpy pandas tqdm matplotlib tensorboard
```
如果需要 PIL 字体渲染/可视化：
```
pip install pillow
```

---

## 使用教程（从零到可跑）

### 1) 预处理（生成 proc/diff）
脚本：`scripts/data_processing/YOLO_input_preprocessing.py`

- 输入：`data/sonar_video/` 下原始视频（mp4/avi/mov/mkv）
- 输出：`data/processed_sonar_video/<video_name>/proc.mp4`、`diff.mp4`、`sector_mask.png`

运行：
```
python scripts/data_processing/YOLO_input_preprocessing.py
```
需要改的关键参数（脚本顶部）：
- `VIDEO_DIR` / `OUTPUT_DIR`
- `MASK_RECTS`（遮 UI/字幕）
- 扇形参数 `USE_MANUAL_SECTOR`/`APEX_X`/`ANGLE_LEFT_DEG` 等
- 归一化/CLAHE/降噪开关

> **重要**：训练与推理必须使用同一套预处理参数（mask/扇形/归一化/resize 等）。

---

### 2) YOLO 检测数据集准备
#### 2.1 手工标注（从 proc 视频导出图像 + YOLO txt）
脚本：`scripts/labeling/annotate_video_yolo.py`

- 输入：`data/processed_sonar_video/<video>/proc.mp4`
- 输出：`YOLO_training/yolo_dataset/images/*.jpg` + `YOLO_training/yolo_dataset/labels/*.txt`

运行：
```
python scripts/labeling/annotate_video_yolo.py
```
需要改的关键参数（脚本顶部）：
- `ROOT_DIR`：默认 `data/processed_sonar_video`
- `OUT_DIR`：**建议设置为** `YOLO_training/yolo_dataset`
- `STRIDE`：标注步长（视频很长时可加速）

常用交互键：
- `n` 保存并跳到下一帧（按步长）
- `c` 复制上一帧框
- `d` 删除最后一个框，`x` 清空

#### 2.2 数据集划分（防止相邻帧泄漏）
脚本：`scripts/training/split_yolo_dataset.py`

```
python scripts/training/split_yolo_dataset.py \
  --input YOLO_training/yolo_dataset \
  --output YOLO_training/yolo_dataset_split \
  --mode temporal --gap 10 --only-labeled \
  --names frogman
```
- `temporal` 模式会按视频时间顺序切分，**比随机切分更稳**。
- `gap` 用于隔离相邻帧，防止训练/验证数据泄漏。

---

### 3) YOLO 训练
脚本：`scripts/training/YOLO_training.py`
```
python scripts/training/YOLO_training.py
```
训练结果写到：
```
YOLO_training/runs_yolo11_from_scratch/<EXP_NAME>/weights/*.pt
```

---

### 4) 动作识别数据集准备（重点）
动作模型训练数据由 **“目标 ROI + 帧级标签”** 自动生成。

#### 4.1 生成并复核 ROI + 帧标签
本步骤需要先自动标注一遍，再人工复核。

**先自动标注（YOLO + Track）**
- YOLO 推理：`scripts/inference/yolo_infer_video.py`
- 跟踪并导出轨迹/裁剪：`scripts/inference/track_and_export_clips.py`

**再人工复核/修正**
- 脚本：`scripts/labeling/yolo_review_and_crop.py`

流程：
1. 运行 `yolo_infer_video.py` 得到初始 detections
2. 运行 `track_and_export_clips.py` 自动生成轨迹/初始标注
3. 使用 `yolo_review_and_crop.py` 人工逐帧复核/修正框
4. 每帧打标签（0: 非蛙人，1: 蛙人）
5. 导出 CSV、overlay 视频、ROI 全量视频
6. 按标签切 5 秒 clip（默认 5s / 7fps = 35 帧）

运行：
```
python scripts/inference/yolo_infer_video.py
python scripts/inference/track_and_export_clips.py
python scripts/labeling/yolo_review_and_crop.py
```
主要输出：
```
video_dataset_process/
  csv/                每段视频的 frame 级标签
  cropped_all/        ROI 拼接视频
  overlay_all/        带框/标签的复核视频
  labeled_clips/      按 label_x 划分的训练 clips
```
关键配置（脚本顶部）：
- `YOLO_WEIGHTS`：你的 best.pt 路径
- `ROI_SIZE`、`BBOX_EXPAND`：裁剪 ROI 规则
- `CLIP_SECONDS` / `STRIDE_SECONDS`：clip 长度与步长
- `LABEL_NAMES`：0/1 的语义（默认 0=NOT_DIVER，1=DIVER）

#### 4.2 划分动作数据集（train/val/test）
脚本：`scripts/training/split_EfficientNet_dataset.py`
```
python scripts/training/split_EfficientNet_dataset.py
```
输出：
```
EfficientNet_training/splitted_dataset/
  train/label_x/*.mp4
  val/label_x/*.mp4
  test/label_x/*.mp4
  splits/split_*.txt
```

#### 4.3 训练动作模型（EfficientNetV2-S + GRU）
脚本：`scripts/training/train_EfficientNet_gru.py`
```
python scripts/training/train_EfficientNet_gru.py
```
输出：
```
EfficientNet_training/runs_action/<RUN_NAME>/{best.pt,last.pt,metrics.csv,curves.png}
```

> 该模型会从每段 clip 中**均匀抽取** `NUM_FRAMES` 帧（默认 16）进行时序建模。

---

## 推理（单独或一体化）

### 一体化推理（YOLO -> Track -> Action）
脚本：`scripts/inference/mian.py`（注意文件名是 `mian.py`）
```
python scripts/inference/mian.py
```
输出：`runs_allinone/01_yolo`、`02_track`、`03_action`

### 仅 YOLO 推理
脚本：`scripts/inference/yolo_infer_video.py`

### 仅跟踪（基于 detections.csv）
脚本：`scripts/inference/track_and_export_clips.py`

> 注意：部分推理脚本默认 `VIDEO_PATH` 是 Windows 绝对路径，请按本机路径修改。

---

## 技术细节（简明版）

### 1) 预处理（proc）
- **ROI 约束**：矩形遮罩 + 扇形 mask
- **强度归一化**：百分位裁剪 + 线性拉伸
- **可选**：CLAHE 局部增强 + speckle 降噪
- **灰度/3通道**：输出兼容 YOLO 输入

### 2) 帧差（diff）
- 使用 `|proc_gray(t) - proc_gray(t-1)|`
- 把运动信息显式化，供动作识别使用

### 3) 跟踪
- IoU 匹配 + 最大失配帧数
- EMA 平滑框
- “最小命中数 + 最近窗口命中”过滤短轨迹噪声

### 4) 动作模型
- **EfficientNetV2-S** 逐帧提特征
- **GRU** 建模时间序列
- 输出二分类或多分类（`NUM_CLASSES` 可扩展）

---

## 常见注意事项
- **预处理一致性**：训练和推理必须用同一套 mask/扇形/归一化参数。
- **视频泄漏问题**：训练/验证拆分必须按时间切分（`split_yolo_dataset.py --mode temporal`）。
- **标注环境**：标注脚本依赖 OpenCV GUI，需在有桌面环境运行。
- **路径修改**：多数脚本“参数集中在顶部”，请优先修改路径参数。

---

## 关键脚本索引
- 预处理：`scripts/data_processing/YOLO_input_preprocessing.py`
- YOLO 标注：`scripts/labeling/annotate_video_yolo.py`
- YOLO 复核 + 导出 clips：`scripts/labeling/yolo_review_and_crop.py`
- YOLO 训练：`scripts/training/YOLO_training.py`
- YOLO 划分：`scripts/training/split_yolo_dataset.py`
- 动作划分：`scripts/training/split_EfficientNet_dataset.py`
- 动作训练：`scripts/training/train_EfficientNet_gru.py`
- 一体化推理：`scripts/inference/mian.py`
