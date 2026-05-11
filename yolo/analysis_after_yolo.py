import cv2
import numpy as np
from collections import defaultdict
import csv
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import os


class Config:
    """配置类，存储全局参数（原有所有参数完全保留）"""
    YOLO_MODEL_PATH = r"C:\Users\Administrator\PycharmProjects\PythonProject\runs\detect\train3\weights\best.pt"
    VIDEO_PATH = r"D:\20\2\channel_2.mp4"
    OUTPUT_DIR = r"D:\20\2\channel_2"
    ERBC_CLASS_ID = 0  # 正常红细胞类别
    IRBC_CLASS_ID = 1  # 感染红细胞类别
    ORBC_CLASS_ID = 2  # 其他红细胞类别
    MIN_TRACK_LENGTH = 5
    MATCH_DISTANCE = 200
    FPS = 280
    PIXEL_TO_MICRON = 0.039
    CONF_THRESH = 0.3

    ROI_Y_MIN = None  # 中部狭窄区域上边界
    ROI_Y_MAX = None  # 中部狭窄区域下边界

    START_FRAME = 1
    END_FRAME = None

    MIN_IRBC_FRAMES = 4  # 门槛：仅保留IRBCFramesCount ≥4 的轨迹
    BOX_EXPAND_PIXEL = 10
    SQUARE_SIZE = 224
    ORIGINAL_FOLDER_NAME = "Full_Original_ERBC"
    RENAMED_FOLDER_NAME = "Full_Renamed_ERBC"

    FULL_FRAME_ORIGINAL = "Full_Frame_Original"
    FULL_FRAME_RENAMED = "Full_Frame_Renamed"

    # 新增：帧缓存上限，防止内存爆掉
    MAX_CACHE_FRAME = 50000


class IRBCTracker:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.tracks = defaultdict(list)
        self.next_id = 1
        self.prev_positions = {}
        self.global_stats = defaultdict(list)
        self.norm_params = {}
        self.valid_tracks = []
        self.keep_tids = set()
        # 帧内存缓存字典 {帧号: 原始frame图像}，核心提速组件
        self.frame_cache = dict()

    def select_roi_mid_zone(self):
        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {self.config.VIDEO_PATH}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_num = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("无法读取中间帧")

        h, w = frame.shape[:2]
        max_display_height = 800
        scale = min(1.0, max_display_height / h)
        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        print(f"图像已缩放至 {int(scale * 100)}%，请框选中部狭窄区域（按回车确认，按 c 取消）")

        roi = cv2.selectROI("Select ROI", resized_frame, showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, w_roi, h_roi = roi
        y_original = int(y / scale)
        h_original = int(h_roi / scale)

        self.config.ROI_Y_MIN = y_original
        self.config.ROI_Y_MAX = y_original + h_original
        print(f"已设置 ROI_Y_MIN = {self.config.ROI_Y_MIN}, ROI_Y_MAX = {self.config.ROI_Y_MAX}")

    def detect_irbc(self, frame, frame_num):
        results = self.model(frame, verbose=False, conf=self.config.CONF_THRESH)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id in [self.config.ERBC_CLASS_ID, self.config.IRBC_CLASS_ID, self.config.ORBC_CLASS_ID]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    if self.config.ROI_Y_MIN is not None and self.config.ROI_Y_MAX is not None:
                        if y1 < self.config.ROI_Y_MIN or y2 > self.config.ROI_Y_MAX:
                            continue
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    el = max(w, h)
                    ey = min(w, h)
                    di = (el - ey) / (el + ey + 1e-6)
                    detections.append({
                        'class_id': class_id,
                        'x': x, 'y': y, 'w': w, 'h': h, 'conf': conf,
                        'di': di
                    })
        return detections

    def update_tracks(self, detections, frame_num):
        current_boxes = [(d['x'], d['y'], d['w'], d['h'], d['conf'], d['class_id'], d['di']) for d in detections]
        used_indices = set()

        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if not track:
                continue

            last_frame, last_x, last_y, _, _, _, _, _ = track[-1]
            if frame_num - last_frame > 3:
                continue

            best_idx = None
            min_dist = float('inf')
            for idx, (x, y, _, _, _, _, _) in enumerate(current_boxes):
                if idx in used_indices:
                    continue
                dist = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                if dist < self.config.MATCH_DISTANCE and dist < min_dist:
                    min_dist = dist
                    best_idx = idx

            if best_idx is not None:
                x, y, w, h, conf, class_id, di = current_boxes[best_idx]
                track.append((frame_num, x, y, w, h, conf, class_id, di))
                used_indices.add(best_idx)

        for idx, (x, y, w, h, conf, class_id, di) in enumerate(current_boxes):
            if idx not in used_indices:
                self.tracks[self.next_id].append((frame_num, x, y, w, h, conf, class_id, di))
                self.next_id += 1

    def draw_tracks(self, frame, frame_num):
        for track_id, history in self.tracks.items():
            if len(history) > 0:
                curr_frame, x, y, w, h, conf, class_id, di = history[-1]
                if class_id != self.config.IRBC_CLASS_ID:
                    continue

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                x1 = max(0, min(x1, frame.shape[1]))
                y1 = max(0, min(y1, frame.shape[0]))
                x2 = max(0, min(x2, frame.shape[1]))
                y2 = max(0, min(y2, frame.shape[0]))

                color = (0, 255, 0)
                label = f"ID:{track_id} DI:{di:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def calculate_global_stats(self):
        all_displacements = []
        all_velocities = []
        all_accelerations = []
        all_areas = []
        all_confidences = []
        all_velocities_pixel = []
        all_dis = []

        for track_id, history in self.tracks.items():
            if len(history) < self.config.MIN_TRACK_LENGTH:
                continue

            for i in range(1, len(history)):
                prev_frame, prev_x, prev_y, prev_w, prev_h, prev_conf, _, prev_di = history[i - 1]
                curr_frame, curr_x, curr_y, curr_w, curr_h, curr_conf, _, curr_di = history[i]

                dx = curr_x - prev_x
                dy = curr_y - prev_y
                disp_pixel = np.sqrt(dx ** 2 + dy ** 2)
                disp_um = disp_pixel * self.config.PIXEL_TO_MICRON
                frame_diff = curr_frame - prev_frame
                vel_um = disp_um * self.config.FPS / frame_diff if frame_diff != 0 else 0
                vel_pixel_frame = disp_pixel / frame_diff if frame_diff != 0 else 0
                area = (curr_w * curr_h) * (self.config.PIXEL_TO_MICRON ** 2)

                all_displacements.append(disp_um)
                all_velocities.append(vel_um)
                all_velocities_pixel.append(vel_pixel_frame)
                all_areas.append(area)
                all_confidences.append(curr_conf)
                all_dis.append(curr_di)

                if i > 1:
                    prev_prev_frame, _, _, _, _, _, _, _ = history[i - 2]
                    prev_disp_pixel = np.sqrt((prev_x - history[i - 2][1]) ** 2 + (prev_y - history[i - 2][2]) ** 2)
                    prev_disp_um = prev_disp_pixel * self.config.PIXEL_TO_MICRON
                    prev_vel_um = prev_disp_um * self.config.FPS / (prev_frame - prev_prev_frame) if (
                                                                                                             prev_frame - prev_prev_frame) != 0 else 0
                    time_diff = (curr_frame - prev_frame) / self.config.FPS
                    accel = (vel_um - prev_vel_um) / time_diff if time_diff != 0 else 0
                    all_accelerations.append(accel)

        self.norm_params = {
            'displacement': self._calculate_stats(all_displacements),
            'velocity': self._calculate_stats(all_velocities),
            'acceleration': self._calculate_stats(all_accelerations),
            'area': self._calculate_stats(all_areas),
            'confidence': self._calculate_stats(all_confidences),
            'velocity_pixel': self._calculate_stats(all_velocities_pixel),
            'di': self._calculate_stats(all_dis)
        }

    def _calculate_stats(self, data):
        data = np.array(data)
        if data.size == 0:
            return {'min': 0, 'max': 1, 'mean': 0, 'std': 1}
        return {
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data)
        }

    def _zscore_normalize(self, value, param_key):
        params = self.norm_params.get(param_key, {'mean': 0, 'std': 1})
        return (value - params['mean']) / (params['std'] + 1e-8)

    def _minmax_normalize(self, value, param_key):
        params = self.norm_params.get(param_key, {'min': 0, 'max': 1})
        return (value - params['min']) / (params['max'] - params['min'] + 1e-8)

    # ===================== 强制关闭断点：永远返回False，必须从头跑 =====================
    def check_tables_exist(self):
        return False  # 强制关闭断点，永不跳过

    # ===================== 【表格逻辑完全不变】仅保留 IRBCFramesCount ≥4 =====================
    def save_results_filtered(self):
        print("\n📊 开始生成统计表格（仅保留 IRBCFramesCount ≥4）")
        self.calculate_global_stats()
        track_stats = {}

        # 第一次遍历：统计所有轨迹参数
        for track_id, history in self.tracks.items():
            if len(history) < self.config.MIN_TRACK_LENGTH:
                continue

            irbc_frames = [h[0] for h in history if h[6] == self.config.IRBC_CLASS_ID]
            cnt = len(irbc_frames)
            # 严格过滤门槛
            if cnt < self.config.MIN_IRBC_FRAMES:
                continue

            first_irbc_frame = min(irbc_frames) if irbc_frames else None
            last_irbc_frame = max(irbc_frames) if irbc_frames else None

            final_state = None
            if history:
                last_class = history[-1][6]
                if last_class == self.config.ERBC_CLASS_ID:
                    final_state = "E"
                elif last_class == self.config.IRBC_CLASS_ID:
                    final_state = "I"
                elif last_class == self.config.ORBC_CLASS_ID:
                    final_state = "O"

            frames = [h[0] for h in history]
            displacements = [
                np.sqrt((h[1] - history[i - 1][1]) ** 2 + (h[2] - history[i - 1][2]) ** 2) * self.config.PIXEL_TO_MICRON
                for i, h in enumerate(history) if i > 0]
            velocities_um = [d * self.config.FPS / (frames[i] - frames[i - 1]) for i, d in enumerate(displacements) if
                             (frames[i] - frames[i - 1]) != 0]
            velocities_pixel = [
                np.sqrt((h[1] - history[i - 1][1]) ** 2 + (h[2] - history[i - 1][2]) ** 2) / (frames[i] - frames[i - 1])
                for i, h in enumerate(history) if i > 0 and (frames[i] - frames[i - 1]) != 0]
            areas = [(h[3] * h[4]) * (self.config.PIXEL_TO_MICRON ** 2) for h in history]
            confidences = [h[5] for h in history]
            dis = [h[7] for h in history if h[6] == self.config.IRBC_CLASS_ID]

            total_dist = sum(displacements)
            avg_vel_um = np.mean(velocities_um) if velocities_um else 0
            max_vel_um = np.max(velocities_um) if velocities_um else 0
            min_vel_um = np.min(velocities_um) if velocities_um else 0
            avg_area = np.mean(areas) if areas else 0
            avg_conf = np.mean(confidences) if confidences else 0
            avg_vel_pixel = np.mean(velocities_pixel) if velocities_pixel else 0
            max_vel_pixel = np.max(velocities_pixel) if velocities_pixel else 0
            min_vel_pixel = np.min(velocities_pixel) if velocities_pixel else 0
            avg_di = np.mean(dis) if dis else 0

            track_stats[track_id] = {
                'first_irbc_frame': first_irbc_frame,
                'first_irbc_time': first_irbc_frame / self.config.FPS if first_irbc_frame else None,
                'last_irbc_frame': last_irbc_frame,
                'last_irbc_time': last_irbc_frame / self.config.FPS if last_irbc_frame else None,
                'transition_time': (
                                               last_irbc_frame - first_irbc_frame) / self.config.FPS if first_irbc_frame and last_irbc_frame else None,
                'final_state': final_state,
                'total_dist': total_dist,
                'avg_vel_um': avg_vel_um,
                'max_vel_um': max_vel_um,
                'min_vel_um': min_vel_um,
                'avg_area': avg_area,
                'avg_conf': avg_conf,
                'avg_vel_pixel': avg_vel_pixel,
                'max_vel_pixel': max_vel_pixel,
                'min_vel_pixel': min_vel_pixel,
                'avg_di': avg_di,
                'irbc_frames_count': cnt
            }

        self.keep_tids = set(track_stats.keys())
        print(f"✅ 符合条件 IRBCFramesCount ≥{self.config.MIN_IRBC_FRAMES} 的轨迹数：{len(self.keep_tids)}")

        # 写入过滤后summary.csv
        sum_path = Path(self.config.OUTPUT_DIR) / "irbc_summary.csv"
        with open(sum_path, "w", newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                "TrackID", "Frames", "Duration(s)",
                "FirstIRBCFrame", "FirstIRBCTime(s)",
                "LastIRBCFrame", "LastIRBCTime(s)",
                "TransitionTime(s)", "IRBCFramesCount",
                "TotalDistance(um)", "TotalDistance(um)_Norm", "TotalDistance(um)_ZNorm",
                "AvgArea(um²)", "AvgArea(um²)_Norm", "AvgArea(um²)_ZNorm",
                "AvgConfidence", "AvgConfidence_Norm",
                "AvgVelocity(pixel/frame)", "AvgVelocity(pixel/frame)_Norm", "AvgVelocity(pixel/frame)_ZNorm",
                "MaxVelocity(pixel/frame)", "MaxVelocity(pixel/frame)_Norm", "MaxVelocity(pixel/frame)_ZNorm",
                "MinVelocity(pixel/frame)", "MinVelocity(pixel/frame)_Norm", "MinVelocity(pixel/frame)_ZNorm",
                "Average_DI"
            ])
            for tid, stats in track_stats.items():
                writer.writerow([
                    tid, len(self.tracks[tid]), f"{len(self.tracks[tid]) / self.config.FPS:.3f}",
                    stats['first_irbc_frame'] if stats['first_irbc_frame'] else "",
                    f"{stats['first_irbc_time']:.3f}" if stats['first_irbc_time'] else "",
                    stats['last_irbc_frame'] if stats['last_irbc_frame'] else "",
                    f"{stats['last_irbc_time']:.3f}" if stats['last_irbc_time'] else "",
                    f"{stats['transition_time']:.3f}" if stats['transition_time'] else "",
                    stats['irbc_frames_count'],
                    f"{stats['total_dist']:.2f}",
                    f"{self._minmax_normalize(stats['total_dist'], 'displacement'):.4f}",
                    f"{self._zscore_normalize(stats['total_dist'], 'displacement'):.4f}",
                    f"{stats['avg_area']:.2f}",
                    f"{self._minmax_normalize(stats['avg_area'], 'area'):.4f}",
                    f"{self._zscore_normalize(stats['avg_area'], 'area'):.4f}",
                    f"{stats['avg_conf']:.2f}",
                    f"{self._minmax_normalize(stats['avg_conf'], 'confidence'):.4f}",
                    f"{stats['avg_vel_pixel']:.2f}",
                    f"{self._minmax_normalize(stats['avg_vel_pixel'], 'velocity_pixel'):.4f}",
                    f"{self._zscore_normalize(stats['avg_vel_pixel'], 'velocity_pixel'):.4f}",
                    f"{stats['max_vel_pixel']:.2f}",
                    f"{self._minmax_normalize(stats['max_vel_pixel'], 'velocity_pixel'):.4f}",
                    f"{self._zscore_normalize(stats['max_vel_pixel'], 'velocity_pixel'):.4f}",
                    f"{stats['min_vel_pixel']:.2f}",
                    f"{self._minmax_normalize(stats['min_vel_pixel'], 'velocity_pixel'):.4f}",
                    f"{self._zscore_normalize(stats['min_vel_pixel'], 'velocity_pixel'):.4f}",
                    f"{stats['avg_di']:.4f}"
                ])

        # 写入过滤后detail.csv
        det_path = Path(self.config.OUTPUT_DIR) / "irbc_details.csv"
        with open(det_path, "w", newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                "TrackID", "Frame", "Time(s)", "FinalState",
                "FirstIRBCFrame", "FirstIRBCTime(s)",
                "LastIRBCFrame", "LastIRBCTime(s)",
                "X(pixel)", "Y(pixel)", "Width(pixel)", "Height(pixel)", "Area(um²)",
                "Displacement(pixel)", "Displacement(um)",
                "Displacement(um)_Norm", "Displacement(um)_ZNorm",
                "Direction(°)", "Velocity(um/s)",
                "Velocity(um/s)_Norm", "Velocity(um/s)_ZNorm",
                "Acceleration(um/s²)", "Acceleration(um/s²)_Norm", "Acceleration(um/s²)_ZNorm",
                "Confidence", "Confidence_Norm",
                "Velocity(pixel/frame)", "Velocity(pixel/frame)_Norm", "Velocity(pixel/frame)_ZNorm",
                "DeformationIndex"
            ])
            for tid in self.keep_tids:
                history = self.tracks[tid]
                stats = track_stats[tid]
                for i in range(1, len(history)):
                    curr_frame, curr_x, curr_y, curr_w, curr_h, curr_conf, curr_class, curr_di = history[i]
                    prev_frame, prev_x, prev_y, prev_w, prev_h, prev_conf, prev_class, prev_di = history[i - 1]

                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    disp_pixel = np.sqrt(dx ** 2 + dy ** 2)
                    disp_um = disp_pixel * self.config.PIXEL_TO_MICRON
                    frame_diff = curr_frame - prev_frame
                    vel_um = disp_um * self.config.FPS / frame_diff if frame_diff != 0 else 0
                    vel_pixel_frame = disp_pixel / frame_diff if frame_diff != 0 else 0
                    direction = np.degrees(np.arctan2(dy, dx)) % 360
                    area = (curr_w * curr_h) * (self.config.PIXEL_TO_MICRON ** 2)

                    accel = 0
                    if i > 1:
                        prev_prev_frame, _, _, _, _, _, _, _ = history[i - 2]
                        prev_disp_pixel = np.sqrt((prev_x - history[i - 2][1]) ** 2 + (prev_y - history[i - 2][2]) ** 2)
                        prev_disp_um = prev_disp_pixel * self.config.PIXEL_TO_MICRON
                        prev_vel_um = prev_disp_um * self.config.FPS / (prev_frame - prev_prev_frame) if (
                                                                                                                 prev_frame - prev_prev_frame) != 0 else 0
                        time_diff = (curr_frame - prev_frame) / self.config.FPS
                        accel = (vel_um - prev_vel_um) / time_diff if time_diff != 0 else 0

                    writer.writerow([
                        tid, curr_frame, f"{curr_frame / self.config.FPS:.3f}", stats['final_state'],
                        stats['first_irbc_frame'] if stats['first_irbc_frame'] else "",
                        f"{stats['first_irbc_time']:.3f}" if stats['first_irbc_time'] else "",
                        stats['last_irbc_frame'] if stats['last_irbc_frame'] else "",
                        f"{stats['last_irbc_time']:.3f}" if stats['last_irbc_time'] else "",
                        f"{curr_x:.2f}", f"{curr_y:.2f}", curr_w, curr_h, f"{area:.2f}",
                        f"{disp_pixel:.2f}", f"{disp_um:.2f}",
                        f"{self._minmax_normalize(disp_um, 'displacement'):.4f}",
                        f"{self._zscore_normalize(disp_um, 'displacement'):.4f}",
                        f"{direction:.1f}", f"{vel_um:.2f}",
                        f"{self._minmax_normalize(vel_um, 'velocity'):.4f}",
                        f"{self._zscore_normalize(vel_um, 'velocity'):.4f}",
                        f"{accel:.2f}",
                        f"{self._minmax_normalize(accel, 'acceleration'):.4f}",
                        f"{self._zscore_normalize(accel, 'acceleration'):.4f}",
                        f"{curr_conf:.2f}",
                        f"{self._minmax_normalize(curr_conf, 'confidence'):.4f}",
                        f"{vel_pixel_frame:.2f}",
                        f"{self._minmax_normalize(vel_pixel_frame, 'velocity_pixel'):.4f}",
                        f"{self._zscore_normalize(vel_pixel_frame, 'velocity_pixel'):.4f}",
                        f"{curr_di:.4f}"
                    ])
        print("✅ 表格已生成（仅保留 ≥4 轨迹）")

    # ===================== 【核心极致提速重构】预加载帧缓存 + 零随机寻址保存图像 =====================
    def preload_all_needed_frames(self):
        """一次性预加载所有轨迹需要用到的帧到内存字典，彻底删掉循环内cap.set随机寻址"""
        print("\n🔄 预加载所有所需视频帧到内存（核心提速步骤）")
        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f"视频打开失败: {self.config.VIDEO_PATH}")

        total_video_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 收集所有有效轨迹需要的全部帧号
        need_frame_set = set()
        for tid in self.keep_tids:
            track = self.tracks[tid]
            irbc_list = [h for h in track if h[6] == self.config.IRBC_CLASS_ID]
            if not irbc_list: continue
            first_irbc = irbc_list[0][0]
            last_irbc = irbc_list[-1][0]
            # E帧范围: IRBC前5帧、I帧全部、O帧范围: IRBC后5帧
            e_start = max(0, first_irbc - 5)
            o_end = min(total_video_frame - 1, last_irbc + 5)
            # 加入全范围帧号
            need_frame_set.update(range(e_start, o_end + 1))

        need_frame_list = sorted(list(need_frame_set))
        print(f"📦 本次总共需要加载 {len(need_frame_list)} 帧图像")

        # 流式顺序加载（视频顺序读是最快的！OpenCV原生最优速度），全部存入内存缓存
        self.frame_cache.clear()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_frame_idx = 0
        with tqdm(total=len(need_frame_list), desc="帧预加载") as pbar:
            target_ptr = 0
            while cap.isOpened() and target_ptr < len(need_frame_list):
                ret, frame = cap.read()
                if not ret: break
                if current_frame_idx == need_frame_list[target_ptr]:
                    self.frame_cache[current_frame_idx] = frame
                    target_ptr += 1
                    pbar.update(1)
                current_frame_idx += 1
        cap.release()
        print(f"✅ 帧预加载完成，内存缓存构建完毕")

    def save_perfect_irbc_frames_ultra_fast(self):
        """极速版图像保存：全程从内存缓存取图，无视频寻址、无重复解码"""
        print("\n🖼️ 开始极速保存 IRBC_Data 图像（仅保留有效≥4轨迹）")
        BOX_SCALE_FACTOR = 1.0
        save_root = Path(self.config.OUTPUT_DIR) / "IRBC_Data"
        save_root.mkdir(exist_ok=True)
        total_video_frame = int(cv2.VideoCapture(self.config.VIDEO_PATH).get(cv2.CAP_PROP_FRAME_COUNT))

        valid_tracks = {tid: self.tracks[tid] for tid in self.keep_tids}

        # 轨迹循环带进度条
        for tid, track in tqdm(valid_tracks.items(), desc="极速保存图像"):
            irbc_frames = sorted([h for h in track if h[6] == self.config.IRBC_CLASS_ID])
            if not irbc_frames:
                continue
            first_irbc = irbc_frames[0][0]
            last_irbc = irbc_frames[-1][0]
            track_dir = save_root / f"TRACK_{tid}"
            track_dir.mkdir(exist_ok=True)

            # 封装缓存取图函数（O(1)内存查询，零耗时）
            def get_cached_frame(fnum):
                return self.frame_cache.get(fnum, None)

            # 封装轨迹内坐标查询
            def get_track_pos(fnum):
                for d in track:
                    if d[0] == fnum:
                        return d[1], d[2], d[3], d[4]
                return None, None, None, None

            # ========== 1. 保存IRBC感染帧(I帧) ==========
            for d in irbc_frames:
                fn = d[0]
                frame = get_cached_frame(fn)
                if frame is None: continue
                is_first = (fn == first_irbc)
                is_last = (fn == last_irbc)
                thickness = 3 if (is_first or is_last) else 2

                cx, cy, w, h = d[1], d[2], d[3], d[4]
                x1 = int(cx - w * BOX_SCALE_FACTOR / 2)
                y1 = int(cy - h * BOX_SCALE_FACTOR / 2)
                x2 = int(cx + w * BOX_SCALE_FACTOR / 2)
                y2 = int(cy + h * BOX_SCALE_FACTOR / 2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # 画图+低压缩极速保存
                draw_frame = frame.copy()
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                prefix = "FIRST_" if is_first else "LAST_" if is_last else ""
                # OpenCV imwrite优化：JPEG压缩质量拉满速度，关闭冗余校验
                cv2.imwrite(str(track_dir / f"{prefix}I_{fn}.png"), draw_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # ========== 2. 保存感染前ERBC帧(E帧) ==========
            e_start = max(0, first_irbc - 5)
            for fn in range(e_start, first_irbc):
                frame = get_cached_frame(fn)
                if frame is None: continue
                cx, cy, w, h = get_track_pos(fn)
                draw_frame = frame.copy()
                if cx is not None:
                    x1 = int(cx - w * BOX_SCALE_FACTOR / 2)
                    y1 = int(cy - h * BOX_SCALE_FACTOR / 2)
                    x2 = int(cx + w * BOX_SCALE_FACTOR / 2)
                    y2 = int(cy + h * BOX_SCALE_FACTOR / 2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(track_dir / f"E_{fn}.png"), draw_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # ========== 3. 保存感染后ORBC帧(O帧) ==========
            o_end = min(total_video_frame - 1, last_irbc + 5)
            for fn in range(last_irbc + 1, o_end + 1):
                frame = get_cached_frame(fn)
                if frame is None: continue
                cx, cy, w, h = get_track_pos(fn)
                draw_frame = frame.copy()
                if cx is not None:
                    x1 = int(cx - w * BOX_SCALE_FACTOR / 2)
                    y1 = int(cy - h * BOX_SCALE_FACTOR / 2)
                    x2 = int(cx + w * BOX_SCALE_FACTOR / 2)
                    y2 = int(cy + h * BOX_SCALE_FACTOR / 2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(track_dir / f"O_{fn}.png"), draw_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        print("✅ 全部图像极速保存完成")

    # ===================== 原有极速清理逻辑完全保留 =====================
    def fast_clean_irbc_data(self):
        print("\n🧹 快速清理无效文件夹（批量加速版）")
        irbc_dir = Path(self.config.OUTPUT_DIR) / "IRBC_Data"
        if not irbc_dir.exists():
            print("ℹ️ 无 IRBC_Data 文件夹，跳过清理")
            return

        all_folders = [f for f in irbc_dir.iterdir() if f.is_dir() and f.name.startswith("TRACK_")]
        to_delete = []

        for f in tqdm(all_folders, desc="扫描文件夹"):
            try:
                tid_part = f.name.split("_")[1]
                tid = int(tid_part)
                if tid not in self.keep_tids:
                    to_delete.append(f)
            except:
                continue

        print(f"🗑️ 待删除无效文件夹总数：{len(to_delete)}")
        for fd in tqdm(to_delete, desc="批量删除中"):
            shutil.rmtree(fd, ignore_errors=True)

        print(f"✅ 清理完成，最终保留有效文件夹：{len(all_folders) - len(to_delete)} 个")

    # ===================== 主流程：强制从头跑，不跳过任何一步 =====================
    def process_video(self):
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # ROI区域框选
        if self.config.ROI_Y_MIN is None:
            self.select_roi_mid_zone()

        # 强制从头跑视频，不跳过
        print("\n🎥 开始处理视频帧检测与轨迹追踪")
        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        s = self.config.START_FRAME
        e = self.config.END_FRAME or total
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)

        with tqdm(total=e - s, desc="视频检测追踪") as pbar:
            fn = s
            while fn < e:
                ret, frame = cap.read()
                if not ret:
                    break
                det = self.detect_irbc(frame, fn)
                self.update_tracks(det, fn)
                self.draw_tracks(frame, fn)
                pbar.update(1)
                fn += 1
        cap.release()
        # 生成过滤后表格
        self.save_results_filtered()

        # 极速版图像保存
        self.preload_all_needed_frames()
        self.save_perfect_irbc_frames_ultra_fast()

        # 批量极速清理
        self.fast_clean_irbc_data()

        print("\n🎉 全流程全部完成！仅保留 IRBCFramesCount ≥4 所有数据！")

    def load_valid_tids_from_summary(self):
        pass


if __name__ == "__main__":
    tracker = IRBCTracker(Config())
    tracker.process_video()