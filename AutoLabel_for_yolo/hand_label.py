import cv2
import numpy as np
import os
import torch
import json
from contextlib import nullcontext
from segment_anything import SamPredictor, sam_model_registry

# ===================== 配置 =====================
MODEL_PATH = r"E:\blood\sam_vit_b_01ec64.pth"
IMAGE_FOLDER = r"H:\20\model\1\images"
OUTPUT_DIR = r"H:\20\model\1\visualization2"

WINDOW_NAME = "SAM标注工具"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

# 简化目录：只在 OUTPUT_DIR 下放 label/ 和 visualization/
LABEL_DIR = os.path.join(OUTPUT_DIR, "label")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

LABELS = [
    {"id": 0, "name": "ERBC", "color": (0, 0, 255)},   # 红 - 进通道
    {"id": 1, "name": "IRBC", "color": (255, 0, 0)},   # 蓝 - 在通道
    {"id": 2, "name": "ORBC", "color": (0, 255, 0)},   # 绿 - 出通道
]

# 鼠标移动节流：距离上次预测多少像素以内不重新预测
HOVER_THROTTLE_PX = 8


# ================================================

def get_autocast_context(device):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def draw_count(canvas, count):
    h, w = canvas.shape[:2]
    cv2.putText(canvas, f"Count: {count}", (w - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return canvas


def make_bbox_visualization(img, annotations):
    """保存用的可视化：纯方框（与 .txt 标签内容对应）。"""
    vis = img.copy()
    for a in annotations:
        bx, by, bw, bh = a["bbox"]
        color = LABELS[a["lbl"]]["color"]
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(vis, LABELS[a["lbl"]]["name"],
                    (bx, max(15, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("模型不存在")
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError("图片文件夹不存在")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  未检测到 GPU，悬停预览会比较卡顿。")
    cv2.setUseOptimized(True)

    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    img_files = sorted([f for f in os.listdir(IMAGE_FOLDER)
                        if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))])

    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            start_idx = json.load(f).get("next_img_idx", 0)

    for idx, img_file in enumerate(img_files):
        if idx < start_idx:
            continue

        img = cv2.imread(os.path.join(IMAGE_FOLDER, img_file))
        if img is None:
            print(f"⚠️ 跳过无法读取的图片：{img_file}")
            continue
        ih, iw = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.inference_mode(), get_autocast_context(device):
            predictor.set_image(img_rgb)

        annotations = []
        # 当前轮廓预览（鼠标悬停时显示，点击后清空，新移动时重建）
        state = {
            "lbl": 0,
            "hover_cnt": None,
            "last_pred_xy": (-1000, -1000),
        }
        canvas = img.copy()
        canvas = draw_count(canvas, 0)

        def predict_contour(x, y):
            """SAM 预测一次，返回最大轮廓或 None。"""
            with torch.inference_mode(), get_autocast_context(device):
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )
                mask = masks[np.argmax(scores)]
            cnts, _ = cv2.findContours(
                (mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not cnts:
                return None
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 20:
                return None
            return c

        def redraw():
            """交互画面：保存的标注画粗轮廓，悬停预览画半透明填充 + 细轮廓。"""
            nonlocal canvas
            canvas = img.copy()

            # 已确认的标注：粗轮廓（不规则形状）
            for a in annotations:
                cv2.drawContours(canvas, [a["cnt"]], -1,
                                 LABELS[a["lbl"]]["color"], 2)

            # 悬停预览：用当前选中的标签颜色，半透明填充 + 1px 描边
            if state["hover_cnt"] is not None:
                color = LABELS[state["lbl"]]["color"]
                overlay = canvas.copy()
                cv2.drawContours(overlay, [state["hover_cnt"]],
                                 -1, color, thickness=cv2.FILLED)
                cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)
                cv2.drawContours(canvas, [state["hover_cnt"]],
                                 -1, color, 1)

            canvas = draw_count(canvas, len(annotations))
            cv2.imshow(WINDOW_NAME, canvas)

        def mouse(event, x, y, flags, param):
            nonlocal annotations

            if event == cv2.EVENT_MOUSEMOVE:
                # 节流：移动距离 < HOVER_THROTTLE_PX 不重新预测
                lx, ly = state["last_pred_xy"]
                if (x - lx) ** 2 + (y - ly) ** 2 < HOVER_THROTTLE_PX ** 2:
                    return
                state["last_pred_xy"] = (x, y)
                state["hover_cnt"] = predict_contour(x, y)
                redraw()

            elif event == cv2.EVENT_LBUTTONDOWN:
                # 直接复用当前悬停预览（鼠标位置 = 点击位置）
                # 兜底：如果还没预览过就现场预测一次
                # 注意：numpy 数组不能用 `or` 做真值判断（会报
                # "truth value ambiguous"），必须显式 is not None。
                c = state["hover_cnt"]
                if c is None:
                    c = predict_contour(x, y)
                if c is None:
                    return
                bx, by, bw, bh = cv2.boundingRect(c)
                # YOLO 检测格式：class cx cy w h（归一化）
                cx_n = (bx + bw / 2) / iw
                cy_n = (by + bh / 2) / ih
                nw_n = bw / iw
                nh_n = bh / ih
                annotations.append({
                    "lbl": state["lbl"],
                    "coord": f"{cx_n:.6f} {cy_n:.6f} {nw_n:.6f} {nh_n:.6f}",
                    "bbox": (bx, by, bw, bh),
                    "cnt": c,  # 仅交互期使用，不保存
                })
                state["hover_cnt"] = None
                state["last_pred_xy"] = (-1000, -1000)
                redraw()

            elif event == cv2.EVENT_RBUTTONDOWN and annotations:
                annotations.pop()
                state["hover_cnt"] = None
                state["last_pred_xy"] = (-1000, -1000)
                redraw()

        # ======================
        # 窗口：屏幕正居中
        # ======================
        max_w, max_h = 1600, 900
        scale = min(max_w / iw, max_h / ih, 1)
        nw_disp, nh_disp = int(iw * scale), int(ih * scale)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, nw_disp, nh_disp)

        screen_w = 1920
        screen_h = 1080
        win_x = (screen_w - nw_disp) // 2
        win_y = (screen_h - nh_disp) // 2
        cv2.moveWindow(WINDOW_NAME, win_x, win_y)

        cv2.imshow(WINDOW_NAME, canvas)
        cv2.setMouseCallback(WINDOW_NAME, mouse)

        print(f"\n===== 标注：{img_file} =====")
        print("✅ 鼠标悬停即预览 | 左键确认 | 右键撤销")
        print("✅ 按 0/1/2 切换标签（进/在/出 通道）")
        print("✅ ENTER 保存并下一张 | ESC 退出（断点保存）")

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 13:
                break
            if k == 27:
                cv2.destroyAllWindows()
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump({"next_img_idx": idx}, f)
                print("\n🛑 已保存断点，下次从这里继续")
                return
            if k == ord('0'):
                state["lbl"] = 0
                print("当前标签：0 ERBC(红) - 进通道")
                redraw()  # 颜色随当前标签变
            if k == ord('1'):
                state["lbl"] = 1
                print("当前标签：1 IRBC(蓝) - 在通道")
                redraw()
            if k == ord('2'):
                state["lbl"] = 2
                print("当前标签：2 ORBC(绿) - 出通道")
                redraw()

        # ======================
        # 保存：方框 .txt + 方框 visualization
        # ======================
        name = os.path.splitext(img_file)[0]
        with open(os.path.join(LABEL_DIR, f"{name}.txt"), "w") as f:
            for a in annotations:
                f.write(f"{a['lbl']} {a['coord']}\n")

        vis = make_bbox_visualization(img, annotations)
        cv2.imwrite(os.path.join(VIS_DIR, img_file), vis)

        cv2.destroyWindow(WINDOW_NAME)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"next_img_idx": idx + 1}, f)

        print(f"✅ 已保存：{img_file}（{len(annotations)} 个标注）")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("\n🎉 全部标注完成！")


if __name__ == "__main__":
    main()