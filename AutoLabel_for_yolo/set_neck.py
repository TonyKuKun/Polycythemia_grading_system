"""Interactive tool to set channel-neck baselines by clicking on an image.

Usage:
    python set_neck.py <sample_dir> [--image FILENAME] [--scale SCALE]

Opens the first (or specified) image from <sample_dir>/images/ and lets
you click/drag two horizontal lines. Writes <sample_dir>/neck.json so
auto_label.py will use these values for every image in the sample.

Controls:
    左键点击 #1    set neck-top
    左键点击 #2    set neck-bottom
    右键点击       撤销最后一条标注的黄线
    'R'           重置所有点击  # 改为大写R
    'S'           save to neck.json and quit  # 改为大写S
    'q' / ESC     quit without saving

If auto-detection works well you do not need this tool — just run
auto_label.py directly.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", nargs='?', default=r"G:\23\model")
    ap.add_argument("--image", default=None,
                    help="specific filename to open (defaults to first)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor for display (1.0 = original size, 0.5 = 50%%)")
    args = ap.parse_args()

    # 验证样本目录和图像目录
    sample = Path(args.sample).resolve()
    images_dir = sample / "images"
    if not images_dir.is_dir():
        print(f"[ERROR] {images_dir} not found")
        sys.exit(1)

    # 选择要打开的图像
    if args.image is not None:
        img_path = images_dir / args.image
        if not img_path.exists():
            print(f"[ERROR] specified image {img_path} not found")
            sys.exit(1)
    else:
        # 查找目录下的有效图像文件
        files = sorted(p for p in images_dir.iterdir()
                       if p.suffix.lower() in VALID_EXT)
        if not files:
            print("[ERROR] no valid images found in {images_dir}")
            sys.exit(1)
        img_path = files[0]

    # 读取原图（保留完整分辨率）
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] could not read image file: {img_path}")
        sys.exit(1)

    # 获取原图尺寸
    H_orig, W_orig = img.shape[:2]
    print(f"[INFO] Original image size: {W_orig}x{H_orig}")

    # 应用缩放（默认1.0=原图，可通过--scale参数调整）
    scale = args.scale
    if scale != 1.0:
        W_disp = int(W_orig * scale)
        H_disp = int(H_orig * scale)
        disp = cv2.resize(img, (W_disp, H_disp), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Display size (scaled {scale}x): {W_disp}x{H_disp}")
    else:
        disp = img.copy()
        W_disp, H_disp = W_orig, H_orig
        print(f"[INFO] Displaying original size: {W_disp}x{H_disp}")

    # 状态变量：存储点击的坐标（显示坐标 + 原图坐标）
    state = {"clicks": []}

    def redraw():
        """重新绘制窗口（包含标注线、文字提示）"""
        canvas = disp.copy()

        # 绘制已点击的颈区线
        for y_disp, y_orig in state["clicks"]:
            cv2.line(canvas, (0, y_disp), (canvas.shape[1], y_disp),
                     (0, 255, 255), 2)

        # 绘制坐标标签
        if len(state["clicks"]) >= 1:
            y_disp, y_orig = state["clicks"][0]
            cv2.putText(canvas, f"top (orig y) = {y_orig}",
                        (8, max(y_disp - 6, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(state["clicks"]) >= 2:
            y_disp, y_orig = state["clicks"][1]
            cv2.putText(canvas, f"bottom (orig y) = {y_orig}",
                        (8, min(y_disp + 22, canvas.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制操作提示（仅修改快捷键提示为大写R/S）
        help_text = [
            "左键1: neck-top | 左键2: neck-bottom | 右键: 撤销最后一条线",
            "[R] 重置全部 | [S] 保存退出 | [q/ESC] 退出不保存",  # 改为大写R/S
            f"原图尺寸: {W_orig}x{H_orig} | 显示缩放: {scale}x",
            f"当前标注: {len(state['clicks'])}/2 条线"
        ]
        for i, text in enumerate(help_text):
            y_pos = 20 + i * 22
            cv2.putText(canvas, text, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        cv2.imshow("Set Channel Neck (Click to mark)", canvas)

    def on_mouse(event, x, y, flags, param):
        """鼠标事件处理：左键标注 + 右键撤销"""
        # 左键点击：添加标注线
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state["clicks"]) >= 2:
                return  # 最多点击2次（上/下边界）

            # 将显示坐标映射回原图坐标
            y_orig = int(y / scale)
            # 边界保护：确保坐标在原图范围内
            y_orig = max(0, min(y_orig, H_orig - 1))

            state["clicks"].append((y, y_orig))
            print(f"[INFO] 新增标注：显示y={y} → 原图y={y_orig} | 当前共{len(state['clicks'])}条线")
            redraw()

        # 右键点击：撤销最后一条标注线（核心新增功能）
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(state["clicks"]) == 0:
                print("[INFO] 暂无标注线可撤销")
                return

            # 弹出最后一条标注
            removed_y_disp, removed_y_orig = state["clicks"].pop()
            print(
                f"[INFO] 撤销标注：显示y={removed_y_disp} → 原图y={removed_y_orig} | 当前剩余{len(state['clicks'])}条线")
            redraw()

    # 创建窗口（支持调整大小 + 自适应图像）
    cv2.namedWindow("Set Channel Neck (Click to mark)", cv2.WINDOW_NORMAL)
    # 调整窗口初始大小为显示图像的尺寸
    cv2.resizeWindow("Set Channel Neck (Click to mark)", W_disp, H_disp)
    # 设置鼠标回调函数
    cv2.setMouseCallback("Set Channel Neck (Click to mark)", on_mouse)

    # 初始绘制
    redraw()

    # 主循环：处理键盘事件（仅修改监听大写R/S）
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):  # ESC / q：退出不保存
            print("[INFO] 退出且未保存")
            break
        elif key == ord("R"):  # 改为大写R：重置所有点击
            state["clicks"] = []
            print("[INFO] 重置所有标注线")
            redraw()
        elif key == ord("S"):  # 改为大写S：保存标注结果
            if len(state["clicks"]) < 2:
                print("[WARN] 需要标注2条线（上+下）才能保存！")
                continue

            # 提取并排序颈区上下边界（确保top < bottom）
            y1 = state["clicks"][0][1]
            y2 = state["clicks"][1][1]
            neck_top = min(y1, y2)
            neck_bottom = max(y1, y2)

            # 保存到neck.json
            neck_data = {
                "neck_top": int(neck_top),
                "neck_bottom": int(neck_bottom),
                "original_image": str(img_path.name),
                "original_size": f"{W_orig}x{H_orig}"
            }
            output_path = sample / "neck.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(neck_data, f, indent=2)

            print(f"[OK] 颈区标注已保存到 {output_path}:")
            print(f"  - 颈区上边界（原图y）: {neck_top}")
            print(f"  - 颈区下边界（原图y）: {neck_bottom}")
            break

    # 清理窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()