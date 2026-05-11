import os
import cv2
import time
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm
import sys


def collect_processing_tasks(output_dir, input_dir):
    """调整帧号提取逻辑：从文件名第一个字段提取帧号"""
    task_dict = defaultdict(list)
    input_images = {}

    # 收集原始图像（建立帧号与路径的映射）
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                image_path = os.path.join(root, file)
                # 原始图像的帧号为文件名本身（不含扩展名），例如 "785.png" 的帧号是 "785"
                frame_str = os.path.splitext(file)[0]  # 不补0，保持原始字符串匹配
                input_images[frame_str] = image_path
                # 同时存储补0后的版本，提高兼容性（例如 "785" 和 "000785" 都能匹配）
                input_images[frame_str.zfill(6)] = image_path

    # 解析待处理文件（从第一个字段提取帧号）
    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"正在处理文件夹: {folder_path}")
        for filename in os.listdir(folder_path):
            if not filename.endswith('.tif'):
                continue

            try:
                file_path = os.path.join(folder_path, filename)
                print(f"正在处理图片: {file_path}")
                base_name = os.path.splitext(filename)[0]
                parts = base_name.split('_')

                # 文件名格式：[帧号]_[其他字段]_[x1]_[y1]_[x2]_[y2]_[额外信息]
                # 从第一个字段提取帧号（parts[0]）
                if len(parts) < 7:
                    print(f"文件名格式错误: {filename}（至少需要7个字段）")
                    continue

                # 提取帧号（第一个字段，如 "785"）
                frame_str = parts[0]
                # 尝试多种匹配方式：原始字符串、补0后字符串
                match_candidates = [frame_str, frame_str.zfill(6)]
                matched_frame = None
                for candidate in match_candidates:
                    if candidate in input_images:
                        matched_frame = candidate
                        break

                if not matched_frame:
                    print(f"未找到对应原始图像: {filename}（尝试帧号: {match_candidates}）")
                    continue

                original_path = input_images[matched_frame]

                # 提取坐标（索引3-6：x1, y1, x2, y2）
                x1, y1, x2, y2 = map(int, parts[3:7])
                # 自动修正坐标顺序
                xmin, xmax = (x1, x2) if x1 < x2 else (x2, x1)
                ymin, ymax = (y1, y2) if y1 < y2 else (y2, y1)

                if xmin >= xmax or ymin >= ymax:
                    print(f"无效坐标: {filename}（修正后: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}）")
                    continue

                task_info = {
                    'output_folder': folder_name,
                    'filename': filename,
                    'coords': (xmin, ymin, xmax, ymax)
                }
                task_dict[original_path].append(task_info)

            except Exception as e:
                print(f"解析 {filename} 失败: {str(e)}")

    return task_dict


def process_single_image(original_path, tasks, output_pro_dir, padding=20, new_size=(224, 224)):
    """处理单张图像（保持不变）"""
    try:
        original_image = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original_image is None:
            print(f"无法读取原始图像: {original_path}")
            return 0

        h, w = original_image.shape[:2]
        processed_count = 0

        for task in tasks:
            output_folder = os.path.join(output_pro_dir, task['output_folder'])
            os.makedirs(output_folder, exist_ok=True)
            xmin, ymin, xmax, ymax = task['coords']

            # 计算带padding的裁剪区域
            ymin_pad = max(0, ymin - padding)
            ymax_pad = min(h, ymax + padding)
            xmin_pad = max(0, xmin - padding)
            xmax_pad = min(w, xmax + padding)

            if ymin_pad >= ymax_pad or xmin_pad >= xmax_pad:
                print(f"裁剪区域为空: {task['filename']}（调整后: {xmin_pad}-{xmax_pad}, {ymin_pad}-{ymax_pad}）")
                continue

            cropped = original_image[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
            resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)

            output_path = os.path.join(output_folder, task['filename'])
            cv2.imwrite(output_path, resized, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            processed_count += 1

        return processed_count
    except Exception as e:
        print(f"处理 {original_path} 失败: {str(e)}")
        return 0


def optimized_pipeline(output_dir, input_dir, output_pro_dir, padding=20, new_size=(224, 224), workers=4):
    print("收集处理任务中...")
    task_dict = collect_processing_tasks(output_dir, input_dir)

    print(f"开始处理 {len(task_dict)} 张原始图像...")
    total = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                original_path,
                tasks,
                output_pro_dir,
                padding,
                new_size
            ) for original_path, tasks in task_dict.items()
        ]

        with tqdm(total=len(futures), desc="处理进度") as pbar:
            for future in concurrent.futures.as_completed(futures):
                total += future.result()
                pbar.update(1)

    print(f"总处理图像数: {total}")


if __name__ == "__main__":
    output_root = ""
    input_root = ""
    output_pro_root = ""
    choice = 2  # 1: 单文件夹处理, 2: 批量处理

    if len(sys.argv) >= 4:
        output_root = sys.argv[1]
        input_root = sys.argv[2]
        output_pro_root = sys.argv[3]

    start_time = time.time()

    if choice == 1:
        os.makedirs(output_pro_root, exist_ok=True)
        optimized_pipeline(
            output_dir=output_root,
            input_dir=input_root,
            output_pro_dir=output_pro_root,
            workers=4,
            padding=20,
            new_size=(224, 224)
        )
    elif choice == 2:
        for folder_name in os.listdir(input_root):
            input_path = os.path.join(input_root, folder_name)
            if os.path.isdir(input_path):
                output_path = os.path.join(output_root, folder_name)
                output_pro_path = os.path.join(output_pro_root, folder_name)
                os.makedirs(output_pro_path, exist_ok=True)
                print(f"正在处理文件夹: {output_pro_path}")
                optimized_pipeline(
                    output_dir=output_path,
                    input_dir=input_path,
                    output_pro_dir=output_pro_path,
                    workers=8,
                    padding=20,
                    new_size=(224, 224)
                )

    print(f"总耗时: {time.time() - start_time:.2f} 秒")