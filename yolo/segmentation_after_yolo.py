import os
import shutil
import time
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
import re
import gc


# ===================== 第一部分：路径配置与文件夹复制筛选 (来自extra.py) =====================
def copy_selected_folders():
    """读取ID列表，复制指定的TRACK_*文件夹并重命名为纯数字"""
    # 路径配置
    BASE_DIR = r"M:\pp\bai11\channel_2"
    IDS_TXT = os.path.join(BASE_DIR, "final_valid_frame_id_frames.txt")
    SOURCE_FOLDER = os.path.join(BASE_DIR, "Full_Frame_Renamed")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "matlabphotos2")

    # 创建输出目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 读取需要的ID
    target_ids = set()
    with open(IDS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for part in line.replace("，", ",").split(","):
                pid = part.strip()
                if pid.isdigit():
                    target_ids.add(pid)

    print(f"✅ 提取ID列表: {sorted(target_ids)}")

    # 复制文件夹并重命名为纯数字
    success = 0
    for folder_name in os.listdir(SOURCE_FOLDER):
        src_path = os.path.join(SOURCE_FOLDER, folder_name)
        if not os.path.isdir(src_path):
            continue

        # 匹配 TRACK_数字 格式
        if not folder_name.startswith("TRACK_"):
            continue

        track_id = folder_name.replace("TRACK_", "")
        if track_id not in target_ids:
            continue

        # 目标文件夹名 = 纯数字
        dst_path = os.path.join(OUTPUT_FOLDER, track_id)

        # 复制整个文件夹
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        print(f"✅ 已复制: {folder_name} → {dst_path}")
        success += 1

    print(f"\n🎉 文件夹复制完成！共提取 {success} 个文件夹")
    print(f"📁 输出路径: {OUTPUT_FOLDER}")
    return OUTPUT_FOLDER  # 返回筛选后的文件夹根路径


# ===================== 第二部分：细胞分割核心函数 (来自yolo-sam.py) =====================
def load_cell_templates(patient_dir):
    """加载 cell1.png, cell2.png, cell3.png 三个模板，并自动裁剪背景"""
    cell_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp']
    templates = {}  # name -> (rgb, path)

    for cell_name in ['cell2', 'cell1', 'cell3']:  # cell2 排在最前
        for ext in cell_extensions:
            cell_path = os.path.join(patient_dir, f"{cell_name}{ext}")
            if os.path.exists(cell_path):
                img = cv2.imread(cell_path)
                if img is not None:
                    # ========== 新增：自动裁剪背景，只保留细胞区域 ==========
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # OTSU阈值分割细胞和背景
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    # 形态学去噪
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    # 找到细胞的边界框
                    y_pixels = np.any(mask, axis=1)
                    x_pixels = np.any(mask, axis=0)
                    try:
                        ymin, ymax = np.where(y_pixels)[0][[0, -1]]
                        xmin, xmax = np.where(x_pixels)[0][[0, -1]]
                        # 裁剪，保留1像素边距
                        cropped_cell = img[max(0, ymin - 1):min(img.shape[0], ymax + 2),
                        max(0, xmin - 1):min(img.shape[1], xmax + 2)]
                        img_rgb = cv2.cvtColor(cropped_cell, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # 如果裁剪失败，使用原图
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # ======================================================

                    templates[cell_name] = (img_rgb, cell_path)
                    print(f"  已加载模板: {cell_path} (裁剪后尺寸: {img_rgb.shape[1]}x{img_rgb.shape[0]})")
                break

    if not templates:
        raise FileNotFoundError(f"在 {patient_dir} 下未找到任何 cell 模板 (cell1/cell2/cell3)")

    if 'cell2' not in templates:
        print(f"  警告: 未找到主模板 cell2，将使用 {list(templates.keys())[0]} 作为主模板")

    return templates


def get_image_files(patient_dir):
    """获取待处理的图像文件（排除cell模板）"""
    valid_ext = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}
    cell_names = {'cell', 'cell1', 'cell2', 'cell3'}
    files = []
    for f in os.listdir(patient_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in valid_ext and name.lower() not in cell_names:
            files.append(f)
    try:
        files = sorted(files, key=lambda x: [int(n) for n in re.findall(r'\d+', x)])
    except Exception:
        files = sorted(files)
    return files


def compute_color_histogram(image_rgb, mask=None):
    """计算颜色直方图"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def compute_solidity(mask_uint8):
    """solidity = 实际面积 / 凸包面积"""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def compute_aspect_ratio(mask_uint8):
    """计算长宽比"""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 999
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        x, y, w, h = cv2.boundingRect(cnt)
        return max(w, h) / max(min(w, h), 1)
    _, (ma, MA), _ = cv2.fitEllipse(cnt)
    if min(ma, MA) == 0:
        return 999
    return max(ma, MA) / min(ma, MA)


def get_cell_template_features(cell_template_rgb):
    """提取细胞模板特征"""
    h, w = cell_template_rgb.shape[:2]
    gray = cv2.cvtColor(cell_template_rgb, cv2.COLOR_RGB2GRAY)

    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(otsu_mask == 255) > np.sum(otsu_mask == 0):
        otsu_mask = 255 - otsu_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    center_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.35)
    cv2.circle(center_mask, (cx, cy), radius, 255, -1)

    pixel_area = np.sum(otsu_mask > 0)
    hist = compute_color_histogram(cell_template_rgb, center_mask)
    solidity = compute_solidity(otsu_mask)

    print(f"  模板尺寸: {w}x{h}")
    print(f"  模板细胞面积: {pixel_area} 像素")
    print(f"  模板 solidity: {solidity:.3f}")

    return {
        'pixel_area': pixel_area,
        'hist': hist,
        'solidity': solidity,
        'template_h': h,
        'template_w': w,
    }


def multi_scale_template_match(image_rgb, template_rgb,
                               scale_range=(0.5, 2.0), scale_steps=30,
                               use_edge=False):
    """多尺度模板匹配"""
    if use_edge:
        gray_image = cv2.Canny(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 30, 100)
        gray_template = cv2.Canny(cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY), 30, 100)
    else:
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_template = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)

    th, tw = gray_template.shape[:2]
    best_score = -1
    best_loc = None
    best_scale = 1.0
    best_tw, best_th = tw, th

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    for scale in scales:
        new_tw = int(tw * scale)
        new_th = int(th * scale)
        if new_tw < 10 or new_th < 10:
            continue
        if new_tw >= gray_image.shape[1] or new_th >= gray_image.shape[0]:
            continue

        resized_template = cv2.resize(gray_template, (new_tw, new_th))
        result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale
            best_tw, best_th = new_tw, new_th

    if best_loc is None:
        return None

    center_x = best_loc[0] + best_tw // 2
    center_y = best_loc[1] + best_th // 2

    return (center_x, center_y, best_scale, best_score, best_tw, best_th)


def locate_cell(image_rgb, template_rgb, scale_range, scale_steps, match_threshold):
    """两阶段定位：灰度 → 边缘"""
    result = multi_scale_template_match(
        image_rgb, template_rgb,
        scale_range=scale_range, scale_steps=scale_steps,
        use_edge=False
    )
    if result is not None and result[3] >= match_threshold:
        return result, "gray"

    result_edge = multi_scale_template_match(
        image_rgb, template_rgb,
        scale_range=scale_range, scale_steps=scale_steps,
        use_edge=True
    )
    if result_edge is not None and result_edge[3] >= match_threshold * 0.7:
        return result_edge, "edge"

    best = result
    method = "gray"
    if result_edge is not None:
        if result is None or result_edge[3] > result[3]:
            best = result_edge
            method = "edge"

    if best is not None and best[3] >= match_threshold * 0.5:
        return best, method + "(weak)"

    return None, "failed"


def generate_grid_points(cx, cy, box_w, box_h, n_ring=8):
    """生成多点 prompt：中心点 + 周围一圈点"""
    points = [[cx, cy]]  # 中心点

    radius_x = int(box_w * 0.3)
    radius_y = int(box_h * 0.3)

    for i in range(n_ring):
        angle = 2 * np.pi * i / n_ring
        px = int(cx + radius_x * np.cos(angle))
        py = int(cy + radius_y * np.sin(angle))
        points.append([px, py])

    return np.array(points)


def sam_multi_point_segment(predictor, image_rgb, cx, cy, box_w, box_h, expand_ratio=1.5):
    """SAM 分割策略：box prompt + 多点 prompt"""
    predictor.set_image(image_rgb)
    img_h, img_w = image_rgb.shape[:2]

    # Box prompt
    half_w = int(box_w * expand_ratio / 2)
    half_h = int(box_h * expand_ratio / 2)
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(img_w, cx + half_w)
    y2 = min(img_h, cy + half_h)
    input_box = np.array([x1, y1, x2, y2])

    # 多点 prompt
    grid_points = generate_grid_points(cx, cy, box_w, box_h, n_ring=8)
    # 裁剪到图像范围内
    grid_points[:, 0] = np.clip(grid_points[:, 0], 0, img_w - 1)
    grid_points[:, 1] = np.clip(grid_points[:, 1], 0, img_h - 1)
    input_labels = np.ones(len(grid_points), dtype=np.int32)  # 全部标记为前景

    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=grid_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True,
        )

    return masks, scores, (x1, y1, x2, y2)


def convex_hull_fill(mask_uint8):
    """凸包填充：填补 SAM 因折叠/模糊而漏切的区域"""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask_uint8

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, [hull], -1, 255, -1)

    return filled


def morphological_cleanup(mask, kernel_size=5):
    """闭运算填空洞 + 开运算去噪"""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def postprocess_mask(mask_raw, use_convex_fill=True):
    """完整的后处理流程"""
    # Step 1: 形态学
    cleaned = morphological_cleanup(mask_raw, kernel_size=5)

    # Step 2: 保留最大连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        result = cleaned
    else:
        # 跳过背景(label 0)，找最大的连通域
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        result = np.zeros_like(cleaned)
        result[labels == largest_label] = 255

    # Step 3: 凸包填充
    if use_convex_fill:
        # 先计算 solidity，如果已经很高就不需要填充
        sol = compute_solidity(result)
        if sol < 0.90:
            # solidity < 0.90 说明有明显凹陷（可能是折叠被切掉了）
            result = convex_hull_fill(result)
            print(f"    [POST] 凸包填充: solidity {sol:.3f} → 1.000")

    return (result > 0).astype(np.uint8)


def compute_mask_color_hist(image_rgb, mask):
    """计算掩膜区域的颜色直方图"""
    mask_uint8 = mask.astype(np.uint8)
    y_pixels = np.any(mask_uint8, axis=1)
    x_pixels = np.any(mask_uint8, axis=0)
    try:
        ymin, ymax = np.where(y_pixels)[0][[0, -1]]
        xmin, xmax = np.where(x_pixels)[0][[0, -1]]
    except IndexError:
        return compute_color_histogram(image_rgb, mask_uint8)

    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    radius = int(min(xmax - xmin, ymax - ymin) * 0.25)

    center_mask = np.zeros_like(mask_uint8)
    cv2.circle(center_mask, (cx, cy), max(radius, 3), 1, -1)
    center_mask = center_mask & mask_uint8

    if np.sum(center_mask) < 10:
        return compute_color_histogram(image_rgb, mask_uint8)

    return compute_color_histogram(image_rgb, center_mask)


def select_best_mask(masks, scores, cell_features, image_rgb,
                     match_scale, prompt_box, debug_name="",
                     area_tolerance=3.0,
                     solidity_min=0.50,
                     aspect_ratio_max=2.5,
                     use_convex_fill=True):
    """筛选最佳掩膜"""
    expected_area = cell_features['pixel_area'] * (match_scale ** 2)
    ref_hist = cell_features['hist']

    box_x1, box_y1, box_x2, box_y2 = prompt_box
    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)

    print(f"    [DEBUG] 模板面积={cell_features['pixel_area']}, "
          f"尺度={match_scale:.2f}, 预期面积={expected_area:.0f}")

    candidates = []

    for i, (mask_raw, score) in enumerate(zip(masks, scores)):
        # ★ 后处理（含凸包填充）
        mask = postprocess_mask(mask_raw, use_convex_fill=use_convex_fill)
        mask_uint8 = mask * 255
        area = int(np.sum(mask))

        area_ratio = area / expected_area if expected_area > 0 else 1.0
        solid = compute_solidity(mask_uint8)
        ar = compute_aspect_ratio(mask_uint8)
        hist = compute_mask_color_hist(image_rgb, mask)
        color_sim = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
        box_overlap = area / box_area if box_area > 0 else 0

        print(f"    [DEBUG] 掩膜{i}: SAM={score:.3f}, "
              f"面积={area}, 比={area_ratio:.2f}, "
              f"sol={solid:.3f}, AR={ar:.2f}, "
              f"色={color_sim:.3f}, box占={box_overlap:.2f}")

        rejected = False

        if area_ratio < (1.0 / area_tolerance) or area_ratio > area_tolerance:
            print(f"    [DEBUG]   -> 拒绝: 面积比超范围")
            rejected = True

        if solid < solidity_min:
            print(f"    [DEBUG]   -> 拒绝: solidity {solid:.3f} < {solidity_min}")
            rejected = True

        if ar > aspect_ratio_max:
            print(f"    [DEBUG]   -> 拒绝: AR {ar:.2f} > {aspect_ratio_max}")
            rejected = True

        if box_overlap > 3.0:
            print(f"    [DEBUG]   -> 拒绝: box占比过大")
            rejected = True

        if rejected:
            continue

        print(f"    [DEBUG]   -> 通过!")

        solid_score = solid
        area_score = 1.0 - min(abs(area_ratio - 1.0), 1.0)
        compact_score = min(box_overlap, 1.0)
        color_score = max(color_sim, 0)

        combined = (solid_score * 0.25
                    + area_score * 0.25
                    + score * 0.20
                    + color_score * 0.15
                    + compact_score * 0.15)

        candidates.append({
            'index': i,
            'mask': mask,
            'area': area,
            'area_ratio': area_ratio,
            'solidity': solid,
            'aspect_ratio': ar,
            'color_sim': color_sim,
            'sam_score': score,
            'combined_score': combined,
        })

    if not candidates:
        return None, None

    best = max(candidates, key=lambda c: c['combined_score'])
    return best['mask'], best


def save_cell_result(image_rgb, mask, out_path, image_name, H_L_image, error_file, score_info=None):
    """保存分割结果"""
    mask_uint8 = mask.astype(np.uint8)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_uint8)

    y_pixels = np.any(mask_uint8, axis=1)
    x_pixels = np.any(mask_uint8, axis=0)
    try:
        ymin, ymax = np.where(y_pixels)[0][[0, -1]]
        xmin, xmax = np.where(x_pixels)[0][[0, -1]]
    except IndexError:
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"[{image_name}] 掩膜边界获取失败\n")
        return

    cropped = masked_image[ymin:ymax + 1, xmin:xmax + 1]
    h, w = cropped.shape[:2]
    mask_area = int(np.sum(mask_uint8))

    if h == 0 or w == 0:
        return

    canvas = np.zeros((H_L_image, H_L_image, 3), dtype=np.uint8)
    if h > w:
        new_w = max(1, int(H_L_image * w / h))
        resized = cv2.resize(cropped, (new_w, H_L_image))
        offset = (H_L_image - new_w) // 2
        canvas[:, offset:offset + new_w] = resized
    elif w > h:
        new_h = max(1, int(H_L_image * h / w))
        resized = cv2.resize(cropped, (H_L_image, new_h))
        offset = (H_L_image - new_h) // 2
        canvas[offset:offset + new_h, :] = resized
    else:
        canvas = cv2.resize(cropped, (H_L_image, H_L_image))

    base_name = os.path.splitext(image_name)[0]
    filename = os.path.join(out_path, f"{base_name}_{xmin}_{ymin}_{xmax}_{ymax}_{mask_area}.tif")

    try:
        cv2.imwrite(filename, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    except Exception as e:
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"保存 {filename} 出错: {str(e)}\n")
        return

    info_str = ""
    if score_info:
        info_str = (f" | 比={score_info['area_ratio']:.2f}, "
                    f"sol={score_info['solidity']:.3f}, "
                    f"AR={score_info['aspect_ratio']:.2f}, "
                    f"色={score_info['color_sim']:.3f}, "
                    f"分={score_info['combined_score']:.3f}")
    print(f"  已保存: {os.path.basename(filename)}{info_str}")


def try_segment_with_template(predictor, image_rgb, template_rgb, cell_features,
                              scale_range, scale_steps, match_threshold,
                              use_convex_fill, img_name, is_retry=False):
    """用指定模板对一张图片尝试定位+分割"""
    # Step 1: 定位
    match_result, method = locate_cell(
        image_rgb, template_rgb,
        scale_range=scale_range, scale_steps=scale_steps,
        match_threshold=match_threshold
    )

    if match_result is None:
        return None, None, f"定位失败({method})"

    cx, cy, best_scale, match_score, mw, mh = match_result
    info = (f"({cx},{cy}), scale={best_scale:.2f}, "
            f"score={match_score:.3f}, method={method}")

    # Step 2: SAM 分割
    try:
        masks, scores, prompt_box = sam_multi_point_segment(
            predictor, image_rgb, cx, cy, mw, mh,
            expand_ratio=1.5
        )
    except Exception as e:
        return None, None, f"SAM出错: {str(e)}"

    # Step 3: 选最佳掩膜
    tol = 3.0 if not is_retry else 4.0
    sol = 0.50 if not is_retry else 0.40
    ar = 2.5 if not is_retry else 3.0

    best_mask, score_info = select_best_mask(
        masks, scores, cell_features, image_rgb,
        match_scale=best_scale,
        prompt_box=prompt_box,
        debug_name=img_name,
        area_tolerance=tol,
        solidity_min=sol,
        aspect_ratio_max=ar,
        use_convex_fill=use_convex_fill,
    )

    # 第一轮失败，扩大 box 重试
    if best_mask is None and not is_retry:
        try:
            masks2, scores2, prompt_box2 = sam_multi_point_segment(
                predictor, image_rgb, cx, cy, mw, mh,
                expand_ratio=2.0
            )
            best_mask, score_info = select_best_mask(
                masks2, scores2, cell_features, image_rgb,
                match_scale=best_scale,
                prompt_box=prompt_box2,
                debug_name=img_name,
                area_tolerance=4.0,
                solidity_min=0.40,
                aspect_ratio_max=3.0,
                use_convex_fill=use_convex_fill,
            )
        except Exception:
            pass

    if best_mask is None:
        return None, None, f"定位成功({info})但无符合掩膜"

    return best_mask, score_info, info


def process_patient_dir(patient_dir, out_path, H_L_image, error_file,
                        sam_checkpoint, model_type="vit_b",
                        match_threshold=0.3,
                        scale_range=(0.5, 2.0), scale_steps=30,
                        use_convex_fill=True):
    """处理单个patient目录的细胞分割"""
    # --- 加载三个模板 ---
    try:
        templates = load_cell_templates(patient_dir)
    except FileNotFoundError as e:
        print(f"跳过 {patient_dir}: {e}")
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"跳过 {patient_dir}: {e}\n")
        return

    # 预计算每个模板的特征
    template_data = {}  # name -> (rgb, features)
    for name, (rgb, path) in templates.items():
        print(f"\n  提取 {name} 特征:")
        features = get_cell_template_features(rgb)
        template_data[name] = (rgb, features)

    # 确定主模板和备选模板
    if 'cell2' in template_data:
        primary_name = 'cell2'
    else:
        primary_name = list(template_data.keys())[0]

    backup_names = [n for n in template_data.keys() if n != primary_name]

    print(f"\n  主模板: {primary_name}, 备选: {backup_names}")

    # --- 加载 SAM ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
    except Exception as e:
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"加载 SAM 模型出错: {str(e)}\n")
        return

    predictor = SamPredictor(sam)
    image_files = get_image_files(patient_dir)
    print(f"共找到 {len(image_files)} 张待处理图片")
    os.makedirs(out_path, exist_ok=True)

    found_count = 0
    not_found_count = 0
    found_by = {primary_name: 0}
    for bn in backup_names:
        found_by[bn] = 0

    for img_name in tqdm(image_files, desc=f"处理 {os.path.basename(patient_dir)}"):
        img_path = os.path.join(patient_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(f"无法读取: {img_path}\n")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(img_name)[0]
        existing = [f for f in os.listdir(out_path) if f.startswith(base_name + "_")]
        if existing:
            continue

        # ====== 第一轮: 用主模板 (cell2) ======
        primary_rgb, primary_features = template_data[primary_name]
        best_mask, score_info, info_str = try_segment_with_template(
            predictor, image_rgb, primary_rgb, primary_features,
            scale_range, scale_steps, match_threshold,
            use_convex_fill, img_name, is_retry=False
        )

        if best_mask is not None:
            print(f"  [{img_name}] {primary_name} 成功: {info_str}")
            save_cell_result(image_rgb, best_mask, out_path, img_name,
                             H_L_image, error_file, score_info)
            found_count += 1
            found_by[primary_name] += 1
            del image, image_rgb
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ====== 第二轮: 用备选模板 (cell1, cell3) ======
        print(f"  [{img_name}] {primary_name} 失败({info_str})，尝试备选模板...")
        found_backup = False

        for backup_name in backup_names:
            backup_rgb, backup_features = template_data[backup_name]
            best_mask, score_info, info_str = try_segment_with_template(
                predictor, image_rgb, backup_rgb, backup_features,
                scale_range, scale_steps, match_threshold * 0.8,  # 备选模板门槛稍降
                use_convex_fill, img_name, is_retry=True
            )

            if best_mask is not None:
                print(f"  [{img_name}] {backup_name} 成功: {info_str}")
                save_cell_result(image_rgb, best_mask, out_path, img_name,
                                 H_L_image, error_file, score_info)
                found_count += 1
                found_by[backup_name] += 1
                found_backup = True
                break

        if not found_backup:
            print(f"  [{img_name}] 所有模板均未找到")
            not_found_count += 1

        del image, image_rgb
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n处理完成: 找到 {found_count} 张，未找到 {not_found_count} 张")
    print(f"  各模板贡献: {found_by}")
    del sam, predictor
    torch.cuda.empty_cache()
    gc.collect()


# ===================== 第三部分：主流程（整合复制+分割） =====================
def main():
    start_time = time.time()
    error_file = "error.txt"
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("错误日志\n")

    # ====== 1. 第一步：复制筛选指定ID的文件夹 ======
    print("=" * 60)
    print("第一步：筛选并复制指定ID的文件夹")
    print("=" * 60)
    filtered_folder_root = copy_selected_folders()

    # ====== 2. 第二步：配置SAM模型 ======
    print("\n" + "=" * 60)
    print("第二步：配置SAM模型并批量分割细胞")
    print("=" * 60)
    # SAM模型路径配置（根据实际情况修改）
    sam_checkpoint = r"E:\blood\sam_vit_l_0b3195.pth"  # vit_l
    # sam_checkpoint = r"C:\Users\Tony_zzk\Desktop\code\code--RBC\SAM\sam_vit_b_01ec64.pth"  # vit_b
    # sam_checkpoint = r"I:\sam_vit_h_4b8939.pth"  # vit_h
    model_type = "vit_l"  # 对应sam_checkpoint的模型类型

    H_L_image = 224  # 输出图像尺寸
    use_convex_fill = True  # 启用凸包填充（对折叠细胞有效）
    match_threshold = 0.3  # 模板匹配阈值
    scale_range = (0.5, 2.0)  # 模板匹配尺度范围
    scale_steps = 30  # 尺度步数

    # 分割结果输出根目录
    seg_output_root = os.path.join(os.path.dirname(filtered_folder_root), "matlabphotos2-seg")
    os.makedirs(seg_output_root, exist_ok=True)

    # ====== 3. 批量处理筛选后的文件夹 ======
    try:
        folder_names = os.listdir(filtered_folder_root)
    except Exception as e:
        print(f"获取筛选后的文件夹列表出错: {str(e)}")
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"获取文件夹列表出错: {str(e)}\n")
        return

    for folder_name in folder_names:
        patient_dir = os.path.join(filtered_folder_root, folder_name)
        if not os.path.isdir(patient_dir):
            continue

        # 检查是否有cell模板
        has_cell = any(
            os.path.exists(os.path.join(patient_dir, f"{cn}{ext}"))
            for cn in ['cell1', 'cell2', 'cell3']
            for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp']
        )
        if not has_cell:
            print(f"\n跳过 {folder_name}：无 cell1/cell2/cell3 模板")
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(f"跳过 {folder_name}：无 cell1/cell2/cell3 模板\n")
            continue

        # 单个文件夹的分割输出路径
        out_path = os.path.join(seg_output_root, folder_name)

        print(f"\n{'=' * 50}")
        print(f"处理文件夹: {patient_dir}")
        print(f"输出路径: {out_path}")
        print(f"{'=' * 50}")

        # 处理当前文件夹的细胞分割
        process_patient_dir(
            patient_dir=patient_dir,
            out_path=out_path,
            H_L_image=H_L_image,
            error_file=error_file,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            match_threshold=match_threshold,
            scale_range=scale_range,
            scale_steps=scale_steps,
            use_convex_fill=use_convex_fill,
        )

    # ====== 4. 完成统计 ======
    end_time = time.time()
    print(f"\n{'=' * 60}")
    print("全部流程完成！")
    print(f"总运行时间: {end_time - start_time:.2f} 秒")
    print(f"文件夹筛选结果路径: {filtered_folder_root}")
    print(f"细胞分割结果路径: {seg_output_root}")
    print(f"错误日志路径: {os.path.abspath(error_file)}")
    print("=" * 60)


if __name__ == "__main__":
    main()