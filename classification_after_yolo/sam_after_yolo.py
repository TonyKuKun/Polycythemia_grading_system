import os
import time
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
import re
import gc

start_time = time.time()

# ========================= 工具函数 =========================

def load_cell_template(patient_dir):
    cell_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp']
    for ext in cell_extensions:
        cell_path = os.path.join(patient_dir, f"cell{ext}")
        if os.path.exists(cell_path):
            img = cv2.imread(cell_path)
            if img is not None:
                print(f"已加载目标细胞模板: {cell_path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cell_path
    raise FileNotFoundError(f"在 {patient_dir} 下未找到 cell 模板图片")


def get_image_files(patient_dir):
    valid_ext = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}
    files = []
    for f in os.listdir(patient_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in valid_ext and not name.lower().startswith('cell'):
            files.append(f)
    try:
        files = sorted(files, key=lambda x: [int(n) for n in re.findall(r'\d+', x)])
    except Exception:
        files = sorted(files)
    return files


def compute_color_histogram(image_rgb, mask=None):
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
    """长宽比，排除多细胞连接"""
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


# ========================= 定位函数 =========================

def multi_scale_template_match(image_rgb, template_rgb,
                               scale_range=(0.5, 2.0), scale_steps=30,
                               use_edge=False):
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


# ========================= SAM 分割（核心改进）=========================

def generate_grid_points(cx, cy, box_w, box_h, n_ring=8):
    """
    生成多点 prompt：中心点 + 周围一圈点。

    为什么多点比单点好：
    - 单点 prompt：SAM 只知道"这个点是前景"，遇到折叠处的色差/纹理变化
      会认为那是边界，只分割一半。
    - 多点 prompt：在细胞范围内撒多个前景点，告诉 SAM
      "这些点全部是同一个物体"，即使折叠处颜色不同，SAM 也被迫包含进来。

    布局：中心 1 点 + 环形 8 点 = 9 个前景点
    环形半径 = 模板匹配框的 30%，确保点在细胞内部
    """
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
    """
    SAM 分割策略：box prompt + 多点 prompt

    组合使用：
    - box prompt: 限定搜索范围，防止选中整个通道
    - 9 个前景点: 覆盖整个细胞区域，包括折叠部分
    """
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
    """
    凸包填充：填补 SAM 因折叠/模糊而漏切的区域。

    原理：细胞（即使折叠、棘状）的真实边界近似于其凸包。
    SAM 因为看到折叠处的色差而"挖掉"了一块，
    用凸包将凹陷区域填回来。

    只对最大轮廓操作，避免填充噪点。
    """
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
    """
    完整的后处理流程：
    1. 形态学清理（填小洞、去噪点）
    2. 只保留最大连通域（去掉远处的碎片）
    3. 可选：凸包填充（修复折叠导致的缺失）
    """
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
    """
    v5 筛选:
      1. 每个候选掩膜先做后处理（形态学 + 最大连通域 + 凸包填充）
      2. 再计算指标做筛选
      3. solidity_min 降到 0.50（因为凸包填充会修复大部分情况）
    """
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


# ========================= 保存 =========================

def save_cell_result(image_rgb, mask, out_path, image_name, H_L_image, error_file, score_info=None):
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


# ========================= 主流程 =========================

def process_patient_dir(patient_dir, out_path, H_L_image, error_file,
                        sam_checkpoint, model_type="vit_b",
                        match_threshold=0.3,
                        scale_range=(0.5, 2.0), scale_steps=30,
                        use_convex_fill=True):

    cell_template_rgb, cell_path = load_cell_template(patient_dir)
    cell_features = get_cell_template_features(cell_template_rgb)

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

    for img_name in tqdm(image_files, desc="处理图片"):
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

        # ====== Step 1: 两阶段定位 ======
        match_result, method = locate_cell(
            image_rgb, cell_template_rgb,
            scale_range=scale_range, scale_steps=scale_steps,
            match_threshold=match_threshold
        )

        if match_result is None:
            print(f"  [{img_name}] 定位失败 ({method})，跳过")
            not_found_count += 1
            del image, image_rgb
            gc.collect()
            continue

        cx, cy, best_scale, match_score, mw, mh = match_result
        print(f"  [{img_name}] 定位: ({cx},{cy}), "
              f"scale={best_scale:.2f}, score={match_score:.3f}, method={method}")

        # ====== Step 2: SAM 多点+box 分割 ======
        try:
            masks, scores, prompt_box = sam_multi_point_segment(
                predictor, image_rgb, cx, cy, mw, mh,
                expand_ratio=1.5
            )
        except Exception as e:
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(f"[{img_name}] SAM出错: {str(e)}\n")
            del image, image_rgb
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ====== Step 3: 选最佳掩膜 ======
        best_mask, score_info = select_best_mask(
            masks, scores, cell_features, image_rgb,
            match_scale=best_scale,
            prompt_box=prompt_box,
            debug_name=img_name,
            area_tolerance=3.0,
            solidity_min=0.50,
            aspect_ratio_max=2.5,
            use_convex_fill=use_convex_fill,
        )

        # ====== Step 3b: 失败重试（更大 box + 更宽松）======
        if best_mask is None:
            print(f"  [{img_name}] 第一轮未找到，重试...")
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
            print(f"  [{img_name}] 未找到符合条件的掩膜")
            not_found_count += 1
        else:
            save_cell_result(image_rgb, best_mask, out_path, img_name,
                             H_L_image, error_file, score_info)
            found_count += 1

        del image, image_rgb, masks, scores
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n处理完成: 找到 {found_count} 张，未找到 {not_found_count} 张")
    del sam, predictor
    torch.cuda.empty_cache()
    gc.collect()


# ========================= 入口 =========================

'''
v5 改进汇总:

  问题1: 折叠细胞 SAM 分割不完整（把折叠处当边界切掉了）
    修复A: 多点 prompt（中心+环形8点 = 9点全标记前景）
           → SAM 被迫把整个区域当成同一个物体
    修复B: 凸包填充后处理
           → 即使 SAM 仍然切掉一块，凸包填充把凹陷补回来
           → 只在 solidity < 0.90 时启动（正常细胞不受影响）

  问题2: 棘状细胞
    继承 v4: solidity 代替 circularity（棘状 solidity ≈ 0.80 通过）

  问题3: 运动模糊
    继承 v4: 灰度→边缘两阶段匹配 + 形态学闭运算

  关于 SAM 模型选择:
    当前使用 vit_b（最快但精度最低）
    如果分割质量仍不满意，强烈建议换用:
      - vit_l: sam_vit_l_0b3195.pth（中等速度，精度更好）
      - vit_h: sam_vit_h_4b8939.pth（最慢但精度最高，推荐用于困难样本）
    只需修改 sam_checkpoint 和 model_type 即可
'''

if __name__ == "__main__":
    error_file = "error.txt"
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("错误日志\n")

    # ====== SAM 模型配置 ======
    # 选项1（当前）: vit_b - 最快，适合大批量初筛
    # sam_checkpoint = r"C:\Users\Tony_zzk\Desktop\code\code--RBC\SAM\sam_vit_b_01ec64.pth"
    # model_type = "vit_b"

    # 选项2: vit_l - 推荐，速度和精度平衡
    # sam_checkpoint = r"E:\blood\sam_vit_l_0b3195.pth"
    # model_type = "vit_l"

    # 选项3: vit_h - 最高精度，处理折叠/模糊细胞效果最好
    sam_checkpoint = r"E:\blood\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    H_L_image = 224

    # 是否启用凸包填充（对折叠细胞有效，默认开启）
    use_convex_fill = True

    print("=" * 50)
    print("  细胞追踪与分割工具 v5")
    print("=" * 50)
    print("choice 1: 处理单个 patient 目录")
    print("choice 2: 批量处理多个 patient 目录")

    try:
        choice = int(input("请输入选择 (1 或 2): "))
    except ValueError:
        print("输入无效")
        exit()

    if choice == 1:
        patient_dir = r"G:\48281242\2-c2\matlabphotos2\45"
        out_path = r"G:\48281242\2-c2\matlabphotos3\45"

        process_patient_dir(
            patient_dir=patient_dir,
            out_path=out_path,
            H_L_image=H_L_image,
            error_file=error_file,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            match_threshold=0.3,
            scale_range=(0.5, 2.0),
            scale_steps=30,
            use_convex_fill=use_convex_fill,
        )

    elif choice == 2:
        input_dir = r"C:\Users\Administrator\Desktop\matlabphotos-error"
        output_dir = r"C:\Users\Administrator\Desktop\new huge"

        try:
            folder_names = os.listdir(input_dir)
        except Exception as e:
            print(f"获取目录列表出错: {str(e)}")
            exit()

        for folder_name in folder_names:
            patient_dir = os.path.join(input_dir, folder_name)
            if not os.path.isdir(patient_dir):
                continue
            has_cell = any(
                os.path.exists(os.path.join(patient_dir, f"cell{ext}"))
                for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp']
            )
            if not has_cell:
                print(f"跳过 {folder_name}：无 cell 模板")
                continue

            out_path = os.path.join(output_dir, folder_name)
            print(f"\n{'='*50}")
            print(f"处理: {patient_dir}")
            print(f"{'='*50}")

            process_patient_dir(
                patient_dir=patient_dir,
                out_path=out_path,
                H_L_image=H_L_image,
                error_file=error_file,
                sam_checkpoint=sam_checkpoint,
                model_type=model_type,
                match_threshold=0.3,
                scale_range=(0.5, 2.0),
                scale_steps=30,
                use_convex_fill=use_convex_fill,
            )
    else:
        print("无效选择")

    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")