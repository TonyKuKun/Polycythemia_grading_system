import os
import time
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator
import re
import gc

start_time = time.time()

# 主函数：执行分割操作，添加无梯度推理
def main(image, adjust=None, show=None, sam=None):
    try:
        with torch.no_grad():
            # 调整参数时使用的掩膜生成器
            if adjust == True:
                mask_generator_2 = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.9,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=5,
                    min_mask_region_area=100,  # 后处理需要 open-cv
                )
                masks = mask_generator_2.generate(image)
            else:
                # 不调整参数时使用的掩膜生成器
                mask_generator = SamAutomaticMaskGenerator(sam)
                masks = mask_generator.generate(image)
        return masks
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return []

# 保存裁剪后的图片，带掩码面积信息
def save_cropped_image(image, mask, idx, H_L_image, pathlis, out_path):
    mask_np = mask['segmentation'].astype(np.uint8)  # 获取并转换掩膜
    masked_image = cv2.bitwise_and(image, image, mask=mask_np)
    y_pixels = np.any(mask_np, axis=1)
    x_pixels = np.any(mask_np, axis=0)
    ymin, ymax = np.where(y_pixels)[0][[0, -1]]
    xmin, xmax = np.where(x_pixels)[0][[0, -1]]
    y, x = ymax - ymin, xmax - xmin
    canvas = np.zeros((H_L_image, H_L_image, 3), dtype=np.uint8)
    cropped_image = masked_image[ymin:ymax + 1, xmin:xmax + 1]

    # 计算掩码的面积
    mask_area = np.sum(mask_np)

    # 根据宽高比调整裁剪图像的大小
    if y != 0 and x != 0 and (mask_area > 20000) and (mask_area < 40000):
        if y > x:
            new_width = int(H_L_image * x / y)
            cropped_image_resized = cv2.resize(cropped_image, (new_width, H_L_image))
            offset = (H_L_image - new_width) // 2
            canvas[:, offset:offset + new_width] = cropped_image_resized
        elif x > y:
            new_height = int(H_L_image * y / x)
            cropped_image_resized = cv2.resize(cropped_image, (H_L_image, new_height))
            offset = (H_L_image - new_height) // 2
            canvas[offset:offset + new_height, :] = cropped_image_resized
        else:
            cropped_image_resized = cv2.resize(cropped_image, (H_L_image, H_L_image))
            canvas = cropped_image_resized

        # 创建文件夹路径
        folder_path = os.path.join(out_path, pathlis)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 文件名加上面积信息
        filename = os.path.join(folder_path, f"{idx}_{xmin}_{ymin}_{xmax}_{ymax}_{mask_area}.tif")

        # 保存裁剪后的图像
        cv2.imwrite(filename, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

# 处理图像文件夹中的所有图像
def ml_handle(osdir, H_L_image, adjust, show, pathlist, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # 按照图像文件名中的数字顺序排序
    sorted_pictures = sorted(osdir, key=lambda x: [int(num) for num in re.findall(r'\d+', x)])

    # 遍历目录中的图像文件
    for path in tqdm(sorted_pictures):
        print(f"使用路径: {path}")
        image_path = os.path.join(pathlist, path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像文件: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检查是否已经处理过该图像（通过文件夹是否存在来判断）
        folder_path = os.path.join(out_path, path.split('.')[0])
        if os.path.exists(folder_path):
            print(f"文件夹已存在，跳过该文件夹: {folder_path}")
            continue

        # 执行分割操作（使用无梯度推理）
        masks = main(image, adjust, show, sam)

        # 对每个掩膜保存裁剪后的图片（跳过最后一个掩膜）
        for idx, mask in enumerate(masks):
            if idx == len(masks) - 1:
                continue
            save_cropped_image(image, mask, idx, H_L_image, path.split('.')[0], out_path)

        # 清理变量，释放显存
        del image, masks
        torch.cuda.empty_cache()
        gc.collect()

'''
作用：使用SAM算法对图像进行分割处理
分割后的图像为带黑色背景的图像
输入图像格式为tif，输出图像格式为tif
'''

if __name__ == "__main__":
    print("choice 1 for single processing")
    print("choice 2 for continual processing")
    choice = int(input("please enter your choice(1 or 2): "))

    if choice == 1:
        pathlist = r'1_datasets\dataset1'
        out_path = r'2_seg\dataset1'
        H_L_image = 224
        os.makedirs(out_path, exist_ok=True)
        osdir = os.listdir(pathlist)
        ml_handle(osdir, H_L_image, adjust=True, show=False, pathlist=pathlist, out_path=out_path)

    elif choice == 2:
        input_dir = '1_datasets'
        output_dir = '2_seg'
        H_L_image = 224
        for folder_name in os.listdir(input_dir):
            pathlist = os.path.join(input_dir, folder_name)
            osdir = os.listdir(pathlist)
            if os.path.isdir(pathlist):
                out_path = os.path.join(output_dir, folder_name)
                os.makedirs(out_path, exist_ok=True)
                ml_handle(osdir, H_L_image, adjust=True, show=False, pathlist=pathlist, out_path=out_path)

end_time = time.time()
total_time = end_time - start_time
print("总运行时间: {:.2f} 秒".format(total_time))
