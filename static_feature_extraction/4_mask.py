import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys


def extract_masks(image_path, save_paths=None):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}，请检查路径是否正确。")
        return None, None, None

    # 图像预处理：高斯滤波去噪，调整核大小
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 将图像从BGR转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 调整红色在HSV中的范围（更宽泛，适应更多红色情况）
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # 创建两个范围的掩膜并合并
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    base_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作：先膨胀再腐蚀（膨胀增强目标，腐蚀去除噪声），调整核大小
    kernel = np.ones((5, 5), np.uint8)
    base_mask = cv2.dilate(base_mask, kernel, iterations=1)
    base_mask = cv2.erode(base_mask, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"未找到轮廓，图像 {image_path} 可能不包含目标对象。")
        return None, None, None

    # 筛选轮廓：选择面积最大的轮廓（放宽圆形度条件）
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_contour = contour

    if best_contour is None:
        print(f"未找到合适的轮廓，图像 {image_path} 可能不包含目标对象。")
        return None, None, None

    # 获取图像中心
    moments = cv2.moments(best_contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # 小形态掩膜：多边形近似圆形并缩放
    epsilon_small = 0.01 * cv2.arcLength(best_contour, True)
    approx_small = cv2.approxPolyDP(best_contour, epsilon_small, True)
    # 调整缩放比例
    small_scale = 0.7
    small_approx_scaled = []
    for point in approx_small:
        x = int(cx + (point[0][0] - cx) * small_scale)
        y = int(cy + (point[0][1] - cy) * small_scale)
        small_approx_scaled.append([[x, y]])
    small_approx_scaled = np.array(small_approx_scaled, dtype=np.int32)
    small_mask = np.zeros_like(base_mask)
    cv2.drawContours(small_mask, [small_approx_scaled], -1, 255, thickness=cv2.FILLED)

    # 中形态掩膜：多边形近似圆形并缩放
    epsilon_medium = 0.02 * cv2.arcLength(best_contour, True)
    approx_medium = cv2.approxPolyDP(best_contour, epsilon_medium, True)
    # 调整缩放比例
    medium_scale = 0.9
    medium_approx_scaled = []
    for point in approx_medium:
        x = int(cx + (point[0][0] - cx) * medium_scale)
        y = int(cy + (point[0][1] - cy) * medium_scale)
        medium_approx_scaled.append([[x, y]])
    medium_approx_scaled = np.array(medium_approx_scaled, dtype=np.int32)
    medium_mask = np.zeros_like(base_mask)
    cv2.drawContours(medium_mask, [medium_approx_scaled], -1, 255, thickness=cv2.FILLED)

    # 大形态掩膜：直接使用最佳轮廓
    large_mask = np.zeros_like(base_mask)
    cv2.drawContours(large_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    # 调整掩码大小
    new_size = (224, 224)
    small_mask = cv2.resize(small_mask, new_size)
    medium_mask = cv2.resize(medium_mask, new_size)
    large_mask = cv2.resize(large_mask, new_size)

    # 将单通道掩码转换为三通道RGB格式
    small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2RGB)
    medium_mask = cv2.cvtColor(medium_mask, cv2.COLOR_GRAY2RGB)
    large_mask = cv2.cvtColor(large_mask, cv2.COLOR_GRAY2RGB)

    # 保存掩膜
    if save_paths:
        if len(save_paths) == 3:
            cv2.imwrite(save_paths[0], small_mask)
            cv2.imwrite(save_paths[1], medium_mask)
            cv2.imwrite(save_paths[2], large_mask)
            print(f"小尺度掩膜已保存到：{save_paths[0]}")
            print(f"中尺度掩膜已保存到：{save_paths[1]}")
            print(f"大尺度掩膜已保存到：{save_paths[2]}")
        else:
            print("保存路径列表长度不正确，无法保存掩膜。")

    # 显示结果（调试用，可取消注释）
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    # plt.axis('off')
    # plt.subplot(2, 2, 2)
    # plt.imshow(small_mask)
    # plt.title('Small Mask')
    # plt.axis('off')
    # plt.subplot(2, 2, 3)
    # plt.imshow(medium_mask)
    # plt.title('Medium Mask')
    # plt.axis('off')
    # plt.subplot(2, 2, 4)
    # plt.imshow(large_mask)
    # plt.title('Large Mask')
    # plt.axis('off')
    # plt.show()

    return small_mask, medium_mask, large_mask


if __name__ == "__main__":
    # 初始化地址
    input_root = ""
    output_root = ""
    choice = 2  # choice 1 for single processing, choice 2 for continual processing

    if len(sys.argv) >= 3:
        input_root = sys.argv[1]
        output_root = sys.argv[2]

    if choice == 1:
        os.makedirs(output_root, exist_ok=True)
        # 遍历根目录
        for folder_name in os.listdir(input_root):
            input_folder = os.path.join(input_root, folder_name)
            output_folder = os.path.join(output_root, folder_name)
            print(f"正在处理文件夹: {input_folder}")

            # 获取该文件夹下所有的 .tif 文件
            image_paths = glob.glob(os.path.join(input_folder, '*.tif'))
            if image_paths:
                # 取文件夹中的第一张图片来生成掩膜
                image_path = image_paths[0]
                print(f"正在处理图片: {image_path}")

                # 定义掩膜的保存路径
                save_paths = [
                    os.path.join(output_folder, 'mask.tif'),
                    os.path.join(output_folder, 'mask2.tif'),
                    os.path.join(output_folder, 'mask3.tif')
                ]

                # 确保输出文件夹存在
                os.makedirs(output_folder, exist_ok=True)

                # 使用函数提取掩膜
                mask1, mask2, mask3 = extract_masks(image_path, save_paths)
            else:
                print(f"文件夹 {input_folder} 中没有图片。")
    elif choice == 2:
        for folder_name in os.listdir(input_root):
            input_dir = os.path.join(input_root, folder_name)
            output_dir = os.path.join(output_root, folder_name)
            # 遍历子文件夹
            for sub_folder_name in os.listdir(input_dir):
                input_folder = os.path.join(input_dir, sub_folder_name)
                output_folder = os.path.join(output_dir, sub_folder_name)
                print(f"正在处理文件夹: {input_folder}")

                # 获取该文件夹下所有的 .tif 文件
                image_paths = glob.glob(os.path.join(input_folder, '*.tif'))
                if image_paths:
                    # 取文件夹中的第一张图片来生成掩膜
                    image_path = image_paths[0]
                    print(f"正在处理图片: {image_path}")

                    # 定义掩膜的保存路径
                    save_paths = [
                        os.path.join(output_folder, 'mask.tif'),
                        os.path.join(output_folder, 'mask2.tif'),
                        os.path.join(output_folder, 'mask3.tif')
                    ]

                    # 确保输出文件夹存在
                    os.makedirs(output_folder, exist_ok=True)

                    # 使用函数提取掩膜
                    mask1, mask2, mask3 = extract_masks(image_path, save_paths)
                else:
                    print(f"文件夹 {input_folder} 中没有图片。")