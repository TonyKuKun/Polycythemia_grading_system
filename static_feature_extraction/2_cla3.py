import os
import shutil
import sys
from collections import defaultdict
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_root', required=True, help='源根目录（大文件夹）')
parser.add_argument('--output_root', required=True, help='输出目录')
args = parser.parse_args()

# 配置路径
input_root = args.input_root
output_root = args.output_root
os.makedirs(output_root, exist_ok=True)

# 获取源根目录下的所有直接子文件夹
try:
    main_subfolders = [f for f in os.listdir(input_root)
                       if os.path.isdir(os.path.join(input_root, f))]
except FileNotFoundError:
    print(f"错误：找不到目录 {input_root}，请检查路径是否正确")
    sys.exit(1)
except PermissionError:
    print(f"错误：没有访问目录 {input_root} 的权限")
    sys.exit(1)

if not main_subfolders:
    print("在源目录下没有找到子文件夹")
    sys.exit(0)

# 辅助函数：从文件名解析y1值（第3个字段）
def get_y1_from_filename(parts):
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return -1

# 处理每个主文件夹
for main_folder in main_subfolders:
    print(f"\n开始处理主文件夹: {main_folder}")
    main_output = os.path.join(output_root, main_folder)
    os.makedirs(main_output, exist_ok=True)

    # 存储预分组文件
    pre_grouped = defaultdict(lambda: defaultdict(list))
    main_folder_path = os.path.join(input_root, main_folder)

    # 1. 递归收集文件并解析y1
    for current_dir, dirs, files in os.walk(main_folder_path):
        folder_name = os.path.basename(current_dir)
        if not folder_name.isdigit():
            continue  # 仅处理纯数字子文件夹
        folder_num = folder_name  # 文件夹数字（排序依据）

        if current_dir == main_folder_path:
            continue

        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(current_dir, file)
                try:
                    base_name = os.path.splitext(file)[0]
                    parts = base_name.split("_")
                    if len(parts) < 3:
                        continue

                    # 解析y1（不再按数字位数分组，统一处理）
                    y1 = get_y1_from_filename(parts)
                    if y1 == -1:
                        continue

                    pre_grouped[folder_num]["all_digits"].append((file_path, parts, y1))

                except (IndexError, ValueError):
                    continue

    # 2. 筛选保留每个子文件夹中y1最大的文件
    filtered_files = defaultdict(list)
    for folder_num, group_list in pre_grouped.items():
        for file_list in group_list.values():
            if not file_list:
                continue
            file_list_sorted = sorted(file_list, key=lambda x: x[2], reverse=True)
            best_file = file_list_sorted[0]
            filtered_files["all_digits"].append((folder_num, best_file[0], best_file[1]))

    # 3. 按文件夹数字排序并复制（文件名前缀为文件夹数字）
    if not filtered_files:
        print(f"  没有符合条件的文件")
        continue

    group_key = "all_digits"
    dest_folder = os.path.join(main_output, group_key)
    os.makedirs(dest_folder, exist_ok=True)

    # 收集文件信息，按文件夹数字排序
    group_files = []
    for folder_num, file_path, parts in filtered_files[group_key]:
        # 新文件名格式：[文件夹数字]_[原始第一个字段]_[文件夹数字]_[剩余字段]
        new_parts = [folder_num, parts[0], folder_num] + parts[1:]
        new_file_name = "_".join(new_parts) + ".tif"
        group_files.append((int(folder_num), new_file_name, file_path))

    # 按文件夹数字升序排序
    group_files.sort(key=lambda x: x[0])

    # 打印排序结果
    print(f"  按文件夹数字排序后的顺序：{[x[0] for x in group_files]}")

    # 复制文件
    for _, new_file_name, file_path in group_files:
        dest_path = os.path.join(dest_folder, new_file_name)
        shutil.copy2(file_path, dest_path)

    print(f"  已按文件夹数字排序并复制 {group_key} 组到 {dest_folder}（共 {len(group_files)} 个文件）")
    print(f"  输出目录：{main_output}")

print("\n所有操作完成！")