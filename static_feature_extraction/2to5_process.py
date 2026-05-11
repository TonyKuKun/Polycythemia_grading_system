import subprocess
import os

# ====================== 只需要填这 1 个根目录 ======================
BASE_ROOT = r"D:\19\9\channel_3"

# 下面全部自动生成，不用你再改！
dataset = os.path.join(BASE_ROOT, "matlabphotos2")
seg_dataset = os.path.join(BASE_ROOT, "matlabphotos2-seg")
background_image_path = os.path.join(BASE_ROOT, "background.png")

root_path = os.path.dirname(seg_dataset)
cla_path = os.path.join(root_path, "3_cla")
background_path = os.path.join(root_path, "4_background")

# 自动生成你要的 background2_dir
background2_dir = os.path.join(BASE_ROOT, "4_background2")
# =================================================================

os.makedirs(cla_path, exist_ok=True)
os.makedirs(background_path, exist_ok=True)
os.makedirs(background2_dir, exist_ok=True)

if __name__ == "__main__":
    print("choice 0 :run all")
    print("choice 1 :从3_add_bakcround开始运行")
    print("choice 2 :从4_mask开始运行")
    print("choice 3 :从5_for_matlab开始运行")
    choice = int(input("please enter your choice(0, 1, 2 or 3): "))

    if choice == 0:
        subprocess.run(["python", "2_cla3.py", "--input_root", seg_dataset, "--output_root", cla_path], check=True)
        subprocess.run(["python", "3_add_background.py", cla_path, dataset, background_path], check=True)
        subprocess.run(["python", "4_mask.py", cla_path, background_path], check=True)
        subprocess.run(["python", "5_for_matlab.py", background_path, background2_dir, background_image_path], check=True)

    if choice == 1:
        subprocess.run(["python", "3_add_background.py", cla_path, dataset, background_path], check=True)
        subprocess.run(["python", "4_mask.py", cla_path, background_path], check=True)
        subprocess.run(["python", "5_for_matlab.py", background_path, background2_dir, background_image_path], check=True)

    if choice == 2:
        subprocess.run(["python", "4_mask.py", cla_path, background_path], check=True)
        subprocess.run(["python", "5_for_matlab.py", background_path, background2_dir, background_image_path], check=True)

    if choice == 3:
        subprocess.run(["python", "5_for_matlab.py", background_path, background2_dir, background_image_path], check=True)