import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings('ignore')  # 忽略无关警告

###############################################
# 1️⃣ 全局参数定义（修复路径+新增全部特征列）
###############################################
# 基础路径（结果保存目录，文件夹）
BASE_DIR = "F:/RBC_Resultsnewmodel2"
os.makedirs(BASE_DIR, exist_ok=True)

# 数据路径（CSV文件路径，逗号分隔）
CSV_PATH = os.path.abspath("C:/Users/Administrator/Desktop/RBCnewmodel.csv")
IMAGE_COL = "Image path"
LABEL_COL = "Disease"
GROUP_COL = "Patient ID"

# ========== 核心修改：新增全部训练参数列 ==========
feature_cols = [
    "volume",
    "HGB Mean",
    "Mean RMS displacement",
    "Max RMS displacement",
    "Area",
    "AvgVelocity(pixel/frame)",
    "MaxVelocity(pixel/frame)",
    "MinVelocity(pixel/frame)",
    "Average_DI",
    "TransitionTime"
]

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 15
PATIENCE = 3
NUM_FOLDS = 5
SEED = 42

# 可视化参数
GRADCAM_NUM = 3  # 每个折生成的Grad-CAM数量
DPI = 300  # 保存图片的DPI

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = DPI


###############################################
# 2️⃣ 工具函数（增强：适配新特征列+更鲁棒的数据清洗）
###############################################
def clean_data(df):
    """数据清洗：处理缺失值、异常值、非数值数据（适配所有新特征列）"""
    print("\n🧹 开始数据清洗...")

    # 1. 处理缺失值
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print(f"⚠️  发现缺失值：\n{missing_info[missing_info > 0]}")
        # 特征列用均值填充，标签/图像路径列删除缺失行
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                mean_val = df[col].astype(float).mean()
                df[col] = df[col].fillna(mean_val)
                print(f"   ✅ {col} 缺失值用均值 {mean_val:.4f} 填充")
        # 删除图像路径/标签缺失的行
        df = df.dropna(subset=[IMAGE_COL, LABEL_COL, GROUP_COL])

    # 2. 转换特征列为数值型（强制转换，兼容特殊格式）
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 再次检查是否有NaN（转换失败产生）
        if df[col].isnull().sum() > 0:
            mean_val = df[col].dropna().astype(float).mean()
            df[col] = df[col].fillna(mean_val)
            print(f"   ✅ {col} 非数值转换后填充均值 {mean_val:.4f}")

    # 3. 移除特征列异常值（3σ原则）
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        outlier_count = len(df[(df[col] < lower) | (df[col] > upper)])
        if outlier_count > 0:
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            print(f"   ✅ {col} 移除 {outlier_count} 个异常值（3σ范围：[{lower:.4f}, {upper:.4f}]）")

    # 4. 检查图像路径有效性
    valid_paths = []
    for idx, path in df[IMAGE_COL].items():
        try:
            path = str(path).strip().replace('\\', '/')
            path = os.path.abspath(path)
            valid_paths.append(os.path.exists(path))
        except:
            valid_paths.append(False)
    invalid_count = len(valid_paths) - sum(valid_paths)
    if invalid_count > 0:
        df = df[valid_paths]
        print(f"   ✅ 移除 {invalid_count} 个无效图像路径")

    # 5. 确保标签为二分类（0/1）
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce')
    label_unique = df[LABEL_COL].unique()
    if len(label_unique) > 2:
        print(f"⚠️  标签类别数 >2 ({label_unique})，自动转为二分类（0/1）")
        df[LABEL_COL] = (df[LABEL_COL] != label_unique[0]).astype(int)
    # 填充标签列可能的NaN
    df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)

    print(f"✅ 数据清洗完成：原始 {len(df)} 样本 → 清洗后 {len(df)} 样本")
    print(f"✅ 参与训练的特征列({len(feature_cols)}个)：{feature_cols}")
    return df.reset_index(drop=True)


###############################################
# 3️⃣ 数据集类定义（适配新特征维度）
###############################################
# 图像预处理（增加数据增强，仅训练集使用）
train_img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class RBCDataset(Dataset):
    def __init__(self, df, scaler=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.img_tf = train_img_tf if is_train else val_img_tf

        # 特征处理（适配新特征列）
        feats = self.df[feature_cols].values
        try:
            feats = feats.astype(np.float32)
        except ValueError as e:
            raise ValueError(f"❌ 特征列包含非数值数据：{feature_cols}\n详细错误：{e}")

        # 标准化
        if scaler:
            feats = scaler.transform(feats)
        self.features = torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            # 图像路径处理
            img_path = self.df.loc[idx, IMAGE_COL]
            img_path = str(img_path).strip().replace('\\', '/')
            img_path = os.path.abspath(img_path)

            # 加载图像
            with Image.open(img_path) as img_file:
                img = img_file.convert("RGB")
            img = self.img_tf(img)

            # 获取特征和标签
            x_tab = self.features[idx]
            y = torch.tensor(self.df.loc[idx, LABEL_COL], dtype=torch.float32)

            return img, x_tab, y, img_path

        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 图像文件不存在：{img_path}")
        except Exception as e:
            raise Exception(f"❌ 样本 {idx} 处理失败：{str(e)} | 路径：{img_path}")


###############################################
# 4️⃣ 模型类定义（适配新特征维度+增强MLP）
###############################################
class MultiModal(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # CNN分支（ResNet18）
        from torchvision.models import ResNet18_Weights
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 分层冻结：仅训练layer4和fc层
        for name, param in self.cnn.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        self.cnn.fc = nn.Identity()  # CNN输出：(batch, 512)
        cnn_out_dim = 512

        # 表格分支MLP（适配新特征维度，增强容量）
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),  # 原64→128，适配10维特征
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        mlp_out_dim = 32

        # 融合分类器（适配新维度）
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + mlp_out_dim, 256),  # 原128→256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab_feats):
        img_feats = self.cnn(img)  # (batch, 512)
        tab_feats = self.mlp(tab_feats)  # (batch, 32)
        fused_feats = torch.cat([img_feats, tab_feats], dim=1)  # (batch, 544)
        return self.classifier(fused_feats).squeeze(1)


###############################################
# 5️⃣ 训练/评估函数（无核心修改，保持原有逻辑）
###############################################
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (img, tab_feats, y, _) in enumerate(dataloader):
        img, tab_feats, y = img.to(device), tab_feats.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(img, tab_feats)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        # 累计指标
        total_loss += loss.item() * len(y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct_preds += (preds == y).sum().item()
        total_samples += len(y)

        # 每10个batch打印一次进度
        if (batch_idx + 1) % 10 == 0:
            batch_acc = correct_preds / total_samples
            print(f"   批次 {batch_idx + 1}/{len(dataloader)} | 训练损失：{loss.item():.4f} | 训练ACC：{batch_acc:.3f}")

    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy


def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for img, tab_feats, y, _ in dataloader:
            img, tab_feats, y = img.to(device), tab_feats.to(device), y.to(device)

            logits = model(img, tab_feats)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    # 计算指标
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0

    # 处理单标签情况
    has_two_classes = len(np.unique(all_labels)) == 2
    f1 = f1_score(all_labels, all_preds) if has_two_classes else np.nan
    auc = roc_auc_score(all_labels, all_probs) if has_two_classes else np.nan

    return avg_loss, accuracy, f1, auc, all_probs, all_labels


###############################################
# 6️⃣ Grad-CAM函数（无核心修改）
###############################################
def generate_gradcam(model, img_tensor, device):
    """
    生成Grad-CAM热力图（仅针对CNN分支）
    :param model: 多模态模型
    :param img_tensor: 单张图像张量 (3, 224, 224)
    :param device: 计算设备
    :return: cam: 归一化的热力图 (224, 224)
    """
    # 获取目标层
    target_layer = model.cnn.layer4[-1].conv2
    feature_map = None
    grads = None

    # 前向钩子
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()

    # 反向钩子
    def full_backward_hook(module, grad_input, grad_output):
        nonlocal grads
        grads = grad_output[0].detach()

    # 注册钩子
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(full_backward_hook)

    try:
        # 前向传播
        model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(device)
        cnn_feats = model.cnn(img_tensor)

        # 临时线性层用于求导
        temp_linear = nn.Linear(512, 1).to(device)
        logit = temp_linear(cnn_feats).squeeze(1)

        # 反向传播
        logit.backward()

        # 检查特征图和梯度
        if feature_map is None or grads is None:
            raise RuntimeError("❌ 未获取到特征图或梯度")
        if feature_map.shape != (1, 512, 7, 7):
            raise ValueError(f"❌ 特征图维度错误：{feature_map.shape}，预期(1,512,7,7)")

        # 计算CAM
        channel_weights = torch.mean(grads, dim=(2, 3))
        cam = torch.sum(channel_weights.unsqueeze(-1).unsqueeze(-1) * feature_map, dim=1)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 上采样到224x224
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        return cam

    finally:
        # 移除钩子
        forward_handle.remove()
        backward_handle.remove()


def plot_gradcam(img_tensor, cam, save_path):
    """
    绘制并保存Grad-CAM图像（修复cv2颜色映射问题）
    :param img_tensor: 图像张量 (3, 224, 224)
    :param cam: 热力图 (224, 224)
    :param save_path: 保存路径
    """
    # 反归一化图像
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    # 生成热力图（兼容cv2）
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    except:
        # 备用方案：使用matplotlib生成热力图
        import matplotlib.cm as cm
        heatmap = cm.jet(cam)[:, :, :3]

    # 叠加图像
    overlay_img = 0.5 * img_np + 0.5 * heatmap
    overlay_img = np.clip(overlay_img, 0, 1)

    # 保存图像
    plt.imsave(save_path, overlay_img)


###############################################
# 7️⃣ 主函数（适配新特征列+增强日志）
###############################################
if __name__ == '__main__':
    # 1. 设置随机种子（确保可复现）
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    # 2. 加载并清洗数据（核心修复：逗号分隔读取）
    print("=" * 80)
    print("📌 开始加载数据")
    print("=" * 80)
    try:
        # 核心修复：使用逗号分隔读取CSV（sep=','）
        df = pd.read_csv(CSV_PATH, sep=',', header=0, encoding='utf-8')
        print(f"✅ 成功读取CSV文件：{CSV_PATH}")
        print(f"📊 原始数据：{len(df)} 样本，{len(df.columns)} 列")
        print(f"📋 文件实际列名：{df.columns.tolist()}")
    except Exception as e:
        raise ValueError(f"❌ 读取CSV失败：{str(e)}")

    # 验证列名（增强容错：忽略大小写/空格）
    df.columns = [col.strip() for col in df.columns]
    required_cols = [IMAGE_COL.strip(), LABEL_COL.strip(), GROUP_COL.strip()] + feature_cols
    missing_cols = []
    for col in required_cols:
        if col not in df.columns and col.lower() not in [c.lower() for c in df.columns]:
            missing_cols.append(col)

    if missing_cols:
        print(f"⚠️  代码需要的列：{required_cols}")
        print(f"⚠️  文件实际的列：{df.columns.tolist()}")
        raise ValueError(f"❌ 缺失必需列：{missing_cols}")
    print("✅ 所有必需列均存在！")

    # 数据清洗
    df = clean_data(df)

    # 3. 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if device == "cuda" else False
    print(f"\n💻 训练设备：{device}")
    if device == "cuda":
        print(f"   GPU名称：{torch.cuda.get_device_name(0)}")

    # 4. 创建保存目录
    dirs = [
        os.path.join(BASE_DIR, "gradcam"),
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "curves")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 5. 初始化交叉验证
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    all_roc_data = []

    # 6. 开始交叉验证训练
    print("\n" + "=" * 80)
    print("📌 开始5折交叉验证训练（使用全部特征列）")
    print("=" * 80)

    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(df, y=df[LABEL_COL]), 1):
        print(f"\n" + "-" * 80)
        print(f"📌 第 {fold_idx}/{NUM_FOLDS} 折训练")
        print("-" * 80)

        # 划分训练/测试集
        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        # 打印折信息
        train_label_dist = train_df[LABEL_COL].value_counts().to_dict()
        test_label_dist = test_df[LABEL_COL].value_counts().to_dict()
        print(f"📊 第 {fold_idx} 折数据分布：")
        print(f"   训练集：{len(train_df)} 样本 | 标签分布：{train_label_dist}")
        print(f"   测试集：{len(test_df)} 样本 | 标签分布：{test_label_dist}")
        print(f"   特征维度：{len(feature_cols)} 维 | 特征列：{feature_cols}")

        # 标准化特征
        tab_scaler = StandardScaler()
        tab_scaler.fit(train_df[feature_cols].values)

        # 构建DataLoader
        train_dataset = RBCDataset(train_df, scaler=tab_scaler, is_train=True)
        test_dataset = RBCDataset(test_df, scaler=tab_scaler, is_train=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )

        # 初始化模型（适配新特征维度）
        model = MultiModal(num_features=len(feature_cols)).to(device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=1e-5  # L2正则化
        )
        loss_fn = nn.BCEWithLogitsLoss()

        # 早停参数
        best_val_loss = float('inf')
        early_stop_count = 0
        train_loss_history = []
        val_loss_history = []

        # 训练循环
        print(f"\n🚀 开始训练（最大轮数：{MAX_EPOCHS}，早停耐心值：{PATIENCE}）")
        for epoch in range(1, MAX_EPOCHS + 1):
            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
            # 验证
            val_loss, val_acc, val_f1, val_auc, val_probs, val_labels = eval_epoch(model, test_loader, loss_fn, device)

            # 记录损失
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
                # 保存最佳模型
                model_save_path = os.path.join(BASE_DIR, "models", f"best_model_fold{fold_idx}.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scaler": tab_scaler,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "feature_cols": feature_cols  # 保存使用的特征列
                }, model_save_path)
                print(f"🔸 第 {epoch} 轮：保存最佳模型 → {model_save_path}")
            else:
                early_stop_count += 1
                if early_stop_count >= PATIENCE:
                    print(f"⚠️  早停触发！第 {epoch} 轮验证损失未改善")
                    break

            # 打印轮次结果
            f1_str = f"{val_f1:.3f}" if not np.isnan(val_f1) else "N/A"
            auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A"
            print(f"📈 第 {epoch:2d}/{MAX_EPOCHS} 轮 | "
                  f"训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f} | "
                  f"验证ACC：{val_acc:.3f} | 验证F1：{f1_str} | 验证AUC：{auc_str}")

        # 记录折结果
        if not np.isnan(val_f1) and not np.isnan(val_auc):
            fold_results.append({
                "fold": fold_idx,
                "accuracy": val_acc,
                "f1": val_f1,
                "auc": val_auc
            })
            all_roc_data.append((val_labels, val_probs))
            print(f"✅ 第 {fold_idx} 折结果已记录")
        else:
            print(f"⚠️  第 {fold_idx} 折标签单一，跳过结果记录")

        # 绘制学习曲线
        if len(train_loss_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history,
                     label="训练损失", color="#2E86AB", linewidth=2, marker='o', markersize=4)
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history,
                     label="验证损失", color="#A23B72", linewidth=2, marker='s', markersize=4)

            # 标记最佳轮次
            best_epoch = np.argmin(val_loss_history) + 1
            plt.scatter(best_epoch, val_loss_history[best_epoch - 1],
                        color="#F18F01", s=100, zorder=5, label=f"最佳轮次 ({best_epoch})")

            plt.xlabel("训练轮次 (Epoch)", fontsize=12)
            plt.ylabel("损失 (Loss)", fontsize=12)
            plt.title(f"第 {fold_idx} 折 - 学习曲线（{len(feature_cols)}维特征）", fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()

            curve_path = os.path.join(BASE_DIR, "curves", f"learning_curve_fold{fold_idx}.png")
            plt.savefig(curve_path, dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"✅ 学习曲线已保存：{curve_path}")

        # 生成Grad-CAM
        if len(np.unique(val_labels)) == 2:
            print(f"\n🎯 生成Grad-CAM示例（{GRADCAM_NUM}个）...")
            cam_count = 0
            model.eval()

            for img_batch, tab_batch, y_batch, path_batch in test_loader:
                if cam_count >= GRADCAM_NUM:
                    break

                for i in range(len(img_batch)):
                    if cam_count >= GRADCAM_NUM:
                        break

                    try:
                        # 生成CAM
                        cam = generate_gradcam(model, img_batch[i], device)
                        # 保存CAM图像
                        cam_save_path = os.path.join(BASE_DIR, "gradcam", f"fold{fold_idx}_cam{cam_count + 1}.png")
                        plot_gradcam(img_batch[i], cam, cam_save_path)

                        print(f"✅ Grad-CAM保存成功：{cam_save_path}")
                        cam_count += 1
                    except Exception as e:
                        print(f"⚠️  样本 {i} 生成Grad-CAM失败：{str(e)}")
                        continue

            if cam_count == 0:
                print(f"❌ 第 {fold_idx} 折未生成任何Grad-CAM示例")
        else:
            print(f"⚠️  第 {fold_idx} 折标签单一，跳过Grad-CAM生成")

    # 8. 结果汇总
    print("\n" + "=" * 80)
    print("📊 5折交叉验证结果汇总（使用全部特征列）")
    print("=" * 80)

    if fold_results:
        # 转换为DataFrame
        results_df = pd.DataFrame(fold_results)

        # 计算平均值和标准差
        avg_acc = results_df["accuracy"].mean()
        std_acc = results_df["accuracy"].std()
        avg_f1 = results_df["f1"].mean()
        std_f1 = results_df["f1"].std()
        avg_auc = results_df["auc"].mean()
        std_auc = results_df["auc"].std()

        # 添加汇总行
        summary_row = {
            "fold": "平均值 (Average)",
            "accuracy": avg_acc,
            "f1": avg_f1,
            "auc": avg_auc
        }
        std_row = {
            "fold": "标准差 (Std)",
            "accuracy": std_acc,
            "f1": std_f1,
            "auc": std_auc
        }
        results_df = pd.concat([results_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)

        # 打印结果
        print("\n📈 详细结果：")
        print(results_df.round(3))

        print(f"\n🎯 最终结果（{len(feature_cols)}维特征）：")
        print(f"   准确率 (Accuracy)：{avg_acc:.3f} ± {std_acc:.3f}")
        print(f"   F1分数 (F1-Score)：{avg_f1:.3f} ± {std_f1:.3f}")
        print(f"   ROC-AUC        ：{avg_auc:.3f} ± {std_auc:.3f}")

        # 保存结果表格
        results_path = os.path.join(BASE_DIR, "cross_validation_results.csv")
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 结果表格已保存：{results_path}")

        # 绘制整体ROC曲线
        all_labels = np.concatenate([data[0] for data in all_roc_data])
        all_probs = np.concatenate([data[1] for data in all_roc_data])
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        overall_auc = roc_auc_score(all_labels, all_probs)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color="#2E86AB", linewidth=3, label=f"多模态模型 (AUC = {overall_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#A23B72", linewidth=2, linestyle="--", label="随机猜测")

        plt.xlabel("假阳性率 (False Positive Rate)", fontsize=12)
        plt.ylabel("真阳性率 (True Positive Rate)", fontsize=12)
        plt.title(f"多模态模型 - 整体ROC曲线（{len(feature_cols)}维特征）", fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

        roc_path = os.path.join(BASE_DIR, "overall_roc_curve.png")
        plt.savefig(roc_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ 整体ROC曲线已保存：{roc_path}")

    else:
        print(f"⚠️  所有折均标签单一，无法统计汇总结果")

    # 9. 最终提示
    print("\n" + "=" * 80)
    print(f"🎉 训练完成！所有结果已保存至：{BASE_DIR}")
    print(f"✅ 参与训练的特征列：{feature_cols}")
    print("=" * 80)