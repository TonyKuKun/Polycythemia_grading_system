import os
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from torchvision import transforms, models
import torch.nn as nn

warnings.filterwarnings("ignore")

# ========== 1. 全局配置 ==========
SUBTYPE_COLOR_MAP = {
    1: "#2ca02c",
    2: "#d62728",
    3: "#ff7f0e",
    4: "#1f77b4",
    5: "#9467bd",
    6: "#8c564b",
    7: "#98df8a",
    8: "#9ecae1",
    9: "#7f7f7f",
    10: "#a9a9a9"
}

DISEASE_COLOR_MAP = {
    "NORMAL": "#2C7FB8",
    "HAPC": "#F28E2B",
}

NUM_TO_DISEASE_TEXT = {
    0: "NORMAL",
    1: "HAPC",
    "0": "NORMAL",
    "1": "HAPC",
    "unknown": "unknown"
}

VIS_CONFIG = {
    "figure_size": (12, 9),
    "axis_label_size": 18,
    "tick_label_size": 15,
    "title_size": 22,
    "legend_size": 16,
    "scatter_size": 80,
    "annot_size": 12
}

# 🚀 已修改为 SVM 配置
CLASSIFIER_CONFIG = {
    "prob_threshold": 0.5,
    "random_state": 42,
    "use_high_dim_features": True,
    "calibrate_probs": True,
    "svm_param_grid": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf"]
    }
}

# ✅ 模型训练时的完整10维特征列
MODEL_TRAIN_FEATURES = [
    "volume", "HGB Mean", "Mean RMS displacement", "Max RMS displacement",
    "TransitionTime", "Area", "Average_DI",
    "AvgVelocity", "MaxVelocity", "MinVelocity"
]

# ✅ 速度特征列
VELOCITY_COLS = [
    "AvgVelocity",
    "MaxVelocity",
    "MinVelocity"
]


# ========== 2. ✅ 模型结构（100% 还原训练代码）==========
class MultiModal(nn.Module):
    def __init__(self, num_features=10):
        super().__init__()
        # CNN分支（ResNet18）
        from torchvision.models import ResNet18_Weights
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 分层冻结
        for name, param in self.cnn.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        self.cnn.fc = nn.Identity()  # CNN输出：(batch, 512)
        cnn_out_dim = 512

        # 表格分支MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
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

        # 融合分类器（必须定义以加载权重）
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + mlp_out_dim, 256),
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
        return fused_feats


# ========== 3. 特征预处理（补全缺失列）==========
def create_derived_features(data_df):
    print("=== 📊 特征预处理（补全缺失列）===")
    df = data_df.copy()

    # 补全10维原始特征（缺失列填充0）
    for feat in MODEL_TRAIN_FEATURES:
        if feat not in df.columns:
            print(f"⚠️ 缺失原始特征 {feat}，填充默认值0")
            df[feat] = 0.0

    print(f"✅ 输入模型特征：10维原始特征")
    print(f"✅ 特征列表：{MODEL_TRAIN_FEATURES}")
    return df, MODEL_TRAIN_FEATURES


# ========== 4. 列名检查 ==========
def check_column_names(data_df, target_cols):
    print("=== 🔍 核心列名检查 ===")
    data_df.columns = [str(x).strip() for x in data_df.columns]
    core_cols = ["Image path", "Disease", "Patient ID"]
    missing_core_cols = [col for col in core_cols if col not in data_df.columns]
    if missing_core_cols:
        raise ValueError(f"❌ 核心列缺失：{missing_core_cols}")
    print("✅ 核心列检查通过")
    return data_df


# ========== 5. Disease 标签映射 ==========
def get_disease_mapping(file_path):
    print("=== 🏷️  解析Disease标签映射 ===")
    data_df = pd.read_excel(file_path, sheet_name=0, header=0, engine="openpyxl")
    data_df.columns = [str(x).strip() for x in data_df.columns]

    unique_diseases = data_df["Disease"].unique()
    print(f"✅ 检测到唯一标签：{unique_diseases}")

    disease_mapping = {}
    for label in unique_diseases:
        if pd.isna(label):
            disease_mapping[label] = "unknown"
        elif str(label).strip() in ["0", "0.0", "NORMAL"]:
            disease_mapping[label] = 0
        elif str(label).strip() in ["1", "1.0", "HAPC"]:
            disease_mapping[label] = 1
        else:
            disease_mapping[label] = "unknown"
    print(f"✅ 标签映射：{disease_mapping}")
    return disease_mapping


# ========== 6. ✅ 强制修复版：确保速度特征归一化一定执行 ==========
def extract_fused_features(model_path, data_df, base_feature_cols, disease_mapping, device):
    print("\n=== 🚀 提取多模态融合特征（含强制速度归一化）===")
    model = MultiModal(num_features=10).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print("✅ 模型权重加载成功")
    except Exception as e:
        raise RuntimeError(f"❌ 模型加载失败：{str(e)}")

    model.eval()

    # ✅ 1. 加载标准化器
    base_scaler = checkpoint["scaler"]
    print(f"✅ [1/4] 加载全局标准化器 (Base Scaler)：输入维度={base_scaler.n_features_in_}")

    velocity_scaler = None
    if "velocity_scaler" in checkpoint:
        velocity_scaler = checkpoint["velocity_scaler"]
        print("✅ [2/4] 加载训练集速度标准化器 (Velocity Scaler)")
    else:
        print("⚠️  [2/4] Checkpoint中未找到 velocity_scaler！")
        print("⚠️  将使用当前数据的统计量临时拟合速度标准化器（这可能与训练集分布有偏差）")
        # 临时拟合一个速度scaler，保证流程不中断
        velocity_scaler = StandardScaler()
        temp_vel_data = data_df[VELOCITY_COLS].values.astype(np.float32)
        temp_vel_data = np.nan_to_num(temp_vel_data, nan=0.0)
        velocity_scaler.fit(temp_vel_data)

    # 图像预处理
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_fused = []
    all_labels = []
    all_subtypes = []
    valid_indices = []
    unknown_samples_info = {}

    print(f"\n✅ [3/4] 开始处理样本，将严格执行：速度归一化 -> 全局归一化")

    # 为了打印日志，取前几个样本看看
    debug_printed = False

    with torch.no_grad():
        total_samples = len(data_df)
        valid_count = 0
        for idx, (row_idx, row) in enumerate(data_df.iterrows(), 1):
            # 路径处理
            img_path = str(row["Image path"]).strip().replace("\\", "/")
            if not os.path.exists(img_path) and os.path.exists(f"./{os.path.basename(img_path)}"):
                img_path = f"./{os.path.basename(img_path)}"

            # ID处理
            try:
                subtype_id = int(row["Patient ID"])
            except Exception:
                subtype_id = f"Unknown_{idx}"

            # 标签处理
            raw_disease = row["Disease"]
            mapped_disease = disease_mapping.get(raw_disease, "unknown")

            # 加载图片
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                continue

            # ✅ 2. 核心修复：特征处理流程与训练代码完全一致
            try:
                # 步骤 A: 提取原始 10 维特征
                base_feat = np.array([row[col] for col in base_feature_cols]).astype(np.float32)
                base_feat = np.nan_to_num(base_feat, nan=0.0)

                # 打印调试信息（仅一次）
                if not debug_printed:
                    print(f"\n   🧪 样本 {idx} 归一化前预览:")
                    for name, val in zip(base_feature_cols, base_feat):
                        print(f"     {name:<25} : {val:.4f}")

                # 步骤 B: 先做速度特征归一化
                # 把 numpy 转成 DataFrame 以确保列名对应
                feat_df = pd.DataFrame([base_feat], columns=base_feature_cols)

                # 提取速度列 -> 归一化 -> 放回
                vel_vals = feat_df[VELOCITY_COLS].values
                vel_vals_scaled = velocity_scaler.transform(vel_vals)
                feat_df[VELOCITY_COLS] = vel_vals_scaled

                # 转回 numpy
                processed_feat = feat_df.values[0]

                # 步骤 C: 再做全局特征归一化
                tab_feat_scaled = base_scaler.transform([processed_feat])[0]

                # 打印调试信息（仅一次）
                if not debug_printed:
                    print(f"   🧪 样本 {idx} 归一化后预览:")
                    for name, val in zip(base_feature_cols, tab_feat_scaled):
                        print(f"     {name:<25} : {val:.4f}")
                    print("   ✅ 速度归一化逻辑已执行！\n")
                    debug_printed = True

            except Exception as e:
                print(f"⚠️ 跳过样本{idx}：特征处理失败 - {e}")
                import traceback
                traceback.print_exc()
                continue

            # 模型推理
            img_tensor = img_transform(img).unsqueeze(0).to(device)
            tab_tensor = torch.tensor([tab_feat_scaled], dtype=torch.float32).to(device)
            fused = model(img_tensor, tab_tensor).cpu().numpy()

            # 保存结果
            all_fused.append(fused)
            all_labels.append(mapped_disease)
            all_subtypes.append(subtype_id)
            valid_indices.append(row_idx)
            valid_count += 1

            # 存入未知样本字典
            if mapped_disease == "unknown":
                if subtype_id not in unknown_samples_info:
                    unknown_samples_info[subtype_id] = {
                        "fused_feats": [],
                        "sample_indices": []
                    }
                unknown_samples_info[subtype_id]["fused_feats"].append(fused[0])
                unknown_samples_info[subtype_id]["sample_indices"].append(len(all_fused) - 1)

            if idx % 50 == 0:
                print(f"🔄 已处理 {idx}/{total_samples} 样本")

    fused_feats = np.vstack(all_fused) if all_fused else np.array([])
    true_labels = np.array(all_labels)
    subtypes = np.array(all_subtypes)

    print(f"\n✅ [4/4] 特征提取完成")
    return fused_feats, true_labels, subtypes, valid_indices, unknown_samples_info


# ========== 7. ✅ 训练 SVM 分类器 ==========
def train_xgboost_classifier(X, y):  # 函数名保持不变，避免修改主流程
    print("\n=== 🚀 训练 SVM 分类器 ===")
    print(f"输入特征维度：{X.shape[1]}，样本数：{len(X)}")

    # 特征标准化（SVM对尺度非常敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 定义 SVM
    svm_clf = SVC(
        probability=True,
        random_state=CLASSIFIER_CONFIG["random_state"],
        cache_size=1000
    )

    # 网格搜索
    grid_search = GridSearchCV(
        svm_clf,
        CLASSIFIER_CONFIG["svm_param_grid"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_scaled, y)
    print(f"✅ 最优超参数：{grid_search.best_params_}")
    print(f"✅ 最优CV AUC分数：{grid_search.best_score_:.4f}")

    # 概率校准
    best_svm = grid_search.best_estimator_
    if CLASSIFIER_CONFIG["calibrate_probs"]:
        clf = CalibratedClassifierCV(best_svm, cv=5, method="sigmoid")
        clf.fit(X_scaled, y)
        print("✅ SVM概率校准完成")
    else:
        clf = best_svm

    return clf, scaler


# ========== 8. 绘制聚类图 ==========
def plot_subtype_and_disease_figures(
        feats_2d, fused_feats, subtypes, true_labels, unknown_samples_info, valid_indices):
    from matplotlib.lines import Line2D
    save_dir = "unknown_samples_prediction_svm_final"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== 📊 绘制聚类图 ===")

    # 标签转文本
    label_text = []
    for l in true_labels:
        if str(l) == "0" or l == 0:
            label_text.append("NORMAL")
        elif str(l) == "1" or l == 1:
            label_text.append("HAPC")
        else:
            label_text.append("unknown")
    label_text = np.array(label_text)

    # 训练分类器并预测
    known_mask = np.isin(label_text, ["HAPC", "NORMAL"])
    X_known = fused_feats[known_mask] if CLASSIFIER_CONFIG["use_high_dim_features"] else feats_2d[known_mask]
    y_known = label_text[known_mask]
    y_known_binary = np.where(y_known == "HAPC", 1, 0)
    clf = None
    clf_scaler = None

    if len(X_known) >= 10 and len(np.unique(y_known)) == 2:
        clf, clf_scaler = train_xgboost_classifier(X_known, y_known_binary)

        print("\n🔮 开始预测未知样本：")
        for subtype_id, info in unknown_samples_info.items():
            subtype_feats = np.array(info["fused_feats"])
            mean_feat = np.mean(subtype_feats, axis=0).reshape(1, -1)
            mean_feat_scaled = clf_scaler.transform(mean_feat)
            hapc_prob = clf.predict_proba(mean_feat_scaled)[0][1]
            pred = "HAPC" if hapc_prob >= CLASSIFIER_CONFIG["prob_threshold"] else "NORMAL"
            info["pred_disease"] = pred
            info["pred_prob"] = float(hapc_prob)
            info["center_xy"] = np.mean(feats_2d[info["sample_indices"]], axis=0)
            print(f"✅ 患者 {subtype_id}: {pred} (Prob: {hapc_prob:.4f})")

    # ======================================
    # Figure 1：样本聚类图
    # ======================================
    fig1, ax1 = plt.subplots(figsize=VIS_CONFIG["figure_size"])
    ax1.set_facecolor("#F8F6F2")

    unique_subtypes = list(dict.fromkeys(subtypes.tolist()))
    for st in unique_subtypes:
        mask = (subtypes == st)
        try:
            st_int = int(st)
            color = SUBTYPE_COLOR_MAP.get(st_int, "#999999")
            label = str(st_int)
        except Exception:
            color = "#999999"
            label = str(st)

        ax1.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            s=VIS_CONFIG["scatter_size"],
            c=color,
            alpha=0.80,
            edgecolors="black",
            linewidths=0.4,
            label=label
        )

    # 标注预测结果（动态防重叠）
    y_range = feats_2d[:, 1].max() - feats_2d[:, 1].min()
    text_offset = 0

    for subtype_id, info in unknown_samples_info.items():
        if "center_xy" not in info:
            continue
        center_xy = info["center_xy"]
        pred = info.get("pred_disease", "unknown")
        prob = info.get("pred_prob", 0.0)

        bbox_color = "#90EE90" if prob >= 0.7 else "#FFFFE0"
        current_y_offset = (y_range * 0.02) + text_offset

        ax1.text(
            center_xy[0] + (feats_2d[:, 0].max() - feats_2d[:, 0].min()) * 0.01,
            center_xy[1] + current_y_offset,
            f"ID:{subtype_id}\n{pred}\n{prob:.4f}",
            fontsize=VIS_CONFIG["annot_size"],
            fontweight="bold",
            color="black",
            bbox=dict(facecolor=bbox_color, alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.3"),
            zorder=10
        )
        text_offset += y_range * 0.06

    ax1.set_xlabel("PCA Component 1", fontsize=VIS_CONFIG["axis_label_size"], fontweight="bold")
    ax1.set_ylabel("PCA Component 2", fontsize=VIS_CONFIG["axis_label_size"], fontweight="bold")
    ax1.set_title("Multi-modal Features Clustering (SVM)", fontsize=24, fontweight="bold", pad=20)
    ax1.grid(True, linestyle="--", alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels1, handles1))
    ax1.legend(by_label.values(), by_label.keys(), title="Patient ID", fontsize=VIS_CONFIG["legend_size"])

    subtype_fig_path = os.path.join(save_dir, "figure_subtype_clustering.png")
    plt.tight_layout()
    plt.savefig(subtype_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"✅ 样本聚类图已保存")

    # ======================================
    # Figure 2：疾病分类图
    # ======================================
    fig2, ax2 = plt.subplots(figsize=VIS_CONFIG["figure_size"])
    ax2.set_facecolor("#EDE2D6")

    disease_plot_labels = label_text.copy()
    for subtype_id, info in unknown_samples_info.items():
        pred = info.get("pred_disease", "unknown")
        if pred in ["HAPC", "NORMAL"]:
            disease_plot_labels[subtypes == subtype_id] = pred

    for cls in ["HAPC", "NORMAL", "unknown"]:
        mask = (disease_plot_labels == cls)
        if np.sum(mask) == 0:
            continue
        color = DISEASE_COLOR_MAP.get(cls, "#7f7f7f")
        ax2.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            s=80,
            c=color,
            alpha=0.80,
            edgecolors="black" if cls == "unknown" else "none",
            label=cls
        )

    ax2.set_xlabel("PCA Component 1", fontsize=VIS_CONFIG["axis_label_size"], fontweight="bold")
    ax2.set_ylabel("PCA Component 2", fontsize=VIS_CONFIG["axis_label_size"], fontweight="bold")
    ax2.set_title("Multimodal Separation (SVM)", fontsize=VIS_CONFIG["title_size"], fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="HAPC", markerfacecolor=DISEASE_COLOR_MAP["HAPC"], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="NORMAL", markerfacecolor=DISEASE_COLOR_MAP["NORMAL"],
               markersize=12)
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=16)

    disease_fig_path = os.path.join(save_dir, "figure_disease_clustering.png")
    plt.tight_layout()
    plt.savefig(disease_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"✅ 疾病分类图已保存")

    # ======================================
    # 保存Excel
    # ======================================
    result_df = pd.DataFrame({
        "Patient_ID": subtypes,
        "True_Label": label_text,
        "Predicted_Label": disease_plot_labels,
        "HAPC_Probability": pd.Series([np.nan] * len(subtypes), dtype="float"),
        "PCA_1": feats_2d[:, 0],
        "PCA_2": feats_2d[:, 1]
    })

    for subtype_id, info in unknown_samples_info.items():
        subtype_mask = (result_df["Patient_ID"] == subtype_id)
        result_df.loc[subtype_mask, "HAPC_Probability"] = info.get("pred_prob", np.nan)

    excel_path = os.path.join(save_dir, "clustering_results.xlsx")
    result_df.to_excel(excel_path, index=False)
    print(f"✅ 结果Excel已保存")

    return subtype_fig_path, disease_fig_path, excel_path


# ========== 10. ✅ 完整主流程 ==========
if __name__ == "__main__":
    # ================= 配置区域 =================
    file_path = r"F:\RBC1.xlsx"  # 你的Excel数据路径
    model_path = r"F:\RBC_Results2\models\best_model_fold5.pth"  # 你的模型路径
    # ===========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("📊 多模态特征聚类（SVM分类 + 双Scaler归一化）")
    print("=" * 80)
    print(f"💻 运行设备：{device}")
    print(f"📁 数据文件：{file_path}")
    print(f"📁 模型文件：{model_path}")
    print("=" * 80)

    try:
        # 1. 解析标签
        disease_mapping = get_disease_mapping(file_path)

        # 2. 读取数据
        data_df = pd.read_excel(file_path, sheet_name=0, header=0, engine="openpyxl")
        data_df = check_column_names(data_df, ["Image path", "Disease", "Patient ID"])

        # 3. 特征预处理
        data_df, base_feature_cols = create_derived_features(data_df)

        # 4. 提取特征
        fused_feats, true_labels, subtypes, valid_indices, unknown_samples_info = extract_fused_features(
            model_path=model_path,
            data_df=data_df,
            base_feature_cols=base_feature_cols,
            disease_mapping=disease_mapping,
            device=device
        )

        if len(fused_feats) == 0:
            print("❌ 无有效样本，程序退出")
            raise SystemExit(1)

        # 5. PCA降维
        print("\n=== 步骤5：PCA降维 ===")
        pca = PCA(n_components=2)
        feats_2d = pca.fit_transform(fused_feats)
        print(f"✅ PCA降维完成，解释方差比：{pca.explained_variance_ratio_.sum():.2%}")

        # 6. 绘图
        print("\n=== 步骤6：绘制聚类图 ===")
        subtype_plot_path, disease_plot_path, excel_path = plot_subtype_and_disease_figures(
            feats_2d=feats_2d,
            fused_feats=fused_feats,
            subtypes=subtypes,
            true_labels=true_labels,
            unknown_samples_info=unknown_samples_info,
            valid_indices=valid_indices
        )

        print("\n" + "=" * 80)
        print("🎉 运行完成！")
        print("=" * 80)
        print(f"📈 样本聚类图：{os.path.abspath(subtype_plot_path)}")
        print(f"📈 疾病分类图：{os.path.abspath(disease_plot_path)}")
        print(f"📋 结果表格：{os.path.abspath(excel_path)}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")
        import traceback

        traceback.print_exc()
        raise