"""
================================================================================
多模态红细胞表征框架（无需任务特异性训练）
更新内容：
  1. 强制 Patient ID 7-11 标记为 HAPC
  2. 新增 HAPC 严重程度亚类标签（Mild/Moderate/Severe）
  3. 新增亚类可视化与结果输出
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cross_decomposition import CCA
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from scipy.stats import mannwhitneyu, kruskal

warnings.filterwarnings("ignore")

# 尝试导入可选依赖
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ umap-learn 未安装，将使用 PCA 替代 UMAP。安装命令：pip install umap-learn")

try:
    import snf
    HAS_SNF = True
except ImportError:
    HAS_SNF = False
    print("⚠️ snfpy 未安装，SNF融合将跳过。安装命令：pip install snfpy")


# ==================== 全局配置 ====================
CONFIG = {
    "data_path": r"C:\Users\Administrator\Desktop\RBC22.xlsx",
    "output_dir": "results_multimodal_framework2",

    "pretrained_model": "resnet18",
    "img_size": 224,
    "img_pca_dim": 20,

    "umap_n_neighbors": 15,
    "umap_min_dist": 0.3,
    "umap_metric": "euclidean",

    "n_clusters_range": [2, 3, 4, 5],
    "clustering_method": "agglomerative",

    "cca_n_components": 5,
    "snf_k": 5,
    "snf_mu": 0.5,

    "figure_dpi": 300,
    "scatter_size": 60,
}

TABULAR_FEATURES = [
    "volume", "HGB Mean", "Mean RMS displacement", "Max RMS displacement",
    "TransitionTime", "Area", "Average_DI",
    "AvgVelocity", "MaxVelocity", "MinVelocity"
]

# 基础颜色方案
DISEASE_COLORS = {"NORMAL": "#2C7FB8", "HAPC": "#F28E2B", "unknown": "#7f7f7f"}
CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                  "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]
PATIENT_COLORS = {
    1: "#2ca02c", 2: "#d62728", 3: "#ff7f0e", 4: "#1f77b4", 5: "#9467bd",
    6: "#8c564b", 7: "#98df8a", 8: "#9ecae1", 9: "#7f7f7f", 10: "#a9a9a9"
}

# 🔴 新增：HAPC 严重程度亚类映射与配色
HAPC_SEVERITY_MAP = {
    1: "Mild", 9: "Mild", 11: "Mild",
    7: "Moderate", 8: "Moderate", 10: "Moderate",
    2: "Severe", 3: "Severe"
}
SEVERITY_COLORS = {
    "Mild": "#2ca02c",      # 绿
    "Moderate": "#ff7f0e",  # 橙
    "Severe": "#d62728",    # 红
    "NORMAL": "#2C7FB8",    # 蓝
    "unknown": "#7f7f7f"    # 灰
}


# ==================== 1. 数据加载 ====================
def load_data(file_path):
    print("=" * 70)
    print("📂 Step 1: 加载数据 & 标签处理")
    print("=" * 70)

    df = pd.read_excel(file_path, sheet_name=0, header=0, engine="openpyxl")
    df.columns = [str(x).strip() for x in df.columns]

    required_cols = ["Image path", "Disease", "Patient ID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ 缺失必要列：{missing}")

    for feat in TABULAR_FEATURES:
        if feat not in df.columns:
            print(f"  ⚠️ 缺失特征列 '{feat}'，填充0")
            df[feat] = 0.0

    def map_disease(label):
        if pd.isna(label):
            return "unknown"
        s = str(label).strip()
        if s in ["0", "0.0", "NORMAL"]:
            return "NORMAL"
        elif s in ["1", "1.0", "HAPC"]:
            return "HAPC"
        return "unknown"

    df["Disease_Label"] = df["Disease"].apply(map_disease)

    # 🔴 强制覆盖：Patient ID 7-11 设为 HAPC
    force_hapc_ids = [7, 8, 9, 10, 11, 12]
    pid_numeric = pd.to_numeric(df["Patient ID"], errors="coerce")
    override_mask = pid_numeric.isin(force_hapc_ids)
    df.loc[override_mask, "Disease_Label"] = "HAPC"
    print(f"  ✅ 已强制覆盖 {override_mask.sum()} 个样本的标签为 HAPC")

    # 🔴 新增：生成严重程度亚类标签
    def map_severity(row):
        disease = row["Disease_Label"]
        if disease != "HAPC":
            return disease
        try:
            pid_int = int(float(row["Patient ID"]))
            return HAPC_SEVERITY_MAP.get(pid_int, "unknown")
        except:
            return "unknown"

    df["Severity_Label"] = df.apply(map_severity, axis=1)

    print(f"  ✅ 总样本数: {len(df)}")
    print(f"  ✅ 疾病标签分布: {df['Disease_Label'].value_counts().to_dict()}")
    print(f"  ✅ 严重程度分布: {df['Severity_Label'].value_counts().to_dict()}")
    return df


# ==================== 2. 预训练ResNet特征提取 ====================
def extract_image_features(df, model_name="resnet18", img_size=224):
    print("\n" + "=" * 70)
    print("🖼️  Step 2: 预训练ResNet图像特征提取（frozen, no training）")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  💻 设备: {device}")

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = 512
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = 2048
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        feat_dim = 2048
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"  ✅ 加载 {model_name}（ImageNet预训练，完全冻结）")

    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_features, valid_indices = [], []
    with torch.no_grad():
        for idx, (row_idx, row) in enumerate(df.iterrows()):
            img_path = str(row["Image path"]).strip().replace("\\", "/")
            if not os.path.exists(img_path):
                basename = os.path.basename(img_path)
                img_path = f"./{basename}" if os.path.exists(f"./{basename}") else None
                if img_path is None:
                    continue

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = img_transform(img).unsqueeze(0).to(device)
                feat = model(img_tensor).cpu().numpy().flatten()
                all_features.append(feat)
                valid_indices.append(row_idx)
            except Exception:
                continue

            if (idx + 1) % 100 == 0:
                print(f"  🔄 已处理 {idx + 1}/{len(df)}")

    img_features = np.array(all_features)
    print(f"  ✅ 成功提取 {len(img_features)} 个样本，维度={img_features.shape[1]}")
    return img_features, valid_indices


# ==================== 3. 表格特征提取 ====================
def extract_tabular_features(df, valid_indices):
    print("\n" + "=" * 70)
    print("📊 Step 3: 表格特征提取与标准化")
    print("=" * 70)

    df_valid = df.loc[valid_indices].copy()
    tab_features = df_valid[TABULAR_FEATURES].values.astype(np.float32)
    tab_features = np.nan_to_num(tab_features, nan=0.0)

    scaler = StandardScaler()
    tab_features_scaled = scaler.fit_transform(tab_features)

    print(f"  ✅ 表格特征维度: {tab_features_scaled.shape[1]}")
    print(f"  ✅ 样本数: {tab_features_scaled.shape[0]}")
    return tab_features_scaled, df_valid["Disease_Label"].values, df_valid["Patient ID"].values


# ==================== 4. 融合方案 ====================
def fuse_concat(img_features, tab_features, img_pca_dim=20):
    print("\n" + "=" * 70)
    print("🔗 Step 4a: 主线融合 —— PCA降维 + Concatenation")
    print("=" * 70)

    img_scaler = StandardScaler()
    img_scaled = img_scaler.fit_transform(img_features)
    pca = PCA(n_components=img_pca_dim)
    img_pca = pca.fit_transform(img_scaled)
    print(f"  ✅ 图像 PCA 降维至 {img_pca_dim} 维，解释方差: {pca.explained_variance_ratio_.sum():.2%}")

    fused = np.hstack([img_pca, tab_features])
    fused_scaler = StandardScaler()
    return fused_scaler.fit_transform(fused), img_pca

def fuse_cca(img_features, tab_features, n_components=5):
    print("\n" + "=" * 70)
    print("🔗 Step 4b: CCA融合")
    print("=" * 70)

    img_scaler = StandardScaler()
    img_scaled = img_scaler.fit_transform(img_features)
    pre_pca_dim = min(50, img_features.shape[0] - 1, img_features.shape[1])
    img_reduced = PCA(n_components=pre_pca_dim).fit_transform(img_scaled)

    n_comp = min(n_components, tab_features.shape[1], img_reduced.shape[1])
    cca = CCA(n_components=n_comp)
    img_cca, tab_cca = cca.fit_transform(img_reduced, tab_features)
    fused_cca = np.hstack([img_cca, tab_cca])

    correlations = [np.corrcoef(img_cca[:, i], tab_cca[:, i])[0, 1] for i in range(n_comp)]
    print(f"  ✅ CCA 各维度相关系数: {[f'{c:.3f}' for c in correlations]}")

    return StandardScaler().fit_transform(fused_cca), correlations

def fuse_snf(img_features, tab_features, labels, k=5, mu=0.5):
    if not HAS_SNF:
        print("\n⚠️ snfpy 未安装，跳过SNF融合"); return None
    print("\n" + "=" * 70)
    print("🔗 Step 4c: SNF融合")
    print("=" * 70)

    img_scaler = StandardScaler()
    img_scaled = img_scaler.fit_transform(img_features)
    img_reduced = PCA(n_components=min(50, img_features.shape[0] - 1)).fit_transform(img_scaled)

    k_actual = min(k, len(labels) - 1)
    affinity_networks = snf.make_affinity([img_reduced, tab_features], metric="euclidean", K=k_actual, mu=mu)
    return snf.snf(affinity_networks, K=k_actual)


# ==================== 5. 降维与聚类 ====================
def reduce_to_2d(features, method="umap", **kwargs):
    if method == "umap" and HAS_UMAP:
        n_neigh = min(kwargs.get("n_neighbors", 15), len(features) - 1)
        return umap.UMAP(n_components=2, n_neighbors=n_neigh, min_dist=kwargs.get("min_dist", 0.3),
                         metric=kwargs.get("metric", "euclidean"), random_state=42).fit_transform(features), "UMAP"
    return PCA(n_components=2).fit_transform(features), "PCA"

def find_optimal_clusters(features, n_clusters_range, method="agglomerative"):
    results = []
    for n_c in n_clusters_range:
        if n_c >= len(features): continue
        clusterer = {"agglomerative": AgglomerativeClustering(n_clusters=n_c),
                     "kmeans": KMeans(n_clusters=n_c, random_state=42, n_init=10),
                     "spectral": SpectralClustering(n_clusters=n_c, random_state=42, affinity="nearest_neighbors")}.get(method, AgglomerativeClustering(n_clusters=n_c))

        cluster_labels = clusterer.fit_predict(features)
        if len(np.unique(cluster_labels)) < 2: continue

        results.append({
            "n_clusters": n_c, "silhouette": silhouette_score(features, cluster_labels),
            "calinski_harabasz": calinski_harabasz_score(features, cluster_labels),
            "davies_bouldin": davies_bouldin_score(features, cluster_labels),
            "labels": cluster_labels
        })
    return results


# ==================== 6. 消融实验 ====================
def run_ablation(img_features, tab_features, fused_features, labels, save_dir):
    print("\n" + "=" * 70)
    print("🧪 Step 6: 消融实验")
    print("=" * 70)

    img_pca_abl = PCA(n_components=min(CONFIG["img_pca_dim"], img_features.shape[0]-1)).fit_transform(StandardScaler().fit_transform(img_features))
    feature_sets = {"Image Only": img_pca_abl, "Tabular Only": tab_features, "Fused (Ours)": fused_features}
    known_mask = np.isin(labels, ["NORMAL", "HAPC"])
    ablation_results = []

    for name, feats in feature_sets.items():
        print(f"\n  --- {name} ---")
        if np.sum(known_mask) >= 4:
            feats_known, labels_known = feats[known_mask], labels[known_mask]
            le = LabelEncoder(); labels_enc = le.fit_transform(labels_known)
            if len(np.unique(labels_enc)) == 2:
                sil = silhouette_score(feats_known, labels_enc)
                ch = calinski_harabasz_score(feats_known, labels_enc)
                db = davies_bouldin_score(feats_known, labels_enc)
                print(f"    Silhouette: {sil:.4f} | CH: {ch:.2f} | DB: {db:.4f}")
                ablation_results.append({"Feature Set": name, "Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db})

        cr = find_optimal_clusters(feats, CONFIG["n_clusters_range"], CONFIG["clustering_method"])
        if cr: print(f"    最优聚类数: {max(cr, key=lambda x: x['silhouette'])['n_clusters']}")

    if ablation_results:
        pd.DataFrame(ablation_results).to_excel(os.path.join(save_dir, "ablation_results.xlsx"), index=False)
    return ablation_results


# ==================== 7. 可视化函数 ====================
def plot_disease_scatter(feats_2d, labels, title, save_path, method_name="UMAP"):
    fig, ax = plt.subplots(figsize=(10, 8)); ax.set_facecolor("#FAFAFA")
    for cls in ["NORMAL", "HAPC", "unknown"]:
        mask = (labels == cls)
        if np.sum(mask) == 0: continue
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], s=CONFIG["scatter_size"], c=DISEASE_COLORS.get(cls, "#999"),
                   alpha=0.7, edgecolors="white", linewidths=0.3, label=cls)
    ax.set_xlabel(f"{method_name} 1"); ax.set_ylabel(f"{method_name} 2"); ax.set_title(title)
    ax.legend(fontsize=12); ax.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ 保存: {save_path}")

def plot_patient_scatter(feats_2d, patient_ids, labels, title, save_path, method_name="UMAP"):
    """绘制按 Patient ID 着色的标准散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#FAFAFA")

    unique_pids = sorted(set(patient_ids))
    for pid in unique_pids:
        mask = (patient_ids == pid)
        try:
            pid_int = int(float(pid))
            color = PATIENT_COLORS.get(pid_int, "#999999")
        except:
            color = "#999999"

        # 统一使用圆形散点，仅按患者ID着色，提升视觉清晰度
        ax.scatter(
            feats_2d[mask, 0], feats_2d[mask, 1],
            s=CONFIG["scatter_size"], c=color, marker="o",
            alpha=0.85, edgecolors="white", linewidths=0.6,
            label=f"P{pid}"
        )

    ax.set_xlabel(f"{method_name} 1", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{method_name} 2", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # 智能图例：自动分列，避免遮挡
    handles, leg_labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ncol = min(4, len(handles))  # 最多4列
        ax.legend(handles, leg_labels, fontsize=10, loc="best", ncol=ncol,
                  frameon=True, edgecolor="lightgray", facecolor="white", framealpha=0.85)

    ax.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: {save_path}")

def plot_cluster_scatter(feats_2d, cluster_labels, title, save_path, method_name="UMAP"):
    fig, ax = plt.subplots(figsize=(10, 8)); ax.set_facecolor("#FAFAFA")
    for cl in sorted(set(cluster_labels)):
        mask = (cluster_labels == cl)
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], s=CONFIG["scatter_size"], c=CLUSTER_COLORS[cl % len(CLUSTER_COLORS)],
                   alpha=0.7, edgecolors="white", linewidths=0.3, label=f"Cluster {cl}")
    ax.set_xlabel(f"{method_name} 1"); ax.set_ylabel(f"{method_name} 2"); ax.set_title(title)
    ax.legend(fontsize=12); ax.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ 保存: {save_path}")

# 🔴 新增：严重程度亚类散点图
def plot_severity_scatter(feats_2d, severity_labels, title, save_path, method_name="UMAP"):
    fig, ax = plt.subplots(figsize=(10, 8)); ax.set_facecolor("#FAFAFA")
    draw_order = ["NORMAL", "Mild", "Moderate", "Severe", "unknown"]
    for cls in draw_order:
        mask = (severity_labels == cls)
        if np.sum(mask) == 0: continue
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], s=CONFIG["scatter_size"], c=SEVERITY_COLORS.get(cls, "#999"),
                   alpha=0.8, edgecolors="white", linewidths=0.4, label=cls, marker="o" if cls != "unknown" else "^")
    ax.set_xlabel(f"{method_name} 1"); ax.set_ylabel(f"{method_name} 2"); ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ordered = [(l, h) for l, h in zip(draw_order, handles) if l in labels]
    if ordered: ax.legend([h for _, h in ordered], [l for l, _ in ordered], fontsize=11, loc="best")
    ax.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ 保存: {save_path}")


def plot_ablation_comparison(feats_dict, severity_labels, save_dir, method_name="UMAP"):
    """消融实验对比图（3合1，带严重程度亚类）"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    draw_order = ["NORMAL", "Mild", "Moderate", "Severe", "unknown"]  # 控制绘制与图例顺序

    for ax, (name, feats) in zip(axes, feats_dict.items()):
        feats_2d, vis_method = reduce_to_2d(feats, method="umap" if HAS_UMAP else "pca",
                                            n_neighbors=CONFIG["umap_n_neighbors"],
                                            min_dist=CONFIG["umap_min_dist"])
        ax.set_facecolor("#FAFAFA")

        # 按顺序绘制亚型，避免遮挡
        for cls in draw_order:
            mask = (severity_labels == cls)
            if np.sum(mask) == 0: continue
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], s=45, c=SEVERITY_COLORS.get(cls, "#999"),
                       alpha=0.75, edgecolors="white", linewidths=0.4, label=cls)

        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.set_xlabel(f"{vis_method} 1", fontsize=12)
        ax.set_ylabel(f"{vis_method} 2", fontsize=12)

        # 固定图例顺序
        handles, legend_labels = ax.get_legend_handles_labels()
        ordered_h = [h for l, h in zip(draw_order, handles) if l in legend_labels]
        ordered_l = [l for l in draw_order if l in legend_labels]
        if ordered_h:
            ax.legend(ordered_h, ordered_l, fontsize=11, loc="best", frameon=True)

        ax.grid(True, linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_comparison.png"), dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 消融对比图（含亚型）已保存: ablation_comparison.png")
def plot_cluster_metrics(cluster_results_list, feature_names, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, mname in zip(axes, ["silhouette", "calinski_harabasz", "davies_bouldin"],
                                 ["Silhouette Score ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓"]):
        for feat_name, results in zip(feature_names, cluster_results_list):
            if not results: continue
            ax.plot([r["n_clusters"] for r in results], [r[metric] for r in results], "o-", label=feat_name, linewidth=2)
        ax.set_xlabel("Number of Clusters"); ax.set_ylabel(mname); ax.set_title(mname); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "cluster_metrics_comparison.png"), dpi=CONFIG["figure_dpi"], bbox_inches="tight"); plt.close(fig)

def plot_robustness_comparison(coords_dict, labels, save_dir):
    n = len(coords_dict); fig, axes = plt.subplots(1, n, figsize=(8*n, 7))
    if n == 1: axes = [axes]
    for ax, (mn, (f2d, vn)) in zip(axes, coords_dict.items()):
        ax.set_facecolor("#FAFAFA")
        for cls in ["NORMAL", "HAPC", "unknown"]:
            mask = (labels == cls)
            if np.sum(mask) == 0: continue
            ax.scatter(f2d[mask, 0], f2d[mask, 1], s=50, c=DISEASE_COLORS.get(cls, "#999"),
                       alpha=0.7, edgecolors="white", linewidths=0.3, label=cls)
        # 🔴 已修复此处缺失的引号
        ax.set_title(mn); ax.set_xlabel(f"{vn} 1"); ax.set_ylabel(f"{vn} 2"); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "robustness_fusion_comparison.png"), dpi=CONFIG["figure_dpi"], bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ 融合方法robustness对比图已保存")

# ==================== 8. 统计检验 ====================
def cluster_clinical_correlation(tab_features, cluster_labels, feature_names, save_dir):
    print("\n" + "=" * 70); print("📈 Step 7: 聚类亚型与临床指标的统计关联分析"); print("=" * 70)
    unique_clusters = sorted(set(cluster_labels)); n_c = len(unique_clusters); results = []
    for i, feat in enumerate(feature_names):
        groups = [tab_features[cluster_labels == cl, i] for cl in unique_clusters]
        stat, pval = mannwhitneyu(groups[0], groups[1]) if n_c == 2 else kruskal(*groups)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        results.append({"Feature": feat, "P-value": pval, "Significance": sig,
                        **{f"Cluster{cl}_mean": np.mean(tab_features[cluster_labels == cl, i]) for cl in unique_clusters}})
        print(f"  {feat:<28} p={pval:.4e} {sig}")
    pd.DataFrame(results).to_excel(os.path.join(save_dir, "cluster_clinical_correlation.xlsx"), index=False)
    return results


# ==================== 9. 主流程 ====================
def main():
    save_dir = CONFIG["output_dir"]; os.makedirs(save_dir, exist_ok=True)

    df = load_data(CONFIG["data_path"])
    img_features, valid_indices = extract_image_features(df, CONFIG["pretrained_model"], CONFIG["img_size"])
    if len(img_features) == 0: print("❌ 无有效图像特征"); return

    tab_features, labels, patient_ids = extract_tabular_features(df, valid_indices)
    fused_concat, _ = fuse_concat(img_features, tab_features, CONFIG["img_pca_dim"])
    fused_cca, _ = fuse_cca(img_features, tab_features, CONFIG["cca_n_components"])
    fused_snf = fuse_snf(img_features, tab_features, labels, CONFIG["snf_k"], CONFIG["snf_mu"])

    print("\n" + "=" * 70); print("🗺️  Step 5: 降维可视化"); print("=" * 70)
    coords_concat, vis_m = reduce_to_2d(fused_concat, method="umap" if HAS_UMAP else "pca", n_neighbors=CONFIG["umap_n_neighbors"], min_dist=CONFIG["umap_min_dist"])
    coords_cca, vis_c = reduce_to_2d(fused_cca, method="umap" if HAS_UMAP else "pca", n_neighbors=CONFIG["umap_n_neighbors"], min_dist=CONFIG["umap_min_dist"])

    # 对齐 Severity_Label
    severity_labels = df.loc[valid_indices, "Severity_Label"].values

    plot_disease_scatter(coords_concat, labels, "Multimodal Fusion: Disease Separation", os.path.join(save_dir, "main_disease_scatter.png"), vis_m)
    plot_patient_scatter(coords_concat, patient_ids, labels, "Multimodal Fusion: Patient Clustering", os.path.join(save_dir, "main_patient_scatter.png"), vis_m)
    plot_severity_scatter(coords_concat, severity_labels, "Multimodal Fusion: HAPC Severity Subtypes", os.path.join(save_dir, "main_severity_scatter.png"), vis_m)

    plot_disease_scatter(coords_cca, labels, "CCA Fusion: Disease Separation", os.path.join(save_dir, "cca_disease_scatter.png"), vis_c)
    plot_severity_scatter(coords_cca, severity_labels, "CCA Fusion: HAPC Severity Subtypes", os.path.join(save_dir, "cca_severity_scatter.png"), vis_c)

    if fused_snf is not None:
        from sklearn.manifold import SpectralEmbedding
        coords_snf = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=42).fit_transform(fused_snf)
        plot_disease_scatter(coords_snf, labels, "SNF Fusion: Disease Separation", os.path.join(save_dir, "snf_disease_scatter.png"), "SE")

    plot_robustness_comparison({"Concat+UMAP": (coords_concat, vis_m), "CCA": (coords_cca, vis_c)}, labels, save_dir)

    # 消融实验
    plot_ablation_comparison({"Image Only": PCA(n_components=CONFIG["img_pca_dim"]).fit_transform(
        StandardScaler().fit_transform(img_features)),
                              "Tabular Only": tab_features, "Fused (Ours)": fused_concat}, severity_labels, save_dir)
    run_ablation(img_features, tab_features, fused_concat, labels, save_dir)

    cr_list, fn_list = [], []
    for name, feats in {"Image Only": PCA(n_components=CONFIG["img_pca_dim"]).fit_transform(StandardScaler().fit_transform(img_features)),
                        "Tabular Only": tab_features, "Fused (Ours)": fused_concat}.items():
        cr_list.append(find_optimal_clusters(feats, CONFIG["n_clusters_range"], CONFIG["clustering_method"]))
        fn_list.append(name)
    plot_cluster_metrics(cr_list, fn_list, save_dir)

    # 最优聚类 & 临床关联
    search = find_optimal_clusters(fused_concat, CONFIG["n_clusters_range"], CONFIG["clustering_method"])
    if search:
        best = max(search, key=lambda x: x["silhouette"])
        print(f"\n  ✅ 最优聚类数: {best['n_clusters']} (Silhouette={best['silhouette']:.4f})")
        plot_cluster_scatter(coords_concat, best["labels"], f"Optimal Clustering (k={best['n_clusters']})", os.path.join(save_dir, "main_cluster_scatter.png"), vis_m)
        cluster_clinical_correlation(tab_features, best["labels"], TABULAR_FEATURES, save_dir)

        res_df = pd.DataFrame({"Patient_ID": patient_ids, "Disease_Label": labels, "Severity_Label": severity_labels,
                               "Cluster": best["labels"], f"{vis_m}_1": coords_concat[:, 0], f"{vis_m}_2": coords_concat[:, 1]})
        for i, f in enumerate(TABULAR_FEATURES): res_df[f] = tab_features[:, i]
        res_df.to_excel(os.path.join(save_dir, "full_results.xlsx"), index=False)

    print("\n" + "=" * 70); print("🎉 全部分析完成！"); print("=" * 70)
    print(f"📁 结果目录: {os.path.abspath(save_dir)}/")


if __name__ == "__main__":
    main()