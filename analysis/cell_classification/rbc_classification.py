import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import patheffects
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline
import os, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
EXCEL_PATH = r"C:\Users\Administrator\Desktop\RBC24.xlsx"
OUTPUT_DIR = r"E:\cell_classification_results2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    'volume', 'HGB Mean', 'Mean RMS displacement', 'Max RMS displacement',
    'TransitionTime', 'Area', 'AvgVelocity(pixel/frame)',
    'MaxVelocity(pixel/frame)', 'MinVelocity(pixel/frame)', 'Average_DI',
]
FEAT6 = ['Average_DI', 'HGB Mean', 'Mean RMS displacement',
         'TransitionTime', 'AvgVelocity(pixel/frame)', 'volume']
FEAT_DISP = {
    'Average_DI': 'DI', 'HGB Mean': 'HGB', 'Mean RMS displacement': 'RMS Disp.',
    'TransitionTime': 'Trans. Time', 'AvgVelocity(pixel/frame)': 'Velocity', 'volume': 'Volume',
}

# ── Patient ordering: Normal → Mild → Moderate → Severe ──
ALL_IDS = [4, 5, 6, 13, 14, 1, 9, 11, 12, 7, 8, 10, 3, 2]

PATIENT_META = {
    4:  dict(label='P4',  group='Normal',   desc='Healthy',       short='Normal',   order=0),
    5:  dict(label='P5',  group='Normal',   desc='Healthy',       short='Normal',   order=1),
    6:  dict(label='P6',  group='Normal',   desc='Healthy',       short='Normal',   order=2),
    13: dict(label='P13', group='Normal', desc='Healthy', short='Normal', order=3),
    14: dict(label='P14', group='Normal', desc='Healthy', short='Normal', order=4),
    1:  dict(label='P1',  group='Mild',     desc='HAPC-Mild',     short='Mild',     order=5),
    9:  dict(label='P9',  group='Mild',     desc='HAPC-Mild',     short='Mild',     order=6),
    11: dict(label='P11', group='Mild',     desc='HAPC-Mild',     short='Mild',     order=7),
    12: dict(label='P12', group='Mild', desc='HAPC-Mild', short='Mild', order=8),
    7:  dict(label='P7',  group='Moderate', desc='HAPC-Moderate', short='Moderate', order=9),
    8:  dict(label='P8',  group='Moderate', desc='HAPC-Moderate', short='Moderate', order=10),
    10: dict(label='P10', group='Moderate', desc='HAPC-Moderate', short='Moderate', order=11),
    3:  dict(label='P3',  group='Severe',   desc='HAPC-Severe',   short='Severe',   order=12),
    2:  dict(label='P2',  group='Severe',   desc='HAPC-Severe',   short='Severe',   order=13),
}

# Disease label → numeric for LDA training
DISEASE_MAP = {'normal': 0, 'hapc-mild': 1, 'hapc-Moderate': 2, 'hapc-sev': 3}

GROUP_PAL = {
    'Normal': '#4A90D9', 'Mild': '#5EC269', 'Moderate': '#F5A623', 'Severe': '#D94040',
}

# 4 cell types — clinically anchored
best_k = 4
TYPE_NAMES = {0: 'Normal-like', 1: 'Mildly Abnormal', 2: 'Moderately Abnormal', 3: 'Severely Abnormal'}
CPAL = ['#4A90D9', '#5EC269', '#F5A623', '#D94040']

P_COLOR = {
    4: '#3A7BC8', 5: '#6DAEE0', 6: '#4A90D9',
    13: '#4A90D9', 14: '#6DAEE0',  # 新增：正常组患者13、14的颜色
    1: '#5EC269', 9: '#45B854', 11: '#73D97E', 12: '#5EC269',  # 新增：轻症患者12
    7: '#F5A623', 8: '#E8963E', 10: '#FFBF47',
    3: '#D85F9A', 2: '#D94040',
}

BG = '#FAFBFE'; BG2 = '#F0F2F8'; GRID_C = '#E8EAF0'
TK = '#1E293B'; TM = '#475569'; TL = '#94A3B8'

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'axes.unicode_minus': False, 'figure.dpi': 150})

# ══════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ══════════════════════════════════════════════════════
print("📖 Loading data …")
df = pd.read_excel(EXCEL_PATH)
id_col = 'Patient ID'
df_all = df[df[id_col].isin(ALL_IDS)].dropna(subset=FEATURES).copy()

scaler = StandardScaler()
X = scaler.fit_transform(df_all[FEATURES].values)

# PCA for visualization
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
df_all['pc1'] = Xp[:, 0]
df_all['pc2'] = Xp[:, 1]
print(f"   {len(df_all)} cells, {len(FEATURES)} features")
print(f"   PCA var explained: {pca.explained_variance_ratio_[:2]*100}")

# ══════════════════════════════════════════════════════
# 2. LDA-BASED CELL TYPE CLASSIFICATION
#    Supervised: trained on Disease labels from clinical diagnosis
#    Each cell is classified as Normal-like / Mildly / Moderately / Severely Abnormal
# ══════════════════════════════════════════════════════
print("\n🔬 LDA Cell Classification (clinically supervised) …")
y_disease = df_all['Disease'].map(DISEASE_MAP).values

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y_disease)
df_all['ld1'] = X_lda[:, 0]
df_all['ld2'] = X_lda[:, 1]
df_all['ld3'] = X_lda[:, 2]

# Cell type = LDA predicted class
df_all['ctype'] = lda.predict(X)
lda_probs = lda.predict_proba(X)
df_all['prob_normal']   = lda_probs[:, 0]
df_all['prob_mild']     = lda_probs[:, 1]
df_all['prob_moderate'] = lda_probs[:, 2]
df_all['prob_severe']   = lda_probs[:, 3]

print(f"   LDA explained variance: {lda.explained_variance_ratio_*100}")
print(f"   Cell type counts:")
for ct in range(best_k):
    print(f"     {TYPE_NAMES[ct]:25s}: {(df_all['ctype']==ct).sum()}")

# ── proportions per patient ──
prop = {}
for pid in ALL_IDS:
    sub = df_all[df_all[id_col] == pid]
    t = len(sub)
    prop[pid] = [(sub['ctype'] == ct).sum() / t * 100 for ct in range(best_k)]

print("\n   Per-patient cell type composition:")
for pid in ALL_IDS:
    meta = PATIENT_META[pid]
    parts = "  ".join(f"{TYPE_NAMES[ct]}={prop[pid][ct]:.1f}%" for ct in range(best_k) if prop[pid][ct] > 0)
    print(f"   {meta['label']:3s} ({meta['short']:10s}): {parts}")

# ══════════════════════════════════════════════════════
# 3. CONTINUOUS PATHOLOGICAL INDEX
#    Weighted sum of LDA posterior probabilities
#    PI = 0*P(Normal) + 33*P(Mild) + 67*P(Moderate) + 100*P(Severe)
# ══════════════════════════════════════════════════════
print("\n📐 Computing Pathological Index (probability-weighted) …")
severity_weights = np.array([0, 33.3, 66.7, 100.0])
df_all['path_idx'] = (lda_probs * severity_weights).sum(axis=1)

print("   Pathological Index per patient:")
for pid in ALL_IDS:
    sub = df_all[df_all[id_col] == pid]
    meta = PATIENT_META[pid]
    print(f"   {meta['label']:3s} ({meta['short']:10s}): "
          f"mean={sub['path_idx'].mean():.1f}  std={sub['path_idx'].std():.1f}  "
          f"range=[{sub['path_idx'].min():.1f}, {sub['path_idx'].max():.1f}]")

# ══════════════════════════════════════════════════════
# 4. SVM DECISION BOUNDARY (on PCA space for plotting)
# ══════════════════════════════════════════════════════
print("\n🖊️ Training SVM for PCA decision boundaries …")
svc = SVC(kernel='rbf', C=5, gamma='scale', probability=True)
svc.fit(Xp, y_disease)

margin = 1.5
x_min, x_max = Xp[:, 0].min() - margin, Xp[:, 0].max() + margin
y_min, y_max = Xp[:, 1].min() - margin, Xp[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

BG_COLORS_4 = ['#D6EAFF', '#D6F5DD', '#FFF3D6', '#FFE0E0']

# ══════════════════════════════════════════════════════
# PLOT 1: ★ PCA + decision boundary — by patient & by cell type
# ══════════════════════════════════════════════════════
print("\n🎨 Plotting …")

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Multi-feature PCA with Clinically-Supervised Disease Stage Boundaries",
             fontsize=20, fontweight='bold', color=TK, y=.97)

for ax_idx, (color_by, title) in enumerate([('patient', 'Colored by Patient'), ('ctype', 'Colored by Cell Type (LDA)')]):
    ax = axes[ax_idx]
    ax.set_facecolor('white')

    # background
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=BG_COLORS_4, alpha=.35)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5],
               colors=['#4A90D9', '#F5A623', '#D94040'], linewidths=1.5, linestyles='--', alpha=.5)

    # region labels
    region_labels = [('Normal\nRegion',   x_min + 1.5, y_max - 1.5, '#4A90D9'),
                     ('Mild\nRegion',     (x_min + x_max) / 2 - 2.5, (y_min + y_max) / 2 + 1.5, '#5EC269'),
                     ('Moderate\nRegion',  (x_min + x_max) / 2 + 2.5, (y_min + y_max) / 2 - 1.5, '#F5A623'),
                     ('Severe\nRegion',    x_max - 2.5, y_min + 1.5, '#D94040')]
    for txt, rx, ry, rc in region_labels:
        ax.text(rx, ry, txt, fontsize=11, fontweight='bold', color=rc, alpha=.45,
                ha='center', va='center', fontstyle='italic')

    if color_by == 'patient':
        for pid in ALL_IDS:
            m = df_all[id_col] == pid
            meta = PATIENT_META[pid]
            ax.scatter(Xp[m.values, 0], Xp[m.values, 1], c=P_COLOR[pid], s=35, alpha=.7,
                       label=f"{meta['label']} ({meta['short']})", edgecolors='white',
                       linewidths=.3, zorder=3)
    else:
        for ct in range(best_k):
            m = (df_all['ctype'] == ct).values
            ax.scatter(Xp[m, 0], Xp[m, 1], c=CPAL[ct], s=35, alpha=.7,
                       label=TYPE_NAMES[ct], edgecolors='white', linewidths=.3, zorder=3)

    ax.set_xlabel(f"PCA Component 1  ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=13, color=TM)
    ax.set_ylabel(f"PCA Component 2  ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=13, color=TM)
    ax.set_title(title, fontsize=16, fontweight='bold', color=TK, pad=12)
    leg = ax.legend(fontsize=9, framealpha=.92, edgecolor=GRID_C, loc='best')
    leg.get_frame().set_facecolor('white')
    ax.grid(True, color=GRID_C, ls='--', alpha=.4)
    ax.tick_params(colors=TM, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)

plt.tight_layout(rect=[0, 0, 1, .95])
plt.savefig(os.path.join(OUTPUT_DIR, "1_PCA_decision_boundary.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 1")

# ══════════════════════════════════════════════════════
# PLOT 2: LDA space — the discriminant's own view
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("LDA Discriminant Space — Maximum Clinical Separation",
             fontsize=20, fontweight='bold', color=TK, y=.97)

# Left: LD1 vs LD2 colored by patient
ax = axes[0]; ax.set_facecolor('white')
for pid in ALL_IDS:
    m = (df_all[id_col] == pid).values
    meta = PATIENT_META[pid]
    ax.scatter(X_lda[m, 0], X_lda[m, 1], c=P_COLOR[pid], s=35, alpha=.65,
               label=f"{meta['label']} ({meta['short']})", edgecolors='white', linewidths=.3, zorder=3)
ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=13, color=TM)
ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=13, color=TM)
ax.set_title("Colored by Patient", fontsize=16, fontweight='bold', color=TK, pad=12)
leg = ax.legend(fontsize=9, framealpha=.92, edgecolor=GRID_C, loc='best')
leg.get_frame().set_facecolor('white')
ax.grid(True, color=GRID_C, ls='--', alpha=.4)
for sp in ax.spines.values(): sp.set_color(GRID_C)
ax.tick_params(colors=TM, labelsize=10)

# Right: LD1 vs LD2 colored by cell type
ax = axes[1]; ax.set_facecolor('white')
for ct in range(best_k):
    m = (df_all['ctype'] == ct).values
    ax.scatter(X_lda[m, 0], X_lda[m, 1], c=CPAL[ct], s=35, alpha=.65,
               label=TYPE_NAMES[ct], edgecolors='white', linewidths=.3, zorder=3)
# group centroids
for ct in range(best_k):
    m = (df_all['ctype'] == ct).values
    cx, cy = X_lda[m, 0].mean(), X_lda[m, 1].mean()
    ax.scatter(cx, cy, c=CPAL[ct], s=250, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
    ax.annotate(TYPE_NAMES[ct].split()[0], xy=(cx, cy), fontsize=10, fontweight='bold',
                color=CPAL[ct], ha='center', va='bottom',
                xytext=(0, 12), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=.2', fc='white', ec=CPAL[ct], alpha=.85))
ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=13, color=TM)
ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=13, color=TM)
ax.set_title("Colored by Cell Type (LDA-classified)", fontsize=16, fontweight='bold', color=TK, pad=12)
leg = ax.legend(fontsize=10, framealpha=.92, edgecolor=GRID_C, loc='best')
leg.get_frame().set_facecolor('white')
ax.grid(True, color=GRID_C, ls='--', alpha=.4)
for sp in ax.spines.values(): sp.set_color(GRID_C)
ax.tick_params(colors=TM, labelsize=10)

plt.tight_layout(rect=[0, 0, 1, .95])
plt.savefig(os.path.join(OUTPUT_DIR, "2_LDA_space.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 2")

# ══════════════════════════════════════════════════════
# PLOT 3: ★ Severe focus — PCA with Severe highlighted + pathological gradient
# ══════════════════════════════════════════════════════
SEVERE_IDS = [2, 3]
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Severe Patients (P2, P3): Heterogeneous Position in Disease Spectrum",
             fontsize=20, fontweight='bold', color=TK, y=.97)

ax = axes[0]; ax.set_facecolor('white')
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=BG_COLORS_4, alpha=.3)
ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5],
           colors=['#4A90D9', '#F5A623', '#D94040'], linewidths=1.5, linestyles='--', alpha=.5)
others = ~df_all[id_col].isin(SEVERE_IDS)
ax.scatter(Xp[others.values, 0], Xp[others.values, 1], c='#CBD5E1', s=15, alpha=.3, edgecolors='none', zorder=2)
sev_m = df_all[id_col].isin(SEVERE_IDS)
sc = ax.scatter(Xp[sev_m.values, 0], Xp[sev_m.values, 1], c=df_all.loc[sev_m, 'path_idx'],
                cmap='RdYlGn_r', s=55, alpha=.85, edgecolors='white', linewidths=.5, zorder=4,
                vmin=0, vmax=100)
cbar = plt.colorbar(sc, ax=ax, fraction=.03, pad=.02, shrink=.7)
cbar.set_label('Pathological Index', fontsize=11, color=TM)
cbar.ax.tick_params(labelsize=9, colors=TM)
for pid in ALL_IDS:
    sub = df_all[df_all[id_col] == pid]
    cx, cy = sub['pc1'].mean(), sub['pc2'].mean()
    meta = PATIENT_META[pid]
    if pid in SEVERE_IDS:
        ax.annotate(f"{meta['label']}\n({meta['short']})", xy=(cx, cy), fontsize=11, fontweight='bold',
                    color=P_COLOR[pid], ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=.3', fc='white', ec=P_COLOR[pid], alpha=.9),
                    xytext=(cx, cy + .8))
    else:
        ax.annotate(meta['label'], xy=(cx, cy), fontsize=9, fontweight='bold',
                    color=P_COLOR[pid], ha='center', va='center', alpha=.7)
ax.set_xlabel("PCA Component 1", fontsize=12, color=TM)
ax.set_ylabel("PCA Component 2", fontsize=12, color=TM)
ax.set_title("Severe patients colored by Pathological Index", fontsize=15, fontweight='bold', color=TK, pad=10)
ax.grid(True, color=GRID_C, ls='--', alpha=.3)
for sp in ax.spines.values(): sp.set_color(GRID_C)
ax.tick_params(colors=TM, labelsize=9)

# Right: violin+box of Pathological Index
ax = axes[1]; ax.set_facecolor(BG2)
patient_data = [df_all[df_all[id_col] == pid]['path_idx'].values for pid in ALL_IDS]
patient_colors_plot = [P_COLOR[pid] for pid in ALL_IDS]
vparts = ax.violinplot(patient_data, positions=range(len(ALL_IDS)),
                       showmeans=False, showmedians=False, showextrema=False)
for i, body in enumerate(vparts['bodies']):
    body.set_facecolor(patient_colors_plot[i]); body.set_edgecolor(patient_colors_plot[i]); body.set_alpha(.4)
bp = ax.boxplot(patient_data, positions=range(len(ALL_IDS)), widths=.15,
                patch_artist=True, showfliers=False, zorder=3)
for i, (box, med) in enumerate(zip(bp['boxes'], bp['medians'])):
    box.set_facecolor(patient_colors_plot[i]); box.set_edgecolor('white'); box.set_alpha(.85); box.set_linewidth(1.2)
    med.set_color('white'); med.set_linewidth(2)
for w in bp['whiskers'] + bp['caps']:
    w.set_color(TL); w.set_linewidth(1)
for i, (d, c) in enumerate(zip(patient_data, patient_colors_plot)):
    jitter = np.random.default_rng(42).normal(0, .04, len(d))
    ax.scatter(i + jitter, d, s=8, color=c, alpha=.4, edgecolors='none', zorder=2)
ax.set_xticks(range(len(ALL_IDS)))
ax.set_xticklabels([f"{PATIENT_META[p]['label']}\n({PATIENT_META[p]['short']})" for p in ALL_IDS],
                   fontsize=9, color=TK, fontweight='600')
ax.set_ylabel("Pathological Index  (0=Normal → 100=Severe)", fontsize=12, color=TM)
ax.set_title("Per-cell Pathological Index Distribution", fontsize=15, fontweight='bold', color=TK, pad=10)
ax.grid(axis='y', color=GRID_C, ls='--', alpha=.5)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID_C)
ax.tick_params(colors=TM, labelsize=9)
plt.tight_layout(rect=[0, 0, 1, .95])
plt.savefig(os.path.join(OUTPUT_DIR, "3_Severe_focus.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 3")

# ══════════════════════════════════════════════════════
# PLOT 4: Radar fingerprints by cell type
# ══════════════════════════════════════════════════════
cluster_means = df_all.groupby('ctype')[FEAT6].mean()
cm_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-10)
angles = np.linspace(0, 2 * np.pi, len(FEAT6), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
for ct in range(best_k):
    vals = cm_norm.iloc[ct].values.tolist() + [cm_norm.iloc[ct].values[0]]
    ax.fill(angles, vals, color=CPAL[ct], alpha=.12)
    ax.plot(angles, vals, color=CPAL[ct], lw=2.8, marker='o', ms=7,
            mfc='white', mew=2.2, label=TYPE_NAMES[ct], zorder=3)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([FEAT_DISP[f] for f in FEAT6], fontsize=12, fontweight='bold', color=TK)
ax.set_ylim(0, 1.18)
ax.set_title("Cell Type Feature Fingerprints (LDA-classified)", fontsize=17, fontweight='bold', color=TK, pad=25, y=1.05)
leg = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=11,
                framealpha=.9, edgecolor=GRID_C)
leg.get_frame().set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_radar.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 4")

# ══════════════════════════════════════════════════════
# PLOT 5: ★ Disease Spectrum Stacked Bar
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
x = np.arange(len(ALL_IDS))
bottom = np.zeros(len(ALL_IDS))
for ct in range(best_k):
    vals = np.array([prop[pid][ct] for pid in ALL_IDS])
    ax.bar(x, vals, .68, bottom=bottom, color=CPAL[ct], edgecolor='white', lw=.8,
           label=TYPE_NAMES[ct], zorder=3)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:
            ax.text(x[i], b + v / 2, f"{v:.1f}%", ha='center', va='center',
                    fontsize=9 if v > 8 else 7, fontweight='bold', color='white',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#00000033')])
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([PATIENT_META[p]['label'] for p in ALL_IDS], fontsize=12, fontweight='bold', color=TK)
for i, pid in enumerate(ALL_IDS):
    ax.text(i, -5.5, PATIENT_META[pid]['desc'], ha='center', va='top', fontsize=8.5,
            color=P_COLOR[pid], fontstyle='italic')
groups_span = [
    (0, 2, 'Normal', GROUP_PAL['Normal']),
    (3, 5, 'Mild', GROUP_PAL['Mild']),
    (6, 8, 'Moderate', GROUP_PAL['Moderate']),
    (9, 10, 'Severe', GROUP_PAL['Severe']),
]
for s, e, nm, gc in groups_span:
    mid = (s + e) / 2
    ax.annotate(nm, xy=(mid, -14), fontsize=11, fontweight='bold', ha='center', va='top',
                color=gc, annotation_clip=False)
    ax.plot([s - .35, e + .35], [-10, -10], color=gc, lw=4, clip_on=False, solid_capstyle='round', alpha=.85)
ax.annotate('', xy=(len(ALL_IDS) - .4, -24), xytext=(-.4, -24),
            arrowprops=dict(arrowstyle='->', color=TL, lw=2), annotation_clip=False)
ax.text((len(ALL_IDS) - 1) / 2, -27, 'Disease  Progression', ha='center', fontsize=12, color=TL,
        fontstyle='italic', fontweight='600', clip_on=False)
ax.set_ylabel("Proportion  (%)", fontsize=14, color=TM); ax.set_ylim(0, 110)
ax.set_title("Cell Type Composition Across Disease Spectrum (LDA-classified)", fontsize=20, fontweight='bold', color=TK, pad=18)
leg = ax.legend(loc='upper right', fontsize=12, framealpha=.92, edgecolor=GRID_C,
                title='Cell Type', title_fontsize=13)
leg.get_frame().set_facecolor('white')
ax.grid(axis='y', color=GRID_C, ls='--', alpha=.5)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color(GRID_C)
plt.subplots_adjust(bottom=.22)
plt.savefig(os.path.join(OUTPUT_DIR, "5_spectrum_bar.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 5")

# ══════════════════════════════════════════════════════
# PLOT 6: Individual Donut charts (3×4)
# ══════════════════════════════════════════════════════
nrows, ncols = 3, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(22, 15))
fig.patch.set_facecolor(BG)
fig.suptitle("Cell Type Distribution — Individual Patients (LDA)", fontsize=20, fontweight='bold', color=TK, y=.98)
for idx in range(nrows * ncols):
    ax = axes[idx // ncols][idx % ncols]; ax.set_facecolor(BG)
    if idx >= len(ALL_IDS):
        ax.axis('off'); continue
    pid = ALL_IDS[idx]; meta = PATIENT_META[pid]; vals = prop[pid]
    nonzero = [(v, ct) for ct, v in enumerate(vals) if v > 0]
    if not nonzero:
        ax.text(.5, .5, 'No Data', transform=ax.transAxes, ha='center', fontsize=12, color=TL); continue
    sizes = [x[0] for x in nonzero]; colors = [CPAL[x[1]] for x in nonzero]
    labels = [f"{TYPE_NAMES[x[1]]}\n{x[0]:.1f}%" if x[0] > 3 else '' for x in nonzero]
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                           textprops={'fontsize': 8, 'fontweight': 'bold', 'color': TK},
                           wedgeprops={'edgecolor': 'white', 'linewidth': 2.5})
    centre = plt.Circle((0, 0), .52, fc='white', ec=GRID_C, lw=1); ax.add_artist(centre)
    n_cells = len(df_all[df_all[id_col] == pid])
    ax.text(0, .06, meta['label'], ha='center', va='center', fontsize=18, fontweight='bold', color=P_COLOR[pid])
    ax.text(0, -.18, f"n={n_cells}", ha='center', va='center', fontsize=10, color=TL)
    ax.set_title(meta['group'], fontsize=13, fontweight='bold', color=P_COLOR[pid], pad=8)
handles = [mpatches.Patch(fc=CPAL[ct], ec='white', lw=1.5, label=TYPE_NAMES[ct]) for ct in range(best_k)]
fig.legend(handles=handles, loc='lower center', ncol=best_k, fontsize=12, frameon=True,
           edgecolor=GRID_C, facecolor='white', bbox_to_anchor=(.5, .01), title='Cell Types', title_fontsize=13)
plt.subplots_adjust(hspace=.35, wspace=.15, bottom=.08)
plt.savefig(os.path.join(OUTPUT_DIR, "6_individual_donuts.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 6")

# ══════════════════════════════════════════════════════
# PLOT 7: Heatmap Patient × Cell Type
# ══════════════════════════════════════════════════════
prop_mat = np.array([prop[pid] for pid in ALL_IDS])
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor(BG)
cmap = mcolors.LinearSegmentedColormap.from_list('', ['#FFFFFF', '#DBEAFE', '#93C5FD', '#3B82F6', '#1E40AF'])
im = ax.imshow(prop_mat, cmap=cmap, aspect='auto', vmin=0, vmax=100)
for i in range(len(ALL_IDS)):
    for j in range(best_k):
        v = prop_mat[i, j]; tc = 'white' if v > 50 else TK
        ax.text(j, i, f"{v:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold', color=tc)
ax.set_xticks(range(best_k))
ax.set_xticklabels([TYPE_NAMES[ct] for ct in range(best_k)], fontsize=11, fontweight='600', color=TK)
ax.set_yticks(range(len(ALL_IDS)))
ax.set_yticklabels([f"{PATIENT_META[p]['label']} ({PATIENT_META[p]['short']})" for p in ALL_IDS], fontsize=11)
for i, pid in enumerate(ALL_IDS):
    ax.add_patch(plt.Rectangle((-.7, i - .45), .25, .9, fc=P_COLOR[pid], clip_on=False, zorder=5))
ax.set_title("Cell Type Proportion Matrix (LDA-classified)", fontsize=18, fontweight='bold', color=TK, pad=18)
for gb, gc in zip([2.5, 5.5, 8.5], [GROUP_PAL['Mild'], GROUP_PAL['Moderate'], GROUP_PAL['Severe']]):
    ax.axhline(gb, color=gc, lw=2, ls='--', alpha=.6)
cbar = plt.colorbar(im, ax=ax, fraction=.015, pad=.03, shrink=.8)
cbar.set_label("Proportion (%)", fontsize=11, color=TM)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_heatmap.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 7")

# ══════════════════════════════════════════════════════
# PLOT 8: ★ Pathological Score progression (bar)
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)

# Patient-level mean Pathological Index
path_scores = [df_all[df_all[id_col] == pid]['path_idx'].mean() for pid in ALL_IDS]
path_stds   = [df_all[df_all[id_col] == pid]['path_idx'].std()  for pid in ALL_IDS]

xs = np.arange(len(ALL_IDS))
for i, (pid, sc) in enumerate(zip(ALL_IDS, path_scores)):
    ratio = sc / 100.0
    bar_color = mcolors.to_hex([ratio * .93 + (1 - ratio) * .23,
                                 ratio * .25 + (1 - ratio) * .51,
                                 ratio * .25 + (1 - ratio) * .96])
    ax.bar(i, sc, .62, color=bar_color, edgecolor='white', lw=1.8, zorder=3, alpha=.88)
    ax.errorbar(i, sc, yerr=path_stds[i], fmt='none', ecolor=TL, elinewidth=1.5, capsize=5, zorder=4)
    ax.text(i, sc + path_stds[i] + 2, f"{sc:.1f}", ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=TK)

try:
    spl = make_interp_spline(xs, path_scores, k=min(3, len(xs) - 1))
    xf = np.linspace(0, len(ALL_IDS) - 1, 200); yf = np.clip(spl(xf), 0, 100)
    ax.plot(xf, yf, color='#EF4444', lw=2.5, ls='--', alpha=.4, zorder=2)
except:
    pass

ax.set_xticks(xs)
ax.set_xticklabels([f"{PATIENT_META[p]['label']}\n({PATIENT_META[p]['short']})" for p in ALL_IDS],
                   fontsize=10, color=TK, fontweight='600')
ax.set_ylabel('Pathological Index  (0=Normal → 100=Severe)', fontsize=12, color=TM)
ax.set_ylim(0, max(v + s for v, s in zip(path_scores, path_stds)) * 1.2 + 5)
ax.set_title("Disease Progression Index — LDA Probability-weighted Score", fontsize=18, fontweight='bold', color=TK, pad=15)
ybot = -max(path_scores) * .12
ax.annotate('', xy=(len(ALL_IDS) - .3, ybot), xytext=(-.3, ybot),
            arrowprops=dict(arrowstyle='->', color=TL, lw=2), annotation_clip=False)
ax.text((len(ALL_IDS) - 1) / 2, ybot - max(path_scores) * .05, 'Disease Progression',
        ha='center', fontsize=11, color=TL, fontstyle='italic', fontweight='600', clip_on=False)
ax.grid(axis='y', color=GRID_C, ls='--', alpha=.5)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
plt.subplots_adjust(bottom=.2)
plt.savefig(os.path.join(OUTPUT_DIR, "8_pathological_score.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 8")

# ══════════════════════════════════════════════════════
# PLOT 9: ★ Stream / stacked area
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
xs = np.arange(len(ALL_IDS))
bottoms = np.zeros(len(ALL_IDS))
for ct in range(best_k):
    vals = np.array([prop[pid][ct] for pid in ALL_IDS])
    ax.fill_between(xs, bottoms, bottoms + vals, color=CPAL[ct], alpha=.7,
                    label=TYPE_NAMES[ct], edgecolor='white', lw=1.5)
    for i in range(len(xs)):
        v = vals[i]
        if v > 5:
            ax.text(xs[i], bottoms[i] + v / 2, f"{v:.0f}%", ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white', alpha=.9)
    bottoms += vals
ax.set_xticks(xs)
ax.set_xticklabels([f"{PATIENT_META[p]['label']}\n{PATIENT_META[p]['group']}" for p in ALL_IDS],
                   fontsize=10, color=TK, fontweight='600', linespacing=1.3)
ax.set_ylabel("Cumulative Proportion (%)", fontsize=13, color=TM)
ax.set_ylim(0, 100); ax.set_xlim(-.3, len(ALL_IDS) - .7)
ax.set_title("Stream View: Cell Population Shift Along Disease Progression (LDA)",
             fontsize=18, fontweight='bold', color=TK, pad=15)
leg = ax.legend(loc='upper right', fontsize=11, framealpha=.92, edgecolor=GRID_C,
                title='Cell Type', title_fontsize=12)
leg.get_frame().set_facecolor('white')
ax.grid(axis='y', color=GRID_C, ls='--', alpha=.3)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.annotate('', xy=(len(ALL_IDS) - .5, -10), xytext=(-.3, -10),
            arrowprops=dict(arrowstyle='->', color=TL, lw=2), annotation_clip=False)
ax.text((len(ALL_IDS) - 1) / 2, -14, 'Normal → Mild → Moderate → Severe',
        ha='center', fontsize=12, color=TL, fontstyle='italic', fontweight='600', clip_on=False)
plt.subplots_adjust(bottom=.18)
plt.savefig(os.path.join(OUTPUT_DIR, "9_stream.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 9")

# ══════════════════════════════════════════════════════
# PLOT 10: ★ Ridge plot — Per-cell Pathological Index
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 9))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
x_kde = np.linspace(0, 100, 300)
for pid in ALL_IDS:
    sub = df_all[df_all[id_col] == pid]['path_idx'].values
    if len(sub) > 2 and sub.std() > 0.01:
        kde = gaussian_kde(sub, bw_method=.25)
        y_kde = kde(x_kde)
        y_kde = y_kde / y_kde.max() * .85
    else:
        y_kde = np.zeros_like(x_kde)
        closest = np.argmin(np.abs(x_kde - sub.mean()))
        y_kde[max(0, closest - 3):closest + 4] = .85
    meta = PATIENT_META[pid]; offset = meta['order']
    ax.fill_between(x_kde, offset, offset + y_kde, color=P_COLOR[pid], alpha=.5, zorder=3)
    ax.plot(x_kde, offset + y_kde, color=P_COLOR[pid], lw=2, zorder=4)
    ax.text(-3, offset + .35, f"{meta['label']}", fontsize=11, fontweight='bold',
            color=P_COLOR[pid], ha='right', va='center')
    ax.text(103, offset + .35, f"({meta['short']})", fontsize=9, color=TM, ha='left', va='center')

# shade severity zones
for x0, x1, c, lbl in [(0, 33.3, '#4A90D9', 'Normal zone'),
                         (33.3, 66.7, '#F5A623', 'Moderate zone'),
                         (66.7, 100, '#D94040', 'Severe zone')]:
    ax.axvspan(x0, x1, color=c, alpha=.04, zorder=0)

ax.set_xlabel("Pathological Index  (0 = Normal → 100 = Severe)", fontsize=13, color=TM)
ax.set_ylabel(""); ax.set_yticks([]); ax.set_xlim(-15, 115)
ax.set_title("Per-cell Pathological Index — Ridge Plot by Patient",
             fontsize=18, fontweight='bold', color=TK, pad=15)
ax.grid(axis='x', color=GRID_C, ls='--', alpha=.5)
for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID_C)
ax.tick_params(colors=TM, labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_ridge_pathological.png"), dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()
print("   ✅ Plot 10")

# ══════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════
out = df_all[[id_col, 'Disease'] + FEATURES + ['ctype', 'path_idx',
      'prob_normal', 'prob_mild', 'prob_moderate', 'prob_severe']].copy()
out['CellTypeName'] = out['ctype'].map(TYPE_NAMES)
out.to_excel(os.path.join(OUTPUT_DIR, "cells_with_LDA_classification.xlsx"), index=False)

print("\n" + "=" * 60)
print(f"🎉 Done! {len(os.listdir(OUTPUT_DIR))} files in '{OUTPUT_DIR}/'")
print("=" * 60)
for pid in ALL_IDS:
    sub = df_all[df_all[id_col] == pid]
    meta = PATIENT_META[pid]
    parts = "  ".join(f"{TYPE_NAMES[ct]}={prop[pid][ct]:.1f}%" for ct in range(best_k) if prop[pid][ct] > 0)
    print(f"  {meta['label']:3s} ({meta['short']:10s}): PathIdx={sub['path_idx'].mean():.1f}±{sub['path_idx'].std():.1f}  {parts}")