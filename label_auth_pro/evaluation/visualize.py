"""
EDA & Evaluation Visualizations
Texture patterns, QR distortions, tampering artifacts,
confusion matrix, ROC curves, attention maps, k-fold bars.
"""

import numpy as np
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2

DARK = "#0f172a"; CARD = "#1e293b"; BORDER = "#334155"
TEXT = "#e2e8f0"; MUTED= "#94a3b8"
PAL  = ["#22c55e","#ef4444","#a855f7","#f59e0b"]
CLASS_NAMES = ["GENUINE","TAMPERED","COUNTERFEIT","DEGRADED"]

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "font.family": "DejaVu Sans",
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  → {path}")


# ─── EDA ──────────────────────────────────────────────────────────────────────

def run_eda(imgs, labels, out_dir="reports/eda"):
    os.makedirs(out_dir, exist_ok=True)
    _class_distribution(labels, out_dir)
    _texture_distributions(imgs, labels, out_dir)
    _edge_maps(imgs, labels, out_dir)
    _fft_analysis(imgs, labels, out_dir)
    _color_channels(imgs, labels, out_dir)
    _pixel_intensity_heatmaps(imgs, labels, out_dir)
    print(f"  EDA complete → {out_dir}/")


def _class_distribution(labels, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11,4), facecolor=DARK)

    counts = [np.sum(labels==i) for i in range(4)]
    axes[0].bar(CLASS_NAMES, counts, color=PAL, width=0.55, edgecolor="none")
    for i,(n,c) in enumerate(zip(counts, PAL)):
        axes[0].text(i, n+2, str(n), ha="center", color=c, fontweight="bold")
    axes[0].set_title("Class Distribution", fontsize=12)
    axes[0].set_ylabel("Count"); axes[0].grid(axis="y", alpha=0.3)

    axes[1].pie(counts, labels=CLASS_NAMES, colors=PAL,
                autopct="%1.1f%%", pctdistance=0.82,
                wedgeprops={"edgecolor":"none"},
                textprops={"color":TEXT,"fontsize":9})
    axes[1].set_title("Class Balance", fontsize=12)

    plt.tight_layout()
    _save(fig, f"{out_dir}/class_distribution.png")


def _texture_distributions(imgs, labels, out_dir):
    metrics = {
        "Mean Intensity":  lambda i: cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).mean(),
        "Std Intensity":   lambda i: cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).std(),
        "Edge Density":    lambda i: cv2.Canny(i,50,150).mean(),
        "Color Variance":  lambda i: np.std([i[:,:,c].mean() for c in range(3)]),
        "Laplacian Var":   lambda i: cv2.Laplacian(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var(),
        "Gradient Mag":    lambda i: np.sqrt(cv2.Sobel(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).astype(np.float32),cv2.CV_32F,1,0)**2+cv2.Sobel(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).astype(np.float32),cv2.CV_32F,0,1)**2).mean(),
    }
    fig, axes = plt.subplots(2,3, figsize=(13,8), facecolor=DARK)
    for ax,(title,fn) in zip(axes.flat, metrics.items()):
        for cls_id,(cname,col) in enumerate(zip(CLASS_NAMES,PAL)):
            vals=[fn(imgs[j]) for j in np.where(labels==cls_id)[0][:80]]
            ax.hist(vals,bins=28,alpha=0.6,color=col,label=cname,edgecolor="none")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7,framealpha=0.2)
        ax.grid(alpha=0.3)
    fig.suptitle("Texture Feature Distributions — Tampering Artifact Analysis", fontsize=12)
    plt.tight_layout()
    _save(fig, f"{out_dir}/texture_distributions.png")


def _edge_maps(imgs, labels, out_dir):
    fig, axes = plt.subplots(4,5, figsize=(13,11), facecolor=DARK)
    for cls_id in range(4):
        cls_imgs = imgs[labels==cls_id][:5]
        for j,img in enumerate(cls_imgs):
            ax   = axes[cls_id,j]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 40, 130)
            ax.imshow(edge, cmap="hot"); ax.axis("off")
            if j==0:
                ax.set_ylabel(CLASS_NAMES[cls_id], color=PAL[cls_id],
                              fontsize=9, rotation=90, va="center")
    fig.suptitle("Canny Edge Maps — Tampering Artifact Detection", color=TEXT, fontsize=12)
    plt.tight_layout()
    _save(fig, f"{out_dir}/edge_maps.png")


def _fft_analysis(imgs, labels, out_dir):
    fig, axes = plt.subplots(4,5, figsize=(13,11), facecolor=DARK)
    for cls_id in range(4):
        cls_imgs = imgs[labels==cls_id][:5]
        for j,img in enumerate(cls_imgs):
            ax   = axes[cls_id,j]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
            fft  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
            ax.imshow(fft, cmap="plasma"); ax.axis("off")
            if j==0:
                ax.set_ylabel(CLASS_NAMES[cls_id], color=PAL[cls_id],
                              fontsize=9, rotation=90, va="center")
    fig.suptitle("FFT Frequency Spectra — QR Code Distortion Analysis", color=TEXT, fontsize=12)
    plt.tight_layout()
    _save(fig, f"{out_dir}/fft_analysis.png")


def _color_channels(imgs, labels, out_dir):
    fig, axes = plt.subplots(1,3, figsize=(13,4), facecolor=DARK)
    for ax,c_idx,cname in zip(axes,[0,1,2],["Blue","Green","Red"]):
        for cls_id,(cls,col) in enumerate(zip(CLASS_NAMES,PAL)):
            vals=[imgs[j][:,:,c_idx].mean()/255 for j in np.where(labels==cls_id)[0][:80]]
            ax.hist(vals,bins=28,alpha=0.6,color=col,label=cls,edgecolor="none")
        ax.set_title(f"{cname} Channel"); ax.legend(fontsize=7,framealpha=0.2)
        ax.grid(alpha=0.3)
    fig.suptitle("RGB Color Channel Distributions — Color Coherence Analysis", fontsize=12)
    plt.tight_layout()
    _save(fig, f"{out_dir}/color_channels.png")


def _pixel_intensity_heatmaps(imgs, labels, out_dir):
    """Average pixel heatmaps per class — shows structural differences."""
    fig, axes = plt.subplots(1,4, figsize=(14,4), facecolor=DARK)
    for cls_id,(ax,col) in enumerate(zip(axes,PAL)):
        cls_imgs = imgs[labels==cls_id]
        mean_img = cls_imgs.astype(float).mean(0)
        gray_mean= cv2.cvtColor(mean_img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        im = ax.imshow(gray_mean, cmap="inferno")
        ax.set_title(CLASS_NAMES[cls_id], color=col, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean Pixel Intensity Maps — Class Structural Profiles", fontsize=12)
    plt.tight_layout()
    _save(fig, f"{out_dir}/intensity_heatmaps.png")


# ─── Evaluation Plots ─────────────────────────────────────────────────────────

def plot_results(eval_results, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    if "val" in eval_results:
        _confusion_matrix(eval_results["val"], out_dir)
        _metric_bars(eval_results["val"], out_dir)
        _per_class_metrics(eval_results["val"], out_dir)
    if "kfold" in eval_results:
        _kfold_bars(eval_results["kfold"], out_dir)
        _inferential_summary(eval_results["kfold"]["inferential"], out_dir)
    print(f"  Eval plots → {out_dir}/")


def _confusion_matrix(val, out_dir):
    cm  = np.array(val["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7,6), facecolor=DARK)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, annot_kws={"size":11})
    ax.set_title(f"Confusion Matrix  (Acc={val['accuracy']:.3f})", fontsize=12)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    plt.tight_layout()
    _save(fig, f"{out_dir}/confusion_matrix.png")


def _metric_bars(val, out_dir):
    metrics = {"Accuracy":val["accuracy"],"Precision":val["precision_macro"],
               "Recall":val["recall_macro"],"F1 Macro":val["f1_macro"],
               "ROC-AUC":val["roc_auc"]}
    cols = ["#38bdf8","#34d399","#a78bfa","#fb923c","#f472b6"]
    fig, ax = plt.subplots(figsize=(9,4), facecolor=DARK)
    bars = ax.bar(metrics.keys(), metrics.values(), color=cols, width=0.5, edgecolor="none")
    for bar,v in zip(bars, metrics.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.009,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0,1.16); ax.set_title("Model Performance Metrics", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{out_dir}/metrics.png")


def _per_class_metrics(val, out_dir):
    rep = val["per_class"]
    x   = np.arange(4); w = 0.25
    fig, ax = plt.subplots(figsize=(10,5), facecolor=DARK)
    for i,(metric,color) in enumerate(zip(["precision","recall","f1-score"],
                                           ["#38bdf8","#34d399","#fb923c"])):
        vals = [rep[c].get(metric,0) for c in CLASS_NAMES]
        ax.bar(x + i*w, vals, w, label=metric.title(), color=color, edgecolor="none")
    ax.set_xticks(x+w); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0,1.15); ax.set_title("Per-Class Precision / Recall / F1", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{out_dir}/per_class_metrics.png")


def _kfold_bars(kfold, out_dir):
    agg  = kfold["aggregate"]
    keys = list(agg.keys())
    means= [agg[k]["mean"] for k in keys]
    stds = [agg[k]["std"]  for k in keys]
    fig, ax = plt.subplots(figsize=(10,5), facecolor=DARK)
    bars = ax.bar(keys, means, yerr=stds, color="#7c3aed", width=0.5, edgecolor="none",
                  capsize=6, error_kw={"color":"#a78bfa","linewidth":2})
    for bar,m,s in zip(bars,means,stds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.01,
                f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0,1.18); ax.set_title("K-Fold Cross-Validation Results (Mean ± Std)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{out_dir}/kfold_results.png")


def _inferential_summary(inf, out_dir):
    fig, ax = plt.subplots(figsize=(8,3), facecolor=DARK)
    ax.axis("off")
    rows = [
        ["Metric","Value"],
        ["t-statistic", f"{inf.get('t_statistic',0):.4f}"],
        ["p-value", f"{inf.get('p_value',0):.4e}"],
        ["Significant (p<0.05)", str(inf.get('significant',False))],
        ["Cohen's d (effect size)", f"{inf.get('effect_size_d',0):.4f}"],
        ["95% CI", f"[{inf.get('ci_95',[0,0])[0]:.4f}, {inf.get('ci_95',[0,0])[1]:.4f}]"],
    ]
    tbl = ax.table(rows[1:], colLabels=rows[0], loc="center",
                   cellLoc="center", colColours=["#1e3a5f","#1e3a5f"])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.4, 1.8)
    ax.set_title("Inferential Statistics — t-test vs Random Baseline", fontsize=12, pad=12)
    plt.tight_layout()
    _save(fig, f"{out_dir}/inferential_stats.png")


def plot_attention_maps(imgs, labels, pipeline, out_dir="reports", n=4):
    """Visualize ViT attention rollout maps."""
    fig, axes = plt.subplots(4, n, figsize=(n*3, 14), facecolor=DARK)
    fig.suptitle("ViT Attention Rollout Maps — XAI Visualization", fontsize=12, color=TEXT)

    for cls_id in range(4):
        cls_imgs = imgs[labels==cls_id][:n]
        for j, img in enumerate(cls_imgs):
            ax  = axes[cls_id, j]
            try:
                attn= pipeline.vit.get_attention_map(img)
                # Resize to match image
                attn_big = cv2.resize(attn.astype(np.float32), (64,64),
                                      interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Overlay
                heatmap = plt.cm.jet(attn_big)[:,:,:3]
                overlay = (0.5*rgb/255 + 0.5*heatmap)
                ax.imshow(overlay); ax.axis("off")
            except Exception:
                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); ax.axis("off")
            if j==0:
                ax.set_ylabel(CLASS_NAMES[cls_id], color=PAL[cls_id],
                              fontsize=9, rotation=90, va="center")
    plt.tight_layout()
    _save(fig, f"{out_dir}/attention_maps.png")
