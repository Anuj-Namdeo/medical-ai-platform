"""
eda.py  →  Convert to Jupyter notebook with: jupytext --to notebook eda.py
=========================================================================
EXPLORATORY DATA ANALYSIS
Chest X-Ray Pneumonia Detection Dataset
=========================================================================
Run each cell block in order inside Jupyter / VS Code Notebook.
"""

# %% [markdown]
# # 🫁 Chest X-Ray Pneumonia — Exploratory Data Analysis
# **Dataset:** Kaggle Chest X-Ray Images (Pneumonia)  
# **Source:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
# **Task:** Binary Classification — NORMAL vs PNEUMONIA  

# %%
# ── IMPORTS ─────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# Configure display
plt.rcParams.update({
    "figure.facecolor":  "#0F172A",
    "axes.facecolor":    "#1E293B",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#F1F5F9",
    "text.color":        "#F1F5F9",
    "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8",
    "grid.color":        "#334155",
    "figure.titlesize":  14,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "font.family":       "sans-serif",
})

sns.set_palette(["#2563EB", "#EF4444", "#22C55E", "#F97316", "#A78BFA"])

print("✅ Imports successful")
print(f"   NumPy  : {np.__version__}")
print(f"   Pandas : {pd.__version__}")
print(f"   OpenCV : {cv2.__version__}")

# %%
# ── DATA PATHS ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"

SPLITS = {
    "train": DATA_DIR / "train",
    "val":   DATA_DIR / "val",
    "test":  DATA_DIR / "test",
}
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_COLORS = {"NORMAL": "#22C55E", "PNEUMONIA": "#EF4444"}

# Verify dataset exists
for split, path in SPLITS.items():
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {split}: {path}")

# %%
# ── 1. DATASET OVERVIEW ──────────────────────────────────────────────────────
print("\n" + "="*55)
print("   DATASET OVERVIEW")
print("="*55)

counts = {}
totals = {}
for split, path in SPLITS.items():
    counts[split] = {}
    for cls in CLASSES:
        cls_path = path / cls
        if cls_path.exists():
            n = len(list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            counts[split][cls] = n
        else:
            counts[split][cls] = 0
    totals[split] = sum(counts[split].values())

df_counts = pd.DataFrame(counts).T
df_counts["TOTAL"] = df_counts.sum(axis=1)
df_counts.index.name = "Split"

print(df_counts.to_string())
print(f"\nGrand Total: {df_counts['TOTAL'].sum()} images")

# %% [markdown]
# ### Observations
# - The **training set** has a significant class imbalance (more PNEUMONIA than NORMAL images).
# - The **validation set** is very small — we rely primarily on the test set for evaluation.
# - We will use **class weights** and **data augmentation** to address the imbalance.

# %%
# ── 2. CLASS DISTRIBUTION BAR CHART ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Class Distribution Across Dataset Splits", fontsize=14, fontweight="bold", y=1.02)

for ax, (split, split_counts) in zip(axes, counts.items()):
    bars = ax.bar(
        list(split_counts.keys()),
        list(split_counts.values()),
        color=[CLASS_COLORS["NORMAL"], CLASS_COLORS["PNEUMONIA"]],
        edgecolor="#0F172A",
        linewidth=0.8,
        width=0.5,
    )
    for bar, (cls, cnt) in zip(bars, split_counts.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(cnt),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=CLASS_COLORS[cls]
        )
    ax.set_title(f"{split.upper()} Split  (n={totals[split]})", fontsize=12, pad=8)
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(list(split_counts.values())) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "docs" / "eda_class_distribution.png"),
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
plt.show()
print("✅ Class distribution chart saved")

# %%
# ── 3. IMBALANCE RATIO ───────────────────────────────────────────────────────
print("\n📊 CLASS IMBALANCE ANALYSIS")
print("-" * 40)
for split in ["train", "test"]:
    n_normal    = counts[split]["NORMAL"]
    n_pneumonia = counts[split]["PNEUMONIA"]
    total       = n_normal + n_pneumonia
    ratio       = n_pneumonia / n_normal if n_normal > 0 else 0
    pct_normal  = 100 * n_normal    / total if total > 0 else 0
    pct_pneum   = 100 * n_pneumonia / total if total > 0 else 0
    print(f"\n{split.upper()}:")
    print(f"  NORMAL    : {n_normal:>5}  ({pct_normal:.1f}%)")
    print(f"  PNEUMONIA : {n_pneumonia:>5}  ({pct_pneum:.1f}%)")
    print(f"  Imbalance ratio (PNEUMONIA:NORMAL) = {ratio:.2f}:1")

# %%
# ── 4. SAMPLE IMAGES ─────────────────────────────────────────────────────────

def load_sample_images(base_path: Path, cls: str, n: int = 5):
    """Load n sample images from a class folder."""
    cls_path = base_path / cls
    if not cls_path.exists():
        print(f"⚠️  Path not found: {cls_path}")
        return []
    files = list(cls_path.glob("*.jpeg"))[:n] + list(cls_path.glob("*.jpg"))[:n]
    images = []
    for f in files[:n]:
        img = cv2.imread(str(f))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((img_rgb, f.name))
    return images


train_path = SPLITS["train"]
samples_normal    = load_sample_images(train_path, "NORMAL",    n=5)
samples_pneumonia = load_sample_images(train_path, "PNEUMONIA", n=5)

if samples_normal or samples_pneumonia:
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Sample X-Ray Images — NORMAL (top) vs PNEUMONIA (bottom)",
                 fontsize=13, fontweight="bold")

    for col, (img, fname) in enumerate(samples_normal[:5]):
        axes[0][col].imshow(img, cmap="gray")
        axes[0][col].set_title(fname[:20], fontsize=7, color="#22C55E")
        axes[0][col].axis("off")

    for col, (img, fname) in enumerate(samples_pneumonia[:5]):
        axes[1][col].imshow(img, cmap="gray")
        axes[1][col].set_title(fname[:20], fontsize=7, color="#EF4444")
        axes[1][col].axis("off")

    # Row labels
    for row, (label, color) in enumerate([("NORMAL", "#22C55E"), ("PNEUMONIA", "#EF4444")]):
        fig.text(-0.01, 0.73 - row * 0.47, label, va="center", ha="right",
                 fontsize=11, fontweight="bold", color=color, rotation=90)

    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "docs" / "eda_sample_images.png"),
                dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.show()
else:
    print("⚠️  No images loaded — verify that the dataset is downloaded.")

# %%
# ── 5. IMAGE DIMENSIONS ANALYSIS ─────────────────────────────────────────────
print("\n📐 IMAGE DIMENSIONS ANALYSIS")
print("   Sampling up to 500 images per class per split...")

dimension_data = []
for split, path in [("train", SPLITS["train"])]:
    for cls in CLASSES:
        cls_path = path / cls
        if not cls_path.exists():
            continue
        files = list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg"))
        files = files[:500]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                dimension_data.append({
                    "class": cls, "split": split,
                    "height": h, "width": w,
                    "aspect_ratio": round(w / h, 3),
                    "megapixels": round((h * w) / 1e6, 3),
                })

if dimension_data:
    df_dim = pd.DataFrame(dimension_data)

    print("\nDimension Summary by Class:")
    print(df_dim.groupby("class")[["height", "width", "aspect_ratio"]].describe().round(1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Image Dimension Distribution (Training Set Sample)", fontsize=13, fontweight="bold")

    # Height distribution
    for cls in CLASSES:
        subset = df_dim[df_dim["class"] == cls]
        axes[0].hist(subset["height"], bins=30, alpha=0.6, label=cls,
                     color=CLASS_COLORS[cls], edgecolor="none")
    axes[0].set_title("Image Heights (px)")
    axes[0].set_xlabel("Height (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Width distribution
    for cls in CLASSES:
        subset = df_dim[df_dim["class"] == cls]
        axes[1].hist(subset["width"], bins=30, alpha=0.6, label=cls,
                     color=CLASS_COLORS[cls], edgecolor="none")
    axes[1].set_title("Image Widths (px)")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Aspect ratio
    for cls in CLASSES:
        subset = df_dim[df_dim["class"] == cls]
        axes[2].hist(subset["aspect_ratio"], bins=30, alpha=0.6, label=cls,
                     color=CLASS_COLORS[cls], edgecolor="none")
    axes[2].set_title("Aspect Ratio (W/H)")
    axes[2].set_xlabel("Aspect Ratio")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "docs" / "eda_dimensions.png"),
                dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.show()
else:
    print("⚠️  No images found for dimension analysis.")

# %%
# ── 6. PIXEL INTENSITY ANALYSIS ──────────────────────────────────────────────
print("\n🔬 PIXEL INTENSITY ANALYSIS")
print("   Computing mean intensity histograms (sample 200 images/class)...")

intensity_data = {"NORMAL": [], "PNEUMONIA": []}

for cls in CLASSES:
    cls_path = SPLITS["train"] / cls
    if not cls_path.exists():
        continue
    files = (list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg")))[:200]
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            intensity_data[cls].append(float(np.mean(img)))

if any(intensity_data.values()):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pixel Intensity Analysis", fontsize=13, fontweight="bold")

    # Mean intensity per image
    for cls in CLASSES:
        if intensity_data[cls]:
            axes[0].hist(intensity_data[cls], bins=40, alpha=0.7,
                         label=cls, color=CLASS_COLORS[cls], edgecolor="none")
    axes[0].set_title("Mean Pixel Intensity per Image")
    axes[0].set_xlabel("Mean Intensity (0=Black, 255=White)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Average histogram over all images
    all_histograms = {}
    for cls in CLASSES:
        cls_path = SPLITS["train"] / cls
        if not cls_path.exists():
            continue
        files = (list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg")))[:100]
        cumulative_hist = np.zeros(256)
        count = 0
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
                cumulative_hist += hist
                count += 1
        if count > 0:
            all_histograms[cls] = cumulative_hist / count

    for cls, hist in all_histograms.items():
        axes[1].plot(hist, label=cls, color=CLASS_COLORS[cls], linewidth=1.5, alpha=0.8)
    axes[1].set_title("Average Pixel Value Distribution")
    axes[1].set_xlabel("Pixel Value (0-255)")
    axes[1].set_ylabel("Average Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "docs" / "eda_pixel_intensity.png"),
                dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.show()

    # Print stats
    for cls in CLASSES:
        if intensity_data[cls]:
            arr = np.array(intensity_data[cls])
            print(f"\n{cls} Mean Intensity:")
            print(f"  Mean  = {arr.mean():.2f}")
            print(f"  Std   = {arr.std():.2f}")
            print(f"  Min   = {arr.min():.2f}")
            print(f"  Max   = {arr.max():.2f}")
else:
    print("⚠️  No images available for intensity analysis.")

# %% [markdown]
# ### Key Insights from Pixel Analysis
# - **PNEUMONIA X-rays** tend to have **lower mean brightness** due to consolidation (white infiltrates).
# - **NORMAL X-rays** are generally **clearer** with more uniform background.
# - This difference justifies CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.

# %%
# ── 7. CLAHE PREPROCESSING COMPARISON ────────────────────────────────────────

def show_preprocessing_pipeline(img_path: Path, title: str):
    """Show original → CLAHE → Denoised comparison."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Could not load {img_path}")
        return

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized  = cv2.resize(denoised, (224, 224))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Preprocessing Pipeline — {title}", fontsize=12, fontweight="bold")

    stages = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original"),
        (gray,    "Grayscale"),
        (enhanced,"CLAHE Enhanced"),
        (resized, "Final (224×224)"),
    ]

    for ax, (stage_img, stage_name) in zip(axes, stages):
        ax.imshow(stage_img, cmap="gray")
        ax.set_title(stage_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "docs" / f"eda_preprocessing_{title.lower().replace(' ', '_')}.png"),
                dpi=130, bbox_inches="tight", facecolor="#0F172A")
    plt.show()


# Show for one image from each class
for cls in CLASSES:
    cls_path = SPLITS["train"] / cls
    if cls_path.exists():
        files = list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg"))
        if files:
            show_preprocessing_pipeline(files[0], f"{cls} Sample")

# %%
# ── 8. DATA AUGMENTATION PREVIEW ─────────────────────────────────────────────
import cv2
import numpy as np

def augment_image(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply a random augmentation to an image."""
    rng = np.random.default_rng(seed)
    aug_type = seed % 5
    if aug_type == 0:  # Horizontal flip
        return cv2.flip(img, 1)
    elif aug_type == 1:  # Rotation
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), 1.0)
        return cv2.warpAffine(img, M, (w, h))
    elif aug_type == 2:  # Brightness
        factor = rng.uniform(0.7, 1.3)
        return np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
    elif aug_type == 3:  # Zoom
        h, w = img.shape[:2]
        zoom = rng.uniform(0.85, 1.0)
        cy, cx = h // 2, w // 2
        crop = img[
            int(cy * (1 - zoom)):int(cy * (1 + zoom)),
            int(cx * (1 - zoom)):int(cx * (1 + zoom))
        ]
        return cv2.resize(crop, (w, h)) if crop.size > 0 else img
    else:  # Gaussian noise
        noise = rng.normal(0, 10, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


cls_path = SPLITS["train"] / "PNEUMONIA"
if cls_path.exists():
    files = list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg"))
    if files:
        base_img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
        if base_img is not None:
            base_img = cv2.resize(base_img, (224, 224))
            aug_labels = ["Original", "H-Flip", "Rotation", "Brightness", "Zoom", "Noise"]
            fig, axes = plt.subplots(1, 6, figsize=(18, 4))
            fig.suptitle("Data Augmentation Techniques Applied to PNEUMONIA X-Ray",
                         fontsize=11, fontweight="bold")

            axes[0].imshow(base_img, cmap="gray")
            axes[0].set_title("Original", fontsize=9)
            axes[0].axis("off")

            for i in range(1, 6):
                aug = augment_image(base_img, seed=i)
                axes[i].imshow(aug, cmap="gray")
                axes[i].set_title(aug_labels[i], fontsize=9)
                axes[i].axis("off")

            plt.tight_layout()
            plt.savefig(str(PROJECT_ROOT / "docs" / "eda_augmentation.png"),
                        dpi=130, bbox_inches="tight", facecolor="#0F172A")
            plt.show()

# %%
# ── 9. FILE SIZE ANALYSIS ─────────────────────────────────────────────────────
file_size_data = []
for split, path in [("train", SPLITS["train"])]:
    for cls in CLASSES:
        cls_path = path / cls
        if not cls_path.exists():
            continue
        for f in list(cls_path.glob("*.jpeg"))[:300] + list(cls_path.glob("*.jpg"))[:300]:
            size_kb = f.stat().st_size / 1024
            file_size_data.append({"class": cls, "size_kb": size_kb})

if file_size_data:
    df_fs = pd.DataFrame(file_size_data)
    print("\n📦 FILE SIZE SUMMARY (KB):")
    print(df_fs.groupby("class")["size_kb"].describe().round(2))

    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in CLASSES:
        subset = df_fs[df_fs["class"] == cls]
        ax.hist(subset["size_kb"], bins=50, alpha=0.7,
                label=cls, color=CLASS_COLORS[cls], edgecolor="none")
    ax.set_title("File Size Distribution (KB) — Training Set")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "docs" / "eda_file_sizes.png"),
                dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.show()

# %%
# ── 10. SUMMARY TABLE ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("   FINAL EDA SUMMARY")
print("="*60)

summary_rows = []
for split in ["train", "val", "test"]:
    n_normal    = counts[split].get("NORMAL",    0)
    n_pneumonia = counts[split].get("PNEUMONIA", 0)
    total       = n_normal + n_pneumonia
    imbalance   = f"{n_pneumonia/n_normal:.2f}:1" if n_normal > 0 else "N/A"
    summary_rows.append({
        "Split":         split.upper(),
        "NORMAL":        n_normal,
        "PNEUMONIA":     n_pneumonia,
        "Total":         total,
        "P(Pneumonia)":  f"{100*n_pneumonia/total:.1f}%" if total > 0 else "N/A",
        "Imbalance":     imbalance,
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
print()
print("📌 KEY TAKEAWAYS:")
print("  1. Training set is IMBALANCED — use class_weight in Keras training")
print("  2. Validation set is small — use test set as primary evaluation")
print("  3. Images vary in size — standardize to 224×224 during preprocessing")
print("  4. CLAHE preprocessing enhances contrast in infiltrate regions")
print("  5. RECALL is the primary metric — missing pneumonia is more dangerous")
print("  6. Apply augmentation only to training set to prevent data leakage")
print()
print("✅ EDA complete. Proceed to src/preprocessing.py")
