"""Plots for the T1 benchmark. Reads results/*.csv written by benchmark.py.

Usage:
    python src/plots.py
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from benchmark import (ADAS_CLASSES, CONF_THRES, DATA_DIR, IMGSZ, OUT_DIR, ROOT,
                       imread, motion_blur)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def blur_rows(df):
    return df[df.family.isin(["none", "motion_blur"])].sort_values("level")


def style_x(ax, levels):
    ax.set_xticks(levels)
    ax.set_xticklabels(["0\n(baseline)"] + [f"{int(v)}" for v in levels[1:]])
    ax.tick_params(length=0)


def fig_blur_sweep(summary):
    d = blur_rows(summary)
    x = d.level.to_numpy()
    panels = [
        ("recall_pct", "Recall (%)", "{:.1f}", (0, 100)),
        ("mean_conf_tp", "Confidence trung bình của detection đúng (0-1)", "{:.2f}", (0, 1)),
        ("blur_score_median", "Blur score, trung vị (phương sai Laplacian)", "{:.0f}", (0, None)),
        ("flag_blur_pct", "Khung hình bị gắn cờ nhòe (%)", "{:.0f}", (0, 100)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for ax, (col, title, fmt, ylim) in zip(axes.ravel(), panels):
        y = d[col].to_numpy()
        ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        # the blur score curve is steep, so its labels sit beside the markers
        offset, ha = ((9, 5), "left") if col == "blur_score_median" else ((0, 8), "center")
        for xi, yi in zip(x, y):
            ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points",
                        xytext=offset, ha=ha, fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_ylim(ylim[0], ylim[1] if ylim[1] else y.max() * 1.2)
        style_x(ax, x)
    for ax in axes[1]:
        ax.set_xlabel("Độ dài kernel motion blur (px)")
    fig.suptitle("Motion blur tăng: recall giảm mạnh, confidence gần như đứng yên",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.01, "YOLOv8n, CPU, 128 ảnh COCO128, 344 object thuộc 7 lớp ADAS. "
             "Ngưỡng gắn cờ = phân vị 10% của blur score ở baseline.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT_DIR / "fig1_blur_sweep.png", dpi=160)
    plt.close(fig)


def fig_recall_by_size(by_size, summary):
    levels = blur_rows(summary)[["condition", "level"]]
    d = by_size.merge(levels, on="condition")
    names = {"large": ("Lớn (≥ 96² px)", BLUE), "medium": ("Vừa (32²-96² px)", ORANGE),
             "small": ("Nhỏ (< 32² px)", AQUA)}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for size, (label, color) in names.items():
        s = d[d["size"] == size].sort_values("level")
        n = int(s.n.iloc[0])
        ax.plot(s.level, s.recall_pct, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label}, n={n}")
        below = size == "small"  # keeps the small and medium labels apart near zero
        for xi, yi in zip(s.level, s.recall_pct):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, -15 if below else 8), ha="center", fontsize=9, color=INK)
    ax.set_ylim(-9, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Recall (%)")
    ax.set_xlabel("Độ dài kernel motion blur (px)")
    style_x(ax, blur_rows(summary).level.to_numpy())
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2, title="Kích thước object")
    ax.set_title("Object vừa và nhỏ biến mất trước khi object lớn bị ảnh hưởng", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_recall_by_size.png", dpi=160)
    plt.close(fig)


def fig_before_after(per_image):
    base = per_image[per_image.condition == "baseline"].set_index("image")
    worst = per_image[per_image.condition == "blur_15px"].set_index("image")
    image = (base.tp - worst.tp).idxmax()
    model = YOLO(str(ROOT / "yolov8n.pt"))
    src = imread(DATA_DIR / "images" / "train2017" / image)
    kernels = [0, 9, 15, 25]
    fig, axes = plt.subplots(1, len(kernels), figsize=(14, 3.7))
    for ax, k in zip(axes, kernels):
        img = motion_blur(src, k)
        res = model.predict(img, imgsz=IMGSZ, conf=CONF_THRES, device="cpu",
                            classes=list(ADAS_CLASSES), verbose=False)[0]
        vis = img.copy()
        for x1, y1, x2, y2 in res.boxes.xyxy.int().tolist():
            cv2.rectangle(vis, (x1, y1), (x2, y2), (214, 120, 42), 2)  # BGR of BLUE
        row = per_image[(per_image.image == image) & (per_image.level == k) &
                        per_image.family.isin(["none", "motion_blur"])].iloc[0]
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{'Baseline' if k == 0 else f'Blur {k} px'}: đúng {int(row.tp)}/{int(row.n_gt)} object\n"
                     f"blur score {row.blur_score:.0f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Cùng một ảnh ({image}), khung xanh là detection của YOLOv8n",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "fig3_before_after.png", dpi=150)
    plt.close(fig)
    return image


def main():
    summary = pd.read_csv(OUT_DIR / "summary.csv")
    per_image = pd.read_csv(OUT_DIR / "per_image.csv")
    by_size = pd.read_csv(OUT_DIR / "recall_by_size.csv")
    fig_blur_sweep(summary)
    fig_recall_by_size(by_size, summary)
    print("before/after image:", fig_before_after(per_image))


if __name__ == "__main__":
    main()
