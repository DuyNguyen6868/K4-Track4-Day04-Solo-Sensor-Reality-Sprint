"""T1 - Camera degradation health score.

Runs one frozen detector (YOLOv8n) over the same COCO128 images under a
baseline and several synthetic degradation levels, and records image health
metrics next to detection quality measured against the ground-truth labels.

Usage:
    python src/benchmark.py --kernels 5 9 15 25 --gains 1.5 2.5 4.0
"""
import argparse
import json
import logging
import platform
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
DATA_DIR = ROOT / "datasets" / "coco128"
OUT_DIR = ROOT / "results"

# COCO class ids relevant to a forward ADAS camera
ADAS_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                5: "bus", 7: "truck", 9: "traffic light"}
CONF_THRES = 0.25
IOU_MATCH = 0.5
IMGSZ = 640
SATURATED_LEVEL = 250

log = logging.getLogger("benchmark")


def imread(path):
    # cv2.imread cannot open non-ASCII paths on Windows
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def ensure_dataset():
    if DATA_DIR.exists():
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.parent / "coco128.zip"
    log.info("Downloading %s", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR.parent)
    zip_path.unlink()


def motion_blur(img, k):
    """Horizontal uniform motion blur with a kernel k pixels long."""
    if k <= 1:
        return img
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def exposure_gain(img, gain):
    """Overexposure: multiply intensities and clip at the sensor ceiling."""
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def health_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return {
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "exposure_mean": float(gray.mean()),
        "saturation_ratio_pct": float((gray >= SATURATED_LEVEL).mean() * 100),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
    }


def load_labels(label_path, w, h):
    """YOLO txt -> list of (class_id, [x1, y1, x2, y2]) for ADAS classes."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls not in ADAS_CLASSES:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append((cls, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                            (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts, dets):
    """Greedy matching, highest confidence first. Returns per-GT confidence
    (None if missed) and the number of unmatched detections."""
    gt_conf = [None] * len(gts)
    fp = 0
    for cls, box, conf in sorted(dets, key=lambda d: -d[2]):
        best, best_iou = -1, IOU_MATCH
        for i, (gcls, gbox) in enumerate(gts):
            if gcls != cls or gt_conf[i] is not None:
                continue
            v = iou(box, gbox)
            if v >= best_iou:
                best, best_iou = i, v
        if best >= 0:
            gt_conf[best] = conf
        else:
            fp += 1
    return gt_conf, fp


def size_bucket(box):
    # COCO convention: small < 32^2 px, medium < 96^2 px, else large
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", type=int, nargs="*", default=[5, 9, 15, 25],
                    help="motion blur kernel lengths in px")
    ap.add_argument("--gains", type=float, nargs="*", default=[1.5, 2.5, 4.0],
                    help="exposure gain factors")
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(OUT_DIR / "run.log", mode="w", encoding="utf-8")])

    conditions = [("baseline", "none", 0.0, lambda im: im)]
    conditions += [(f"blur_{k}px", "motion_blur", float(k), lambda im, k=k: motion_blur(im, k))
                   for k in args.kernels]
    conditions += [(f"gain_x{g:g}", "exposure_gain", g, lambda im, g=g: exposure_gain(im, g))
                   for g in args.gains]

    ensure_dataset()
    images = sorted((DATA_DIR / "images" / "train2017").glob("*.jpg"))
    model = YOLO(str(ROOT / args.weights))

    config = {
        "dataset": "COCO128 (first 128 images of COCO train2017)", "dataset_url": DATA_URL,
        "n_images": len(images), "weights": args.weights, "device": "cpu",
        "classes": ADAS_CLASSES, "conf_thres": CONF_THRES, "iou_match": IOU_MATCH,
        "imgsz": IMGSZ, "saturated_level": SATURATED_LEVEL,
        "motion_blur_kernels_px": args.kernels, "exposure_gains": args.gains,
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
        "numpy": np.__version__, "cpu": platform.processor(),
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log.info("Config: %s", json.dumps(config))

    # warm-up so the first condition does not absorb model start-up time
    model.predict(imread(images[0]), imgsz=IMGSZ, conf=CONF_THRES, device="cpu", verbose=False)

    img_rows, obj_rows = [], []
    for name, family, level, degrade in conditions:
        for path in images:
            img = degrade(imread(path))
            h, w = img.shape[:2]
            gts = load_labels(DATA_DIR / "labels" / "train2017" / f"{path.stem}.txt", w, h)
            res = model.predict(img, imgsz=IMGSZ, conf=CONF_THRES, device="cpu",
                                classes=list(ADAS_CLASSES), verbose=False)[0]
            dets = [(int(c), b.tolist(), float(s)) for c, b, s in
                    zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf)]
            gt_conf, fp = match(gts, dets)

            row = {"condition": name, "family": family, "level": level, "image": path.name,
                   "n_gt": len(gts), "tp": sum(c is not None for c in gt_conf), "fp": fp,
                   "conf_sum_tp": sum(c for c in gt_conf if c is not None),
                   "latency_ms": sum(res.speed.values())}
            row.update(health_metrics(img))
            img_rows.append(row)
            for i, ((cls, box), c) in enumerate(zip(gts, gt_conf)):
                obj_rows.append({"condition": name, "family": family, "level": level,
                                 "image": path.name, "gt_index": i, "class": ADAS_CLASSES[cls],
                                 "size": size_bucket(box), "detected": c is not None,
                                 "confidence": c})
        log.info("Finished %s", name)

    per_image = pd.DataFrame(img_rows)
    per_object = pd.DataFrame(obj_rows)
    per_image.to_csv(OUT_DIR / "per_image.csv", index=False)
    per_object.to_csv(OUT_DIR / "per_object.csv", index=False)

    # Health thresholds are set from the baseline distribution only
    base = per_image[per_image.condition == "baseline"]
    tau_blur = float(base.blur_score.quantile(0.10))
    tau_sat = float(base.saturation_ratio_pct.quantile(0.90))
    per_image["flag_blur"] = per_image.blur_score < tau_blur
    per_image["flag_sat"] = per_image.saturation_ratio_pct > tau_sat

    summary = per_image.groupby(["condition", "family", "level"], sort=False).agg(
        n_gt=("n_gt", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        conf_sum_tp=("conf_sum_tp", "sum"),
        blur_score_median=("blur_score", "median"),
        exposure_mean=("exposure_mean", "mean"),
        saturation_ratio_pct=("saturation_ratio_pct", "mean"),
        entropy_bits=("entropy_bits", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        flag_blur_pct=("flag_blur", lambda s: s.mean() * 100),
        flag_sat_pct=("flag_sat", lambda s: s.mean() * 100),
    ).reset_index()
    summary["recall_pct"] = summary.tp / summary.n_gt * 100
    summary["precision_pct"] = summary.tp / (summary.tp + summary.fp) * 100
    summary["mean_conf_tp"] = summary.conf_sum_tp / summary.tp
    summary["recall_rel_pct"] = summary.recall_pct / summary.recall_pct.iloc[0] * 100
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    by_size = (per_object.groupby(["condition", "size"], sort=False).detected
               .agg(n="size", recall_pct=lambda s: s.mean() * 100).reset_index())
    by_size.to_csv(OUT_DIR / "recall_by_size.csv", index=False)

    log.info("Thresholds from baseline: blur_score < %.1f (p10), saturation_ratio > %.2f%% (p90)",
             tau_blur, tau_sat)
    cols = ["condition", "n_gt", "tp", "fp", "recall_pct", "recall_rel_pct", "precision_pct",
            "mean_conf_tp", "blur_score_median", "exposure_mean", "saturation_ratio_pct",
            "entropy_bits", "latency_ms_mean", "flag_blur_pct", "flag_sat_pct"]
    log.info("Summary:\n%s", summary[cols].round(2).to_string(index=False))
    log.info("Recall by object size:\n%s", by_size.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
