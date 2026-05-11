"""YOLO Auto-Labeling Tool for Red Blood Cells in Microfluidic Channels.

Two run modes:

  Single sample:
      python auto_label.py /path/to/sample
  Batch (multiple samples under one parent dir):
      python auto_label.py /path/to/parent --batch

Outputs per sample:
  <sample>/labels/<stem>.txt           YOLO labels
  <sample>/visualization/<stem>.jpg    annotated images
  <sample>/summary.csv                 per-image detection report

Class labels:
  0 = entering (进), green bbox
  1 = in       (在), blue  bbox
  2 = exiting  (出), red   bbox

Cells whose bounding box straddles a neck baseline (i.e. they are
actively transitioning) are marked with an asterisk in the visualization
and reported in the CSV as "transitions". Their YOLO class is still
assigned by centroid; the asterisk is only a hint for manual review.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cell_detect import detect_cells  # noqa: E402
from neck_detect import detect_channel_neck, build_channel_interior_mask  # noqa: E402


VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CLASS_NAMES = {0: "enter", 1: "in", 2: "exit"}
CLASS_CN = {0: "进", 1: "在", 2: "出"}
# BGR colors
CLASS_COLORS = {0: (0, 220, 0), 1: (255, 128, 0), 2: (0, 0, 255)}


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", s)]


def classify_status(bbox, neck_top, neck_bottom):
    """Classify by bbox edges:

      0 enter:  bbox bottom is strictly above neck_top
                -> cell hasn't started entering the neck
      1 in:     bbox overlaps the neck region in any way
                -> at least part of the cell is in (or past the start of)
                   the neck, and not all of it has exited
      2 exit:   bbox top is at or below neck_bottom
                -> whole cell has cleared the neck exit line

    This matches the user's rule:
      - bbox bottom touching neck_top line => no longer "enter"
      - bbox top touching neck_bottom line => becomes "exit"
    """
    x, y, w, h = bbox
    y_top = y
    y_bottom = y + h
    if y_top >= neck_bottom:
        return 2
    if y_bottom < neck_top:
        return 0
    return 1


def bbox_crosses_neck(bbox, neck_top, neck_bottom):
    """True if bbox straddles either neck line — useful review hint.
    With the new classification rule these cases are still class 1,
    but flagging them helps a human review transitions."""
    x, y, w, h = bbox
    y2 = y + h
    return (y <= neck_top <= y2) or (y <= neck_bottom <= y2)


def bbox_to_yolo(bbox, img_w, img_h):
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return cx, cy, nw, nh


def draw_visualization(image_bgr, cells, neck_top, neck_bottom,
                       manual_neck=False):
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    neck_color = (0, 255, 255)
    cv2.line(vis, (0, neck_top), (w, neck_top), neck_color, 2)
    cv2.line(vis, (0, neck_bottom), (w, neck_bottom), neck_color, 2)

    tag = "neck (manual)" if manual_neck else "neck (auto)"
    cv2.putText(vis, f"{tag} top y={neck_top}", (8, max(neck_top - 6, 14)),
                font, 0.6, neck_color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"{tag} bot y={neck_bottom}",
                (8, min(neck_bottom + 22, h - 4)),
                font, 0.6, neck_color, 2, cv2.LINE_AA)

    for c in cells:
        x, y, bw, bh = c["bbox"]
        cls = c["class"]
        color = CLASS_COLORS[cls]
        thick = 3 if c.get("transition", False) else 2
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, thick)

        label = f"{cls}:{CLASS_NAMES[cls]}"
        if c.get("transition", False):
            label += "*"
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        ty = max(y - 6, th + 4)
        cv2.rectangle(vis, (x, ty - th - 4), (x + tw + 4, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(vis, label, (x + 2, ty - 2), font, 0.6, color, 2,
                    cv2.LINE_AA)
        cx_i, cy_i = int(c["centroid"][0]), int(c["centroid"][1])
        cv2.circle(vis, (cx_i, cy_i), 3, color, -1)

    return vis


def process_image(image_path, labels_dir, vis_dir,
                  manual_neck=None, detect_kwargs=None,
                  save_vis=True, debug=False):
    detect_kwargs = detect_kwargs or {}

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [ERROR] could not read {image_path}")
        return None
    H, W = img.shape[:2]

    # ALWAYS run channel/neck auto-detection to get per-row wall positions.
    # Even when manual_neck is provided, we still need the wall info to
    # build the channel-interior mask (which excludes out-of-channel noise).
    neck_info = detect_channel_neck(img, debug=debug)
    channel_mask = build_channel_interior_mask(
        img.shape, neck_info["lefts"], neck_info["rights"], margin=0
    )

    is_manual = manual_neck is not None
    if is_manual:
        neck_top, neck_bottom = manual_neck
        neck_top = max(0, min(H - 1, int(neck_top)))
        neck_bottom = max(0, min(H - 1, int(neck_bottom)))
        if neck_top > neck_bottom:
            neck_top, neck_bottom = neck_bottom, neck_top
    else:
        neck_top = neck_info["neck_top"]
        neck_bottom = neck_info["neck_bottom"]

    cells, _ = detect_cells(
        img,
        channel_mask=channel_mask,
        neck_y_range=(neck_top, neck_bottom),
        debug=debug,
        **detect_kwargs,
    )

    yolo_lines = []
    transitions = 0
    for c in cells:
        cls = classify_status(c["bbox"], neck_top, neck_bottom)
        c["class"] = cls
        c["transition"] = bbox_crosses_neck(c["bbox"], neck_top, neck_bottom)
        if c["transition"]:
            transitions += 1
        ncx, ncy, nw, nh = bbox_to_yolo(c["bbox"], W, H)
        yolo_lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

    label_file = labels_dir / f"{image_path.stem}.txt"
    label_file.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                          encoding="utf-8")

    if save_vis:
        vis = draw_visualization(img, cells, neck_top, neck_bottom,
                                 manual_neck=is_manual)
        vis_path = vis_dir / f"{image_path.stem}.jpg"
        vis_for_save = vis
        if max(vis.shape[:2]) > 2400:
            scale = 2400 / max(vis.shape[:2])
            vis_for_save = cv2.resize(
                vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(vis_path), vis_for_save,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {
        "n_cells": len(cells),
        "neck_top": neck_top,
        "neck_bottom": neck_bottom,
        "classes": [c["class"] for c in cells],
        "transitions": transitions,
        "img_w": W,
        "img_h": H,
    }


def load_manual_neck(sample_dir, cli_neck):
    """CLI > JSON > None (auto-detect)."""
    if cli_neck is not None:
        return cli_neck
    neck_json = sample_dir / "neck.json"
    if neck_json.exists():
        try:
            data = json.loads(neck_json.read_text(encoding="utf-8"))
            return (int(data["neck_top"]), int(data["neck_bottom"]))
        except Exception as e:
            print(f"  [WARN] could not parse {neck_json}: {e}")
    return None


def process_sample(sample_dir, detect_kwargs, cli_neck=None,
                   save_vis=True, debug=False):
    sample_dir = Path(sample_dir).resolve()
    images_dir = sample_dir / "images"
    labels_dir = sample_dir / "labels"
    vis_dir = sample_dir / "visualization"

    if not images_dir.is_dir():
        print(f"  [SKIP] {sample_dir}: no images/ subdir")
        return None

    labels_dir.mkdir(exist_ok=True)
    if save_vis:
        vis_dir.mkdir(exist_ok=True)

    manual_neck = load_manual_neck(sample_dir, cli_neck)

    image_files = sorted(
        (p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXT),
        key=lambda p: natural_sort_key(p.name),
    )

    print(f"\n[{sample_dir.name}] {len(image_files)} images, "
          + (f"manual neck {manual_neck}" if manual_neck
             else "auto neck per image"))

    class_counts = {0: 0, 1: 0, 2: 0}
    total_cells = 0
    total_empty = 0
    total_transitions = 0
    rows = []

    for img_path in image_files:
        res = process_image(
            img_path, labels_dir, vis_dir,
            manual_neck=manual_neck,
            detect_kwargs=detect_kwargs,
            save_vis=save_vis,
            debug=debug,
        )
        if res is None:
            continue
        total_cells += res["n_cells"]
        total_transitions += res["transitions"]
        if res["n_cells"] == 0:
            total_empty += 1
        for c in res["classes"]:
            class_counts[c] += 1

        rows.append({
            "image": img_path.name,
            "img_w": res["img_w"],
            "img_h": res["img_h"],
            "neck_top": res["neck_top"],
            "neck_bottom": res["neck_bottom"],
            "n_cells": res["n_cells"],
            "n_enter": sum(1 for c in res["classes"] if c == 0),
            "n_in": sum(1 for c in res["classes"] if c == 1),
            "n_exit": sum(1 for c in res["classes"] if c == 2),
            "transitions": res["transitions"],
            "classes": ";".join(str(c) for c in res["classes"]),
        })

        status_str = ",".join(str(c) for c in res["classes"]) or "none"
        trans_tag = f" ({res['transitions']}*)" if res["transitions"] else ""
        print(f"  {img_path.name}: neck={res['neck_top']}-{res['neck_bottom']}, "
              f"cells={res['n_cells']} [{status_str}]{trans_tag}")

    if rows:
        csv_path = sample_dir / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"  -> {total_cells} cells "
          f"({class_counts[0]} enter, {class_counts[1]} in, "
          f"{class_counts[2]} exit), {total_empty} empty, "
          f"{total_transitions} transitions*")

    return {
        "sample": sample_dir.name,
        "n_images": len(image_files),
        "total_cells": total_cells,
        "class_counts": class_counts,
        "total_empty": total_empty,
        "total_transitions": total_transitions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label red blood cells in channel images for YOLO."
    )
    parser.add_argument(
        "path", nargs="?", default=r"G:\23\model",
        help="Single sample dir (default) or parent of sample dirs with --batch"
    )
    parser.add_argument("--batch", action="store_true",
                        help="Treat <path> as parent; process every "
                             "subdirectory that contains an images/ folder")
    parser.add_argument("--neck_top", type=int, default=None)
    parser.add_argument("--neck_bottom", type=int, default=None)
    parser.add_argument("--b_percentile", type=float, default=95.0)
    parser.add_argument("--min_area_ratio", type=float, default=0.001)
    parser.add_argument("--max_area_ratio", type=float, default=0.04)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cli_neck = None
    if args.neck_top is not None and args.neck_bottom is not None:
        cli_neck = (args.neck_top, args.neck_bottom)

    detect_kwargs = {
        "b_percentile": args.b_percentile,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
    }

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"[ERROR] {root} does not exist")
        sys.exit(1)

    if args.batch:
        samples = sorted(
            (d for d in root.iterdir()
             if d.is_dir() and (d / "images").is_dir()),
            key=lambda d: natural_sort_key(d.name),
        )
        if not samples:
            print(f"[ERROR] no sample dirs under {root} "
                  f"(expected subdirs with images/ inside)")
            sys.exit(1)
        print(f"[BATCH] {len(samples)} samples under {root}")
        all_stats = []
        for sdir in samples:
            s = process_sample(sdir, detect_kwargs, cli_neck=cli_neck,
                               save_vis=not args.no_vis, debug=args.debug)
            if s is not None:
                all_stats.append(s)

        if all_stats:
            grand_cells = sum(s["total_cells"] for s in all_stats)
            grand_enter = sum(s["class_counts"][0] for s in all_stats)
            grand_in = sum(s["class_counts"][1] for s in all_stats)
            grand_exit = sum(s["class_counts"][2] for s in all_stats)
            grand_empty = sum(s["total_empty"] for s in all_stats)
            grand_trans = sum(s["total_transitions"] for s in all_stats)
            grand_images = sum(s["n_images"] for s in all_stats)
            print("\n" + "=" * 60)
            print(f"BATCH TOTAL: {len(all_stats)} samples, "
                  f"{grand_images} images, {grand_cells} cells")
            print(f"  0 enter: {grand_enter}")
            print(f"  1 in   : {grand_in}")
            print(f"  2 exit : {grand_exit}")
            print(f"  empty  : {grand_empty}")
            print(f"  transitions*: {grand_trans}")

            batch_csv = root / "batch_summary.csv"
            with batch_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sample", "n_images", "total_cells",
                                 "n_enter", "n_in", "n_exit",
                                 "n_empty", "n_transitions"])
                for s in all_stats:
                    writer.writerow([
                        s["sample"], s["n_images"], s["total_cells"],
                        s["class_counts"][0], s["class_counts"][1],
                        s["class_counts"][2],
                        s["total_empty"], s["total_transitions"],
                    ])
            print(f"  summary: {batch_csv}")
    else:
        process_sample(root, detect_kwargs, cli_neck=cli_neck,
                       save_vis=not args.no_vis, debug=args.debug)


if __name__ == "__main__":
    main()