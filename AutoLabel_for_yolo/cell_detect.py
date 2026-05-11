"""Cell detection v2: LAB-B threshold + shape filters + watershed for overlap."""
import cv2
import numpy as np


def _compute_shape_metrics(single_mask_u8):
    contours, _ = cv2.findContours(
        single_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(cnt))
    if area == 0:
        return None

    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw == 0 or bh == 0:
        return None

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    fill = area / (bw * bh)
    aspect = min(bw, bh) / max(bw, bh)

    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = x + bw / 2
        cy = y + bh / 2

    return {
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "centroid": (float(cx), float(cy)),
        "area": area,
        "solidity": float(solidity),
        "fill_ratio": float(fill),
        "aspect_ratio": float(aspect),
        "contour": cnt,
    }


def _passes_shape_filter(m, min_ar=0.4, min_fill=0.45, min_solidity=0.80):
    if m is None:
        return False
    if m["aspect_ratio"] < min_ar:
        return False
    if m["fill_ratio"] < min_fill:
        return False
    if m["solidity"] < min_solidity:
        return False
    return True


def _watershed_split(blob_mask_u8, peak_rel_threshold=0.7):
    """Split possibly merged cells in a single blob using watershed.

    Uses the standard OpenCV watershed marker scheme:
      - 1 = known background (outside the blob)
      - 2, 3, ... = sure foreground markers (cell centers from distance-
        transform peaks)
      - 0 = unknown (rest of the blob interior) — watershed fills these in

    Args:
        peak_rel_threshold: only consider distance-transform peaks that are
          at least this fraction of the global maximum. Higher = stricter
          (fewer splits, less over-splitting). 0.7 treats peaks within
          ~30% of the max as legitimate cell centers and ignores smaller
          bumps from biconcave/irregular single-cell shapes.
    """
    dist = cv2.distanceTransform(blob_mask_u8, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return [blob_mask_u8]

    # Local maxima of distance transform = cell centers
    min_distance = max(5, int(dist.max() * 0.6))
    kernel = np.ones((min_distance * 2 + 1, min_distance * 2 + 1), np.uint8)
    dilated = cv2.dilate(dist, kernel)
    peaks = ((dist == dilated) & (dist > dist.max() * peak_rel_threshold)
             & (blob_mask_u8 > 0)).astype(np.uint8)
    # Grow each peak into a small region so watershed has real seeds
    peaks = cv2.dilate(peaks, np.ones((5, 5), np.uint8))
    peaks = cv2.bitwise_and(peaks, blob_mask_u8)

    n_peaks, peak_labels = cv2.connectedComponents(peaks)
    if n_peaks <= 2:  # 0 + 1 peak = no split needed
        return [blob_mask_u8]

    # Build markers:
    #   outside blob -> 1 (known background)
    #   peak regions -> 2, 3, ... (cell centers)
    #   rest of blob -> 0 (unknown)
    markers = np.zeros(blob_mask_u8.shape, dtype=np.int32)
    markers[blob_mask_u8 == 0] = 1
    mask_peaks = peaks > 0
    offset_pl = peak_labels.copy()
    offset_pl[offset_pl > 0] += 1
    markers[mask_peaks] = offset_pl[mask_peaks]

    rgb = cv2.cvtColor(blob_mask_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(rgb, markers)

    splits = []
    for lv in np.unique(markers):
        if lv <= 1:  # skip bg and watershed boundary (-1)
            continue
        part = ((markers == lv) & (blob_mask_u8 > 0)).astype(np.uint8) * 255
        if part.sum() < 100:  # skip tiny fragments
            continue
        splits.append(part)

    return splits if len(splits) > 1 else [blob_mask_u8]


def _annotation_overlay_mask(image_bgr):
    """Return a mask (H,W) uint8 = 255 on pixels that look like yellow/cyan
    overlay text (very bright, saturated non-magenta).

    The tool's own visualizations draw yellow neck lines and colored bbox
    text. If an already-annotated image is fed back in (easy to do by
    accident), those pixels have high LAB-B and would otherwise be
    mistaken for cells. We detect them in BGR (pure yellow R&G high, B
    low; pure cyan G&B high, R low) and mark them as non-cell.
    """
    B, G, R = cv2.split(image_bgr)
    # Yellow text: R high, G high, B low
    yellow = (R > 200) & (G > 180) & (B < 140)
    # Cyan/white-ish crosshairs: G & B high
    cyan = (G > 200) & (B > 200) & (R < 200)
    # Very bright pixels (labels on dark background): all high
    bright = (R > 230) & (G > 230) & (B > 230)
    return ((yellow | cyan | bright).astype(np.uint8)) * 255


def _merge_colinear_fragments(blobs_with_metrics,
                              max_gap_ratio=0.4,
                              x_overlap_ratio=0.5,
                              combined_ar_max=3.0,
                              min_ar_for_standalone=0.4,
                              min_fill_for_standalone=0.5,
                              debug=False):
    """Merge pairs of blobs that look like fragments of one motion-blurred
    cell (e.g. a streak split by a brief signal dip).

    Only merges when BOTH pieces individually look like fragments rather
    than complete cells — i.e. each has poor aspect ratio or low fill.
    Two genuinely separate round cells stacked vertically will have good
    shape metrics individually, so they won't be merged.

    A pair is merged when ALL of these are true:
      - both blobs individually have poor shape (AR or fill too low)
      - bboxes have significant horizontal overlap (same X track)
      - vertical gap between them is smaller than max_gap_ratio × avg
        bbox height
      - the combined region still has reasonable aspect ratio
    """
    if len(blobs_with_metrics) < 2:
        return blobs_with_metrics

    def looks_fragment(m):
        """A fragment has poor standalone shape. A round intact cell has
        AR>=0.4 and fill>=0.5; anything failing these likely isn't a
        complete cell by itself."""
        return (m["aspect_ratio"] < min_ar_for_standalone
                or m["fill_ratio"] < min_fill_for_standalone)

    items = []
    for single, m in blobs_with_metrics:
        x, y, w, h = m["bbox"]
        items.append({
            "mask": single, "metrics": m,
            "x": x, "y": y, "w": w, "h": h,
            "merged": False,
        })
    items.sort(key=lambda it: it["y"])

    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            if items[i]["merged"]:
                continue
            a = items[i]
            # Skip if this blob is already a nice cell by itself
            if not looks_fragment(a["metrics"]):
                continue
            for j in range(i + 1, len(items)):
                if items[j]["merged"]:
                    continue
                b = items[j]
                if not looks_fragment(b["metrics"]):
                    continue

                # horizontal overlap
                x_overlap = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                min_w = min(a["w"], b["w"])
                if x_overlap < x_overlap_ratio * min_w:
                    continue

                # vertical gap
                a_bottom = a["y"] + a["h"]
                b_top = b["y"]
                vgap = max(0, b_top - a_bottom)
                avg_h = (a["h"] + b["h"]) / 2
                if vgap > max_gap_ratio * avg_h:
                    continue

                merged_mask = cv2.bitwise_or(a["mask"], b["mask"])
                mm = _compute_shape_metrics(merged_mask)
                if mm is None:
                    continue
                mx, my, mw, mh = mm["bbox"]
                combined_ar = max(mw, mh) / max(min(mw, mh), 1)
                if combined_ar > combined_ar_max:
                    continue

                if debug:
                    print(f"    merge fragments: "
                          f"({a['x']},{a['y']} {a['w']}x{a['h']}, "
                          f"AR={a['metrics']['aspect_ratio']:.2f}, "
                          f"fill={a['metrics']['fill_ratio']:.2f}) + "
                          f"({b['x']},{b['y']} {b['w']}x{b['h']}, "
                          f"AR={b['metrics']['aspect_ratio']:.2f}, "
                          f"fill={b['metrics']['fill_ratio']:.2f}) "
                          f"-> ({mx},{my} {mw}x{mh})")
                a["mask"] = merged_mask
                a["metrics"] = mm
                a["x"], a["y"], a["w"], a["h"] = mx, my, mw, mh
                b["merged"] = True
                changed = True
                break
            if changed:
                break

    return [(it["mask"], it["metrics"]) for it in items if not it["merged"]]


def _non_max_suppression(cells, iou_threshold=0.3, contain_threshold=0.7):
    """Remove duplicate/nested bounding boxes.

    Drops cell B if either:
      - its IoU with a kept cell A > iou_threshold
      - it is mostly contained in a kept cell A (overlap / B_area > contain_threshold)

    Keeps larger-area cells first.
    """
    if not cells:
        return cells
    order = sorted(range(len(cells)), key=lambda i: -cells[i]["area"])
    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        c = cells[idx]
        keep.append(c)
        x1, y1, w1, h1 = c["bbox"]
        a1 = w1 * h1
        for j in order:
            if j == idx or j in suppressed:
                continue
            c2 = cells[j]
            x2, y2, w2, h2 = c2["bbox"]
            a2 = w2 * h2
            ix1 = max(x1, x2); iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2); iy2 = min(y1 + h1, y2 + h2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            iou = inter / (a1 + a2 - inter)
            contain = inter / a2  # how much of c2 is in c
            if iou > iou_threshold or contain > contain_threshold:
                suppressed.add(j)
    return keep


def detect_cells(
    image_bgr,
    min_area_ratio=0.001,
    max_area_ratio=0.04,
    b_percentile=95.0,
    min_aspect_ratio=0.4,
    min_fill_ratio=0.45,
    min_solidity=0.75,
    # Relaxed filters for compressed cells inside the neck region
    neck_min_aspect_ratio=0.15,
    neck_min_fill_ratio=0.35,
    neck_min_solidity=0.75,
    split_solidity_th=0.85,
    split_area_multiplier=1.5,
    # Spatial constraints
    channel_mask=None,  # optional (H,W) uint8: 255=inside channel, 0=outside
    neck_y_range=None,  # optional (neck_top, neck_bottom) for adaptive filter
    morph_kernel=5,  # morphology kernel size
    debug=False,
):
    """Detect individual red blood cells.

    When channel_mask is provided, the thresholded mask is intersected with
    it BEFORE morphological closing. This prevents out-of-channel noise
    (reflections, artifacts) from being detected AND prevents closing from
    bridging cells to wall fragments or annotation lines.

    When neck_y_range is provided, blobs whose centroid falls within that
    Y-range are evaluated with the looser `neck_*` thresholds — compressed
    cells in the narrow neck legitimately have lower aspect ratios and
    fill ratios than free cells.
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w

    # First, inpaint any baked-in annotation pixels (yellow/cyan/white text)
    # BEFORE computing LAB. This prevents two problems:
    #  (1) annotation pixels having high LAB-B, polluting cell detection
    #  (2) a cell sitting under annotation text being split into pieces
    #      by the annotation's color interfering with the cell signal
    anno_mask = _annotation_overlay_mask(image_bgr)
    if anno_mask.any():
        # Dilate slightly to cover anti-aliased edges
        dil = cv2.dilate(anno_mask, np.ones((3, 3), np.uint8))
        image_bgr = cv2.inpaint(image_bgr, dil, 3, cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    B = lab[:, :, 2]

    threshold = max(np.percentile(B, b_percentile), B.mean() + B.std() * 1.2)
    raw_mask = (B > threshold).astype(np.uint8) * 255

    # (Annotation already inpainted above, so no need to mask raw_mask)

    # Apply channel interior mask BEFORE morphology, so closing doesn't
    # bridge cell pixels to wall / annotation artifacts outside channel
    if channel_mask is not None:
        raw_mask = cv2.bitwise_and(raw_mask, channel_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Vertical-only close kernel: fuses vertically-stacked fragments
    # separated by a SHORT gap (~5 pixels), as happens with motion-
    # blurred cells where the LAB-B signal has a brief dip mid-streak.
    # Use a tall thin RECT kernel so it closes short vertical gaps
    # without bridging horizontally-separated cells. Keep it short (3x7)
    # so it doesn't fuse two actual stacked cells with a real 15+ pixel
    # gap between them.
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_vert)

    # Fill interior holes (biconcave center)
    flood = mask.copy()
    fh, fw = flood.shape
    ff_mask = np.zeros((fh + 2, fw + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    mask_filled = cv2.bitwise_or(mask, holes)

    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask_filled, 8)

    def _thresholds_for_metrics(m):
        """Return (min_ar, min_fill, min_sol) based on location.

        Use relaxed neck thresholds when the blob's bbox overlaps the
        neck region (even partially) — cells entering the taper or
        crossing the boundary are legitimately elongated.
        """
        if neck_y_range is not None:
            nt, nb = neck_y_range
            by = m["bbox"][1]
            by2 = by + m["bbox"][3]
            # Buffer zone: also relax filter within 15% of neck length
            # above neck_top (taper region leading into neck)
            buffer = max(30, int((nb - nt) * 0.15))
            if by2 >= (nt - buffer) and by <= nb:
                return (neck_min_aspect_ratio, neck_min_fill_ratio,
                        neck_min_solidity)
        return (min_aspect_ratio, min_fill_ratio, min_solidity)

    # First pass: collect all blobs that meet the area bound, without
    # shape filtering yet. This gives us the raw "connected regions with
    # cell-like color signal" which we'll try to merge before filtering.
    raw_blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        single = (labels == i).astype(np.uint8) * 255
        m = _compute_shape_metrics(single)
        if m is None:
            continue
        raw_blobs.append((single, m))

    # Merge motion-blur fragments: pairs that are co-linear vertically
    # with small gap look like one cell broken by a brief signal dip.
    raw_blobs = _merge_colinear_fragments(raw_blobs, debug=debug)

    # Now apply shape filtering
    initial_blobs = []
    big_blobs = []
    for single, m in raw_blobs:
        area = m["area"]
        cy = m["centroid"][1]
        ar_th, fill_th, sol_th = _thresholds_for_metrics(m)

        if _passes_shape_filter(m, ar_th, fill_th, sol_th):
            initial_blobs.append((single, m))
        else:
            # Candidate for watershed splitting if its shape is borderline
            if m["aspect_ratio"] >= ar_th * 0.7 and m["fill_ratio"] >= 0.3:
                big_blobs.append((single, m))
            elif debug:
                print(f"    reject blob area={area} cy={cy:.0f} "
                      f"sol={m['solidity']:.2f} AR={m['aspect_ratio']:.2f} "
                      f"fill={m['fill_ratio']:.2f} "
                      f"(thresholds AR>={ar_th}, fill>={fill_th}, "
                      f"sol>={sol_th})")

    median_area = (float(np.median([m["area"] for _, m in initial_blobs]))
                   if initial_blobs else None)

    more_blobs = []
    still_good = []
    for single, m in initial_blobs:
        too_big = (median_area is not None
                   and m["area"] > median_area * split_area_multiplier)
        # A single cell typically has fill_ratio > 0.60. Two overlapping
        # circles packed into a bbox leave lots of empty corner space
        # (fill drops to ~0.45-0.55) AND have low convex-hull solidity.
        # Use both signals. Biconcave single cells still have relatively
        # high fill from the interior-hole flood fill.
        looks_merged = m["fill_ratio"] < 0.60 and m["solidity"] < split_solidity_th

        if not (too_big or looks_merged):
            still_good.append((single, m))
            continue

        splits = _watershed_split(single)
        if len(splits) <= 1:
            still_good.append((single, m))
            continue

        # Evaluate each candidate split piece
        passing = []
        for s in splits:
            sm = _compute_shape_metrics(s)
            if sm is None:
                continue
            ar_th, fill_th, sol_th = _thresholds_for_metrics(sm)
            if _passes_shape_filter(sm, ar_th, fill_th, sol_th * 0.9):
                passing.append((s, sm))

        # Size sanity check: pieces produced by splitting a single
        # irregular cell are typically MUCH smaller than a real cell
        # (e.g. 30-50% of median size). Pieces from splitting two
        # truly overlapping cells are closer to median size (60%+).
        # Reject the split if any piece is suspiciously small.
        min_piece_area_ratio = 0.55  # piece must be >=55% of median
        if median_area is not None and passing:
            min_piece_area = median_area * min_piece_area_ratio
            too_small = [p for p, pm in passing if pm["area"] < min_piece_area]
            if too_small:
                if debug:
                    sizes = [pm["area"] for _, pm in passing]
                    print(f"    reject split: piece sizes {sizes} vs "
                          f"median {median_area:.0f} (min_piece_area="
                          f"{min_piece_area:.0f}) -> keep original")
                still_good.append((single, m))
                continue

        if passing:
            more_blobs.extend(passing)
            continue

        # Split produced no valid pieces -> keep original
        if debug:
            print(f"    watershed split {len(splits)} pieces, "
                  f"none passed filter -> keep original")
        still_good.append((single, m))

    for single, m in big_blobs:
        splits = _watershed_split(single)
        for s in splits:
            sm = _compute_shape_metrics(s)
            if sm is None:
                continue
            ar_th, fill_th, sol_th = _thresholds_for_metrics(sm)
            if _passes_shape_filter(sm, ar_th, fill_th, sol_th * 0.9):
                more_blobs.append((s, sm))

    all_blobs = still_good + more_blobs

    cells = []
    for single, m in all_blobs:
        cells.append({
            "bbox": m["bbox"],
            "centroid": m["centroid"],
            "area": m["area"],
            "solidity": m["solidity"],
            "aspect_ratio": m["aspect_ratio"],
            "fill_ratio": m["fill_ratio"],
            "mask": single,
        })

    # Final NMS to guard against any nested/duplicate bboxes that might
    # slip through (e.g. watershed producing a piece overlapping the
    # original, or multi-pass detection creating duplicates)
    cells = _non_max_suppression(cells)

    if debug:
        print(f"  threshold={threshold:.1f}, "
              f"initial good={len(initial_blobs)}, big={len(big_blobs)}, "
              f"final={len(cells)}")

    return cells, mask_filled


if __name__ == "__main__":
    from pathlib import Path
    imgs = Path("/home/claude/test_sample/images")
    for p in sorted(imgs.glob("*.png")):
        img = cv2.imread(str(p))
        print(f"\n=== {p.name}: {img.shape[1]}x{img.shape[0]} ===")
        cells, _ = detect_cells(img, debug=True)
        for c in cells:
            print(f"  cell: bbox={c['bbox']}, cent=({c['centroid'][0]:.0f},"
                  f"{c['centroid'][1]:.0f}), area={c['area']}, "
                  f"sol={c['solidity']:.2f}, AR={c['aspect_ratio']:.2f}, "
                  f"fill={c['fill_ratio']:.2f}")