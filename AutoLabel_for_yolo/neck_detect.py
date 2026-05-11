"""Detect the narrow neck region of a microfluidic channel.

Algorithm (gradient-based, robust across varying channel appearances):
1. Strip any baked-in annotation pixels (yellow text, cyan labels).
2. Apply horizontal Sobel on LAB-L to find vertical edges = channel walls.
3. For each row, pick the strongest negative gradient (left wall) in the
   left half and the strongest positive gradient (right wall) in the right
   half. Require both to exceed a row-noise threshold.
4. Median-filter the left/right wall positions along Y to smooth out
   anomalies (cells in the interior, brief shadows).
5. Width profile = right - left. Neck rows = rows whose smoothed width is
   within neck_ratio × min_width.
6. Return neck_top, neck_bottom = longest contiguous neck segment.

This works whether walls are bright pink on purple (old images) or subtly
brighter on nearly-uniform saturation (new images) because a channel wall
is always a *local edge* regardless of absolute color.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter


def _strip_annotations(image_bgr):
    """Return a copy of the image with yellow/cyan/white annotation pixels
    inpainted from surrounding content.

    Uses cv2.inpaint (Telea algorithm) instead of a single median-BG fill,
    which avoids creating a horizontal dark strip through any cell that
    happens to sit under the annotation text.
    """
    B_ch, G_ch, R_ch = cv2.split(image_bgr)
    anno = (((R_ch > 200) & (G_ch > 180) & (B_ch < 140))
            | ((G_ch > 200) & (B_ch > 200) & (R_ch < 200))
            | ((R_ch > 230) & (G_ch > 230) & (B_ch > 230)))
    if not anno.any():
        return image_bgr
    # Slightly dilate the mask so the text's anti-aliased edges are
    # also inpainted; otherwise a thin colored halo remains.
    mask = anno.astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    # Telea inpainting with small radius is fast and preserves local color
    return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)


def _gradient_wall_positions(image_bgr, min_edge_strength_factor=1.5):
    """Find left and right wall x-positions for every row using horizontal
    gradient.

    Returns:
        (lefts, rights, widths) int32 arrays of length H
        -1 where detection failed for that row
    """
    H, W = image_bgr.shape[:2]

    # LAB-L is robust lightness. Vertical Gaussian blur stabilizes walls
    # (continuous vertically) while preserving horizontal edges.
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    L_smooth = cv2.GaussianBlur(L, (9, 31), 0)

    # Horizontal gradient. Walls are vertical lines = strong ±grad_x.
    grad_x = cv2.Sobel(L_smooth, cv2.CV_32F, 1, 0, ksize=5)

    lefts = np.full(H, -1, dtype=np.int32)
    rights = np.full(H, -1, dtype=np.int32)
    widths = np.full(H, -1, dtype=np.int32)

    mid = W // 2
    overlap = 20  # allow a bit past mid for asymmetric channels

    for y in range(H):
        row = grad_x[y]
        # Left wall: strongest negative gradient in the left half
        left_half = row[:mid + overlap]
        L_x = int(np.argmin(left_half))
        L_val = left_half[L_x]
        # Right wall: strongest positive gradient in the right half
        right_half = row[mid - overlap:]
        R_x = int(np.argmax(right_half)) + (mid - overlap)
        R_val = row[R_x]

        # Both must be stronger than noise in this row
        row_noise = np.std(row) * min_edge_strength_factor
        if L_val < -row_noise and R_val > row_noise and R_x > L_x:
            lefts[y] = L_x
            rights[y] = R_x
            widths[y] = R_x - L_x

    return lefts, rights, widths


def detect_channel_neck(
    image_bgr,
    neck_ratio=1.3,
    smooth_window=31,
    fallback_margin=0.33,
    max_gap_rows=60,
    min_edge_strength_factor=1.5,
    debug=False,
):
    """Detect (neck_top_y, neck_bottom_y) of the vertical channel.

    Returns dict with: neck_top, neck_bottom, lefts, rights, widths
    """
    H, W = image_bgr.shape[:2]

    img_clean = _strip_annotations(image_bgr)

    lefts_raw, rights_raw, widths_raw = _gradient_wall_positions(
        img_clean, min_edge_strength_factor=min_edge_strength_factor
    )

    # Median-filter the wall positions along Y to suppress outliers
    # (cells in the interior, brief shadows). This gives us stable,
    # monotonically-varying wall curves.
    valid = (lefts_raw >= 0).astype(np.float32)
    valid_ratio = float(valid.sum() / H)
    if debug:
        print(f"  [neck] valid_ratio={valid_ratio:.2f} "
              f"({int(valid.sum())}/{H})")

    fallback_top = int(H * fallback_margin)
    fallback_bottom = int(H * (1 - fallback_margin))

    if valid_ratio < 0.30:
        if debug:
            print(f"  [neck] too few valid rows, fallback to "
                  f"({fallback_top}, {fallback_bottom})")
        return {
            "neck_top": fallback_top,
            "neck_bottom": fallback_bottom,
            "widths": widths_raw,
            "lefts": lefts_raw,
            "rights": rights_raw,
        }

    # For smoothing: replace -1 with median so filter doesn't get skewed
    lefts_f = lefts_raw.astype(np.float32)
    rights_f = rights_raw.astype(np.float32)
    med_l = float(np.median(lefts_raw[lefts_raw >= 0]))
    med_r = float(np.median(rights_raw[rights_raw >= 0]))
    lefts_f[lefts_raw < 0] = med_l
    rights_f[rights_raw < 0] = med_r

    lefts_s = median_filter(lefts_f, size=31)
    rights_s = median_filter(rights_f, size=31)
    widths_s = rights_s - lefts_s

    # Light moving-average smoothing of widths for neck detection
    if smooth_window > 1:
        kern = np.ones(smooth_window, dtype=np.float32) / smooth_window
        widths_smooth = np.convolve(widths_s, kern, mode="same")
    else:
        widths_smooth = widths_s

    # Neck = rows where width is close to the minimum
    # Use robust estimate: min of moderately-smoothed widths, ignoring
    # the smallest 10% (which can include wall-detection anomalies)
    # Filter out only actively-valid rows for percentile calc
    widths_for_stats = widths_smooth[widths_smooth > 0]
    if len(widths_for_stats) < 10:
        if debug:
            print(f"  [neck] too few width samples, fallback")
        return {
            "neck_top": fallback_top, "neck_bottom": fallback_bottom,
            "widths": widths_s.astype(np.int32),
            "lefts": lefts_s.astype(np.int32),
            "rights": rights_s.astype(np.int32),
        }
    # 10th percentile gives a robust "typical minimum" excluding
    # anomalous tiny spikes from wall-detection glitches
    robust_min = float(np.percentile(widths_for_stats, 10))
    threshold = robust_min * neck_ratio
    neck_rows = np.where(widths_smooth <= threshold)[0]

    if len(neck_rows) < 5:
        if debug:
            print(f"  [neck] few neck rows, fallback")
        return {
            "neck_top": fallback_top,
            "neck_bottom": fallback_bottom,
            "widths": widths_s.astype(np.int32),
            "lefts": lefts_s.astype(np.int32),
            "rights": rights_s.astype(np.int32),
        }

    # Longest contiguous segment (allow small gaps)
    segments = []
    start = neck_rows[0]
    prev = start
    for y in neck_rows[1:]:
        if y - prev > max_gap_rows:
            segments.append((start, prev))
            start = y
        prev = y
    segments.append((start, prev))
    longest = max(segments, key=lambda s: s[1] - s[0])
    neck_top, neck_bottom = longest

    if debug:
        print(f"  [neck] robust_min={robust_min:.0f}, "
              f"threshold={threshold:.0f}, top={neck_top}, bottom={neck_bottom}, "
              f"segments={len(segments)}")

    return {
        "neck_top": int(neck_top),
        "neck_bottom": int(neck_bottom),
        "widths": widths_s.astype(np.int32),
        "lefts": lefts_s.astype(np.int32),
        "rights": rights_s.astype(np.int32),
    }


def build_channel_interior_mask(image_shape, lefts, rights, margin=0):
    """Build a binary mask = 255 inside the channel, 0 outside.

    Interpolates across invalid rows if any. margin shrinks interior by
    that many pixels on each side (useful when walls are thick).
    """
    H, W = image_shape[:2]
    valid = (lefts >= 0) & (rights >= 0)

    if not valid.any():
        return np.full((H, W), 255, dtype=np.uint8)

    valid_idx = np.where(valid)[0]
    y_arr = np.arange(H)
    lefts_i = np.interp(y_arr, valid_idx, lefts[valid]).astype(np.int32)
    rights_i = np.interp(y_arr, valid_idx, rights[valid]).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        l = max(0, int(lefts_i[y]) + margin)
        r = min(W, int(rights_i[y]) - margin + 1)
        if r > l:
            mask[y, l:r] = 255
    return mask


# Kept for any callers importing it — unused by the new algorithm
def _per_row_channel_width(wall_mask, min_gap=5):
    """Legacy HSV-based helper, kept for compatibility."""
    h, w = wall_mask.shape
    widths = np.full(h, -1, dtype=np.int32)
    lefts = np.full(h, -1, dtype=np.int32)
    rights = np.full(h, -1, dtype=np.int32)
    for y in range(h):
        row = wall_mask[y, :]
        idx = np.where(row > 0)[0]
        if len(idx) < 2:
            continue
        diffs = np.diff(idx)
        breaks = np.where(diffs > min_gap)[0]
        if len(breaks) == 0:
            continue
        left_inner = idx[breaks[0]]
        right_inner = idx[breaks[-1] + 1]
        if right_inner > left_inner:
            widths[y] = right_inner - left_inner
            lefts[y] = left_inner
            rights[y] = right_inner
    return widths, lefts, rights


if __name__ == "__main__":
    import sys
    from pathlib import Path
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("H:/20/model/images")
    for p in sorted(d.iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        print(f"\n=== {p.name}: {img.shape[1]}x{img.shape[0]} ===")
        res = detect_channel_neck(img, debug=True)
        print(f"  neck: {res['neck_top']} -> {res['neck_bottom']}")