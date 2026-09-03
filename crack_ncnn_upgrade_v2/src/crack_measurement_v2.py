import math
import cv2
import numpy as np


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _skeletonize(binary_u8):
    src = (binary_u8 > 0).astype(np.uint8) * 255
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(src)
    except Exception:
        pass

    img = src.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
    return skel


def _skeleton_length_px(skel_u8):
    """8-connected skeleton length without double-counting 2x2 stair-step diagonals."""
    s = skel_u8 > 0
    if not np.any(s):
        return 0.0

    horizontal = np.count_nonzero(s[:, :-1] & s[:, 1:])
    vertical = np.count_nonzero(s[:-1, :] & s[1:, :])

    # Count a diagonal only when there is no orthogonal intermediate pixel.
    # This prevents thick/stair-step skeletons from creating triangle edges.
    d1 = s[:-1, :-1] & s[1:, 1:]
    d1 &= ~(s[:-1, 1:] | s[1:, :-1])
    d2 = s[:-1, 1:] & s[1:, :-1]
    d2 &= ~(s[:-1, :-1] | s[1:, 1:])
    diagonal = np.count_nonzero(d1) + np.count_nonzero(d2)

    return float(horizontal + vertical + math.sqrt(2.0) * diagonal)


def _endpoint_count(skel_u8):
    s = (skel_u8 > 0).astype(np.uint8)
    if not np.any(s):
        return 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(s, -1, kernel, borderType=cv2.BORDER_CONSTANT) - s
    ep = ((s > 0) & (neighbors == 1)).astype(np.uint8) * 255
    n, _, _, _ = cv2.connectedComponentsWithStats(ep, connectivity=8)
    return max(int(n) - 1, 0)


def _branchpoint_count(skel_u8):
    """Count clusters of true branch pixels (degree >= 3)."""
    s = (skel_u8 > 0).astype(np.uint8)
    if not np.any(s):
        return 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(s, -1, kernel, borderType=cv2.BORDER_CONSTANT) - s
    bp = ((s > 0) & (neighbors >= 3)).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(bp, connectivity=8)
    count = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 1:
            count += 1
    return count


def _orientation_from_points(xs, ys, cfg):
    if len(xs) < 2:
        return "unknown", 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, int(np.argmax(vals))]
    angle = abs(math.degrees(math.atan2(float(principal[1]), float(principal[0]))))
    if angle > 90.0:
        angle = 180.0 - angle

    h_deg = float(cfg.get("orientation_horizontal_deg", 22.5))
    v_deg = float(cfg.get("orientation_vertical_deg", 67.5))
    if angle <= h_deg:
        ori = "horizontal"
    elif angle >= v_deg:
        ori = "vertical"
    else:
        ori = "diagonal"
    return ori, angle


def _component_angle(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 2:
        return None
    _, angle = _orientation_from_points(xs, ys, {})
    return angle


def _angle_diff(a, b):
    if a is None or b is None:
        return 0.0
    d = abs(float(a) - float(b))
    return min(d, 90.0 - d if d > 45.0 else d)


def _bbox_gap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ar, ab = ax + aw - 1, ay + ah - 1
    br, bb = bx + bw - 1, by + bh - 1
    dx = max(bx - ar - 1, ax - br - 1, 0)
    dy = max(by - ab - 1, ay - bb - 1, 0)
    return math.hypot(dx, dy)


def _clean_and_group_small(mask_small, cfg):
    """
    Work only at model resolution (160x160):
      1) bridge tiny segmentation gaps
      2) discard tiny noise
      3) merge nearby/aligned fragments into one physical crack group
    """
    binary = (mask_small > 0).astype(np.uint8) * 255

    k = int(cfg.get("model_close_kernel", 3))
    if k >= 3:
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(cfg.get("model_close_iterations", 1)),
        )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(cfg.get("min_model_component_area_px", 3))

    comps = []
    clean = np.zeros_like(binary)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cmask = (labels == label).astype(np.uint8) * 255
        clean[cmask > 0] = 255
        cskel = _skeletonize(cmask)
        c_branchpoints = _branchpoint_count(cskel)
        c_endpoints = _endpoint_count(cskel)
        comps.append({
            "mask": cmask,
            "bbox": (x, y, w, h),
            "area": area,
            "angle": _component_angle(cmask),
            # Branching hint is computed BEFORE grouping, so two nearby
            # disconnected fragments do not become "branching" merely
            # because the merged group has four endpoints.
            "branching_hint": (c_branchpoints >= 1 and c_endpoints >= 3),
        })

    if not comps:
        return clean, []

    gap = float(cfg.get("group_merge_gap_model_px", 3.0))
    angle_tol = float(cfg.get("group_merge_angle_deg", 35.0))
    small_fragment_area = int(cfg.get("group_merge_small_fragment_area_px", 12))
    uf = _UnionFind(len(comps))

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _bbox_gap(comps[i]["bbox"], comps[j]["bbox"]) > gap:
                continue
            aligned = _angle_diff(comps[i]["angle"], comps[j]["angle"]) <= angle_tol
            tiny_fragment = min(comps[i]["area"], comps[j]["area"]) <= small_fragment_area
            if aligned or tiny_fragment:
                uf.union(i, j)

    buckets = {}
    for i in range(len(comps)):
        buckets.setdefault(uf.find(i), []).append(i)

    groups = []
    for members in buckets.values():
        gmask = np.zeros_like(binary)
        for i in members:
            gmask[comps[i]["mask"] > 0] = 255
        ys, xs = np.where(gmask > 0)
        if not len(xs):
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        groups.append({
            "mask": gmask,
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "source_components": len(members),
            "branching_hint": any(comps[i]["branching_hint"] for i in members),
        })

    return clean, groups


def _map_model_bbox_to_frame(bbox, model_w, model_h, frame_w, frame_h, margin_model_px):
    x, y, w, h = bbox
    x0m = max(x - margin_model_px, 0)
    y0m = max(y - margin_model_px, 0)
    x1m = min(x + w + margin_model_px, model_w)
    y1m = min(y + h + margin_model_px, model_h)

    x0 = max(int(math.floor(x0m * frame_w / model_w)), 0)
    y0 = max(int(math.floor(y0m * frame_h / model_h)), 0)
    x1 = min(int(math.ceil(x1m * frame_w / model_w)), frame_w)
    y1 = min(int(math.ceil(y1m * frame_h / model_h)), frame_h)
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def _refine_original_roi(gray_roi, guide_small_crop, out_w, out_h, cfg):
    """
    Refine crack pixels from the ORIGINAL image, but only inside a guided ROI.
    This preserves 160x160 NCNN speed while measuring width/length using 640x480 pixels.
    """
    guide_seed = cv2.resize(
        guide_small_crop,
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    )

    dilate_px = int(cfg.get("guide_dilate_px", 5))
    if dilate_px > 0:
        dk = 2 * dilate_px + 1
        guide = cv2.dilate(
            guide_seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)),
            iterations=1,
        )
    else:
        guide = guide_seed

    sigma = float(cfg.get("refine_background_sigma", 3.0))
    background = cv2.GaussianBlur(gray_roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Dark cracks become positive values.
    darkness = cv2.subtract(background, gray_roi)

    vals = darkness[guide > 0]
    if vals.size == 0:
        return guide_seed, True

    percentile = float(cfg.get("refine_darkness_percentile", 65.0))
    min_dark = float(cfg.get("refine_min_darkness", 5.0))
    thr = max(min_dark, float(np.percentile(vals, percentile)))
    refined = ((darkness >= thr) & (guide > 0)).astype(np.uint8) * 255

    close_k = int(cfg.get("refine_close_kernel", 3))
    if close_k >= 3:
        if close_k % 2 == 0:
            close_k += 1
        refined = cv2.morphologyEx(
            refined,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=1,
        )

    # Keep only refined regions that overlap/are close to the NCNN seed.
    seed_near = cv2.dilate(
        guide_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    min_area = int(cfg.get("min_refined_component_area_px", 12))
    filtered = np.zeros_like(refined)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == i
        if np.any(seed_near[comp] > 0):
            filtered[comp] = 255

    min_total = int(cfg.get("min_refined_total_area_px", 18))
    if cv2.countNonZero(filtered) < min_total:
        # Refinement can fail on unusual lighting. Fall back to the model guide
        # rather than deleting a real model detection.
        return guide_seed, True
    return filtered, False


def measure_cracks_from_small_mask(frame_bgr, mask_small, mm_per_px, cfg):
    """
    V2 measurement pipeline.

    - grouping/noise removal happens at 160x160
    - ONLY grouped ROIs are mapped to the original 640x480 frame
    - width/length are measured inside those original-resolution ROIs
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    model_h, model_w = mask_small.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    _, groups = _clean_and_group_small(mask_small, cfg)
    if not groups:
        return []

    pctl = float(cfg.get("width_percentile", 95.0))
    margin_model_px = int(cfg.get("roi_margin_model_px", 3))
    min_length_px = float(cfg.get("min_measured_length_px", 12.0))
    branch_min = int(cfg.get("branch_min_branchpoints", 1))

    features = []
    for group in groups:
        gx, gy, gw, gh = group["bbox"]
        x, y, w, h = _map_model_bbox_to_frame(
            group["bbox"], model_w, model_h, frame_w, frame_h, margin_model_px
        )

        roi_gray = gray[y:y + h, x:x + w]

        # Crop the group's low-res guide with the SAME model-space margin used
        # to create the full-resolution ROI.
        x0m = max(gx - margin_model_px, 0)
        y0m = max(gy - margin_model_px, 0)
        x1m = min(gx + gw + margin_model_px, model_w)
        y1m = min(gy + gh + margin_model_px, model_h)
        guide_crop = group["mask"][y0m:y1m, x0m:x1m]

        refined, fallback = _refine_original_roi(
            roi_gray, guide_crop, w, h, cfg
        )

        # Tight crop after original-resolution refinement => faster skeleton.
        ry, rx = np.where(refined > 0)
        if not len(rx):
            continue
        tx0, tx1 = int(rx.min()), int(rx.max()) + 1
        ty0, ty1 = int(ry.min()), int(ry.max()) + 1
        tight = refined[ty0:ty1, tx0:tx1]

        skel = _skeletonize(tight)
        dist = cv2.distanceTransform(tight, cv2.DIST_L2, 5)
        sy, sx = np.where(skel > 0)
        if not len(sx):
            continue

        length_px = _skeleton_length_px(skel)
        if length_px < min_length_px:
            continue

        local_widths = 2.0 * dist[sy, sx]
        width_px = float(np.percentile(local_widths, pctl))
        max_width_px = float(np.max(local_widths))

        orientation, angle_deg = _orientation_from_points(sx, sy, cfg)
        branchpoints = _branchpoint_count(skel)
        endpoints = _endpoint_count(skel)
        # Require both original model-space branching evidence and a real
        # branch in the refined skeleton. This reduces false branching from
        # grouped fragments or digital stair-step artifacts.
        if group.get("branching_hint", False) and branchpoints >= branch_min and endpoints >= 3:
            orientation = "branching"

        area_px = int(cv2.countNonZero(tight))
        full_x = x + tx0
        full_y = y + ty0
        tight_h, tight_w = tight.shape[:2]

        features.append({
            "bbox": (full_x, full_y, tight_w, tight_h),
            "area_px": area_px,
            "area_mm2": float(area_px) * float(mm_per_px) ** 2,
            "length_px": length_px,
            "length_mm": length_px * float(mm_per_px),
            "width_px": width_px,
            "width_mm": width_px * float(mm_per_px),
            "max_width_px": max_width_px,
            "max_width_mm": max_width_px * float(mm_per_px),
            "orientation": orientation,
            "angle_deg": angle_deg,
            "branchpoint_count": branchpoints,
            "endpoint_count": endpoints,
            "source_components": int(group["source_components"]),
            "refine_fallback": bool(fallback),
        })

    features.sort(key=lambda f: (f["width_mm"], f["length_mm"]), reverse=True)
    return features
