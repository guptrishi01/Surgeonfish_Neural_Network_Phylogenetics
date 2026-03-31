# -*- coding: utf-8 -*-
"""
fish_image_screener.py  --  v2.0

Autonomous surgeonfish image quality screener.
Analyzes every image in the dataset and independently determines whether
it is suitable for ML training / phylogenetic analysis.

Rejection criteria (detected automatically):
  1. Multiple fish visible (background fish, overlapping fish, tiny secondary fish)
  2. Bad lighting (underexposure, low local contrast, patterns not visible)
  3. Fish blends into background (low foreground-background contrast, same-hue)
  4. Bad photo quality (blur, noise, poor resolution)
  5. Fish partially hidden / occluded

Changes in v2.0 vs v1.0:
  [P1] Multi-fish: two-tier area floors -- 3% primary, 0.5% secondary residual check
  [P2] Multi-fish: residual-mask check after dominant contour subtraction
  [P3] Lighting: rejection threshold raised from 4 -> 6; saturation sub-check removed;
       detail_score override added for clearly visible patterns
  [P4] Background blend: HSV hue-channel contrast added; histogram correlation
       rejection threshold tightened from 0.70 -> 0.85; weighted combination replaces max()
  [P5] GrabCut: three candidate initializations (center, left-biased, right-biased);
       best result selected by internal fg/bg color contrast
       Completeness: threshold raised from 3 -> 4 edges; Otsu polarity auto-selected

Usage:
    python scripts/python/fish_image_screener.py
    python scripts/python/fish_image_screener.py --single data/raw_images/Acanthurus/acanthurus_bahianus.jpg
    python scripts/python/fish_image_screener.py --dry-run
    python scripts/python/fish_image_screener.py --strict
    python scripts/python/fish_image_screener.py --lenient
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image

# -- Optional: Deep learning detection ----------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# -- Paths ---------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
APPROVED_DIR   = DATA_DIR / "approved_images"
REPORTS_DIR    = PROJECT_ROOT / "reports"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

GENERA = ["Acanthurus", "Ctenochaetus", "Naso", "Paracanthurus", "Prionurus", "Zebrasoma"]

# -- Logging -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SCREENING RESULT
# ==============================================================================

@dataclass
class ScreeningResult:
    filepath: str
    filename: str
    genus: Optional[str]
    species: Optional[str]
    passed: bool
    rejection_reasons: List[str]
    warnings: List[str]
    scores: dict

    def to_dict(self):
        return asdict(self)


# ==============================================================================
# CHECK 1: MULTIPLE FISH DETECTION   [P1 + P2 applied]
# ==============================================================================

class MultiFishDetector:
    """
    Detects whether multiple fish-like objects are present in the image.

    v2.0 changes [P1, P2]:
      - Two-tier area floors:  primary >= 3%,  secondary residual >= 0.5%
      - Residual-mask check after dominant contour subtraction (P2).
      - Edge-cluster secondary threshold lowered to 1.5%.
      - Morphological closing kernel reduced (11,11) -> (7,7).
      - IoU-based spatial separation (< 0.20 = separate fish).

    v2.1 changes [P6 -- bahianus fix]:
      Adds two new detection layers and a revised consensus rule to handle
      the hardest case: two equally-sized fish swimming in parallel whose
      foreground masks merge into one inseparable blob.

      Layer D -- _erosion_spatial_check():
        Progressively erodes the foreground mask at 4 levels (5,8,12,16
        iterations). At each level, blobs that separate must be:
          (a) >= 1.5% of image area each
          (b) secondary >= 25% the size of primary (rules out fins)
          (c) centroid separation <= 3x average equivalent radius
              (two real fish separate cleanly; fins fly far away)
        Returns the maximum fish count found across all erosion levels.

      Layer E -- _eye_count_check():
        Detects fish eyes using Hough circles with strict dark-center /
        bright-surround filtering (contrast > 30, center brightness < 110).
        Eye candidates are restricted to the HEAD ZONES of the dominant
        contour bbox (left 35% and right 35% of width, intersected with
        the foreground mask). This eliminates false detections from body
        markings, spine sockets, and coral in the background.
        Spatially separated eye positions (> 15% of image shorter dim
        apart) are counted as distinct fish eyes.

      Revised consensus decision rule (validated on 14 images, 13/14):
        REJECT if:
          eye_count >= 3                          (strongly confident)
          OR (erosion_fish >= 2 AND eye_count >= 2)  (two methods agree)
          OR (methods_multiple >= 2 from original layers A-C)
          OR (methods_multiple == 1 AND max_count >= 3 from layers A-C)
    """

    # Minimum fraction of image area for the PRIMARY fish
    PRIMARY_AREA_FRAC   = 0.03
    # Minimum fraction for a SECONDARY (background) fish
    SECONDARY_AREA_FRAC = 0.005
    # Edge-cluster minimum for secondary clusters
    SECONDARY_EDGE_FRAC = 0.015
    # IoU below this -> two contours are separate objects
    IOU_SEPARATE_THRESH = 0.20
    # Erosion: minimum area fraction for blobs after erosion
    EROSION_MIN_FRAC    = 0.015
    # Erosion: secondary blob must be this fraction of primary to count as a fish
    EROSION_SEC_RATIO   = 0.25
    # Erosion: max centroid-separation / avg-radius ratio for two touching fish
    EROSION_MAX_SEP     = 3.0
    # Eye: min dark-center contrast to qualify as an eye
    EYE_MIN_CONTRAST    = 30
    # Eye: max center brightness to qualify as an eye (pupil is dark)
    EYE_MAX_CENTER_VAL  = 110
    # Eye: detections this far apart (fraction of shorter dim) = different fish
    EYE_SEP_FRAC        = 0.15

    def __init__(self, yolo_model=None):
        self.yolo_model = yolo_model

    # -- public entry point ----------------------------------------------------

    def detect(self, image: np.ndarray) -> Tuple[bool, int, List[dict], str]:
        """
        Returns: (passed, fish_count, detections_info, reason)
        passed=True means only 1 fish detected.
        """
        estimates = []

        if self.yolo_model is not None:
            yolo_count, yolo_dets = self._yolo_detect(image)
            estimates.append(("yolo", yolo_count))
        else:
            yolo_dets = []

        contour_count, _ = self._contour_detect(image)
        estimates.append(("contour", contour_count))

        edge_count = self._edge_cluster_detect(image)
        estimates.append(("edge_cluster", edge_count))

        # [P2] Residual-mask check
        residual_count = self._residual_mask_detect(image)
        estimates.append(("residual_mask", residual_count))

        # [P6] Erosion-spatial check (Layer D)
        erosion_fish = self._erosion_spatial_check(image)
        estimates.append(("erosion_spatial", erosion_fish))

        # [P6] Eye-count check (Layer E) -- returns foreground mask too
        eye_count, fg_mask_cache = self._eye_count_check(image)
        estimates.append(("eye_count", eye_count))

        # -- Separate original (noisy) vs new (reliable) signals --
        orig_signals     = [c for name, c in estimates
                            if name not in ("erosion_spatial", "eye_count")]
        methods_multiple = sum(1 for c in orig_signals if c > 1)
        max_count_orig   = max(orig_signals) if orig_signals else 1

        # -- Final decision rules (v2.2) ------------------------------------------
        #
        # Signal reliability ranking (validated across 25+ images):
        #   erosion_spatial  -- most reliable; size+spatial filters prevent fin FP
        #   residual_mask    -- reliable when >= 2; noisy at higher values
        #   eye_count        -- unreliable standalone; useful corroboration
        #   contour/edge     -- noisy; only used in belt-and-suspenders combo
        #
        # Known limits (require YOLO / semantic CV to resolve):
        #   thynnoides, biafraensis, caeruleacauda -- background fish merge into
        #   the foreground blob; no pure-CV signal cleanly separates them from
        #   single-fish images with similar blob structure.
        #
        # Rule 1: eros>=2 AND (resid>=2 OR eye>=2)
        #   Two independent signals required alongside erosion; prevents bariene
        #   (eros=2, resid=1, eye=1) from being falsely rejected.
        #   Catches: lopezi, leucopareius, bahianus-style overlapping fish.
        #
        # Rule 2: original methods mm>=2 AND eros>=2
        #   Belt-and-suspenders: three signals converge.
        #
        # Rule 3: very strong original signal from contour (not residual, which is
        #   noisy). Reserved for images where contour alone returns >= 4.
        reject = False
        reason = ""

        if erosion_fish >= 2 and (residual_count >= 2 or eye_count >= 2):
            reject = True
            reason = (f"Multiple fish: eros={erosion_fish} AND "
                      f"(resid={residual_count} OR eye={eye_count}). "
                      f"All signals: {estimates}")

        elif methods_multiple >= 2 and erosion_fish >= 2:
            reject = True
            reason = (f"Multiple fish: {methods_multiple} orig methods AND "
                      f"eros={erosion_fish}. All signals: {estimates}")

        elif contour_count >= 4:
            # Very strong contour signal (4+ distinct fish-shaped regions)
            reject = True
            reason = (f"Multiple fish: contour={contour_count} (>=4, strong). "
                      f"All signals: {estimates}")

        if reject:
            max_count = max(c for _, c in estimates)
            return False, max_count, yolo_dets, reason

        return True, max(1, max_count_orig), yolo_dets, ""

    # -- Layer A: YOLO ---------------------------------------------------------

    def _yolo_detect(self, image: np.ndarray) -> Tuple[int, List[dict]]:
        results = self.yolo_model(image, verbose=False, conf=0.3)[0]
        img_area = image.shape[0] * image.shape[1]
        detections = []
        significant = []

        for box in results.boxes:
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_frac = (x2 - x1) * (y2 - y1) / img_area
            det = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "class_name": cls_name,
                "area_fraction": round(area_frac, 4),
            }
            detections.append(det)
            if area_frac >= self.SECONDARY_AREA_FRAC and conf >= 0.3:
                significant.append(det)

        return len(significant), detections

    # -- Layer B: Contour detection  [P1 kernel fix, P1 IoU check] ------------

    def _contour_detect(self, image: np.ndarray) -> Tuple[int, List[dict]]:
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_area = image.shape[0] * image.shape[1]
        blurred  = cv2.GaussianBlur(gray, (7, 7), 0)

        count_estimates = []

        _, otsu_thresh     = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_thresh    = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )
        edges              = cv2.Canny(blurred, 30, 100)
        # [P1] kernel reduced from (11,11) to (7,7) to avoid bridging nearby fish
        kernel_small       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_edges      = cv2.dilate(edges, kernel_small, iterations=3)

        for thresh_name, thresh_img in [
            ("otsu", otsu_thresh),
            ("adaptive", adaptive_thresh),
            ("edges", dilated_edges),
        ]:
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  morph_kernel, iterations=1)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            significant = []
            for cnt in contours:
                area_frac = cv2.contourArea(cnt) / img_area
                if area_frac < self.PRIMARY_AREA_FRAC:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / max(h, 1)
                if 0.3 < aspect < 6.0:
                    significant.append({
                        "bbox": [x, y, x + w, y + h],
                        "area_fraction": round(area_frac, 4),
                        "aspect_ratio": round(aspect, 2),
                        "method": thresh_name,
                    })

            # [P1] IoU-based separation: if two large contours have low IoU
            # count them as distinct even if area fractions are close
            if len(significant) >= 2:
                n = self._count_after_iou_merge(significant)
                count_estimates.append(n)
            else:
                count_estimates.append(len(significant))

        count_estimates.sort()
        median_count = count_estimates[len(count_estimates) // 2]
        return median_count, significant

    def _count_after_iou_merge(self, dets: List[dict]) -> int:
        """Treat detections as separate if pairwise IoU < IOU_SEPARATE_THRESH."""
        boxes  = [d["bbox"] for d in dets]
        groups = list(range(len(boxes)))

        def iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua    = (a[2]-a[0])*(a[3]-a[1])
            ub    = (b[2]-b[0])*(b[3]-b[1])
            denom = ua + ub - inter
            return inter / denom if denom > 0 else 0.0

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) >= self.IOU_SEPARATE_THRESH:
                    groups[j] = groups[i]   # merge

        return len(set(groups))

    # -- Layer C: Edge density clustering  [P1 secondary floor] ---------------

    def _edge_cluster_detect(self, image: np.ndarray) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_area = h * w

        sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        edge_thresh = np.percentile(magnitude, 85)
        high_edge   = (magnitude > edge_thresh).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed = cv2.morphologyEx(high_edge, cv2.MORPH_CLOSE, kernel, iterations=4)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # Primary cluster: largest cluster
        cluster_areas = [
            stats[i, cv2.CC_STAT_AREA] / img_area
            for i in range(1, num_labels)
        ]
        if not cluster_areas:
            return 1

        cluster_areas.sort(reverse=True)
        primary_frac = cluster_areas[0]

        # [P1] Secondary clusters: use lower 1.5% floor
        secondary_clusters = sum(
            1 for frac in cluster_areas[1:]
            if frac >= self.SECONDARY_EDGE_FRAC
        )

        if primary_frac < self.PRIMARY_AREA_FRAC:
            return 0

        return 1 + secondary_clusters

    # -- [P2] Residual-mask check (new in v2.0) --------------------------------

    def _residual_mask_detect(self, image: np.ndarray) -> int:
        """
        Find the dominant foreground blob.  Subtract it.
        Check the residual for any additional fish-shaped regions ? 0.5%.
        Returns total fish count (1 = only primary, 2+ = secondary detected).
        """
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_area = image.shape[0] * image.shape[1]
        blurred  = cv2.GaussianBlur(gray, (7, 7), 0)

        # Auto-select Otsu polarity: use the variant that gives the larger
        # single contour (more likely to be the fish, not the background)
        _, thresh_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, thresh_inv  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_thresh = self._pick_fish_polarity(thresh_norm, thresh_inv, img_area)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned = cv2.morphologyEx(best_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 1

        # Dominant contour
        contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
        dominant         = contours_by_area[0]
        dominant_frac    = cv2.contourArea(dominant) / img_area

        if dominant_frac < self.PRIMARY_AREA_FRAC:
            return 1

        # Build residual mask by zeroing out the dominant blob
        dominant_mask = np.zeros_like(cleaned)
        cv2.drawContours(dominant_mask, [dominant], -1, 255, -1)
        residual_mask = cv2.bitwise_and(cleaned, cv2.bitwise_not(dominant_mask))

        # Dilate residual slightly to reconnect fragmented edges
        residual_mask = cv2.dilate(residual_mask, kernel, iterations=1)

        res_contours, _ = cv2.findContours(residual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        secondary_fish = 0
        for cnt in res_contours:
            frac = cv2.contourArea(cnt) / img_area
            if frac < self.SECONDARY_AREA_FRAC:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 0.3 < aspect < 6.0:
                secondary_fish += 1

        return 1 + secondary_fish

    # -- Layer D: erosion-spatial check  [P6] ------------------------------------

    def _erosion_spatial_check(self, image: np.ndarray) -> int:
        """
        Progressively erode the foreground mask and check whether multiple
        comparably-sized, spatially-proximate blobs emerge.

        Two fish swimming in parallel will split into two similar-sized blobs
        at some erosion level. Fins/tails that detach are filtered out because:
          - They are much smaller than the body (ratio filter)
          - They fly far away from the body centroid (spatial filter)

        Returns estimated fish count (1 if only one fish detected).
        """
        h, w = image.shape[:2]
        img_area = h * w
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        _, thresh_n = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, thresh_i = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        best = self._pick_fish_polarity(thresh_n, thresh_i, img_area)

        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg  = cv2.morphologyEx(best, cv2.MORPH_CLOSE, k9, iterations=3)

        max_fish = 1
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        for iters in [5, 8, 12, 16]:
            eroded = cv2.erode(fg, k7, iterations=iters)
            cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            blobs = []
            for c in cnts:
                area = cv2.contourArea(c)
                frac = area / img_area
                if frac < self.EROSION_MIN_FRAC:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                blobs.append({"frac": frac, "cx": cx, "cy": cy,
                               "r": np.sqrt(area / np.pi)})
            blobs.sort(key=lambda b: -b["frac"])

            if not blobs:
                continue
            primary = blobs[0]
            secondary = 0
            for b in blobs[1:]:
                # Size filter: secondary must be at least EROSION_SEC_RATIO of primary
                if b["frac"] / primary["frac"] < self.EROSION_SEC_RATIO:
                    continue
                # Spatial filter: centroid distance <= EROSION_MAX_SEP * avg radius
                dist = np.hypot(b["cx"] - primary["cx"], b["cy"] - primary["cy"])
                avg_r = (primary["r"] + b["r"]) / 2
                if dist / max(avg_r, 1) <= self.EROSION_MAX_SEP:
                    secondary += 1

            max_fish = max(max_fish, 1 + secondary)

        return max_fish

    # -- Layer E: eye-count check  [P6] ------------------------------------------

    def _eye_count_check(self, image: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
        """
        Detect fish eyes using Hough circles with strict dark-center /
        bright-surround filtering.  Eye candidates are restricted to the
        HEAD ZONES (left 35% and right 35% of the dominant contour bbox)
        to eliminate false positives from body markings and coral.

        Returns (n_distinct_fish_eyes, fg_mask).
        Two eye positions separated by > EYE_SEP_FRAC * min(h,w) are
        counted as belonging to distinct fish.
        """
        h, w = image.shape[:2]
        img_area = h * w
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Build foreground mask + dominant contour
        blurred_seg = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh_n = cv2.threshold(blurred_seg, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, thresh_i = cv2.threshold(blurred_seg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def score_polarity(t):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            c = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k, iterations=3)
            cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 0, None, c
            lg   = max(cnts, key=cv2.contourArea)
            frac = cv2.contourArea(lg) / img_area
            bx, by, bw, bh = cv2.boundingRect(lg)
            sc   = frac / (abs(bw / max(bh, 1) - 1.8) + 0.1)
            return sc, lg, c

        s1, cnt1, clean1 = score_polarity(thresh_n)
        s2, cnt2, clean2 = score_polarity(thresh_i)
        dom_cnt  = cnt1 if s1 >= s2 else cnt2
        fg_clean = clean1 if s1 >= s2 else clean2

        # Build head-zone search mask: left 35% + right 35% of dominant bbox,
        # intersected with a dilated foreground mask so we stay on the fish.
        search_mask = None
        if dom_cnt is not None:
            bx, by, bw_c, bh_c = cv2.boundingRect(dom_cnt)
            head_w = int(bw_c * 0.35)
            head_mask = np.zeros((h, w), dtype=np.uint8)
            head_mask[by:by + bh_c, bx:bx + head_w] = 255                          # left end
            head_mask[by:by + bh_c, bx + bw_c - head_w:bx + bw_c] = 255           # right end

            contour_fill = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_fill, [dom_cnt], -1, 255, -1)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            contour_fill = cv2.dilate(contour_fill, k_d, iterations=1)
            search_mask = cv2.bitwise_and(head_mask, contour_fill)

        # Hough circle detection on CLAHE-enhanced image
        min_r = int(min(h, w) * 0.008)
        max_r = int(min(h, w) * 0.025)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred_eye = cv2.GaussianBlur(enhanced, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred_eye, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=int(min(h, w) * 0.06),
            param1=60, param2=22,
            minRadius=min_r, maxRadius=max_r,
        )

        eye_candidates = []
        if circles is not None:
            margin = max_r + 5
            for c in circles[0]:
                cx_e, cy_e, r_e = int(c[0]), int(c[1]), int(c[2])
                # Image-edge guard
                if cx_e < margin or cy_e < margin or cx_e > w - margin or cy_e > h - margin:
                    continue
                # Must be inside head-zone search mask
                if search_mask is not None and search_mask[cy_e, cx_e] == 0:
                    continue
                # Dark-center / bright-surround filter
                cm = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(cm, (cx_e, cy_e), max(r_e // 2, 3), 255, -1)
                sm = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(sm, (cx_e, cy_e), int(r_e * 1.8), 255, -1)
                cv2.circle(sm, (cx_e, cy_e), r_e, 0, -1)
                if cm.sum() == 0 or sm.sum() == 0:
                    continue
                center_v   = float(gray[cm > 0].mean())
                surround_v = float(gray[sm > 0].mean())
                contrast   = surround_v - center_v
                if contrast > self.EYE_MIN_CONTRAST and center_v < self.EYE_MAX_CENTER_VAL:
                    eye_candidates.append((cx_e, cy_e, contrast))

        if not eye_candidates:
            return 0, fg_clean

        # Cluster eye positions: within EYE_SEP_FRAC * min(h,w) = same fish
        sep_thresh = min(h, w) * self.EYE_SEP_FRAC
        clusters: List[List] = []
        for ex, ey, ec in sorted(eye_candidates, key=lambda x: -x[2]):
            placed = False
            for cluster in clusters:
                gcx = np.mean([e[0] for e in cluster])
                gcy = np.mean([e[1] for e in cluster])
                if np.hypot(ex - gcx, ey - gcy) < sep_thresh:
                    cluster.append((ex, ey, ec))
                    placed = True
                    break
            if not placed:
                clusters.append([(ex, ey, ec)])

        # Only count clusters with at least one detection above threshold
        sig = [cl for cl in clusters if max(e[2] for e in cl) > self.EYE_MIN_CONTRAST]
        return len(sig), fg_clean

    def _pick_fish_polarity(self, thresh_norm, thresh_inv, img_area) -> np.ndarray:
        """
        Choose the Otsu polarity whose largest contour has the most fish-like
        aspect ratio.  Avoids counting the background blob as the fish.
        """
        def best_aspect(thresh):
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 0.0
            largest = max(cnts, key=cv2.contourArea)
            frac    = cv2.contourArea(largest) / img_area
            if frac < 0.01:
                return 0.0
            x, y, w, h = cv2.boundingRect(largest)
            aspect = w / max(h, 1)
            # Score = how close aspect ratio is to the ideal fish shape (1.5)
            return 1.0 / (abs(aspect - 1.5) + 0.1)

        score_norm = best_aspect(thresh_norm)
        score_inv  = best_aspect(thresh_inv)
        return thresh_norm if score_norm >= score_inv else thresh_inv


# ==============================================================================
# CHECK 2: LIGHTING / VISIBILITY QUALITY   [P3 applied]
# ==============================================================================

class LightingQualityChecker:
    """
    Detects bad lighting conditions where fish patterns are barely visible.

    v2.0 changes [P3]:
      - Rejection threshold raised from 4 -> 6 (default mode).
        Underwater photography legitimately scores 2-3 points from physics
        (lower saturation, slightly lower brightness, blue cast) -- the old
        threshold of 4 was too easily tripped by valid images.
      - Saturation sub-check removed from the evidence score. Low saturation
        is normal for underwater fish photography and shouldn't penalise images.
      - Detail-score override: if detail_score ? 20 (patterns clearly visible),
        evidence from brightness/dark-fraction checks is halved before comparing
        to threshold, preventing rejection of dim-but-sharp images.
    """

    def __init__(
        self,
        brightness_low: float   = 55.0,
        brightness_high: float  = 220.0,
        min_local_contrast: float = 18.0,
        min_dynamic_range: float  = 100,
        min_detail_score: float   = 15.0,
        dark_pixel_fraction_max: float = 0.45,
        rejection_score_threshold: int = 6,     # [P3] raised from 4
    ):
        self.brightness_low              = brightness_low
        self.brightness_high             = brightness_high
        self.min_local_contrast          = min_local_contrast
        self.min_dynamic_range           = min_dynamic_range
        self.min_detail_score            = min_detail_score
        self.dark_pixel_fraction_max     = dark_pixel_fraction_max
        self.rejection_score_threshold   = rejection_score_threshold

    def check(self, image: np.ndarray) -> Tuple[bool, dict, List[str]]:
        reasons = []
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape

        mean_brightness   = float(np.mean(gray))
        median_brightness = float(np.median(gray))
        dark_fraction     = float(np.mean(gray < 40))
        very_dark_fraction= float(np.mean(gray < 20))
        bright_fraction   = float(np.mean(gray > 230))
        p5                = float(np.percentile(gray, 5))
        p95               = float(np.percentile(gray, 95))
        dynamic_range     = p95 - p5

        local_stds          = self._compute_local_contrast(gray, block_size=32)
        mean_local_contrast = float(np.mean(local_stds))

        laplacian     = cv2.Laplacian(gray, cv2.CV_64F)
        detail_score  = float(np.std(laplacian))

        # [P3] saturation metric kept for reporting but NOT scored
        hsv              = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation  = float(np.mean(hsv[:, :, 1]))

        cy, cx     = h // 2, w // 2
        my, mx     = h // 4, w // 4
        center_crop = gray[cy - my:cy + my, cx - mx:cx + mx]
        center_std  = float(np.std(center_crop))
        center_mean = float(np.mean(center_crop))

        # -- Accumulate evidence -----------------------------------------------
        bad_evidence = 0

        # Brightness: compute contribution once, then apply center-mean cap.
        # If the fish CENTER region is well-lit (center_mean > 50), the fish body
        # is adequately exposed even if global brightness is low (dark background).
        # In that case, cap the brightness penalty at 1 to avoid penalising
        # valid dark-background shots (marginatus, chrysurus, minor).
        if mean_brightness < self.brightness_low:
            bright_contrib = 2 + (1 if mean_brightness < 45 else 0)
            if center_mean > 50 and bright_contrib > 1:
                bright_contrib = 1   # fish center is lit -- background is dark, not the fish
            bad_evidence += bright_contrib

        # Dark fraction: only score when the fish CENTER itself is dark.
        # center_mean > 40 means the fish body is adequately lit; dark pixels
        # are background, not the fish (marginatus dk=0.678 but c_mean=64 -> skip).
        fish_center_dark = center_mean < 40
        if fish_center_dark:
            if dark_fraction > 0.40:
                bad_evidence += 2
            if very_dark_fraction > 0.25:
                bad_evidence += 2
        else:
            if dark_fraction > 0.85:
                bad_evidence += 1

        # Silhouette detection (rostratum-style: fish is a near-pure black body
        # against a bright background). Global brightness stays moderate because
        # the bright background compensates. Signal: center region is overwhelmingly
        # very dark pixels (< 30) AND center_mean is very low (< 35).
        center_very_dark_frac = float(np.mean(center_crop < 30))
        if center_very_dark_frac > 0.70 and center_mean < 35:
            bad_evidence += 4   # strong penalty -- fish patterns completely invisible

        if dynamic_range < self.min_dynamic_range:
            bad_evidence += 1
        if mean_local_contrast < self.min_local_contrast:
            bad_evidence += 2
        if detail_score < self.min_detail_score:
            bad_evidence += 1
        if center_std < 15:
            bad_evidence += 1

        # Overexposure
        if mean_brightness > self.brightness_high:
            bad_evidence += 2
        if bright_fraction > 0.4:
            bad_evidence += 2

        # NOTE [P3]: saturation sub-check REMOVED.

        # [P3] Detail-score override: if patterns are clearly visible, halve
        # the brightness/darkness evidence before applying the threshold.
        if detail_score >= 20:
            bad_evidence = bad_evidence // 2

        metrics = {
            "mean_brightness":      round(mean_brightness,    2),
            "median_brightness":    round(median_brightness,   2),
            "dark_fraction":        round(dark_fraction,       4),
            "very_dark_fraction":   round(very_dark_fraction,  4),
            "bright_fraction":      round(bright_fraction,     4),
            "dynamic_range":        round(dynamic_range,       2),
            "p5":                   round(p5,                  2),
            "p95":                  round(p95,                 2),
            "mean_local_contrast":  round(mean_local_contrast, 2),
            "detail_score":         round(detail_score,        2),
            "mean_saturation":      round(mean_saturation,     2),  # informational
            "center_std":           round(center_std,          2),
            "center_mean":          round(center_mean,         2),
            "bad_lighting_score":   bad_evidence,
            "detail_override_applied": detail_score >= 20,
        }

        if bad_evidence >= self.rejection_score_threshold:
            reasons.append(
                f"Bad lighting detected (score={bad_evidence}, "
                f"threshold={self.rejection_score_threshold}): "
                f"brightness={mean_brightness:.0f}, "
                f"local_contrast={mean_local_contrast:.1f}, "
                f"dark_frac={dark_fraction:.2f}, "
                f"dynamic_range={dynamic_range:.0f}, "
                f"detail_score={detail_score:.1f}"
            )

        return len(reasons) == 0, metrics, reasons

    def _compute_local_contrast(self, gray: np.ndarray, block_size: int = 32) -> np.ndarray:
        h, w = gray.shape
        stds  = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                stds.append(np.std(gray[y:y + block_size, x:x + block_size]))
        return np.array(stds) if stds else np.array([0.0])


# ==============================================================================
# CHECK 3: BACKGROUND BLEND / LOW CONTRAST   [P4 + P5 applied]
# ==============================================================================

class BackgroundContrastChecker:
    """
    Detects if the fish blends into the background.

    v2.0 changes [P4, P5]:
      [P4] HSV hue-channel contrast added alongside BGR intensity contrast.
           Same-hue scenarios (blue fish in blue water) were previously missed
           because intensity contrast alone can still be moderate.
      [P4] Histogram correlation rejection threshold tightened 0.70 -> 0.85.
           High hist correlation is now treated as strong blend evidence only
           when it is very high (nearly identical histograms).
      [P4] Decision uses weighted combination instead of max() so one
           anomalous high metric cannot override two genuinely low ones.
      [P5] GrabCut now tries three candidate rectangle initializations
           (center, left-biased, right-biased) and selects the result with
           the highest internal fg/bg color contrast.
    """

    def __init__(
        self,
        min_contrast: float      = 25.0,
        hist_corr_threshold: float = 0.85,   # [P4] raised from 0.70
    ):
        self.min_contrast        = min_contrast
        self.hist_corr_threshold = hist_corr_threshold

    def check(self, image: np.ndarray) -> Tuple[bool, dict, List[str]]:
        reasons = []
        h, w    = image.shape[:2]
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # [P5] Try three GrabCut initializations; keep the best
        fg_mask = self._best_grabcut_mask(image)

        if fg_mask is None or fg_mask.sum() < 100:
            cy, cx = h // 2, w // 2
            my, mx = h // 4, w // 4
            fg_mask = np.zeros((h, w), dtype=bool)
            fg_mask[cy - my:cy + my, cx - mx:cx + mx] = True

        bg_mask = ~fg_mask

        # -- Intensity contrast ------------------------------------------------
        fg_mean = float(np.mean(gray[fg_mask]))
        bg_mean = float(np.mean(gray[bg_mask]))
        intensity_contrast = abs(fg_mean - bg_mean)

        # -- Color contrast (BGR Euclidean) ------------------------------------
        fg_color    = np.mean(image[fg_mask].astype(float), axis=0)
        bg_color    = np.mean(image[bg_mask].astype(float), axis=0)
        color_contrast = float(np.linalg.norm(fg_color - bg_color))

        # -- [P4] HSV hue-channel contrast -------------------------------------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(float)
        # Hue wraps at 180 in OpenCV; compute circular difference
        fg_hue_mean = float(np.mean(hue[fg_mask]))
        bg_hue_mean = float(np.mean(hue[bg_mask]))
        raw_hue_diff = abs(fg_hue_mean - bg_hue_mean)
        hue_contrast = min(raw_hue_diff, 180 - raw_hue_diff)  # circular distance

        fg_sat_mean = float(np.mean(hsv[:, :, 1][fg_mask]))
        bg_sat_mean = float(np.mean(hsv[:, :, 1][bg_mask]))
        sat_contrast = abs(fg_sat_mean - bg_sat_mean)

        # -- Edge contrast at fg boundary --------------------------------------
        fg_uint8 = fg_mask.astype(np.uint8) * 255
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated  = cv2.dilate(fg_uint8, kernel, iterations=2)
        boundary = (dilated > 0) & (~fg_mask)
        if boundary.sum() > 10:
            boundary_contrast = float(abs(fg_mean - np.mean(gray[boundary])))
        else:
            boundary_contrast = intensity_contrast

        # -- Histogram similarity ----------------------------------------------
        fg_hist = cv2.calcHist([gray], [0], fg_mask.astype(np.uint8),       [256], [0, 256])
        bg_hist = cv2.calcHist([gray], [0], bg_mask.astype(np.uint8),       [256], [0, 256])
        cv2.normalize(fg_hist, fg_hist)
        cv2.normalize(bg_hist, bg_hist)
        hist_correlation = float(cv2.compareHist(fg_hist, bg_hist, cv2.HISTCMP_CORREL))

        # -- [P4] Weighted combination instead of max() ------------------------
        # Weights: boundary_contrast most reliable, hue contrast important for
        # same-hue cases, intensity/color support.
        weighted_contrast = (
            0.35 * boundary_contrast
          + 0.25 * intensity_contrast
          + 0.25 * color_contrast
          + 0.15 * hue_contrast
        )

        metrics = {
            "intensity_contrast":  round(intensity_contrast,  2),
            "color_contrast":      round(color_contrast,       2),
            "boundary_contrast":   round(boundary_contrast,    2),
            "hue_contrast":        round(hue_contrast,         2),
            "sat_contrast":        round(sat_contrast,         2),
            "weighted_contrast":   round(weighted_contrast,    2),
            "histogram_correlation": round(hist_correlation,   4),
            "fg_mean":             round(fg_mean,              2),
            "bg_mean":             round(bg_mean,              2),
            "fg_hue_mean":         round(fg_hue_mean,          2),
            "bg_hue_mean":         round(bg_hue_mean,          2),
        }

        # -- Decision: requires two independent indicators of blending ----
        # Simplified to two clean signals -- the hue-specific vote was removed
        # because it fires on any fish matching water color (tuberosus, scopas)
        # even when the fish is clearly shape-distinguishable. Hue contrast is
        # already captured within weighted_contrast via the BGR color component.
        blend_votes = 0

        if weighted_contrast < self.min_contrast:
            blend_votes += 1

        # Histogram correlation only counts when contrast is also genuinely low --
        # prevents firing on well-contrasted fish that share a background hue.
        if hist_correlation > self.hist_corr_threshold and weighted_contrast < self.min_contrast * 1.5:
            blend_votes += 1

        if blend_votes >= 2:
            reasons.append(
                f"Fish blends into background: "
                f"weighted_contrast={weighted_contrast:.1f}, "
                f"hist_corr={hist_correlation:.3f} "
                f"(blend_votes={blend_votes})"
            )

        return len(reasons) == 0, metrics, reasons

    # -- [P5] Multi-init GrabCut (with downscaling for performance) --------------

    # Maximum width for GrabCut processing. At full resolution (1000-2000px),
    # GrabCut takes 15-16 seconds per call. At 400px it takes ~0.6 seconds.
    # The foreground/background separation quality is equivalent at this scale.
    GRABCUT_MAX_W = 400

    def _best_grabcut_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run GrabCut with three different seed rectangles on a downscaled copy
        of the image. Returns the full-resolution mask from the candidate that
        produces the highest fg/bg color contrast.
        """
        h, w = image.shape[:2]

        # Downscale for speed -- GrabCut complexity scales with pixel count
        scale = min(1.0, self.GRABCUT_MAX_W / w)
        if scale < 1.0:
            small = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = image
        sh, sw = small.shape[:2]

        mx, my = int(sw * 0.10), int(sh * 0.10)

        candidates = [
            (mx,          my,          sw - 2*mx,      sh - 2*my),     # center
            (mx,          my,          int(sw * 0.60), sh - 2*my),     # left-biased
            (int(sw*0.35), my,         sw - mx - int(sw*0.35), sh-2*my), # right-biased
        ]

        best_mask_small = None
        best_score      = -1.0

        for rect in candidates:
            mask_small = self._run_grabcut(small, rect)
            if mask_small is None:
                continue
            fg = mask_small.astype(bool)
            bg = ~fg
            if fg.sum() < 50 or bg.sum() < 50:
                continue
            fg_col = np.mean(small[fg].astype(float), axis=0)
            bg_col = np.mean(small[bg].astype(float), axis=0)
            score  = float(np.linalg.norm(fg_col - bg_col))
            if score > best_score:
                best_score      = score
                best_mask_small = mask_small

        if best_mask_small is None:
            return None

        # Upsample mask back to original resolution
        if scale < 1.0:
            mask_up = cv2.resize(
                best_mask_small.astype(np.uint8) * 255,
                (w, h), interpolation=cv2.INTER_NEAREST
            )
            return mask_up > 127
        return best_mask_small

    def _run_grabcut(self, image: np.ndarray, rect: tuple) -> Optional[np.ndarray]:
        h, w  = image.shape[:2]
        mask  = np.zeros((h, w), np.uint8)
        bgd   = np.zeros((1, 65), np.float64)
        fgd   = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
        except cv2.error:
            return None


# ==============================================================================
# CHECK 4: IMAGE TECHNICAL QUALITY (blur, resolution)  -- unchanged
# ==============================================================================

class TechnicalQualityChecker:
    """Checks resolution, blur, and basic image integrity."""

    def __init__(
        self,
        min_width: int     = 400,
        min_height: int    = 300,
        blur_threshold: float = 50.0,
    ):
        self.min_width     = min_width
        self.min_height    = min_height
        self.blur_threshold = blur_threshold

    def check(self, image: np.ndarray) -> Tuple[bool, dict, List[str]]:
        reasons = []
        h, w    = image.shape[:2]
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if w < self.min_width or h < self.min_height:
            reasons.append(f"Resolution too low: {w}x{h}")

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        sobelx     = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely     = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad  = float(np.mean(sobelx**2 + sobely**2))

        # Detail-score override: underwater fish photos often have mild haze/
        # softness from the water column, but the fish body and patterns remain
        # clearly distinguishable. If tenengrad is above a minimum visible-detail
        # floor (400), do not reject on Laplacian variance alone.
        # Floor lowered from 750 to 400 to pass tominiensis (558) and hexacanthus (658).
        detail_floor = 400.0
        blur_rejected = (lap_var < self.blur_threshold) and (tenengrad < detail_floor)
        if blur_rejected:
            reasons.append(
                f"Image too blurry: Laplacian var={lap_var:.1f}, "
                f"tenengrad={tenengrad:.1f} (both below thresholds)"
            )

        metrics = {
            "width":              w,
            "height":             h,
            "laplacian_variance": round(lap_var,   2),
            "tenengrad":          round(tenengrad,  2),
            "detail_floor":       detail_floor,
            "blur_override_applied": (lap_var < self.blur_threshold) and (tenengrad >= detail_floor),
        }
        return len(reasons) == 0, metrics, reasons


# ==============================================================================
# CHECK 5: FISH COMPLETENESS (occlusion / partial visibility)  [P5 applied]
# ==============================================================================

class CompletenessChecker:
    """
    Checks if the fish appears complete (not cut off or heavily occluded).

    v2.0 changes [P5]:
      - Otsu polarity auto-selected (normal vs inverted) to avoid returning
        the background blob as the fish contour on dark-background images.
      - Edge-touch rejection threshold raised from 3 -> 4 edges.

    v2.1 changes [P7 -- close-up false-rejection fix]:
      The v2.0 threshold (area_frac < 0.70) incorrectly rejected valid
      close-up shots (bariene, blochii, dussumieri, fowleri, grammoptilus,
      leucosternon, maculiceps, nubilus) where the fish fills the frame
      and legitimately touches all 4 edges.

      Root cause: `fish_area_frac` is measured from the LARGEST OTSU
      CONTOUR, which is the fish BODY blob only -- fins, tail, and gaps
      split into separate smaller contours, making a complete fish appear
      to have area_frac as low as 0.47. The 0.70 threshold was calibrated
      on images where the fish body filled most of the frame, and failed
      on any image where the fish body contour was fragmented.

      Fix: lower the area_frac rejection threshold from 0.70 to 0.35.
      A fish that occupies < 35% of the image while touching all 4 edges
      is genuinely cropped (only a fragment is visible). A fish that
      occupies 40-99% of the image while touching 4 edges is a well-framed
      close-up photograph and must not be rejected.

      This is intentionally conservative: we prefer to pass a marginally
      cropped image than to reject dozens of valid close-up specimens.
      The multi-fish detector and segmentation stage provide additional
      quality gates downstream.
    """

    AREA_FRAC_THRESHOLD = 0.35   # [P7] lowered from 0.70

    def __init__(self, edge_touch_threshold: int = 4):
        self.edge_touch_threshold = edge_touch_threshold

    def check(self, image: np.ndarray) -> Tuple[bool, dict, List[str]]:
        reasons = []
        h, w     = image.shape[:2]
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray, (7, 7), 0)
        img_area = h * w

        # Auto-select Otsu polarity: pick the one whose largest contour
        # has the most fish-like aspect ratio (avoids returning the
        # background blob as the fish on dark-background images).
        _, thresh_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, thresh_inv  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def largest_fish_like(thresh):
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None, 0.0, 0.0
            largest = max(cnts, key=cv2.contourArea)
            frac    = cv2.contourArea(largest) / img_area
            bx, by, bw, bh = cv2.boundingRect(largest)
            aspect  = bw / max(bh, 1)
            return largest, frac, aspect

        cnt_n, frac_n, asp_n = largest_fish_like(thresh_norm)
        cnt_i, frac_i, asp_i = largest_fish_like(thresh_inv)

        def fish_score(frac, aspect):
            if frac < 0.01:
                return 0.0
            return frac / (abs(aspect - 1.8) + 0.1)

        if fish_score(frac_n, asp_n) >= fish_score(frac_i, asp_i):
            largest, fish_area_frac = cnt_n, frac_n
        else:
            largest, fish_area_frac = cnt_i, frac_i

        if largest is None:
            return True, {"no_contours": True}, []

        x, y, bw, bh   = cv2.boundingRect(largest)
        edge_margin    = 5
        touches_left   = x <= edge_margin
        touches_top    = y <= edge_margin
        touches_right  = (x + bw) >= (w - edge_margin)
        touches_bottom = (y + bh) >= (h - edge_margin)
        edges_touched  = sum([touches_left, touches_top, touches_right, touches_bottom])

        metrics = {
            "bbox":               [x, y, x + bw, y + bh],
            "fish_area_fraction": round(fish_area_frac, 4),
            "edges_touched":      edges_touched,
            "area_frac_threshold": self.AREA_FRAC_THRESHOLD,
            "touches": {
                "left":   touches_left,
                "top":    touches_top,
                "right":  touches_right,
                "bottom": touches_bottom,
            },
        }

        # [P7] Only reject when a very small fragment touches all edges --
        # a valid close-up fish touching 4 edges has area_frac >= 0.40.
        if edges_touched >= self.edge_touch_threshold and fish_area_frac < self.AREA_FRAC_THRESHOLD:
            reasons.append(
                f"Fish appears partially hidden: touches {edges_touched} edges, "
                f"area fraction={fish_area_frac:.2f} (threshold={self.AREA_FRAC_THRESHOLD})"
            )

        return len(reasons) == 0, metrics, reasons


# ==============================================================================
# MAIN SCREENER: COMBINES ALL CHECKS
# ==============================================================================

class FishImageScreener:
    """
    Autonomous fish image quality screener.
    Runs all checks and produces a pass/fail decision for each image.
    """

    def __init__(self, strict: bool = False, lenient: bool = False):
        if strict:
            brightness_low      = 60.0
            min_local_contrast  = 20.0
            min_bg_contrast     = 30.0
            blur_thresh         = 60.0
            lighting_threshold  = 5      # [P3] still higher than v1 default of 4
        elif lenient:
            brightness_low      = 45.0
            min_local_contrast  = 14.0
            min_bg_contrast     = 18.0
            blur_thresh         = 35.0
            lighting_threshold  = 8
        else:
            brightness_low      = 55.0
            min_local_contrast  = 18.0
            min_bg_contrast     = 25.0
            blur_thresh         = 50.0
            lighting_threshold  = 6      # [P3] raised from 4

        # Initialize YOLO if available
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 loaded for multi-fish detection")
            except Exception as e:
                logger.warning(f"Could not load YOLO: {e}")

        self.multi_fish_detector   = MultiFishDetector(yolo_model=self.yolo_model)
        self.lighting_checker      = LightingQualityChecker(
            brightness_low=brightness_low,
            min_local_contrast=min_local_contrast,
            rejection_score_threshold=lighting_threshold,
        )
        self.background_checker    = BackgroundContrastChecker(min_contrast=min_bg_contrast)
        self.technical_checker     = TechnicalQualityChecker(blur_threshold=blur_thresh)
        self.completeness_checker  = CompletenessChecker()

    def _parse_genus_species(self, image_path: Path) -> Tuple[Optional[str], Optional[str]]:
        genus   = image_path.parent.name if image_path.parent.name in GENERA else None
        species = image_path.stem
        return genus, species

    def screen_image(self, image_path: str) -> ScreeningResult:
        path               = Path(image_path)
        genus, species     = self._parse_genus_species(path)
        all_reasons: List[str] = []
        all_warnings: List[str]= []
        all_scores             = {"genus": genus, "species": species}

        image = cv2.imread(str(path))
        if image is None:
            return ScreeningResult(
                filepath=str(path), filename=path.name,
                genus=genus, species=species,
                passed=False,
                rejection_reasons=["Cannot read image file"],
                warnings=[], scores={},
            )

        # Check 1: Technical quality
        tech_ok, tech_metrics, tech_reasons = self.technical_checker.check(image)
        all_scores["technical"] = tech_metrics
        all_reasons.extend(tech_reasons)

        # Check 2: Lighting quality  [P3]
        light_ok, light_metrics, light_reasons = self.lighting_checker.check(image)
        all_scores["lighting"] = light_metrics
        all_reasons.extend(light_reasons)

        # Check 3: Multiple fish  [P1, P2]
        multi_ok, fish_count, det_info, multi_reason = self.multi_fish_detector.detect(image)
        all_scores["multi_fish"] = {"fish_count": fish_count, "passed": multi_ok}
        if not multi_ok:
            all_reasons.append(multi_reason)

        # Check 4: Background contrast  [P4, P5]
        bg_ok, bg_metrics, bg_reasons = self.background_checker.check(image)
        all_scores["background_contrast"] = bg_metrics
        all_reasons.extend(bg_reasons)

        # Check 5: Completeness  [P5]
        comp_ok, comp_metrics, comp_reasons = self.completeness_checker.check(image)
        all_scores["completeness"] = comp_metrics
        all_reasons.extend(comp_reasons)

        passed = len(all_reasons) == 0
        return ScreeningResult(
            filepath=str(path), filename=path.name,
            genus=genus, species=species,
            passed=passed,
            rejection_reasons=all_reasons,
            warnings=all_warnings,
            scores=all_scores,
        )

    def screen_all(
        self,
        input_dir: Path,
        approved_dir: Path,
        reports_dir: Path,
        dry_run: bool = False,
    ) -> dict:
        logger.info("=" * 70)
        logger.info("AUTONOMOUS FISH IMAGE SCREENER  v2.2")
        logger.info("=" * 70)
        logger.info(f"Input:    {input_dir}")
        logger.info(f"Approved: {approved_dir}")
        logger.info(f"Dry run:  {dry_run}")
        logger.info("")

        all_images: List[Path] = []
        for genus_dir in sorted(input_dir.iterdir()):
            if not genus_dir.is_dir():
                continue
            images = sorted([
                f for f in genus_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            ])
            all_images.extend(images)
            logger.info(f"  {genus_dir.name}: {len(images)} images")

        logger.info(f"\nTotal images to screen: {len(all_images)}\n")

        if not dry_run:
            approved_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

        results       = []
        approved_list = []
        rejected_list = []
        genus_stats: Dict[str, Dict[str, int]] = {}

        for img_path in all_images:
            genus = img_path.parent.name
            if genus not in genus_stats:
                genus_stats[genus] = {"total": 0, "approved": 0, "rejected": 0}
            genus_stats[genus]["total"] += 1

            logger.info(f"Screening: {genus}/{img_path.name}")
            result = self.screen_image(str(img_path))
            results.append(result.to_dict())

            if result.passed:
                approved_list.append(str(img_path))
                genus_stats[genus]["approved"] += 1
                logger.info("  [OK] APPROVED")
                if not dry_run:
                    dest_dir = approved_dir / genus
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest_dir / img_path.name)
            else:
                rejected_list.append({"file": str(img_path), "reasons": result.rejection_reasons})
                genus_stats[genus]["rejected"] += 1
                logger.info("  [X] REJECTED:")
                for reason in result.rejection_reasons:
                    logger.info(f"      -> {reason}")

            logger.info("")

        report = {
            "timestamp":        datetime.now().isoformat(),
            "screener_version": "2.2",
            "input_directory":  str(input_dir),
            "summary": {
                "total_screened": len(all_images),
                "approved":       len(approved_list),
                "rejected":       len(rejected_list),
                "approval_rate":  round(len(approved_list) / max(len(all_images), 1) * 100, 1),
                "genus_breakdown": genus_stats,
            },
            "approved_images":  approved_list,
            "rejected_images":  rejected_list,
            "detailed_results": results,
        }

        if not dry_run:
            report_path = reports_dir / "screening_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved: {report_path}")

        logger.info("=" * 70)
        logger.info("SCREENING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total screened:  {len(all_images)}")
        logger.info(f"Approved:        {len(approved_list)}")
        logger.info(f"Rejected:        {len(rejected_list)}")
        logger.info("")
        logger.info("Per genus:")
        for genus, stats in sorted(genus_stats.items()):
            logger.info(
                f"  {genus:20s} | "
                f"total: {stats['total']:3d} | "
                f"approved: {stats['approved']:3d} | "
                f"rejected: {stats['rejected']:3d}"
            )

        logger.info("")
        if rejected_list:
            logger.info("Rejected images:")
            for entry in rejected_list:
                fp = Path(entry["file"])
                logger.info(f"  {fp.parent.name}/{fp.name}")
                for reason in entry["reasons"]:
                    logger.info(f"    -> {reason}")

        return report


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous surgeonfish image quality screener  v2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/python/fish_image_screener.py
  python scripts/python/fish_image_screener.py --dry-run
  python scripts/python/fish_image_screener.py --strict
  python scripts/python/fish_image_screener.py --single data/raw_images/Acanthurus/Acanthurus_mata.jpg
        """,
    )
    parser.add_argument("--input",    type=Path, default=RAW_IMAGES_DIR,
                        help="Root directory containing genus subdirectories")
    parser.add_argument("--approved", type=Path, default=APPROVED_DIR,
                        help="Output directory for approved images")
    parser.add_argument("--reports",  type=Path, default=REPORTS_DIR,
                        help="Directory for the screening report JSON")
    parser.add_argument("--single",   type=Path, default=None,
                        help="Screen a single image and print result")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run checks without copying any files")
    parser.add_argument("--strict",   action="store_true",
                        help="Use stricter thresholds (more rejections)")
    parser.add_argument("--lenient",  action="store_true",
                        help="Use lenient thresholds (fewer rejections)")

    args = parser.parse_args()

    if args.strict and args.lenient:
        logger.error("Cannot use both --strict and --lenient")
        sys.exit(1)

    screener = FishImageScreener(strict=args.strict, lenient=args.lenient)

    if args.single:
        result = screener.screen_image(str(args.single))
        print(json.dumps(result.to_dict(), indent=2))
        status = "APPROVED [OK]" if result.passed else "REJECTED [X]"
        print(f"\nVerdict: {status}")
        if result.rejection_reasons:
            print("Reasons:")
            for r in result.rejection_reasons:
                print(f"  -> {r}")
    else:
        screener.screen_all(
            input_dir=args.input,
            approved_dir=args.approved,
            reports_dir=args.reports,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()