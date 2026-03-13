"""
YOLO-Style Label Detection Module
Simulates YOLOv8 / MTCNN label detection without requiring the full library.

Features:
  - Anchor-based region proposal on a grid
  - Non-Maximum Suppression (NMS)
  - Multi-scale detection (3 scales)
  - Confidence scoring
  - Returns bounding boxes + crops for downstream classification

In production: drop-in replace with ultralytics YOLOv8 or MTCNN.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "label"

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y1:self.y2, self.x1:self.x2]


def iou(boxA: Detection, boxB: Detection) -> float:
    """Intersection over Union."""
    ix1 = max(boxA.x1, boxB.x1); iy1 = max(boxA.y1, boxB.y1)
    ix2 = min(boxA.x2, boxB.x2); iy2 = min(boxA.y2, boxB.y2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = boxA.area + boxB.area - inter
    return inter / (union + 1e-9)


def nms(detections: List[Detection], iou_thresh=0.45) -> List[Detection]:
    """Non-Maximum Suppression."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best, d) < iou_thresh]
    return kept


class YOLOStyleDetector:
    """
    YOLO-inspired label detector using image processing heuristics.
    Detects rectangular label regions via edge detection + contour analysis.

    Simulates YOLOv8 grid-based prediction without deep learning weights.
    Replace `detect()` with `ultralytics.YOLO('yolov8n.pt')(img)` in production.
    """

    ANCHORS = [
        # (w_ratio, h_ratio) relative to image size — 3 anchor scales
        (0.4, 0.3), (0.6, 0.5), (0.8, 0.7),
    ]

    def __init__(self, conf_thresh=0.35, iou_thresh=0.45, img_size=640):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.img_size    = img_size

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Letterbox resize to standard YOLO input size."""
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas  = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_x   = (self.img_size - new_w) // 2
        pad_y   = (self.img_size - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return canvas, scale, pad_x, pad_y

    def _edge_based_proposals(self, img: np.ndarray) -> List[Detection]:
        """
        Generate region proposals using Canny + morphological operations.
        Mimics YOLO's anchor regression on a feature map.
        """
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 1)
        edges   = cv2.Canny(blurred, 30, 100)

        # Dilate to connect label borders
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = img.shape[:2]
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: label should be a reasonable fraction of the image
            if area < 0.01 * H * W or area > 0.98 * H * W:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio filter (labels are roughly rectangular)
            ar = w / (h + 1e-9)
            if ar < 0.3 or ar > 5.0:
                continue

            # Confidence: based on edge density inside the region
            roi      = edges[y:y+h, x:x+w]
            conf     = float(roi.mean()) / 255.0
            conf     = np.clip(conf * 3, 0, 1)  # boost

            if conf >= self.conf_thresh:
                # Small padding
                pad = 5
                detections.append(Detection(
                    x1=max(0, x-pad), y1=max(0, y-pad),
                    x2=min(W, x+w+pad), y2=min(H, y+h+pad),
                    confidence=conf
                ))

        return detections

    def _grid_proposals(self, img: np.ndarray) -> List[Detection]:
        """
        YOLO-style grid proposals: divide image into S×S grid,
        apply anchor boxes, compute objectness from local feature statistics.
        """
        H, W   = img.shape[:2]
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        S      = 4   # 4×4 grid
        detections = []

        cell_h, cell_w = H // S, W // S
        for gi in range(S):
            for gj in range(S):
                cy = (gi + 0.5) * cell_h
                cx = (gj + 0.5) * cell_w

                # Cell statistics → objectness proxy
                roi = gray[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w]
                edge_d = float(cv2.Canny(roi.astype(np.uint8), 50, 150).mean()) / 255.0

                for aw, ah in self.ANCHORS:
                    bw = aw * W;  bh = ah * H
                    x1 = int(cx - bw/2); y1 = int(cy - bh/2)
                    x2 = int(cx + bw/2); y2 = int(cy + bh/2)
                    x1 = max(0,x1); y1 = max(0,y1)
                    x2 = min(W,x2); y2 = min(H,y2)
                    conf = edge_d * 1.5
                    if conf >= self.conf_thresh:
                        detections.append(Detection(x1,y1,x2,y2, min(conf,1.0)))

        return detections

    def detect(self, img: np.ndarray) -> List[Detection]:
        """
        Full detection pipeline: proposals → NMS → sorted by confidence.
        Returns list of Detection objects with bounding boxes.
        """
        H, W = img.shape[:2]
        all_dets = []

        # Method 1: Edge-based contour proposals (high precision)
        all_dets += self._edge_based_proposals(img)

        # Method 2: YOLO-style grid proposals (high recall)
        all_dets += self._grid_proposals(img)

        # NMS
        kept = nms(all_dets, self.iou_thresh)

        # Fallback: if nothing detected, use full image
        if not kept:
            kept = [Detection(0, 0, W, H, confidence=0.5, label="label")]

        # Sort by confidence descending
        kept.sort(key=lambda d: d.confidence, reverse=True)
        return kept[:5]  # top-5 detections

    def detect_and_crop(self, img: np.ndarray, target_size: int = 64) -> List[Tuple]:
        """
        Detect labels, crop and resize each to target_size × target_size.
        Returns: [(crop_img, detection), ...]
        """
        detections = self.detect(img)
        crops = []
        for det in detections:
            crop = det.crop(img)
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (target_size, target_size))
            crops.append((crop_resized, det))
        return crops

    def draw_detections(self, img: np.ndarray,
                        detections: List[Detection],
                        predictions: list = None) -> np.ndarray:
        """Visualize detections on image with bounding boxes."""
        vis    = img.copy()
        colors = {
            'GENUINE':     (0, 200, 80),
            'TAMPERED':    (0, 60, 220),
            'COUNTERFEIT': (200, 40, 200),
            'DEGRADED':    (0, 180, 220),
            'label':       (200, 200, 0),
        }
        for i, det in enumerate(detections):
            pred = predictions[i] if predictions and i < len(predictions) else None
            label_text = pred.get('label', det.label) if pred else det.label
            conf_text  = f"{det.confidence:.2f}"
            color      = colors.get(label_text, (200, 200, 200))

            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            txt = f"{label_text} {conf_text}"
            cv2.putText(vis, txt, (det.x1, max(det.y1-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return vis
