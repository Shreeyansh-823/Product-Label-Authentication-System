"""
Advanced Data Augmentation Pipeline
Implements geometric, photometric, and label-specific augmentations.
Generates diverse training data to improve model generalization.
"""

import numpy as np
import cv2
from typing import Tuple, List


class AugmentationPipeline:
    """
    Configurable augmentation pipeline with probability-weighted transforms.
    Mimics Albumentations-style API without the dependency.
    """

    def __init__(self, p=0.5, seed=None):
        self.p   = p
        self.rng = np.random.default_rng(seed)

    def _apply(self, img: np.ndarray, fn, p=None) -> np.ndarray:
        prob = p if p is not None else self.p
        return fn(img) if self.rng.random() < prob else img

    # ── Geometric Transforms ─────────────────────────────────────────

    def random_flip(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            img = cv2.flip(img, 1)   # horizontal
        if self.rng.random() < 0.15:
            img = cv2.flip(img, 0)   # vertical (rare for labels)
        return img

    def random_rotate(self, img: np.ndarray, max_deg=12) -> np.ndarray:
        h, w  = img.shape[:2]
        angle = self.rng.uniform(-max_deg, max_deg)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def random_scale_crop(self, img: np.ndarray, scale=(0.85, 1.15)) -> np.ndarray:
        h, w   = img.shape[:2]
        factor = self.rng.uniform(*scale)
        new_h  = max(8, int(h * factor))
        new_w  = max(8, int(w * factor))
        resized = cv2.resize(img, (new_w, new_h))
        # Center crop back to original size
        if new_h >= h and new_w >= w:
            y0 = (new_h - h) // 2
            x0 = (new_w - w) // 2
            return resized[y0:y0+h, x0:x0+w]
        else:
            canvas = np.full((h, w, img.shape[2]), 230, dtype=np.uint8)
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            canvas[y0:y0+new_h, x0:x0+new_w] = resized
            return canvas

    def random_shear(self, img: np.ndarray, shear=0.08) -> np.ndarray:
        h, w = img.shape[:2]
        sx   = self.rng.uniform(-shear, shear)
        M    = np.float32([[1, sx, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def random_perspective(self, img: np.ndarray, distort=0.08) -> np.ndarray:
        h, w = img.shape[:2]
        d    = int(min(h, w) * distort)
        rng  = self.rng
        src  = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst  = np.float32([
            [rng.integers(0,d), rng.integers(0,d)],
            [w-rng.integers(0,d), rng.integers(0,d)],
            [w-rng.integers(0,d), h-rng.integers(0,d)],
            [rng.integers(0,d), h-rng.integers(0,d)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ── Photometric Transforms ────────────────────────────────────────

    def random_brightness_contrast(self, img: np.ndarray,
                                    bright=(-40,40), contrast=(0.75,1.3)) -> np.ndarray:
        alpha = self.rng.uniform(*contrast)
        beta  = self.rng.uniform(*bright)
        out   = np.clip(img.astype(float) * alpha + beta, 0, 255).astype(np.uint8)
        return out

    def random_gamma(self, img: np.ndarray, gamma=(0.7, 1.5)) -> np.ndarray:
        g     = self.rng.uniform(*gamma)
        table = np.array([((i/255.0)**(1/g))*255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    def random_hsv(self, img: np.ndarray,
                   h_shift=15, s_scale=(0.7,1.3), v_scale=(0.8,1.2)) -> np.ndarray:
        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + self.rng.uniform(-h_shift, h_shift)) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * self.rng.uniform(*s_scale), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * self.rng.uniform(*v_scale),  0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_noise(self, img: np.ndarray, scale=10) -> np.ndarray:
        noise = self.rng.normal(0, self.rng.uniform(2, scale), img.shape)
        return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

    def random_blur(self, img: np.ndarray) -> np.ndarray:
        choice = self.rng.integers(3)
        if choice == 0:
            k = int(self.rng.choice([3, 5]))
            return cv2.GaussianBlur(img, (k,k), 0)
        elif choice == 1:
            k = int(self.rng.choice([3, 5]))
            return cv2.medianBlur(img, k)
        else:
            return cv2.bilateralFilter(img, 7, 50, 50)

    def random_jpeg_compression(self, img: np.ndarray, quality=(60,95)) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        q    = int(self.rng.uniform(*quality))
        enc  = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def random_channel_dropout(self, img: np.ndarray) -> np.ndarray:
        """Randomly zero out one color channel (simulates sensor failure)."""
        ch = self.rng.integers(3)
        out = img.copy()
        out[:,:,ch] = 0
        return out

    def cutout(self, img: np.ndarray, n_holes=1, hole_ratio=0.2) -> np.ndarray:
        """CutOut regularization: mask random square patches."""
        h, w = img.shape[:2]
        out  = img.copy()
        size = int(min(h, w) * hole_ratio)
        for _ in range(n_holes):
            y = int(self.rng.integers(0, h))
            x = int(self.rng.integers(0, w))
            y1, y2 = max(0, y-size//2), min(h, y+size//2)
            x1, x2 = max(0, x-size//2), min(w, x+size//2)
            out[y1:y2, x1:x2] = 128  # gray fill
        return out

    # ── Label-Specific Transforms ─────────────────────────────────────

    def simulate_scan_artifact(self, img: np.ndarray) -> np.ndarray:
        """Simulate scanner line artifacts."""
        out = img.copy()
        h   = img.shape[0]
        for _ in range(self.rng.integers(1, 4)):
            y = int(self.rng.integers(0, h))
            out[y, :] = np.clip(out[y, :].astype(int) + self.rng.integers(-60,60), 0, 255)
        return out

    def simulate_edge_glare(self, img: np.ndarray) -> np.ndarray:
        """Simulate lighting glare on label surface."""
        out = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        cx = self.rng.integers(w//4, 3*w//4)
        cy = self.rng.integers(h//4, 3*h//4)
        r  = self.rng.integers(h//6, h//3)
        Y, X = np.ogrid[:h, :w]
        dist  = np.sqrt((X-cx)**2 + (Y-cy)**2)
        glare = np.clip(1 - dist/r, 0, 1) * 80
        out  += glare[:,:,np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)

    # ── Full Pipeline ─────────────────────────────────────────────────

    def __call__(self, img: np.ndarray, strong=False) -> np.ndarray:
        """Apply random subset of augmentations."""
        p_geo   = 0.6 if strong else 0.4
        p_photo = 0.7 if strong else 0.5

        img = self._apply(img, self.random_flip,   p=0.5)
        img = self._apply(img, self.random_rotate,  p=p_geo)
        img = self._apply(img, self.random_scale_crop, p=p_geo*0.7)
        img = self._apply(img, self.random_shear,   p=p_geo*0.4)
        img = self._apply(img, self.random_perspective, p=p_geo*0.3)
        img = self._apply(img, self.random_brightness_contrast, p=p_photo)
        img = self._apply(img, self.random_hsv,     p=p_photo*0.7)
        img = self._apply(img, self.random_gamma,   p=p_photo*0.5)
        img = self._apply(img, self.random_noise,   p=p_photo*0.5)
        img = self._apply(img, self.random_blur,    p=0.3)
        img = self._apply(img, self.random_jpeg_compression, p=0.3)
        img = self._apply(img, self.cutout,         p=0.25)
        img = self._apply(img, self.simulate_scan_artifact,  p=0.2)
        img = self._apply(img, self.simulate_edge_glare,     p=0.15)
        return img

    def augment_batch(self, imgs: np.ndarray, strong=False) -> np.ndarray:
        return np.array([self(img, strong) for img in imgs], dtype=np.uint8)
