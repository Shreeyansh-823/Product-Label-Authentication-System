"""
Multi-Scale Feature Extractor
Combines ViT embeddings with handcrafted texture, frequency,
and structural features for robust label classification.
"""

import numpy as np
import cv2

TEXTURE_FEATURE_NAMES = [
    "edge_density","gradient_mean","gradient_std","gradient_max",
    "scratch_score","surface_roughness","laplacian_var",
    "bright_patch_ratio","corner_anomaly","peel_area",
    "reflectance_std","qr_pattern_score","microtext_density",
    "hf_energy","freq_entropy","lf_energy",
    "color_coherence","saturation_mean","hue_variance","color_shift",
    "entropy","kurtosis","skewness","edge_ratio",
    "r_mean","g_mean","b_mean","rg_diff","rb_diff",
]


class TextureFeatureExtractor:
    """29-dimensional handcrafted feature vector per image."""

    def extract(self, img: np.ndarray) -> np.ndarray:
        h, w   = img.shape[:2]
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        feats  = []

        # ── Edge & Gradient ───────────────────────────────────────────
        edges = cv2.Canny(img, 50, 150)
        feats.append(float(edges.mean()) / 255.0)

        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        feats += [float(mag.mean())/255, float(mag.std())/255,
                  float(mag.max())/255]

        # ── Surface / Scratch ─────────────────────────────────────────
        thresh = mag.mean() + 2.0*mag.std()
        feats.append(float((mag > thresh).mean()))

        kern   = np.ones((5,5), np.float32)/25
        mean_f = cv2.filter2D(gray,-1,kern)
        var_f  = cv2.filter2D(gray**2,-1,kern) - mean_f**2
        feats.append(float(var_f.mean()))

        lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F)
        feats.append(float(lap.var()) / 1000.0)

        # ── Adhesive / Peel ───────────────────────────────────────────
        bright = (gray > 215).astype(np.float32)
        feats.append(float(bright.mean()))

        qs = min(h,w)//4
        corners = [gray[:qs,:qs], gray[:qs,-qs:], gray[-qs:,:qs], gray[-qs:,-qs:]]
        feats.append(float(np.mean([c.std() for c in corners])))

        _, thresh2 = cv2.threshold(gray.astype(np.uint8), 210, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        peel = sum(cv2.contourArea(c) for c in cnts) / (h*w+1e-9)
        feats.append(float(peel))

        feats.append(float(np.std([img[:,:,c].mean() for c in range(3)])))

        # ── QR / Microtext / Frequency ────────────────────────────────
        _, bw = cv2.threshold(gray.astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        runs = np.abs(np.diff(bw.flatten().astype(int)))
        feats.append(float((runs>100).mean()))

        feats.append(float(np.abs(lap).mean()) / 255.0)

        fft  = np.abs(np.fft.fft2(gray))
        fftS = np.abs(np.fft.fftshift(fft))
        hmask = np.zeros_like(fftS, dtype=bool)
        hmask[:h//4,:] = True; hmask[-h//4:,:] = True
        hmask[:,:w//4] = True; hmask[:,-w//4:] = True
        feats.append(float(fftS[hmask].mean() / (fftS.mean()+1e-9)))

        fn  = fftS / (fftS.sum()+1e-9)
        fn  = fn[fn>1e-15]
        feats.append(float(-np.sum(fn*np.log(fn+1e-15))/(h*w)))

        lmask = ~hmask
        feats.append(float(fftS[lmask].mean() / (fftS.mean()+1e-9)))

        # ── Color ─────────────────────────────────────────────────────
        rm,gm,bm = img[:,:,2].mean(),img[:,:,1].mean(),img[:,:,0].mean()
        feats.append(float(np.std([rm,gm,bm])))
        feats.append(float(hsv[:,:,1].mean()))
        feats.append(float(hsv[:,:,0].var()))
        cen = img[h//4:3*h//4, w//4:3*w//4]
        feats.append(float(np.abs(img.mean((0,1))-cen.mean((0,1))).mean()))

        # ── Statistical ───────────────────────────────────────────────
        hist = cv2.calcHist([gray.astype(np.uint8)],[0],None,[256],[0,256]).flatten()
        hn   = hist/(hist.sum()+1e-9)
        feats.append(float(-np.sum(hn[hn>0]*np.log2(hn[hn>0]+1e-15))/8.0))

        gn = (gray - gray.mean())/(gray.std()+1e-9)
        feats.append(float(np.mean(gn**4)-3))
        feats.append(float(np.mean(gn**3)))
        feats.append(float((edges>0).sum()/(h*w)))

        # ── Per-channel means + diffs ─────────────────────────────────
        feats += [rm/255, gm/255, bm/255,
                  abs(rm-gm)/255, abs(rm-bm)/255]

        assert len(feats) == len(TEXTURE_FEATURE_NAMES), \
            f"Feature dim {len(feats)} != {len(TEXTURE_FEATURE_NAMES)}"
        return np.array(feats, dtype=np.float32)
