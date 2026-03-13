"""
Synthetic Label Image Generator
Generates 4 classes of product authentication labels with rich visual features:
  0 - GENUINE      : pristine labels with correct QR, hologram, microtext
  1 - TAMPERED     : physical damage, adhesive failure, ink smear
  2 - COUNTERFEIT  : QR distortions, font errors, color deviations
  3 - DEGRADED     : age, UV fade, water damage, cracking
"""

import numpy as np
import cv2
import os


CLASS_NAMES  = ["GENUINE", "TAMPERED", "COUNTERFEIT", "DEGRADED"]
CLASS_COLORS_BGR = {
    "GENUINE":     (80,  197, 34),
    "TAMPERED":    (68,  68,  239),
    "COUNTERFEIT": (247, 85,  168),
    "DEGRADED":    (36,  191, 251),
}


def _safe_rect(img, x, y, w, h):
    """Clamp coords to image bounds."""
    H, W = img.shape[:2]
    x1, y1 = int(x),        int(y)
    x2, y2 = min(W, x1+int(w)), min(H, y1+int(h))
    return x1, y1, x2, y2


def _draw_qr(img, ox, oy, size, distort=False, rng=None):
    """Draw a QR-code-like pattern."""
    if rng is None: rng = np.random.RandomState(42)
    cell = max(1, size // 8)
    grid = rng.randint(0, 2, (8, 8))
    H, W = img.shape[:2]
    for i in range(8):
        for j in range(8):
            v = 0 if grid[i,j] else 255
            if distort and rng.rand() < 0.35:
                v = int(rng.randint(60, 200))
            px1 = int(ox + j*cell); py1 = int(oy + i*cell)
            px2 = min(W-1, px1+cell); py2 = min(H-1, py1+cell)
            cv2.rectangle(img, (px1,py1), (px2,py2), (v,v,v), -1)
    # Finder patterns
    for fx, fy in [(ox,oy), (ox+size-3*cell,oy), (ox,oy+size-3*cell)]:
        fx,fy = int(fx), int(fy)
        for r_outer, r_inner, val in [(3*cell,3*cell-2,0),(3*cell-2,3*cell-4,255),(2,2,0)]:
            cv2.rectangle(img, (fx,fy), (min(W-1,fx+r_outer),min(H-1,fy+r_outer)),
                          (val,val,val), -1)


def generate_genuine(size=64, seed=0):
    rng = np.random.RandomState(seed)
    img = np.full((size,size,3), 248, dtype=np.uint8)

    # Clean border
    cv2.rectangle(img, (1,1), (size-2,size-2), (25,35,130), 2)
    # Header
    cv2.rectangle(img, (1,1), (size-2,15), (20,55,175), -1)
    # Header text (simulated)
    for xp in range(5, size-5, 5):
        cv2.line(img, (xp,5), (xp,12), (200,220,255), 1)

    # Text lines
    for yp in [22,29,36]:
        w = rng.randint(18, size-16)
        cv2.line(img, (6,yp), (6+w,yp), (45,45,60), 1)

    # QR code (crisp)
    qs = size//4
    _draw_qr(img, size-qs-3, size-qs-3, qs, distort=False, rng=rng)

    # Hologram diagonal lines
    for d in range(0, size*2, 7):
        x2, y2 = min(d, size-1), max(d-size,0)
        x1, y1 = max(d-size,0), min(d, size-1)
        cv2.line(img, (x1,y1), (x2,y2), (210,215,255), 1)

    # Microtext strip
    cv2.rectangle(img, (2, size-10), (size-3, size-3), (225,225,242), -1)
    for xp in range(4, size-4, 2):
        cv2.line(img, (xp, size-9), (xp, size-4), (110,110,160), 1)

    # Serial number dots
    for xp in range(6, size//2, 8):
        cv2.circle(img, (xp, 46), 1, (80,80,120), -1)

    img = cv2.GaussianBlur(img, (3,3), 0.3)
    return img


def generate_tampered(size=64, seed=0):
    rng = np.random.RandomState(seed)
    img = generate_genuine(size, seed)

    # Scratches
    for _ in range(rng.randint(3,9)):
        x1 = int(rng.randint(0, size)); y1 = int(rng.randint(0, size))
        x2 = int(np.clip(x1+rng.randint(-22,22), 0, size-1))
        y2 = int(np.clip(y1+rng.randint(-18,18), 0, size-1))
        cv2.line(img, (x1,y1), (x2,y2), (int(rng.randint(140,190)),)*3,
                 int(rng.randint(1,3)))

    # Adhesive failure (safe)
    for _ in range(rng.randint(1,4)):
        x = int(rng.randint(0, size-12)); y = int(rng.randint(0, size-12))
        x1,y1,x2,y2 = _safe_rect(img, x, y, rng.randint(6,13), rng.randint(5,12))
        if x2>x1 and y2>y1:
            c = (int(rng.randint(200,252)),int(rng.randint(185,232)),int(rng.randint(150,205)))
            fill = np.full((y2-y1, x2-x1, 3), c, dtype=np.uint8)
            img[y1:y2,x1:x2] = cv2.addWeighted(img[y1:y2,x1:x2], 0.5, fill, 0.5, 0)

    # Color shift in region
    if rng.rand() > 0.4:
        sx = int(rng.randint(0, size//2))
        patch = img[4:size-4, sx:].astype(int)
        patch += rng.randint(-45,45,3).tolist()
        img[4:size-4, sx:] = np.clip(patch,0,255).astype(np.uint8)

    # Edge peel
    pts = np.array([[0,0],[rng.randint(10,26),0],[0,rng.randint(10,26)]], np.int32)
    cv2.fillPoly(img, [pts], (232,222,202))

    # Ink smear
    x0 = int(rng.randint(5, size-20))
    r  = min(x0+16, size-1)
    roi = img[18:min(38,size), max(x0-2,0):r]
    if roi.size>0:
        img[18:min(38,size), max(x0-2,0):r] = cv2.filter2D(
            roi, -1, np.ones((3,7),np.float32)/21)

    return img


def generate_counterfeit(size=64, seed=0):
    rng = np.random.RandomState(seed)
    img = np.full((size,size,3), 242, dtype=np.uint8)

    # Slightly-off border
    bc = (int(rng.randint(18,50)), int(rng.randint(45,92)), int(rng.randint(140,215)))
    cv2.rectangle(img, (1,1), (size-2,size-2), bc, 2)

    # Off-shade header
    hc = (int(rng.randint(0,42)), int(rng.randint(38,82)), int(rng.randint(148,205)))
    cv2.rectangle(img, (1,1), (size-2,15), hc, -1)
    for xp in range(5, size-5, 5):
        v = int(rng.randint(150,220))
        cv2.line(img, (xp,5), (xp,12), (v,v,255), 1)

    # Misaligned text lines
    for yp in [22,29,36]:
        w   = int(rng.randint(12, size-13))
        off = int(rng.randint(-3,4))
        cv2.line(img, (6+off,yp+off), (6+w+off,yp+off), (85,65,65), 1)

    # Distorted QR
    qs = size//4
    _draw_qr(img, size-qs-3, size-qs-3, qs, distort=True, rng=rng)

    # Wavy font (counterfeit tells)
    for xp in range(5, size-5, 4):
        yo = int(2*np.sin(xp*0.55))
        cv2.circle(img, (xp, 45+yo), 1, (65,65,90), -1)

    # Sparse hologram
    for d in range(0, size*2, 14):
        x2, y2 = min(d, size-1), max(d-size,0)
        x1, y1 = max(d-size,0), min(d, size-1)
        cv2.line(img, (x1,y1), (x2,y2), (185,190,248), 1)

    img = cv2.GaussianBlur(img, (5,5), 1.1)
    noise = rng.randint(-18,18,img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16)+noise, 0,255).astype(np.uint8)
    return img


def generate_degraded(size=64, seed=0):
    rng = np.random.RandomState(seed)
    img = generate_genuine(size, seed)

    # Yellowing
    y = np.zeros_like(img)
    y[:,:,0] = int(rng.randint(8,42)); y[:,:,1] = int(rng.randint(4,26))
    img = np.clip(img.astype(int)+y, 0,255).astype(np.uint8)

    # Fade
    img = cv2.addWeighted(img, 0.62, np.full_like(img,205), 0.38, 0)

    # Water stains
    for _ in range(rng.randint(2,7)):
        cx,cy = int(rng.randint(5,size-5)), int(rng.randint(5,size-5))
        r     = int(rng.randint(4,14))
        ov    = img.copy()
        stain = (int(rng.randint(178,222)),int(rng.randint(168,212)),int(rng.randint(138,192)))
        cv2.ellipse(ov,(cx,cy),(r, r//2+int(rng.randint(0,5))),
                    int(rng.randint(0,181)),0,360,stain,-1)
        img = cv2.addWeighted(ov,0.38,img,0.62,0)

    # Cracks
    for _ in range(rng.randint(2,6)):
        pts=[]; x,y=int(rng.randint(0,size)),int(rng.randint(0,size))
        for _ in range(rng.randint(3,8)):
            x+=int(rng.randint(-9,10)); y+=int(rng.randint(-9,10))
            pts.append([int(np.clip(x,0,size-1)), int(np.clip(y,0,size-1))])
        if len(pts)>=2:
            cv2.polylines(img,[np.array(pts,np.int32)],False,(158,148,128),1)

    img = cv2.GaussianBlur(img,(3,3),0.7)
    return img


GENERATORS = [generate_genuine, generate_tampered, generate_counterfeit, generate_degraded]


def generate_dataset(n_per_class=200, img_size=64, output_dir=None, seed=42, verbose=True):
    rng = np.random.RandomState(seed)
    imgs, labels = [], []
    for cls_id, fn in enumerate(GENERATORS):
        for i in range(n_per_class):
            s   = int(rng.randint(0,1000000))
            img = fn(img_size, seed=s)
            imgs.append(img); labels.append(cls_id)
        if verbose:
            print(f"  Generated {n_per_class}× {CLASS_NAMES[cls_id]}")

    imgs   = np.array(imgs,   dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    idx    = np.random.permutation(len(labels))
    imgs, labels = imgs[idx], labels[idx]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/images.npy", imgs)
        np.save(f"{output_dir}/labels.npy", labels)
        if verbose:
            print(f"  Saved {len(labels)} samples → {output_dir}/")

    return imgs, labels


def save_sample_grid(imgs, labels, path="reports/sample_grid.png", n_per_class=6):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, n_per_class, figsize=(n_per_class*2, 10),
                             facecolor="#0f172a")
    fig.suptitle("Synthetic Label Dataset — All 4 Authenticity Classes",
                 color="white", fontsize=13, y=0.99)
    pal = ["#22c55e","#ef4444","#a855f7","#f59e0b"]

    for cls_id in range(4):
        cls_imgs = imgs[labels==cls_id][:n_per_class]
        for j, img in enumerate(cls_imgs):
            ax = axes[cls_id, j]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if j==0:
                ax.set_ylabel(CLASS_NAMES[cls_id], color=pal[cls_id],
                              fontsize=10, rotation=90, va="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  Sample grid → {path}")
