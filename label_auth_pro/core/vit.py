"""
Vision Transformer (ViT) + Hybrid CNN-ViT Architecture
Built from scratch in NumPy — no PyTorch/TF required.

Implements:
  - PatchEmbedding with learnable CLS token & positional encodings
  - Multi-Head Self-Attention with scaled dot-product
  - Transformer Encoder blocks (GELU, LayerNorm, residuals)
  - Attention Rollout map for XAI visualization
  - CNN stem (conv-like pooling) for Hybrid CNN-ViT
  - Efficient batch processing

Reference: Dosovitskiy et al. "An Image is Worth 16x16 Words" (2020)
"""

import numpy as np
import joblib, os


# ── Activations ──────────────────────────────────────────────────────────────

def gelu(x):
    """Gaussian Error Linear Unit — smoother than ReLU."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

def layer_norm(x, gamma, beta, eps=1e-6):
    mu    = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return gamma * (x - mu) / (sigma + eps) + beta


# ── CNN Stem (Hybrid head) ────────────────────────────────────────────────────

class CNNStem:
    """
    Lightweight CNN stem for feature extraction before ViT encoder.
    Simulates 2 conv layers with average pooling using NumPy operations.
    Extracts local spatial features (edges, textures) more efficiently than raw patches.
    """
    def __init__(self, in_ch=3, out_ch=32, seed=42):
        rng = np.random.default_rng(seed)
        # Two 3x3 filters per output channel (simplified)
        self.filters1 = rng.normal(0, 0.1, (out_ch, in_ch, 3, 3)).astype(np.float32)
        self.bias1    = np.zeros(out_ch, dtype=np.float32)
        self.out_ch   = out_ch

    def _conv2d(self, img, filters, bias):
        """Manual 2D convolution with zero padding."""
        C_in, H, W = img.shape
        C_out, _, kH, kW = filters.shape
        pH, pW = kH // 2, kW // 2
        padded = np.pad(img, ((0,0),(pH,pH),(pW,pW)), mode='reflect')
        out = np.zeros((C_out, H, W), dtype=np.float32)
        for f in range(C_out):
            for c in range(C_in):
                for i in range(kH):
                    for j in range(kW):
                        out[f] += filters[f, c, i, j] * padded[c, i:i+H, j:j+W]
            out[f] += bias[f]
        return np.maximum(0, out)  # ReLU

    def forward(self, img: np.ndarray) -> np.ndarray:
        """img: (H,W,C) → feature_map: (H//2, W//2, out_ch)"""
        # Transpose to (C,H,W)
        x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = self._conv2d(x, self.filters1, self.bias1)  # (out_ch, H, W)
        # Average pool 2x2
        C, H, W = x.shape
        x = x[:, :H//2*2, :W//2*2].reshape(C, H//2, 2, W//2, 2).mean(axis=(2,4))
        return x.transpose(1, 2, 0)  # (H//2, W//2, out_ch)


# ── Patch Embedding ───────────────────────────────────────────────────────────

class PatchEmbedding:
    """
    Splits image (or CNN feature map) into non-overlapping patches,
    linearly projects each patch, prepends CLS token, adds positional embedding.
    """
    def __init__(self, img_size=64, patch_size=8, in_ch=3, d_model=128, seed=42):
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.patch_dim  = patch_size * patch_size * in_ch
        self.d_model    = d_model

        rng = np.random.default_rng(seed)
        s   = np.sqrt(2.0 / self.patch_dim)
        self.proj_W    = rng.normal(0, s, (self.patch_dim, d_model)).astype(np.float32)
        self.proj_b    = np.zeros(d_model, dtype=np.float32)
        self.cls_token = rng.normal(0, 0.02, (1, d_model)).astype(np.float32)
        # Sinusoidal positional encoding
        self.pos_embed = self._sinusoidal_pe(self.n_patches + 1, d_model)

    def _sinusoidal_pe(self, seq_len, d_model):
        pe   = np.zeros((seq_len, d_model), dtype=np.float32)
        pos  = np.arange(seq_len).reshape(-1, 1)
        div  = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model//2])
        return pe

    def _extract_patches(self, img):
        H, W, C = img.shape
        ps = self.patch_size
        patches = []
        for i in range(0, H - H % ps, ps):
            for j in range(0, W - W % ps, ps):
                patches.append(img[i:i+ps, j:j+ps, :].flatten())
        return np.array(patches, dtype=np.float32)

    def forward(self, img: np.ndarray) -> np.ndarray:
        """img: (H,W,C) float [0,1] → tokens: (N+1, d_model)"""
        patches = self._extract_patches(img)        # (N, patch_dim)
        x       = patches @ self.proj_W + self.proj_b  # (N, d_model)
        x       = np.vstack([self.cls_token, x])    # (N+1, d_model)
        return x + self.pos_embed[:len(x)]          # add PE


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MHSA:
    """Multi-Head Self-Attention with scaled dot-product."""
    def __init__(self, d_model=128, n_heads=4, dropout=0.0, seed=42):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.drop    = dropout

        rng = np.random.default_rng(seed)
        s   = np.sqrt(2.0 / d_model)
        self.Wq = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.Wk = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.Wv = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        self.Wo = rng.normal(0, s, (d_model, d_model)).astype(np.float32)
        for b in ['bq','bk','bv','bo']:
            setattr(self, b, np.zeros(d_model, dtype=np.float32))

    def forward(self, x, training=False):
        """x: (seq, d_model) → (seq, d_model), attn_weights"""
        seq = x.shape[0]
        Q = (x @ self.Wq + self.bq).reshape(seq, self.n_heads, self.d_k).transpose(1,0,2)
        K = (x @ self.Wk + self.bk).reshape(seq, self.n_heads, self.d_k).transpose(1,0,2)
        V = (x @ self.Wv + self.bv).reshape(seq, self.n_heads, self.d_k).transpose(1,0,2)

        scores = (Q @ K.transpose(0,2,1)) / np.sqrt(self.d_k)  # (heads, seq, seq)
        attn   = softmax(scores, axis=-1)

        if training and self.drop > 0:
            mask = np.random.rand(*attn.shape) > self.drop
            attn = attn * mask / (1 - self.drop + 1e-9)

        out = (attn @ V).transpose(1,0,2).reshape(seq, self.d_model)
        return out @ self.Wo + self.bo, attn


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FFN:
    def __init__(self, d_model=128, d_ff=512, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/d_model),  (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff,    dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/d_ff),     (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


# ── Transformer Encoder Block ─────────────────────────────────────────────────

class TransformerBlock:
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1, seed=42):
        self.attn = MHSA(d_model, n_heads, dropout, seed)
        self.ffn  = FFN(d_model, d_ff, seed+1)
        # Pre-norm (more stable than post-norm)
        self.g1 = np.ones(d_model,  dtype=np.float32)
        self.b1 = np.zeros(d_model, dtype=np.float32)
        self.g2 = np.ones(d_model,  dtype=np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x, training=False):
        # Pre-LN attention
        xn, attn = self.attn.forward(layer_norm(x, self.g1, self.b1), training)
        x = x + xn
        # Pre-LN FFN
        x = x + self.ffn.forward(layer_norm(x, self.g2, self.b2))
        return x, attn


# ── Full Vision Transformer ───────────────────────────────────────────────────

class ViT:
    """
    Pure-NumPy Vision Transformer.
    forward(img) → (cls_feat: d_model,)
    get_attention_map(img) → (grid_h, grid_w) heatmap via attention rollout.
    """
    def __init__(self, img_size=64, patch_size=8, in_ch=3,
                 d_model=128, n_heads=4, n_layers=4, d_ff=256,
                 dropout=0.1, seed=42):
        self.img_size   = img_size
        self.patch_size = patch_size
        self.d_model    = d_model
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, d_model, seed)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout, seed + i)
            for i in range(n_layers)
        ]
        self.norm_g = np.ones(d_model,  dtype=np.float32)
        self.norm_b = np.zeros(d_model, dtype=np.float32)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Normalize uint8 image to float [0,1]."""
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        return img

    def forward(self, img: np.ndarray, training=False):
        """Returns (cls_feat, list_of_attn_maps)."""
        x     = self.patch_embed.forward(self._preprocess(img))
        attns = []
        for blk in self.blocks:
            x, attn = blk.forward(x, training)
            attns.append(attn)
        x = layer_norm(x, self.norm_g, self.norm_b)
        return x[0], attns  # CLS token

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        feat, _ = self.forward(img)
        return feat

    def batch_extract(self, imgs) -> np.ndarray:
        return np.array([self.extract_features(img) for img in imgs], dtype=np.float32)

    def get_attention_map(self, img: np.ndarray) -> np.ndarray:
        """Attention rollout (Abnar & Zuidema 2020) → patch-grid heatmap."""
        _, attns = self.forward(img)
        n_patches = self.patch_embed.n_patches
        ps        = int(np.sqrt(n_patches))
        R = np.eye(n_patches + 1)
        for attn in attns:
            a = attn.mean(0)             # avg over heads
            a = a + np.eye(a.shape[-1]) # add residual
            a /= a.sum(-1, keepdims=True) + 1e-9
            R  = a @ R
        mask = R[0, 1:]  # CLS → patch weights
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
        return mask.reshape(ps, ps)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


# ── Hybrid CNN-ViT ────────────────────────────────────────────────────────────

class HybridCNNViT:
    """
    CNN stem extracts local features, then ViT encoder processes them.
    Better for small images: CNN captures fine-grained textures,
    ViT captures long-range relationships between label regions.
    """
    def __init__(self, img_size=64, patch_size=8, d_model=128,
                 n_heads=4, n_layers=4, d_ff=256, seed=42):
        self.cnn_stem    = CNNStem(in_ch=3, out_ch=32, seed=seed)
        cnn_out_size     = img_size // 2  # after 2x2 pooling
        self.vit         = ViT(
            img_size=cnn_out_size, patch_size=patch_size//2,
            in_ch=32, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, seed=seed
        )

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        cnn_feat = self.cnn_stem.forward(img)          # (H//2, W//2, 32)
        vit_feat = self.vit.extract_features(cnn_feat) # (d_model,)
        return vit_feat

    def batch_extract(self, imgs) -> np.ndarray:
        return np.array([self.extract_features(img) for img in imgs], dtype=np.float32)

    def get_attention_map(self, img: np.ndarray) -> np.ndarray:
        cnn_feat = self.cnn_stem.forward(img)
        return self.vit.get_attention_map(cnn_feat)
