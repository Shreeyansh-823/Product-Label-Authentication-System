"""
Label Authentication Model Pipeline
ViT + CNN-ViT Hybrid + Texture features → Ensemble (GBM+RF+MLP+SVC)
Optuna-style HPO · k-fold CV · Inferential statistics
"""

import numpy as np
import joblib, os, json, time
import scipy.stats as stats

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               VotingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA

from core.vit import ViT, HybridCNNViT
from core.feature_extractor import TextureFeatureExtractor
from core.data_gen import CLASS_NAMES
from augmentation.augment import AugmentationPipeline


# ─── Optuna-style Hyperparameter Optimization ─────────────────────────────────

def optuna_hpo(X_tr, y_tr, n_trials=12, seed=42):
    """
    Bayesian-inspired Optuna-style HPO using progressive random search.
    Optimizes GBM hyperparameters via macro-F1 3-fold CV.
    """
    rng = np.random.default_rng(seed)
    best_score, best_params = -1.0, {}
    history = []

    # Exploration bounds
    space = {
        "n_estimators":    [80, 120, 160, 200, 250],
        "learning_rate":   [0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
        "max_depth":       [3, 4, 5, 6],
        "subsample":       [0.75, 0.80, 0.85, 0.90, 1.0],
        "min_samples_leaf":[3, 5, 8, 12],
    }

    print(f"  [Optuna] {n_trials} trials …")
    for trial in range(n_trials):
        params = {k: rng.choice(v) for k,v in space.items()}
        clf    = GradientBoostingClassifier(**params, random_state=seed)
        scores = cross_val_score(clf, X_tr, y_tr, cv=3, scoring="f1_macro", n_jobs=1)
        m      = float(scores.mean())
        history.append({"trial": trial, "params": params, "f1": m})

        if m > best_score:
            best_score, best_params = m, params.copy()
        print(f"    Trial {trial+1:02d}/{n_trials}  F1={m:.4f}  best={best_score:.4f}", end="\r")

    print(f"\n  [Optuna] Best params → {best_params}  F1={best_score:.4f}")
    return best_params, history, best_score


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class LabelAuthPipeline:

    def __init__(self, use_hybrid=True, img_size=64, patch_size=8,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, seed=42):
        self.use_hybrid = use_hybrid
        self.img_size   = img_size
        self.d_model    = d_model

        # Feature extractors
        if use_hybrid:
            self.vit = HybridCNNViT(
                img_size=img_size, patch_size=patch_size//2,
                d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, seed=seed
            )
        else:
            self.vit = ViT(
                img_size=img_size, patch_size=patch_size,
                in_ch=3, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, seed=seed
            )

        self.tex     = TextureFeatureExtractor()
        self.aug     = AugmentationPipeline(seed=seed)
        self.scaler  = StandardScaler()
        self.pca     = PCA(n_components=40, random_state=seed)
        self.model   = None

        self.eval_results = {}
        self.hpo_history  = []
        self.hpo_best_f1  = 0.0
        self._seed        = seed

    # ── Feature Extraction ────────────────────────────────────────────

    def _feats(self, imgs):
        """Extract [ViT | Texture] features for each image."""
        all_f = []
        for img in imgs:
            vf = self.vit.extract_features(img)
            tf = self.tex.extract(img)
            all_f.append(np.concatenate([vf, tf]))
        return np.array(all_f, dtype=np.float32)

    # ── Training ──────────────────────────────────────────────────────

    def train(self, imgs, labels, run_hpo=True, augment_factor=1):
        t0 = time.time()
        print(f"  Extracting features from {len(imgs)} images …")

        X = self._feats(imgs)

        # Augment
        if augment_factor > 0:
            aug_imgs, aug_y = [], []
            for i, img in enumerate(imgs):
                for _ in range(augment_factor):
                    aug_imgs.append(self.aug(img, strong=False))
                    aug_y.append(labels[i])
            X_aug = self._feats(aug_imgs)
            X     = np.vstack([X, X_aug])
            labels_all = np.concatenate([labels, np.array(aug_y, dtype=np.int32)])
        else:
            labels_all = labels

        print(f"  Feature shape: {X.shape}  ({time.time()-t0:.1f}s)")

        X_s = self.scaler.fit_transform(X)
        X_p = self.pca.fit_transform(X_s)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_p, labels_all, test_size=0.2, stratify=labels_all, random_state=self._seed
        )

        # Optuna HPO
        if run_hpo:
            best_params, self.hpo_history, self.hpo_best_f1 = \
                optuna_hpo(X_tr, y_tr, n_trials=10, seed=self._seed)
        else:
            best_params = {"n_estimators":150,"learning_rate":0.1,
                           "max_depth":4,"subsample":0.85,"min_samples_leaf":5}

        # Ensemble
        self.model = self._build_ensemble(best_params)
        print("  Training ensemble classifier …")
        self.model.fit(X_tr, y_tr)

        y_pred  = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        self.eval_results["val"] = self._metrics(y_val, y_pred, y_proba)
        print(f"  Val  Acc={self.eval_results['val']['accuracy']:.4f}  "
              f"F1={self.eval_results['val']['f1_macro']:.4f}  "
              f"AUC={self.eval_results['val']['roc_auc']:.4f}")

    def _build_ensemble(self, best_params):
        """Soft-voting ensemble: GBM(3) + RF(2) + ET(2) + MLP(2) + SVC(1)."""
        gbm = GradientBoostingClassifier(**best_params, random_state=self._seed)
        rf  = RandomForestClassifier(n_estimators=150, max_depth=9,
                                     class_weight="balanced",
                                     random_state=self._seed, n_jobs=-1)
        et  = ExtraTreesClassifier(n_estimators=150, max_depth=9,
                                   class_weight="balanced",
                                   random_state=self._seed, n_jobs=-1)
        mlp = MLPClassifier(hidden_layer_sizes=(128,64,32),
                            activation="relu", max_iter=400,
                            random_state=self._seed)
        svc = CalibratedClassifierCV(
                SVC(kernel="rbf", C=8, gamma="scale", probability=False), cv=3)

        return VotingClassifier(
            estimators=[("gbm",gbm),("rf",rf),("et",et),("mlp",mlp),("svc",svc)],
            voting="soft", weights=[3,2,2,2,1]
        )

    # ── K-Fold CV + Inferential Statistics ───────────────────────────

    def kfold_cv(self, imgs, labels, k=5):
        print(f"\n  Running {k}-fold cross-validation …")
        X    = self._feats(imgs)
        X_s  = self.scaler.transform(X)
        X_p  = self.pca.transform(X_s)
        cv   = StratifiedKFold(n_splits=k, shuffle=True, random_state=self._seed)
        fold_res = []

        for fold, (tr_i, val_i) in enumerate(cv.split(X_p, labels)):
            clf = GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                             learning_rate=0.1, random_state=self._seed)
            clf.fit(X_p[tr_i], labels[tr_i])
            yp = clf.predict(X_p[val_i])
            yb = clf.predict_proba(X_p[val_i])
            m  = self._metrics(labels[val_i], yp, yb)
            fold_res.append(m)
            print(f"    Fold {fold+1}/{k}: Acc={m['accuracy']:.4f}  "
                  f"F1={m['f1_macro']:.4f}  AUC={m['roc_auc']:.4f}")

        keys = ["accuracy","precision_macro","recall_macro","f1_macro","roc_auc"]
        agg  = {k_: {"mean": float(np.mean([r[k_] for r in fold_res])),
                     "std":  float(np.std([r[k_]  for r in fold_res]))}
                for k_ in keys}

        # Inferential: one-sample t-test vs random-chance baseline (0.25)
        accs = [r["accuracy"] for r in fold_res]
        t_stat, p_val = stats.ttest_1samp(accs, 0.25)
        effect_size   = (np.mean(accs) - 0.25) / (np.std(accs) + 1e-9)

        # 95% confidence interval
        ci_lo, ci_hi = stats.t.interval(
            0.95, df=len(accs)-1, loc=np.mean(accs), scale=stats.sem(accs))

        self.eval_results["kfold"] = {
            "per_fold": fold_res, "aggregate": agg,
            "inferential": {
                "t_statistic":  float(t_stat),
                "p_value":      float(p_val),
                "significant":  bool(p_val < 0.05),
                "effect_size_d": float(effect_size),
                "ci_95":        [float(ci_lo), float(ci_hi)],
            }
        }

        print(f"\n  Aggregate ({k}-fold):")
        for k_, v in agg.items():
            print(f"    {k_:<20}: {v['mean']:.4f} ± {v['std']:.4f}")
        inf = self.eval_results["kfold"]["inferential"]
        print(f"  t={inf['t_statistic']:.2f}  p={inf['p_value']:.4e}  "
              f"Cohen's d={inf['effect_size_d']:.2f}  "
              f"CI95=[{inf['ci_95'][0]:.4f},{inf['ci_95'][1]:.4f}]")
        return agg

    # ── Prediction ────────────────────────────────────────────────────

    def predict(self, img: np.ndarray) -> dict:
        """Predict single image. Returns full classification result."""
        vf   = self.vit.extract_features(img)
        tf   = self.tex.extract(img)
        feat = np.concatenate([vf, tf]).reshape(1,-1)
        X_s  = self.scaler.transform(feat)
        X_p  = self.pca.transform(X_s)
        pred  = self.model.predict(X_p)[0]
        proba = self.model.predict_proba(X_p)[0]

        # Attention map for XAI
        try:
            attn = self.vit.get_attention_map(img).tolist()
        except Exception:
            attn = []

        return {
            "label":         CLASS_NAMES[pred],
            "class_id":      int(pred),
            "confidence":    float(proba[pred]),
            "probabilities": {CLASS_NAMES[i]: float(p) for i,p in enumerate(proba)},
            "attention_map": attn,
        }

    # ── Metrics ───────────────────────────────────────────────────────

    def _metrics(self, y_true, y_pred, y_proba):
        y_bin = label_binarize(y_true, classes=[0,1,2,3])
        return {
            "accuracy":         float(accuracy_score(y_true, y_pred)),
            "precision_macro":  float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
            "recall_macro":     float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
            "f1_macro":         float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
            "roc_auc":          float(roc_auc_score(y_bin,y_proba,multi_class="ovr",average="weighted")),
            "confusion_matrix": confusion_matrix(y_true,y_pred).tolist(),
            "per_class":        classification_report(
                                    y_true,y_pred,target_names=CLASS_NAMES,
                                    output_dict=True,zero_division=0),
        }

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, out_dir="models"):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.model,   f"{out_dir}/ensemble.pkl")
        joblib.dump(self.scaler,  f"{out_dir}/scaler.pkl")
        joblib.dump(self.pca,     f"{out_dir}/pca.pkl")
        joblib.dump(self.vit,     f"{out_dir}/vit.pkl")
        with open(f"{out_dir}/eval.json","w") as f:
            json.dump(self.eval_results, f, indent=2, default=str)
        with open(f"{out_dir}/hpo_history.json","w") as f:
            json.dump(self.hpo_history, f, indent=2, default=str)
        print(f"  Models saved → {out_dir}/")

    def load(self, out_dir="models"):
        self.model  = joblib.load(f"{out_dir}/ensemble.pkl")
        self.scaler = joblib.load(f"{out_dir}/scaler.pkl")
        self.pca    = joblib.load(f"{out_dir}/pca.pkl")
        self.vit    = joblib.load(f"{out_dir}/vit.pkl")
        try:
            with open(f"{out_dir}/eval.json") as f:
                self.eval_results = json.load(f)
        except Exception:
            pass
        print(f"  Models loaded ← {out_dir}/")
