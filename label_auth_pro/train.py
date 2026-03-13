"""
Label Authentication System — Training Entry Point
============================================================
Runs the complete pipeline:
  1. Synthetic dataset generation (4 classes)
  2. EDA — texture, edge, FFT, color channel analysis
  3. ViT / Hybrid CNN-ViT feature extraction
  4. Optuna hyperparameter tuning
  5. Ensemble training (GBM + RF + ET + MLP + SVC)
  6. k-fold cross-validation + inferential statistics
  7. Evaluation plots (confusion matrix, ROC, per-class, k-fold)
  8. Attention map visualization (XAI)
  9. Blockchain provenance chain + LLM-embedding RAG

Usage:
  python train.py                    # full run
  python train.py --fast             # quick demo (fewer samples, no augment)
"""

import os, sys, json, argparse, time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use("Agg")

from core.data_gen             import generate_dataset, save_sample_grid, CLASS_NAMES
from pipeline.model_pipeline   import LabelAuthPipeline
from evaluation.visualize      import run_eda, plot_results, plot_attention_maps
from utils.blockchain_rag      import build_blockchain, EmbeddingRAG


def banner(msg):
    print("\n" + "─"*62)
    print(f"  {msg}")
    print("─"*62)


def main(fast=False):
    t_start = time.time()
    N        = 80  if fast else 200
    AUG      = 0   if fast else 1
    KFOLD    = 3   if fast else 5
    HPO      = not fast

    for d in ["models","reports/eda","reports","data"]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Dataset ────────────────────────────────────────────────────
    banner("STEP 1/8 — Generating Synthetic Label Dataset")
    imgs, labels = generate_dataset(n_per_class=N, img_size=64,
                                    output_dir="data", seed=42, verbose=True)
    save_sample_grid(imgs, labels, path="reports/sample_grid.png")
    print(f"  Total: {len(imgs)} images × {len(CLASS_NAMES)} classes")

    # ── 2. EDA ───────────────────────────────────────────────────────
    banner("STEP 2/8 — Exploratory Data Analysis")
    run_eda(imgs, labels, out_dir="reports/eda")

    # ── 3. Train ─────────────────────────────────────────────────────
    banner("STEP 3/8 — Training Hybrid CNN-ViT + Ensemble (Optuna HPO)")
    pipeline = LabelAuthPipeline(
        use_hybrid=True, img_size=64, patch_size=8,
        d_model=128, n_heads=4, n_layers=3, d_ff=256, seed=42
    )
    pipeline.train(imgs, labels, run_hpo=HPO, augment_factor=AUG)

    # ── 4. K-Fold ─────────────────────────────────────────────────────
    banner(f"STEP 4/8 — {KFOLD}-Fold Cross-Validation + Inferential Stats")
    pipeline.kfold_cv(imgs, labels, k=KFOLD)

    # ── 5. Save ───────────────────────────────────────────────────────
    banner("STEP 5/8 — Saving Models")
    pipeline.save("models")

    # ── 6. Eval Plots ─────────────────────────────────────────────────
    banner("STEP 6/8 — Evaluation Plots")
    plot_results(pipeline.eval_results, out_dir="reports")

    # ── 7. Attention Maps ─────────────────────────────────────────────
    banner("STEP 7/8 — ViT Attention Rollout Maps (XAI)")
    plot_attention_maps(imgs, labels, pipeline, out_dir="reports", n=4)

    # ── 8. Blockchain + RAG ───────────────────────────────────────────
    banner("STEP 8/8 — Blockchain Provenance + RAG Knowledge Base")
    blockchain = build_blockchain(n_products=25, rng_seed=42)
    print(f"  Chain: {len(blockchain.chain)} blocks, "
          f"{len(blockchain.records)} products, valid={blockchain.verify()}")
    rag = EmbeddingRAG()
    rag.build(blockchain)
    rag.save("models/rag.pkl")

    # ── Summary ───────────────────────────────────────────────────────
    v  = pipeline.eval_results.get("val", {})
    kf = pipeline.eval_results.get("kfold", {}).get("aggregate", {})
    inf= pipeline.eval_results.get("kfold", {}).get("inferential", {})
    elapsed = time.time() - t_start

    print("\n" + "═"*62)
    print("  ✅  TRAINING COMPLETE")
    print("═"*62)
    print(f"  Val Accuracy   : {v.get('accuracy',0):.4f}")
    print(f"  Val Precision  : {v.get('precision_macro',0):.4f}")
    print(f"  Val Recall     : {v.get('recall_macro',0):.4f}")
    print(f"  Val F1 Macro   : {v.get('f1_macro',0):.4f}")
    print(f"  Val ROC-AUC    : {v.get('roc_auc',0):.4f}")
    if kf:
        print(f"  {KFOLD}-Fold Accuracy: {kf['accuracy']['mean']:.4f} ± {kf['accuracy']['std']:.4f}")
        print(f"  {KFOLD}-Fold F1 Macro: {kf['f1_macro']['mean']:.4f} ± {kf['f1_macro']['std']:.4f}")
        print(f"  {KFOLD}-Fold ROC-AUC : {kf['roc_auc']['mean']:.4f} ± {kf['roc_auc']['std']:.4f}")
    if inf:
        print(f"  t-statistic    : {inf.get('t_statistic',0):.4f}")
        print(f"  p-value        : {inf.get('p_value',0):.4e}  (significant={inf.get('significant')})")
        print(f"  Cohen's d      : {inf.get('effect_size_d',0):.4f}")
        ci = inf.get("ci_95",[0,0])
        print(f"  95% CI         : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Total time     : {elapsed:.1f}s")
    print(f"\n  Models saved   : models/")
    print(f"  Reports        : reports/")
    print(f"\n  Launch API     : python api/app.py  →  http://localhost:5001")
    print("═"*62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick demo run (fewer samples, no HPO)")
    args = parser.parse_args()
    main(fast=args.fast)
