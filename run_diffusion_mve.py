"""Minimum Viable Experiment: Conditional Flow Matching vs baselines.

Tests whether a tiny CFM model can learn from ~229 resolved markets.
Compares against: market price baseline, logistic regression, regularized LightGBM.

Usage:
    python run_diffusion_mve.py
    python run_diffusion_mve.py --augment          # with counterfactual augmentation
    python run_diffusion_mve.py --device cuda       # force GPU
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add package paths (same pattern as run_multi_backtest.py)
for pkg in [
    "shared", "schemas", "market_ingest", "rules", "forecasting",
    "calibration", "execution", "training", "evidence", "diffusion",
]:
    sys.path.insert(0, str(Path(__file__).parent / "packages" / pkg / "src"))

import numpy as np
import pandas as pd
import torch
import structlog

from shared.config import BaseAppSettings
from shared.db import create_sync_engine, create_sync_session_factory
from calibration.calibrator import Calibrator, FEATURE_COLUMNS

from diffusion.model import DenoisingMLP
from diffusion.flow_matching import ConditionalFlowMatcher, CFMConfig
from diffusion.dataset import (
    ForecastDataset,
    build_dataset_from_db,
    temporal_cv_splits,
    augment_with_counterfactuals,
)
from diffusion.evaluate import (
    compute_metrics,
    reliability_diagram,
    bootstrap_brier_test,
)

logger = structlog.get_logger()

OUTPUT_DIR = Path("data/diffusion_mve")


def get_db():
    settings = BaseAppSettings()
    engine = create_sync_engine(settings.database_url_sync)
    factory = create_sync_session_factory(engine)
    return factory()


def train_and_eval_cfm(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
    config: CFMConfig,
    device: str,
    augment: bool = False,
) -> dict:
    """Train a CFM model on one fold and evaluate on validation set."""
    # Optionally augment training data
    if augment:
        train_features, train_labels = augment_with_counterfactuals(
            train_features, train_labels, n_augmented=3,
        )

    train_ds = ForecastDataset(train_features, train_labels, device=device)
    val_ds = ForecastDataset(val_features, val_labels, stats=train_ds.stats, device=device)

    feature_dim = train_ds.features.shape[1]
    model = DenoisingMLP(
        target_dim=1,
        feature_dim=feature_dim,
        hidden_dim=128,
        dropout=0.2,
    ).to(device)

    cfm = ConditionalFlowMatcher(model, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_brier = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        loss = cfm.train_epoch(train_ds.targets, train_ds.features, optimizer)
        scheduler.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_probs, val_samples = _predict_cfm(cfm, val_ds.features, device)
            val_brier = float(np.mean((val_probs - val_ds.labels_raw.cpu().numpy()) ** 2))

            if val_brier < best_val_brier:
                best_val_brier = val_brier
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.patience // 10:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    val_probs, val_samples = _predict_cfm(cfm, val_ds.features, device)
    val_labels_np = val_ds.labels_raw.cpu().numpy()
    metrics = compute_metrics(val_probs, val_labels_np, val_samples)

    return {
        "metrics": metrics,
        "probs": val_probs,
        "samples": val_samples,
        "model": model,
        "stats": train_ds.stats,
    }


def _predict_cfm(
    cfm: ConditionalFlowMatcher,
    features: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get probability predictions and samples from CFM model."""
    logit_samples = cfm.sample(features, n_samples=32)  # (N, 32, 1)
    prob_samples = torch.sigmoid(logit_samples).squeeze(-1).cpu().numpy()  # (N, 32)
    point_probs = prob_samples.mean(axis=1)  # (N,)
    return point_probs, prob_samples


def train_and_eval_logreg(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
) -> dict:
    """Train logistic regression baseline."""
    from sklearn.linear_model import LogisticRegression

    available_cols = [c for c in FEATURE_COLUMNS if c in train_features.columns]
    X_train = train_features[available_cols].fillna(0).values
    X_val = val_features[available_cols].fillna(0).values
    y_train = train_labels.values
    y_val = val_labels.values

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(val_probs, y_val)

    return {"metrics": metrics, "probs": val_probs}


def train_and_eval_lgbm(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
) -> dict:
    """Train regularized LightGBM baseline (Phase 0 config)."""
    cal = Calibrator()
    # Combine for the Calibrator.train API (it does its own internal val split,
    # but we pass the full train set and evaluate on our held-out val set)
    combined_features = pd.concat([train_features, val_features], ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels], ignore_index=True)
    cal.train(combined_features, combined_labels, val_fraction=len(val_labels) / len(combined_labels))

    # Predict on val set
    available_cols = [c for c in FEATURE_COLUMNS if c in val_features.columns]
    val_probs = []
    for i in range(len(val_features)):
        feats = {c: float(val_features.iloc[i][c]) for c in available_cols}
        out = cal.predict(feats)
        val_probs.append(out.calibrated_probability)

    val_probs = np.array(val_probs)
    metrics = compute_metrics(val_probs, val_labels.values)

    return {"metrics": metrics, "probs": val_probs}


def market_price_baseline(
    val_features: pd.DataFrame,
    val_labels: pd.Series,
) -> dict:
    """Market price baseline: use market_price as the prediction."""
    probs = val_features["market_price"].fillna(0.5).values.astype(float)
    probs = np.clip(probs, 0.01, 0.99)
    metrics = compute_metrics(probs, val_labels.values)
    return {"metrics": metrics, "probs": probs}


def constant_baseline(val_labels: pd.Series) -> dict:
    """Constant 0.5 baseline (no information)."""
    probs = np.full(len(val_labels), 0.5)
    metrics = compute_metrics(probs, val_labels.values)
    return {"metrics": metrics, "probs": probs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion MVE Experiment")
    parser.add_argument("--augment", action="store_true", help="Use counterfactual augmentation")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    # Load data from database
    print("\n--- Loading data from database ---")
    db = get_db()
    features_df, labels, close_times = build_dataset_from_db(db)
    db.close()

    n_total = len(labels)
    n_positive = int(labels.sum())
    print(f"Loaded {n_total} resolved markets ({n_positive} YES, {n_total - n_positive} NO)")
    print(f"Date range: {close_times.min()} to {close_times.max()}")

    if n_total < 30:
        print("ERROR: Need at least 30 resolved markets. Run more backtests first.")
        return

    # Cross-validation
    splits = temporal_cv_splits(n_total, n_folds=args.folds)
    print(f"\nRunning {len(splits)}-fold temporal cross-validation")

    config = CFMConfig(epochs=args.epochs, patience=30)

    # Accumulate per-fold results
    all_results: dict[str, list] = {
        "constant": [],
        "market_price": [],
        "logistic_regression": [],
        "lightgbm": [],
        "cfm": [],
    }
    if args.augment:
        all_results["cfm_augmented"] = []

    all_val_labels = []
    all_cfm_probs = []
    all_lgbm_probs = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(splits)}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"{'='*60}")

        train_feat = features_df.iloc[train_idx].reset_index(drop=True)
        train_lab = labels.iloc[train_idx].reset_index(drop=True)
        val_feat = features_df.iloc[val_idx].reset_index(drop=True)
        val_lab = labels.iloc[val_idx].reset_index(drop=True)

        # Constant baseline
        res = constant_baseline(val_lab)
        all_results["constant"].append(res)
        print(f"  Constant 0.5:    {res['metrics']}")

        # Market price baseline
        res = market_price_baseline(val_feat, val_lab)
        all_results["market_price"].append(res)
        print(f"  Market price:    {res['metrics']}")

        # Logistic regression
        res = train_and_eval_logreg(train_feat, train_lab, val_feat, val_lab)
        all_results["logistic_regression"].append(res)
        print(f"  Logistic reg:    {res['metrics']}")

        # LightGBM (regularized)
        res = train_and_eval_lgbm(train_feat, train_lab, val_feat, val_lab)
        all_results["lightgbm"].append(res)
        all_lgbm_probs.append(res["probs"])
        print(f"  LightGBM:        {res['metrics']}")

        # CFM (no augmentation)
        res = train_and_eval_cfm(
            train_feat, train_lab, val_feat, val_lab, config, device, augment=False,
        )
        all_results["cfm"].append(res)
        all_cfm_probs.append(res["probs"])
        print(f"  CFM:             {res['metrics']}")

        # CFM with augmentation
        if args.augment:
            res = train_and_eval_cfm(
                train_feat, train_lab, val_feat, val_lab, config, device, augment=True,
            )
            all_results["cfm_augmented"].append(res)
            print(f"  CFM (augmented): {res['metrics']}")

        all_val_labels.append(val_lab.values)

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean across folds)")
    print(f"{'='*60}")

    for model_name, fold_results in all_results.items():
        briers = [r["metrics"].brier_score for r in fold_results]
        log_losses = [r["metrics"].log_loss for r in fold_results]
        accs = [r["metrics"].accuracy for r in fold_results]
        print(
            f"  {model_name:25s} | Brier: {np.mean(briers):.4f} +/- {np.std(briers):.4f} | "
            f"LogLoss: {np.mean(log_losses):.4f} | Acc: {np.mean(accs):.1%}"
        )

    # Bootstrap significance test: CFM vs LightGBM
    if all_cfm_probs and all_lgbm_probs:
        cfm_all = np.concatenate(all_cfm_probs)
        lgbm_all = np.concatenate(all_lgbm_probs)
        labels_all = np.concatenate(all_val_labels)

        test = bootstrap_brier_test(cfm_all, lgbm_all, labels_all)
        print(f"\n--- Bootstrap Significance Test: CFM vs LightGBM ---")
        print(f"  Brier diff (CFM - LightGBM): {test['observed_diff']:.4f}")
        print(f"  95% CI: [{test['ci_low']:.4f}, {test['ci_high']:.4f}]")
        print(f"  p-value: {test['p_value']:.3f}")
        print(f"  CFM better: {test['a_better']}")

    # Save reliability diagram for CFM (using all folds combined)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if all_cfm_probs:
        cfm_all = np.concatenate(all_cfm_probs)
        labels_all = np.concatenate(all_val_labels)
        reliability_diagram(
            cfm_all, labels_all,
            save_path=str(OUTPUT_DIR / "reliability_cfm.png"),
            title="CFM Reliability Diagram (all folds)",
        )
        print(f"\nReliability diagram saved to {OUTPUT_DIR / 'reliability_cfm.png'}")

    # Save the final model (trained on all data except last 20%)
    print("\n--- Training final model on full dataset ---")
    n_val = max(1, int(n_total * 0.2))
    final_train_feat = features_df.iloc[:-n_val].reset_index(drop=True)
    final_train_lab = labels.iloc[:-n_val].reset_index(drop=True)

    if args.augment:
        final_train_feat, final_train_lab = augment_with_counterfactuals(
            final_train_feat, final_train_lab, n_augmented=3,
        )

    final_ds = ForecastDataset(final_train_feat, final_train_lab, device=device)
    feature_dim = final_ds.features.shape[1]

    final_model = DenoisingMLP(target_dim=1, feature_dim=feature_dim, hidden_dim=128).to(device)
    final_cfm = ConditionalFlowMatcher(final_model, config)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        final_cfm.train_epoch(final_ds.targets, final_ds.features, optimizer)
        scheduler.step()

    # Save
    model_path = OUTPUT_DIR / "cfm_mve.pt"
    torch.save(final_model.state_dict(), str(model_path))
    final_ds.stats.save(str(OUTPUT_DIR / "cfm_mve.stats.npz"))
    print(f"Final model saved to {model_path}")
    print(f"Parameters: {sum(p.numel() for p in final_model.parameters()):,}")

    print("\n--- MVE Complete ---")
    print("Next steps:")
    print("  - If CFM >= LogReg on Brier: proceed to Phase 2 (evidence conditioning)")
    print("  - If CFM < Constant 0.5:     accumulate more data before neural approaches")


if __name__ == "__main__":
    main()
