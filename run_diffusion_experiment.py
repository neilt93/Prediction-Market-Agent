"""Full diffusion experiment: Phase 1 (fixed) + Phase 2 (evidence-conditioned) + benchmarks.

Compares:
  - Constant 0.5 baseline
  - Market price baseline
  - Logistic regression (13 features)
  - Platt scaling (on raw_probability)
  - LightGBM (regularized, Phase 0)
  - CFM Phase 1 (13 features, fixed normalization)
  - CFM Phase 2 (13 features + 384D title embedding, FiLM conditioning)
  - CFM Phase 2 + augmentation

Usage:
    python run_diffusion_experiment.py
    python run_diffusion_experiment.py --device cuda --folds 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

from diffusion.model import DenoisingMLP, FiLMDenoisingMLP, ConditionEncoder
from diffusion.flow_matching import ConditionalFlowMatcher, CFMConfig, sigmoid
from diffusion.dataset import (
    ForecastDataset,
    build_dataset_from_db,
    temporal_cv_splits,
    augment_with_counterfactuals,
)
from diffusion.flow_matching import logit as flow_logit
from diffusion.evaluate import (
    compute_metrics,
    reliability_diagram,
    bootstrap_brier_test,
    EvalMetrics,
)

logger = structlog.get_logger()
OUTPUT_DIR = Path("data/diffusion_experiment")


def get_db():
    settings = BaseAppSettings()
    engine = create_sync_engine(settings.database_url_sync)
    factory = create_sync_session_factory(engine)
    return factory()


# ---------------------------------------------------------------------------
# Title embedding computation
# ---------------------------------------------------------------------------

_ENCODER = None

def get_title_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


def encode_titles(titles: list[str], device: str = "cpu") -> torch.Tensor:
    """Encode market titles into 384D embeddings using frozen sentence-transformer."""
    encoder = get_title_encoder()
    embeddings = encoder.encode(titles, show_progress_bar=False, convert_to_numpy=True)
    return torch.tensor(embeddings, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Phase 2 dataset: features + title embeddings
# ---------------------------------------------------------------------------

class Phase2Dataset:
    """Combines numerical features with title embeddings for FiLM conditioning."""

    def __init__(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        titles: list[str],
        stats=None,
        label_smoothing: float = 0.025,
        device: str = "cpu",
    ):
        self.feat_ds = ForecastDataset(
            features_df, labels, stats=stats,
            label_smoothing=label_smoothing, device=device,
        )
        self.stats = self.feat_ds.stats
        self.title_embeddings = encode_titles(titles, device=device)  # (N, 384)
        self.features = self.feat_ds.features
        self.targets = self.feat_ds.targets
        self.labels_raw = self.feat_ds.labels_raw

    def __len__(self):
        return len(self.feat_ds)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def _market_prior(features: torch.Tensor, device: str) -> torch.Tensor:
    """Sample from market-price-informed prior for Phase 2 FiLM training."""
    mp = features[:, 0].clamp(0.01, 0.99)  # market_price is first feature (passthrough)
    mu = flow_logit(mp).unsqueeze(-1)
    sigma = (4.0 * mp * (1.0 - mp)).unsqueeze(-1).clamp(min=0.1)
    return mu + sigma * torch.randn(features.shape[0], 1, device=device)


def _predict_cfm(cfm, features, device):
    logit_samples = cfm.sample(features, n_samples=32, n_steps=5)
    prob_samples = sigmoid(logit_samples).squeeze(-1).cpu().numpy()
    return prob_samples.mean(axis=1), prob_samples


def train_cfm_phase1(train_feat, train_lab, val_feat, val_lab, config, device, augment=False):
    """Phase 1: DenoisingMLP on 13 features (fixed normalization)."""
    if augment:
        train_feat, train_lab = augment_with_counterfactuals(train_feat, train_lab, n_augmented=3)

    train_ds = ForecastDataset(train_feat, train_lab, device=device)
    val_ds = ForecastDataset(val_feat, val_lab, stats=train_ds.stats, device=device)

    model = DenoisingMLP(target_dim=1, feature_dim=train_ds.features.shape[1], hidden_dim=128, dropout=0.2).to(device)
    cfm = ConditionalFlowMatcher(model, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_brier = float("inf")
    best_state = None
    patience = 0

    for epoch in range(config.epochs):
        cfm.train_epoch(train_ds.targets, train_ds.features, optimizer)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            probs, _ = _predict_cfm(cfm, val_ds.features, device)
            brier = float(np.mean((probs - val_ds.labels_raw.cpu().numpy()) ** 2))
            if brier < best_val_brier:
                best_val_brier = brier
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= config.patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    probs, samples = _predict_cfm(cfm, val_ds.features, device)
    return {"metrics": compute_metrics(probs, val_ds.labels_raw.cpu().numpy(), samples), "probs": probs, "samples": samples}


def train_cfm_phase2(
    train_feat, train_lab, train_titles,
    val_feat, val_lab, val_titles,
    config, device, augment=False,
):
    """Phase 2: FiLM-conditioned model with title embeddings."""
    if augment:
        train_feat, train_lab = augment_with_counterfactuals(train_feat, train_lab, n_augmented=3)
        # Repeat titles for augmented samples
        n_orig = len(train_titles)
        n_aug = len(train_feat) - n_orig
        aug_titles = []
        for i in range(n_aug):
            aug_titles.append(train_titles[i % n_orig])
        train_titles = list(train_titles) + aug_titles

    train_ds = Phase2Dataset(train_feat, train_lab, train_titles, device=device)
    val_ds = Phase2Dataset(val_feat, val_lab, val_titles, stats=train_ds.stats, device=device)

    feature_dim = train_ds.features.shape[1]
    condition_dim = 128

    # Condition encoder: 384D title + 13D features -> 128D
    cond_encoder = ConditionEncoder(
        evidence_dim=384, context_dim=0, feature_dim=feature_dim, output_dim=condition_dim,
    ).to(device)
    # Override the input projection to match actual input dims (384 + feature_dim, no context)
    cond_input_dim = 384 + feature_dim
    cond_encoder.net = torch.nn.Sequential(
        torch.nn.Linear(cond_input_dim, condition_dim),
        torch.nn.LayerNorm(condition_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(condition_dim, condition_dim),
    ).to(device)

    denoiser = FiLMDenoisingMLP(
        target_dim=1, condition_dim=condition_dim, time_dim=64,
        hidden_dim=128, n_layers=2, dropout=0.2,
    ).to(device)

    all_params = list(denoiser.parameters()) + list(cond_encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_brier = float("inf")
    best_denoiser_state = None
    best_encoder_state = None
    patience = 0

    for epoch in range(config.epochs):
        denoiser.train()
        cond_encoder.train()
        optimizer.zero_grad()

        # Build condition vectors
        # Use zeros for context_embedding since we don't have separate context
        cond = cond_encoder.net(torch.cat([train_ds.title_embeddings, train_ds.features], dim=-1))

        # CFM loss with market-price-informed prior
        B = train_ds.targets.shape[0]
        x_0 = _market_prior(train_ds.features, device)
        t = torch.rand(B, device=device)
        t_expand = t.unsqueeze(-1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * train_ds.targets
        target_v = train_ds.targets - x_0
        pred_v = denoiser(x_t, t, cond)
        loss = torch.mean((pred_v - target_v) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            denoiser.eval()
            cond_encoder.eval()
            with torch.no_grad():
                val_cond = cond_encoder.net(torch.cat([val_ds.title_embeddings, val_ds.features], dim=-1))
                probs, samples = _sample_film(denoiser, val_cond, device, features=val_ds.features)
            brier = float(np.mean((probs - val_ds.labels_raw.cpu().numpy()) ** 2))
            if brier < best_val_brier:
                best_val_brier = brier
                best_denoiser_state = {k: v.clone() for k, v in denoiser.state_dict().items()}
                best_encoder_state = {k: v.clone() for k, v in cond_encoder.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= config.patience // 10:
                break

    if best_denoiser_state:
        denoiser.load_state_dict(best_denoiser_state)
        cond_encoder.net.load_state_dict(best_encoder_state)

    denoiser.eval()
    cond_encoder.eval()
    with torch.no_grad():
        val_cond = cond_encoder.net(torch.cat([val_ds.title_embeddings, val_ds.features], dim=-1))
        probs, samples = _sample_film(denoiser, val_cond, device)
    return {"metrics": compute_metrics(probs, val_ds.labels_raw.cpu().numpy(), samples), "probs": probs, "samples": samples}


def _sample_film(denoiser, condition, device, features=None, n_samples=32, n_steps=5):
    """Sample from FiLM model with market-price-informed prior."""
    B = condition.shape[0]
    cond_expanded = condition.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

    # Market-price prior
    if features is not None:
        mp = features[:, 0].clamp(0.01, 0.99)  # market_price
        mu = flow_logit(mp)
        sigma = (4.0 * mp * (1.0 - mp)).clamp(min=0.1)
        mu_exp = mu.unsqueeze(1).expand(B, n_samples).reshape(B * n_samples, 1)
        sigma_exp = sigma.unsqueeze(1).expand(B, n_samples).reshape(B * n_samples, 1)
        z = mu_exp + sigma_exp * torch.randn(B * n_samples, 1, device=device)
    else:
        z = torch.randn(B * n_samples, 1, device=device)

    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B * n_samples,), i * dt, device=device)
        v = denoiser(z, t, cond_expanded)
        z = z + dt * v
    prob_samples = torch.sigmoid(z).view(B, n_samples).cpu().numpy()
    return prob_samples.mean(axis=1), prob_samples


def train_lgbm(train_feat, train_lab, val_feat, val_lab):
    cal = Calibrator()
    combined = pd.concat([train_feat, val_feat], ignore_index=True)
    combined_lab = pd.concat([train_lab, val_lab], ignore_index=True)
    cal.train(combined, combined_lab, val_fraction=len(val_lab) / len(combined_lab))
    available_cols = [c for c in FEATURE_COLUMNS if c in val_feat.columns]
    probs = np.array([cal.predict({c: float(val_feat.iloc[i][c]) for c in available_cols}).calibrated_probability for i in range(len(val_feat))])
    return {"metrics": compute_metrics(probs, val_lab.values), "probs": probs}


def train_logreg(train_feat, train_lab, val_feat, val_lab):
    from sklearn.linear_model import LogisticRegression
    cols = [c for c in FEATURE_COLUMNS if c in train_feat.columns]
    X_tr = train_feat[cols].fillna(0).values
    X_val = val_feat[cols].fillna(0).values
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_tr, train_lab.values)
    probs = model.predict_proba(X_val)[:, 1]
    return {"metrics": compute_metrics(probs, val_lab.values), "probs": probs}


def platt_scaling(train_feat, train_lab, val_feat, val_lab):
    """Platt scaling: logistic regression on raw_probability only."""
    from sklearn.linear_model import LogisticRegression
    X_tr = train_feat[["raw_probability"]].fillna(0.5).values
    X_val = val_feat[["raw_probability"]].fillna(0.5).values
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_tr, train_lab.values)
    probs = model.predict_proba(X_val)[:, 1]
    return {"metrics": compute_metrics(probs, val_lab.values), "probs": probs}


def market_price_baseline(val_feat, val_lab):
    probs = np.clip(val_feat["market_price"].fillna(0.5).values.astype(float), 0.01, 0.99)
    return {"metrics": compute_metrics(probs, val_lab.values), "probs": probs}


def constant_baseline(val_lab):
    probs = np.full(len(val_lab), 0.5)
    return {"metrics": compute_metrics(probs, val_lab.values), "probs": probs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diffusion Full Experiment")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    print("\n--- Loading data ---")
    db = get_db()
    features_df, labels, close_times, titles = build_dataset_from_db(db)
    db.close()

    n = len(labels)
    print(f"Loaded {n} resolved markets ({int(labels.sum())} YES, {n - int(labels.sum())} NO)")
    print(f"Date range: {close_times.min()} to {close_times.max()}")

    if n < 30:
        print("ERROR: Need at least 30 resolved markets.")
        return

    print("\n--- Computing title embeddings ---")
    t0 = time.time()
    _ = encode_titles(titles[:1])  # warm up model
    all_title_embs = encode_titles(titles, device="cpu")
    print(f"Encoded {len(titles)} titles in {time.time() - t0:.1f}s, shape={all_title_embs.shape}")

    splits = temporal_cv_splits(n, n_folds=args.folds)
    print(f"\n{len(splits)}-fold temporal CV")

    config = CFMConfig(epochs=args.epochs, patience=30)

    models = [
        "constant", "market_price", "platt_scaling", "logistic_regression",
        "lightgbm", "cfm_phase1", "cfm_phase1_aug", "cfm_phase2", "cfm_phase2_aug",
    ]
    all_results: dict[str, list] = {m: [] for m in models}
    all_probs: dict[str, list] = {m: [] for m in models}
    all_labels_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx+1}/{len(splits)}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"{'='*70}")

        tr_f = features_df.iloc[train_idx].reset_index(drop=True)
        tr_l = labels.iloc[train_idx].reset_index(drop=True)
        va_f = features_df.iloc[val_idx].reset_index(drop=True)
        va_l = labels.iloc[val_idx].reset_index(drop=True)
        tr_t = [titles[i] for i in train_idx]
        va_t = [titles[i] for i in val_idx]

        all_labels_list.append(va_l.values)

        # Baselines
        for name, fn in [
            ("constant", lambda: constant_baseline(va_l)),
            ("market_price", lambda: market_price_baseline(va_f, va_l)),
            ("platt_scaling", lambda: platt_scaling(tr_f, tr_l, va_f, va_l)),
            ("logistic_regression", lambda: train_logreg(tr_f, tr_l, va_f, va_l)),
            ("lightgbm", lambda: train_lgbm(tr_f, tr_l, va_f, va_l)),
        ]:
            res = fn()
            all_results[name].append(res)
            all_probs[name].append(res["probs"])
            print(f"  {name:25s} {res['metrics']}")

        # Phase 1 CFM (fixed normalization)
        res = train_cfm_phase1(tr_f, tr_l, va_f, va_l, config, device, augment=False)
        all_results["cfm_phase1"].append(res)
        all_probs["cfm_phase1"].append(res["probs"])
        print(f"  {'cfm_phase1':25s} {res['metrics']}")

        res = train_cfm_phase1(tr_f, tr_l, va_f, va_l, config, device, augment=True)
        all_results["cfm_phase1_aug"].append(res)
        all_probs["cfm_phase1_aug"].append(res["probs"])
        print(f"  {'cfm_phase1_aug':25s} {res['metrics']}")

        # Phase 2 CFM (with title embeddings + FiLM)
        res = train_cfm_phase2(tr_f, tr_l, tr_t, va_f, va_l, va_t, config, device, augment=False)
        all_results["cfm_phase2"].append(res)
        all_probs["cfm_phase2"].append(res["probs"])
        print(f"  {'cfm_phase2':25s} {res['metrics']}")

        res = train_cfm_phase2(tr_f, tr_l, tr_t, va_f, va_l, va_t, config, device, augment=True)
        all_results["cfm_phase2_aug"].append(res)
        all_probs["cfm_phase2_aug"].append(res["probs"])
        print(f"  {'cfm_phase2_aug':25s} {res['metrics']}")

    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (mean +/- std across folds)")
    print(f"{'='*70}")
    for name in models:
        folds = all_results[name]
        briers = [r["metrics"].brier_score for r in folds]
        lls = [r["metrics"].log_loss for r in folds]
        accs = [r["metrics"].accuracy for r in folds]
        uncs = [r["metrics"].mean_uncertainty for r in folds]
        covs = [r["metrics"].interval_coverage for r in folds]
        print(
            f"  {name:25s} | Brier: {np.mean(briers):.4f}+/-{np.std(briers):.4f} | "
            f"LL: {np.mean(lls):.4f} | Acc: {np.mean(accs):.1%} | "
            f"Unc: {np.mean(uncs):.3f} | Cov: {np.mean(covs):.1%}"
        )

    # Bootstrap significance: best CFM vs LightGBM, best CFM vs market price
    labels_all = np.concatenate(all_labels_list)
    print(f"\n--- Bootstrap Significance Tests (N={len(labels_all)}) ---")
    for cfm_name in ["cfm_phase1", "cfm_phase1_aug", "cfm_phase2", "cfm_phase2_aug"]:
        cfm_p = np.concatenate(all_probs[cfm_name])
        for baseline in ["lightgbm", "market_price"]:
            bl_p = np.concatenate(all_probs[baseline])
            test = bootstrap_brier_test(cfm_p, bl_p, labels_all)
            direction = "BETTER" if test["a_better"] else "worse"
            sig = "*" if test["p_value"] < 0.05 else ""
            print(
                f"  {cfm_name:20s} vs {baseline:15s}: "
                f"diff={test['observed_diff']:+.4f} p={test['p_value']:.3f}{sig} ({direction})"
            )

    # Reliability diagrams
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["cfm_phase1", "cfm_phase2", "lightgbm", "market_price"]:
        p = np.concatenate(all_probs[name])
        reliability_diagram(
            p, labels_all,
            save_path=str(OUTPUT_DIR / f"reliability_{name}.png"),
            title=f"Reliability: {name}",
        )
    print(f"\nReliability diagrams saved to {OUTPUT_DIR}/")

    # Per-fold detail table
    print(f"\n--- Per-Fold Brier Scores ---")
    header = f"{'Fold':>5}"
    for name in models:
        header += f" | {name[:12]:>12}"
    print(header)
    for fold_idx in range(len(splits)):
        row = f"{fold_idx+1:>5}"
        for name in models:
            b = all_results[name][fold_idx]["metrics"].brier_score
            row += f" | {b:>12.4f}"
        print(row)

    print("\n--- Experiment Complete ---")


if __name__ == "__main__":
    main()
