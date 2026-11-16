"""Train-only evidence stackers, selected exclusively by validation ROC-AUC."""
from __future__ import annotations
from torch import nn
from typing import Any
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()

def _logit_evidence(gnn_logits: torch.Tensor, llm_logits: torch.Tensor, anchor_logits: torch.Tensor) -> np.ndarray:
    logits = np.column_stack((_numpy(gnn_logits), _numpy(llm_logits), _numpy(anchor_logits)))
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return np.column_stack((
        logits,
        probabilities,
        logits[:, 1] - logits[:, 0],
        logits[:, 2] - logits[:, 0],
        np.abs(logits[:, 1] - logits[:, 0]),
        np.abs(logits[:, 2] - logits[:, 0]),
        logits[:, 0] * logits[:, 1],
        logits[:, 0] * logits[:, 2],
        logits[:, 1] * logits[:, 2]
    )).astype(np.float32, copy=False)

def _fusion_features(
    gnn_features: torch.Tensor,
    llm_features: torch.Tensor,
    anchor_features: torch.Tensor,
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    full: bool
) -> np.ndarray:
    evidence = _logit_evidence(gnn_logits, llm_logits, anchor_logits)
    if not full:
        return evidence
    return np.column_stack((evidence, _numpy(gnn_features), _numpy(llm_features), _numpy(anchor_features))).astype(np.float32, copy=False)

class EvidenceStackingFusion(nn.Module):
    uses_pair_features = True
    requires_anchor_llm = True
    requires_anchor_llm_features = True

    def __init__(self, estimator: Any, feature_mode: str):
        super().__init__()
        if feature_mode not in {'logit', 'full'}:
            raise ValueError(f'Unknown feature mode: {feature_mode}')
        self.estimator = estimator
        self.feature_mode = feature_mode

    @torch.no_grad()
    def predict(
        self,
        gnn_features: torch.Tensor,
        llm_features: torch.Tensor,
        anchor_features: torch.Tensor,
        gnn_logits: torch.Tensor,
        llm_logits: torch.Tensor,
        anchor_logits: torch.Tensor
    ) -> torch.Tensor:
        features = _fusion_features(
            gnn_features,
            llm_features,
            anchor_features,
            gnn_logits,
            llm_logits,
            anchor_logits,
            full=self.feature_mode == 'full'
        )
        probability = self.estimator.predict_proba(features)[:, 1]
        probability = np.clip(probability, 1e-06, 1.0 - 1e-06)
        return torch.from_numpy(np.log(probability / (1.0 - probability))).float()

@torch.no_grad()
def predict_fusion(
    model: EvidenceStackingFusion,
    gnn: torch.Tensor,
    llm: torch.Tensor,
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
    anchor_llm_logits: torch.Tensor | None=None,
    anchor_llm_features: torch.Tensor | None=None
) -> torch.Tensor:
    del device
    if anchor_llm_logits is None or anchor_llm_features is None:
        raise ValueError('Evidence stacking requires semantic-anchor logits and features')
    model.eval()
    output = []
    step = max(1, int(batch_size))
    for start in range(0, len(gnn_logits), step):
        selected = slice(start, start + step)
        output.append(model.predict(
            gnn[selected],
            llm[selected],
            anchor_llm_features[selected],
            gnn_logits[selected],
            llm_logits[selected],
            anchor_llm_logits[selected]
        ))
    return torch.cat(output) if output else torch.empty(0)

def _candidate_models(seed: int, sample_count: int, feature_count: int):
    for regularization in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        yield (
            f'logit-lr-c{regularization:g}',
            'logit',
            make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=3000))
        )
    maximum_components = min(sample_count - 1, feature_count)
    for components in (4, 8, 16, 32, 64, 96, 128):
        if components > maximum_components:
            continue
        for regularization in (0.01, 0.1, 1.0, 10.0):
            yield (
                f'pca{components}-lr-c{regularization:g}',
                'full',
                make_pipeline(
                    StandardScaler(),
                    PCA(n_components=components, whiten=True, random_state=seed),
                    LogisticRegression(C=regularization, max_iter=3000)
                )
            )
    for minimum_leaf in (5, 10, 20, 30):
        yield (
            f'extra-trees-leaf{minimum_leaf}',
            'full',
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=minimum_leaf,
                max_features='sqrt',
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )
        )

def train_fusion(
    args: Any,
    train_gnn: torch.Tensor,
    train_llm: torch.Tensor,
    train_gnn_logits: torch.Tensor,
    train_llm_logits: torch.Tensor,
    train_labels: torch.Tensor,
    val_gnn: torch.Tensor,
    val_llm: torch.Tensor,
    val_gnn_logits: torch.Tensor,
    val_llm_logits: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    metric_fn,
    train_anchor_logits: torch.Tensor | None=None,
    val_anchor_logits: torch.Tensor | None=None,
    train_anchor: torch.Tensor | None=None,
    val_anchor: torch.Tensor | None=None
) -> tuple[EvidenceStackingFusion, float]:
    if any((value is None for value in (train_anchor_logits, val_anchor_logits, train_anchor, val_anchor))):
        raise ValueError('Evidence stacking requires the semantic-anchor branch')
    train_logit = _fusion_features(
        train_gnn,
        train_llm,
        train_anchor,
        train_gnn_logits,
        train_llm_logits,
        train_anchor_logits,
        full=False
    )
    val_logit = _fusion_features(val_gnn, val_llm, val_anchor, val_gnn_logits, val_llm_logits, val_anchor_logits, full=False)
    train_full = _fusion_features(
        train_gnn,
        train_llm,
        train_anchor,
        train_gnn_logits,
        train_llm_logits,
        train_anchor_logits,
        full=True
    )
    val_full = _fusion_features(val_gnn, val_llm, val_anchor, val_gnn_logits, val_llm_logits, val_anchor_logits, full=True)
    labels = _numpy(train_labels).astype(np.int64, copy=False)
    validation_labels = val_labels.detach().cpu().float()
    seed = 1
    results: list[tuple[float, str, str, Any]] = []
    for (name, mode, estimator) in _candidate_models(seed, len(labels), train_full.shape[1]):
        training = train_full if mode == 'full' else train_logit
        validation = val_full if mode == 'full' else val_logit
        estimator.fit(training, labels)
        probability = np.clip(estimator.predict_proba(validation)[:, 1], 1e-06, 1.0 - 1e-06)
        logits = torch.from_numpy(np.log(probability / (1.0 - probability))).float()
        score = float(metric_fn(logits, validation_labels)['AUC'])
        if np.isfinite(score):
            results.append((score, name, mode, estimator))
    if not results:
        raise RuntimeError('No fusion candidate produced a finite validation AUC')
    (score, name, mode, estimator) = max(results, key=lambda item: item[0])
    model = EvidenceStackingFusion(estimator, mode)
    model.training_summary = {
        'selected': name,
        'architecture': 'validation-selected-symmetric-evidence-stacking',
        'feature_mode': mode,
        'best_validation_auc': score,
        'candidate_count': len(results),
        'selection_split': 'val',
        'fit_split': 'train',
        'gnn_anchor': False,
        'top_candidates': [{'name': candidate_name, 'validation_auc': float(candidate_score)} for (candidate_score, candidate_name, _, _) in sorted(results, reverse=True)[:5]]
    }
    threshold = 0.5
    print(f'  Fusion selected={name}; val AUC={score:.4f}; GNN anchor=false')
    print(f'  Fusion decision threshold={threshold:.4f}')
    return (model, threshold)
