"""SEGMDAL configuration, local paths and evaluation metrics."""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass, field
DATA_FILES = {
    'rna_sim': 'data/miRNA_total_similarity.csv',
    'drug_sim': 'data/drug_total_similarity.csv',
    'disease_sim': 'data/disease_total_similarity.csv',
    'rna_drug': 'data/miRNA_drug.csv',
    'rna_disease': 'data/miRNA_disease.csv',
    'drug_disease': 'data/drug_disease.csv'
}
NODE_FEATURE_FILES = {
    'rna': {
        'doc2vec': ('data/feature/miRNA/miRNA_doc2vec_features.csv', 'name', 128),
        'kmer': ('data/feature/miRNA/miRNA_kmer_features.csv', 'name', 64),
        'rnafm': ('data/feature/miRNA/miRNA_rnafm_features.csv', 'name', 640)
    },
    'drug': {
        'chemberta': ('data/feature/drug/chemberta_features.csv', 'drug_name', 768),
        'ecfp': ('data/feature/drug/ecfp_features.csv', 'drug_name', 2048),
        'gin': ('data/feature/drug/gin_features.csv', 'drug_name', 300)
    },
    'disease': {'ontology': ('data/feature/disease/disease_features.csv', 'disease_name', 3124)}
}

@dataclass(frozen=True)
class GraphConfig:
    folds: int = 5
    seed: int = 2027
    knn: int = 15
    similarity_threshold: float = 0.0
    partition_passes: int = 8
    negative_ratio: float = 1.0
    train_ratio: float = 0.64
    val_ratio: float = 0.16
    test_ratio: float = 0.2
    candidate_multiplier: int = 20
    entropy_eval_multiplier: int = 5
    easy_negative_ratio: float = 0.45
    hard_negative_ratio: float = 0.45
    random_negative_ratio: float = 0.1
    entropy_weight: float = 0.5
    module_weight: float = 0.3
    context_weight: float = 0.2

@dataclass(frozen=True)
class RiemannianGNNConfig:
    epochs: int = 200
    patience: int = 30
    hidden_dim: int = 128
    encoder_width: int = 256
    num_layers: int = 2
    dropout: float = 0.25
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    curvature_threshold: float = 0.15
    metric_consistency_weight: float = 0.05
    smoothness_weight: float = 0.05
    similarity_edge_scale: float = 1.5
    se_prior_weight: float = 0.9

@dataclass(frozen=True)
class GraphRAGConfig:
    topk: int = 10
    max_hops: int = 3
    min_hops: int = 2
    max_branch_per_step: int = 80
    max_module_changes: int = 2
    sampling_ratio: float = 0.8
    minhash_threshold: float = 0.85
    minhash_permutations: int = 64
    quality_strength: float = 1.5
    sampling_quality_mix: float = 0.9
    disease_freq_threshold: float = 50.0
    disease_freq_quantile: float = 0.9
    disease_penalty_alpha: float = 0.5
    disease_penalty_min: float = 0.25
    bridge_factor: float = 0.65
    degree_penalty_gamma: float = 0.05
    module_penalty_floor: float = 0.2
    disease_penalty_floor: float = 0.45
    degree_penalty_floor: float = 0.5

@dataclass(frozen=True)
class LLMConfig:
    epochs: int = 100
    patience: int = 30
    node_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 1e-05
    hard_negative_weight: float = 0.25
    ranking_weight: float = 0.02
    ranking_margin: float = 0.2
    se_path_dim: int = 32
    se_path_residual_weight: float = 0.8
    bridge_dim: int = 32
    lora_epochs: float = 3.0
    lora_batch_size: int = 8
    lora_gradient_accumulation_steps: int = 8
    lora_learning_rate: float = 0.0001
    lora_warmup_ratio: float = 0.03
    lora_max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    extract_batch_size: int = 8

@dataclass(frozen=True)
class SEGMDALConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    gnn: RiemannianGNNConfig = field(default_factory=RiemannianGNNConfig)
    rag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
MODEL_NAME = 'SEGMDAL'
CONFIG = SEGMDALConfig()
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
METRIC_POLICY = {
    'primary': 'validation ROC-AUC',
    'AUC': 'ROC-AUC',
    'AUPR': 'auc(recall, precision), full precision_recall_curve',
    'lora_selection': 'validation LM loss',
    'threshold': 0.5,
    'aggregation': 'five-fold equal mean, sample SD ddof=1'
}

def binary_metrics(logits, labels, threshold=0.5):
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError('Evaluation logits produced NaN or Inf probabilities')
    if np.unique(truth).size != 2:
        raise ValueError('Every evaluation split must contain both classes')
    predicted = (scores >= threshold).astype(np.int64)
    (precision, recall, _) = precision_recall_curve(truth, scores, drop_intermediate=False)
    return {
        'AUC': float(roc_auc_score(truth, scores)),
        'AUPR': float(auc(recall, precision)),
        'ACC': float(accuracy_score(truth, predicted)),
        'F1-Score': float(f1_score(truth, predicted, zero_division=0)),
        'Sensitivity': float(recall_score(truth, predicted, zero_division=0)),
        'Precision': float(precision_score(truth, predicted, zero_division=0))
    }
PROJECT_ROOT = Path(__file__).resolve().parent
LLM_BASE_MODEL = Path('E:/llama/Llama-3.2-1B')
DATA_ROOT = Path('.')
OUTPUT_ROOT = Path('outputs')


def project_path(path):
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

def classification_arguments():
    (g, h) = (CONFIG.gnn, CONFIG.llm)
    return Namespace(
        seed=CONFIG.graph.seed,
        batch_size=512,
        gnn_epochs=g.epochs,
        gnn_patience=g.patience,
        gnn_hidden_dim=g.hidden_dim,
        gnn_encoder_width=g.encoder_width,
        gnn_num_layers=g.num_layers,
        gnn_dropout=g.dropout,
        gnn_lr=g.learning_rate,
        gnn_weight_decay=g.weight_decay,
        curvature_threshold=g.curvature_threshold,
        similarity_edge_scale=g.similarity_edge_scale,
        metric_consistency_weight=g.metric_consistency_weight,
        smoothness_weight=g.smoothness_weight,
        se_prior_weight=g.se_prior_weight,
        llm_epochs=h.epochs,
        llm_patience=h.patience,
        llm_node_dim=h.node_dim,
        llm_hidden_dim=h.hidden_dim,
        llm_dropout=h.dropout,
        llm_lr=h.learning_rate,
        llm_weight_decay=h.weight_decay,
        llm_hard_negative_weight=h.hard_negative_weight,
        llm_ranking_weight=h.ranking_weight,
        llm_ranking_margin=h.ranking_margin,
        llm_bridge_dim=h.bridge_dim,
        llm_se_path_dim=h.se_path_dim,
        llm_se_path_residual_weight=h.se_path_residual_weight
    )

def lora_arguments(fold, graph_dir, directory, base_model, resume=None, device='cuda:0'):
    h = CONFIG.llm
    (corpus, adapter) = (directory / 'corpus', directory / 'lora_adapter')
    return Namespace(
        model_name_or_path=str(base_model),
        train_jsonl=str(corpus / 'train_corpus.jsonl'),
        eval_jsonl=str(corpus / 'val_corpus.jsonl'),
        node_vocab_json=str(corpus / 'node_vocab.json'),
        split_data=str(graph_dir / 'split_data.pt'),
        output_dir=str(adapter),
        epochs=h.lora_epochs,
        batch_size=h.lora_batch_size,
        eval_batch_size=h.lora_batch_size,
        gradient_accumulation_steps=h.lora_gradient_accumulation_steps,
        lr=h.lora_learning_rate,
        warmup_ratio=h.lora_warmup_ratio,
        max_length=h.lora_max_length,
        lora_r=h.lora_r,
        lora_alpha=h.lora_alpha,
        lora_dropout=0.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj',
        modules_to_save='embed_tokens,lm_head',
        weight_decay=0.0,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
        seed=CONFIG.graph.seed + fold,
        resume_from_checkpoint=resume,
        device=device
    )

def extraction_arguments(graph_dir, directory, base_model, device='cuda:0'):
    return Namespace(
        base_model_name_or_path=str(base_model),
        adapter_dir=str(directory / 'lora_adapter'),
        node_vocab_json=str(directory / 'corpus/node_vocab.json'),
        output_dir=str(directory),
        output_file=str(directory / 'llm_hidden.pt'),
        extract_prompts_jsonl=str(directory / 'corpus/extract_prompts.jsonl'),
        split_data=str(graph_dir / 'split_data.pt'),
        batch_size=CONFIG.llm.extract_batch_size,
        max_length=CONFIG.llm.lora_max_length,
        device=device
    )
