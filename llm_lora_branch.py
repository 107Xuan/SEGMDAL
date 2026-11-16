"""Main-model LoRA stages, causal hidden extraction, semantic anchor and SE evidence head."""
from __future__ import annotations
from pathlib import Path
from torch import nn
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import argparse
import copy
import hashlib
import inspect
import json
import math
import numpy as np
import os
import re
import shutil
import time
import torch
import torch.nn.functional as F
from config import CONFIG

def configure_torch_determinism(enabled: bool) -> None:
    if not enabled:
        return
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False

def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')
NODE_TYPES = ('rna', 'drug', 'disease')
SE_PATH_FEATURE_NAMES = (
    'shared_disease_evidence',
    'soft_se_evidence',
    'soft_se_to_raw_ratio',
    'intra_module_evidence_ratio',
    'hub_adjusted_evidence_ratio',
    'shared_disease_module_count',
    'module_contribution_entropy',
    'endpoint_same_module',
    'module_cohesion_prior',
    'module_change_penalty',
    'rna_similarity_metapath',
    'drug_similarity_metapath',
    'two_sided_similarity_metapath',
    'similarity_metapath_agreement',
    'low_entropy_metapath_score',
    'low_entropy_metapath_consistency'
)

def _node_tables(value: Any, field: str, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not set(NODE_TYPES).issubset(value):
        raise KeyError(f'{path} is missing one or more {field} node tables: {NODE_TYPES}')
    return {node_type: torch.as_tensor(value[node_type], dtype=torch.float32) for node_type in NODE_TYPES}

def load_llm_embeddings(root: Path, fold: int) -> dict[str, dict[str, torch.Tensor]]:
    path = root / f'fold_{fold}' / 'llm_hidden.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing LLM embedding file: {path}. Set --llm-cache-dir to the directory containing fold_*/llm_hidden.pt.')
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise TypeError(f'Invalid LLM embedding payload: {path}')
    llm = _node_tables(payload.get('llm_node_embeddings'), 'llm_node_embeddings', path)
    for node_type in NODE_TYPES:
        if llm[node_type].ndim != 2:
            raise ValueError(f'LLM table must be rank-2 for {node_type} in {path}')
        if not torch.isfinite(llm[node_type]).all():
            raise ValueError(f'Non-finite LLM embeddings for {node_type} in {path}')
    llm_dims = {node_type: int(table.shape[1]) for (node_type, table) in llm.items()}
    if len(set(llm_dims.values())) != 1:
        raise ValueError(f'Node-type LLM embedding dimensions differ: {llm_dims}')
    return {'llm': llm}

def batch_indices(size: int, batch_size: int, shuffle: bool) -> Iterable[torch.Tensor]:
    order = torch.randperm(size) if shuffle else torch.arange(size)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]

class LLMNodeProjection(nn.Module):

    def __init__(self, llm_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(llm_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout))
        self.refine = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout)
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, llm: torch.Tensor) -> torch.Tensor:
        base = self.projection(llm)
        return self.output_norm(base + self.refine(base))

def _modules_for_node_type(graph: Any, node_type: str, count: int) -> torch.Tensor:
    partition = torch.as_tensor(np.asarray(graph.global_partition), dtype=torch.long)
    global_ids = []
    for index in range(count):
        if hasattr(graph, 'get_global_id'):
            global_id = int(graph.get_global_id(node_type, index))
        else:
            global_id = int(graph.global_node_id.get((node_type, index), -1))
        if global_id < 0 or global_id >= len(partition):
            raise IndexError(f'Missing global module for {node_type}:{index}')
        global_ids.append(global_id)
    return partition[torch.tensor(global_ids, dtype=torch.long)]

def _soften_tensor(raw: torch.Tensor, floor: float) -> torch.Tensor:
    residual = min(max(float(floor), 0.0), 1.0)
    return residual + (1.0 - residual) * raw.clamp(0.0, 1.0)

def _similarity_transition(graph: Any, node_type: str, node_count: int, modules: torch.Tensor, module_penalty_floor: float) -> torch.Tensor:
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float32)
    relation = graph.sim_edges.get(node_type)
    if relation is not None and relation[0].numel():
        (edge_index, edge_weight) = relation
        source = edge_index[0].long().cpu()
        target = edge_index[1].long().cpu()
        weight = edge_weight.detach().cpu().float().clamp_min(0.0)
        same_module = modules[source].eq(modules[target]).float()
        module_gate = float(module_penalty_floor) + (1.0 - float(module_penalty_floor)) * same_module
        adjacency[source, target] = torch.maximum(adjacency[source, target], weight * module_gate)
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-08)

def _log_unit_scale(values: torch.Tensor) -> torch.Tensor:
    transformed = torch.log1p(values.clamp_min(0.0))
    return transformed / transformed.amax().clamp_min(1e-08)

def _structural_entropy_path_features(
    graph: Any,
    rna_adjacency: torch.Tensor,
    drug_adjacency: torch.Tensor,
    *,
    max_module_changes: int,
    module_penalty_floor: float,
    disease_penalty_floor: float,
    degree_penalty_floor: float
) -> torch.Tensor:
    (rna_count, disease_count) = rna_adjacency.shape
    drug_count = drug_adjacency.shape[0]
    rna_modules = _modules_for_node_type(graph, 'rna', rna_count)
    drug_modules = _modules_for_node_type(graph, 'drug', drug_count)
    disease_modules = _modules_for_node_type(graph, 'disease', disease_count)
    disease_frequency = (rna_adjacency.gt(0).sum(dim=0) + drug_adjacency.gt(0).sum(dim=0)).float()
    positive_frequency = disease_frequency[disease_frequency > 0]
    quantile_threshold = float(np.quantile(positive_frequency.numpy(), CONFIG.rag.disease_freq_quantile)) if positive_frequency.numel() else 0.0
    frequency_threshold = max(float(CONFIG.rag.disease_freq_threshold), quantile_threshold)
    disease_raw = torch.ones_like(disease_frequency)
    frequent = disease_frequency > frequency_threshold
    if frequent.any() and CONFIG.rag.disease_penalty_alpha > 0:
        disease_raw[frequent] = (frequency_threshold / disease_frequency[frequent].clamp_min(1.0)).pow(float(CONFIG.rag.disease_penalty_alpha)).clamp(min=float(CONFIG.rag.disease_penalty_min), max=1.0)
    disease_soft = _soften_tensor(disease_raw, disease_penalty_floor)
    disease_degree = torch.zeros(disease_count, dtype=torch.float32)
    sim_relation = graph.sim_edges.get('disease')
    if sim_relation is not None and sim_relation[0].numel():
        source = sim_relation[0][0].long().cpu()
        disease_degree += torch.bincount(source, minlength=disease_count).float()
    for ((src_type, _), relation) in graph.inter_edges.items():
        if src_type != 'disease' or not relation[0].numel():
            continue
        source = relation[0][0].long().cpu()
        disease_degree += torch.bincount(source, minlength=disease_count).float()
    degree_raw = 1.0 / (1.0 + float(CONFIG.rag.degree_penalty_gamma) * torch.log1p(disease_degree))
    degree_soft = _soften_tensor(degree_raw, degree_penalty_floor)
    rna_root = rna_adjacency.clamp_min(0.0).sqrt()
    drug_root = drug_adjacency.clamp_min(0.0).sqrt()
    raw_evidence = rna_root @ drug_root.transpose(0, 1)
    hub_evidence = rna_root * degree_soft.unsqueeze(0) @ drug_root.transpose(0, 1)
    se_evidence = torch.zeros_like(raw_evidence)
    intra_evidence = torch.zeros_like(raw_evidence)
    module_contributions = []
    for module in torch.unique(disease_modules, sorted=True).tolist():
        disease_mask = disease_modules == int(module)
        if not disease_mask.any():
            continue
        rna_slice = rna_root[:, disease_mask]
        drug_slice = drug_root[:, disease_mask]
        contribution = rna_slice @ drug_slice.transpose(0, 1)
        module_contributions.append(contribution)
        weighted = rna_slice * (disease_soft[disease_mask] * degree_soft[disease_mask]).unsqueeze(0) @ drug_slice.transpose(0, 1)
        module_changes = rna_modules[:, None].ne(int(module)).long() + drug_modules[None, :].ne(int(module)).long()
        module_raw = torch.where(
            module_changes == 0,
            torch.ones_like(weighted),
            torch.full_like(weighted, float(CONFIG.rag.bridge_factor))
        )
        excess = (module_changes - int(max_module_changes)).clamp_min(0).float()
        if excess.any():
            module_raw = module_raw * float(CONFIG.rag.bridge_factor) ** excess
        se_evidence += weighted * _soften_tensor(module_raw, module_penalty_floor)
        same_module = rna_modules[:, None].eq(int(module)) & drug_modules[None, :].eq(int(module))
        intra_evidence += contribution * same_module.float()
    if module_contributions:
        contribution_stack = torch.stack(module_contributions, dim=0)
        active_modules = contribution_stack.gt(0).sum(dim=0).float()
        probabilities = contribution_stack / raw_evidence.clamp_min(1e-08).unsqueeze(0)
        module_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=0)
        entropy_scale = active_modules.clamp_min(2.0).log()
        module_entropy = torch.where(active_modules > 1, module_entropy / entropy_scale, torch.zeros_like(module_entropy))
    else:
        active_modules = torch.zeros_like(raw_evidence)
        module_entropy = torch.zeros_like(raw_evidence)
    raw_scale = torch.log1p(raw_evidence).amax().clamp_min(1.0)
    se_scale = torch.log1p(se_evidence).amax().clamp_min(1.0)
    module_scale = float(np.log1p(max(1, len(module_contributions))))
    endpoint_same = rna_modules[:, None].eq(drug_modules[None, :]).float()
    module_cohesion_prior = torch.exp(-0.45 * (1.0 - endpoint_same))
    module_change_penalty = torch.where(
        endpoint_same > 0.5,
        torch.ones_like(endpoint_same),
        torch.full_like(endpoint_same, float(CONFIG.rag.bridge_factor))
    )
    rna_transition = _similarity_transition(graph, 'rna', rna_count, rna_modules, module_penalty_floor)
    drug_transition = _similarity_transition(graph, 'drug', drug_count, drug_modules, module_penalty_floor)
    association = torch.as_tensor(np.asarray(graph.rna_drug_matrix), dtype=torch.float32)
    rna_path = rna_transition @ association
    drug_path = association @ drug_transition.transpose(0, 1)
    two_sided_path = rna_transition @ association @ drug_transition.transpose(0, 1)
    rna_path_unit = _log_unit_scale(rna_path)
    drug_path_unit = _log_unit_scale(drug_path)
    two_sided_unit = _log_unit_scale(two_sided_path)
    metapath_agreement = torch.sqrt((rna_path_unit * drug_path_unit).clamp_min(0.0))
    low_entropy_score = (0.35 * rna_path_unit + 0.25 * drug_path_unit + 0.2 * two_sided_unit + 0.2 * metapath_agreement) * (0.65 + 0.35 * module_cohesion_prior)
    metapath_consistency = (1.0 - (rna_path_unit - drug_path_unit).abs()) * low_entropy_score.gt(0).float()
    features = torch.stack(
        (
            torch.log1p(raw_evidence) / raw_scale,
            torch.log1p(se_evidence) / se_scale,
            se_evidence / raw_evidence.clamp_min(1e-08),
            intra_evidence / raw_evidence.clamp_min(1e-08),
            hub_evidence / raw_evidence.clamp_min(1e-08),
            torch.log1p(active_modules) / max(module_scale, 1.0),
            module_entropy,
            endpoint_same,
            module_cohesion_prior,
            module_change_penalty,
            rna_path_unit,
            drug_path_unit,
            two_sided_unit,
            metapath_agreement,
            low_entropy_score,
            metapath_consistency
        ),
        dim=-1
    ).clamp(0.0, 1.0).contiguous()
    if features.shape != (rna_count, drug_count, len(SE_PATH_FEATURE_NAMES)):
        raise RuntimeError(f'Unexpected SE path feature shape: {tuple(features.shape)}')
    if not torch.isfinite(features).all():
        raise FloatingPointError('Structural-entropy path features contain non-finite values')
    return features

def attach_disease_mediator_context(
    embeddings: dict[str, dict[str, torch.Tensor]],
    graph: Any,
    *,
    max_module_changes: int=CONFIG.rag.max_module_changes,
    module_penalty_floor: float=CONFIG.rag.module_penalty_floor,
    disease_penalty_floor: float=CONFIG.rag.disease_penalty_floor,
    degree_penalty_floor: float=CONFIG.rag.degree_penalty_floor
) -> dict[str, Any]:
    disease = embeddings['llm']['disease'].detach().cpu().float()
    mediator: dict[str, torch.Tensor] = {}
    membership: dict[str, torch.Tensor] = {}
    for node_type in ('rna', 'drug'):
        node_count = int(embeddings['llm'][node_type].shape[0])
        context = torch.zeros((node_count, disease.shape[1]), dtype=disease.dtype)
        normalizer = torch.zeros(node_count, dtype=disease.dtype)
        adjacency = torch.zeros((node_count, disease.shape[0]), dtype=torch.float32)
        relation = graph.inter_edges.get((node_type, 'disease'))
        if relation is not None:
            (edge_index, edge_weight) = relation
            if edge_index.numel():
                source = edge_index[0].long().cpu()
                target = edge_index[1].long().cpu()
                weight = edge_weight.detach().cpu().float().clamp_min(0.0)
                context.index_add_(0, source, disease[target] * weight.unsqueeze(1))
                normalizer.index_add_(0, source, weight)
                adjacency[source, target] = torch.maximum(adjacency[source, target], weight)
        context /= normalizer.clamp_min(1.0).unsqueeze(1)
        mediator[node_type] = context
        membership[node_type] = adjacency
    rna_adjacency = membership['rna']
    drug_adjacency = membership['drug']
    shared = rna_adjacency @ drug_adjacency.transpose(0, 1)
    rna_degree = rna_adjacency.sum(dim=1, keepdim=True)
    drug_degree = drug_adjacency.sum(dim=1).unsqueeze(0)
    union = rna_degree + drug_degree - shared
    minimum_degree = torch.minimum(rna_degree, drug_degree)
    degree_product = (rna_degree * drug_degree).clamp_min(1.0)
    rna_context = F.normalize(mediator['rna'], dim=1)
    drug_context = F.normalize(mediator['drug'], dim=1)
    context_similarity = (rna_context @ drug_context.transpose(0, 1) + 1.0) * 0.5
    maximum_degree = max(float(rna_degree.max()), float(drug_degree.max()), 1.0)
    bridge = torch.stack(
        (
            torch.log1p(shared) / np.log1p(maximum_degree),
            shared / union.clamp_min(1.0),
            shared / minimum_degree.clamp_min(1.0),
            shared / degree_product.sqrt(),
            shared / rna_degree.clamp_min(1.0),
            shared / drug_degree.clamp_min(1.0),
            context_similarity,
            torch.log1p(minimum_degree.expand_as(shared)) / np.log1p(maximum_degree)
        ),
        dim=-1
    ).contiguous()
    se_path = _structural_entropy_path_features(
        graph,
        rna_adjacency,
        drug_adjacency,
        max_module_changes=max_module_changes,
        module_penalty_floor=module_penalty_floor,
        disease_penalty_floor=disease_penalty_floor,
        degree_penalty_floor=degree_penalty_floor
    )
    se_mode = 'soft_structural_entropy_features'
    return {
        'llm': embeddings['llm'],
        'mediator': mediator,
        'bridge': {'pair': bridge},
        'se_path': {
            'pair': se_path,
            'feature_names': SE_PATH_FEATURE_NAMES,
            'mode': se_mode,
            'structural_entropy_guidance': bool(True)
        }
    }

class LLMBranch(nn.Module):
    architecture_name = 'semantic'

    def __init__(self, llm_dim: int, node_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.semantic_projection = LLMNodeProjection(llm_dim, node_dim, dropout)
        self.type_adapters = nn.ModuleDict({node_type: nn.Sequential(nn.LayerNorm(node_dim), nn.Linear(node_dim, node_dim), nn.GELU(), nn.Dropout(dropout)) for node_type in ('rna', 'drug')})
        gate_width = max(8, node_dim // 4)
        self.mediator_gates = nn.ModuleDict({node_type: nn.Sequential(nn.Linear(node_dim * 2, gate_width), nn.GELU(), nn.Linear(gate_width, 1), nn.Sigmoid()) for node_type in ('rna', 'drug')})
        self.node_norms = nn.ModuleDict({node_type: nn.LayerNorm(node_dim) for node_type in ('rna', 'drug')})
        pair_dim = node_dim * 4
        half = max(1, hidden_dim // 2)
        self.predictor = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half, 1)
        )

    def pair_features(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes: dict[str, torch.Tensor] = {}
        for node_type in ('rna', 'drug'):
            base = self.semantic_projection(modalities[f'{node_type}_llm'])
            context_input = modalities[f'{node_type}_mediator']
            context = self.semantic_projection(context_input)
            present = context_input.abs().sum(dim=-1, keepdim=True).gt(0).to(context.dtype)
            gate = self.mediator_gates[node_type](torch.cat((base, context), dim=-1))
            nodes[node_type] = self.node_norms[node_type](base + self.type_adapters[node_type](base) + present * gate * context)
        (rna, drug) = (nodes['rna'], nodes['drug'])
        return torch.cat((rna, drug, (rna - drug).abs(), rna * drug), dim=-1)

    def forward(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.predictor(self.pair_features(modalities)).squeeze(-1)

class EvidenceAwareLLMBranch(LLMBranch):
    architecture_name = 'evidence_aware'

    def __init__(
        self,
        llm_dim: int,
        node_dim: int,
        hidden_dim: int,
        dropout: float,
        bridge_dim: int,
        se_path_dim: int,
        se_path_residual_weight: float
    ):
        super().__init__(llm_dim, node_dim, hidden_dim, dropout)
        self.bridge_encoder = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim, bridge_dim),
            nn.GELU()
        )
        self.se_path_encoder = nn.Sequential(
            nn.LayerNorm(len(SE_PATH_FEATURE_NAMES)),
            nn.Linear(len(SE_PATH_FEATURE_NAMES), se_path_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(se_path_dim, se_path_dim),
            nn.GELU()
        )
        gate_hidden = max(8, se_path_dim // 2)
        self.se_gate = nn.Sequential(
            nn.Linear(8 + len(SE_PATH_FEATURE_NAMES), gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid()
        )
        self.se_residual_head = nn.Sequential(
            nn.LayerNorm(len(SE_PATH_FEATURE_NAMES)),
            nn.Linear(len(SE_PATH_FEATURE_NAMES), gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1, bias=False)
        )
        self.se_path_residual_weight = float(se_path_residual_weight)
        pair_dim = node_dim * 4 + bridge_dim + se_path_dim
        half = max(1, hidden_dim // 2)
        self.predictor = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half, 1)
        )

    def se_gate_values(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.se_gate(torch.cat((modalities['disease_bridge'], modalities['se_path_stats']), dim=-1))

    def pair_features(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        semantic = LLMBranch.pair_features(self, modalities)
        disease_bridge = modalities['disease_bridge']
        se_statistics = modalities['se_path_stats']
        bridge = self.bridge_encoder(disease_bridge)
        gate = self.se_gate_values(modalities)
        encoded_se_path = self.se_path_encoder(se_statistics)
        se_path = gate * encoded_se_path
        return torch.cat((semantic, bridge, se_path), dim=-1)

    def forward(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.predictor(self.pair_features(modalities)).squeeze(-1)
        if self.se_path_residual_weight > 0.0:
            residual = self.se_residual_head(modalities['se_path_stats']).squeeze(-1)
            logits = logits + self.se_path_residual_weight * torch.tanh(residual)
        return logits

def _modalities(embeddings: dict[str, dict[str, torch.Tensor]], pairs: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    (rna, drug) = (pairs[:, 0].long(), pairs[:, 1].long())
    return {
        'rna_llm': embeddings['llm']['rna'][rna].to(device),
        'drug_llm': embeddings['llm']['drug'][drug].to(device),
        'rna_mediator': embeddings['mediator']['rna'][rna].to(device),
        'drug_mediator': embeddings['mediator']['drug'][drug].to(device),
        'disease_bridge': embeddings['bridge']['pair'][rna, drug].to(device),
        'se_path_stats': embeddings['se_path']['pair'][rna, drug].to(device)
    }

@torch.no_grad()
def predict_llm(
    model: nn.Module,
    embeddings: dict[str, dict[str, torch.Tensor]],
    pairs: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    model.eval()
    outputs = [model(_modalities(embeddings, pairs[index], device)).cpu() for index in batch_indices(len(pairs), batch_size, shuffle=False)]
    return torch.cat(outputs) if outputs else torch.empty(0)

def train_llm_branch(args: Any, embeddings, split, device: torch.device, metric_fn) -> EvidenceAwareLLMBranch:
    model = EvidenceAwareLLMBranch(
        int(embeddings['llm']['rna'].shape[1]),
        args.llm_node_dim,
        args.llm_hidden_dim,
        args.llm_dropout,
        args.llm_bridge_dim,
        args.llm_se_path_dim,
        args.llm_se_path_residual_weight
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.llm_lr, weight_decay=args.llm_weight_decay)
    (train_pairs, train_labels) = (split['train_pairs'].long(), split['train_labels'].float())
    (val_pairs, val_labels) = (split['val_pairs'].long(), split['val_labels'].float())
    sample_weight = torch.ones_like(train_labels)
    sample_scores = split.get('train_sample_scores')
    if isinstance(sample_scores, torch.Tensor) and sample_scores.shape == train_labels.shape:
        scores = sample_scores.float()
        scored_negative = (train_labels == 0) & (scores > 0)
        if scored_negative.any():
            negative_scores = scores[scored_negative]
            (score_min, score_max) = (negative_scores.min(), negative_scores.max())
            hardness = (score_max - scores) / (score_max - score_min).clamp_min(1e-08)
            sample_weight[scored_negative] += float(args.llm_hard_negative_weight) * hardness[scored_negative]
    (best_score, best_state, best_epoch, stale) = (-np.inf, None, 0, 0)
    for epoch in range(1, args.llm_epochs + 1):
        model.train()
        for index in batch_indices(len(train_pairs), args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(_modalities(embeddings, train_pairs[index], device))
            labels = train_labels[index].to(device)
            weights = sample_weight[index].to(device)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
            positive_logits = logits[labels > 0.5]
            negative_logits = logits[labels <= 0.5]
            if positive_logits.numel() and negative_logits.numel():
                ranking = F.softplus(negative_logits[:, None] - positive_logits[None, :] + float(args.llm_ranking_margin)).mean()
                loss = loss + float(args.llm_ranking_weight) * ranking
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite LLMBranch loss at epoch {epoch}: {loss.detach().cpu().item()}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        score = metric_fn(predict_llm(model, embeddings, val_pairs, args.batch_size, device), val_labels)['AUC']
        if score > best_score + 1e-08:
            (best_score, best_state, best_epoch, stale) = (score, copy.deepcopy(model.state_dict()), epoch, 0)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'  LLM epoch {epoch:03d}: val AUC={score:.4f}')
        if stale >= args.llm_patience:
            break
    if best_state is None:
        raise RuntimeError('LLM branch did not produce a valid state')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_gates = torch.cat([model.se_gate_values(_modalities(embeddings, val_pairs[index], device)).cpu() for index in batch_indices(len(val_pairs), args.batch_size, shuffle=False)]).flatten()
    model.training_summary = {
        'best_epoch': int(best_epoch),
        'best_validation_auc': float(best_score),
        'architecture_name': str(model.architecture_name),
        'validation_se_gate_mean': float(validation_gates.mean()),
        'validation_se_gate_std': float(validation_gates.std(unbiased=False)),
        'validation_se_gate_min': float(validation_gates.min()),
        'validation_se_gate_max': float(validation_gates.max()),
        'se_path_contribution_enabled': bool(True)
    }
    return model

def train_semantic_anchor(args: Any, embeddings, split, device: torch.device, metric_fn) -> LLMBranch:
    model = LLMBranch(int(embeddings['llm']['rna'].shape[1]), args.llm_node_dim, args.llm_hidden_dim, args.llm_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.llm_lr, weight_decay=args.llm_weight_decay)
    train_pairs = split['train_pairs'].long()
    train_labels = split['train_labels'].float()
    val_pairs = split['val_pairs'].long()
    val_labels = split['val_labels'].float()
    (best_score, best_state, stale) = (-np.inf, None, 0)
    for epoch in range(1, args.llm_epochs + 1):
        model.train()
        for index in batch_indices(len(train_pairs), args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(_modalities(embeddings, train_pairs[index], device))
            labels = train_labels[index].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            positive = logits[labels > 0.5]
            negative = logits[labels <= 0.5]
            if positive.numel() and negative.numel():
                loss = loss + float(args.llm_ranking_weight) * F.softplus(negative[:, None] - positive[None, :] + float(args.llm_ranking_margin)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        score = metric_fn(predict_llm(model, embeddings, val_pairs, args.batch_size, device), val_labels)['AUC']
        if score > best_score + 1e-08:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'  semantic anchor epoch {epoch:03d}: val AUC={score:.4f}')
        if stale >= args.llm_patience:
            break
    if best_state is None:
        raise RuntimeError('semantic anchor LLM anchor did not produce a valid state')
    model.load_state_dict(best_state)
    model.training_summary = {
        'best_epoch': int(epoch - stale),
        'best_validation_auc': float(best_score),
        'architecture_name': 'semantic'
    }
    return model

@torch.no_grad()
def llm_pair_features(
    model: nn.Module,
    embeddings: dict[str, dict[str, torch.Tensor]],
    pairs: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    model.eval()
    outputs = []
    for index in batch_indices(len(pairs), batch_size, shuffle=False):
        modalities = _modalities(embeddings, pairs[index], device)
        learned = model.pair_features(modalities)
        outputs.append(learned.cpu())
    return torch.cat(outputs) if outputs else torch.empty((0, 0))

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def build_lm_corpus(args, fold) -> None:
    NODE_TYPES = ('rna', 'drug', 'disease')
    NODE_PREFIX = {'rna': 'RNA', 'drug': 'DRUG', 'disease': 'DISEASE'}
    REL_PREFIX = 'REL'

    def torch_load_cpu(path: str) -> Any:
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(path, map_location='cpu')

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)

    def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        n = 0
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json_dumps(row) + '\n')
                n += 1
        return n

    def load_graph_state(final_graph_path: str) -> Dict[str, Any]:
        payload = torch_load_cpu(final_graph_path)
        if isinstance(payload, dict) and 'state' in payload:
            return payload['state']
        if isinstance(payload, dict):
            return payload
        raise TypeError(f'无法解析图文件: {final_graph_path}')

    def infer_node_count(state: Dict[str, Any], node_type: str) -> int:
        names = state.get('names', {})
        if node_type in names and names[node_type] is not None:
            return len(names[node_type])
        feats = state.get('node_features', {})
        if node_type in feats and feats[node_type] is not None:
            return int(feats[node_type].shape[0])
        raw_sim = state.get('raw_similarity', {})
        if node_type in raw_sim and raw_sim[node_type] is not None:
            return int(raw_sim[node_type].shape[0])
        return 0

    def get_node_names(state: Dict[str, Any]) -> Dict[str, List[str]]:
        names_in = state.get('names', {}) or {}
        out: Dict[str, List[str]] = {}
        for ntype in NODE_TYPES:
            n = infer_node_count(state, ntype)
            values = list(names_in.get(ntype, [])) if ntype in names_in else []
            if len(values) < n:
                values += [f'{ntype}_{i}' for i in range(len(values), n)]
            out[ntype] = [str(x) for x in values[:n]]
        return out

    def token_for(ntype: str, idx: int) -> str:
        prefix = NODE_PREFIX.get(ntype.lower(), ntype.upper())
        return f'[{prefix}_{int(idx)}]'

    def rel_token(rel: str) -> str:
        rel = str(rel).strip()
        rel = re.sub('\\s+', '_', rel)
        rel = re.sub('[^0-9A-Za-z_\\-:.]', '_', rel)
        return f'[{REL_PREFIX}_{rel}]'

    def normalize_node(node: Any) -> Optional[Tuple[str, int]]:
        if isinstance(node, (list, tuple)) and len(node) >= 2:
            return (str(node[0]).lower(), int(node[1]))
        if isinstance(node, dict):
            ntype = node.get('type', node.get('ntype', node.get('node_type')))
            idx = node.get('idx', node.get('index', node.get('id')))
            if ntype is not None and idx is not None:
                return (str(ntype).lower(), int(idx))
        return None

    def make_path_tokens(nodes: List[Any], rels: Optional[List[str]]=None) -> str:
        typed = [normalize_node(n) for n in nodes]
        typed = [x for x in typed if x is not None]
        pieces: List[str] = []
        for (i, (ntype, idx)) in enumerate(typed):
            pieces.append(token_for(ntype, idx))
            if rels is not None and i < len(rels):
                pieces.append(rel_token(str(rels[i])))
        return ' '.join(pieces)

    def make_lm_text(path_nodes: List[Any], rels: Optional[List[str]]) -> str:
        path = make_path_tokens(path_nodes, rels)
        if not path:
            raise ValueError('GraphRAG produced an empty meta-path')
        return 'PATH: ' + path

    def is_safe_val_row(label: Optional[int], source: str) -> bool:
        if source not in {'val', 'val_pos'}:
            return False
        return label != 0

    def build_one_fold(args: argparse.Namespace, fold: int) -> Dict[str, Any]:
        fold_name = f'fold_{fold}'
        fold_dir = args.fold_dir
        rag_dir = args.rag_dir
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        final_graph_path = os.path.join(fold_dir, 'final_train_graph.pt')
        graph_paths_path = os.path.join(rag_dir, 'graph_rag_paths.jsonl')
        sampled_paths_path = os.path.join(rag_dir, 'sampled_graph_rag_sentences.jsonl')
        if not os.path.exists(final_graph_path):
            raise FileNotFoundError(f'缺少 {final_graph_path}')
        if not os.path.exists(graph_paths_path) or not os.path.exists(sampled_paths_path):
            raise FileNotFoundError(f'在 {rag_dir} 下找不到 graph_rag_paths.jsonl 或 sampled_graph_rag_sentences.jsonl；请先运行结构熵 GraphRAG 阶段。')
        state = load_graph_state(final_graph_path)
        names = get_node_names(state)
        relation_tokens = set()
        train_rows: List[Dict[str, Any]] = []
        val_rows: List[Dict[str, Any]] = []
        supervised_rows: List[Dict[str, Any]] = []
        extract_rows: List[Dict[str, Any]] = []
        for sentence_record in read_jsonl(sampled_paths_path):
            source = str(sentence_record.get('source', ''))
            if source != 'graph_train':
                raise ValueError(f'采样 GraphRAG 路径 source 必须为 graph_train，实际为 {source!r}')
            pair_source = str(sentence_record.get('pair_source', ''))
            if pair_source != 'train_pos':
                raise ValueError(f'LoRA 训练路径 pair_source 必须为 train_pos，实际为 {pair_source!r}')
            query_rna_idx = int(sentence_record['rna_idx'])
            query_drug_idx = int(sentence_record['drug_idx'])
            nodes = sentence_record.get('nodes', [])
            rels = [str(value) for value in sentence_record.get('rels', [])]
            text = make_lm_text(path_nodes=nodes, rels=rels)
            for relation in rels:
                relation_tokens.add(rel_token(relation))
            row = {
                'text': text,
                'rna_idx': query_rna_idx,
                'drug_idx': query_drug_idx,
                'label': None,
                'source': 'train_pos',
                'path_nodes': [normalize_node(node) for node in nodes],
                'path_relations': rels,
                'path_score': float(sentence_record.get('score', 1.0)),
                'direction': 'forward'
            }
            train_rows.append(row)
            reverse_nodes = list(reversed(nodes))
            reverse_rels = list(reversed(rels))
            train_rows.append({
                'text': make_lm_text(path_nodes=reverse_nodes, rels=reverse_rels),
                'rna_idx': query_rna_idx,
                'drug_idx': query_drug_idx,
                'label': None,
                'source': 'train_pos',
                'path_nodes': [normalize_node(node) for node in reverse_nodes],
                'path_relations': reverse_rels,
                'path_score': float(sentence_record.get('score', 1.0)),
                'direction': 'reverse'
            })
        for pair_record in read_jsonl(graph_paths_path):
            rna_idx = int(pair_record.get('rna_idx'))
            drug_idx = int(pair_record.get('drug_idx'))
            label = pair_record.get('label', None)
            source = str(pair_record.get('source', ''))
            pair_supervised = {
                'pair_id': pair_record.get('pair_id'),
                'rna_idx': rna_idx,
                'drug_idx': drug_idx,
                'label': label,
                'source': source,
                'path_texts': [],
                'path_scores': [],
                'meta_paths': []
            }
            for p in pair_record.get('paths', []):
                nodes = p.get('nodes', [])
                rels = p.get('rels', [])
                for r in rels:
                    relation_tokens.add(rel_token(r))
                text = make_lm_text(path_nodes=nodes, rels=rels)
                pair_supervised['path_texts'].append(text)
                pair_supervised['path_scores'].append(float(p.get('score', 0.0)))
                pair_supervised['meta_paths'].append(' -> '.join(map(str, p.get('meta_path', []))))
                if source in {'train', 'train_pos'} and label != 0:
                    path_score = max(float(p.get('score', 0.0)), 1e-12)
                    typed_nodes = [normalize_node(node) for node in nodes]
                    extract_rows.append({
                        'text': text,
                        'label': None,
                        'source': 'graph_train',
                        'path_nodes': typed_nodes,
                        'path_relations': [str(relation) for relation in rels],
                        'path_score': path_score,
                        'direction': 'forward'
                    })
                    reverse_nodes = list(reversed(nodes))
                    reverse_rels = list(reversed(rels))
                    extract_rows.append({
                        'text': make_lm_text(path_nodes=reverse_nodes, rels=reverse_rels),
                        'label': None,
                        'source': 'graph_train',
                        'path_nodes': [normalize_node(node) for node in reverse_nodes],
                        'path_relations': [str(relation) for relation in reverse_rels],
                        'path_score': path_score,
                        'direction': 'reverse'
                    })
                if is_safe_val_row(label, source):
                    val_rows.append({'text': text, 'rna_idx': rna_idx, 'drug_idx': drug_idx, 'label': label, 'source': source})
            supervised_rows.append(pair_supervised)
        node_tokens = []
        for ntype in NODE_TYPES:
            for (idx, name) in enumerate(names[ntype]):
                node_tokens.append({
                    'type': ntype,
                    'index': int(idx),
                    'name': str(name),
                    'token': token_for(ntype, idx),
                    'has_text': False
                })
        default_rels = ['sim', 'rna_drug', 'rna_disease', 'drug_disease', '1']
        for r in default_rels:
            relation_tokens.add(rel_token(r))
        node_vocab = {
            'fold': int(fold),
            'fold_dir': os.path.abspath(fold_dir),
            'rag_dir': os.path.abspath(rag_dir),
            'node_tokens': node_tokens,
            'relation_tokens': sorted(relation_tokens),
            'all_special_tokens': [x['token'] for x in node_tokens] + sorted(relation_tokens),
            'num_nodes': {ntype: len(names[ntype]) for ntype in NODE_TYPES},
            'raw_description_text_used': False
        }
        train_path = os.path.join(out_dir, 'train_corpus.jsonl')
        val_path = os.path.join(out_dir, 'val_corpus.jsonl')
        supervised_out_path = os.path.join(out_dir, 'supervised_pair_corpus.jsonl')
        extract_path = os.path.join(out_dir, 'extract_prompts.jsonl')
        vocab_path = os.path.join(out_dir, 'node_vocab.json')
        summary_path = os.path.join(out_dir, 'corpus_summary.json')
        n_train = write_jsonl(train_path, train_rows)
        n_val = write_jsonl(val_path, val_rows)
        n_sup = write_jsonl(supervised_out_path, supervised_rows)
        n_extract = write_jsonl(extract_path, extract_rows)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(node_vocab, f, ensure_ascii=False, indent=2)
        summary = {
            'fold': int(fold),
            'num_train_corpus_rows': int(n_train),
            'num_val_corpus_rows': int(n_val),
            'num_supervised_pair_rows': int(n_sup),
            'num_extract_prompts': int(n_extract),
            'include_raw_text_in_corpus': False,
            'corpus_format': 'se_graphrag_bidirectional_causal',
            'embedding_prompt_source': 'all_train_pos_graphrag_paths_bidirectional',
            'use_relation_tokens': bool(True),
            'lm_corpus_safety': {
                'labels_removed_from_text': True,
                'train_sources': ['train_pos'],
                'val_sources': ['val', 'val_pos'],
                'heldout_rows_allowed_in_train': bool(False),
                'heldout_extract_prompts': bool(False)
            },
            'num_nodes': node_vocab['num_nodes'],
            'num_text_features': {k: 0 for k in NODE_TYPES},
            'output_files': {
                'train_corpus': os.path.abspath(train_path),
                'val_corpus': os.path.abspath(val_path),
                'supervised_pair_corpus': os.path.abspath(supervised_out_path),
                'extract_prompts': os.path.abspath(extract_path),
                'node_vocab': os.path.abspath(vocab_path)
            }
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'[fold_{fold}] train={n_train}, val={n_val}, supervised={n_sup}, extract={n_extract} -> {out_dir}')
        return summary
    return build_one_fold(args, fold)

def make_experiment_progress():
    from transformers import TrainerCallback

    def write(path, obj):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temp.replace(path)

    def digest_parameters(model):
        h = hashlib.sha256()
        count = 0
        names = []
        with torch.no_grad():
            for (name, p) in model.named_parameters():
                if p.requires_grad:
                    h.update(name.encode())
                    h.update(str(tuple(p.shape)).encode())
                    h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                    count += p.numel()
                    names.append(name)
        return dict(sha256=h.hexdigest(), trainable_parameters=count, names=names)

    class ExperimentProgress(TrainerCallback):

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.started = time.time()
            self.output = Path(args.output_dir)
            fold = int(self.output.parent.name.split('_')[-1])
            initial = digest_parameters(model)
            initial.update(seed=int(args.seed), fold=fold, step_at_begin=int(state.global_step), started_at=self.started)
            if not state.global_step:
                fold_initial = self.output.parent / 'initial_trainable_fingerprint.json'
                if fold_initial.exists():
                    expected = json.loads(fold_initial.read_text(encoding='utf-8'))
                    assert expected['sha256'] == initial['sha256'], 'Matched fresh initialization failed'
                else:
                    write(fold_initial, initial)
                write(self.output / 'fresh_initialization.json', initial)
            else:
                write(self.output / 'resume_initialization.json', initial)
            self.witness_name = next((name for (name, p) in model.named_parameters() if 'lora_B' in name and p.requires_grad))
            self.witness_initial = dict(model.named_parameters())[self.witness_name].detach().cpu().clone()
            self.on_log(args, state, control, logs={'event': 'train_begin'})

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            for (key, value) in logs.items():
                if isinstance(value, (int, float)) and (not math.isfinite(value)):
                    raise FloatingPointError(f'Nonfinite training metric {key}={value}')
            row = dict(
                global_step=int(state.global_step),
                max_steps=int(state.max_steps),
                epoch=state.epoch,
                updated_at=time.time(),
                elapsed_this_session=time.time() - getattr(self, 'started', time.time()),
                metrics=logs,
                device=str(args.device),
                gpu_allocated_mib=torch.cuda.memory_allocated() / 1048576
            )
            output = Path(args.output_dir)
            write(output / 'progress.json', row)
            with open(output / 'progress.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if int(state.global_step) <= 10 and (not (Path(args.output_dir) / 'optimizer_witness.json').exists()):
                current = dict(model.named_parameters())[self.witness_name].detach().cpu()
                difference = float((current - self.witness_initial).abs().max())
                if difference > 0:
                    write(
                        Path(args.output_dir) / 'optimizer_witness.json',
                        dict(
                            parameter=self.witness_name,
                            max_abs_change=difference,
                            step=int(state.global_step),
                            passed=True,
                            zero_lr_warmup_step_allowed=True
                        )
                    )
                elif int(state.global_step) == 10:
                    raise AssertionError('LoRA parameter did not update within first ten optimizer steps')

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            needed = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors')
            assert all(((checkpoint / name).is_file() for name in needed)), str(checkpoint)
            write(
                checkpoint / 'resume_ready.json',
                dict(
                    step=int(state.global_step),
                    saved_at=time.time(),
                    files={name: (checkpoint / name).stat().st_size for name in needed}
                )
            )

        def on_train_end(self, args, state, control, model=None, **kwargs):
            final = digest_parameters(model)
            initial = Path(args.output_dir) / 'fresh_initialization.json'
            assert initial.exists()
            assert final['sha256'] != json.loads(initial.read_text())['sha256'], 'Trainable parameters unchanged'
            final.update(
                global_step=int(state.global_step),
                epoch=state.epoch,
                ended_at=time.time(),
                best_metric=state.best_metric,
                best_model_checkpoint=state.best_model_checkpoint
            )
            write(Path(args.output_dir) / 'trained_parameters.json', final)
    return ExperimentProgress()

def train_lora(args) -> None:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    def load_special_tokens(node_vocab_json: str) -> List[str]:
        with open(node_vocab_json, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        if 'all_special_tokens' in vocab:
            tokens = list(vocab['all_special_tokens'])
        else:
            tokens = [x['token'] for x in vocab.get('node_tokens', [])]
            tokens += list(vocab.get('relation_tokens', []))
        seen = set()
        out = []
        for t in tokens:
            t = str(t)
            if t and t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def validate_training_corpus(path: str) -> Dict[str, int]:
        stats = {'rows': 0, 'missing_source': 0, 'heldout_source': 0, 'label_text': 0}
        with open(path, 'r', encoding='utf-8') as f:
            for (line_no, line) in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                stats['rows'] += 1
                source = str(row.get('source', '')).strip()
                text = str(row.get('text', ''))
                if not source:
                    stats['missing_source'] += 1
                elif source not in {'train', 'train_pos'}:
                    stats['heldout_source'] += 1
                if re.search('\\blabel\\s*:', text, flags=re.IGNORECASE):
                    stats['label_text'] += 1
                if not source:
                    raise ValueError(f'训练语料第 {line_no} 行缺少 source。请用修改后的 corpus 阶段重新生成。')
                if source not in {'train', 'train_pos'}:
                    raise ValueError(f'训练语料第 {line_no} 行来自 {source}，拒绝把验证/测试样本用于 LoRA 训练。')
                if re.search('\\blabel\\s*:', text, flags=re.IGNORECASE):
                    raise ValueError(f'训练语料第 {line_no} 行文本包含 Label:，存在直接标签泄露。')
        if stats['rows'] == 0:
            raise ValueError(f'训练语料为空: {path}')
        return stats

    def validate_eval_corpus(path: str) -> Dict[str, int]:
        stats = {'rows': 0, 'missing_source': 0, 'wrong_source': 0, 'negative_rows': 0, 'label_text': 0}
        allowed_sources = {'val', 'val_pos'}
        with open(path, 'r', encoding='utf-8') as f:
            for (line_no, line) in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                stats['rows'] += 1
                source = str(row.get('source', '')).strip()
                text = str(row.get('text', ''))
                label = row.get('label', None)
                if not source:
                    stats['missing_source'] += 1
                elif source not in allowed_sources:
                    stats['wrong_source'] += 1
                if label == 0:
                    stats['negative_rows'] += 1
                if re.search('\\blabel\\s*:', text, flags=re.IGNORECASE):
                    stats['label_text'] += 1
                if not source:
                    raise ValueError(f'LLM 验证语料第 {line_no} 行缺少 source。请重新运行 corpus 阶段。')
                if source not in allowed_sources:
                    raise ValueError(f'LLM 验证语料第 {line_no} 行来自 {source}；只允许 val/val_pos。')
                if label == 0:
                    raise ValueError(f'LLM 验证语料第 {line_no} 行是负样本。当前 causal-LM 验证只评估真实图语言正样本。')
                if re.search('\\blabel\\s*:', text, flags=re.IGNORECASE):
                    raise ValueError(f'LLM 验证语料第 {line_no} 行文本包含 Label:，存在直接标签泄露。')
        if stats['rows'] == 0:
            raise ValueError(f'LLM 验证语料为空: {path}。请确认 GraphRAG 使用 --pair_source llm_train_val_pos，并重新生成 corpus。')
        return stats

    def load_corpus_pair_keys(path: str) -> Set[Tuple[int, int]]:
        keys: Set[Tuple[int, int]] = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if 'rna_idx' in row and 'drug_idx' in row:
                    keys.add((int(row['rna_idx']), int(row['drug_idx'])))
        return keys

    def canonical_pair_hash(keys: Set[Tuple[int, int]]) -> str:
        encoded = '\n'.join((f'{rna},{drug}' for (rna, drug) in sorted(keys)))
        return hashlib.sha256(encoded.encode('utf-8')).hexdigest()

    def split_pair_keys(split: Dict[str, Any], key: str) -> Set[Tuple[int, int]]:
        values = split.get(key)
        if not torch.is_tensor(values) or values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f'split_data.pt 缺少有效的 {key}')
        return {(int(row[0]), int(row[1])) for row in values.tolist()}

    def smart_resize_tokenizer_and_embeddings(tokenizer, model, special_tokens: List[str]) -> int:
        added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        return int(added)

    def sync_model_token_ids(tokenizer, model) -> None:
        token_ids = {
            'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
        for config in (model.config, getattr(model, 'generation_config', None)):
            if config is None:
                continue
            for (name, value) in token_ids.items():
                if value is not None:
                    setattr(config, name, int(value))

    class PathOnlyCausalCollator:

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            labels = [list(feature['labels']) for feature in features]
            model_features = [{key: value for (key, value) in feature.items() if key != 'labels'} for feature in features]
            batch = self.tokenizer.pad(model_features, padding=True, return_tensors='pt')
            max_length = int(batch['input_ids'].shape[1])
            padded_labels = []
            for label in labels:
                padding = [-100] * (max_length - len(label))
                if self.tokenizer.padding_side == 'left':
                    padded_labels.append(padding + label)
                else:
                    padded_labels.append(label + padding)
            batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
            return batch

    def tokenize_dataset(dataset, tokenizer, max_length: int, supervised_token_ids: Optional[Set[int]]=None):
        supervised_token_ids = set(supervised_token_ids or ())
        stats = {'examples': 0, 'valid_path_tokens': 0, 'min_valid_path_tokens': None, 'max_valid_path_tokens': 0}

        def last_subsequence(sequence: List[int], target: List[int]) -> int:
            for start in range(len(sequence) - len(target), -1, -1):
                if sequence[start:start + len(target)] == target:
                    return start
            return -1
        token_builder = getattr(tokenizer, 'build_inputs_with_special_tokens', None)
        if callable(token_builder):
            special_count = len(token_builder([]))

            def add_special_tokens(token_ids: List[int]) -> List[int]:
                return list(token_builder(token_ids))
        else:
            probe_plain = tokenizer('PATH:', add_special_tokens=False, padding=False)['input_ids']
            probe_formatted = tokenizer('PATH:', add_special_tokens=True, padding=False)['input_ids']
            probe_start = last_subsequence(probe_formatted, probe_plain)
            if not probe_plain or probe_start < 0:
                raise RuntimeError('Tokenizer cannot expose a usable single-sequence special-token template')
            special_prefix = probe_formatted[:probe_start]
            special_suffix = probe_formatted[probe_start + len(probe_plain):]
            special_count = len(special_prefix) + len(special_suffix)

            def add_special_tokens(token_ids: List[int]) -> List[int]:
                return special_prefix + list(token_ids) + special_suffix

        def tokenize_fn(batch):
            output = {'input_ids': [], 'attention_mask': [], 'labels': []}
            for text in batch['text']:
                path_start = text.find('PATH:')
                if path_start < 0:
                    raise ValueError('Every causal-LM example must contain a PATH: line')
                context_ids = tokenizer(text[:path_start], add_special_tokens=False, padding=False)['input_ids']
                path_ids = tokenizer(text[path_start:], add_special_tokens=False, padding=False)['input_ids']
                context_budget = max_length - special_count - len(path_ids)
                if not path_ids or context_budget < 0:
                    raise ValueError(f'LoRA PATH supervision does not fit within max_length; PATH tokens={len(path_ids)}, max_length={max_length}. Increase --lora-max-length.')
                combined = add_special_tokens(context_ids[:context_budget] + path_ids)
                supervised_start = last_subsequence(combined, path_ids)
                if supervised_start < 0:
                    raise ValueError('Tokenizer could not preserve the PATH token sequence')
                labels = [-100] * len(combined)
                path_end = supervised_start + len(path_ids)
                for position in range(supervised_start, path_end):
                    token_id = int(combined[position])
                    if not supervised_token_ids or token_id in supervised_token_ids:
                        labels[position] = token_id
                valid = sum((label != -100 for label in labels))
                if valid == 0:
                    raise ValueError('LoRA sample has zero supervised PATH tokens; increase max_length or reduce context length.')
                output['input_ids'].append(combined)
                output['attention_mask'].append([1] * len(combined))
                output['labels'].append(labels)
                stats['examples'] += 1
                stats['valid_path_tokens'] += valid
                current_min = stats['min_valid_path_tokens']
                stats['min_valid_path_tokens'] = valid if current_min is None else min(current_min, valid)
                stats['max_valid_path_tokens'] = max(stats['max_valid_path_tokens'], valid)
            return output
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc='Tokenizing corpus',
            load_from_cache_file=False
        )
        if stats['examples'] == 0 or stats['valid_path_tokens'] == 0:
            raise ValueError('LoRA corpus contains no supervised PATH tokens')
        stats['mean_valid_path_tokens'] = stats['valid_path_tokens'] / stats['examples']
        print(
            f"[PATH supervision] examples={stats['examples']}, total_valid_tokens={stats['valid_path_tokens']}, min/mean/max={stats['min_valid_path_tokens']}/{stats['mean_valid_path_tokens']:.2f}/{stats['max_valid_path_tokens']}",
            flush=True
        )
        return (tokenized, stats)

    def parse_target_modules(value: str) -> List[str]:
        if isinstance(value, list):
            return value
        return [x.strip() for x in value.split(',') if x.strip()]
    configure_torch_determinism(True)
    adapter_root = Path(args.output_dir)
    if adapter_root.exists() and any(adapter_root.iterdir()) and (args.resume_from_checkpoint is None):
        raise RuntimeError('Nonempty adapter requires a verified resumable checkpoint; existing files are preserved')
    os.makedirs(args.output_dir, exist_ok=True)
    corpus_safety = validate_training_corpus(args.train_jsonl)
    if not args.eval_jsonl:
        raise ValueError('Validation corpus is required for LM checkpoint selection')
    split_payload = torch_load_cpu(Path(args.split_data))
    if not isinstance(split_payload, dict):
        raise TypeError(f'split_data.pt 必须是字典: {args.split_data}')
    train_corpus_pairs = load_corpus_pair_keys(args.train_jsonl)
    if not train_corpus_pairs:
        raise ValueError('LoRA 训练语料无法解析任何 RNA-Drug 路径端点；拒绝生成空集合上的隔离证明')
    train_positive_pairs = split_pair_keys(split_payload, 'train_positive_pairs')
    test_pairs = split_pair_keys(split_payload, 'test_pairs')
    test_positive_pairs = split_pair_keys(split_payload, 'test_positive_pairs')
    outside_train = train_corpus_pairs - train_positive_pairs
    test_overlap = train_corpus_pairs & test_pairs
    test_positive_overlap = train_corpus_pairs & test_positive_pairs
    if outside_train or test_overlap or test_positive_overlap:
        raise ValueError(f'LoRA 训练语料未与 outer-test 隔离: outside_train={len(outside_train)}, test_overlap={len(test_overlap)}, test_positive_overlap={len(test_positive_overlap)}')
    split_isolation = {
        'split_data': os.path.abspath(args.split_data),
        'split_data_sha256': _sha256_file(Path(args.split_data)),
        'train_corpus_pair_count': len(train_corpus_pairs),
        'train_corpus_pair_sha256': canonical_pair_hash(train_corpus_pairs),
        'train_positive_pair_count': len(train_positive_pairs),
        'train_positive_pair_sha256': canonical_pair_hash(train_positive_pairs),
        'test_pair_count': len(test_pairs),
        'test_pair_sha256': canonical_pair_hash(test_pairs),
        'test_positive_pair_count': len(test_positive_pairs),
        'test_positive_pair_sha256': canonical_pair_hash(test_positive_pairs),
        'train_corpus_outside_train_count': len(outside_train),
        'train_test_pair_overlap': len(test_overlap),
        'train_test_positive_pair_overlap': len(test_positive_overlap)
    }
    eval_corpus_safety = None
    train_eval_pair_overlap = 0
    if args.eval_jsonl:
        eval_corpus_safety = validate_eval_corpus(args.eval_jsonl)
        train_pairs = load_corpus_pair_keys(args.train_jsonl)
        eval_pairs = load_corpus_pair_keys(args.eval_jsonl)
        overlap = train_pairs & eval_pairs
        train_eval_pair_overlap = len(overlap)
        if overlap:
            preview = sorted(overlap)[:10]
            raise ValueError(f'LoRA train/eval pair 不互斥，检测到 {len(overlap)} 个重叠 pair，示例: {preview}')
    if not eval_pairs or not eval_pairs.issubset(split_pair_keys(split_payload, 'val_positive_pairs')):
        raise ValueError('LM validation must use only this fold validation positives')
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={'': args.device},
        dtype=torch.bfloat16,
        trust_remote_code=False
    )
    special_tokens = load_special_tokens(args.node_vocab_json)
    num_added = smart_resize_tokenizer_and_embeddings(tokenizer, model, special_tokens)
    sync_model_token_ids(tokenizer, model)
    with open(args.node_vocab_json, 'r', encoding='utf-8') as handle:
        node_vocab = json.load(handle)
    node_token_ids = {int(tokenizer.convert_tokens_to_ids(str(item['token']))) for item in node_vocab.get('node_tokens', [])}
    if not node_token_ids or tokenizer.unk_token_id in node_token_ids:
        raise ValueError('node_vocab contains no valid tokenizer node tokens')
    relation_token_ids = {int(tokenizer.convert_tokens_to_ids(str(token))) for token in node_vocab.get('relation_tokens', [])}
    if not relation_token_ids or tokenizer.unk_token_id in relation_token_ids:
        raise ValueError('node_vocab contains no valid tokenizer relation tokens')
    supervised_token_ids = node_token_ids | relation_token_ids
    model.gradient_checkpointing_enable()
    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=parse_target_modules(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        modules_to_save=parse_target_modules(args.modules_to_save)
    )
    if bool(getattr(model.config, 'tie_word_embeddings', False)) and 'ensure_weight_tying' in inspect.signature(LoraConfig).parameters:
        lora_kwargs['ensure_weight_tying'] = True
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    original_save_pretrained = model.save_pretrained

    def save_pretrained_without_auto_embedding(*save_args, **save_kwargs):
        save_kwargs.setdefault('save_embedding_layers', False)
        return original_save_pretrained(*save_args, **save_kwargs)
    model.save_pretrained = save_pretrained_without_auto_embedding
    model.print_trainable_parameters()
    raw_train = load_dataset('json', data_files=args.train_jsonl, split='train')
    (train_dataset, train_path_stats) = tokenize_dataset(raw_train, tokenizer, args.max_length, supervised_token_ids)
    eval_dataset = None
    eval_path_stats = None
    if args.eval_jsonl:
        raw_eval = load_dataset('json', data_files=args.eval_jsonl, split='train')
        (eval_dataset, eval_path_stats) = tokenize_dataset(raw_eval, tokenizer, args.max_length, supervised_token_ids)
    data_collator = PathOnlyCausalCollator(tokenizer)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    if eval_dataset is not None and args.eval_steps <= 0:
        raise ValueError('--eval_steps 必须 > 0')
    if args.save_steps <= 0:
        raise ValueError('--save_steps 必须 > 0')
    effective_save_steps = args.save_steps
    if eval_dataset is not None and effective_save_steps % args.eval_steps != 0:
        effective_save_steps = args.eval_steps
        print(f'[LLM-val] 为满足最佳模型选择要求，save_steps 自动调整为 eval_steps={args.eval_steps}。', flush=True)
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError('--warmup_ratio must be in [0, 1)')
    updates_per_epoch = max(1, math.ceil(len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps)))
    total_update_steps = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = int(math.ceil(total_update_steps * args.warmup_ratio))
    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=effective_save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy='steps',
        fp16=False,
        bf16=True,
        optim='adamw_torch',
        report_to='none',
        remove_unused_columns=False,
        seed=args.seed,
        use_cpu=args.device == 'cpu'
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = 'eval_strategy' if 'eval_strategy' in ta_params else 'evaluation_strategy'
    training_kwargs.update({
        eval_key: 'steps',
        'eval_steps': args.eval_steps,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'restore_callback_states_from_checkpoint': True
    })
    if 'group_by_length' in ta_params:
        training_kwargs['group_by_length'] = True
    load_best_model_at_end = True
    train_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = {
        'model': model,
        'args': train_args,
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
    early_stopping_enabled = eval_dataset is not None and load_best_model_at_end and (args.early_stopping_patience > 0)
    if early_stopping_enabled:
        trainer_kwargs['callbacks'] = [EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )]
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if 'tokenizer' in trainer_params:
        trainer_kwargs['tokenizer'] = tokenizer
    elif 'processing_class' in trainer_params:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        print('[兼容] 当前 transformers.Trainer 不支持 tokenizer/processing_class 参数，已自动忽略。', flush=True)
    trainer = Trainer(**trainer_kwargs)
    trainer.add_callback(make_experiment_progress())
    model.config.use_cache = False
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.state.best_metric is None or not math.isfinite(float(trainer.state.best_metric)):
        raise RuntimeError('LoRA produced no valid validation-loss-selected checkpoint')
    eval_metrics: Dict[str, float] = {}
    if eval_dataset is not None:
        eval_metrics = dict(trainer.evaluate())
        if 'eval_loss' in eval_metrics:
            try:
                eval_metrics['eval_perplexity'] = float(math.exp(float(eval_metrics['eval_loss'])))
            except OverflowError:
                eval_metrics['eval_perplexity'] = float('inf')
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    train_metrics = {str(k): float(v) if isinstance(v, (int, float)) else v for (k, v) in dict(train_result.metrics).items()}
    eval_metrics = {str(k): float(v) if isinstance(v, (int, float)) else v for (k, v) in eval_metrics.items()}
    metadata = {
        'model_name_or_path': args.model_name_or_path,
        'train_jsonl': os.path.abspath(args.train_jsonl),
        'eval_jsonl': os.path.abspath(args.eval_jsonl) if args.eval_jsonl else None,
        'node_vocab_json': os.path.abspath(args.node_vocab_json),
        'split_data': os.path.abspath(args.split_data),
        'output_dir': os.path.abspath(args.output_dir),
        'num_added_special_tokens': num_added,
        'num_special_tokens': len(special_tokens),
        'num_supervised_node_tokens': len(node_token_ids),
        'num_supervised_relation_tokens': len(relation_token_ids),
        'supervision': 'next_graph_node_and_relation_tokens',
        'max_length': args.max_length,
        'warmup_ratio': float(args.warmup_ratio),
        'warmup_steps': int(warmup_steps),
        'lora_target_modules': parse_target_modules(args.lora_target_modules),
        'modules_to_save': parse_target_modules(args.modules_to_save),
        'corpus_safety': corpus_safety,
        'eval_corpus_safety': eval_corpus_safety,
        'path_supervision': {'train': train_path_stats, 'eval': eval_path_stats},
        'train_eval_pair_overlap': int(train_eval_pair_overlap),
        'split_isolation': split_isolation,
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'best_metric': float(trainer.state.best_metric) if getattr(trainer.state, 'best_metric', None) is not None else None,
        'best_model_checkpoint': getattr(trainer.state, 'best_model_checkpoint', None),
        'load_best_model_at_end': bool(load_best_model_at_end),
        'early_stopping_enabled': bool(early_stopping_enabled),
        'early_stopping_patience': int(args.early_stopping_patience),
        'early_stopping_threshold': float(args.early_stopping_threshold),
        'eval_steps': int(args.eval_steps) if eval_dataset is not None else None,
        'save_steps': int(effective_save_steps),
    }
    with open(os.path.join(args.output_dir, 'train_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    adapter_root = Path(args.output_dir).resolve()
    for checkpoint in adapter_root.glob('checkpoint-*'):
        if checkpoint.is_dir() and checkpoint.parent == adapter_root:
            shutil.rmtree(checkpoint)
    print(f'\nLoRA 训练完成，输出目录：{args.output_dir}')

def extract_hidden(args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    NODE_TYPES = ('rna', 'drug', 'disease')

    def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load_node_vocab(path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_special_tokens(vocab: Dict[str, Any]) -> List[str]:
        tokens = list(vocab.get('all_special_tokens', []))
        if not tokens:
            tokens = [x['token'] for x in vocab.get('node_tokens', [])]
            tokens += list(vocab.get('relation_tokens', []))
        return list(dict.fromkeys((str(x) for x in tokens if str(x))))

    def add_tokens_if_needed(tokenizer, model, special_tokens: List[str]) -> None:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))

    def freeze_model_for_inference(model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trainable = [name for (name, parameter) in model.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError('Frozen LoRA still has trainable parameters: ' + ', '.join(trainable[:10]))
        model.eval()
        print(
            '[LoRA frozen] Base LLM and adapter parameters are read-only during hidden-state extraction.',
            flush=True
        )
        return model

    def token_id(tokenizer, token: str) -> int:
        idx = tokenizer.convert_tokens_to_ids(token)
        if idx is None or idx == tokenizer.unk_token_id:
            raise ValueError(f'token 不在 tokenizer 中: {token}')
        return int(idx)

    def extract_static_node_tables(model, tokenizer, node_vocab: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        weight = model.get_input_embeddings().weight.detach().float().cpu()
        dim = int(weight.shape[1])
        counts = {ntype: int(node_vocab.get('num_nodes', {}).get(ntype, 0)) for ntype in NODE_TYPES}
        for item in node_vocab.get('node_tokens', []):
            (ntype, idx) = (str(item['type']).lower(), int(item['index']))
            counts[ntype] = max(counts.get(ntype, 0), idx + 1)
        tables = {ntype: torch.zeros((counts.get(ntype, 0), dim), dtype=torch.float32) for ntype in NODE_TYPES}
        for item in node_vocab.get('node_tokens', []):
            (ntype, idx, tok) = (str(item['type']).lower(), int(item['index']), str(item['token']))
            if ntype in tables and idx < tables[ntype].shape[0]:
                tables[ntype][idx] = weight[token_id(tokenizer, tok)]
        return tables

    def contextualize_node_tables(model, tokenizer, node_vocab: Dict[str, Any], prompts_path: str, batch_size: int, max_length: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        static = extract_static_node_tables(model, tokenizer, node_vocab)
        accum = {k: torch.zeros_like(v) for (k, v) in static.items()}
        counts = {k: torch.zeros((v.shape[0],), dtype=torch.long) for (k, v) in static.items()}
        weight_sums = {k: torch.zeros((v.shape[0],), dtype=torch.float32) for (k, v) in static.items()}
        token_lookup: Dict[int, Tuple[str, int]] = {}
        for item in node_vocab.get('node_tokens', []):
            (ntype, idx, tok) = (str(item['type']).lower(), int(item['index']), str(item['token']))
            token_lookup[token_id(tokenizer, tok)] = (ntype, idx)
        rows = list(read_jsonl(prompts_path))
        if not rows:
            raise ValueError(f'extract_prompts.jsonl 为空: {prompts_path}')
        device = model.get_input_embeddings().weight.device
        for start in range(0, len(rows), max(1, batch_size)):
            batch = rows[start:start + max(1, batch_size)]
            texts = [str(r.get('text', '')) for r in batch]
            path_weights = [max(float(r.get('path_score', 1.0)), 1e-12) for r in batch]
            for (row_index, text) in enumerate(texts, start=start + 1):
                if not text.startswith('PATH:'):
                    raise ValueError(f'Hidden extraction row {row_index} is not a pure PATH sequence')
                if any((marker in text for marker in ('NODE_TEXT:', 'description:', 'text='))):
                    raise ValueError(f'Raw description content found in hidden extraction row {row_index}')
            encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=False)
            lengths = encoded.get('attention_mask', torch.ones_like(encoded['input_ids'])).sum(dim=1)
            too_long = torch.nonzero(lengths > max_length, as_tuple=False).flatten().tolist()
            if too_long:
                details = [int(lengths[index].item()) for index in too_long]
                raise ValueError(f'Complete meta-path does not fit within max_length; rows={[start + index + 1 for index in too_long]}, tokens={details}, max_length={max_length}. Increase --max_length.')
            encoded = {k: v.to(device) for (k, v) in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
                hidden = output.hidden_states[-1].detach().float().cpu()
            input_ids = encoded['input_ids'].detach().cpu()
            attention = encoded.get('attention_mask', torch.ones_like(encoded['input_ids'])).detach().cpu()
            for b in range(input_ids.shape[0]):
                valid_ids = input_ids[b][attention[b].bool()]
                valid_hidden = hidden[b][attention[b].bool()]
                node_positions = [position for (position, token) in enumerate(valid_ids.tolist()) if int(token) in token_lookup]
                for target_position in node_positions[1 + 0:]:
                    context_position = target_position - 1
                    if context_position < 0:
                        continue
                    (ntype, idx) = token_lookup[int(valid_ids[target_position].item())]
                    if ntype not in accum or idx >= accum[ntype].shape[0]:
                        continue
                    weight = path_weights[b]
                    accum[ntype][idx] += valid_hidden[context_position] * weight
                    counts[ntype][idx] += 1
                    weight_sums[ntype][idx] += weight
        contextual = {}
        coverage = {}
        for ntype in NODE_TYPES:
            table = static[ntype].clone()
            mask = counts[ntype] > 0
            if mask.any():
                table[mask] = accum[ntype][mask] / weight_sums[ntype][mask].unsqueeze(1)
            contextual[ntype] = table
            coverage[ntype] = {
                'num_contextualized': int(mask.sum().item()),
                'num_static_fallback': int((~mask).sum().item()),
                'total_occurrences': int(counts[ntype].sum().item()),
                'num_nodes': int(len(mask)),
                'coverage': float(mask.float().mean().item()) if len(mask) else 0.0
            }
        return (
            contextual,
            {
                'num_meta_paths': len(rows),
                'coverage': coverage,
                'contextual_count': {ntype: counts[ntype].tolist() for ntype in NODE_TYPES},
                'pooling': 'score_weighted_mean',
                'minimum_graph_prefix_nodes': int(1 + 0),
                'prompt_directions': ['forward', 'reverse'],
                'path_weight': 'structural_entropy_graphrag_score',
                'hidden_alignment': 'all_causal_prefix_hidden_at_target_minus_one',
                'fallback': 'static_lora_node_token_embedding'
            }
        )
    configure_torch_determinism(True)
    os.makedirs(args.output_dir, exist_ok=True)
    validate_extraction_prompts(args.extract_prompts_jsonl, Path(args.split_data))
    node_vocab = load_node_vocab(args.node_vocab_json)
    special_tokens = load_special_tokens(node_vocab)
    tokenizer_path = args.adapter_dir if os.path.exists(os.path.join(args.adapter_dir, 'tokenizer_config.json')) else args.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        device_map={'': args.device},
        dtype=torch.bfloat16,
        trust_remote_code=False
    )
    add_tokens_if_needed(tokenizer, model, special_tokens)
    model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model = freeze_model_for_inference(model)
    contextual_info = None
    if not args.extract_prompts_jsonl:
        raise ValueError('contextual 模式必须提供 --extract_prompts_jsonl')
    (node_emb, contextual_info) = contextualize_node_tables(model, tokenizer, node_vocab, args.extract_prompts_jsonl, args.batch_size, args.max_length)
    result_path = args.output_file or os.path.join(args.output_dir, 'llm_hidden.pt')
    metadata = {
        'base_model_name_or_path': args.base_model_name_or_path,
        'adapter_dir': os.path.abspath(args.adapter_dir),
        'node_vocab_json': os.path.abspath(args.node_vocab_json),
        'embedding_mode': 'contextual',
        'extract_prompts_jsonl': os.path.abspath(args.extract_prompts_jsonl) if args.extract_prompts_jsonl else None,
        'contextual_info': contextual_info,
        'num_nodes': {k: int(v.shape[0]) for (k, v) in node_emb.items()},
        'embedding_dim': int(node_emb['rna'].shape[1]) if node_emb else None,
        'llm_embedding_dim': int(node_emb['rna'].shape[1]),
        'llm_embedding_source': 'meta_path_hidden_state',
        'pooling': 'score_weighted_mean',
        'minimum_graph_prefix_nodes': int(1 + 0),
        'hidden_alignment': 'all_causal_prefix_hidden_at_target_minus_one',
        'path_weight': 'structural_entropy_graphrag_score',
        'prompt_directions': ['forward', 'reverse'],
        'fallback': 'static_lora_node_token_embedding',
        'raw_description_text_used': False,
        'model_parameters_frozen': True
    }
    temporary_path = result_path + '.tmp'
    torch.save(
        {
            'node_embeddings': node_emb,
            'llm_node_embeddings': node_emb,
            'llm_embedding_source': 'meta_path_hidden_state',
            'pooling': 'score_weighted_mean',
            'minimum_graph_prefix_nodes': int(1 + 0),
            'hidden_alignment': 'all_causal_prefix_hidden_at_target_minus_one',
            'path_weight': 'structural_entropy_graphrag_score',
            'prompt_directions': ['forward', 'reverse'],
            'fallback': 'static_lora_node_token_embedding',
            'model_parameters_frozen': True,
            'metadata': metadata
        },
        temporary_path
    )
    os.replace(temporary_path, result_path)
    metadata['result_path'] = os.path.abspath(result_path)
    metadata_path = os.path.join(args.output_dir, 'llm_metadata.json')
    metadata_temporary_path = metadata_path + '.tmp'
    with open(metadata_temporary_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(metadata_temporary_path, metadata_path)
    print(f"Embedding 抽取完成（mode={'contextual'}）：{result_path}")

def validate_extraction_prompts(path, split_path):
    split = torch_load_cpu(split_path)
    allowed = {tuple(map(int, p)) for p in split['train_positive_pairs'].tolist()}
    count = 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            nodes = row.get('path_nodes', [])
            if row.get('source') != 'graph_train' or len(nodes) < 3:
                raise ValueError('Hidden extraction requires training-only graph paths')
            (first, last) = (nodes[0], nodes[-1])
            if (first[0], last[0]) == ('drug', 'rna'):
                (first, last) = (last, first)
            if (first[0], last[0]) != ('rna', 'drug') or (int(first[1]), int(last[1])) not in allowed:
                raise ValueError('Held-out/non-training pair in hidden extraction')
            for (a, b) in zip(nodes, nodes[1:]):
                if (a[0], b[0]) == ('drug', 'rna'):
                    (a, b) = (b, a)
                if (a[0], b[0]) == ('rna', 'drug') and (int(a[1]), int(b[1])) not in allowed:
                    raise ValueError('Held-out association edge in a training path')
            count += 1
    if not count:
        raise ValueError('Hidden extraction corpus is empty')
