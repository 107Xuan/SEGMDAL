"""SE graph Euclidean/hyperbolic GNN with curvature routing."""
from __future__ import annotations
from torch import nn
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Tuple
import copy
import numpy as np
import torch
from dataclasses import dataclass
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from data_processing import HeteroGraphData
NodeType = str
Relation = Tuple[str, str, str]
TARGET_LINK_TYPES = frozenset({('rna', 'drug'), ('drug', 'rna')})

@dataclass(frozen=True)
class EdgeSet:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    def to(self, device: torch.device) -> 'EdgeSet':
        return EdgeSet(self.edge_index.to(device), self.edge_weight.to(device))

def _relation_key(relation: Relation) -> str:
    return '__'.join(relation)

def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(1e-08)

def _deterministic_scatter_sum(index: torch.Tensor, values: torch.Tensor, size: int) -> torch.Tensor:
    output_shape = (size, *values.shape[1:])
    if index.numel() == 0:
        return values.new_zeros(output_shape)
    order = torch.argsort(index, stable=True)
    sorted_index = index[order]
    sorted_values = values[order]
    cumulative = sorted_values.cumsum(dim=0)
    is_end = torch.ones_like(sorted_index, dtype=torch.bool)
    is_end[:-1] = sorted_index[:-1] != sorted_index[1:]
    end_positions = torch.nonzero(is_end, as_tuple=False).flatten()
    segment_sums = cumulative[end_positions]
    if len(end_positions) > 1:
        previous = cumulative[end_positions[:-1]]
        segment_sums = torch.cat((segment_sums[:1], segment_sums[1:] - previous), dim=0)
    unique_index = sorted_index[end_positions]
    output = values.new_zeros(output_shape)
    return output.index_copy(0, unique_index, segment_sums)

def poincare_expmap0(v: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    c = curvature.clamp_min(1e-05)
    sqrt_c = c.sqrt()
    norm = _safe_norm(v)
    x = torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm)
    max_norm = (1.0 - 1e-05) / sqrt_c
    return x * torch.clamp(max_norm / _safe_norm(x), max=1.0)

def poincare_logmap0(x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    c = curvature.clamp_min(1e-05)
    sqrt_c = c.sqrt()
    norm = _safe_norm(x)
    scaled = (sqrt_c * norm).clamp(max=1.0 - 1e-05)
    return torch.atanh(scaled) * x / (sqrt_c * norm)

def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    c = curvature.clamp_min(1e-05)
    x2 = x.square().sum(dim=-1).clamp_max((1.0 - 1e-05) / c)
    y2 = y.square().sum(dim=-1).clamp_max((1.0 - 1e-05) / c)
    diff2 = (x - y).square().sum(dim=-1)
    denominator = ((1.0 - c * x2) * (1.0 - c * y2)).clamp_min(1e-08)
    argument = (1.0 + 2.0 * c * diff2 / denominator).clamp_min(1.0 + 1e-07)
    return torch.acosh(argument) / c.sqrt()

class AdaptiveMetricRelationLayer(nn.Module):

    def __init__(
        self,
        node_types: Iterable[str],
        relations: Iterable[Relation],
        dim: int,
        output_types: Iterable[str] | None=None
    ):
        super().__init__()
        self.node_types = tuple(node_types)
        self.relations = tuple(relations)
        self.output_types = tuple(output_types or self.node_types)
        invalid_outputs = set(self.output_types).difference(self.node_types)
        if invalid_outputs:
            raise ValueError(f'Unknown output node types: {sorted(invalid_outputs)}')
        self.self_linear = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.output_types})
        self.metric_net = nn.ModuleDict({t: nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim)) for t in self.node_types})
        self.relation_linear = nn.ModuleDict({_relation_key(r): nn.Linear(dim, dim, bias=False) for r in self.relations if r[2] in self.output_types})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(dim) for t in self.output_types})

    def _neighbour_mean(self, x: Dict[NodeType, torch.Tensor], edges: Dict[Relation, EdgeSet]) -> Dict[NodeType, torch.Tensor]:
        total = {t: torch.zeros_like(v) for (t, v) in x.items()}
        degree = {t: v.new_zeros(v.shape[0]) for (t, v) in x.items()}
        for ((src_type, _, dst_type), edge) in edges.items():
            (src, dst) = edge.edge_index
            if src.numel() == 0:
                continue
            w = edge.edge_weight.clamp_min(0.0)
            total[dst_type] = total[dst_type] + _deterministic_scatter_sum(dst, x[src_type][src] * w[:, None], total[dst_type].shape[0])
            degree[dst_type] = degree[dst_type] + _deterministic_scatter_sum(dst, w, degree[dst_type].shape[0])
        return {t: total[t] / degree[t].clamp_min(1.0)[:, None] for t in x}

    def forward(self, x: Dict[NodeType, torch.Tensor], edges: Dict[Relation, EdgeSet]) -> tuple[Dict[NodeType, torch.Tensor], Dict[NodeType, torch.Tensor], torch.Tensor, torch.Tensor]:
        neighbour = self._neighbour_mean(x, edges)
        metric = {}
        for t in x:
            g = F.softplus(self.metric_net[t](torch.cat((x[t], neighbour[t]), dim=-1))) + 0.0001
            metric[t] = (g / g.mean(dim=-1, keepdim=True).clamp_min(1e-06)).clamp(0.05, 20.0)
        aggregated = {t: torch.zeros_like(x[t]) for t in self.output_types}
        mass = {t: x[t].new_zeros(x[t].shape[0]) for t in self.output_types}
        smooth_terms = []
        consistency_sum = {t: torch.zeros_like(metric[t]) for t in x}
        consistency_degree = {t: x[t].new_zeros(x[t].shape[0]) for t in x}
        for (relation, edge) in edges.items():
            (src_type, _, dst_type) = relation
            (src, dst) = edge.edge_index
            if src.numel() == 0:
                continue
            w = edge.edge_weight.clamp_min(0.0)
            source = x[src_type][src]
            target = x[dst_type][dst]
            g_src = metric[src_type][src]
            g_dst = metric[dst_type][dst]
            if dst_type in aggregated:
                transformed = self.relation_linear[_relation_key(relation)](source)
                delta = source - target
                direction = delta / _safe_norm(delta)
                g_edge = torch.sqrt((g_src * g_dst).clamp_min(1e-08))
                tau = (direction.square() * torch.tanh(torch.log(g_edge))).sum(dim=-1)
                numerator = (g_edge * source * target).sum(dim=-1)
                source_norm = (g_edge * source.square()).sum(dim=-1).clamp_min(1e-08).sqrt()
                target_norm = (g_edge * target.square()).sum(dim=-1).clamp_min(1e-08).sqrt()
                denominator = source_norm * target_norm
                alpha = (numerator / denominator).clamp(-1.0, 1.0)
                gate = torch.sigmoid(alpha + tau) * w
                aggregated[dst_type] = aggregated[dst_type] + _deterministic_scatter_sum(dst, transformed * gate[:, None], aggregated[dst_type].shape[0])
                mass[dst_type] = mass[dst_type] + _deterministic_scatter_sum(dst, gate, mass[dst_type].shape[0])
            metric_delta = metric[src_type][src] - g_dst
            smooth_terms.append((metric_delta.square().mean(dim=-1) * w).mean())
            consistency_sum[dst_type] = consistency_sum[dst_type] + _deterministic_scatter_sum(dst, metric_delta * w[:, None], consistency_sum[dst_type].shape[0])
            consistency_degree[dst_type] = consistency_degree[dst_type] + _deterministic_scatter_sum(dst, w, consistency_degree[dst_type].shape[0])
        output = {}
        for t in self.output_types:
            message = aggregated[t] / mass[t].clamp_min(1.0)[:, None]
            output[t] = self.norm[t](x[t] + self.self_linear[t](x[t]) + message)
        zero = next(iter(x.values())).new_zeros(())
        smooth_loss = torch.stack(smooth_terms).mean() if smooth_terms else zero
        metric_consistency_loss = torch.stack([(consistency_sum[t] / consistency_degree[t].clamp_min(1.0)[:, None]).square().mean() for t in x]).mean()
        return (output, metric, metric_consistency_loss, smooth_loss)

class GeometryExpert(nn.Module):

    def __init__(
        self,
        geometry: str,
        node_types: Iterable[str],
        relations: Iterable[Relation],
        dim: int,
        layers: int,
        dropout: float
    ):
        super().__init__()
        self.geometry = geometry
        self.dropout = dropout
        self.curvature_raw = nn.Parameter(torch.tensor(-1.5)) if geometry != 'euclidean' else None
        if layers < 1:
            raise ValueError('GeometryExpert requires at least one relation layer')
        node_types = tuple(node_types)
        relations = tuple(relations)
        endpoint_types = tuple((t for t in ('rna', 'drug') if t in node_types))
        self.layers = nn.ModuleList([AdaptiveMetricRelationLayer(node_types, relations, dim, output_types=endpoint_types if index == layers - 1 else node_types) for index in range(layers)])

    def curvature(self) -> torch.Tensor:
        if self.curvature_raw is None:
            return next(self.parameters()).new_tensor(0.0)
        return F.softplus(self.curvature_raw) + 0.0001

    def _to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        if self.geometry == 'hyperbolic':
            return poincare_expmap0(x, self.curvature())
        return x

    def _to_tangent(self, x: torch.Tensor) -> torch.Tensor:
        if self.geometry == 'hyperbolic':
            return poincare_logmap0(x, self.curvature())
        return x

    def forward(self, x: Dict[str, torch.Tensor], edges: Dict[Relation, EdgeSet]):
        manifold = {t: self._to_manifold(v) for (t, v) in x.items()}
        metric_consistency = next(iter(x.values())).new_zeros(())
        smooth = metric_consistency.clone()
        metrics: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            tangent = {t: self._to_tangent(v) for (t, v) in manifold.items()}
            (tangent, metrics, consistency, s) = layer(tangent, edges)
            tangent = {t: F.dropout(F.gelu(v), self.dropout, self.training) for (t, v) in tangent.items()}
            manifold = {t: self._to_manifold(v) for (t, v) in tangent.items()}
            (metric_consistency, smooth) = (metric_consistency + consistency, smooth + s)
        tangent = {t: self._to_tangent(v) for (t, v) in manifold.items()}
        return (tangent, manifold, metrics, metric_consistency / len(self.layers), smooth / len(self.layers))

class FeatureEncoder(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, encoder_width: int, dropout: float):
        super().__init__()
        intermediate_dim = min(encoder_width, max(input_dim, output_dim))
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)

class GeometricLinkDecoder(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.classifier = nn.Linear(2 * hidden_dim + 3, 1)

    def forward(self, rna: torch.Tensor, drug: torch.Tensor, distances: list[torch.Tensor]) -> torch.Tensor:
        distance_matrix = torch.log1p(torch.cat(distances, dim=-1).clamp_min(0.0))
        pair_features = torch.cat((rna * drug, (rna - drug).abs(), distance_matrix), dim=-1)
        return self.classifier(pair_features).squeeze(-1)

class RiemannianMixtureGNN(nn.Module):

    def __init__(
        self,
        input_dims: Dict[str, int],
        relations: Iterable[Relation],
        hidden_dim: int=128,
        encoder_width: int=256,
        num_layers: int=2,
        dropout: float=0.25,
        curvature_threshold: float=0.15
    ):
        super().__init__()
        self.node_types = tuple(input_dims)
        self.relations = tuple(relations)
        self.dropout = dropout
        self.curvature_threshold = curvature_threshold
        if encoder_width < 1:
            raise ValueError('encoder_width 必须为正整数')
        self.feature_encoders = nn.ModuleDict({t: FeatureEncoder(d, hidden_dim, encoder_width, dropout) for (t, d) in input_dims.items()})
        self.input_norm = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in self.node_types})
        self.experts = nn.ModuleDict({name: GeometryExpert(name, self.node_types, self.relations, hidden_dim, num_layers, dropout) for name in ('euclidean', 'hyperbolic')})
        # Keep seeded decoder initialization consistent without retaining a third expert.
        GeometryExpert('hyperbolic', self.node_types, self.relations, hidden_dim, num_layers, dropout)
        self.decoder = GeometricLinkDecoder(hidden_dim)

    def _base(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing_types = set(self.node_types).difference(features)
        if missing_types:
            raise ValueError(f'缺少节点特征: {sorted(missing_types)}')
        base = {t: self.input_norm[t](self.feature_encoders[t](features[t])) for t in self.node_types}
        base = {t: F.dropout(F.gelu(value), self.dropout, self.training) for (t, value) in base.items()}
        return base

    def forward(self, features, edges, pairs, curvature_prior, return_aux: bool=False):
        (fused, _, aux) = self.encode(features, edges, curvature_prior)
        logits = self.decode(fused, aux['manifold'], pairs)
        return (logits, aux) if return_aux else logits

    def encode(self, features, edges, curvature_prior):
        base = self._base(features)
        (outputs, manifold, metric) = ({}, {}, {})
        (consistency, smoothness) = ([], [])
        for (name, expert) in self.experts.items():
            (outputs[name], manifold[name], metric[name], c, s) = expert(base, edges)
            consistency.append(c)
            smoothness.append(s)
        (gates, fused) = ({}, {})
        for kind in next(iter(outputs.values())):
            gates[kind] = self.hard_curvature_gates(curvature_prior[kind].reshape(-1))
            fused[kind] = sum((gates[kind][:, ORDER.index(name), None] * value[kind] for (name, value) in outputs.items()))
        return (
            fused,
            outputs,
            dict(
                manifold=manifold,
                metrics=metric,
                gates=gates,
                metric_consistency_loss=torch.stack(consistency).mean(),
                smoothness_loss=torch.stack(smoothness).mean()
            )
        )

    def decode(self, fused, manifold, pairs):
        (left, right) = (fused['rna'][pairs[:, 0]], fused['drug'][pairs[:, 1]])
        distances = distances_for(self, manifold, pairs, left)
        return self.decoder(left, right, distances)

    def hard_curvature_gates(self, kappa):
        index = (kappa < -self.curvature_threshold).long()
        return F.one_hot(index, num_classes=3).to(kappa.dtype)

def graph_tensors(graph: HeteroGraphData, device: torch.device, similarity_edge_scale: float=1.0) -> tuple[dict[str, torch.Tensor], dict[Relation, EdgeSet]]:
    if similarity_edge_scale <= 0.0:
        raise ValueError('similarity_edge_scale must be positive')
    feature_info = getattr(graph, 'node_feature_info', {})
    invalid_sources = [node_type for node_type in graph.node_features if feature_info.get(node_type, {}).get('source') != 'feature_folder_only']
    if invalid_sources:
        raise ValueError(f'Graph node features must come from data/feature for {invalid_sources}. Rebuild the graph cache before training.')
    features = {node_type: values.float().to(device) for (node_type, values) in graph.node_features.items()}
    edges: dict[Relation, EdgeSet] = {}
    for (node_type, (edge_index, edge_weight)) in graph.sim_edges.items():
        edges[node_type, 'similar', node_type] = EdgeSet(edge_index.long(), edge_weight.float() * similarity_edge_scale).to(device)
    for ((src_type, dst_type), (edge_index, edge_weight)) in graph.inter_edges.items():
        if (src_type, dst_type) in TARGET_LINK_TYPES:
            continue
        edges[src_type, 'interacts', dst_type] = EdgeSet(edge_index.long(), edge_weight.float()).to(device)
    return (features, edges)

def topology_curvature_prior(graph: HeteroGraphData, device: torch.device, max_neighbours: int=96) -> dict[str, torch.Tensor]:
    total_nodes = len(graph.local_node_id)
    neighbours: list[set[int]] = [set() for _ in range(total_nodes)]

    def add_edges(src_type: str, dst_type: str, edge_index: torch.Tensor) -> None:
        if edge_index.numel() == 0:
            return
        for (src, dst) in edge_index.t().tolist():
            left = graph.get_global_id(src_type, int(src))
            right = graph.get_global_id(dst_type, int(dst))
            if left >= 0 and right >= 0 and (left != right):
                neighbours[left].add(right)
                neighbours[right].add(left)
    for (node_type, (edge_index, _)) in graph.sim_edges.items():
        add_edges(node_type, node_type, edge_index)
    for (src_type, dst_type) in (('rna', 'disease'), ('drug', 'disease')):
        edge_index = graph.inter_edges.get((src_type, dst_type), (torch.empty((2, 0)), None))[0]
        add_edges(src_type, dst_type, edge_index)
    degrees = np.asarray([len(values) for values in neighbours], dtype=np.float64)
    max_log_degree = max(float(np.log1p(degrees).max()), 1.0)
    curvature = np.zeros(total_nodes, dtype=np.float32)
    for (node, adjacent) in enumerate(neighbours):
        degree = len(adjacent)
        if degree < 2:
            curvature[node] = -1.0 if degree == 1 else 0.0
            continue
        ordered = sorted(adjacent)
        if degree > max_neighbours:
            positions = np.linspace(0, degree - 1, max_neighbours, dtype=np.int64)
            ordered = [ordered[index] for index in positions]
        closed = 0
        possible = 0
        for (left, neighbour) in enumerate(ordered):
            for other in ordered[left + 1:]:
                possible += 1
                closed += int(other in neighbours[neighbour])
        clustering = closed / max(1, possible)
        branching = np.log1p(degree) / max_log_degree
        curvature[node] = float(np.tanh(2.5 * clustering - branching))
    result: dict[str, torch.Tensor] = {}
    for node_type in graph.node_features:
        values = [curvature[graph.get_global_id(node_type, index)] for index in range(graph.num_nodes(node_type))]
        result[node_type] = torch.tensor(values, dtype=torch.float32, device=device)
    return result

@torch.no_grad()
def predict_gnn(
    model: RiemannianMixtureGNN,
    features: dict[str, torch.Tensor],
    edges: dict[Relation, EdgeSet],
    pairs: torch.Tensor,
    curvature: dict[str, torch.Tensor],
    device: torch.device,
    pair_prior: torch.Tensor | None=None,
    se_prior_weight: float=0.0
) -> torch.Tensor:
    model.eval()
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    logits = model(features, edges, pair_tensor, curvature)
    if pair_prior is not None and se_prior_weight > 0.0:
        pair_values = pair_prior.to(device)[pair_tensor[:, 0], pair_tensor[:, 1]]
        prior_score = 0.45 * pair_values[:, 0] + 0.35 * pair_values[:, 1] + 0.2 * pair_values[:, 2]
        logits = logits + float(se_prior_weight) * torch.tanh(prior_score)
    return logits.detach().cpu()

def train_gnn_branch(args: Any, graph: HeteroGraphData, split: dict[str, Any], device: torch.device, metric_fn) -> tuple[
    RiemannianMixtureGNN,
    dict[str, torch.Tensor],
    dict[Relation, EdgeSet],
    dict[str, torch.Tensor],
    torch.Tensor | None
]:
    (features, edges) = graph_tensors(graph, device, similarity_edge_scale=args.similarity_edge_scale)
    curvature = topology_curvature_prior(graph, device)
    pair_prior = getattr(graph, 'se_pair_prior', None)
    if pair_prior is not None:
        pair_prior = pair_prior.float().to(device)
    model = RiemannianMixtureGNN(
        input_dims={node_type: int(value.shape[1]) for (node_type, value) in features.items()},
        relations=tuple(edges),
        hidden_dim=args.gnn_hidden_dim,
        encoder_width=args.gnn_encoder_width,
        num_layers=args.gnn_num_layers,
        dropout=args.gnn_dropout,
        curvature_threshold=args.curvature_threshold
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    train_pairs = split['train_pairs'].long().to(device)
    train_labels = split['train_labels'].float().to(device)
    val_pairs = split['val_pairs'].long()
    val_labels = split['val_labels'].float()
    best_score = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.gnn_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        (logits, aux) = model(features, edges, train_pairs, curvature, return_aux=True)
        if pair_prior is not None and args.se_prior_weight > 0.0:
            pair_values = pair_prior[train_pairs[:, 0], train_pairs[:, 1]]
            prior_score = 0.45 * pair_values[:, 0] + 0.35 * pair_values[:, 1] + 0.2 * pair_values[:, 2]
            logits = logits + args.se_prior_weight * torch.tanh(prior_score)
        task_loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss = task_loss + args.metric_consistency_weight * aux['metric_consistency_loss'] + args.smoothness_weight * aux['smoothness_loss']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite GNN loss at epoch {epoch}: {loss.detach().cpu().item()}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        metrics = metric_fn(
            predict_gnn(
                model,
                features,
                edges,
                val_pairs,
                curvature,
                device,
                pair_prior=pair_prior,
                se_prior_weight=args.se_prior_weight
            ),
            val_labels
        )
        score = metrics['AUC']
        if score > best_score + 1e-08:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'  GNN epoch {epoch:03d}: loss={loss.item():.4f}, val AUC={score:.4f}')
        if stale >= args.gnn_patience:
            break
    if best_state is None:
        raise RuntimeError('GNN branch did not produce a valid state')
    model.load_state_dict(best_state)
    model.se_calibration = None
    if pair_prior is not None:
        raw_val_logits = predict_gnn(
            model,
            features,
            edges,
            val_pairs,
            curvature,
            device,
            pair_prior=pair_prior,
            se_prior_weight=args.se_prior_weight
        ).numpy()
        calibration_x = raw_val_logits.reshape(-1, 1)
        calibration_mean = calibration_x.mean(axis=0)
        calibration_scale = np.maximum(calibration_x.std(axis=0), 1e-08)
        calibration_x = (calibration_x - calibration_mean) / calibration_scale
        calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        calibrator.fit(calibration_x, val_labels.numpy().astype(np.int64))
        model.se_calibration = {
            'mean': calibration_mean,
            'scale': calibration_scale,
            'coef': calibrator.coef_[0].copy(),
            'intercept': float(calibrator.intercept_[0])
        }
    model.training_summary = {
        'best_epoch': int(best_epoch),
        'best_validation_auc': float(best_score),
        'se_calibration_coefficients': model.se_calibration['coef'].tolist() if model.se_calibration is not None else None,
        'se_prior_weight': float(args.se_prior_weight),
        'similarity_edge_gate': 'intra_module=1.8,cross_module=0.35',
        'geometry_routing_method': 'hard_topology_curvature',
        'hard_curvature_routing': True
    }
    return (model, features, edges, curvature, pair_prior)

def apply_se_calibration(model: RiemannianMixtureGNN, raw_logits: torch.Tensor) -> torch.Tensor:
    if model.se_calibration is None:
        return raw_logits
    state = model.se_calibration
    values = raw_logits.detach().cpu().numpy().reshape(-1, 1)
    values = (values - state['mean']) / state['scale']
    calibrated = values @ state['coef'] + state['intercept']
    return torch.from_numpy(calibrated.astype(np.float32))
ORDER = ('euclidean', 'hyperbolic')

def distances_for(model, manifold, pairs, reference):
    a = manifold['euclidean']['rna'][pairs[:, 0]]
    b = manifold['euclidean']['drug'][pairs[:, 1]]
    euclidean = ((a - b).square().sum(-1, keepdim=True) + 1e-08).sqrt()
    a = manifold['hyperbolic']['rna'][pairs[:, 0]]
    b = manifold['hyperbolic']['drug'][pairs[:, 1]]
    hyperbolic = poincare_distance(a, b, model.experts['hyperbolic'].curvature())[:, None]
    # The zero distance retains native checkpoint and fusion feature widths.
    return [euclidean, hyperbolic, reference.new_zeros((len(pairs), 1))]

@torch.no_grad()
def pair_features(model, features, edges, curvature, pairs):
    model.eval()
    (fused, _, aux) = model.encode(features, edges, curvature)
    pairs = pairs.to(fused['rna'].device).long()
    (left, right) = (fused['rna'][pairs[:, 0]], fused['drug'][pairs[:, 1]])
    ds = distances_for(model, aux['manifold'], pairs, left)
    a = aux['manifold']['euclidean']['rna'][pairs[:, 0]]
    b = aux['manifold']['euclidean']['drug'][pairs[:, 1]]
    ds[0] = (a - b).square().sum(-1, keepdim=True).clamp_min(1e-08).sqrt()
    return torch.cat((left, right, (left - right).abs(), left * right, *ds), -1).cpu()
