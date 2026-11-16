"""Strict data/feature alignment and leakage-safe five-fold SE graph preparation."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import argparse
import numpy as np
import os
import time
import torch
import pandas as pd
from sklearn.model_selection import KFold
from config import CONFIG, DATA_FILES, NODE_FEATURE_FILES
from structural_entropy import adjacency_to_edge_index, build_similarity_adjacency, compute_delta_h2_add, compute_module_distance, low_entropy_partition, two_dim_structural_entropy, load_matrix_interactions, load_pair_interactions, load_similarity_matrix, normalize_identifier
DEFAULT_BOTH_MISSING_CONTEXT_DISSIMILARITY = 0.5

def _progress(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)

def _load_aligned_node_feature_view(path: str, id_column: str, expected_dim: int, node_names: List[str]) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Node feature file not found: {path}')
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise ValueError(f'{path} is missing identifier column {id_column!r}')
    feature_columns = [column for column in frame.columns if column != id_column]
    if len(feature_columns) != expected_dim:
        raise ValueError(f'{path} must contain {expected_dim} feature columns, got {len(feature_columns)}')
    identifiers = frame[id_column].map(normalize_identifier).tolist()
    expected_names = [normalize_identifier(name) for name in node_names]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f'{path} contains duplicate node identifiers')
    if len(set(expected_names)) != len(expected_names):
        raise ValueError('Graph contains duplicate node identifiers')
    available = set(identifiers)
    expected = set(expected_names)
    missing = [name for name in expected_names if name not in available]
    unexpected = [name for name in identifiers if name not in expected]
    if missing or unexpected:
        raise ValueError(f'{path} does not match graph nodes; missing={missing[:10]}, unexpected={unexpected[:10]}')
    numeric = frame[feature_columns].apply(pd.to_numeric, errors='raise')
    row_by_name = {name: index for (index, name) in enumerate(identifiers)}
    row_order = [row_by_name[name] for name in expected_names]
    values = numeric.to_numpy(dtype=np.float32, copy=True)[row_order]
    if not np.isfinite(values).all():
        raise ValueError(f'{path} contains NaN or infinite feature values')
    return torch.from_numpy(values)

def load_node_features(data_dir: str, node_names: Dict[str, List[str]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    features: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Any] = {}
    for (node_type, modalities) in NODE_FEATURE_FILES.items():
        if node_type not in node_names:
            raise ValueError(f'Missing node names for type {node_type!r}')
        views = []
        modality_dims = {}
        for (name, (relative_path, id_column, expected_dim)) in modalities.items():
            view = _load_aligned_node_feature_view(os.path.join(data_dir, relative_path), id_column, expected_dim, node_names[node_type])
            views.append(view)
            modality_dims[name] = int(view.shape[1])
        features[node_type] = torch.cat(views, dim=1)
        metadata[node_type] = {
            'source': 'feature_folder_only',
            'modalities': modality_dims,
            'total_dim': int(features[node_type].shape[1])
        }
    return (features, metadata)

def _elapsed(start_time: float) -> str:
    seconds = max(0.0, time.perf_counter() - start_time)
    if seconds < 60:
        return f'{seconds:.2f}s'
    (minutes, seconds) = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m {seconds:.1f}s'
    (hours, minutes) = divmod(minutes, 60)
    return f'{int(hours)}h {int(minutes)}m {seconds:.1f}s'

def _ensure_parent_dir(path: str) -> str:
    absolute_path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(absolute_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return absolute_path

def _torch_save_safe(obj, path: str) -> None:
    absolute_path = _ensure_parent_dir(path)
    tmp_path = absolute_path + '.tmp'
    try:
        with open(tmp_path, 'wb') as f:
            torch.save(obj, f)
        os.replace(tmp_path, absolute_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

class HeteroGraphData:

    def __init__(self):
        self.node_features: Dict[str, torch.Tensor] = {}
        self.sim_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.inter_edges: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.partitions: Dict[str, np.ndarray] = {}
        self.names: Dict[str, List[str]] = {}
        self.se_pair_prior: Optional[torch.Tensor] = None
        self.se_filter_info: Dict[str, Dict] = {}
        self.module_info: Dict[str, Dict] = {}
        self.global_node_id: Dict[Tuple[str, int], int] = {}
        self.local_node_id: Dict[int, Tuple[str, int]] = {}
        self.rna_drug_matrix_full: Optional[np.ndarray] = None
        self.rna_drug_matrix: Optional[np.ndarray] = None
        self.sim_degree: Dict[str, np.ndarray] = {}
        self.raw_similarity: Dict[str, np.ndarray] = {}
        self.node_feature_info: Dict[str, Any] = {}
        self.global_adjacency: Optional[np.ndarray] = None
        self.global_partition: Optional[np.ndarray] = None
        self.global_module_distance: Optional[np.ndarray] = None
        self.global_module_to_pos: Dict[int, int] = {}
        self.global_structure_info: Dict = {}
        self.initial_global_adjacency: Optional[np.ndarray] = None
        self.initial_structure_info: Dict = {}
        self.fold_info: Dict = {}
        self.negative_samples: Dict[str, Any] = {}

    def num_nodes(self, ntype: str) -> int:
        return self.node_features[ntype].shape[0]

    def get_global_id(self, ntype: str, idx: int) -> int:
        return self.global_node_id.get((ntype, idx), -1)

    def build_global_node_mapping(self):
        global_id = 0
        for ntype in ['rna', 'drug', 'disease']:
            if ntype in self.node_features:
                for local_idx in range(self.num_nodes(ntype)):
                    self.global_node_id[ntype, local_idx] = global_id
                    self.local_node_id[global_id] = (ntype, local_idx)
                    global_id += 1

    def build_module_info(self):
        for (ntype, labels) in self.partitions.items():
            node_to_module = {i: int(labels[i]) for i in range(len(labels))}
            module_to_nodes: Dict[int, List[int]] = {}
            for (node, mod) in node_to_module.items():
                module_to_nodes.setdefault(mod, []).append(node)
            module_size = {m: len(nodes) for (m, nodes) in module_to_nodes.items()}
            if ntype in self.sim_degree:
                deg = self.sim_degree[ntype]
                module_volume = {m: float(deg[nodes].sum()) for (m, nodes) in module_to_nodes.items()}
            else:
                module_volume = {m: len(nodes) for (m, nodes) in module_to_nodes.items()}
            self.module_info[ntype] = {
                'node_to_module': node_to_module,
                'module_to_nodes': module_to_nodes,
                'module_size': module_size,
                'module_volume': module_volume
            }

def _inter_edge_index(mat: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    (src, dst) = np.nonzero(mat > 0)
    if src.size == 0:
        return (torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float))
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    ew = torch.ones(src.size, dtype=torch.float)
    return (ei, ew)
CANONICAL_INTER_RELATIONS = (('rna', 'drug'), ('rna', 'disease'), ('drug', 'disease'))

def _build_global_adjacency(graph: HeteroGraphData, similarity_scale: float=1.0, interaction_scale: float=1.0) -> np.ndarray:
    total_nodes = len(graph.local_node_id)
    if total_nodes == 0:
        raise RuntimeError('请先调用 graph.build_global_node_mapping()')
    W = np.zeros((total_nodes, total_nodes), dtype=np.float64)

    def add_edge(global_src: int, global_dst: int, weight: float):
        if global_src < 0 or global_dst < 0:
            raise ValueError('节点全局编号不存在')
        if global_src == global_dst or weight <= 0:
            return
        value = max(W[global_src, global_dst], float(weight))
        W[global_src, global_dst] = value
        W[global_dst, global_src] = value
    for (node_type, (edge_index, edge_weight)) in graph.sim_edges.items():
        for k in range(edge_index.shape[1]):
            src = int(edge_index[0, k])
            dst = int(edge_index[1, k])
            add_edge(
                graph.get_global_id(node_type, src),
                graph.get_global_id(node_type, dst),
                similarity_scale * float(edge_weight[k])
            )
    for (src_type, dst_type) in CANONICAL_INTER_RELATIONS:
        (edge_index, edge_weight) = graph.inter_edges.get(
            (src_type, dst_type),
            (torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float))
        )
        for k in range(edge_index.shape[1]):
            src = int(edge_index[0, k])
            dst = int(edge_index[1, k])
            add_edge(graph.get_global_id(src_type, src), graph.get_global_id(dst_type, dst), interaction_scale * 1.0)
    return W

def build_global_structural_context(
    graph: HeteroGraphData,
    partition_passes: int=10,
    random_seed: Optional[int]=None,
    similarity_scale: float=1.0,
    interaction_scale: float=1.0,
    repartition: bool=True,
    verbose: bool=False
) -> Dict:
    W = _build_global_adjacency(graph, similarity_scale=similarity_scale, interaction_scale=interaction_scale)
    if repartition or graph.global_partition is None:
        labels = low_entropy_partition(W, max_pass=partition_passes, random_seed=random_seed, verbose=verbose)
        graph.global_partition = labels
    else:
        labels = graph.global_partition
    module_ids = np.unique(labels)
    graph.global_adjacency = W
    graph.global_module_to_pos = {int(module_id): pos for (pos, module_id) in enumerate(module_ids)}
    graph.global_module_distance = compute_module_distance(labels, W, verbose=verbose)
    graph.global_structure_info = {
        'num_nodes': int(W.shape[0]),
        'num_edges': int(np.count_nonzero(np.triu(W, k=1))),
        'num_modules': int(len(module_ids)),
        'H2': float(two_dim_structural_entropy(W, labels))
    }
    return graph.global_structure_info
DEFAULT_EASY_NEGATIVE_RATIO = 0.45
DEFAULT_HARD_NEGATIVE_RATIO = 0.45
DEFAULT_RANDOM_NEGATIVE_RATIO = 0.1
DEFAULT_FILES = DATA_FILES

def validate_negative_sample_ratios(easy_ratio: float, hard_ratio: float, random_ratio: float) -> Tuple[float, float, float]:
    ratios = tuple(map(float, (easy_ratio, hard_ratio, random_ratio)))
    names = ('easy_negative_ratio', 'hard_negative_ratio', 'random_negative_ratio')
    for (name, ratio) in zip(names, ratios):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f'{name} must be in [0, 1], got {ratio}')
    if not np.isclose(sum(ratios), 1.0, rtol=0.0, atol=1e-08):
        raise ValueError(f'easy_negative_ratio + hard_negative_ratio + random_negative_ratio must equal 1.0, got {sum(ratios):.12g}')
    return ratios

def add_negative_sampling_ratio_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--easy_negative_ratio', type=float, default=DEFAULT_EASY_NEGATIVE_RATIO)
    parser.add_argument('--hard_negative_ratio', type=float, default=DEFAULT_HARD_NEGATIVE_RATIO)
    parser.add_argument('--random_negative_ratio', type=float, default=DEFAULT_RANDOM_NEGATIVE_RATIO)
    return parser

def _load_relation_interactions(path: str, row_names: List[str], col_names: List[str], row_candidates, col_candidates) -> np.ndarray:
    header = pd.read_csv(path, nrows=0)
    columns = {str(c).strip().lower(): c for c in header.columns}
    available_rows = [columns[c.lower()] for c in row_candidates if c.lower() in columns]
    available_cols = [columns[c.lower()] for c in col_candidates if c.lower() in columns]
    if available_rows and available_cols:
        usecols = list(dict.fromkeys(available_rows + available_cols))
        values = pd.read_csv(path, usecols=usecols)
        row_known = {normalize_identifier(value) for value in row_names}
        col_known = {normalize_identifier(value) for value in col_names}
        row_col = max(available_rows, key=lambda col: int(values[col].map(normalize_identifier).isin(row_known).sum()))
        col_col = max(available_cols, key=lambda col: int(values[col].map(normalize_identifier).isin(col_known).sum()))
        return load_pair_interactions(path, row_names, col_names, row_col, col_col)
    return load_matrix_interactions(path, row_names, col_names)

def _load_rna_drug_interactions(
    path: str,
    rna_names: List[str],
    drug_names: List[str],
    strict_id_alignment: bool=True,
    max_unmatched_frac: float=0.0
) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f'RNA-Drug file not found: {path}')
    df = pd.read_csv(path)
    lower_to_col = {str(c).strip().lower(): c for c in df.columns}

    def _select_best_column(candidates, known_ids):
        available = [lower_to_col[c.lower()] for c in candidates if c.lower() in lower_to_col]
        if not available:
            return None
        known = set(known_ids)
        return max(available, key=lambda col: int(df[col].map(normalize_identifier).isin(known).sum()))
    normalized_rna_names = [normalize_identifier(n) for n in rna_names]
    normalized_drug_names = [normalize_identifier(n) for n in drug_names]
    rna_col = _select_best_column(('miRNA_name', 'RNA_ID', 'miRNA', 'miRNA_ID', 'rna_id', 'RNA', 'rna'), normalized_rna_names)
    drug_col = _select_best_column(('drug_name', 'CID', 'Drug_ID', 'drug_id', 'PubChem_CID', 'Drug', 'drug'), normalized_drug_names)
    if rna_col is not None and drug_col is not None:
        rna_ids = normalized_rna_names
        drug_ids = normalized_drug_names
        duplicated_rna_ids = pd.Series(rna_ids).duplicated()
        duplicated_drug_ids = pd.Series(drug_ids).duplicated()
        if bool(duplicated_rna_ids.any()):
            examples = pd.Series(rna_ids)[duplicated_rna_ids].drop_duplicates().head(10).tolist()
            raise ValueError(f'RNA 节点名存在重复 ID，无法唯一对齐。示例: {examples}')
        if bool(duplicated_drug_ids.any()):
            examples = pd.Series(drug_ids)[duplicated_drug_ids].drop_duplicates().head(10).tolist()
            raise ValueError(f'Drug 节点名存在重复 ID，无法唯一对齐。示例: {examples}')
        row2idx = {n: i for (i, n) in enumerate(rna_ids)}
        col2idx = {n: i for (i, n) in enumerate(drug_ids)}
        mat = np.zeros((len(rna_ids), len(drug_ids)), dtype=np.float32)
        unmatched = []
        duplicate_pairs = 0
        matched_pairs = 0
        seen = set()
        for (row_i, (raw_r, raw_c)) in enumerate(zip(df[rna_col], df[drug_col])):
            r = normalize_identifier(raw_r)
            c = normalize_identifier(raw_c)
            if r not in row2idx or c not in col2idx:
                unmatched.append({
                    'row_number': int(row_i),
                    'rna_id': r,
                    'drug_id': c,
                    'rna_known': bool(r in row2idx),
                    'drug_known': bool(c in col2idx)
                })
                continue
            pair = (row2idx[r], col2idx[c])
            if pair in seen:
                duplicate_pairs += 1
                continue
            seen.add(pair)
            mat[pair[0], pair[1]] = 1.0
            matched_pairs += 1
        unmatched_frac = len(unmatched) / max(1, len(df))
        if unmatched and (strict_id_alignment or unmatched_frac > max_unmatched_frac):
            raise ValueError(f'RNA-Drug pair 表 ID 对齐失败：{path}\n无法匹配 {len(unmatched)} / {len(df)} 条 ({unmatched_frac:.2%})。\n示例: {unmatched[:10]}\n检测到列: RNA={rna_col!r}, Drug={drug_col!r}。\n默认禁止静默跳过正样本；否则这些真实正边可能在后续负采样中被当作负样本。')
        if matched_pairs == 0 and len(df) > 0:
            raise ValueError(f'RNA-Drug pair 表没有任何样本成功匹配：{path}。请检查 RNA_ID/CID 是否与相似性矩阵行列名完全一致。')
        if duplicate_pairs > 0:
            print(f'[RNA-Drug] 检测到重复 pair {duplicate_pairs} 条，已去重。', flush=True)
        return mat
    try:
        return load_matrix_interactions(
            path,
            rna_names,
            drug_names,
            allow_missing=not strict_id_alignment,
            max_missing_frac=max_unmatched_frac
        )
    except Exception as exc:
        raise ValueError(f'无法解析 RNA-Drug 文件: {path}\n请使用 pair 表列 RNA_ID/CID，或使用带行名/列名的 0/1 矩阵。原始错误: {exc}')

def save_hetero_graph(graph: HeteroGraphData, path: str, include_full_matrix: bool=False) -> None:
    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    os.makedirs(directory, exist_ok=True)
    state = dict(graph.__dict__)
    if not include_full_matrix:
        state['rna_drug_matrix_full'] = None
    payload = {'schema_name': 'graph_split', 'state': state}
    _torch_save_safe(payload, absolute_path)

def load_hetero_graph(path: str) -> HeteroGraphData:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('state'), dict):
        raise ValueError('图文件缺少有效的 state 载荷')
    schema_name = payload.get('schema_name')
    if schema_name not in (None, 'graph_split'):
        raise ValueError(f'不支持的图文件格式：{schema_name}')
    graph = HeteroGraphData()
    graph.__dict__.update(payload['state'])
    return graph

def _sync_type_partitions_from_global(graph: HeteroGraphData) -> None:
    if graph.global_partition is None:
        raise RuntimeError('global_partition 尚未构建')
    graph.partitions = {}
    for node_type in graph.node_features:
        graph.partitions[node_type] = np.asarray(
            [int(graph.global_partition[graph.get_global_id(node_type, idx)]) for idx in range(graph.num_nodes(node_type))],
            dtype=np.int64
        )
    graph.build_module_info()

def build_entropy_guided_similarity_prior(graph: HeteroGraphData, verbose: bool=False) -> Dict[str, Dict]:
    if graph.global_partition is None:
        raise RuntimeError('global structural-entropy partition has not been built')
    _sync_type_partitions_from_global(graph)
    (edge_index, _) = graph.inter_edges['rna', 'drug']
    rna_labels = np.asarray(graph.partitions['rna'], dtype=np.int64)
    drug_labels = np.asarray(graph.partitions['drug'], dtype=np.int64)
    if edge_index.numel():
        src = edge_index[0].cpu().numpy().astype(np.int64)
        dst = edge_index[1].cpu().numpy().astype(np.int64)
    else:
        src = dst = np.empty(0, dtype=np.int64)
    num_modules = int(max(rna_labels.max(), drug_labels.max()) + 1)
    rna_sizes = np.bincount(rna_labels, minlength=num_modules).astype(np.float64)
    drug_sizes = np.bincount(drug_labels, minlength=num_modules).astype(np.float64)
    pair_count = np.zeros((num_modules, num_modules), dtype=np.float64)
    if src.size:
        np.add.at(pair_count, (rna_labels[src], drug_labels[dst]), 1.0)
    possible = rna_sizes[:, None] * drug_sizes[None, :]
    density = pair_count / np.maximum(np.sqrt(possible), 1.0)
    density_score = density[rna_labels[:, None], drug_labels[None, :]]
    endpoint_alignment = (rna_labels[:, None] == drug_labels[None, :]).astype(np.float64)
    density_score = density_score + 0.75 * endpoint_alignment
    transition: Dict[str, np.ndarray] = {}
    for node_type in ('rna', 'drug'):
        (sim_index, sim_weight) = graph.sim_edges[node_type]
        size = graph.num_nodes(node_type)
        adjacency = np.zeros((size, size), dtype=np.float64)
        if sim_index.numel():
            (rows, cols) = sim_index.cpu().numpy()
            adjacency[rows, cols] = sim_weight.cpu().numpy()
        labels = np.asarray(graph.partitions[node_type], dtype=np.int64)
        adjacency *= np.where(labels[:, None] == labels[None, :], 1.8, 0.35)
        transition[node_type] = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-12)
        (edge_index, edge_weight) = graph.sim_edges[node_type]
        if edge_index.numel():
            (rows, cols) = edge_index.cpu().numpy()
            raw_weight = edge_weight.cpu().numpy().astype(np.float32)
            gated = raw_weight * np.where(labels[rows] == labels[cols], 1.8, 0.35).astype(np.float32)
            gated *= raw_weight.mean() / max(float(gated.mean()), 1e-08)
            graph.sim_edges[node_type] = (edge_index, torch.from_numpy(gated.astype(np.float32)))
    association = graph.rna_drug_matrix.astype(np.float64, copy=False)
    channels = (transition['rna'] @ association, association @ transition['drug'].T, density_score)

    def standardize(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / max(float(values.std()), 1e-08)
    graph.se_pair_prior = torch.from_numpy(np.clip(np.stack(tuple((standardize(channel) for channel in channels)), axis=-1), -4.0, 4.0).astype(np.float32))
    retained = int((rna_labels[src] == drug_labels[dst]).sum()) if src.size else 0
    graph.se_filter_info = {'_global': {
        'strategy': 'entropy_gated_similarity_diffusion_pair_prior',
        'original_similarity_topology_unchanged': True,
        'similarity_edge_weight_gate': {'intra_module': 1.8, 'cross_module': 0.35},
        'num_entropy_supported_edges': retained,
        'prior_channels': ['rna_diffusion', 'drug_diffusion', 'module_density_alignment']
    }}
    _progress(
        verbose,
        f'      [SE similarity] original KNN edges unchanged; pair prior shape={tuple(graph.se_pair_prior.shape)}'
    )
    return graph.se_filter_info

def build_hetero_graph(
    data_dir: str='.',
    files: Optional[Dict[str, str]]=None,
    knn: int=15,
    sim_threshold: float=0.0,
    partition_passes: int=8,
    random_seed: Optional[int]=None,
    remove_rna_drug_edges: Optional[Set[Tuple[int, int]]]=None,
    build_partition: bool=True,
    save_path: Optional[str]=None,
    verbose: bool=True
) -> HeteroGraphData:
    build_start = time.perf_counter()
    files = {**DEFAULT_FILES, **(files or {})}

    def _p(key: str) -> str:
        return os.path.join(data_dir, files[key])
    _progress(verbose, '      [数据 1/6] 读取 RNA 相似性矩阵')
    (rna_sim, rna_names) = load_similarity_matrix(_p('rna_sim'))
    _progress(verbose, f'          RNA：节点数={len(rna_names)}，形状={rna_sim.shape}')
    _progress(verbose, '      [数据 2/6] 读取 Drug 相似性矩阵')
    (drug_sim, drug_names) = load_similarity_matrix(_p('drug_sim'))
    _progress(verbose, f'          Drug：节点数={len(drug_names)}，形状={drug_sim.shape}')
    _progress(verbose, '      [数据 3/6] 读取 Disease 相似性矩阵')
    (disease_sim, disease_names) = load_similarity_matrix(_p('disease_sim'))
    _progress(verbose, f'          Disease：节点数={len(disease_names)}，形状={disease_sim.shape}')
    graph = HeteroGraphData()
    graph.names = {'rna': rna_names, 'drug': drug_names, 'disease': disease_names}
    graph.raw_similarity = {'rna': rna_sim.copy(), 'drug': drug_sim.copy(), 'disease': disease_sim.copy()}
    _progress(verbose, '      [特征] 读取并拼接 feature 目录中的节点特征')
    (graph.node_features, graph.node_feature_info) = load_node_features(data_dir, graph.names)
    for (node_type, info) in graph.node_feature_info.items():
        _progress(verbose, f"          {node_type.upper()}：特征维度={info['total_dim']}，模态={list(info['modalities'])}")
    _progress(verbose, f'      [建图] 构建 KNN 相似性边：knn={knn}, threshold={sim_threshold}')
    for (node_type, sim) in graph.raw_similarity.items():
        adjacency = build_similarity_adjacency(sim, knn=knn, threshold=sim_threshold)
        graph.sim_edges[node_type] = adjacency_to_edge_index(adjacency)
        graph.sim_degree[node_type] = adjacency.sum(axis=1)
        unique_edges = int(np.count_nonzero(np.triu(adjacency, k=1)))
        _progress(verbose, f'          {node_type.upper()} 相似性边数={unique_edges}')
    _progress(verbose, '      [数据 4/6] 读取 RNA-Drug 正样本关系')
    rna_drug_full = _load_rna_drug_interactions(_p('rna_drug'), rna_names, drug_names)
    _progress(verbose, f'          RNA-Drug 正样本数={int(rna_drug_full.sum())}')
    _progress(verbose, '      [数据 5/6] 读取 RNA-Disease 关系矩阵')
    rna_disease = _load_relation_interactions(
        _p('rna_disease'),
        rna_names,
        disease_names,
        ('miRNA', 'miRNA_ID', 'rna_id', 'RNA_ID', 'rna'),
        ('disease_id', 'disease', 'Disease', 'DOID')
    )
    _progress(verbose, f'          RNA-Disease 边数={int((rna_disease > 0).sum())}')
    _progress(verbose, '      [数据 6/6] 读取 Drug-Disease 关系矩阵')
    drug_disease = _load_relation_interactions(
        _p('drug_disease'),
        drug_names,
        disease_names,
        ('drug', 'drug_id', 'Drug_ID', 'CID'),
        ('disease_id', 'disease', 'Disease', 'DOID')
    )
    graph.rna_drug_matrix_full = (rna_drug_full > 0).astype(np.float32)
    rna_drug_train = graph.rna_drug_matrix_full.copy()
    for edge in remove_rna_drug_edges or set():
        (rna_idx, drug_idx) = (int(edge[0]), int(edge[1]))
        if 0 <= rna_idx < rna_drug_train.shape[0] and 0 <= drug_idx < rna_drug_train.shape[1]:
            rna_drug_train[rna_idx, drug_idx] = 0.0
    graph.rna_drug_matrix = rna_drug_train

    def _add_bidir(mat: np.ndarray, src: str, dst: str) -> None:
        binary_mat = (mat > 0).astype(np.float32)
        (edge_index, edge_weight) = _inter_edge_index(binary_mat)
        graph.inter_edges[src, dst] = (edge_index, edge_weight)
        graph.inter_edges[dst, src] = (edge_index.flip(0) if edge_index.numel() else edge_index, edge_weight)
    _add_bidir(rna_drug_train, 'rna', 'drug')
    _add_bidir(rna_disease, 'rna', 'disease')
    _add_bidir(drug_disease, 'drug', 'disease')
    graph.build_global_node_mapping()
    if build_partition:
        build_global_structural_context(
            graph,
            partition_passes=partition_passes,
            random_seed=random_seed,
            repartition=True,
            verbose=verbose
        )
        graph.initial_global_adjacency = graph.global_adjacency.copy()
        graph.initial_structure_info = dict(graph.global_structure_info)
        _sync_type_partitions_from_global(graph)
    if save_path is not None:
        save_hetero_graph(graph, save_path, include_full_matrix=False)
    if verbose:
        print(
            f'      [初始图完成] 训练 RNA-Drug 边={int(rna_drug_train.sum())}，完整 RNA-Drug 边={int(rna_drug_full.sum())}，RNA-Disease 边={int(rna_disease.sum())}，Drug-Disease 边={int(drug_disease.sum())}；耗时={_elapsed(build_start)}',
            flush=True
        )
    return graph

def build_train_graph_for_fold(data_dir: str, val_pos_edges, test_pos_edges=None, **kwargs) -> HeteroGraphData:

    def _to_tuple(edge) -> Tuple[int, int]:
        if torch.is_tensor(edge):
            edge = edge.detach().cpu().tolist()
        return (int(edge[0]), int(edge[1]))
    remove_edges = {_to_tuple(edge) for edge in val_pos_edges}
    if test_pos_edges is not None:
        remove_edges.update((_to_tuple(edge) for edge in test_pos_edges))
    kwargs.setdefault('build_partition', False)
    return build_hetero_graph(data_dir=data_dir, remove_rna_drug_edges=remove_edges, **kwargs)

def _weighted_disease_dissimilarity(
    rna_idx: int,
    drug_idx: int,
    rna_disease_adj: np.ndarray,
    drug_disease_adj: np.ndarray,
    disease_weight: np.ndarray
) -> float:
    rna_vector = rna_disease_adj[rna_idx] * disease_weight
    drug_vector = drug_disease_adj[drug_idx] * disease_weight
    intersection = np.minimum(rna_vector, drug_vector).sum()
    union = np.maximum(rna_vector, drug_vector).sum()
    if union <= 0:
        return DEFAULT_BOTH_MISSING_CONTEXT_DISSIMILARITY
    return float(1.0 - intersection / union)

def _normalized_module_distance(graph: HeteroGraphData, rna_idx: int, drug_idx: int) -> float:
    labels = graph.global_partition
    rna_module = int(labels[graph.get_global_id('rna', rna_idx)])
    drug_module = int(labels[graph.get_global_id('drug', drug_idx)])
    rna_pos = graph.global_module_to_pos[rna_module]
    drug_pos = graph.global_module_to_pos[drug_module]
    distance = float(graph.global_module_distance[rna_pos, drug_pos])
    finite_distance = graph.global_module_distance[np.isfinite(graph.global_module_distance)]
    max_finite = float(finite_distance.max()) if finite_distance.size > 0 and finite_distance.max() > 0 else 1.0
    if not np.isfinite(distance):
        return 1.0
    return float(np.clip(distance / max_finite, 0.0, 1.0))

def generate_negative_samples_entropy_guided(
    graph: HeteroGraphData,
    num_samples: int,
    forbidden_set: Set[Tuple[int, int]],
    rng: np.random.RandomState,
    candidate_multiplier: int=20,
    entropy_eval_multiplier: int=5,
    added_edge_weight: float=1.0,
    easy_negative_ratio: float=DEFAULT_EASY_NEGATIVE_RATIO,
    hard_negative_ratio: float=DEFAULT_HARD_NEGATIVE_RATIO,
    random_negative_ratio: float=DEFAULT_RANDOM_NEGATIVE_RATIO,
    entropy_weight: float=0.5,
    module_weight: float=0.3,
    context_weight: float=0.2,
    verbose: bool=False,
    progress_label: str='negative'
) -> Tuple[List[Tuple[int, int]], List[float], List[Dict]]:
    if graph.global_adjacency is None:
        raise RuntimeError('请先构建全局结构上下文')
    if num_samples <= 0:
        return ([], [], [])
    ratios = validate_negative_sample_ratios(easy_negative_ratio, hard_negative_ratio, random_negative_ratio)
    total_weight = entropy_weight + module_weight + context_weight
    if not np.isclose(total_weight, 1.0):
        raise ValueError('三个评分权重之和必须为 1')
    raw_counts = np.asarray(ratios, dtype=np.float64) * num_samples
    counts = np.floor(raw_counts).astype(int)
    remainder = num_samples - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw_counts - counts), kind='stable')
        counts[order[:remainder]] += 1
    (target_easy, target_hard, target_random) = map(int, counts)
    num_rna = graph.num_nodes('rna')
    num_drug = graph.num_nodes('drug')
    num_disease = graph.num_nodes('disease')
    rna_disease_adj = _edge_index_to_binary_adj(graph.inter_edges['rna', 'disease'][0], num_rna, num_disease)
    drug_disease_adj = _edge_index_to_binary_adj(graph.inter_edges['drug', 'disease'][0], num_drug, num_disease)
    disease_frequency = rna_disease_adj.sum(axis=0) + drug_disease_adj.sum(axis=0)
    disease_weight = 1.0 / np.maximum(np.log1p(disease_frequency + 1.0), 1e-12)
    sampling_start = time.perf_counter()
    pool_target = max(num_samples, num_samples * candidate_multiplier)
    initial_exact_target = max(num_samples, num_samples * entropy_eval_multiplier)
    _progress(
        verbose,
        f'      [{progress_label}] 目标数={num_samples}，简单={target_easy}，困难={target_hard}，随机={target_random}，候选池目标={pool_target}，初始精确 H2 评估上限={initial_exact_target}'
    )
    candidate_pool = []
    seen = set(forbidden_set)
    attempts = 0
    max_attempts = max(pool_target * 20, 1000)
    while len(candidate_pool) < pool_target and attempts < max_attempts:
        pair = (int(rng.randint(num_rna)), int(rng.randint(num_drug)))
        if pair not in seen:
            seen.add(pair)
            candidate_pool.append(pair)
        attempts += 1
    _progress(verbose, f'      [{progress_label}] 候选池构建完成：{len(candidate_pool)}')
    if len(candidate_pool) < num_samples:
        raise RuntimeError(f'候选负边不足：需要 {num_samples}，实际只有 {len(candidate_pool)}')
    random_records = []
    if target_random:
        random_indices = set(map(int, rng.choice(len(candidate_pool), size=target_random, replace=False)))
        structural_pool = []
        for (index, pair) in enumerate(candidate_pool):
            if index in random_indices:
                random_records.append({
                    'pair': pair,
                    'score': 0.0,
                    'sample_type': 'random',
                    'entropy_score': None,
                    'delta_h2_add': None,
                    'module_distance': None,
                    'context_dissimilarity': None
                })
            else:
                structural_pool.append(pair)
    else:
        structural_pool = candidate_pool
    structural_target = target_easy + target_hard
    if len(structural_pool) < structural_target:
        raise RuntimeError(f'结构熵负样本候选不足：需要 {structural_target}，实际 {len(structural_pool)}')
    cheap_records = []
    for (rna_idx, drug_idx) in structural_pool:
        distance_score = _normalized_module_distance(graph, rna_idx, drug_idx)
        context_score = _weighted_disease_dissimilarity(rna_idx, drug_idx, rna_disease_adj, drug_disease_adj, disease_weight)
        cheap_score = module_weight * distance_score + context_weight * context_score
        cheap_records.append({
            'pair': (rna_idx, drug_idx),
            'module_distance': distance_score,
            'context_dissimilarity': context_score,
            'cheap_score': cheap_score
        })
    cheap_records.sort(key=lambda record: record['cheap_score'], reverse=True)
    exact_h2_before = two_dim_structural_entropy(graph.global_adjacency, graph.global_partition)

    def _evaluate_h2(records: List[Dict], offset: int) -> None:
        if not records:
            return
        exact_progress_step = max(1, len(records) // 10)
        for (record_index, record) in enumerate(records, start=1):
            global_index = offset + record_index
            if verbose and (record_index == 1 or record_index % exact_progress_step == 0 or record_index == len(records)):
                percent = int(round(record_index * 100 / max(1, len(records))))
                _progress(
                    verbose,
                    f'          [{progress_label} H2] {global_index}/{len(cheap_records)} 本批 {record_index}/{len(records)} ({percent}%)'
                )
            (rna_idx, drug_idx) = record['pair']
            rna_global = graph.get_global_id('rna', rna_idx)
            drug_global = graph.get_global_id('drug', drug_idx)
            delta_h2 = compute_delta_h2_add(
                graph.global_adjacency,
                graph.global_partition,
                rna_global,
                drug_global,
                weight=added_edge_weight,
                h2_before=exact_h2_before
            )
            record['delta_h2_add'] = float(delta_h2)

    def _refresh_scores(records: List[Dict]) -> None:
        max_positive_delta = max((max(record['delta_h2_add'], 0.0) for record in records), default=0.0)
        for record in records:
            if max_positive_delta > 0:
                entropy_score = max(record['delta_h2_add'], 0.0) / max_positive_delta
            else:
                entropy_score = 0.0
            record['entropy_score'] = float(entropy_score)
            record['score'] = float(entropy_weight * entropy_score + module_weight * record['module_distance'] + context_weight * record['context_dissimilarity'])
    exact_count = min(len(cheap_records), initial_exact_target) if structural_target else 0
    if exact_count < structural_target:
        raise RuntimeError(f'精确 H2 评估预算不足：需要至少 {structural_target}，实际只有 {exact_count}。请增大 entropy_eval_multiplier。')
    extra_exact_budget = exact_count - structural_target
    if structural_target:
        easy_extra = int(np.floor(extra_exact_budget * target_easy / structural_target))
    else:
        easy_extra = 0
    easy_exact_count = target_easy + easy_extra
    hard_exact_count = exact_count - easy_exact_count
    easy_shortlist = cheap_records[:easy_exact_count]
    hard_shortlist = cheap_records[len(cheap_records) - hard_exact_count:] if hard_exact_count else []
    for record in easy_shortlist:
        record['prefilter_band'] = 'high'
    for record in hard_shortlist:
        record['prefilter_band'] = 'low'
    shortlist = easy_shortlist + hard_shortlist
    _progress(
        verbose,
        f'      [{progress_label}] 低成本双端预筛完成，精确 H2 评估数={len(shortlist)} (easy 高端={len(easy_shortlist)}，hard 低端={len(hard_shortlist)})'
    )
    _evaluate_h2(shortlist, offset=0)
    _refresh_scores(shortlist)
    easy_ranked = sorted(easy_shortlist, key=lambda record: record['score'], reverse=True)
    hard_ranked = sorted(hard_shortlist, key=lambda record: record['score'])
    selected_easy = easy_ranked[:target_easy]
    selected_hard = hard_ranked[:target_hard]
    for record in selected_easy:
        record['sample_type'] = 'easy'
    for record in selected_hard:
        record['sample_type'] = 'hard'
    selected = selected_easy + selected_hard + random_records
    selected.sort(key=lambda record: record['score'], reverse=True)
    negative_pairs = [record['pair'] for record in selected]
    negative_scores = [record['score'] for record in selected]
    _progress(
        verbose,
        f'      [{progress_label}] 选中={len(selected)}，简单={len(selected_easy)}，困难={len(selected_hard)}，随机={len(random_records)}，目标比例={easy_negative_ratio:.2%}/{hard_negative_ratio:.2%}/{random_negative_ratio:.2%}，耗时={_elapsed(sampling_start)}'
    )
    return (negative_pairs, negative_scores, selected)

def _assemble_labeled_split(positive: np.ndarray, negative: np.ndarray, negative_scores: List[float], shuffle: bool, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = np.vstack([positive, negative])
    labels = np.concatenate([np.ones(len(positive), dtype=np.float32), np.zeros(len(negative), dtype=np.float32)])
    scores = np.concatenate([np.full(len(positive), -1.0, dtype=np.float32), np.asarray(negative_scores, dtype=np.float32)])
    if shuffle:
        order = np.random.RandomState(seed).permutation(len(pairs))
        (pairs, labels, scores) = (pairs[order], labels[order], scores[order])
    return (torch.from_numpy(pairs).long(), torch.from_numpy(labels).float(), torch.from_numpy(scores).float())

def prepare_and_save_five_fold_graphs(data_dir: str='.', save_dir: str='processed_folds', files: Optional[Dict[str, str]]=None, **kwargs) -> List[Dict[str, int]]:
    kwargs = dict(kwargs)
    verbose = bool(kwargs.pop('verbose', True))
    n_splits = int(kwargs.pop('n_splits', CONFIG.graph.folds))
    shuffle = bool(kwargs.pop('shuffle', True))
    random_state = int(kwargs.get('random_state', CONFIG.graph.seed))
    train_ratio = float(kwargs.get('train_ratio', CONFIG.graph.train_ratio))
    val_ratio = float(kwargs.get('val_ratio', CONFIG.graph.val_ratio))
    test_ratio = float(kwargs.get('test_ratio', CONFIG.graph.test_ratio))
    if n_splits != 5:
        raise ValueError(f'SEGMDAL requires exactly five folds, got {n_splits}')
    if not np.allclose([train_ratio, val_ratio, test_ratio], [0.64, 0.16, 0.2]):
        raise ValueError('outer five-fold protocol requires train/val/test ratios 0.64/0.16/0.20')
    resolved_files = {**DEFAULT_FILES, **(files or {})}

    def _p(key: str) -> str:
        return os.path.join(data_dir, resolved_files[key])
    _progress(verbose, '[预检查] 读取节点名称和完整 RNA-Drug 正样本')
    (_, rna_names) = load_similarity_matrix(_p('rna_sim'))
    (_, drug_names) = load_similarity_matrix(_p('drug_sim'))
    full_matrix = _load_rna_drug_interactions(_p('rna_drug'), rna_names, drug_names)
    positive_pairs = np.argwhere(full_matrix > 0).astype(np.int64)
    if len(positive_pairs) < n_splits:
        raise ValueError(f'五折划分至少需要 {n_splits} 个正样本，当前为 {len(positive_pairs)}')
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    _progress(verbose, f'标准五折：全部正样本={len(positive_pairs)}，每折外层训练池约 80%，测试约 20%；外层训练池再按 train=80% / val=20% 划分')
    summaries: List[Dict[str, int]] = []
    prepare_start = time.perf_counter()
    inner_train_fraction = train_ratio / (train_ratio + val_ratio)
    for (fold_number, (outer_train_indices, test_indices)) in enumerate(splitter.split(positive_pairs), start=1):
        outer_train_positive = positive_pairs[outer_train_indices]
        test_positive = positive_pairs[test_indices]
        inner_order = np.random.RandomState(random_state + 1000 + fold_number).permutation(len(outer_train_positive))
        inner_train_count = int(round(len(outer_train_positive) * inner_train_fraction))
        inner_train_count = min(max(1, inner_train_count), len(outer_train_positive) - 1)
        train_positive = outer_train_positive[inner_order[:inner_train_count]]
        val_positive = outer_train_positive[inner_order[inner_train_count:]]
        _progress(
            verbose,
            f'[Fold {fold_number}/{n_splits}] 正样本 train={len(train_positive)}, val={len(val_positive)}, test={len(test_positive)}'
        )
        summaries.extend(build_fold(
            data_dir=data_dir,
            save_dir=save_dir,
            files=files,
            verbose=verbose,
            _positive_splits=(train_positive, val_positive, test_positive),
            _fold_number=fold_number,
            _n_splits=n_splits,
            **kwargs
        ))
    _progress(verbose, f'[完成] 已生成 {len(summaries)} 折标准交叉验证缓存，耗时 {_elapsed(prepare_start)}')
    return summaries

def build_fold(
    data_dir: str='.',
    save_dir: str='processed_simple_split',
    files: Optional[Dict[str, str]]=None,
    train_ratio: float=CONFIG.graph.train_ratio,
    val_ratio: float=CONFIG.graph.val_ratio,
    test_ratio: float=CONFIG.graph.test_ratio,
    random_state: int=CONFIG.graph.seed,
    neg_ratio: float=CONFIG.graph.negative_ratio,
    knn: int=CONFIG.graph.knn,
    sim_threshold: float=CONFIG.graph.similarity_threshold,
    partition_passes: int=CONFIG.graph.partition_passes,
    candidate_multiplier: int=CONFIG.graph.candidate_multiplier,
    entropy_eval_multiplier: int=CONFIG.graph.entropy_eval_multiplier,
    easy_negative_ratio: float=DEFAULT_EASY_NEGATIVE_RATIO,
    hard_negative_ratio: float=DEFAULT_HARD_NEGATIVE_RATIO,
    random_negative_ratio: float=DEFAULT_RANDOM_NEGATIVE_RATIO,
    entropy_weight: float=CONFIG.graph.entropy_weight,
    module_weight: float=CONFIG.graph.module_weight,
    context_weight: float=CONFIG.graph.context_weight,
    verbose: bool=True,
    _positive_splits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]=None,
    _fold_number: int=1,
    _n_splits: int=1
) -> List[Dict[str, int]]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0.0) or not np.isclose(float(ratios.sum()), 1.0):
        raise ValueError('train_ratio、val_ratio、test_ratio 必须均大于 0 且总和为 1')
    if neg_ratio <= 0:
        raise ValueError('neg_ratio 必须 > 0')
    validate_negative_sample_ratios(easy_negative_ratio, hard_negative_ratio, random_negative_ratio)
    resolved_files = {**DEFAULT_FILES, **(files or {})}

    def _p(key: str) -> str:
        return os.path.join(data_dir, resolved_files[key])
    _progress(verbose, '[预检查] 读取节点名称和完整 RNA-Drug 正样本')
    (_, rna_names) = load_similarity_matrix(_p('rna_sim'))
    (_, drug_names) = load_similarity_matrix(_p('drug_sim'))
    full_matrix = _load_rna_drug_interactions(_p('rna_drug'), rna_names, drug_names)
    positive_pairs = np.argwhere(full_matrix > 0).astype(np.int64)
    if _positive_splits is None or _n_splits != 5:
        raise ValueError('build_fold is internal to the fixed five-fold protocol')
    (train_positive, val_positive, test_positive) = [np.asarray(p, dtype=np.int64) for p in _positive_splits]
    if len(positive_pairs) < 3:
        raise ValueError('正样本数至少需要 3 条才能构造 train/val/test')
    if len(train_positive) == 0 or len(val_positive) == 0 or len(test_positive) == 0:
        raise ValueError('train/val/test 不能出现空集')
    fold_seed = int(random_state) + int(_fold_number) - 1
    _progress(
        verbose,
        f'第 {_fold_number}/{_n_splits} 折 train/val/test：train={len(train_positive)}, val={len(val_positive)}, test={len(test_positive)}'
    )
    positive_set = {tuple(map(int, pair)) for pair in positive_pairs}
    heldout_positive = np.vstack([val_positive, test_positive])
    fold_graph = build_train_graph_for_fold(
        data_dir=data_dir,
        files=files,
        val_pos_edges=heldout_positive,
        knn=knn,
        sim_threshold=sim_threshold,
        random_seed=fold_seed,
        build_partition=False,
        verbose=verbose
    )
    actual_train_set = {tuple(map(int, pair)) for pair in np.argwhere(fold_graph.rna_drug_matrix > 0)}
    expected_train_set = {tuple(map(int, pair)) for pair in train_positive}
    if actual_train_set != expected_train_set:
        raise ValueError('训练图 RNA-Drug 正边与 train_positive 不一致')
    build_global_structural_context(
        fold_graph,
        partition_passes=partition_passes,
        random_seed=fold_seed,
        repartition=True,
        verbose=verbose
    )
    fold_graph.initial_global_adjacency = fold_graph.global_adjacency.copy()
    fold_graph.initial_structure_info = dict(fold_graph.global_structure_info)
    _sync_type_partitions_from_global(fold_graph)
    split_specs = [('train', train_positive, 100000), ('val', val_positive, 200000), ('test', test_positive, 300000)]
    sampled: Dict[str, Tuple[np.ndarray, List[float], List[Dict]]] = {}
    used_negative: set[tuple[int, int]] = set()
    for (split_name, positives, seed_offset) in split_specs:
        (neg_list, neg_scores, neg_details) = generate_negative_samples_entropy_guided(
            graph=fold_graph,
            num_samples=max(1, int(round(len(positives) * neg_ratio))),
            forbidden_set=positive_set | used_negative,
            rng=np.random.RandomState(seed_offset + fold_seed),
            candidate_multiplier=candidate_multiplier,
            entropy_eval_multiplier=entropy_eval_multiplier,
            easy_negative_ratio=easy_negative_ratio,
            hard_negative_ratio=hard_negative_ratio,
            random_negative_ratio=random_negative_ratio,
            entropy_weight=entropy_weight,
            module_weight=module_weight,
            context_weight=context_weight,
            verbose=verbose,
            progress_label=f'{split_name} 负样本'
        )
        neg_array = np.asarray(neg_list, dtype=np.int64).reshape(-1, 2)
        used_negative.update((tuple(map(int, x)) for x in neg_array))
        sampled[split_name] = (neg_array, list(neg_scores), list(neg_details))
    build_entropy_guided_similarity_prior(fold_graph, verbose=verbose)
    (train_negative, train_scores, train_details) = sampled['train']
    (val_negative, val_scores, val_details) = sampled['val']
    (test_negative, test_scores, test_details) = sampled['test']
    train_t = _assemble_labeled_split(train_positive, train_negative, train_scores, True, 510000 + fold_seed)
    val_t = _assemble_labeled_split(val_positive, val_negative, val_scores, True, 520000 + fold_seed)
    test_t = _assemble_labeled_split(test_positive, test_negative, test_scores, True, 530000 + fold_seed)
    fold_dir = os.path.join(save_dir, f'fold_{_fold_number}')
    os.makedirs(fold_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    graph_path = os.path.join(fold_dir, 'final_train_graph.pt')
    save_hetero_graph(fold_graph, graph_path, include_full_matrix=False)
    paths['train_graph'] = graph_path
    split_payload = {
        'fold': int(_fold_number),
        'n_splits': int(_n_splits),
        'split_protocol': 'outer_5fold_train_val_test',
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'train_pairs': train_t[0],
        'train_labels': train_t[1],
        'train_sample_scores': train_t[2],
        'val_pairs': val_t[0],
        'val_labels': val_t[1],
        'val_sample_scores': val_t[2],
        'test_pairs': test_t[0],
        'test_labels': test_t[1],
        'test_sample_scores': test_t[2],
        'train_positive_pairs': torch.from_numpy(train_positive.copy()).long(),
        'train_negative_pairs': torch.from_numpy(train_negative.copy()).long(),
        'train_negative_details': train_details,
        'val_positive_pairs': torch.from_numpy(val_positive.copy()).long(),
        'val_negative_pairs': torch.from_numpy(val_negative.copy()).long(),
        'val_negative_details': val_details,
        'test_positive_pairs': torch.from_numpy(test_positive.copy()).long(),
        'test_negative_pairs': torch.from_numpy(test_negative.copy()).long(),
        'test_negative_details': test_details,
        'initial_structure_info': fold_graph.initial_structure_info,
        'final_structure_info': fold_graph.global_structure_info,
        'negative_exclusion_scope': 'all_known_positive_pairs',
        'random_state': fold_seed
    }
    split_path = os.path.join(fold_dir, 'split_data.pt')
    _torch_save_safe(split_payload, split_path)
    paths['split_data'] = split_path
    fold_graph.fold_info = {
        'fold': int(_fold_number),
        'n_splits': int(_n_splits),
        'split_protocol': 'outer_5fold_train_val_test',
        'train_positive_count': int(len(train_positive)),
        'val_positive_count': int(len(val_positive)),
        'test_positive_count': int(len(test_positive)),
        'train_negative_count': int(len(train_negative)),
        'val_negative_count': int(len(val_negative)),
        'test_negative_count': int(len(test_negative)),
        'random_state': fold_seed,
        'saved_paths': paths
    }
    return [{
        'fold': int(_fold_number),
        'n_splits': int(_n_splits),
        'train_samples': int(len(train_t[1])),
        'train_positive': int(train_t[1].sum().item()),
        'val_samples': int(len(val_t[1])),
        'val_positive': int(val_t[1].sum().item()),
        'test_samples': int(len(test_t[1])),
        'test_positive': int(test_t[1].sum().item())
    }]

def _edge_index_to_binary_adj(edge_index: torch.Tensor, num_src: int, num_dst: int) -> np.ndarray:
    adj = np.zeros((num_src, num_dst), dtype=np.float32)
    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        adj[src, dst] = 1.0
    return adj


def load_split(path: Path, graph: Any | None=None) -> dict[str, Any]:
    split = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(split, dict):
        raise TypeError(f'{path} must contain a dictionary')
    if split.get('split_protocol') == 'outer_5fold_train_val_test':
        required = {'train_pairs', 'train_labels', 'val_pairs', 'val_labels', 'test_pairs', 'test_labels'}
        missing = required.difference(split)
        if missing:
            raise KeyError(f'{path} 缺少 train/val/test 划分字段: {sorted(missing)}')
        simple_names = ('train', 'val', 'test')
        seen_pairs: set[tuple[int, int]] = set()
        num_rna = graph.num_nodes('rna') if graph is not None else None
        num_drug = graph.num_nodes('drug') if graph is not None else None
        for name in simple_names:
            pairs = split[f'{name}_pairs']
            labels = split[f'{name}_labels']
            if not torch.is_tensor(pairs) or pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f'{path}: {name}_pairs must be a [N, 2] tensor')
            if not torch.is_tensor(labels) or labels.ndim != 1 or len(labels) != len(pairs):
                raise ValueError(f'{path}: {name}_labels must be a length-N tensor')
            if len(pairs) == 0:
                raise ValueError(f'{path}: {name} split is empty')
            if not torch.isfinite(labels).all():
                raise ValueError(f'{path}: {name}_labels contains NaN or Inf')
            if set(labels.detach().cpu().tolist()) != {0.0, 1.0}:
                raise ValueError(f'{path}: {name}_labels must contain both binary classes')
            integer_pairs = pairs.detach().cpu().long()
            if not torch.equal(integer_pairs.to(pairs.dtype), pairs.detach().cpu()):
                raise ValueError(f'{path}: {name}_pairs contains non-integer indices')
            if (integer_pairs < 0).any():
                raise IndexError(f'{path}: {name}_pairs contains negative indices')
            if num_rna is not None and ((integer_pairs[:, 0] >= num_rna).any() or (integer_pairs[:, 1] >= num_drug).any()):
                raise IndexError(f'{path}: {name}_pairs contains out-of-range node indices')
            local_pairs = [tuple(map(int, row)) for row in integer_pairs.tolist()]
            if len(set(local_pairs)) != len(local_pairs):
                raise ValueError(f'{path}: {name} contains duplicate pairs')
            overlap = seen_pairs.intersection(local_pairs)
            if overlap:
                raise ValueError(f'{path}: pair {next(iter(overlap))} occurs in multiple splits')
            seen_pairs.update(local_pairs)
        return split
    raise ValueError(f'{path} uses an unsupported split protocol; rebuild it with five train/val/test folds')
