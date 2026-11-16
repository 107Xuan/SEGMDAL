"""Complete SE-RAG: structural scoring, diversity selection and MinHash training-path sampling."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import hashlib
import json
import math
import numpy as np
import os
import re
import torch
from dataclasses import dataclass
from collections import defaultdict, Counter
from config import CONFIG
Node = Tuple[str, int]

def make_json_safe(obj: Any):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(make_json_safe(k)): make_json_safe(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    if hasattr(obj, '__fspath__'):
        return os.fspath(obj)
    return str(obj)

@dataclass(frozen=True)
class Edge:
    src_type: str
    src_idx: int
    dst_type: str
    dst_idx: int
    rel: str
    weight: float

@dataclass
class QueryPair:
    rna_idx: int
    drug_idx: int
    label: Optional[int] = None
    sample_score: Optional[float] = None
    source: str = ''

def torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def sanitize_token(text: str) -> str:
    text = str(text).strip()
    text = re.sub('\\s+', '_', text)
    text = re.sub('[^0-9A-Za-z_\\-.:]', '_', text)
    return text

def relation_name(src_type: str, dst_type: str) -> str:
    if src_type == dst_type:
        return 'sim'
    pair = {src_type, dst_type}
    if pair == {'rna', 'drug'}:
        return 'rna_drug'
    if pair == {'rna', 'disease'}:
        return 'rna_disease'
    if pair == {'drug', 'disease'}:
        return 'drug_disease'
    return f'{src_type}_{dst_type}'

def path_hash(nodes: List[Node], rels: List[str]) -> str:
    raw = '|'.join([f'{t}:{i}' for (t, i) in nodes]) + '||' + '|'.join(rels)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def normalize_global_node_id(global_node_id: Dict[Any, Any]) -> Dict[Tuple[str, int], int]:
    out: Dict[Tuple[str, int], int] = {}
    for (k, v) in global_node_id.items():
        if isinstance(k, tuple) and len(k) == 2:
            (ntype, idx) = k
        elif isinstance(k, list) and len(k) == 2:
            (ntype, idx) = k
        elif isinstance(k, str):
            s = k.strip().replace('(', '').replace(')', '').replace("'", '').replace('"', '')
            parts = [p for p in re.split('[,|\\s:]+', s) if p]
            if len(parts) < 2:
                continue
            (ntype, idx) = (parts[0], parts[1])
        else:
            continue
        out[str(ntype), int(idx)] = int(v)
    return out

class GraphIndex:

    def __init__(
        self,
        node_names: Dict[str, List[str]],
        adj: Dict[Node, List[Edge]],
        global_node_id: Dict[Tuple[str, int], int],
        module_partition: np.ndarray,
        disease_freq: Dict[int, int],
        disease_penalty: Dict[int, float],
        high_freq_disease_threshold: float,
        bridge_factor: float=CONFIG.rag.bridge_factor,
        degree_penalty_gamma: float=CONFIG.rag.degree_penalty_gamma,
        module_penalty_floor: float=CONFIG.rag.module_penalty_floor,
        disease_penalty_floor: float=CONFIG.rag.disease_penalty_floor,
        degree_penalty_floor: float=CONFIG.rag.degree_penalty_floor
    ):
        self.node_names = node_names
        self.adj = adj
        self.global_node_id = global_node_id
        self.module_partition = module_partition.astype(np.int64)
        self.disease_freq = disease_freq
        self.disease_penalty = disease_penalty
        self.high_freq_disease_threshold = float(high_freq_disease_threshold)
        self.bridge_factor = float(bridge_factor)
        self.degree_penalty_gamma = float(degree_penalty_gamma)
        self.module_penalty_floor = float(module_penalty_floor)
        self.disease_penalty_floor = float(disease_penalty_floor)
        self.degree_penalty_floor = float(degree_penalty_floor)
        self.node_degree = {node: len(edges) for (node, edges) in adj.items()}

    def has_node(self, node: Node) -> bool:
        return node in self.global_node_id

    def global_id(self, node: Node) -> int:
        return int(self.global_node_id[node])

    def module_id(self, node: Node) -> int:
        gid = self.global_id(node)
        return int(self.module_partition[gid])

    def token(self, node: Node, use_raw_name: bool=True) -> str:
        (ntype, idx) = node
        prefix = {'rna': 'miRNA', 'drug': 'Drug', 'disease': 'Disease'}.get(ntype, ntype)
        if use_raw_name and ntype in self.node_names and (idx < len(self.node_names[ntype])):
            raw = sanitize_token(self.node_names[ntype][idx])
            return f'<{prefix}:{raw}>'
        return f'<{prefix}:{idx}>'

    def neighbors(self, node: Node) -> List[Edge]:
        return self.adj.get(node, [])

def _normalize_edge_dict(obj: Any) -> Tuple[Any, Any]:
    if isinstance(obj, dict):
        return (obj['edge_index'], obj['edge_weight'])
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return (obj[0], obj[1])
    raise TypeError(f'未知边结构: {type(obj)}')

def build_fallback_global_node_id(node_features: Dict[str, Any], node_names: Dict[str, List[str]]) -> Dict[Tuple[str, int], int]:
    global_node_id: Dict[Tuple[str, int], int] = {}
    gid = 0
    for ntype in ['rna', 'drug', 'disease']:
        if ntype in node_features:
            num_nodes = int(node_features[ntype].shape[0])
        elif ntype in node_names:
            num_nodes = len(node_names[ntype])
        else:
            num_nodes = 0
        for idx in range(num_nodes):
            global_node_id[ntype, idx] = gid
            gid += 1
    return global_node_id

def load_graph_index(
    fold_dir: str,
    disease_freq_threshold: float=CONFIG.rag.disease_freq_threshold,
    disease_freq_quantile: float=CONFIG.rag.disease_freq_quantile,
    disease_penalty_alpha: float=CONFIG.rag.disease_penalty_alpha,
    disease_penalty_min: float=CONFIG.rag.disease_penalty_min,
    bridge_factor: float=CONFIG.rag.bridge_factor,
    degree_penalty_gamma: float=CONFIG.rag.degree_penalty_gamma,
    module_penalty_floor: float=CONFIG.rag.module_penalty_floor,
    disease_penalty_floor: float=CONFIG.rag.disease_penalty_floor,
    degree_penalty_floor: float=CONFIG.rag.degree_penalty_floor
) -> GraphIndex:
    final_graph_path = os.path.join(fold_dir, 'final_train_graph.pt')
    if not os.path.exists(final_graph_path):
        raise FileNotFoundError(f'找不到 {final_graph_path}。本脚本需要 final_train_graph.pt 来获取 node names、global_node_id 和图结构。')
    payload = torch_load_cpu(final_graph_path)
    state = payload['state'] if isinstance(payload, dict) and 'state' in payload else payload
    node_names = state.get('names', {})
    node_features = state.get('node_features', {})
    if state.get('global_node_id', None) is not None:
        global_node_id = normalize_global_node_id(state['global_node_id'])
    else:
        global_node_id = build_fallback_global_node_id(node_features, node_names)
    if state.get('global_partition', None) is not None:
        module_partition = to_numpy(state['global_partition'])
    else:
        raise FileNotFoundError(f'{final_graph_path} does not contain global_partition')
    sim_edges = state.get('sim_edges', {})
    inter_edges = state.get('inter_edges', {})
    adj: Dict[Node, List[Edge]] = defaultdict(list)
    for (ntype, edge_obj) in sim_edges.items():
        (edge_index, edge_weight) = _normalize_edge_dict(edge_obj)
        ei = to_numpy(edge_index).astype(np.int64)
        ew = to_numpy(edge_weight).astype(np.float64)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f'{ntype} sim edge_index 应为 [2, E]，当前 {ei.shape}')
        for k in range(ei.shape[1]):
            src = int(ei[0, k])
            dst = int(ei[1, k])
            if src == dst:
                continue
            w = float(ew[k]) if k < len(ew) else 1.0
            if w <= 0:
                continue
            src_node = (str(ntype), src)
            dst_node = (str(ntype), dst)
            if src_node not in global_node_id or dst_node not in global_node_id:
                continue
            adj[src_node].append(Edge(src_type=str(ntype), src_idx=src, dst_type=str(ntype), dst_idx=dst, rel='sim', weight=w))
    for (key, edge_obj) in inter_edges.items():
        if len(key) != 2:
            continue
        (src_type, dst_type) = (str(key[0]), str(key[1]))
        (edge_index, edge_weight) = _normalize_edge_dict(edge_obj)
        ei = to_numpy(edge_index).astype(np.int64)
        ew = to_numpy(edge_weight).astype(np.float64)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f'{key} edge_index 应为 [2, E]，当前 {ei.shape}')
        rel = relation_name(src_type, dst_type)
        for k in range(ei.shape[1]):
            src = int(ei[0, k])
            dst = int(ei[1, k])
            w = float(ew[k]) if k < len(ew) else 1.0
            if w <= 0:
                continue
            src_node = (src_type, src)
            dst_node = (dst_type, dst)
            if src_node not in global_node_id or dst_node not in global_node_id:
                continue
            adj[src_node].append(Edge(src_type=src_type, src_idx=src, dst_type=dst_type, dst_idx=dst, rel=rel, weight=w))
    disease_assoc = set()
    for (_, edges) in adj.items():
        for e in edges:
            if e.rel not in {'rna_disease', 'drug_disease'}:
                continue
            if e.dst_type == 'disease' and e.src_type in {'rna', 'drug'}:
                disease_assoc.add((e.src_type, int(e.src_idx), int(e.dst_idx)))
            elif e.src_type == 'disease' and e.dst_type in {'rna', 'drug'}:
                disease_assoc.add((e.dst_type, int(e.dst_idx), int(e.src_idx)))
    disease_freq_counter: Counter = Counter()
    for (_, _, dis_idx) in disease_assoc:
        disease_freq_counter[int(dis_idx)] += 1
    disease_freq = dict(disease_freq_counter)
    if len(disease_freq) > 0:
        freqs = np.asarray(list(disease_freq.values()), dtype=np.float64)
        q = min(max(float(disease_freq_quantile), 0.0), 1.0)
        q_threshold = float(np.quantile(freqs, q))
        if disease_freq_threshold and disease_freq_threshold > 0:
            high_freq_threshold = max(float(disease_freq_threshold), q_threshold)
        else:
            high_freq_threshold = q_threshold
    else:
        high_freq_threshold = float(disease_freq_threshold if disease_freq_threshold and disease_freq_threshold > 0 else 50.0)
    disease_penalty: Dict[int, float] = {}
    alpha = max(float(disease_penalty_alpha), 0.0)
    min_penalty = min(max(float(disease_penalty_min), 1e-06), 1.0)
    for (dis_idx, freq) in disease_freq.items():
        if freq <= high_freq_threshold or alpha <= 0:
            disease_penalty[dis_idx] = 1.0
        else:
            raw_penalty = (high_freq_threshold / max(float(freq), 1e-12)) ** alpha
            disease_penalty[dis_idx] = float(max(min_penalty, min(1.0, raw_penalty)))
    degree_tmp = Counter()
    for (src_node, edges) in adj.items():
        degree_tmp[src_node] += len(edges)
    for node in list(adj.keys()):
        adj[node].sort(
            key=lambda e: (float(e.weight), -float(degree_tmp.get((e.dst_type, int(e.dst_idx)), 0))),
            reverse=True
        )
    return GraphIndex(
        node_names=node_names,
        adj=dict(adj),
        global_node_id=global_node_id,
        module_partition=module_partition,
        disease_freq=disease_freq,
        disease_penalty=disease_penalty,
        high_freq_disease_threshold=high_freq_threshold,
        bridge_factor=bridge_factor,
        degree_penalty_gamma=degree_penalty_gamma,
        module_penalty_floor=module_penalty_floor,
        disease_penalty_floor=disease_penalty_floor,
        degree_penalty_floor=degree_penalty_floor
    )

def enumerate_meta_path_templates(max_hops: int=3) -> List[Tuple[str, ...]]:
    schema_adj = {
        'rna': ['rna', 'drug', 'disease'],
        'drug': ['drug', 'rna', 'disease'],
        'disease': ['disease', 'rna', 'drug']
    }
    templates = []

    def dfs(types: List[str], hops_left: int):
        cur = types[-1]
        if len(types) > 1 and cur == 'drug':
            templates.append(tuple(types))
        if hops_left == 0:
            return
        for nxt in schema_adj[cur]:
            if len(types) >= 2 and nxt == types[-2]:
                continue
            dfs(types + [nxt], hops_left - 1)
    dfs(['rna'], max_hops)
    templates = sorted(set(templates), key=lambda x: (len(x), x))
    return templates

def classify_module_pattern(graph: GraphIndex, nodes: List[Node], max_module_changes: int=2) -> Tuple[str, List[int], int]:
    modules = [graph.module_id(n) for n in nodes]
    unique_modules = set(modules)
    module_changes = sum((int(modules[i] != modules[i - 1]) for i in range(1, len(modules))))
    if len(unique_modules) == 1:
        return ('intra_module', modules, module_changes)
    if module_changes <= max_module_changes:
        return ('bridge', modules, module_changes)
    return ('cross_module', modules, module_changes)

def soften_penalty(raw_factor: float, residual_floor: float) -> float:
    floor = min(max(float(residual_floor), 0.0), 1.0)
    raw = min(max(float(raw_factor), 0.0), 1.0)
    return float(floor + (1.0 - floor) * raw)

def disease_downweight(graph: GraphIndex, nodes: List[Node]) -> Tuple[float, List[Dict[str, Any]]]:
    factor = 1.0
    info = []
    for (ntype, idx) in nodes:
        if ntype != 'disease':
            continue
        freq = int(graph.disease_freq.get(idx, 0))
        penalty = float(graph.disease_penalty.get(idx, 1.0))
        factor *= penalty
        info.append({
            'disease_idx': int(idx),
            'freq': freq,
            'penalty': penalty,
            'is_high_freq': bool(freq > graph.high_freq_disease_threshold)
        })
    return (float(factor), info)

def path_edge_score(edges: List[Edge]) -> float:
    if not edges:
        return 0.0
    logs = [math.log(max(float(e.weight), 1e-12)) for e in edges]
    return float(math.exp(sum(logs) / len(logs)))

def path_degree_penalty(graph: GraphIndex, nodes: List[Node]) -> float:
    if len(nodes) <= 2:
        return 1.0
    mids = nodes[1:-1]
    degrees = [graph.node_degree.get(n, 0) for n in mids]
    if not degrees:
        return 1.0
    avg_log_deg = float(np.mean([math.log1p(d) for d in degrees]))
    return float(1.0 / (1.0 + graph.degree_penalty_gamma * avg_log_deg))

def score_path_instance(graph: GraphIndex, nodes: List[Node], edges: List[Edge], max_module_changes: int) -> Optional[Dict[str, Any]]:
    (keep_type, modules, module_changes) = classify_module_pattern(graph=graph, nodes=nodes, max_module_changes=max_module_changes)
    edge_score = path_edge_score(edges)
    (disease_raw_factor, disease_info) = disease_downweight(graph, nodes)
    degree_raw_factor = path_degree_penalty(graph, nodes)
    module_pairs = list(zip(modules[:-1], modules[1:]))
    same_module_ratio = float(sum((int(left == right) for (left, right) in module_pairs))) / max(1, len(module_pairs))
    endpoint_same_module = float(modules[0] == modules[-1]) if modules else 0.0
    module_span = len(set(modules)) if modules else 0
    if keep_type == 'intra_module':
        module_raw_factor = 1.0
    elif keep_type == 'bridge':
        module_raw_factor = graph.bridge_factor
    else:
        excess = max(1, module_changes - max_module_changes)
        module_raw_factor = graph.bridge_factor ** (1 + excess)
    module_factor = soften_penalty(module_raw_factor, graph.module_penalty_floor)
    disease_factor = soften_penalty(disease_raw_factor, graph.disease_penalty_floor)
    degree_factor = soften_penalty(degree_raw_factor, graph.degree_penalty_floor)
    cohesion_factor = 0.35 + 0.65 * same_module_ratio
    endpoint_factor = 1.35 if endpoint_same_module else 0.65
    span_factor = 1.0 / (1.0 + 0.25 * max(0, module_span - 1))
    change_factor = math.exp(-0.45 * module_changes)
    score = edge_score * disease_factor * degree_factor * module_factor * cohesion_factor * endpoint_factor * span_factor * change_factor
    rels = [e.rel for e in edges]
    meta_path = tuple([nodes[0][0]] + [e.dst_type for e in edges])
    return {
        'score': float(score),
        'edge_score': float(edge_score),
        'disease_factor': float(disease_factor),
        'degree_factor': float(degree_factor),
        'module_factor': float(module_factor),
        'disease_raw_factor': float(disease_raw_factor),
        'degree_raw_factor': float(degree_raw_factor),
        'module_raw_factor': float(module_raw_factor),
        'module_cohesion': float(same_module_ratio),
        'endpoint_same_module': bool(endpoint_same_module),
        'module_span': int(module_span),
        'cohesion_factor': float(cohesion_factor),
        'endpoint_factor': float(endpoint_factor),
        'module_span_factor': float(span_factor),
        'module_change_factor': float(change_factor),
        'keep_type': keep_type,
        'modules': [int(x) for x in modules],
        'module_changes': int(module_changes),
        'disease_info': disease_info,
        'meta_path': list(meta_path),
        'rels': rels,
        'structural_entropy_guidance': True,
        'path_ranking': 'low_entropy_cohesion_bottleneck_score'
    }

def min_schema_hops_to_drug(ntype: str) -> int:
    if ntype == 'drug':
        return 0
    if ntype in {'rna', 'disease'}:
        return 1
    return 999

def jaccard_nodes(path_a: List[Node], path_b: List[Node]) -> float:
    a = set(path_a)
    b = set(path_b)
    if not a and (not b):
        return 1.0
    return len(a & b) / max(1, len(a | b))

def retrieve_top_paths_for_pair(
    graph: GraphIndex,
    rna_idx: int,
    drug_idx: int,
    topk: int=CONFIG.rag.topk,
    max_hops: int=CONFIG.rag.max_hops,
    min_hops: int=CONFIG.rag.min_hops,
    max_branch_per_step: int=CONFIG.rag.max_branch_per_step,
    max_module_changes: int=CONFIG.rag.max_module_changes,
    jaccard_dedup_threshold: float=0.85
) -> List[Dict[str, Any]]:
    start: Node = ('rna', int(rna_idx))
    target: Node = ('drug', int(drug_idx))
    allowed_templates = set(enumerate_meta_path_templates(max_hops=max_hops))
    allowed_prefixes = {template[:length] for template in allowed_templates for length in range(1, len(template) + 1)}
    if not graph.has_node(start) or not graph.has_node(target):
        return []
    if start not in graph.adj:
        return []
    candidates: List[Dict[str, Any]] = []
    seen_hash = set()

    def dfs(cur: Node, nodes: List[Node], edges: List[Edge], visited: set) -> None:
        hops = len(edges)
        if cur == target and hops > 0:
            if tuple((node_type for (node_type, _) in nodes)) not in allowed_templates:
                return
            if hops == 1:
                return
            if hops < min_hops:
                return
            scored = score_path_instance(graph=graph, nodes=nodes, edges=edges, max_module_changes=max_module_changes)
            if scored is None:
                return
            h = path_hash(nodes, scored['rels'])
            if h in seen_hash:
                return
            seen_hash.add(h)
            scored.update({
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'nodes': [[t, int(i)] for (t, i) in nodes],
                'path_hash': h
            })
            candidates.append(scored)
            return
        if hops >= max_hops:
            return
        remaining_after_next = max_hops - hops - 1
        neighbors = graph.neighbors(cur)[:max_branch_per_step]
        for e in neighbors:
            nxt: Node = (e.dst_type, int(e.dst_idx))
            if nxt in visited:
                continue
            type_prefix = tuple((node_type for (node_type, _) in nodes + [nxt]))
            if type_prefix not in allowed_prefixes:
                continue
            if min_schema_hops_to_drug(nxt[0]) > remaining_after_next:
                continue
            if len(edges) == 0 and nxt == target:
                continue
            if len(edges) >= 2:
                if edges[-1].rel == 'sim' and edges[-2].rel == 'sim' and (e.rel == 'sim'):
                    continue
            if nxt[0] == 'drug' and nxt != target and (remaining_after_next == 0):
                continue
            dfs(cur=nxt, nodes=nodes + [nxt], edges=edges + [e], visited=visited | {nxt})
    dfs(cur=start, nodes=[start], edges=[], visited={start})
    candidates.sort(
        key=lambda x: (x['score'], x.get('module_cohesion', 0.0), -x.get('module_span', 999), x.get('edge_score', 0.0)),
        reverse=True
    )
    selected: List[Dict[str, Any]] = []
    selected_node_paths: List[List[Node]] = []
    remaining = list(candidates)
    while remaining and len(selected) < topk:
        best_index = 0
        best_rank = -float('inf')
        for (index, item) in enumerate(remaining):
            nodes = [(str(t), int(i)) for (t, i) in item['nodes']]
            novelty = 1.0
            for old_nodes in selected_node_paths:
                novelty = min(novelty, 1.0 - jaccard_nodes(nodes, old_nodes))
            rank = float(item['score']) * (0.75 + 0.25 * novelty)
            if rank > best_rank:
                best_rank = rank
                best_index = index
        item = remaining.pop(best_index)
        nodes = [(str(t), int(i)) for (t, i) in item['nodes']]
        too_similar = False
        for old_nodes in selected_node_paths:
            if jaccard_nodes(nodes, old_nodes) >= jaccard_dedup_threshold:
                too_similar = True
                break
        if too_similar:
            continue
        selected.append(item)
        selected_node_paths.append(nodes)
    return selected

def to_graph_sentence(graph: GraphIndex, nodes: List[List[Any]]) -> str:
    typed_nodes = [(str(t), int(i)) for (t, i) in nodes]
    tokens = [graph.token(n) for n in typed_nodes]
    return ' '.join(tokens)

def to_graph_sentence_with_rel(graph: GraphIndex, nodes: List[List[Any]], rels: List[str]) -> str:
    typed_nodes = [(str(t), int(i)) for (t, i) in nodes]
    node_tokens = [graph.token(n) for n in typed_nodes]
    pieces = []
    for (i, tok) in enumerate(node_tokens):
        pieces.append(tok)
        if i < len(rels):
            pieces.append(f'<REL:{rels[i]}>')
    return ' '.join(pieces)

def _minhash_clusters(node_sets: List[set[int]], threshold: float, num_permutations: int, seed: int) -> Tuple[List[int], Dict[str, Any]]:
    if not node_sets:
        return ([], {'candidate_pairs': 0, 'verified_pairs': 0})
    if num_permutations < 4:
        raise ValueError('num_permutations must be at least 4')
    prime = np.uint64(4294967311)
    parameter_rng = np.random.default_rng(seed)
    coefficients_a = parameter_rng.integers(1, int(prime), size=num_permutations, dtype=np.uint64)
    coefficients_b = parameter_rng.integers(0, int(prime), size=num_permutations, dtype=np.uint64)
    signatures = np.full((len(node_sets), num_permutations), prime, dtype=np.uint64)
    for (row, values) in enumerate(node_sets):
        if not values:
            continue
        ids = np.asarray(sorted(values), dtype=np.uint64)[:, None]
        signatures[row] = ((ids * coefficients_a + coefficients_b) % prime).min(axis=0)
    rows_per_band = 4
    band_count = max(1, num_permutations // rows_per_band)
    representative_buckets: Dict[Tuple[int, bytes], List[int]] = defaultdict(list)
    representative_cluster: Dict[int, int] = {}
    cluster_ids: List[int] = []
    comparisons = 0
    assigned = 0
    for row in range(len(node_sets)):
        candidate_representatives = set()
        band_keys = []
        for band in range(band_count):
            start = band * rows_per_band
            stop = min(start + rows_per_band, num_permutations)
            key = (band, signatures[row, start:stop].tobytes())
            band_keys.append(key)
            candidate_representatives.update(representative_buckets.get(key, ()))
        best_representative = None
        best_similarity = -1.0
        for representative in sorted(candidate_representatives):
            comparisons += 1
            union = node_sets[row] | node_sets[representative]
            similarity = len(node_sets[row] & node_sets[representative]) / max(1, len(union))
            if similarity >= threshold and similarity > best_similarity:
                best_representative = representative
                best_similarity = similarity
        if best_representative is None:
            cluster_id = len(representative_cluster)
            representative_cluster[row] = cluster_id
            for key in band_keys:
                representative_buckets[key].append(row)
        else:
            cluster_id = representative_cluster[best_representative]
            assigned += 1
        cluster_ids.append(cluster_id)
    return (
        cluster_ids,
        {
            'candidate_pairs': int(comparisons),
            'verified_pairs': int(assigned),
            'representative_clusters': int(len(representative_cluster)),
            'transitive_merging': False
        }
    )

def sample_graphrag_training_paths(
    graph: GraphIndex,
    pair_records: List[Dict[str, Any]],
    out_dir: str,
    sampling_ratio: float,
    minhash_threshold: float,
    minhash_permutations: int,
    quality_strength: float,
    sampling_quality_mix: float,
    seed: int
) -> Dict[str, Any]:
    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError('sampling_ratio must be in (0, 1]')
    if not 0.0 <= minhash_threshold <= 1.0:
        raise ValueError('minhash_threshold must be in [0, 1]')
    if minhash_permutations < 4:
        raise ValueError('minhash_permutations must be at least 4')
    if quality_strength < 0.0:
        raise ValueError('quality_strength must be non-negative')
    if not 0.0 <= sampling_quality_mix <= 1.0:
        raise ValueError('sampling_quality_mix must be in [0, 1]')
    candidates: List[Dict[str, Any]] = []
    for pair_record in pair_records:
        source = str(pair_record.get('source', ''))
        label = pair_record.get('label')
        if source not in {'train', 'train_pos'} or label == 0:
            continue
        for path in pair_record.get('paths', []):
            candidates.append({
                **path,
                'pair_id': int(pair_record['pair_id']),
                'rna_idx': int(pair_record['rna_idx']),
                'drug_idx': int(pair_record['drug_idx']),
                'pair_source': source
            })
    if not candidates:
        raise RuntimeError('结构熵 GraphRAG 未为 train_pos 产生可用于 LoRA 的路径；请检查 topk、hop 约束或图构建结果。')
    node_sets = [{graph.global_id((str(node_type), int(index))) for (node_type, index) in path['nodes']} for path in candidates]
    (cluster_ids, cluster_stats) = _minhash_clusters(node_sets, minhash_threshold, minhash_permutations, seed)
    cluster_sizes = Counter(cluster_ids)
    log_scores = np.log(np.asarray([max(float(path.get('score', 0.0)), 1e-12) for path in candidates], dtype=np.float64))
    score_z = (log_scores - log_scores.mean()) / max(float(log_scores.std()), 1e-08)
    quality_weights = np.asarray([1.0 / cluster_sizes[cluster_id] for cluster_id in cluster_ids], dtype=np.float64)
    if quality_strength > 0.0:
        quality_weights *= np.exp(np.clip(quality_strength * score_z, -8.0, 8.0))
    quality_weights /= quality_weights.sum()
    uniform_weights = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    weights = (1.0 - float(sampling_quality_mix)) * uniform_weights + float(sampling_quality_mix) * quality_weights
    weights /= weights.sum()
    sample_size = max(1, min(len(candidates), int(round(sampling_ratio * len(candidates)))))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=sample_size, replace=False, p=weights).tolist()
    rng.shuffle(chosen)
    output_path = os.path.join(out_dir, 'sampled_graph_rag_sentences.jsonl')
    selected_cluster_counts: Counter = Counter()
    with open(output_path, 'w', encoding='utf-8') as handle:
        for index in chosen:
            path = dict(candidates[index])
            cluster_id = int(cluster_ids[index])
            path.update({
                'path_id': int(index),
                'cluster_id': cluster_id,
                'cluster_size': int(cluster_sizes[cluster_id]),
                'sampling_weight': float(weights[index]),
                'source': 'graph_train'
            })
            selected_cluster_counts[cluster_id] += 1
            handle.write(json.dumps(make_json_safe(path), ensure_ascii=False) + '\n')
    return {
        'num_train_path_candidates': int(len(candidates)),
        'num_sampled_train_paths': int(sample_size),
        'sampling_ratio': float(sampling_ratio),
        'minhash_threshold': float(minhash_threshold),
        'minhash_permutations': int(minhash_permutations),
        'quality_strength': float(quality_strength),
        'sampling_quality_mix': float(sampling_quality_mix),
        'sampling_distribution': 'uniform_plus_se_quality_mixture',
        'structural_entropy_guidance': bool(True),
        'path_quality_source': 'soft_structural_entropy_score',
        'num_clusters': int(len(cluster_sizes)),
        'largest_cluster': int(max(cluster_sizes.values())),
        'selected_cluster_counts': dict(selected_cluster_counts),
        'minhash': cluster_stats,
        'output_file': os.path.abspath(output_path)
    }

def _dedup_query_pairs(items: List[QueryPair]) -> List[QueryPair]:
    seen = set()
    out: List[QueryPair] = []
    for q in items:
        key = (int(q.rna_idx), int(q.drug_idx), q.label, q.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out

def _pairs_from_array(pairs: np.ndarray, label: Optional[int], source: str, scores: Optional[np.ndarray]=None) -> List[QueryPair]:
    pairs = np.asarray(pairs).astype(np.int64)
    out: List[QueryPair] = []
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
    for (idx, row) in enumerate(pairs):
        sample_score = None
        if scores is not None and idx < len(scores):
            sample_score = float(scores[idx])
        out.append(QueryPair(rna_idx=int(row[0]), drug_idx=int(row[1]), label=label, sample_score=sample_score, source=source))
    return out

def load_query_pairs(fold_dir, pair_source='train_val_pos'):
    split = torch_load_cpu(os.path.join(fold_dir, 'split_data.pt'))
    sources = ('train_pos', 'val_pos') if pair_source == 'train_val_pos' else (pair_source,)
    items = []
    for source in sources:
        if source not in ('train_pos', 'val_pos'):
            raise ValueError('Only training positives and separate LM validation positives are permitted')
        key = 'train_positive_pairs' if source == 'train_pos' else 'val_positive_pairs'
        items.extend(_pairs_from_array(to_numpy(split[key]), 1, source))
    return _dedup_query_pairs(items)

def build_corpus(
    fold_dir: str,
    out_dir: str,
    pair_source: str='train_val_pos',
    topk: int=10,
    max_hops: int=3,
    min_hops: int=2,
    max_branch_per_step: int=80,
    max_module_changes: int=2,
    jaccard_dedup_threshold: float=0.85,
    disease_freq_threshold: float=CONFIG.rag.disease_freq_threshold,
    disease_freq_quantile: float=CONFIG.rag.disease_freq_quantile,
    disease_penalty_alpha: float=CONFIG.rag.disease_penalty_alpha,
    disease_penalty_min: float=CONFIG.rag.disease_penalty_min,
    bridge_factor: float=CONFIG.rag.bridge_factor,
    degree_penalty_gamma: float=CONFIG.rag.degree_penalty_gamma,
    module_penalty_floor: float=CONFIG.rag.module_penalty_floor,
    disease_penalty_floor: float=CONFIG.rag.disease_penalty_floor,
    degree_penalty_floor: float=CONFIG.rag.degree_penalty_floor,
    sampling_ratio: float=CONFIG.rag.sampling_ratio,
    minhash_threshold: float=CONFIG.rag.minhash_threshold,
    minhash_permutations: int=CONFIG.rag.minhash_permutations,
    quality_strength: float=CONFIG.rag.quality_strength,
    sampling_quality_mix: float=CONFIG.rag.sampling_quality_mix,
    seed: int=CONFIG.graph.seed
) -> Dict[str, Any]:
    if not 0.0 <= disease_freq_quantile <= 1.0:
        raise ValueError('disease_freq_quantile must be in [0, 1]')
    if disease_penalty_alpha < 0.0 or degree_penalty_gamma < 0.0:
        raise ValueError('disease and degree penalty strengths must be non-negative')
    bounded = (
        disease_penalty_min,
        bridge_factor,
        module_penalty_floor,
        disease_penalty_floor,
        degree_penalty_floor,
        sampling_quality_mix
    )
    if any((value < 0.0 or value > 1.0 for value in bounded)):
        raise ValueError('GraphRAG factors, residual floors and sampling mix must be in [0, 1]')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[1/5] 读取图结构: {fold_dir}')
    graph = load_graph_index(
        fold_dir,
        disease_freq_threshold=disease_freq_threshold,
        disease_freq_quantile=disease_freq_quantile,
        disease_penalty_alpha=disease_penalty_alpha,
        disease_penalty_min=disease_penalty_min,
        bridge_factor=bridge_factor,
        degree_penalty_gamma=degree_penalty_gamma,
        module_penalty_floor=module_penalty_floor,
        disease_penalty_floor=disease_penalty_floor,
        degree_penalty_floor=degree_penalty_floor
    )
    print('[2/5] 枚举候选元路径模板')
    templates = enumerate_meta_path_templates(max_hops=max_hops)
    print(f'      候选 schema 元路径数: {len(templates)}')
    for t in templates:
        print('      ' + ' -> '.join(t))
    print(f'[3/5] 读取 query pairs: {pair_source}')
    query_pairs = load_query_pairs(fold_dir=fold_dir, pair_source=pair_source)
    label_counter = Counter((q.label for q in query_pairs))
    print(f'      pairs = {len(query_pairs)}')
    print(f'      labels = {dict(label_counter)}')
    jsonl_path = os.path.join(out_dir, 'graph_rag_paths.jsonl')
    meta_summary_path = os.path.join(out_dir, 'meta_path_summary.json')
    meta_counter = Counter()
    keep_counter = Counter()
    label_path_counter = Counter()
    all_records = []
    print('[4/5] GraphRAG 检索 top 路径实例 + 去重')
    with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
        for (idx, q) in enumerate(query_pairs):
            paths = retrieve_top_paths_for_pair(
                graph=graph,
                rna_idx=int(q.rna_idx),
                drug_idx=int(q.drug_idx),
                topk=topk,
                max_hops=max_hops,
                min_hops=min_hops,
                max_branch_per_step=max_branch_per_step,
                max_module_changes=max_module_changes,
                jaccard_dedup_threshold=jaccard_dedup_threshold
            )
            pair_record = {
                'pair_id': idx,
                'rna_idx': int(q.rna_idx),
                'drug_idx': int(q.drug_idx),
                'label': q.label,
                'sample_score': q.sample_score,
                'source': q.source,
                'num_paths': len(paths),
                'paths': []
            }
            for p in paths:
                sent = to_graph_sentence(graph, p['nodes'])
                sent_rel = to_graph_sentence_with_rel(graph, p['nodes'], p['rels'])
                p['sentence'] = sent
                p['sentence_with_rel'] = sent_rel
                p['pair_label'] = q.label
                p['pair_sample_score'] = q.sample_score
                p['pair_source'] = q.source
                pair_record['paths'].append(p)
                meta_counter[' -> '.join(p['meta_path'])] += 1
                keep_counter[p['keep_type']] += 1
                label_path_counter[str(q.label)] += 1
            f_jsonl.write(json.dumps(make_json_safe(pair_record), ensure_ascii=False) + '\n')
            all_records.append(pair_record)
            if (idx + 1) % 100 == 0:
                print(f'      已处理 {idx + 1}/{len(query_pairs)} pairs')
    print('[5/6] 对已筛选的 GraphRAG 训练路径执行全局 MinHash 去冗余采样')
    sampling_summary = sample_graphrag_training_paths(
        graph=graph,
        pair_records=all_records,
        out_dir=out_dir,
        sampling_ratio=sampling_ratio,
        minhash_threshold=minhash_threshold,
        minhash_permutations=minhash_permutations,
        quality_strength=quality_strength,
        sampling_quality_mix=sampling_quality_mix,
        seed=seed
    )
    meta_summary = {
        'fold_dir': fold_dir,
        'pair_source': pair_source,
        'max_hops': max_hops,
        'min_hops': min_hops,
        'topk': topk,
        'max_module_changes': max_module_changes,
        'allow_direct_evidence': False,
        'allow_long_sim_chain': False,
        'disease_freq_threshold': disease_freq_threshold,
        'disease_freq_quantile': disease_freq_quantile,
        'disease_penalty_alpha': disease_penalty_alpha,
        'disease_penalty_min': disease_penalty_min,
        'bridge_factor': bridge_factor,
        'degree_penalty_gamma': degree_penalty_gamma,
        'module_penalty_floor': module_penalty_floor,
        'disease_penalty_floor': disease_penalty_floor,
        'degree_penalty_floor': degree_penalty_floor,
        'structural_entropy_guidance': bool(True),
        'path_debias_mode': 'structural_entropy_soft_residual',
        'global_minhash_sampling': sampling_summary,
        'num_pairs': int(len(query_pairs)),
        'pair_label_counts': {str(k): int(v) for (k, v) in label_counter.items()},
        'num_lm_graph_sentences': int(sum((r['num_paths'] for r in all_records))),
        'num_all_retrieved_paths': int(sum((r['num_paths'] for r in all_records))),
        'path_label_counts': dict(label_path_counter),
        'keep_type_counts': dict(keep_counter),
        'meta_path_counts': dict(meta_counter.most_common()),
        'candidate_schema_templates': [list(t) for t in templates],
        'high_freq_disease': {
            'threshold': float(graph.high_freq_disease_threshold),
            'num_disease_with_freq': int(len(graph.disease_freq)),
            'top20_by_freq': [{'disease_idx': int(k), 'freq': int(v), 'penalty': float(graph.disease_penalty.get(k, 1.0))} for (k, v) in Counter(graph.disease_freq).most_common(20)]
        },
        'output_files': {
            'graph_rag_paths': jsonl_path,
            'sampled_graph_rag_sentences': sampling_summary['output_file'],
            'meta_path_summary': meta_summary_path
        }
    }
    print('[6/6] 保存统计')
    with open(meta_summary_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(meta_summary), f, ensure_ascii=False, indent=2)
    print('\n完成输出：')
    print(f'  {jsonl_path}')
    print(f"  {sampling_summary['output_file']}")
    print(f'  {meta_summary_path}')
    return meta_summary
