"""Structural entropy, graph partitions and strict identifier alignment."""
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
import os
import torch
import pandas as pd

def normalize_identifier(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if text.endswith('.0'):
        head = text[:-2]
        if head.lstrip('-').isdigit():
            return head
    return text

def _find_duplicates(values: List[str]) -> List[str]:
    seen = set()
    dup = []
    for v in values:
        if v in seen and v not in dup:
            dup.append(v)
        seen.add(v)
    return dup

def _preview(values: List[Any], limit: int=10) -> List[Any]:
    return list(values[:limit])

def _check_unique_ids(values: List[str], label: str, source: str) -> None:
    dup = _find_duplicates(values)
    if dup:
        raise ValueError(f'{source} 中 {label} 存在重复 ID，共 {len(dup)} 个。示例: {_preview(dup)}。重复 ID 会导致关系矩阵/特征无法唯一对齐，请先修正输入文件。')

def _missing_fraction(num_missing: int, total: int) -> float:
    return float(num_missing) / float(max(1, total))

def validate_similarity_matrix(
    mat: np.ndarray,
    row_names: Optional[List[str]]=None,
    col_names: Optional[List[str]]=None,
    symmetrize: bool=True,
    normalize_similarity: bool=False,
    check_row_col_match: bool=True
) -> np.ndarray:
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f'相似度矩阵必须是方阵，当前形状: {mat.shape}')
    if row_names is not None and col_names is not None and check_row_col_match:
        if len(row_names) != len(col_names):
            raise ValueError(f'行名数量({len(row_names)}) != 列名数量({len(col_names)})')
        if row_names != col_names:
            raise ValueError('相似度矩阵的行名和列名必须完全一致')
    if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
        raise ValueError('相似度矩阵包含 NaN 或 Inf')
    if np.any(mat < 0):
        raise ValueError('相似度矩阵包含负值')
    if symmetrize:
        mat = np.maximum(mat, mat.T)
    if normalize_similarity:
        (vmin, vmax) = (mat.min(), mat.max())
        if vmax > vmin:
            mat = (mat - vmin) / (vmax - vmin)
    return mat.astype(np.float32)

def load_similarity_matrix(sim_file: str, symmetrize: bool=True, normalize_similarity: bool=False, allow_reindex: bool=False) -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(sim_file):
        raise FileNotFoundError(f'Similarity file not found: {sim_file}')
    df = pd.read_csv(sim_file, index_col=0)
    df.index = [normalize_identifier(x) for x in df.index]
    df.columns = [normalize_identifier(x) for x in df.columns]
    _check_unique_ids(list(df.index), '相似度矩阵行 ID', str(sim_file))
    _check_unique_ids(list(df.columns), '相似度矩阵列 ID', str(sim_file))
    if df.shape[0] != df.shape[1] or list(df.index) != list(df.columns):
        if not allow_reindex:
            raise ValueError(f'相似度矩阵行列名/形状不一致: {sim_file}。请先修正输入文件，或在确认安全时显式设置 allow_reindex=True。')
        df = df.reindex(index=df.index, columns=df.index, fill_value=0).fillna(0)
    names = df.index.tolist()
    col_names = df.columns.tolist()
    mat = np.nan_to_num(df.values.astype(np.float32), nan=0.0)
    mat = validate_similarity_matrix(
        mat,
        row_names=names,
        col_names=col_names,
        symmetrize=symmetrize,
        normalize_similarity=normalize_similarity,
        check_row_col_match=True
    )
    return (mat, names)

def load_pair_interactions(
    link_file,
    row_names,
    col_names,
    row_col,
    col_col,
    *,
    allow_missing: bool=False,
    max_unmatched_frac: float=0.0
):
    if not os.path.exists(link_file):
        raise FileNotFoundError(f'Link file not found: {link_file}')
    row_ids = [normalize_identifier(x) for x in row_names]
    col_ids = [normalize_identifier(x) for x in col_names]
    _check_unique_ids(row_ids, 'row_names', str(link_file))
    _check_unique_ids(col_ids, 'col_names', str(link_file))
    row2idx = {n: i for (i, n) in enumerate(row_ids)}
    col2idx = {n: i for (i, n) in enumerate(col_ids)}
    mat = np.zeros((len(row_ids), len(col_ids)), dtype=np.float32)
    df = pd.read_csv(link_file)
    if row_col not in df.columns or col_col not in df.columns:
        raise ValueError(f'{link_file} 缺少必要列：{row_col!r} 或 {col_col!r}。实际列名: {list(df.columns)}')
    unmatched = []
    duplicated = 0
    matched = 0
    seen = set()
    for (row_i, (raw_r, raw_c)) in enumerate(zip(df[row_col], df[col_col])):
        r = normalize_identifier(raw_r)
        c = normalize_identifier(raw_c)
        if r not in row2idx or c not in col2idx:
            unmatched.append({
                'row_number': int(row_i),
                'row_id': r,
                'col_id': c,
                'row_known': bool(r in row2idx),
                'col_known': bool(c in col2idx)
            })
            continue
        pair = (row2idx[r], col2idx[c])
        if pair in seen:
            duplicated += 1
            continue
        seen.add(pair)
        mat[pair[0], pair[1]] = 1.0
        matched += 1
    unmatched_frac = _missing_fraction(len(unmatched), len(df))
    if unmatched and (not allow_missing or unmatched_frac > max_unmatched_frac):
        raise ValueError(f'{link_file} 有 {len(unmatched)} / {len(df)} 条关系对无法按 ID 对齐 ({unmatched_frac:.2%})。默认禁止静默跳过，否则会把真实关系误当成不存在。\n无法匹配示例: {_preview(unmatched)}\n请检查 {row_col}/{col_col} 是否与相似性矩阵的行列名一致。')
    if matched == 0 and len(df) > 0:
        raise ValueError(f'{link_file} 没有任何关系对成功匹配。请检查 ID 格式是否一致，例如 11302 vs 11302.0、CID 前缀、大小写或空格。')
    return mat

def load_matrix_interactions(
    matrix_file,
    row_names,
    col_names,
    *,
    allow_missing: bool=False,
    max_missing_frac: float=0.0,
    allow_extra_ids: bool=True,
    binary: bool=True
):
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f'Matrix file not found: {matrix_file}')
    df = pd.read_csv(matrix_file, index_col=0)
    df.index = [normalize_identifier(x) for x in df.index]
    df.columns = [normalize_identifier(x) for x in df.columns]
    row_ids = [normalize_identifier(x) for x in row_names]
    col_ids = [normalize_identifier(x) for x in col_names]
    _check_unique_ids(row_ids, '目标行 ID', str(matrix_file))
    _check_unique_ids(col_ids, '目标列 ID', str(matrix_file))
    _check_unique_ids(list(df.index), '矩阵行 ID', str(matrix_file))
    _check_unique_ids(list(df.columns), '矩阵列 ID', str(matrix_file))
    row_set = set(row_ids)
    col_set = set(col_ids)
    df_row_set = set(df.index)
    df_col_set = set(df.columns)
    missing_rows = sorted(row_set - df_row_set)
    missing_cols = sorted(col_set - df_col_set)
    extra_rows = sorted(df_row_set - row_set)
    extra_cols = sorted(df_col_set - col_set)
    missing_row_frac = _missing_fraction(len(missing_rows), len(row_ids))
    missing_col_frac = _missing_fraction(len(missing_cols), len(col_ids))
    if extra_rows and (not allow_extra_ids):
        raise ValueError(f'{matrix_file} 存在 {len(extra_rows)} 个额外行 ID。示例: {_preview(extra_rows)}')
    if extra_cols and (not allow_extra_ids):
        raise ValueError(f'{matrix_file} 存在 {len(extra_cols)} 个额外列 ID。示例: {_preview(extra_cols)}')
    if missing_rows or missing_cols:
        if not allow_missing or missing_row_frac > max_missing_frac or missing_col_frac > max_missing_frac:
            raise ValueError(f'{matrix_file} ID 对齐失败：缺失行 {len(missing_rows)} / {len(row_ids)} ({missing_row_frac:.2%})，缺失列 {len(missing_cols)} / {len(col_ids)} ({missing_col_frac:.2%})。\n缺失行示例: {_preview(missing_rows)}\n缺失列示例: {_preview(missing_cols)}\n额外行示例: {_preview(extra_rows)}\n额外列示例: {_preview(extra_cols)}\n默认禁止把缺失 ID 静默补 0，因为这会把数据工程错误伪装成真实无关联。')
    aligned = df.reindex(index=row_ids, columns=col_ids)
    if aligned.isna().any().any():
        if not allow_missing:
            raise ValueError(f'{matrix_file} reindex 后出现 NaN。请检查 ID 对齐，或显式 allow_missing=True。')
        aligned = aligned.fillna(0)
    numeric = aligned.apply(pd.to_numeric, errors='coerce')
    if numeric.isna().any().any():
        (bad_rows, bad_cols) = np.where(numeric.isna().values)
        examples = []
        for (rr, cc) in zip(bad_rows[:10], bad_cols[:10]):
            examples.append({'row_id': row_ids[int(rr)], 'col_id': col_ids[int(cc)], 'value': aligned.iat[int(rr), int(cc)]})
        raise ValueError(f'{matrix_file} 包含非数值关系条目，无法安全转为矩阵。示例: {examples}')
    values = numeric.values.astype(np.float32)
    if np.any(values < 0):
        raise ValueError(f'{matrix_file} 包含负值，不应作为 0/1 关系矩阵。')
    if binary:
        values = (values > 0).astype(np.float32)
    return values

def _build_sim_adjacency(sim: np.ndarray, knn: int=0, threshold: float=0.0, remove_self_loop: bool=True) -> np.ndarray:
    n = sim.shape[0]
    W = np.array(sim, dtype=np.float64, copy=True)
    if remove_self_loop:
        np.fill_diagonal(W, 0.0)
    W[W < threshold] = 0.0
    if knn and knn < n:
        mask = np.zeros_like(W, dtype=bool)
        idx = np.argpartition(-W, kth=knn - 1, axis=1)[:, :knn]
        rows = np.repeat(np.arange(n), knn)
        mask[rows, idx.reshape(-1)] = True
        W = np.where(mask, W, 0.0)
        W = np.maximum(W, W.T)
    else:
        W = np.maximum(W, W.T)
    return W

def build_similarity_adjacency(sim: np.ndarray, knn: int=0, threshold: float=0.0, remove_self_loop: bool=True) -> np.ndarray:
    return _build_sim_adjacency(sim, knn=knn, threshold=threshold, remove_self_loop=remove_self_loop)

def two_dim_structural_entropy(W: np.ndarray, labels: np.ndarray) -> float:
    deg = W.sum(axis=1)
    vol = deg.sum()
    if vol <= 0:
        return 0.0
    entropy = 0.0
    for c in np.unique(labels):
        nodes = np.where(labels == c)[0]
        V_X = deg[nodes].sum()
        if V_X <= 0:
            continue
        in_block = np.zeros(W.shape[0], dtype=bool)
        in_block[nodes] = True
        g_X = W[np.ix_(nodes, ~in_block)].sum()
        di = deg[nodes]
        di_pos = di[di > 0]
        entropy -= np.sum(di_pos / vol * np.log2(di_pos / V_X))
        if g_X > 0:
            entropy -= g_X / vol * np.log2(V_X / vol)
    return float(entropy)

def _community_cost(V, L, vol):
    if V <= 0:
        return 0.0
    cost = V * np.log2(V)
    cut = V - L
    if cut > 0 and vol > 0:
        cost -= cut * np.log2(V / vol)
    return float(cost)

def low_entropy_partition(
    W: np.ndarray,
    max_pass: int=10,
    min_delta: float=1e-06,
    random_seed: Optional[int]=None,
    verbose: bool=False
) -> np.ndarray:
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random
    n = W.shape[0]
    labels = np.arange(n)
    if n <= 1 or W.sum() <= 0:
        return labels
    deg = W.sum(axis=1)
    vol = float(deg.sum())
    V = deg.astype(np.float64).copy()
    L = np.zeros(n, dtype=np.float64)
    neighbors = [np.where(W[i] > 0)[0] for i in range(n)]
    for pass_idx in range(max_pass):
        improved = False
        order = rng.permutation(n)
        for i in order:
            cur = labels[i]
            nbr = neighbors[i]
            if nbr.size == 0:
                continue
            di = deg[i]
            comm_link = {}
            for (c, w) in zip(labels[nbr].tolist(), W[i, nbr].tolist()):
                comm_link[c] = comm_link.get(c, 0.0) + w
            k_i_cur = comm_link.get(cur, 0.0)
            V_cur_out = V[cur] - di
            L_cur_out = L[cur] - 2.0 * k_i_cur
            base_removed = _community_cost(V_cur_out, L_cur_out, vol) - _community_cost(V[cur], L[cur], vol)
            best_delta = -min_delta
            best_c = cur
            for (c, k_i_c) in comm_link.items():
                if c == cur:
                    continue
                add = _community_cost(V[c] + di, L[c] + 2.0 * k_i_c, vol) - _community_cost(V[c], L[c], vol)
                if base_removed + add < best_delta:
                    best_delta = base_removed + add
                    best_c = c
            if best_c != cur:
                V[cur] = V_cur_out
                L[cur] = L_cur_out
                V[best_c] += di
                L[best_c] += 2.0 * comm_link[best_c]
                labels[i] = best_c
                improved = True
        if verbose:
            h2 = two_dim_structural_entropy(W, labels)
            print(f'[模块划分] 轮次={pass_idx + 1}，模块数={len(np.unique(labels))}，H2={h2:.4f}', flush=True)
        if not improved:
            break
    (_, labels) = np.unique(labels, return_inverse=True)
    return labels

def compute_delta_h2_add(
    W: np.ndarray,
    labels: np.ndarray,
    i: int,
    j: int,
    weight: float=1.0,
    h2_before: Optional[float]=None
) -> float:
    W_temp = W.copy()
    W_temp[i, j] += weight
    W_temp[j, i] += weight
    if h2_before is None:
        h2_before = two_dim_structural_entropy(W, labels)
    h2_after = two_dim_structural_entropy(W_temp, labels)
    return h2_after - h2_before

def adjacency_to_edge_index(W: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    (src, dst) = np.nonzero(W)
    if src.size == 0:
        return (torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float))
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(W[src, dst], dtype=torch.float)
    return (edge_index, edge_weight)

def compute_module_distance(labels: np.ndarray, W: np.ndarray, epsilon: float=1e-06, verbose: bool=False) -> np.ndarray:
    unique_labels = np.unique(labels)
    num_modules = len(unique_labels)
    label_to_idx = {label: idx for (idx, label) in enumerate(unique_labels)}
    module_weight = np.zeros((num_modules, num_modules))
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[0]):
            if W[i, j] > 0:
                mi = label_to_idx[labels[i]]
                mj = label_to_idx[labels[j]]
                if mi != mj:
                    module_weight[mi, mj] += W[i, j]
                    module_weight[mj, mi] += W[i, j]
    module_dist = np.full((num_modules, num_modules), np.inf)
    np.fill_diagonal(module_dist, 0.0)
    for i in range(num_modules):
        for j in range(i + 1, num_modules):
            if module_weight[i, j] > 0:
                dist = 1.0 / (module_weight[i, j] + epsilon)
                module_dist[i, j] = dist
                module_dist[j, i] = dist
    progress_step = max(1, num_modules // 10)
    for k in range(num_modules):
        if verbose and (k == 0 or (k + 1) % progress_step == 0 or k + 1 == num_modules):
            percent = int(round((k + 1) * 100 / max(1, num_modules)))
            print(f'        [模块距离] {k + 1}/{num_modules} ({percent}%)', flush=True)
        for i in range(num_modules):
            for j in range(num_modules):
                if module_dist[i, k] + module_dist[k, j] < module_dist[i, j]:
                    module_dist[i, j] = module_dist[i, k] + module_dist[k, j]
    return module_dist
