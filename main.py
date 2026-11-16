"""Train SEGMDAL and predict associations from colocated data."""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import os
import torch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from contextlib import contextmanager
import csv
from dataclasses import asdict
import gc
import random
from threadpoolctl import threadpool_limits
import config
import data_processing as data
import graph_rag as rag
import llm_lora_branch as llm
import riemannian_gnn_branch as gnn
from dual_branch_fusion import predict_fusion, train_fusion
from structural_entropy import load_similarity_matrix
METRICS = ('AUC', 'AUPR', 'ACC', 'F1-Score', 'Sensitivity', 'Precision')
STAGE_SEEDS = {'gnn': 20000, 'anchor': 10000, 'evidence': 30000, 'fusion': 40000}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda : handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    temp.replace(path)

def tsave(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)

def hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and (not p.name.endswith('.tmp'))}

def source_hashes():
    return {p.name: sha(p) for p in sorted(config.PROJECT_ROOT.glob('*.py'))}

def seed_stage(fold, stage):
    seed = config.CONFIG.graph.seed + 100 * fold + STAGE_SEEDS[stage]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not handle.read(1):
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def check_data(data_root):
    (names, sizes) = ({}, {})
    for node in ('rna', 'drug', 'disease'):
        (matrix, names[node]) = load_similarity_matrix(str(data_root / config.DATA_FILES[node + '_sim']))
        sizes[node] = list(matrix.shape)
    (features, _) = data.load_node_features(str(data_root), names)
    positives = data._load_rna_drug_interactions(str(data_root / config.DATA_FILES['rna_drug']), names['rna'], names['drug'])
    for (kind, left) in (('rna_disease', 'rna'), ('drug_disease', 'drug')):
        data._load_relation_interactions(
            str(data_root / config.DATA_FILES[kind]),
            names[left],
            names['disease'],
            row_candidates=('mirna', 'rna', 'miRNA', 'drug', 'Drug'),
            col_candidates=('disease', 'Disease')
        )
    return dict(
        similarity_shapes=sizes,
        feature_shapes={k: list(v.shape) for (k, v) in features.items()},
        positive_pairs=int((positives > 0).sum()),
        data_sha256=hashes(data_root / 'data')
    )

def stage(directory, name, inputs, action, outputs):
    marker = directory / (name + '_complete.json')
    if marker.exists():
        value = read(marker)
        if value['inputs'] != inputs:
            raise ValueError(f'Changed dependency: {marker}')
        for (relative, expected) in value['outputs'].items():
            if sha(directory / relative) != expected:
                raise ValueError(f'Changed output: {directory / relative}')
        print(f'[verified] {directory.name}/{name}', flush=True)
        return
    print(f'[start] {directory.name}/{name}', flush=True)
    action()
    artifacts = {}
    for relative in outputs:
        path = directory / relative
        if path.is_dir():
            artifacts.update({str(Path(relative) / k): v for (k, v) in hashes(path).items()})
        else:
            artifacts[relative] = sha(path)
    if not artifacts:
        raise RuntimeError(f'No output from {name}')
    save(marker, dict(inputs=inputs, outputs=artifacts))
    release()

def latest_lora_checkpoint(directory):
    valid = []
    for path in (directory / 'lora_adapter').glob('checkpoint-*'):
        marker = path / 'resume_ready.json'
        if marker.is_file():
            value = read(marker)
            if all(((path / k).is_file() and (path / k).stat().st_size == size for (k, size) in value['files'].items())):
                valid.append((int(value['step']), str(path)))
    return max(valid)[1] if valid else None

def embeddings(root, fold, graph):
    h = config.CONFIG.rag
    return llm.attach_disease_mediator_context(
        llm.load_llm_embeddings(root, fold),
        graph,
        max_module_changes=h.max_module_changes,
        module_penalty_floor=h.module_penalty_floor,
        disease_penalty_floor=h.disease_penalty_floor,
        degree_penalty_floor=h.degree_penalty_floor
    )

def load_gnn(directory, graph, args, device):
    (features, edges) = gnn.graph_tensors(graph, device, args.similarity_edge_scale)
    curvature = gnn.topology_curvature_prior(graph, device)
    model = gnn.RiemannianMixtureGNN(
        input_dims={k: v.shape[1] for (k, v) in features.items()},
        relations=tuple(edges),
        hidden_dim=args.gnn_hidden_dim,
        encoder_width=args.gnn_encoder_width,
        num_layers=args.gnn_num_layers,
        dropout=args.gnn_dropout,
        curvature_threshold=args.curvature_threshold
    ).to(device)
    checkpoint = llm.torch_load_cpu(directory / 'gnn.pt')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.se_calibration = checkpoint['calibration']
    return (model.eval().requires_grad_(False), features, edges, curvature, graph.se_pair_prior.to(device))

@torch.no_grad()
def gnn_values(runtime, pairs, args, device):
    (model, features, edges, curvature, prior) = runtime
    return dict(
        logits=gnn.apply_se_calibration(
            model,
            gnn.predict_gnn(
                model,
                features,
                edges,
                pairs,
                curvature,
                device,
                pair_prior=prior,
                se_prior_weight=args.se_prior_weight
            )
        ),
        features=gnn.pair_features(model, features, edges, curvature, pairs)
    )

def save_head(path, model):
    tsave(
        path,
        dict(
            state_dict={k: v.detach().cpu() for (k, v) in model.state_dict().items()},
            training_summary=model.training_summary
        )
    )

def load_head(directory, kind, dims, args, device):
    if kind == 'anchor':
        model = llm.LLMBranch(dims, args.llm_node_dim, args.llm_hidden_dim, args.llm_dropout)
    else:
        model = llm.EvidenceAwareLLMBranch(
            dims,
            args.llm_node_dim,
            args.llm_hidden_dim,
            args.llm_dropout,
            args.llm_bridge_dim,
            args.llm_se_path_dim,
            args.llm_se_path_residual_weight
        )
    model.load_state_dict(llm.torch_load_cpu(directory / (kind + '.pt'))['state_dict'], strict=True)
    return model.to(device).eval()

def evidence_values(heads, emb, pairs, args, device):
    values = {}
    for (kind, model) in heads.items():
        values[kind + '_logits'] = llm.predict_llm(model, emb, pairs, args.batch_size, device)
        values[kind + '_features'] = llm.llm_pair_features(model, emb, pairs, args.batch_size, device)
    return values

def fused_logits(model, geometry, values, args, device):
    return predict_fusion(
        model,
        geometry['features'],
        values['evidence_features'],
        geometry['logits'],
        values['evidence_logits'],
        args.batch_size,
        device,
        anchor_llm_logits=values['anchor_logits'],
        anchor_llm_features=values['anchor_features']
    )

def train_fold(root, fold, device):
    (directory, graph_dir) = (root / f'fold_{fold}', root / 'graphs' / f'fold_{fold}')
    directory.mkdir(parents=True, exist_ok=True)
    args = config.classification_arguments()
    graph = data.load_hetero_graph(str(graph_dir / 'final_train_graph.pt'))
    full_split = data.load_split(graph_dir / 'split_data.pt', graph)
    split = {k: v for (k, v) in full_split.items() if k.startswith(('train_', 'val_'))}
    dependency = dict(
        code=source_hashes(),
        graph=sha(graph_dir / 'final_train_graph.pt'),
        split=sha(graph_dir / 'split_data.pt'),
        manifest=sha(root / 'run_manifest.json')
    )
    stage(
        directory,
        'rag',
        dependency,
        lambda : rag.build_corpus(str(graph_dir), str(directory / 'rag'), **asdict(config.CONFIG.rag)),
        ['rag']
    )
    stage(
        directory,
        'corpus',
        dict(rag=sha(directory / 'rag_complete.json')),
        lambda : llm.build_lm_corpus(
            argparse.Namespace(fold_dir=str(graph_dir), rag_dir=str(directory / 'rag'), out_dir=str(directory / 'corpus')),
            fold
        ),
        ['corpus']
    )
    base_model = Path(read(root / 'run_manifest.json')['base_model'])
    stage(
        directory,
        'lora',
        dict(corpus=sha(directory / 'corpus_complete.json')),
        lambda : llm.train_lora(config.lora_arguments(fold, graph_dir, directory, base_model, latest_lora_checkpoint(directory), device=str(device))),
        [
            'lora_adapter/adapter_model.safetensors',
            'lora_adapter/adapter_config.json',
            'lora_adapter/tokenizer.json',
            'lora_adapter/train_metadata.json',
            'lora_adapter/trained_parameters.json'
        ]
    )
    stage(
        directory,
        'hidden',
        dict(lora=sha(directory / 'lora_complete.json')),
        lambda : llm.extract_hidden(config.extraction_arguments(graph_dir, directory, base_model, device=str(device))),
        ['llm_hidden.pt', 'llm_metadata.json']
    )

    def fit_gnn():
        seed_stage(fold, 'gnn')
        (model, *_) = gnn.train_gnn_branch(args, graph, split, device, config.binary_metrics)
        tsave(
            directory / 'gnn.pt',
            dict(
                state_dict={k: v.detach().cpu() for (k, v) in model.state_dict().items()},
                calibration=model.se_calibration,
                training_summary=model.training_summary
            )
        )
    stage(directory, 'gnn', dependency, fit_gnn, ['gnn.pt'])
    emb = embeddings(root, fold, graph)
    for (kind, trainer) in (('anchor', llm.train_semantic_anchor), ('evidence', llm.train_llm_branch)):

        def fit_head(kind=kind, trainer=trainer):
            seed_stage(fold, kind)
            save_head(directory / (kind + '.pt'), trainer(args, emb, split, device, config.binary_metrics))
        stage(
            directory,
            kind,
            dict(hidden=sha(directory / 'hidden_complete.json'), **dependency),
            fit_head,
            [kind + '.pt']
        )

    def fit_fusion():
        seed_stage(fold, 'fusion')
        runtime = load_gnn(directory, graph, args, device)
        heads = {k: load_head(directory, k, emb['llm']['rna'].shape[1], args, device) for k in ('anchor', 'evidence')}
        (geometry, values) = ({}, {})
        for name in ('train', 'val'):
            pairs = split[name + '_pairs']
            geometry[name] = gnn_values(runtime, pairs, args, device)
            values[name] = evidence_values(heads, emb, pairs, args, device)
        (tr, va) = (values['train'], values['val'])
        (model, threshold) = train_fusion(
            args,
            geometry['train']['features'],
            tr['evidence_features'],
            geometry['train']['logits'],
            tr['evidence_logits'],
            split['train_labels'],
            geometry['val']['features'],
            va['evidence_features'],
            geometry['val']['logits'],
            va['evidence_logits'],
            split['val_labels'],
            device,
            config.binary_metrics,
            train_anchor_logits=tr['anchor_logits'],
            val_anchor_logits=va['anchor_logits'],
            train_anchor=tr['anchor_features'],
            val_anchor=va['anchor_features']
        )
        tsave(
            directory / 'fusion.pt',
            dict(model=model, threshold=threshold, training_summary=model.training_summary)
        )
        logits = fused_logits(model, geometry['val'], va, args, device).cpu()
        tsave(
            directory / 'validation_predictions.pt',
            dict(pairs=split['val_pairs'], labels=split['val_labels'], logits=logits)
        )
    stage(
        directory,
        'selection',
        dict(
            **dependency,
            **{k: sha(directory / (k + '_complete.json')) for k in ('gnn', 'anchor', 'evidence')}
        ),
        fit_fusion,
        ['fusion.pt', 'validation_predictions.pt']
    )
    release()

def evaluate_fold(root, fold, device):
    locked = read(root / 'selection_lock.json')
    if set(locked) != {str(f) for f in range(1, 6)}:
        raise RuntimeError('All five selections must be locked before any test evaluation')
    for (selected_fold, expected) in locked.items():
        if sha(root / f'fold_{selected_fold}' / 'selection_complete.json') != expected:
            raise ValueError('A locked selection changed')
    (directory, graph_dir) = (root / f'fold_{fold}', root / 'graphs' / f'fold_{fold}')
    args = config.classification_arguments()

    def action():
        graph = data.load_hetero_graph(str(graph_dir / 'final_train_graph.pt'))
        split = data.load_split(graph_dir / 'split_data.pt', graph)
        emb = embeddings(root, fold, graph)
        heads = {k: load_head(directory, k, emb['llm']['rna'].shape[1], args, device) for k in ('anchor', 'evidence')}
        geometry = gnn_values(load_gnn(directory, graph, args, device), split['test_pairs'], args, device)
        values = evidence_values(heads, emb, split['test_pairs'], args, device)
        fusion = llm.torch_load_cpu(directory / 'fusion.pt')
        assert fusion['threshold'] == 0.5
        logits = fused_logits(fusion['model'], geometry, values, args, device).cpu()
        probabilities = torch.sigmoid(logits)
        tsave(
            directory / 'test_predictions.pt',
            dict(pairs=split['test_pairs'], labels=split['test_labels'], logits=logits,
                 probabilities=probabilities, predictions=(probabilities >= 0.5).long(), threshold=0.5)
        )
    stage(
        directory,
        'test',
        dict(selection_lock=sha(root / 'selection_lock.json')),
        action,
        ['test_predictions.pt']
    )

def export_metrics(root):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
    rows = []
    for fold in range(1, 6):
        pred = llm.torch_load_cpu(root / f'fold_{fold}' / 'test_predictions.pt')
        split = data.load_split(root / 'graphs' / f'fold_{fold}' / 'split_data.pt')
        assert torch.equal(pred['pairs'], split['test_pairs'])
        assert torch.equal(pred['labels'], split['test_labels'])
        (scores, labels) = (torch.sigmoid(pred['logits']).numpy(), pred['labels'].numpy().astype(int))
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite test probabilities')
        predicted = scores >= 0.5
        (precision, recall, _) = precision_recall_curve(labels, scores, drop_intermediate=False)
        metrics = dict(zip(
            METRICS,
            map(
                float,
                (
                    roc_auc_score(labels, scores),
                    auc(recall, precision),
                    accuracy_score(labels, predicted),
                    f1_score(labels, predicted, zero_division=0),
                    recall_score(labels, predicted, zero_division=0),
                    precision_score(labels, predicted, zero_division=0)
                )
            )
        ))
        expected = config.binary_metrics(pred['logits'], pred['labels'])
        assert all((abs(metrics[k] - expected[k]) < 1e-10 for k in METRICS))
        rows.append(dict(fold=fold, threshold=0.5, **metrics))
    summary = [dict(
        metric=k,
        mean=float(np.mean([r[k] for r in rows])),
        std=float(np.std([r[k] for r in rows], ddof=1)),
        folds=5,
        ddof=1,
        threshold=0.5
    ) for k in METRICS]
    for (name, values) in (('fold_metrics.csv', rows), ('summary_metrics.csv', summary)):
        with (root / name).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

def train(options):
    root = options.output.resolve()
    if root == config.PROJECT_ROOT or config.PROJECT_ROOT.is_relative_to(root):
        raise ValueError('Use a dedicated run output directory')
    if root.exists() and any(root.iterdir()) and (not options.resume):
        raise FileExistsError('Existing run preserved. Use --resume to verify and continue it.')
    if not options.base_model.is_dir():
        raise FileNotFoundError(options.base_model)
    if options.device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is unavailable')
    device = torch.device(options.device)
    torch.set_num_threads(4)
    llm.configure_torch_determinism(True)
    with run_lock(root), threadpool_limits(limits=4):
        preflight = check_data(options.data_root)
        binding = dict(
            model_name=config.MODEL_NAME,
            recipe=asdict(config.CONFIG),
            code=source_hashes(),
            data=preflight['data_sha256'],
            data_root=str(options.data_root),
            base_model=str(options.base_model),
            base_model_sha256=hashes(options.base_model),
            metric_policy=config.METRIC_POLICY,
            device=options.device,
            fresh_gnn=True
        )
        manifest = root / 'run_manifest.json'
        if manifest.exists() and read(manifest) != binding:
            raise ValueError('Run code, data, model or recipe changed; use a fresh output directory')
        save(manifest, binding)
        try:
            stage(
                root,
                'graphs',
                dict(manifest=sha(manifest)),
                lambda : data.prepare_and_save_five_fold_graphs(data_dir=str(options.data_root), save_dir=str(root / 'graphs')),
                ['graphs']
            )
            for fold in range(1, 6):
                save(root / 'status.json', dict(stage='training', fold=fold))
                train_fold(root, fold, device)
            locked = {str(f): sha(root / f'fold_{f}' / 'selection_complete.json') for f in range(1, 6)}
            lock = root / 'selection_lock.json'
            if lock.exists() and read(lock) != locked:
                raise ValueError('Locked selections changed')
            save(lock, locked)
            for fold in range(1, 6):
                save(root / 'status.json', dict(stage='testing', fold=fold))
                evaluate_fold(root, fold, device)
            export_metrics(root)
            save(root / 'status.json', dict(stage='complete'))
        except BaseException as exc:
            save(
                root / 'status.json',
                dict(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
            )
            raise

def main():
    parser = argparse.ArgumentParser(description=f'{config.MODEL_NAME}: complete five-fold training and prediction')
    parser.add_argument('--data-root', type=Path, default=config.DATA_ROOT,
                        help='Directory containing data/, relative to the code directory')
    parser.add_argument('--base-model', type=Path, default=config.LLM_BASE_MODEL)
    parser.add_argument('--output', type=Path, default=config.OUTPUT_ROOT,
                        help='Output directory, relative to the code directory (default: outputs)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true', help='Verify and resume completed stages')
    parser.add_argument('--check-data', action='store_true', help='Validate data/features without training')
    parser.add_argument('--metrics-only', action='store_true', help='Recompute metric CSVs from saved predictions')
    options = parser.parse_args()
    options.data_root = config.project_path(options.data_root)
    options.output = config.project_path(options.output)
    options.base_model = config.project_path(options.base_model)
    if options.check_data:
        print(json.dumps(check_data(options.data_root), ensure_ascii=False, indent=2))
    elif options.metrics_only:
        export_metrics(options.output.resolve())
    else:
        train(options)
if __name__ == '__main__':
    main()
