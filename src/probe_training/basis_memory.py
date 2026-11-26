"""Equal-task/equal-class training activation SVD, fixed at the own preceding Probe.

The compressed objects are protection directions, not LoRA features. No replay,
centering, output sensitivity, validation rows or risk constraint is used.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from torch import nn
from threadpoolctl import threadpool_limits


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_train(payload, audit):
    ids = payload['record_ids']
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate feature record IDs')
    index, owner = {rid: i for i, rid in enumerate(ids)}, {}
    splits = audit['split_record_ids']
    if set(splits) != {'train', 'validation', 'test'} or not splits['train']:
        raise ValueError('Expected nonempty original train/validation/test split')
    for split, members in splits.items():
        if len(set(members)) != len(members):
            raise ValueError('Duplicate split record IDs')
        for rid in members:
            if rid not in index or rid in owner:
                raise ValueError('Missing or overlapping split record ID')
            owner[rid] = split
    if set(owner) != set(ids):
        raise ValueError('Incomplete data split membership')
    group_owner = {}
    for rid, split in owner.items():
        group = tuple(payload['groups'][index[rid]])
        if False and group in group_owner and group_owner[group] != split:
            raise ValueError('Trajectory crosses original data splits')
        group_owner[group] = split
    chosen = list(splits['train'])
    return torch.tensor([index[rid] for rid in chosen]), chosen


def right_singular_basis(factors, cap):
    """Deterministic exact top right singular space, avoiding d-by-d matrices.

    For the wide input layer, exact dual Gram eigendecomposition gives the same
    right singular subspace as a thin SVD, up to numerical roundoff. The small
    hidden layer uses direct thin SVD. Numerically null columns are not padded.
    """
    f = factors.detach().cpu().double().contiguous().numpy()
    if f.ndim != 2 or not len(f) or not np.isfinite(f).all() or cap <= 0:
        raise ValueError('Expected finite nonempty activation factors and positive cap')
    n, d = f.shape
    requested = min(int(cap), n, d)
    with threadpool_limits(limits=2):
        if n < d:
            gram = f @ f.T
            gram = (gram + gram.T) * .5
            wanted = min(n, requested + 1)
            eigenvalues, eigenvectors = scipy.linalg.eigh(gram,
                subset_by_index=(n-wanted, n-1), check_finite=False, driver='evr')
            eigenvalues = eigenvalues[::-1].copy()
            eigenvectors = eigenvectors[:, ::-1].copy()
            tolerance = 64 * np.finfo(np.float64).eps * max(n,d) * max(float(eigenvalues[0]), 0)
            actual = min(requested, int(np.sum(eigenvalues[:requested] > tolerance)))
            initial = ((f.T @ eigenvectors[:, :actual]) /
                np.sqrt(eigenvalues[:actual])[None, :]) if actual else np.empty((d,0))
            if actual:
                u = np.linalg.qr(initial, mode='reduced')[0]
                fu = f @ u
                strengths, rotation = scipy.linalg.eigh(fu.T @ fu, check_finite=False)
                u = u @ rotation[:, ::-1]
                strengths = np.maximum(strengths[::-1], 0)
            else:
                u, strengths = initial, np.empty(0)
            spectrum = np.sqrt(np.maximum(eigenvalues, 0))
            algorithm = 'exact_dual_gram_top_eigh_then_right_vector_qr'
        else:
            _, singular, vh = scipy.linalg.svd(f, full_matrices=False,
                check_finite=False, lapack_driver='gesdd')
            tolerance = max(1e-10, float(singular[0]) * 1e-8)
            actual = min(requested, int(np.sum(singular > tolerance)))
            u, strengths = vh[:actual].T.copy(), singular[:actual] ** 2
            spectrum = singular[:min(len(singular),requested+1)]
            algorithm = 'exact_thin_svd'
        # Fix the otherwise arbitrary sign for reproducible serialization.
        if actual:
            signs = np.sign(u[np.abs(u).argmax(axis=0), np.arange(actual)])
            signs[signs == 0] = 1
            u *= signs[None, :]
        error = float(np.linalg.norm(u.T @ u - np.eye(actual)))
        total = float(np.square(f).sum())
        retained = float(strengths.sum())
    if error > 1e-8:
        raise ValueError('Compressed basis is not orthonormal')
    return torch.from_numpy(u.copy()), dict(dimension=d, factor_rows=n,
        rank_cap=cap, rank=actual, numerical_threshold=float(tolerance),
        retained_activation_energy_fraction=retained/total if total else 1.0,
        singular_values=spectrum.tolist(), retained_eigenvalues=strengths.tolist(),
        orthogonality_error=error, algorithm=algorithm)


@torch.no_grad()
def build_bases(model, sources, rank_caps=(128,32), batch_size=64,
                expected_train_count=1120, expected_class_count=560):
    history=sorted(map(str,sources),key=int)
    if len(history) not in (3,4) or history != [str(i) for i in range(1,len(history)+1)]:
        raise ValueError('Protect all contiguous preceding Task training sets')
    n_history=len(history)
    layers = [(name,module) for name,module in model.named_modules() if isinstance(module,nn.Linear)]
    if (len(layers) != 2 or layers[0][1].out_features != layers[1][1].in_features
            or layers[1][1].out_features != 1 or any(layer.bias is None for _,layer in layers)):
        raise ValueError('Expected biased one-hidden-layer scalar MLP')
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('Historical statistics are built on CPU only')
    modes = [(module,module.training) for module in model.modules()]
    tasks, bins = {}, []
    total = n_history * expected_train_count
    dimensions = [layer.in_features+1 for _,layer in layers]
    factors = [torch.empty(total,dim,dtype=torch.float64) for dim in dimensions]
    offset = 0
    model.eval()
    try:
        for task in history:
            directory = Path(sources[task])
            feature_file, split_file = directory/f'task{task}.pt', directory/f'task{task}.splits.json'
            audit = json.loads(split_file.read_text())
            digest = sha(feature_file)
            if audit['features_sha256'] != digest:
                raise ValueError('Historical feature bytes differ from split audit')
            payload = torch.load(feature_file,map_location='cpu',weights_only=False)
            indices, train_ids = select_train(payload,audit)
            features, labels = payload['features'], payload['labels']
            if (len(indices) != expected_train_count or features.ndim != 2
                    or features.shape[1] != dimensions[0]-1 or len(features) != len(labels)):
                raise ValueError('Historical train quota or input dimension differs')
            selected_labels = labels[indices]
            if not ((selected_labels == 0)|(selected_labels == 1)).all():
                raise ValueError('Non-binary historical training labels')
            info = dict(feature_file=str(feature_file.resolve()),feature_file_sha256=digest,
                split_file=str(split_file.resolve()),split_file_sha256=sha(split_file),
                train_count=len(indices),train_record_ids=train_ids,
                train_record_ids_sha256=hashlib.sha256(json.dumps(train_ids,separators=(',',':')).encode()).hexdigest(),
                class_counts={},class_weights={},row_scales={},protection_split='train')
            for label in (0,1):
                selected = indices[selected_labels == label]
                count = len(selected)
                if count <= 0 or (expected_class_count is not None and count != expected_class_count):
                    raise ValueError('Historical task/class quota differs')
                scale = math.sqrt(1/(2*n_history*count))
                lo = offset
                for start in range(0,count,batch_size):
                    ix = selected[start:start+batch_size]
                    # Exactly the eval-mode preceding Probe activations: no dropout,
                    # no sensitivity Jacobian, no additional normalization.
                    x = features[ix].to(dtype=layers[0][1].weight.dtype)
                    if not torch.isfinite(x).all():
                        raise ValueError('Nonfinite historical training input')
                    captured = {}
                    handles = [module.register_forward_pre_hook(
                        lambda mod,args,name=name: captured.__setitem__(name,args[0].detach()))
                        for name,module in layers]
                    try:
                        model(x)
                    finally:
                        for handle in handles:
                            handle.remove()
                    for j,(name,_) in enumerate(layers):
                        value = captured[name].reshape(len(x),-1).cpu().double()
                        if not torch.isfinite(value).all():
                            raise ValueError('Nonfinite historical activation')
                        h = torch.cat([value,torch.ones(len(value),1,dtype=torch.float64)],1)
                        factors[j][offset:offset+len(x)] = h * scale
                    offset += len(x)
                bins.append(dict(task=task,label=label,start=lo,stop=offset))
                info['class_counts'][str(label)] = count
                info['class_weights'][str(label)] = 1/(2*n_history)
                info['row_scales'][str(label)] = scale
            tasks[task] = info
            del payload, features, labels
    finally:
        for module,mode in modes:
            module.training = mode
    if offset != total:
        raise ValueError('Unexpected historical factor row count')
    bases, statistics = {}, {}
    for j,(name,_) in enumerate(layers):
        u, stat = right_singular_basis(factors[j],int(rank_caps[j]))
        per_bin = {}
        for section in bins:
            f = factors[j][section['start']:section['stop']]
            energy = float(f.square().sum())
            captured = float((f @ u).square().sum())
            per_bin[f"task{section['task']}:class{section['label']}"] = dict(
                rows=len(f),total_weight=1/(2*n_history),
                retained_activation_energy_fraction=captured/energy if energy else 1.0)
        stat['task_class_energy'] = per_bin
        bases[name], statistics[name] = u, stat
        factors[j] = None
    metadata = dict(history_tasks=list(map(int,history)),history_split='train',
        task_class_balance='equal_task_equal_class',total_train_records=total,
        centering=False,logit_sensitivity=False,replay=False,dropout_enabled=False,
        reference='fixed selected own preceding Probe',reference_activation_dtype=str(layers[0][1].weight.dtype),
        compression_dtype='float64',bias_coordinate=1.0,input_dimension=dimensions[0]-1,
        rank_caps=list(rank_caps),basis_ranks=[bases[name].shape[1] for name,_ in layers],
        tasks=tasks,layers=statistics,
        limitations=['Fixed rank may replace historical directions; All preceding Tasks contribute but not every direction survives.',
            'Layer input protection is not exact nonlinear prediction invariance or downstream model safety.'])
    return bases,metadata
