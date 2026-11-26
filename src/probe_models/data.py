"""Read only frozen source features; recover step order from locked metadata."""
import hashlib
import json
import math
from pathlib import Path
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'features'
MODEL_CONFIG = Path(__file__).resolve().parents[1] / 'inputs/model_config.json'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def build_lock():
    files = [SOURCE / 'inputs/protocol.json', SOURCE / 'inputs/group_splits.json',
             SOURCE / 'features/COMPLETE.json', MODEL_CONFIG,
             SOURCE / 'operations/deep_probe/projection.pt']
    for task in (1, 2, 3):
        files += [SOURCE / 'features' / f'task{task}.pt', SOURCE / 'features' / f'task{task}.splits.json',
                  SOURCE / 'collection' / f'task{task}' / 'probe-collection-run.json']
        files += [SOURCE / 'collection' / f'task{task}' / f'round-{r:03d}' / 'index.json' for r in range(40)]
    return {str(p): sha(p) for p in files}


def load_data(device='cpu'):
    lock = json.loads((ROOT / 'inputs/LOCK.json').read_text())['files']
    for path, digest in lock.items():
        if sha(path) != digest:
            raise RuntimeError('Input changed: ' + path)
    cfg = json.loads(MODEL_CONFIG.read_text())
    assert (cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads']) == (4096, 32, 8)
    frozen = json.loads((SOURCE / 'features/COMPLETE.json').read_text())['tasks']
    output, audit, all_ids = {}, {}, set()
    for task in (1, 2, 3):
        path = SOURCE / 'features' / f'task{task}.pt'
        assert sha(path) == frozen[str(task)]['features_sha256']
        raw = torch.load(path, map_location='cpu', weights_only=False)
        split_audit = json.loads((SOURCE / 'features' / f'task{task}.splits.json').read_text())
        meta = {}
        for r in range(40):
            ip = SOURCE / 'collection' / f'task{task}' / f'round-{r:03d}' / 'index.json'
            assert sha(ip) == frozen[str(task)]['index_hashes'][str(ip)]
            for entry in json.loads(ip.read_text())['records']:
                m = entry['metadata']
                if m['record_id'] in meta:
                    raise ValueError('Duplicate record')
                meta[m['record_id']] = m
        assert len(meta) == len(raw['record_ids']) == len(set(raw['record_ids'])) == 1600
        assert not all_ids.intersection(raw['record_ids'])
        all_ids.update(raw['record_ids'])
        ids_to_index = {rid: i for i, rid in enumerate(raw['record_ids'])}
        groups = {}
        for rid in raw['record_ids']:
            m = meta[rid]
            i = ids_to_index[rid]
            assert tuple(raw['groups'][i]) == (m['round_index'], m['client_id'])
            assert int(raw['labels'][i]) == m['label']
            groups.setdefault((m['round_index'], m['client_id']), []).append(rid)
        x = raw['features'].float()
        assert x.shape == (1600, 40960) and torch.isfinite(x).all()
        assert torch.allclose(x.norm(dim=1), torch.ones(1600), atol=1e-5)
        spec = next(iter(meta.values()))['lora_spec']
        assert [spec['tensors'][k]['shape'] for k in spec['ordered_keys']] == [[4096, 8], [1024, 8]]
        assert raw['feature_metadata']['ordered_keys'] == spec['ordered_keys']
        output[task], audit[task], used = {}, {}, set()
        for split in ('train', 'validation', 'test'):
            selected = set(split_audit['split_record_ids'][split])
            assert len(selected) == len(split_audit['split_record_ids'][split])
            assert not used.intersection(selected)
            used |= selected
            indices, record_ids, group_ids, labels = [], [], [], []
            for (r, c), rids in sorted(groups.items()):
                overlap = selected.intersection(rids)
                if not overlap:
                    continue
                assert len(overlap) == len(rids) == 10
                ordered = sorted(rids, key=lambda rid: meta[rid]['step_index'])
                assert [meta[rid]['step_index'] for rid in ordered] == list(range(10))
                assert len({meta[rid]['label'] for rid in ordered}) == 1
                for j, rid in enumerate(ordered):
                    assert meta[rid]['previous_record_id'] == (ordered[j - 1] if j else None)
                indices.append([ids_to_index[rid] for rid in ordered])
                record_ids.append(ordered)
                group_ids.append(f'task{task}:round{r}:client{c}')
                labels.append(meta[ordered[0]]['label'])
            ix = torch.tensor(indices)
            n_expected = {'train': 112, 'validation': 16, 'test': 32}[split]
            assert len(labels) == n_expected and sum(labels) == n_expected // 2
            raw_x = x[ix].to(device)
            output[task][split] = dict(raw=raw_x, tokens=raw_x.reshape(len(ix),10,40,1024),
                                       y=torch.tensor(labels, dtype=torch.float32, device=device),
                                       ids=record_ids, groups=group_ids)
            audit[task][split] = dict(trajectories=len(labels), records=len(labels) * 10,
                                      benign_trajectories=labels.count(0), malicious_trajectories=labels.count(1),
                                      group_ids=group_ids, record_ids=record_ids)
        assert used == set(raw['record_ids'])
        audit[task]['source_text_overlap_counts'] = split_audit['source_text_overlap_counts']
    write_json(ROOT / 'inputs/data_audit.json', audit)
    torch.save(dict(transform='identity; lossless reshape for Transformer', raw_dimension=40960, tokens=[40,1024], normalization='original global L2, no further scaling'), ROOT/'inputs/transforms.pt')
    return output


def memory_selection(data, through_task, seed, capacity=32):
    """Balanced history reservoir, fixed 32 trajectories across tasks/classes."""
    slots = [(t, y) for t in range(1, through_task + 1) for y in (0, 1)]
    assert capacity % len(slots) == 0
    count = capacity // len(slots)
    selected = []
    for task, label in slots:
        d = data[task]['train']
        candidates = [i for i, y in enumerate(d['y'].cpu().tolist()) if y == label]
        # Stable priorities allow shrinking the older-task allocation as new tasks arrive.
        order = sorted(candidates, key=lambda i: hashlib.sha256(f'{seed}:{d["groups"][i]}'.encode()).hexdigest())
        selected.extend((task, i) for i in order[:count])
    assert len(selected) == capacity
    return selected
