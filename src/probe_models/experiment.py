"""Train tiny probes only; do not modify federated models or source experiments."""
import copy
import datetime
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from data import ROOT, sha, write_json, load_data, memory_selection
from models import ARCHITECTURES, make_model


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def status(**kwargs):
    write_json(ROOT / 'state/status.json', dict(updated_utc=now(), **kwargs))


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def metric(logits, y):
    logits, y = logits.flatten().double(), y.flatten().int()
    probs = logits.sigmoid()
    pred = probs >= .8
    tp = int((pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return dict(samples=len(y), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp + tn) / len(y), malicious_recall=tp / (tp + fn),
                benign_fpr=fp / (fp + tn), f1=float(f1_score(y.numpy(), pred.numpy(), zero_division=0)),
                auroc=float(roc_auc_score(y.numpy(), logits.numpy())),
                auprc=float(average_precision_score(y.numpy(), logits.numpy())),
                bce=float(F.binary_cross_entropy_with_logits(logits, y.double())), threshold=.8)


@torch.no_grad()
def predict(model, d):
    model.eval()
    output = torch.cat([model(x).cpu() for x in d[model.input_kind].split(16)])
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite predictions')
    return output


@torch.no_grad()
def evaluate(model, data, through_task, destination):
    result, predictions = {}, {}
    for task in range(1, through_task + 1):
        result[str(task)] = {}
        for split in ('train', 'validation', 'test'):
            d = data[task][split]
            logits = predict(model, d)
            y = d['y'].cpu()
            metrics = metric(logits, y[:, None].expand_as(logits))
            metrics['last_step'] = metric(logits[:, -1], y)
            metrics['per_step'] = [metric(logits[:, step], y) for step in range(10)]
            metrics['trajectories'] = len(y)
            result[str(task)][split] = metrics
            predictions[f'{task}:{split}'] = dict(logits=logits, labels=y,
                                                  record_ids=d['ids'], group_ids=d['groups'])
    torch.save(predictions, destination / 'predictions.pt')
    return result


def memory_batch(data, selection, input_kind, generator, device, count=16):
    slots = sorted({(t, int(data[t]['train']['y'][i])) for t, i in selection})
    assert count % len(slots) == 0
    chosen = []
    for task, label in slots:
        pool = [(t, i) for t, i in selection if t == task and int(data[t]['train']['y'][i]) == label]
        draw = torch.randint(len(pool), (count // len(slots),), generator=generator).tolist()
        chosen.extend(pool[j] for j in draw)
    x = torch.stack([data[t]['train'][input_kind][i] for t, i in chosen])
    y = torch.stack([data[t]['train']['y'][i] for t, i in chosen])
    return x, y


def fit(model, data, task, recipe, seed, method, folder, protocol, selection=None):
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed + 1000 * task)
    generator = torch.Generator().manual_seed(seed + 1000 * task)
    train, validation = data[task]['train'], data[task]['validation']
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=.01)
    epochs, patience = protocol['training']['epochs'], protocol['training']['patience']
    best, best_epoch, bad, best_model, best_optimizer = float('inf'), 0, 0, None, None
    best_rng = None
    start, updates, sample_count = time.perf_counter(), 0, 0
    device = next(model.parameters()).device
    with (folder / 'epochs.jsonl').open('w') as log:
        for epoch in range(1, epochs + 1):
            model.train()
            running, batches = 0., 0
            order = torch.randperm(len(train['y']), generator=generator)
            for ix in order.split(16):
                x, y = train[model.input_kind][ix], train['y'][ix]
                if task > 1:
                    if method == 'replay':
                        hx, hy = memory_batch(data, selection, model.input_kind, generator, device)
                    else:
                        # Match replay's batch size/number of scored prefixes using current-only resampling.
                        extra = torch.randint(len(train['y']), (16,), generator=generator)
                        hx, hy = train[model.input_kind][extra], train['y'][extra]
                    x, y = torch.cat([x, hx]), torch.cat([y, hy])
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                losses = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction='none')
                loss = losses.mean(1).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite train loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1; sample_count += len(x) * 10
                running += float(loss.detach()); batches += 1
            v_logits = predict(model, validation)
            val = float(F.binary_cross_entropy_with_logits(v_logits, validation['y'].cpu()[:, None].expand_as(v_logits)))
            if val < best - 1e-4:
                best, best_epoch, bad = val, epoch, 0
                best_model, best_optimizer = cpu_state(model), copy.deepcopy(optimizer.state_dict())
                best_rng = dict(torch=torch.get_rng_state(), generator=generator.get_state(),
                                cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            else:
                bad += 1
            train_eval = float(F.binary_cross_entropy_with_logits(predict(model, train), train['y'].cpu()[:, None].expand(-1,10)))
            log.write(json.dumps(dict(epoch=epoch,train_bce=running/batches,train_eval_bce=train_eval,validation_bce=val,
                                      best_epoch=best_epoch,best_validation_bce=best,updates=updates))+'\n')
            log.flush()
            if epoch == 1 or epoch % 10 == 0:
                status(phase='training', architecture=protocol['_active_arch'], seed=seed, task=task,
                       method=method, epoch=epoch, best_validation_bce=best,
                       completed_fits=protocol['_completed_fits'])
            if bad >= patience:
                break
    torch.save(dict(model_state_dict=cpu_state(model), epoch=epoch, train_eval_bce=train_eval, validation_bce=val), folder/'last_epoch.pt')
    model.load_state_dict(best_model)
    model.eval()
    restored = float(F.binary_cross_entropy_with_logits(predict(model, validation),
                     validation['y'].cpu()[:, None].expand(-1, 10)))
    assert abs(restored - best) < 1e-6
    if device.type == 'cuda':
        torch.cuda.synchronize()
    info = dict(recipe=recipe,epochs_completed=epoch,best_epoch=best_epoch,best_validation_bce=best,
                restored_validation_bce=restored,updates=updates,scored_training_prefixes=sample_count,
                elapsed_seconds=time.perf_counter()-start,selection='current-task validation BCE, min_delta 1e-4; selected checkpoint continues', early_stopping=True, selected_epoch=best_epoch, stop_reason='patience' if bad>=patience else 'epoch_cap', final_train_eval_bce=train_eval, final_validation_bce=val,
                optimizer_between_tasks='reset',training_record_ids=train['ids'])
    if selection:
        info['memory_groups'] = [data[t]['train']['groups'][i] for t, i in selection]
        info['memory_record_ids'] = [data[t]['train']['ids'][i] for t, i in selection]
        info['memory_raw_float32_feature_bytes'] = len(selection)*10*40960*4
        per_record = 40*32 if model.input_kind == 'tokens' else 256
        info['memory_encoded_float32_feature_bytes'] = len(selection)*10*per_record*4
    return info, best_optimizer, best_rng


def save_checkpoint(model, folder, info, optimizer, rng, architecture, seed, task, method, parent=None):
    path = folder / 'probe.pt'
    torch.save(dict(model_state_dict=cpu_state(model),optimizer_state=optimizer,rng_state=rng,
                    architecture=architecture,seed=seed,task=task,method=method,training=info,
                    previous_checkpoint=str(parent) if parent else None,
                    previous_checkpoint_sha256=sha(parent) if parent else None,
                    transforms_sha256=sha(ROOT/'inputs/transforms.pt'),
                    protocol_sha256=sha(ROOT/'inputs/protocol.json')),path)
    return path

