"""Existing layer-input SVD + final AdamW update projection, adapted to Conv/MHA.

The mathematical rule and rank schedule follow the project's depth-protected run.
LayerNorm, positional parameters and embeddings remain trainable but unprojected.
This is not output-Jacobian projection and makes no nonlinear invariance claim.
"""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F


def blocks(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            assert module._qkv_same_embed_dim and module.batch_first
            result[name + '.in_proj'] = (module, 'in_proj_weight', 'in_proj_bias')
            result[name + '.out_proj'] = (module.out_proj, 'weight', 'bias')
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            if name.endswith('.self_attn.out_proj'):
                continue
            if isinstance(module, nn.Conv1d):
                assert module.groups == 1 and module.padding == (0,)
            result[name] = (module, 'weight', 'bias')
    return result


def packed(block):
    module, wname, bname = block
    w, b = getattr(module, wname), getattr(module, bname)
    assert b is not None
    return torch.cat((w.detach().reshape(w.shape[0], -1), b.detach()[:, None]), 1)


def coverage(model):
    protected = set()
    for module, wname, bname in blocks(model).values():
        protected.update((id(getattr(module, wname)), id(getattr(module, bname))))
    names = {name: p.numel() for name, p in model.named_parameters() if id(p) not in protected}
    return dict(projected_parameters=sum(p.numel() for p in model.parameters() if id(p) in protected),
                unprojected_parameters=names, total_parameters=sum(p.numel() for p in model.parameters()))


def conv_patches(x, module):
    kernel, dilation, stride = module.kernel_size[0], module.dilation[0], module.stride[0]
    windows = x.unfold(2, dilation * (kernel - 1) + 1, stride)[..., ::dilation]
    return windows.permute(0, 2, 1, 3).reshape(-1, x.shape[1] * kernel)


@torch.no_grad()
def attention_context(module, x):
    b, length, width = x.shape
    q, k, v = F.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
    def heads(t):
        return t.reshape(b, length, module.num_heads, width // module.num_heads).transpose(1, 2)
    context = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0.)
    return context.transpose(1, 2).contiguous().reshape(b, length, width)


@torch.no_grad()
def collect_inputs(model, x):
    model.eval()
    captured = {name: [] for name in blocks(model)}
    handles = []
    context_error = 0.
    def capture(name, module, inputs):
        value = inputs[0]
        if isinstance(module, nn.Conv1d):
            value = conv_patches(value, module)
        else:
            value = value.reshape(-1, value.shape[-1])
        captured[name].append(value.detach().cpu().double())
    def attention_hook(name):
        def hook(module, inputs, output):
            nonlocal context_error
            x = inputs[0]
            assert len(inputs) == 3 and inputs[0] is inputs[1] and inputs[1] is inputs[2]
            context = attention_context(module, x)
            recovered = F.linear(context, module.out_proj.weight, module.out_proj.bias)
            torch.testing.assert_close(recovered, output[0], atol=2e-5, rtol=2e-5)
            context_error = max(context_error, float((recovered - output[0]).abs().max()))
            captured[name + '.in_proj'].append(x.detach().reshape(-1, x.shape[-1]).cpu().double())
            captured[name + '.out_proj'].append(context.detach().reshape(-1, context.shape[-1]).cpu().double())
        return hook
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(attention_hook(name)))
        elif isinstance(module, (nn.Linear, nn.Conv1d)) and name in captured:
            handles.append(module.register_forward_pre_hook(lambda m, a, n=name: capture(n, m, a)))
    try:
        for batch in x.split(16):
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert all(captured.values())
    return {name: torch.cat(values) for name, values in captured.items()}, context_error


@torch.no_grad()
def extend_bases(model, train_x, previous, completed_task):
    assert completed_task in (1, 2, 3)
    if completed_task == 3 and set(previous) != set(blocks(model)):
        raise ValueError('Task4 requires the preceding task basis for every affine layer')
    activations, context_error = collect_inputs(model, train_x)
    bases, statistics = {}, {}
    for name, x in activations.items():
        h = torch.cat((x, x.new_ones((len(x), 1))), 1)
        dim = h.shape[1]
        cap = min(128, dim // 2)
        budget = cap // 2 if completed_task == 1 else cap
        old = previous.get(name, h.new_empty((dim, 0))).cpu().double()
        if completed_task == 3:
            # Task4 retains the existing protection span and original rank caps.
            # Do not add Task3 directions or rotate the inherited basis with QR.
            if old.ndim != 2 or old.shape[0] != dim or not 0 < old.shape[1] <= cap:
                raise ValueError('Task4 inherited basis has an invalid shape or rank')
            u = old.clone()
            valid, add = None, 0
        else:
            residual = h - (h @ old) @ old.T
            residual -= (residual @ old) @ old.T
            _, s, vh = torch.linalg.svd(residual, full_matrices=False)
            valid = int((s > max(1e-10, float(s[0]) * 1e-8)).sum())
            add = max(0, min(budget - old.shape[1], valid))
            candidates = vh[:add].T
            candidates -= old @ (old.T @ candidates)
            u = torch.linalg.qr(torch.cat((old, candidates), 1), mode='reduced').Q
        assert u.shape[1] <= cap
        assert torch.allclose(u.T @ u, torch.eye(u.shape[1], dtype=u.dtype), atol=1e-9, rtol=0)
        if old.shape[1]:
            assert float((old - u @ (u.T @ old)).norm()) < 1e-8
        bases[name] = u
        statistics[name] = dict(input_dimension_with_bias=dim, total_rank_cap=cap, stage_budget=budget,
            previous_rank=old.shape[1], added_rank=add, rank=u.shape[1], train_activation_rows=len(h),
            train_activation_energy_fraction=float((h @ u).square().sum() / h.square().sum()),
            residual_numerical_rank=valid)
    return bases, dict(layers=statistics, attention_context_max_error=context_error, parameter_coverage=coverage(model))


class Projector:
    def __init__(self, model, bases, log=None):
        self.blocks = blocks(model)
        assert set(self.blocks) == set(bases)
        self.bases = {name: u.to(device=packed(self.blocks[name]).device, dtype=torch.float64) for name, u in bases.items()}
        self.log = log
        self.steps = 0
        self.max_leakage = 0.
        self.max_relative_leakage = 0.
        self.initial = self.snapshot()

    def snapshot(self):
        return {name: packed(block).double() for name, block in self.blocks.items()}

    @torch.no_grad()
    def project(self, before):
        worst, proposed_norm2, actual_norm2 = 0., 0., 0.
        for name, block in self.blocks.items():
            module, wname, bname = block
            proposed = packed(block).double() - before[name]
            u = self.bases[name]
            actual = proposed - (proposed @ u) @ u.T
            w, b = getattr(module, wname), getattr(module, bname)
            value = before[name] + actual
            # Preserve bitwise identity when the subspace is empty.
            if u.shape[1]:
                w.copy_(value[:, :-1].reshape_as(w)); b.copy_(value[:, -1])
            displacement = packed(block).double() - before[name]
            leakage = float((displacement @ u).norm())
            rounding_bound = 3 * torch.finfo(w.dtype).eps * (float(before[name].norm()) + float(value.norm())) + 1e-10
            if leakage > rounding_bound:
                raise RuntimeError(f'Update projection failed: {name}, {leakage} > {rounding_bound}')
            worst = max(worst, leakage)
            proposed_norm2 += float(proposed.square().sum())
            actual_norm2 += float(displacement.square().sum())
        self.steps += 1
        self.max_leakage = max(self.max_leakage, worst)
        relative = worst / (actual_norm2**.5 + 1e-12)
        self.max_relative_leakage = max(self.max_relative_leakage, relative)
        if self.log:
            self.log(dict(step=self.steps, max_layer_leakage=worst, relative_leakage=relative,
                          proposed_update_norm=proposed_norm2**.5, projected_update_norm=actual_norm2**.5))

    def summary(self):
        drift = {name: float(((packed(block).double() - self.initial[name]) @ self.bases[name]).norm())
                 for name, block in self.blocks.items()}
        return dict(steps=self.steps, maximum_update_leakage=self.max_leakage,
                    maximum_relative_update_leakage=self.max_relative_leakage,
                    restored_checkpoint_layer_drift=drift,
                    arithmetic='float64 projection, original float32 parameters; rounding residual reported')


@contextlib.contextmanager
def projected_adamw(projector):
    """Reuse the exact original trainer; change only the optimizer's final step."""
    base_class = torch.optim.AdamW
    class ProjectedAdamW(base_class):
        def step(self, closure=None):
            before = projector.snapshot()
            result = super().step(closure)
            projector.project(before)
            return result
    torch.optim.AdamW = ProjectedAdamW
    try:
        yield
    finally:
        torch.optim.AdamW = base_class
