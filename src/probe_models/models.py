"""Small LoRA-update classifiers. All forwards return [trajectory, step] logits."""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ('mlp1', 'mlp2', 'causal_cnn', 'transformer')


class TokenTransformer(nn.Module):
    input_kind = 'tokens'

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1024, 64)
        self.position = nn.Parameter(torch.empty(1, 40, 64))
        self.module_type = nn.Embedding(2, 64)
        self.register_buffer('module_ids', torch.tensor([0] * 32 + [1] * 8))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(64, 4, 128, dropout=.1, activation='gelu',
                                       batch_first=True, norm_first=True) for _ in range(2)])
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 1)
        nn.init.normal_(self.position, std=.02)

    def forward(self, x):
        b, steps, tokens, width = x.shape
        z = self.embed(x.reshape(b * steps, tokens, width))
        z = z + self.position + self.module_type(self.module_ids).unsqueeze(0)
        for layer in self.layers:
            z = layer(z)
        return self.head(self.norm(z).mean(1)).reshape(b, steps)


class StepEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(40960, 64), nn.LayerNorm(64), nn.GELU())

    def forward(self, x):
        return self.net(x)


class CausalCNN(nn.Module):
    input_kind = 'raw'

    def __init__(self):
        super().__init__()
        self.encoder = StepEncoder()
        self.convs = nn.ModuleList([nn.Conv1d(64, 64, 3, dilation=d) for d in (1, 2, 4)])
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        z = self.encoder(x).transpose(1, 2)
        for conv in self.convs:
            h = conv(F.pad(z, (2 * conv.dilation[0], 0)))
            z = z + self.dropout(F.gelu(h))
        return self.head(z.transpose(1, 2)).squeeze(-1)


class RawMLP(nn.Module):
    input_kind = 'raw'

    def __init__(self, depth):
        super().__init__()
        widths = [40960, 64] if depth == 1 else [40960, 128, 64]
        modules = []
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            modules.extend([nn.Linear(a, b), nn.GELU()])
            if i == 0:
                modules.append(nn.Dropout(.1))
        modules.append(nn.Linear(widths[-1], 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_model(architecture):
    return {'mlp1': lambda: RawMLP(1), 'mlp2': lambda: RawMLP(2),
            'transformer': TokenTransformer, 'causal_cnn': CausalCNN}[architecture]()


def parameter_breakdown(model):
    # Input = trainable map into first hidden representation; hidden includes
    # later blocks/norms/position embeddings; output = final classifier.
    if isinstance(model, RawMLP):
        input_module, output_module = model.net[0], model.net[-1]
    elif isinstance(model, CausalCNN):
        input_module, output_module = model.encoder.net[0], model.head
    else:
        input_module, output_module = model.embed, model.head
    inputs = sum(p.numel() for p in input_module.parameters())
    outputs = sum(p.numel() for p in output_module.parameters())
    total = sum(p.numel() for p in model.parameters())
    return dict(input_mapping=inputs, hidden_blocks=total-inputs-outputs,
                output_mapping=outputs, total=total,
                named_parameters={n:p.numel() for n,p in model.named_parameters()})
