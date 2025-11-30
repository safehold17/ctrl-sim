import torch
import torch.nn as nn

from utils.train_utils import weight_init


class ResidualMLP(nn.Module):
    """
    Residual feed-forward block with configurable depth.

    Args:
        input_dim: size of the input features.
        hidden_dim: size of the hidden/residual features.
        n_hidden: number of Linear -> LayerNorm blocks before the residual add.
        output_dim: if provided, applies an extra Linear(hidden_dim, output_dim)
                    after the residual + ReLU. If None, returns hidden_dim.
    """

    def __init__(self, input_dim, hidden_dim, n_hidden=2, output_dim=None):
        super().__init__()
        assert n_hidden >= 1, "ResidualMLP needs at least one hidden layer"

        self.linears = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_hidden)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_hidden)])

        self.transform_linear = nn.Linear(input_dim, hidden_dim)
        self.transform_norm = nn.LayerNorm(hidden_dim)

        self.linear_out = nn.Linear(hidden_dim, output_dim) if output_dim is not None else None
        self.relu = nn.ReLU(inplace=True)
        self.apply(weight_init)

    def forward(self, x):
        out = x
        for i, (lin, norm) in enumerate(zip(self.linears, self.norms)):
            out = lin(out)
            out = norm(out)
            if i < len(self.linears) - 1:
                out = self.relu(out)

        res = self.transform_norm(self.transform_linear(x))
        out = self.relu(out + res)

        if self.linear_out is not None:
            out = self.linear_out(out)
        return out


class MLPLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)
