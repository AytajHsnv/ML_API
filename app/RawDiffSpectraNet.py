import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected x with shape (B,L) or (B,1,L), got {tuple(x.shape)}")


def diff_same_length(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for _ in range(order):
        d = torch.diff(x, dim=-1)
        x = F.pad(d, (0, 1))
    return x


class RawDiffSpectralNet(nn.Module):
    """
    Self-contained raw + diff path spectral classifier.

    Input:
        (B, L) or (B, 1, L)

    Output:
        logits: (B, n_classes)
    """

    def __init__(
        self,
        n_classes: int,
        width: int = 32,
        dropout: float = 0.25,
        stem_kernel_size: int = 7,
        inception_kernels=(3, 7, 9),
        use_mean_max_pooling: bool = True,
        with_batch_norm: bool = True,
    ):
        super().__init__()

        self.n_classes = int(n_classes)
        self.width = int(width)
        self.dropout = float(dropout)
        self.stem_kernel_size = int(stem_kernel_size)
        self.inception_kernels = tuple(int(k) for k in inception_kernels)
        self.use_mean_max_pooling = bool(use_mean_max_pooling)
        self.with_batch_norm = bool(with_batch_norm)

        # Stems
        self.raw_stem = self._make_stem(c_in=1, c_out=width, kernel_size=stem_kernel_size, with_batch_norm=False)
        self.diff_stem = self._make_stem(c_in=1, c_out=width, kernel_size=stem_kernel_size, with_batch_norm=False)

        # Raw branch
        self.raw_inc1 = self._make_inception_block(c_in=width, c_out=width * 2, kernels=inception_kernels, dropout=dropout, with_batch_norm=False)
        self.raw_pool1 = nn.MaxPool1d(kernel_size=2)

        self.raw_inc2 = self._make_inception_block(c_in=width * 2, c_out=width * 4, kernels=inception_kernels, dropout=dropout, with_batch_norm=with_batch_norm)
        self.raw_pool2 = nn.MaxPool1d(kernel_size=2)

        self.raw_final = self._make_conv_block(c_in=width * 4, c_out=width * 4, kernel_size=5, dropout=dropout, with_batch_norm=with_batch_norm)

        # Diff branch
        self.diff_inc1 = self._make_inception_block(c_in=width, c_out=width * 2, kernels=inception_kernels, dropout=dropout, with_batch_norm=with_batch_norm)
        self.diff_pool1 = nn.MaxPool1d(kernel_size=2)

        self.diff_inc2 = self._make_inception_block(c_in=width * 2, c_out=width * 4, kernels=inception_kernels, dropout=dropout, with_batch_norm=with_batch_norm)
        self.diff_pool2 = nn.MaxPool1d(kernel_size=2)

        self.diff_final = self._make_conv_block(c_in=width * 4, c_out=width * 4, kernel_size=5, dropout=dropout, with_batch_norm=with_batch_norm)

        # Head
        if self.use_mean_max_pooling:
            branch_feat_dim = width * 4 * 2
        else:
            branch_feat_dim = width * 4

        fused_dim = branch_feat_dim * 2

        self.head = nn.Sequential(
            nn.Linear(fused_dim, width * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 8, width * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, n_classes))

    def _make_stem(self, c_in: int, c_out: int, kernel_size: int, with_batch_norm: bool) -> nn.Sequential:
        layers = [nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=not with_batch_norm)]

        if with_batch_norm:
            layers.append(nn.BatchNorm1d(c_out))
            
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_conv_block(self, c_in: int, c_out: int, kernel_size: int = 5, dropout: float = 0.0, with_batch_norm: bool = True) -> nn.Sequential:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd.")

        layers = [ nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=not with_batch_norm)]

        if with_batch_norm:
            layers.append(nn.BatchNorm1d(c_out))

        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout1d(dropout))

        return nn.Sequential(*layers)

    def _make_inception_block(self, c_in: int, c_out: int, kernels=(3, 7, 9), dropout: float = 0.0, with_batch_norm: bool = True) -> nn.ModuleDict:
        b = c_out // 4
        r = c_out - 4 * b

        b1 = b
        b2 = b
        b3 = b
        b4 = b + r

        block = nn.ModuleDict(
            {
                "p1": nn.Conv1d(
                    c_in,
                    b1,
                    kernel_size=kernels[0],
                    padding=kernels[0] // 2,
                    bias=False,
                ),
                "p2": nn.Conv1d(
                    c_in,
                    b2,
                    kernel_size=kernels[1],
                    padding=kernels[1] // 2,
                    bias=False,
                ),
                "p3": nn.Conv1d(
                    c_in,
                    b3,
                    kernel_size=kernels[2],
                    padding=kernels[2] // 2,
                    bias=False,
                ),
                "p4": nn.Sequential(
                    nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
                    nn.Conv1d(c_in, b4, kernel_size=1, bias=False),
                ),
                "bn": nn.BatchNorm1d(c_out) if with_batch_norm else nn.Identity(),
                "drop": nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            }
        )

        return block


    def _forward_inception_block(self, block: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            [
                block["p1"](x),
                block["p2"](x),
                block["p3"](x),
                block["p4"](x),
            ],
            dim=1,
        )

        y = block["bn"](y)
        y = F.relu(y, inplace=True)
        y = block["drop"](y)

        return y

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mean_max_pooling:
            x_mean = x.mean(dim=-1)
            x_max = x.max(dim=-1).values
            return torch.cat([x_mean, x_max], dim=1)
        return x.mean(dim=-1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)               # (B, 1, L)

        # raw path
        f_raw = self.raw_stem(x)
        f_raw = self._forward_inception_block(self.raw_inc1, f_raw)
        f_raw = self.raw_pool1(f_raw)
        f_raw = self._forward_inception_block(self.raw_inc2, f_raw)
        f_raw = self.raw_pool2(f_raw)
        f_raw = self.raw_final(f_raw)

        # diff path
        f_diff = self.diff_stem(x)
        f_diff = diff_same_length(f_diff)   # (B, 1, L)
        f_diff = self._forward_inception_block(self.diff_inc1, f_diff)
        f_diff = self.diff_pool1(f_diff)
        f_diff = self._forward_inception_block(self.diff_inc2, f_diff)
        f_diff = self.diff_pool2(f_diff)
        f_diff = self.diff_final(f_diff)

        f_raw = self._pool_features(f_raw)
        f_diff = self._pool_features(f_diff)

        f = torch.cat([f_raw, f_diff], dim=1)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(x)
        return self.head(f)
