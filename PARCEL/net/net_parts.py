import torch
import torch.nn as nn
from mri_tools import AtA


class Dw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dw, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        return x + self.layers(x)


class ConjugatedGrad(nn.Module):
    def __init__(self):
        super(ConjugatedGrad, self).__init__()

    def forward(self, rhs, csm, mask, lam):
        rhs = torch.view_as_complex(rhs.permute(0, 2, 3, 1).contiguous())
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real
        num_iter, epsilon = 10, 1e-10
        for i in range(num_iter):
            Ap = AtA(p, csm, mask) + lam * p
            alpha = rTr / torch.sum(torch.conj(p) * Ap, dim=(-2, -1)).real
            x = x + alpha[:, None, None] * p
            r = r - alpha[:, None, None] * Ap
            rTrNew = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real
            if rTrNew.max() < epsilon:
                break
            beta = rTrNew / rTr
            rTr = rTrNew
            p = r + beta[:, None, None] * p
        return x
