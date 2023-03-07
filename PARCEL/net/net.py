from net.net_parts import *
import torch.nn as nn


class MoDL(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, rank):
        super(MoDL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rank = rank
        self.layers = Dw(self.in_channels, self.out_channels)
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).cuda(self.rank), requires_grad=True)
        self.CG = ConjugatedGrad()

    def forward(self, under_img, csm, under_mask):
        x = under_img
        for i in range(self.num_layers):
            x = self.layers(x)
            x = under_img + self.lam * x
            x = self.CG(x, csm, under_mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        x_final = x
        return x_final


class ParallelNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, rank):
        super(ParallelNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rank = rank
        self.up_network = MoDL(self.in_channels, self.out_channels, self.num_layers, self.rank)
        self.down_network = MoDL(self.in_channels, self.out_channels, self.num_layers, self.rank)

    def forward(self, under_image_up, mask_up, under_image_down, mask_down, csm):

        output_up = self.up_network(under_image_up, csm, mask_up)
        output_down = self.down_network(under_image_down, csm, mask_down)

        return output_up, output_down


