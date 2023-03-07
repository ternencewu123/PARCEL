import torch
import torch.fft


def fft2(data):
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def ifft2(data):
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def A(data, csm, mask):
    data = data[:, None, ...] * csm
    data = fft2(data)
    data = data * mask[:, None, ...]
    return data


def At(data, csm, mask):
    data = data * mask[:, None, ...]
    data = ifft2(data)
    data = torch.sum(data * torch.conj(csm), dim=1)
    return data


def AtA(data, csm, mask):
    data = data[:, None, ...] * csm
    data = fft2(data)
    data = data * mask[:, None, ...]
    data = ifft2(data)
    data = torch.sum(data * torch.conj(csm), dim=1)
    return data

