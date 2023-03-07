import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error


def complex2real(data, axis=-1):
    assert type(data) is np.ndarray
    data = np.stack((data.real, data.imag), axis=axis)
    return data


def real2complex(data, axis=-1):
    assert type(data) is np.ndarray
    assert data.shape[axis] == 2
    mid = data.shape[axis] // 2
    data = data[..., 0:mid] + data[..., mid:] * 1j
    return data.squeeze(axis=axis)


def get_cos_similar(v1, v2):
    v1, v2 = torch.abs(v1), torch.abs(v2)
    batch_size = v1.shape[0]
    similar = torch.tensor(0.)
    for i in range(batch_size):
        num = torch.dot(v1[i], v2[i])
        denom = torch.linalg.norm(v1[i] * torch.linalg.norm(v2[i]))
        res = 0.5 + 0.5 * (num/denom) if denom != 0 else 0
        similar = torch.add(similar, res)
    return similar/batch_size


def mse_slice(gt, pred):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    MSE = 0.0
    for i in range(batch_size):
        mse = mean_squared_error(gt[i], pred[i])
        MSE += mse
    return MSE/batch_size


def psnr_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    PSNR = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        psnr = peak_signal_noise_ratio(gt[i], pred[i], data_range=max_val)
        PSNR += psnr

    return PSNR / batch_size


def ssim_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    SSIM = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        ssim = structural_similarity(gt[i], pred[i], data_range=max_val)
        SSIM += ssim
    return SSIM / batch_size

