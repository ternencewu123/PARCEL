# System / Python
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# PyTorch
import torch
from torch.utils.data.dataloader import DataLoader
# Custom
from fastmri_dataset import FastMRIData as Dataset
from net import ParallelNetwork as Network
from mri_tools import *
from utils import real2complex

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to model
parser.add_argument('--num-layers', type=int, default=5, help='number of iterations')
parser.add_argument('--in-channels', type=int, default=2, help='number of model input channels')
parser.add_argument('--out-channels', type=int, default=2, help='number of model output channels')
# batch size, num workers
parser.add_argument('--batch-size', type=int, default=4, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
# parameters related to test data
parser.add_argument('--test-path', type=str, default='/home/sswang/data/fastmri_multicoil2/test', help='path of validation data')
parser.add_argument('--u-mask-path', type=str, default='./mask1/undersampling_mask/mask_3.00x_acs24.mat', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask1/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask1/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=1.0, help='sampling rate of test data')
# others
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--results-save-path', type=str, default='./results/CL/', help='save path of reconstruction results')
parser.add_argument('--save-results', '-sr', type=bool, default=True, help='whether save results')


def save_results(name, under_slice, output_slice, label_slice, psnr_zerof, ssim_zerof, psnr, ssim):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    diff_img = label_slice - output_slice
    norm = colors.Normalize(vmin=0.0, vmax=0.15)

    plt.ion()
    # plt.figure(10)
    plt.subplot(221)
    plt.imshow(label_slice, cmap='gray')
    plt.axis('off')
    plt.title('full_img')
    plt.subplot(222)
    plt.imshow(under_slice, cmap='gray')
    plt.axis('off')
    plt.title('under_img, psnr:{:.5f}, ssim:{:.5f}'.format(psnr_zerof, ssim_zerof))
    plt.subplot(223)
    plt.imshow(output_slice, cmap='gray')
    plt.axis('off')
    plt.title('infer_img, psnr:{:.5f}, ssim:{:.5f}'.format(psnr, ssim))
    plt.subplot(224)
    plt.imshow(diff_img, norm=norm)
    plt.colorbar()
    plt.axis('off')
    plt.title('diff_img')
    plt.savefig(name)
    # plt.pause(0.1)
    plt.close()


def validate(args):
    torch.cuda.set_device(0)

    test_set = Dataset(args.test_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = Network(in_channels=args.in_channels, out_channels=args.out_channels, num_layers=args.num_layers, rank=0)
    # load checkpoint
    model_path = os.path.join(args.model_save_path, 'best_checkpoint_cl_3x(1).pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(0))
    model.load_state_dict(checkpoint['model'])
    print('The model is loaded.')
    model = model.cuda(0)

    print('Now testing {}.'.format(args.exp_name))
    model.eval()
    with torch.no_grad():
        mean_psnr, std_psnr, mean_ssim, std_ssim, mean_psnr_zerof, std_psnr_zerof, \
        mean_ssim_zerof, std_ssim_zerof, average_time, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        t = tqdm(test_loader, desc='testing', total=int(len(test_loader)))
        batch_psnr, batch_ssim, batch_psnr_zerof, batch_ssim_zerof = [], [], [], []
        for iter_num, data_batch in enumerate(t):
            full_kspace = data_batch[0].to(0, non_blocking=True)
            csm = data_batch[1].to(0, non_blocking=True)
            mask_under = data_batch[2].to(0, non_blocking=True)

            fname = data_batch[5]
            slice_id = data_batch[6]

            label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

            under_img = At(full_kspace, csm, mask_under)
            net_under_img = torch.view_as_real(under_img).permute(0, 3, 1, 2).contiguous()
            # inference
            start_time = time.time()
            output_up, output_down = model(net_under_img, mask_under, net_under_img, mask_under, csm)

            output_up = torch.view_as_complex(output_up.permute(0, 2, 3, 1).contiguous())
            output_down = torch.view_as_complex(output_down.permute(0, 2, 3, 1).contiguous())
            output = (output_up+output_down)/2.0

            infer_time = time.time() - start_time
            average_time += infer_time
            # calculate and print test information
            under_img, output, label = under_img.detach().cpu().numpy(), output.detach().cpu().numpy(), label.detach().cpu().numpy()

            total_num += under_img.shape[0]

            for i in range(under_img.shape[0]):
                name = args.results_save_path + fname[i] + '_' + str(slice_id[i].item()) + '_3x(1).png'
                under_slice, output_slice, label_slice = np.abs(under_img[i]), np.abs(output[i]), np.abs(label[i])
                psnr = peak_signal_noise_ratio(label_slice, output_slice, data_range=label_slice.max())
                psnr_zerof = peak_signal_noise_ratio(label_slice, under_slice, data_range=label_slice.max())
                ssim = structural_similarity(label_slice, output_slice, data_range=label_slice.max())
                ssim_zerof = structural_similarity(label_slice, under_slice, data_range=label_slice.max())
                if args.save_results:
                    if not os.path.exists(args.results_save_path):
                        os.makedirs(args.results_save_path)
                    # save_results(name, under_slice, output_slice, label_slice, psnr_zerof, ssim_zerof, psnr, ssim)
                batch_psnr.append(psnr)
                batch_ssim.append(ssim)
                batch_psnr_zerof.append(psnr_zerof)
                batch_ssim_zerof.append(ssim_zerof)

        # np.savetxt('./test/dc_psnr_3x(1).txt', batch_psnr, fmt='%.5f', delimiter=" ")
        # np.savetxt('./test/dc_ssim_3x(1).txt', batch_ssim, fmt='%.5f', delimiter=" ")

        mean_psnr, std_psnr = np.mean(batch_psnr), np.std(batch_psnr, ddof=1)
        mean_ssim, std_ssim = np.mean(batch_ssim), np.std(batch_ssim, ddof=1)
        mean_psnr_zerof, std_psnr_zerof = np.mean(batch_psnr_zerof), np.std(batch_psnr_zerof, ddof=1)
        mean_ssim_zerof, std_ssim_zerof = np.mean(batch_ssim_zerof), np.std(batch_ssim_zerof, ddof=1)
        average_time /= total_num

    print('average_time:{:.5f}s\tmean_zerof_psnr:{:.5f}\tstd_zerof_psnr:{:.5f}\tmean_zerof_ssim:{:.5f}\tstd_zerof_ssim:{:.5f}'
          '\tmean_test_psnr:{:.5f}\tstd_test_psnr:{:.5f}\tmean_test_ssim:{:.5f}\tstd_test_ssim:{:.5f}'.format(
        average_time, mean_psnr_zerof, std_psnr_zerof, mean_ssim_zerof, std_ssim_zerof,
        mean_psnr, std_psnr, mean_ssim, std_ssim))


if __name__ == '__main__':
    args_ = parser.parse_args()
    validate(args_)
