# System / Python
import os
import argparse
import logging
import random
import shutil
import time

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
# PyTorch

import torch.fft
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from net import ParallelNetwork as Network
from fastmri_dataset import FastMRIData as Dataset
from mri_tools import *
from utils import psnr_slice, ssim_slice, get_cos_similar

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default='tcp://localhost:1836', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help='number of gpus per node')
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=False, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=5, help='number of iterations')  # 5
parser.add_argument('--in-channels', type=int, default=2, help='number of model input channels')
parser.add_argument('--out-channels', type=int, default=2, help='number of model output channels')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=20, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=4, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number'
                                                               ' of workers')
parser.add_argument('--warmup-epochs', type=int, default=5, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=200, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='/home/sswang/data/fastmri_multicoil2/train', help='path of training data')
parser.add_argument('--val-path', type=str, default='/home/sswang/data/fastmri_multicoil2/val', help='path of validation data')
parser.add_argument('--test-path', type=str, default='/home/sswang/data/fastmri_multicoil2/test', help='path of test data')
parser.add_argument('--u-mask-path', type=str, default='./mask1/undersampling_mask/mask_3.00x_acs24.mat', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask1/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask1/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.2, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.1, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=1.0, help='sampling rate of test data')
# save path
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--loss-curve-path', type=str, default='./runs/loss_curve/', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class similarity_loss(nn.Module):
    def __init__(self):
        super(similarity_loss, self).__init__()

    def forward(self, x, y):
        s = torch.exp(get_cos_similar(x, y))
        return -torch.log(s/(s+torch.tensor(0.05)))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(320*320, 1024)

    def forward(self, x):
        x = x.view(-1, 320*320)
        x = self.fc1(x)
        x = F.relu(x)
        return x


def forward(mode, rank, model, dataloader, criterion, optimizer, log, mlp):
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under = data_batch[2].to(rank, non_blocking=True)
        mask_net_up = data_batch[3].to(rank, non_blocking=True)
        mask_net_down = data_batch[4].to(rank, non_blocking=True)
        # fname = data_batch[5]
        # slice_id = data_batch[6]

        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

        under_img = At(full_kspace, csm, mask_under)
        under_img = torch.view_as_real(under_img).permute(0, 3, 1, 2).contiguous()

        net_img_up = At(full_kspace, csm, mask_net_up)
        net_img_up = torch.view_as_real(net_img_up).permute(0, 3, 1, 2).contiguous()

        net_img_down = At(full_kspace, csm, mask_net_down)
        net_img_down = torch.view_as_real(net_img_down).permute(0, 3, 1, 2).contiguous()

        if mode == 'test':
            net_img_up = net_img_down = under_img
            mask_net_up = mask_net_down = mask_under

        output_up, output_down = model(net_img_up, mask_net_up, net_img_down, mask_net_down, csm)

        output_up = torch.view_as_complex(output_up.permute(0, 2, 3, 1).contiguous())
        output_down = torch.view_as_complex(output_down.permute(0, 2, 3, 1).contiguous())

        output_up_kspace = fft2(output_up[:, None, ...] * csm)
        output_down_kspace = fft2(output_down[:, None, ...] * csm)

        recon_up_masked = At(output_up_kspace, csm, mask_under)
        recon_down_masked = At(output_down_kspace, csm, mask_under)

        # undersampled calibration loss
        under_img = torch.view_as_complex(under_img.permute(0, 2, 3, 1).contiguous())
        recon_loss_up = criterion(torch.abs(recon_up_masked), torch.abs(under_img))
        recon_loss_down = criterion(torch.abs(recon_down_masked), torch.abs(under_img))

        # data consistency
        e_up_kspace = output_up_kspace * (1-mask_under)[:, None, ...] + full_kspace * mask_under[:, None, ...]
        dc_up_out = torch.sum(ifft2(e_up_kspace) * torch.conj(csm), dim=1)

        e_down_kspace = output_down_kspace * (1-mask_under)[:, None, ...] + full_kspace * mask_under[:, None, ...]
        dc_down_out = torch.sum(ifft2(e_down_kspace) * torch.conj(csm), dim=1)

        # reconstructed calibration loss
        dc_up_loss = criterion(torch.abs(dc_up_out), torch.abs(output_up))
        dc_down_loss = criterion(torch.abs(dc_down_out), torch.abs(output_down))

        # constrast_loss
        cl = similarity_loss()
        s = cl(mlp(torch.abs(output_up)), mlp(torch.abs(output_down)))

        batch_loss = recon_loss_up + recon_loss_down + 0.01 * dc_up_loss + 0.01 * dc_down_loss + 0.1 * s

        f_output = (output_up + output_down)/2.0

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            psnr += psnr_slice(label, f_output)
            ssim += ssim_slice(label, f_output)
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    model = Network(in_channels=args.in_channels, out_channels=args.out_channels, num_layers=args.num_layers, rank=rank)
    mlp = MLP().to('cuda')
    # whether load checkpoint
    if args.pretrained or args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'best_checkpoint_cl_3x(1).pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim = checkpoint['best_ssim']
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    mlp = mlp.to(rank)
    mlp = DDP(mlp, device_ids=[rank])

    # criterion, optimizer, learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr}, {'params': mlp.parameters(), 'lr': args.lr}])
    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=10)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    # test step
    if args.mode == 'test':
        test_set = Dataset(args.test_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        if rank == 0:
            logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, mlp)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return

    # training step
    train_set = Dataset(args.train_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.train_sample_rate)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler
    )
    val_set = Dataset(args.val_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.val_sample_rate)
    val_sampler = DistributedSampler(val_set)
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=(val_sampler is None),
        pin_memory=True, sampler=val_sampler
    )
    if rank == 0:
        logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        logger.info('Now training {}.'.format(args.exp_name))
        writer = SummaryWriter(args.loss_curve_path)
    # loss curve
    epochs = []
    trains = []
    vals = []
    psnr = []
    ssim = []
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()
        model.train()

        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, mlp)
        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, mlp)
        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]
        val_loss = train_log[3]
        val_psnr = train_log[4]
        val_ssim = train_log[5]

        # add loss
        epochs.append(epoch)
        trains.append(train_loss)
        vals.append(val_loss)
        psnr.append(val_psnr)
        ssim.append(val_ssim)

        is_best = val_ssim > best_ssim
        best_ssim = max(val_ssim, best_ssim)
        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)

            # save checkpoint
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim': best_ssim,
                'model': model.module.state_dict()
            }
            if not os.path.exists(args.model_save_path):
                os.makedirs(args.model_save_path)
            model_path = os.path.join(args.model_save_path, 'checkpoint_cl_3x(1).pth.tar')
            best_model_path = os.path.join(args.model_save_path, 'best_checkpoint_cl_3x(1).pth.tar')
            torch.save(checkpoint, model_path)
            if is_best:
                shutil.copy(model_path, best_model_path)
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
            break
    if rank == 0:
        writer.close()
    np.savetxt('./fig/cl_train_loss_3x(1).txt', trains, fmt='%.5f', delimiter=" ")
    np.savetxt('./fig/cl_val_loss_3x(1).txt', vals, fmt='%.5f', delimiter=" ")
    np.savetxt('./fig/cl_val_psnr_3x(1).txt', psnr, fmt='%.5f', delimiter=" ")
    np.savetxt('./fig/cl_val_ssim_3x(1).txt', ssim, fmt='%.5f', delimiter=" ")

    # plot curve
    plt.ion()
    x = range(0, len(epochs))
    plt.subplot(1, 3, 1)
    plt.plot(x, trains)
    plt.plot(x, vals)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 3, 2)
    plt.plot(x, psnr)
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.subplot(1, 3, 3)
    plt.plot(x, ssim)
    plt.xlabel('epoch')
    plt.ylabel('ssim')
    plt.show()
    name = "./fig/"+str(time.time()) + '_cl_3x(1).jpg'
    plt.savefig(name)


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus  # 1*2
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('begin')
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))
    print('end')


if __name__ == '__main__':
    main()
