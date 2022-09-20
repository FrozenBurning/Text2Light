import os
import random
import torch
import torch.nn as nn
import torchvision
import time
import datetime
import numpy as np

from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist

from sritmo.global_sritmo import SRToneMapper
from sritmo.dataset import PatchHDRDataset
from sritmo.scheduler import make_lr_scheduler
from sritmo.loss import get_loss
from sritmo.util import make_recorder, save_model, ToTorchFormatTensor


def main(args):
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.ddp
    ngpus_per_node = torch.cuda.device_count()

    if args.cache:
        data_list = os.listdir(os.path.join(args.dir, 'train'))
        full_list = []
        for fname in data_list:
            full_list.append(os.path.join(args.dir, 'train', fname))
        random.shuffle(full_list)
        total_len = len(full_list)
        args.training_list = full_list[:total_len - total_len % ngpus_per_node]

    if args.ddp:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.ddp:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    print("curent thread rank:",args.rank)

    model = SRToneMapper(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.bs = int(args.bs / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            print("Using DDP!")
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            print("Using DDP with all GPUs!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)


    recorder = make_recorder(args)
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer type of [{}] is not supported'.format(args.optim))
    scheduler = make_lr_scheduler(args, optimizer)

    if args.cache and args.distributed:
        split_len = len(args.training_list) // args.world_size
        train_dataset = PatchHDRDataset(args.training_list[args.rank*split_len: (args.rank+1)*split_len], transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=False, single_patch=True),]), cache='in_memory',mode='train')
    else:
        train_dataset = PatchHDRDataset(args.dir, transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=False, single_patch=True),]), mode='train')
    val_dataset = PatchHDRDataset(args.dir, transform=torchvision.transforms.Compose([ToTorchFormatTensor(div=False, single_patch=False),]), mode='val')

    if args.distributed:
        if args.cache:
            train_sampler = None
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
    
    for ep in range(args.epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        end = time.time()

        # training...
        model.train()
        for i, batch in enumerate(train_loader):
            data_time = time.time() - end
            optimizer.zero_grad()
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()
            hr_ldr_pred, hr_hdr_pred = model(batch)
            hdr_gt = batch["hr_hdr"]
            ldr_gt = batch["hr_ldr"]
            loss_fn = get_loss(args)
            l1loss = nn.L1Loss()
            ldr_recon_loss = l1loss(hr_ldr_pred, ldr_gt)
            hdr_recon_loss = loss_fn(hr_hdr_pred, hdr_gt)

            loss = ldr_recon_loss + hdr_recon_loss
            loss.backward()
            optimizer.step()

            if args.rank == 0:
                recorder.epoch = ep
                recorder.step += 1
                recorder.update_loss_stats({'Loss': loss, 'LDR_L1': ldr_recon_loss, 'HDR_recon': hdr_recon_loss})
                batch_time = time.time() - end
                end = time.time()
                recorder.batch_time.update(batch_time)
                recorder.data_time.update(data_time)

            if args.rank == 0:
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (len(train_loader) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}'])
                training_state = training_state.format(eta_string, str(recorder), lr)
                print(training_state)
                recorder.record('train')
        scheduler.step()

        # validation
        if (ep + 1) % args.val_ep == 0:
            model.eval()
            end = time.time()
            with torch.no_grad():
                all_loss = []
                all_l1_loss = []
                for i, batch in enumerate(val_loader):
                    data_time = time.time() - end
                    for k in batch:
                        if k != 'meta':
                            batch[k] = batch[k].cuda()
                    hr_ldr_pred, hr_hdr_pred = model(batch)
                    hdr_gt = batch["hr_hdr"]
                    ldr_gt = batch["hr_ldr"]
                    loss_fn = get_loss(args)
                    l1loss = nn.L1Loss()
                    ldr_recon_loss = l1loss(hr_ldr_pred, ldr_gt)
                    hdr_recon_loss = loss_fn(hr_hdr_pred, hdr_gt)

                    loss = ldr_recon_loss + hdr_recon_loss
                    all_loss.append(loss.detach().cpu().item())
                    all_l1_loss.append(ldr_recon_loss.detach().cpu().item())
                total_loss_avg = np.mean(all_loss)
                ldr_recon_avg = np.mean(all_l1_loss)
                hdr_recon_avg = total_loss_avg - ldr_recon_avg
                recorder.record('val', loss_stats={'Loss': total_loss_avg, 'LDR_L1': ldr_recon_avg, 'HDR_recon': hdr_recon_avg})

        if (ep + 1) % args.val_ep == 0 and args.rank == 0:
            save_model(model, optimizer, scheduler, recorder, args.save_dir, ep)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='directory path of HDRs.')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the logs and checkpoints.')
    parser.add_argument('--bs', type=int, default=32, help='batch size over all gpus.')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for pytocrh dataloader.')
    parser.add_argument('--distributed', action='store_true', help='enable distributed training.')
    parser.add_argument('--lr', type=float, default=7e-5, help='base learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--val_ep', type=int, default=5, help='evaluate per #val_ep')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer type.')
    parser.add_argument('--scheduler', type=str, default='exponential', help='learning rate scheduler.')
    parser.add_argument('--loss', type=str, default='simse', help='type of loss for HDR reconstruction.')
    parser.add_argument('--cache', action='store_true', help='deprecated.')
    parser.add_argument('--gpu', type=int, help='the gpu to use when training. ignored when using DDP.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:8889')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--training_list', type=list, default=None)

    args = parser.parse_args()
    main(args)