import os
import cv2
import torch
import numpy as np
from collections import deque, defaultdict
from tensorboardX import SummaryWriter
from termcolor import colored

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True, single_patch=True):
        self.div = div
        self.single_patch = single_patch

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if self.single_patch:
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            else:
                img = torch.from_numpy(pic).permute(0, 3, 1, 2).contiguous()
        else:
            raise NotImplementedError
        return img.float().div(255) if self.div else img.float()

def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

def load_raw_hdr(path):
    _, ext = os.path.splitext(path)
    if 'exr' in ext.lower():
        bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    elif 'hdr' in ext.lower():
        bgr = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else:
        raise NotImplementedError('images with extension of [{}] is not supported'.format(ext))
    return np.clip(bgr, 0., 65536.)

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    coord = make_coord(img.shape[-2:])
    rgb = img.reshape(3, -1).permute(1, 0)
    return coord, rgb

def batchify(model, img, glb_coord, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(img)
        n = coord.shape[1]
        ql = 0
        preds_ldr = []
        preds_hdr = []
        while ql < n:
            qr = min(ql+bsize, n)
            pred_ldr, pred_hdr = model.query(glb_coord[:, ql:qr, :], coord[:, ql: qr, :], cell[:, ql:qr, :])
            preds_ldr.append(pred_ldr)
            preds_hdr.append(pred_hdr)
            ql = qr
    return torch.cat(preds_ldr, dim=1), torch.cat(preds_hdr, dim=1)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, args):
        self.local_rank = args.rank
        if self.local_rank > 0:
            return

        record_dir = args.save_dir
        os.system('mkdir -p {}'.format(record_dir))
        log_dir = record_dir
        if not args.resume:
            print(colored('remove contents of directory %s' % log_dir, 'red'))
            os.system('rm -r %s/*' % log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        self.processor = None

    def update_loss_stats(self, loss_dict):
        if self.local_rank > 0:
            return
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if self.local_rank > 0:
            return
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        if self.local_rank > 0:
            return

        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        image_stats = image_stats if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step, dataformats='HWC')

    def state_dict(self):
        if self.local_rank > 0:
            return
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        if self.local_rank > 0:
            return
        self.step = scalar_dict['step']

    def __str__(self):
        if self.local_rank > 0:
            return
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(args):
    return Recorder(args)
