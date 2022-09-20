import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from argparse import ArgumentParser


def reinhard_tmo(hdr, gamma=1.5, intensity = 0., light_adapt = 0., color_adapt = 0.):
    scale = hdr.max() - hdr.min()
    if hdr.max() - hdr.min() > 1e-6:
        hdr = (hdr - hdr.min()) / (hdr.max() - hdr.min())
    # the input is opencv BGR format
    gray = 0.299*hdr[..., 2] + 0.587*hdr[..., 1] + 0.114*hdr[..., 0]
    log_ = np.log(np.clip(gray, 1e-4, np.inf))
    log_mean = log_.mean()
    log_min = log_.min()
    log_max = log_.max()
    key = (log_max - log_mean) / (log_max - log_min)
    map_key = 0.3 + 0.7*(key**1.4)
    gray_mean = gray.mean()

    intensity = np.exp(-intensity)
    hdr = hdr*(1/(gray_mean**map_key + hdr))
    hdr = hdr**(1/gamma)
    return hdr, scale*(gray_mean**map_key)

def load_raw_hdr(path):
    name, ext = os.path.splitext(path)
    if 'exr' in ext.lower():
        bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    elif 'hdr' in ext.lower():
        bgr = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else:
        raise NotImplementedError('images with extension of [{}] is not supported'.format(ext))
    return np.clip(bgr, 0., 65536.)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='the directory that contains raw HDRIs.')
    parser.add_argument('--dst', type=str, default='./data', help='the directory to store processed data.')
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst
    if not os.path.isdir(src_dir):
        raise NotImplementedError("source directory [{}] is not valid!".format(src_dir))
    
    os.makedirs(dst_dir, exist_ok=True)
    train_dst_dir = os.path.join(dst_dir, 'train')
    val_dst_dir = os.path.join(dst_dir, 'val')
    os.makedirs(train_dst_dir, exist_ok=True)
    os.makedirs(val_dst_dir, exist_ok=True)

    train_ldr_dir = os.path.join(train_dst_dir, 'ldr')
    train_ldr1k_dir = os.path.join(train_dst_dir, 'ldr1k')
    train_hdr_dir = os.path.join(train_dst_dir, 'calib_hdr')
    os.makedirs(train_ldr_dir, exist_ok=True)
    os.makedirs(train_ldr1k_dir, exist_ok=True)
    os.makedirs(train_hdr_dir, exist_ok=True)

    val_ldr_dir = os.path.join(val_dst_dir, 'ldr')
    val_ldr1k_dir = os.path.join(val_dst_dir, 'ldr1k')
    val_hdr_dir = os.path.join(val_dst_dir, 'calib_hdr')
    os.makedirs(val_ldr_dir, exist_ok=True)
    os.makedirs(val_ldr1k_dir, exist_ok=True)
    os.makedirs(val_hdr_dir, exist_ok=True)

    src_file_list = os.listdir(src_dir)
    random.shuffle(src_file_list)
    total_len = len(src_file_list)
    train_file_list = src_file_list[:int(total_len * 0.9)]
    val_file_list = src_file_list[int(total_len * 0.9):]

    # train
    train_1k_list = []
    train_aug_list = []
    for f in tqdm(train_file_list):
        full_path = os.path.join(src_dir, f)
        full_hdr = load_raw_hdr(full_path)
        full_ldr, _ = reinhard_tmo(full_hdr)
        full_ldr[np.isnan(full_ldr)] = 0.
        LDR = full_ldr
        LDR_1k = cv2.resize(full_ldr, (1024, 512), interpolation=cv2.INTER_LANCZOS4)
        LDR_I = np.mean(LDR, axis=2)
        mask = np.where(LDR_I > 0.83, 0, 1)
        # pre-processing for HDR image
        HDR = np.maximum(full_hdr, 1e-8)
        HDR = np.minimum(HDR, 1e4)
        HDR_I = np.mean(HDR, axis=2)
        mean_HDR = np.mean(mask * HDR_I)
        mean_LDR = np.mean(mask * LDR_I)
        HDR = HDR * mean_LDR / mean_HDR
        cv2.imwrite(os.path.join(train_hdr_dir, f), HDR)
        cv2.imwrite(os.path.join(train_ldr_dir, f[:-4]+'.jpg'), LDR*255)

        for i in range(10):
            cv2.imwrite(os.path.join(train_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i)), LDR_1k*255)
            train_aug_list.append(os.path.join(train_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i)))
            LDR_1k = np.roll(LDR_1k, 1024 // 10, axis=1)
        cv2.imwrite(os.path.join(train_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i+1)), LDR_1k*255)
        train_1k_list.append(os.path.join(train_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i+1)))
        train_aug_list.append(os.path.join(train_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i+1)))


    # val
    val_1k_list = []
    val_aug_list = []
    for f in tqdm(val_file_list):
        full_path = os.path.join(src_dir, f)
        full_hdr = load_raw_hdr(full_path)
        full_ldr, _ = reinhard_tmo(full_hdr)
        full_ldr[np.isnan(full_ldr)] = 0.
        LDR = full_ldr
        LDR_1k = cv2.resize(full_ldr, (1024, 512), interpolation=cv2.INTER_LANCZOS4)
        LDR_I = np.mean(LDR, axis=2)
        mask = np.where(LDR_I > 0.83, 0, 1)
        # pre-processing for HDR image
        HDR = np.maximum(full_hdr, 1e-8)
        HDR = np.minimum(HDR, 1e4)
        HDR_I = np.mean(HDR, axis=2)
        mean_HDR = np.mean(mask * HDR_I)
        mean_LDR = np.mean(mask * LDR_I)
        HDR = HDR * mean_LDR / mean_HDR
        cv2.imwrite(os.path.join(val_hdr_dir, f), HDR)
        cv2.imwrite(os.path.join(val_ldr_dir, f[:-4]+'.jpg'), LDR*255)

        for i in range(10):
            cv2.imwrite(os.path.join(val_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i)), LDR_1k*255)
            val_aug_list.append(os.path.join(val_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i)))
            LDR_1k = np.roll(LDR_1k, 1024 // 10, axis=1)
        cv2.imwrite(os.path.join(val_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i+1)), LDR_1k*255)
        val_1k_list.append(os.path.join(val_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i+1)))
        val_aug_list.append(os.path.join(val_ldr1k_dir, f[:-4]+'_{}.jpg'.format(i+1)))

    meta_dir = os.path.join(dst_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, 'train.txt'), 'w') as fp:
        fp.write('\n'.join(train_1k_list))
    with open(os.path.join(meta_dir, 'val.txt'), 'w') as fp:
        fp.write('\n'.join(val_1k_list))
    with open(os.path.join(meta_dir, 'train_aug.txt'), 'w') as fp:
        fp.write('\n'.join(train_aug_list))
    with open(os.path.join(meta_dir, 'val_aug.txt'), 'w') as fp:
        fp.write('\n'.join(val_aug_list))


