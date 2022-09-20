import argparse, os, sys, glob
import cv2
import torch
import faiss
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import clip
from taming.util import instantiate_from_config
from sritmo.global_sritmo import SRiTMO


def save_image(x, path):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)


def get_knn(database: np.array, index: faiss.Index, txt_emb, k = 5):
    dist, idx  = index.search(txt_emb, k)
    return database[idx], idx #[bs, k, 512]


@torch.no_grad()
def text2light(models: dict, prompts, outdir, params: dict):
    # models
    global_sampler = models["gs"]
    local_sampler = models["ls"]
    # params
    batch_size = len(prompts)
    top_k = params["top_k"]
    temperature = params['temperature']
    database = params['data4knn']
    faiss_index = params['index4knn']
    device = params['device']

    # embed input texts
    lan_model, _ = clip.load("ViT-B/32", device=device)
    lan_model.eval()
    text = clip.tokenize(prompts).to(device)
    text_features = lan_model.encode_text(text)
    target_txt_emb = text_features / text_features.norm(dim=-1, keepdim=True)
    cond, _ = get_knn(database, faiss_index, target_txt_emb.cpu().numpy().astype('float32'))
    txt_cond = torch.from_numpy(cond.reshape(batch_size, 5, cond.shape[-1]))
    txt_cond = torch.cat([txt_cond, txt_cond,], dim=-1).to(device)

    # sample holistic condition
    bs = batch_size
    start = 0
    idx = torch.zeros(bs, 1, dtype=int)[:, :start].to(device)
    cshape = [bs, 256, 8, 16]
    sample = True

    for i in tqdm(range(start, cshape[2]*cshape[3])):
        logits, _ = global_sampler.transformer(idx, embeddings=txt_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            logits = global_sampler.top_k_logits(logits, top_k)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, ix), dim=1)

    xsample_holistic = global_sampler.decode_to_img(idx, cshape)
    for i in range(xsample_holistic.shape[0]):
        save_image(xsample_holistic[i], os.path.join(outdir, "holistic", "holistic_[{}].png".format(prompts[i])))

    # synthesize patch by patch according to holistic condition
    h = 512
    w = 1024
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    screen_points = np.stack([xx, yy], axis=-1)
    coord = (screen_points * 2 - 1) * np.array([np.pi, np.pi/2])
    spe = torch.from_numpy(coord).to(xsample_holistic).repeat(xsample_holistic.shape[0], 1, 1, 1).permute(0, 3, 1, 2)
    spe = torch.nn.functional.interpolate(spe, scale_factor=1/8,
                                            mode="bicubic", recompute_scale_factor=False, align_corners=True)
    spe = local_sampler.embedder(spe.permute(0, 2, 3, 1))
    spe = spe.permute(0, 3, 1, 2)

    _, h_indices = local_sampler.encode_to_h(xsample_holistic)
    cshape = [xsample_holistic.shape[0], 256, h // 16, w // 16]
    idx = torch.randint(0, 1024, (cshape[0], cshape[2], cshape[3])).to(h_indices)
    idx = idx.reshape(cshape[0], cshape[2], cshape[3])

    start = 0
    start_i = start // cshape[3]
    start_j = start % cshape[3]
    sample = True

    for i in tqdm(range(start_i, cshape[2])):
        if i <= 8:
            local_i = i
        elif cshape[2]-i < 8:
            local_i = 16-(cshape[2]-i)
        else:
            local_i = 8
        for j in tqdm(range(start_j, cshape[3])):
            if j <= 8:
                local_j = j
            elif cshape[3]-j < 8:
                local_j = 16-(cshape[3]-j)
            else:
                local_j = 8

            i_start = i-local_i
            i_end = i_start+16
            j_start = j-local_j
            j_end = j_start+16
            patch = idx[:,i_start:i_end,j_start:j_end]
            patch = patch.reshape(patch.shape[0],-1)
            cpatch = spe[:, :, i_start*2:i_end*2,j_start*2:j_end*2]
            cpatch = cpatch.reshape(cpatch.shape[0], local_sampler.cdim, -1)
            patch = torch.cat((h_indices, patch), dim=1)
            logits, _ = local_sampler.transformer(patch[:,:-1], embeddings=cpatch)
            logits = logits[:, -256:, :]
            logits = logits.reshape(cshape[0],16,16,-1)
            logits = logits[:,local_i,local_j,:]
            logits = logits / temperature

            if top_k is not None:
                logits = local_sampler.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            idx[:,i,j] = ix.reshape(-1)
    xsample = local_sampler.decode_to_img(idx, cshape)
    for i in range(xsample.shape[0]):
        save_image(xsample[i], os.path.join(outdir, "ldr", "ldr_[{}].png".format(prompts[i])))

    # super-resolution inverse tone mapping
    if params['sritmo'] is not None:
        ldr_hr_samples, hdr_hr_samples = SRiTMO(xsample, params)
    else:
        print("no checkpoint provided, skip Stage II (SR-iTMO)...")
        return
    
    for i in range(xsample.shape[0]):
        cv2.imwrite(os.path.join(outdir, "ldr", "hrldr_[{}].png".format(prompts[i])), (ldr_hr_samples[i].permute(1, 2, 0).detach().cpu().numpy() + 1) * 127.5)
        cv2.imwrite(os.path.join(outdir, "hdr", "hdr_[{}].exr".format(prompts[i])), hdr_hr_samples[i].permute(1, 2, 0).detach().cpu().numpy())


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rg",
        "--resume_global",
        type=str,
        nargs="?",
        help="load global sampler from logdir or checkpoint in logdir.",
    )
    parser.add_argument(
        "-rl",
        "--resume_local",
        type=str,
        nargs="?",
        help="load local sampler from logdir or checkpoint in logdir.",
    )
    parser.add_argument(
        "--sritmo",
        type=str,
        nargs="?",
        default=None,
        help="load super-resolution inverse tone mapping operator from the given path.",
    )
    parser.add_argument(
        "--sr_factor",
        type=int,
        nargs="?",
        default=4,
        help="upscaling factor for super-resolution."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="output directory.",
    )
    parser.add_argument(
        '--clip',
        required=True,
        type=str,
        default='clip_emb.npy',
        help="the path to numpy file of CLIP embeddings database.",
    )
    parser.add_argument(
        "--text",
        required=True,
        type=str,
        help="input scene descriptions. Can be a single sentence typed via command line or the file path to a list of texts.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=4,
        help="batch size. Tune it according to your GPU capacity.",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.holistic_config.params:
            config.params.holistic_config.params.ckpt_path = None
            print("Deleting the global sampler restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        raw_model = torch.load(ckpt, map_location="cpu")
        state_dict = raw_model["state_dict"]
    else:
        raise NotImplementedError("checkpoint at [{}] is not found!".format(ckpt))
    model = load_model_from_config(config.model, state_dict, gpu=gpu, eval_mode=eval_mode)["model"]
    return model


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    gpu = True
    eval_mode = True
    show_config = False

    base = list()

    ckpt = None
    if opt.resume_global:
        if not os.path.exists(opt.resume_global):
            raise ValueError("Cannot find {}".format(opt.resume_global))
        print("Resuming from global sampler ckpt...")
        assert os.path.isdir(opt.resume_global), opt.resume_global
        logdir = opt.resume_global.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        config2load = base_configs + base

    configs = [OmegaConf.load(cfg) for cfg in config2load]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    print(ckpt)
    if show_config:
        print(OmegaConf.to_container(config))
    
    global_sampler = load_model(config, ckpt, gpu, eval_mode)

    ckpt = None
    if opt.resume_local:
        if not os.path.exists(opt.resume_local):
            raise ValueError("Cannot find {}".format(opt.resume_local))
        print("Resuming from local sampler ckpt...")
        assert os.path.isdir(opt.resume_local), opt.resume_local
        logdir = opt.resume_local.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        config2load = base_configs + base

    configs = [OmegaConf.load(cfg) for cfg in config2load]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    print(ckpt)
    if show_config:
        print(OmegaConf.to_container(config))

    local_sampler = load_model(config, ckpt, gpu, eval_mode)

    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)
    print("Writing samples to ", outdir)
    for k in ["holistic", "ldr", "hdr"]:
        os.makedirs(os.path.join(outdir, k), exist_ok=True)
    
    prompts_file = opt.text
    if os.path.exists(prompts_file):
        # list of prompts for text2light tasks
        with open(prompts_file, 'r') as f:
            prompts = f.read().splitlines()
    else:
        # a single prompt
        prompts = [prompts_file]

    # construct knn searching base
    if os.path.isfile(opt.clip):
        clip_emb = np.load(opt.clip).astype('float32')
    else:
        raise NotImplementedError('The path [{}] to clip embedding is not valid.'.format(opt.clip))
    
    knn_index = faiss.IndexFlatIP(clip_emb.shape[-1])
    knn_index.add(clip_emb)

    input_models = {
        'gs': global_sampler,
        'ls': local_sampler,
    }

    input_params = {
        'top_k': opt.top_k,
        'temperature': opt.temperature,
        'device': 'cuda' if gpu else 'cpu',
        'data4knn': clip_emb,
        'index4knn': knn_index,
        'sritmo': opt.sritmo,
        'sr_factor': opt.sr_factor,
    }
    for i in range(0, len(prompts), opt.bs):
        end_i = min(len(prompts), i + opt.bs)
        prompt = prompts[i: i+opt.bs]
        text2light(input_models, prompt, outdir, input_params)
