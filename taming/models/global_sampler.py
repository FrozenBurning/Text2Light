import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from taming.models.base_sampler import BaseSampler

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        RandomCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class GlobalSampler(BaseSampler):
    def forward(self, x, c):
        _, z_indices = self.encode_to_z(x)
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices
        target = z_indices
        cb, _, knn, cd = c.shape
        c = c.reshape(cb, knn, cd)
        c = torch.cat([c, c,], dim=-1)
        logits, _ = self.transformer(a_indices[:, :-1], embeddings=c)
        logits = logits[:, knn-1:]

        return logits, target
    
    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()
        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)
        quant_z, z_indices = self.encode_to_z(x)

        # half
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1, sample=False, top_k=None, callback=lambda k: None):
        cb, _, knn, cd = c.shape
        c = c.reshape(cb, knn, cd)
        c = torch.cat([c, c,], dim=-1)

        block_size = self.transformer.get_block_size() - 5
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            raise NotImplementedError
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond, embeddings=c)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
        return x


class GlobalSamplerWithCLIP(GlobalSampler):
    def __init__(self, transformer_config, first_stage_config, cond_stage_config, permuter_config=None, ckpt_path=None, ignore_keys=[], first_stage_key="image", cond_stage_key="depth", downsample_cond_size=-1, pkeep=1, sos_token=0, unconditional=False):
        super().__init__(transformer_config, first_stage_config, cond_stage_config, permuter_config, ckpt_path, ignore_keys, first_stage_key, cond_stage_key, downsample_cond_size, pkeep, sos_token, unconditional)
        import clip
        self.clip, _ = clip.load("ViT-B/32", device="cpu")
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        z_indices = target
        z_start_indices = z_indices[:, :0]

        cb, _, knn, cd = c.shape
        c = c.reshape(cb, knn, cd)
        c = torch.cat([c, c,], dim=-1)

        steps = z_indices.shape[1]
        block_size = self.transformer.get_block_size() - 5
        callback = lambda k: None
        if self.pkeep <= 0.0:
            raise NotImplementedError
        else:
            for k in range(steps):
                callback(k)
                assert z_start_indices.size(1) <= block_size # make sure model can see conditioning
                x_cond = z_start_indices if z_start_indices.size(1) <= block_size else z_start_indices[:, -block_size:]  # crop context if needed
                lo, _ = self.transformer(x_cond, embeddings=c)
                # pluck the logits at the final step and scale by temperature
                lo = lo[:, -1, :] / 1.0
                # optionally crop probabilities to only the top k options
                lo = self.top_k_logits(lo, 100)
                # apply softmax to convert to probabilities
                probs = F.softmax(lo, dim=-1)
                # sample from the distribution or take the most likely
                ix = torch.multinomial(probs, num_samples=1)
                # append to the sequence and continue
                z_start_indices = torch.cat((z_start_indices, ix), dim=1)

        index_sample = z_start_indices
        x_sample_nopix = self.decode_to_img(index_sample, [index_sample.shape[0], 256, 8, 16])
        preprocess = _transform(224)
        gen_img_emb = self.clip.encode_image(preprocess(x_sample_nopix))
        gen_img_emb /= gen_img_emb.norm(dim=-1, keepdim=True)
        
        psed_emb = batch['psed_emb']
        sim = torch.cosine_similarity(gen_img_emb.unsqueeze(1), psed_emb.unsqueeze(0), dim=-1)
        temp = 0.5
        sim = torch.exp(sim/temp)
        sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
        contra_loss = torch.log(sim2).mean()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss - contra_loss, -contra_loss

    def training_step(self, batch, batch_idx):
        loss, contra_loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/contra_loss", contra_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, contra_loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/contra_loss", contra_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

