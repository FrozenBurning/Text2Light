import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from taming.util import instantiate_from_config
from taming.models.base_sampler import BaseSampler, disabled_train

class LocalSampler(BaseSampler):
    def __init__(self,
                transformer_config,
                first_stage_config,
                cond_stage_config,
                permuter_config=None,
                ckpt_path=None,
                ignore_keys=[],
                first_stage_key="image",
                cond_stage_key="depth",
                downsample_cond_size=-1,
                pkeep=1.0,
                sos_token=0,
                unconditional=False,
                ):
        super().__init__(
                transformer_config,
                first_stage_config,
                cond_stage_config,
                permuter_config,
                ckpt_path,
                ignore_keys,
                first_stage_key,
                cond_stage_key,
                downsample_cond_size,
                pkeep,
                sos_token,
                unconditional)
    
    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        holistic = self.get_input('holistic', batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
            holistic = holistic[:N]
        return x, c, holistic

    def shared_step(self, batch, batch_idx):
        x, c, holistic = self.get_xc(batch)
        logits, target = self(x, c, holistic)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def forward(self, x, c, holistic):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)
        _, h_indices = self.encode_to_z(holistic)

        c_indices = torch.cat((h_indices, c_indices), dim=1)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]

        return logits, target

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c, holistic = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)
        holistic = holistic.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)
        _, h_indices = self.encode_to_z(holistic)

        c_indices = torch.cat((h_indices, c_indices), dim=1)

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["holistic"] = holistic

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

class LocalSamplerDualCodebook(LocalSampler):
    def __init__(self,
                transformer_config,
                first_stage_config,
                holistic_config,
                cond_stage_config,
                permuter_config=None,
                ckpt_path=None,
                ignore_keys=[],
                first_stage_key="image",
                cond_stage_key="depth",
                downsample_cond_size=-1,
                pkeep=1.0,
                sos_token=0,
                unconditional=False,
                ):
        super().__init__(
                transformer_config,
                first_stage_config,
                cond_stage_config,
                permuter_config,
                ckpt_path,
                ignore_keys,
                first_stage_key,
                cond_stage_key,
                downsample_cond_size,
                pkeep,
                sos_token,
                unconditional)
        model = instantiate_from_config(holistic_config)
        model = model.eval()
        model.train = disabled_train
        self.holistic_model = model

    def forward(self, x, c, holistic):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)
        _, h_indices = self.encode_to_h(holistic)

        c_indices = torch.cat((h_indices, c_indices), dim=1)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]

        return logits, target

    @torch.no_grad()
    def encode_to_h(self, holistic):
        quant_h, _, info = self.holistic_model.encode(holistic)
        indices = info[2].view(quant_h.shape[0], -1)
        indices = self.permuter(indices)
        return quant_h, indices

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c, holistic = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)
        holistic = holistic.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)
        _, h_indices = self.encode_to_h(holistic)

        c_indices = torch.cat((h_indices, c_indices), dim=1)

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["holistic"] = holistic

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

class LocalSamplerDualCodebookPE(LocalSampler):
    def __init__(self,
                transformer_config,
                first_stage_config,
                holistic_config,
                cond_stage_config,
                permuter_config=None,
                ckpt_path=None,
                ignore_keys=[],
                first_stage_key="image",
                cond_stage_key="depth",
                downsample_cond_size=-1,
                pkeep=1.0,
                sos_token=0,
                unconditional=False,
                ):
        super().__init__(
                transformer_config,
                first_stage_config,
                cond_stage_config,
                permuter_config,
                ckpt_path,
                ignore_keys,
                first_stage_key,
                cond_stage_key,
                downsample_cond_size,
                pkeep,
                sos_token,
                unconditional)
        model = instantiate_from_config(holistic_config)
        model = model.eval()
        model.train = disabled_train
        self.holistic_model = model
        self.embedder, self.cdim = get_embedder(4, input_dims=2)

    def forward(self, x, c, holistic):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, h_indices = self.encode_to_h(holistic)
        
        c = torch.nn.functional.interpolate(c, scale_factor=1/8,
                                            mode="bicubic", recompute_scale_factor=False, align_corners=True)
        c = self.embedder(c.permute(0, 2, 3, 1))
        c = c.permute(0, 3, 1, 2).reshape(c.shape[0], self.cdim, -1)
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((h_indices, a_indices), dim=1)
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        logits, _ = self.transformer(cz_indices[:, :-1], embeddings=c)
        logits = logits[:, -target.shape[1]:]
        return logits, target

    @torch.no_grad()
    def encode_to_h(self, holistic):
        quant_h, _, info = self.holistic_model.encode(holistic)
        indices = info[2].view(quant_h.shape[0], -1)
        indices = self.permuter(indices)
        return quant_h, indices

    @torch.no_grad()
    def sample(self, x, h, c, steps, temperature=1, sample=False, top_k=None, callback=lambda k: None):
        x = torch.cat((h,x),dim=1)
        c = torch.nn.functional.interpolate(c, scale_factor=1/8,
                                            mode="bicubic", recompute_scale_factor=False, align_corners=True)
        c = self.embedder(c.permute(0, 2, 3, 1))
        c = c.permute(0, 3, 1, 2).reshape(c.shape[0], self.cdim, -1)
        block_size = self.transformer.get_block_size() - c.shape[1]
        assert not self.transformer.training
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
        # cut off conditioning
        x = x[:, h.shape[1]:]
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c, holistic = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)
        holistic = holistic.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        _, h_indices = self.encode_to_h(holistic)

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, h_indices, c,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, h_indices, c,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, h_indices, c,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["holistic"] = holistic

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim