import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sritmo.net import make_edsr_baseline
from sritmo.mlp import MLP
from sritmo.util import make_coord, batchify

class SRToneMapper(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self.ldr_encoder = make_edsr_baseline(no_upsampling=True)
        self.srmapper = MLP(in_dim = self.ldr_encoder.out_dim * 9 + 4, out_dim = 3, hidden_list = [256, 256, 256, 256], final_act = None)
        self.hdrmapper = MLP(in_dim = 256 + 2, out_dim = 3, hidden_list = [256, 256], final_act = None)

    def forward(self, batch):
        lr_ldr = batch['lr_ldr']
        coord = batch['local_coord']
        cell = batch['cell']
        glb_coord = batch['global_coord']
        feat = self.ldr_encoder(lr_ldr)
        feat = F.unfold(feat, 3, padding = 1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        intermediate = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps
                coord_[:, :, 1] += vy * ry + eps
                coord_.clamp_(-1 + eps, 1 - eps)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                hr_ldr, inter = self.srmapper(inp.reshape(bs*q, -1), 1)
                hr_ldr = hr_ldr.reshape(bs, q, -1)
                inter = inter.reshape(bs, q, -1)
                preds.append(hr_ldr)
                intermediate.append(inter)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / total_area).unsqueeze(-1)
        feat4hdr = 0
        for inte, area in zip(intermediate, areas):
            feat4hdr = feat4hdr + inte * (area / total_area).unsqueeze(-1)
        hr_ldr = ret

        input2hdrmapper = torch.cat([feat4hdr, glb_coord], dim=-1)
        hr_hdr, _ = self.hdrmapper(input2hdrmapper)

        return hr_ldr, hr_hdr
    
    def gen_feat(self, img):
        self.feat = self.ldr_encoder(img)
        return self.feat
    
    def query(self, glb_coord, coord, cell=None):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        intermediate = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps
                coord_[:, :, 1] += vy * ry + eps
                coord_.clamp_(-1 + eps, 1 - eps)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                hr_ldr, inter = self.srmapper(inp.reshape(bs*q, -1), 1)
                hr_ldr = hr_ldr.reshape(bs, q, -1)
                inter = inter.reshape(bs, q, -1)
                preds.append(hr_ldr)
                intermediate.append(inter)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / total_area).unsqueeze(-1)
        feat4hdr = 0
        for inte, area in zip(intermediate, areas):
            feat4hdr = feat4hdr + inte * (area / total_area).unsqueeze(-1)
        hr_ldr = ret

        input2hdrmapper = torch.cat([feat4hdr, glb_coord], dim=-1)
        hr_hdr, _ = self.hdrmapper(input2hdrmapper)

        return hr_ldr, hr_hdr

@torch.no_grad()
def SRiTMO(ldr_samples: torch.Tensor, params: dict):
    model_path = params['sritmo']
    sr_factor = params['sr_factor']
    device = params['device']

    model = SRToneMapper()
    state_dict = torch.load(model_path, map_location='cpu')['net']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # TODO: normalize ldr_samples?
    bs, channels, height, width = ldr_samples.shape
    ldr_samples = ldr_samples[:, [2, 1, 0], :, :] # RGB2BGR, the stage II model is trained on BGR format
    height = int(height * sr_factor)
    width = int(width * sr_factor)
    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    screen_points = np.stack([xx, yy], axis=-1)
    glb_coords = (screen_points * 2 - 1) * np.array([np.pi, np.pi/2])
    glb_coords = torch.from_numpy(glb_coords.astype('float32').reshape(-1, 2)).to(device)

    coord = make_coord((height, width)).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / height
    cell[:, 1] *= 2 / width
    hr_output, hdr_output = batchify(model, ldr_samples, glb_coords.repeat(bs, 1, 1), coord.repeat(bs, 1, 1), cell.repeat(bs, 1, 1), bsize=30000)

    hr_output = hr_output.reshape(bs, height, width, channels).permute(0, 3, 1, 2)
    hdr_output = hdr_output.reshape(bs, height, width, channels).permute(0, 3, 1, 2)

    gamma = 1
    boost = 4
    balance = 0.7
    luma = torch.mean(hdr_output, dim=1, keepdim=True)
    mask = torch.clip(luma / luma.max() - 0.83, 0, 1)
    hdr_output += hdr_output * mask * boost

    hdr_output = torch.exp((hdr_output - hdr_output.mean()) * gamma - balance)

    return hr_output, hdr_output