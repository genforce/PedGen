import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)

    dct_m = torch.from_numpy(dct_m).float()
    idct_m = torch.from_numpy(idct_m).float()
    return dct_m, idct_m


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim, cond_type):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.cond_type = cond_type
        if self.cond_type != "film":
            self.norm = nn.LayerNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.time_proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                              dropout)
        self.cond_proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                              dropout)

    def forward(self, x, time_emb, cond_emb):
        B, T, D = x.shape
        if self.cond_type != "film":
            y = self.linear2(
                self.dropout(self.activation(self.linear1(self.norm(x)))))
            y = y + x
        else:
            y = self.linear2(self.dropout(self.activation(self.linear1(x))))
            time_res = self.time_proj_out(y, time_emb)
            if cond_emb is not None:
                cond_res = self.cond_proj_out(y[:, :T // 2, :], cond_emb)
            y = x + time_res
            if cond_emb is not None:
                y[:, :T // 2, :] += cond_res
        return y


class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim,
                 cond_type):
        super().__init__()
        self.num_head = num_head
        self.cond_type = cond_type
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.time_proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                              dropout)
        self.cond_proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                              dropout)

    def forward(self, x, time_emb, cond_emb):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(
            D // H)

        # apply attention mask
        # attention[:, :T // 2, T // 2:] = -torch.inf
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        if self.cond_type != "film":
            y = x + y
        else:
            time_res = self.time_proj_out(y, time_emb)
            if cond_emb is not None:
                cond_res = self.cond_proj_out(y[:, :T // 2, :], cond_emb)
            y = x + time_res
            if cond_emb is not None:
                y[:, :T // 2, :] += cond_res
        return y


class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.time_proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                              dropout)
        self.cond_proj_out = StylizationBlock(latent_dim, time_embed_dim,
                                              dropout)

    def forward(self, x, xf, time_emb, cond_emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(
            D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        time_res = self.time_proj_out(y, time_emb)
        if cond_emb is not None:
            cond_res = self.cond_proj_out(y[:, :T // 2, :], cond_emb)
        y = x + time_res
        if cond_emb is not None:
            y[:, :T // 2, :] += cond_res
        return y


class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        latent_dim=32,
        time_embed_dim=128,
        ffn_dim=256,
        num_head=4,
        dropout=0.5,
        cond_type="film",
    ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(latent_dim,
                                              num_head,
                                              dropout,
                                              time_embed_dim,
                                              cond_type=cond_type)
        self.ffn = FFN(latent_dim,
                       ffn_dim,
                       dropout,
                       time_embed_dim,
                       cond_type=cond_type)

    def forward(self, x, time_emb, cond_emb):
        x = self.sa_block(x, time_emb, cond_emb)
        x = self.ffn(x, time_emb, cond_emb)
        return x


class MotionTransformer(nn.Module):

    def __init__(
            self,
            input_feats,
            num_frames=240,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.2,
            activation="gelu",
            trans_rot_sep=True,
            cond_type="film",  # between film or cross-attn
            **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.trans_rot_sep = trans_rot_sep
        self.cond_type = cond_type

        if self.cond_type == "film":
            self.sequence_embedding = nn.Parameter(
                torch.randn(num_frames, latent_dim))
        else:
            self.sequence_embedding = nn.Parameter(
                torch.randn(num_frames + 1, latent_dim))

        if self.trans_rot_sep:
            self.trans_rot_embedding = nn.Parameter(torch.randn(2, latent_dim))

            # Input Embedding
            self.trans_embed = nn.Linear(3, self.latent_dim)
            self.rot_embed = nn.Linear(self.input_feats - 3, self.latent_dim)
        else:
            self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    cond_type=cond_type,
                ))

        # Output Module
        if self.trans_rot_sep:
            self.trans_out = zero_module(nn.Linear(self.latent_dim, 3))
            self.rot_out = zero_module(
                nn.Linear(self.latent_dim, self.input_feats - 3))
        else:
            self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, cond_embed=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        time_emb = self.time_embed(
            timestep_embedding(timesteps, self.latent_dim))

        if not self.trans_rot_sep and cond_embed is not None:
            time_emb += cond_embed
            cond_embed = None

        # B, T, latent_dim
        if self.trans_rot_sep:
            h_trans = self.trans_embed(x[..., :3])
            h_rot = self.rot_embed(x[..., 3:])
            h = torch.cat([h_trans, h_rot], dim=1)
            h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :].repeat(
                1, 2,
                1) + self.trans_rot_embedding.unsqueeze(0).repeat_interleave(
                    T, 1)
        else:
            h = self.joint_embed(x)
            if self.cond_type == "film":
                h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
            else:
                h = torch.cat([h, time_emb.unsqueeze(1)], dim=1)
                h = h + self.sequence_embedding.unsqueeze(0)[:, :T + 1, :]

        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, time_emb, cond_embed)
            elif i >= (self.num_layers // 2):
                h = module(h, time_emb, None)
                if self.cond_type == "film":
                    h += prelist[-1]
                    prelist.pop()
            i += 1

        if self.trans_rot_sep:
            output_trans = self.trans_out(h[:, :T, :]).view(B, T,
                                                            -1).contiguous()
            output_rot = self.rot_out(h[:, T:, :]).view(B, T, -1).contiguous()
            output = torch.cat([output_trans, output_rot], dim=-1)
        else:
            if self.cond_type != "film":
                h = h[:, :T, :]
            output = self.out(h).view(B, T, -1).contiguous()
        return output


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MLPHead(nn.Module):

    def __init__(self, input_channels, output_channels, num_basic_blocks=2):
        super(MLPHead, self).__init__()
        head_layers = []
        num_channels = output_channels

        head_layers.append(
            nn.Sequential(nn.Linear(input_channels, num_channels),
                          nn.LayerNorm(num_channels), nn.SiLU()))

        head_layers.append(nn.Linear(num_channels, output_channels))
        self.head_layers = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head_layers(x)
