import os
import math
from pathlib import Path
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from dataset.dataloader import Train_Dataset, Test_Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from models.attend import Attend
from models.version import __version__
from models.encoder_decoder import ReconNet
from metric import get_ssim_torch, get_psnr_torch
from models.color_loss import ColorHistogramKLLoss

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def tensor_gpu(batch, check_on=True):
    def check_on_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            tensor_g = tensor_
        else:
            tensor_g = tensor_.cuda()
        return tensor_g

    def check_off_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            return tensor_

        if tensor_.is_cuda:
            tensor_c = tensor_.cpu()
        else:
            tensor_c = tensor_
        tensor_c = tensor_c.detach().numpy()
        return tensor_c

    if torch.cuda.is_available():
        if check_on:
            for k, v in batch.items():
                batch[k] = check_on_gpu(v)
        else:
            for k, v in batch.items():
                batch[k] = check_off_gpu(v)
    else:
        if check_on:
            batch = batch
        else:
            for k, v in batch.items():
                batch[k] = v.detach().numpy()

    return batch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model

class Unet(Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4,
            full_attn=None,  # defaults to full attention only for inner most layer
            flash_attn=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        input_channels = channels * 2

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_cond):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[
                                                                     -2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        x = torch.cat((x_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Module):
    def __init__(
            self,
            model,
            *,
            timesteps=1000,
            sampling_timesteps=None,
            beta_schedule='sigmoid',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            auto_normalize=True,
            offset_noise_strength=0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
            min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma=5,
            training_stage='stage1'
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.training_stage = training_stage

        self.model = model
        self.stage1_Net = ReconNet(training_stage=self.training_stage)
        self.channels = self.model.channels

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, clip_denoised=True):
        preds = self.model_predictions(x, t, x_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def ddim_sample(self, x_cond, return_all_timesteps=False):

        bs, c, h, w = x_cond.shape

        batch, device, total_timesteps, sampling_timesteps, eta = \
            bs, self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn((bs, 3, h, w), device=device)
        imgs = [img]

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond,
                                                             clip_x_start=True, rederive_pred_noise=True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, data_batch, return_all_timesteps=False):

        if self.training_stage == 'stage1':
            low_RAW_img, high_RAW_img, high_RGB_img = data_batch[
                "low_RAW_img"], data_batch["high_RAW_img"], data_batch["high_RGB_img"]

            high_RGB_img_shuffle = F.pixel_unshuffle(high_RGB_img, 2)

            input_img = torch.cat((low_RAW_img, high_RAW_img, high_RGB_img_shuffle), dim=1)

            b, c, h, w = input_img.shape
            img_h_32 = int(32 * np.ceil(h / 32.0))
            img_w_32 = int(32 * np.ceil(w / 32.0))
            input_img = F.pad(input_img, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')

            pred_img = F.pixel_shuffle(self.stage1_Net(input_img)["high_rgb_img_ori"], 2)
            pred_img = pred_img[:, :, :high_RGB_img.shape[2], :high_RGB_img.shape[3]]

            return pred_img, high_RGB_img

        else:
            low_RAW_img, high_RGB_img = data_batch["low_RAW_img"], data_batch["high_RGB_img"]
            high_RGB_img_shuffle = F.pixel_unshuffle(high_RGB_img, 2)

            input_img = torch.cat((low_RAW_img, high_RGB_img_shuffle), dim=1)

            b, c, h, w = input_img.shape
            img_h_32 = int(256 * np.ceil(h / 256.0))
            img_w_32 = int(256 * np.ceil(w / 256.0))
            input_img = F.pad(input_img, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')

            raw_fea = self.stage1_Net(input_img, pred_fea=None)["raw_fea"]

            x_cond = self.normalize(raw_fea)
            pred_fea = self.ddim_sample(x_cond, return_all_timesteps=return_all_timesteps)
            pred_img = F.pixel_shuffle(self.stage1_Net(input_img, pred_fea=pred_fea)["pred_img"], 2)

            pred_img = pred_img[:, :, :high_RGB_img.shape[2], :high_RGB_img.shape[3]]

            return pred_img, high_RGB_img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def ccl_loss(self, input_img, gt_img):

        input_img, gt_img = torch.clip(input_img * 255, 0, 255), \
            torch.clip(gt_img * 255, 0, 255)

        obj = ColorHistogramKLLoss()

        loss = obj(input_img, gt_img).abs()

        return loss

    def icl_loss(self, low_raw, low_raw_relight, gt_raw):

        low_raw_ill = torch.max(low_raw, dim=1, keepdim=True)[0]
        low_raw_relight_ill = torch.max(low_raw_relight, dim=1, keepdim=True)[0]
        gt_raw_ill = torch.max(gt_raw, dim=1, keepdim=True)[0]

        low_raw_relight_ref = low_raw_relight / (low_raw_relight_ill + 1e-6)
        low_raw_ref = low_raw / (low_raw_ill + 1e-6)

        loss = F.l1_loss(low_raw_relight_ref, low_raw_ref) + \
               F.l1_loss(low_raw_relight_ill, gt_raw_ill)

        return loss

    def forward(self, data_batch, noise=None, offset_noise_strength=None):

        if self.training_stage == 'stage1':
            low_RAW_img, high_RAW_img, high_RGB_img = data_batch[
                "low_RAW_img"], data_batch["high_RAW_img"], data_batch["high_RGB_img"]

            high_RGB_img_shuffle = F.pixel_unshuffle(high_RGB_img, 2)

            input_img = torch.cat((low_RAW_img, high_RAW_img, high_RGB_img_shuffle), dim=1)

            output = self.stage1_Net(input_img)
            low_raw_img_ori, high_raw_img_ori, high_rgb_img_ori = \
                output["low_raw_img_ori"], output["high_raw_img_ori"], output["high_rgb_img_ori"],
            low_raw_fea, low_raw_relight, gt_raw_fea = \
                output["low_raw_fea"], output["low_raw_fea_relight"], output["gt_raw_fea"]

            high_rgb_img_ori = F.pixel_shuffle(high_rgb_img_ori, 2)

            raw_loss = F.l1_loss(low_raw_img_ori, low_RAW_img) + F.l1_loss(high_raw_img_ori, high_RAW_img)
            rgb_loss = F.l1_loss(high_rgb_img_ori, high_RGB_img)

            icl_loss = self.icl_loss(low_raw_fea, low_raw_relight, gt_raw_fea)

            return raw_loss, rgb_loss, icl_loss

        else:
            low_RAW_img, high_RGB_img = data_batch["low_RAW_img"], data_batch["high_RGB_img"]

            high_RGB_img_shuffle = F.pixel_unshuffle(high_RGB_img, 2)

            input_img = torch.cat((low_RAW_img, high_RGB_img_shuffle), dim=1)

            encoded_features = self.stage1_Net(input_img, pred_fea=None)
            raw_fea, gt_fea = encoded_features["raw_fea"], \
                encoded_features["gt_fea"]

            b, c, h, w = raw_fea.shape

            device = raw_fea.device

            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            x_cond = raw_fea

            x_cond_norm = self.normalize(x_cond)
            x0_norm = self.normalize(gt_fea)

            noise = default(noise, lambda: torch.randn_like(x0_norm))

            offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

            if offset_noise_strength > 0.:
                offset_noise = torch.randn(x0_norm.shape[:2], device=self.device)
                noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

            x0 = self.q_sample(x_start=x0_norm, t=t, noise=noise)

            x0_out = self.model(x0, t, x_cond_norm)

            diff_loss = F.mse_loss(x0_out, x0_norm)

            pred_fea = self.unnormalize(x0_out)
            pred_img = self.stage1_Net(input_img, pred_fea=pred_fea)["pred_img"]
            pred_img = F.pixel_shuffle(pred_img, 2)
            content_loss = F.l1_loss(pred_img, high_RGB_img)

            Color_loss = 0.1 * self.ccl_loss(pred_fea, gt_fea)

            return diff_loss, content_loss, Color_loss


class Trainer:
    def __init__(
            self,
            diffusion_model,
            data_dir,
            train_dataset,
            val_dataset,
            *,
            train_batch_size=16,
            patch_size=[128, 128],
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            ckpt_path='ckpt/stage2',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            convert_image_to=None,
            max_grad_norm=1.
    ):
        super().__init__()

        # accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            kwargs_handlers=[ddp_kwargs]
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.training_stage = self.model.training_stage

        self.dir = data_dir
        self.train_dataset = train_dataset
        self.va_dataset = val_dataset
        self.patch_size = patch_size

        self.ckpt_path = ckpt_path

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.train_ds = Train_Dataset(image_dir=self.dir,
                                      filelist='{}.txt'.format(self.train_dataset),
                                      patch_size=self.patch_size)
        self.val_ds = Test_Dataset(image_dir=self.dir,
                                   filelist='{}.txt'.format(self.va_dataset))

        train_dl = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True,
                              pin_memory=True, num_workers=8)
        val_dl = DataLoader(self.val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=8)

        train_dl = self.accelerator.prepare(train_dl)

        self.train_dl = cycle(train_dl)
        self.val_dl = val_dl
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.best_psnr, self.best_ssim = 0, 0

    @property
    def device(self):
        return self.accelerator.device

    def save(self, save_path, save_name):

        os.makedirs(save_path, exist_ok=True)

        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, os.path.join(save_path, '{}.pt'.format(save_name)))

    def load(self, ckpt_path):
        device = self.accelerator.device
        data = torch.load(os.path.join(ckpt_path), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        # self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        # if self.accelerator.is_main_process:
            # self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
            # self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        if self.training_stage == 'stage1':
            for name, param in self.model.named_parameters():
                if "stage1_Net" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        else:
            for name, param in self.model.named_parameters():
                if "stage1_Net" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process,
                  ncols=100) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dl)
                    data = tensor_gpu(data)

                    with self.accelerator.autocast():
                        if self.training_stage == 'stage1':
                            raw_loss, rgb_loss, icl_loss = self.model(data)
                            loss = raw_loss + rgb_loss + icl_loss
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        else:
                            diff_loss, content_loss, Color_loss = self.model(data)
                            loss = diff_loss + content_loss + Color_loss
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    self.accelerator.backward(loss)
                if self.training_stage == 'stage1':
                    pbar.set_description('raw:{:.4f} rgb:{:.4f} icl:{:.4f}'
                                         .format(raw_loss.item(), rgb_loss.item(), icl_loss.item()))
                else:
                    pbar.set_description('diff:{:.4f} cont:{:.4f} col:{:.4f}'
                                         .format(diff_loss.item(), content_loss.item(), Color_loss.item()))

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.model.eval()

                        self.save(self.ckpt_path, 'model_latest')

                        psnr, ssim = 0, 0

                        with torch.inference_mode():
                            for i, data_batch in enumerate(self.val_dl):
                                data_batch = tensor_gpu(data_batch)

                                img_name = data_batch["img_name"][-1]

                                pred_img, high_img = self.model.sample(data_batch, return_all_timesteps=False)

                                pred_img = torch.clip(pred_img * 255.0, 0, 255.0)
                                high_img = torch.clip(high_img * 255.0, 0, 255.0)

                                psnr += get_psnr_torch(pred_img, high_img).item()
                                ssim += get_ssim_torch(pred_img, high_img).item()

                                pred_img = pred_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')

                                pred_img = Image.fromarray(pred_img)
                                pred_img.save(os.path.join(self.results_folder, img_name))

                            avg_psnr = psnr / len(self.val_dl)
                            avg_ssim = ssim / len(self.val_dl)
                            print('img_num {} avg psnr:{:.4f} ssim:{:.4f}'
                                  .format(len(self.val_dl), avg_psnr, avg_ssim))

                            if avg_psnr > self.best_psnr:
                                self.best_psnr = avg_psnr
                                self.save(self.ckpt_path, 'best_{:.3f}_{:.3f}'.format(avg_psnr, avg_ssim))
                            elif avg_ssim > self.best_ssim:
                                self.best_ssim = avg_ssim
                                self.save(self.ckpt_path, 'best_{:.3f}_{:.3f}'.format(avg_psnr, avg_ssim))

                pbar.update(1)

        accelerator.print('training complete')
