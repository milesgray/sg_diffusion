import math
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from sg.samplers.common import DiffusionSamplerWrapper, DiffusionSampler

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# neural net helpers

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class MonotonicLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())

# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device

        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        x = self.net(x)

        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)

class ContinuousTimeGaussianDiffusionSampler(DiffusionSampler):
    def __init__(self, model):
        self.model = model

    def make_schedule(self, num_steps, **kwargs):
        noise_schedule = kwargs.get("noise_schedule", "linear")
        self.timesteps = torch.linspace(1., 0., num_steps + 1, device=device)

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        return self.timesteps
    
    @torch.no_grad()
    def _sampling(self, cond, shape, timesteps, **kwargs):
        kwargs.get("x_T", None)
        kwargs.get("x0", None)
        kwargs.get("mask", None)
        kwargs.get("temperature", 1.)
        kwargs.get("noise_dropout", 0.)
        kwargs.get("unconditional_guidance_scale", 1.)
        kwargs.get("unconditional_conditioning", None)
        kwargs.get("quantize_denoised", False)
        kwargs.get("callback", None)
        kwargs.get("img_callback", None)
        kwargs.get("score_corrector", None)
        kwargs.get("corrector_kwargs", None)
        kwargs.get("log_every_t", 100)
        kwargs.get("verbose", False)

        device = self.model.betas.device
        batch = shape[0]

        if x_T is None:
            x_T = torch.randn(shape, device = device)
        
        intermediates = {'x_inter': [x_T], 'pred_x0': [x_T]}
        total_steps = timesteps.shape[0]

        if verbose: 
            print(f"Running Continuous Time Sampling with {total_steps} timesteps")
            iterator = tqdm(range(total_steps), desc='sampling loop time step', total=total_steps)
        else:
            iterator = iter(range(total_steps))

        for i in iterator:
            index = total_steps - i - 1

            times = timesteps[i]
            times_next = timesteps[i + 1]

            if mask is not None:
                assert x0 is not None
                ts = timesteps[times: times_next]
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                x_T = img_orig * mask + (1. - mask) * x_T

            x_T = self.p_sample(x_T, times, times_next)

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_T)
                intermediates['pred_x0'].append(pred_x0)

        img = x_T
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img       

    @torch.no_grad()
    def p_sample(self, x, c, t, t_2, **kwargs):
        quantize_denoised = kwargs.get("quantize_denoised", False)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", None)
        unconditional_guidance_scale = kwargs.get("unconditional_guidance_scale", 1.)
        unconditional_conditioning = kwargs.get("unconditional_conditioning", None)

        batch, *_, device = *x.shape, x.device

        model_mean, post_variance, pred_x0 = self.p_mean_variance(x=x, time=t, time_next=t_2)

        if t_2 == 0:
            x_prev = model_mean
        else:
            noise = torch.randn_like(x)
            x_prev = model_mean + sqrt(post_variance) * noise
        return x_prev, pred_x0

    def p_mean_variance(self, x, time, time_next, clip_sample_denoised=True):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        e_t = self.model(x, batch_log_snr)

        if clip_sample_denoised:
            pred_x0 = (x - sigma * e_t) / alpha

            # in Imagen, this was changed to dynamic thresholding
            pred_x0.clamp_(-1., 1.)

            model_mean = alpha_next * (x * (1 - c) / alpha + c * pred_x0)
        else:
            pred_x0 = x - c * sigma * e_t
            model_mean = alpha_next / alpha * pred_x0

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, **kwargs):
        noise = kwargs.get("noise", None)
        if noise is None:
            noise = torch.randn_like(x0)

        log_snr = self.log_snr(times)
        
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())

        return (alpha * x0 +
                sigma * noise) 

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, **kwargs):
        total_steps = timesteps.shape[0]

        if verbose: 
            print(f"Running Continuous Time Sampling with {total_steps} timesteps")
            iterator = tqdm(range(total_steps), desc='sampling loop time step', total=total_steps)
        else:
            iterator = iter(range(total_steps))
        
        x_dec = x_latent
        for i in iterator:
            index = total_steps - i - 1

            times = timesteps[i]
            times_next = timesteps[i + 1]

            if mask is not None:
                assert x0 is not None
                ts = timesteps[times: times_next]
                img_orig = self.q_sample(x0, ts)  # TODO: deterministic forward pass?
                x_T = img_orig * mask + (1. - mask) * x_T

            x_dec, _ = self.p_sample(x_dec, times, times_next)
            
        return x_dec

class ContinuousTimeGaussianDiffusionTrainer(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        loss_type = 'l1',
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        p2_loss_weight_gamma = 0.,                   # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_k = 1
    ):
        super().__init__()
        assert model.learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff

        self.loss_type = loss_type

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() \
                                        for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # p2 loss weight
        # proposed https://arxiv.org/abs/2204.00227

        assert p2_loss_weight_gamma <= 2, 'in paper, they noticed any gamma greater than 2 is harmful'

        self.p2_loss_weight_gamma = p2_loss_weight_gamma  # recommended to be 0.5 or 1
        self.p2_loss_weight_k = p2_loss_weight_k

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_mean_variance(self, x, time, time_next):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred_noise = self.model(x, batch_log_snr)

        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha

            # in Imagen, this was changed to dynamic thresholding
            x_start.clamp_(-1., 1.)

            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, steps, batch_size, shape, **kwargs):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def random_times(self, batch_size):
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr)

        losses = self.loss_fn(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        if self.p2_loss_weight_gamma >= 0:
            # following eq 8. in https://arxiv.org/abs/2204.00227
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -self.p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)