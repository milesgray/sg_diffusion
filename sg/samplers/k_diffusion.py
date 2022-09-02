import torch
import torch.nn as nn

import diffuson.samplers.util as util
from diffuson.samplers.registry import register

class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.quantize = quantize

    def get_sigmas(self, n=None):
        if n is None:
            return util.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return util.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        dists = torch.abs(sigma - self.sigmas[:, None])
        if quantize:
            return torch.argmin(dists, dim=0).view(sigma.shape)
        low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
        low, high = self.sigmas[low_idx], self.sigmas[high_idx]
        w = (low - sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        return (1 - w) * self.sigmas[low_idx] + w * self.sigmas[high_idx]

class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_out, c_in = [util.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * util.append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_out, c_in = [util.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out

class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""

    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

@register("k_diffusion")
class KDiffusionSampler:
    def __init__(self, model, funcname='sample_heun'):
        self.model_wrap = CompVisDenoiser(model)
        self.funcname = funcname
        self.func = sample_funcs[self.funcname]
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def sample(self, S: int,
               conditioning: torch.Tensor=None, 
               batch_size: int=1,
               shape=None,
               verbose: bool=False,
               unconditional_guidance_scale: float=1.0,
               unconditional_conditioning: torch.Tensor=None,
               eta: float=0.0,
               x_T: torch.Tensor=None):
        sigmas = self.model_wrap.get_sigmas(S)
        if x_T is None:
            x_T = torch.randn([batch_size]+shape).cuda()
        x = x_T * sigmas[0]

        samples = self.func(self.model_wrap_cfg, x, sigmas, 
                            extra_args={
                                'cond': conditioning, 
                                'uncond': unconditional_conditioning, 
                                'cond_scale': unconditional_guidance_scale
                                }, 
                            disable=False)
        return samples


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = util.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = util.get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = util.to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + torch.randn_like(x) * sigma_up
    return x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = util.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = util.to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = util.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
        d_2 = util.to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x

@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Ancestral sampling with DPM-Solver inspired second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = util.get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = util.to_d(x, sigmas[i], denoised)
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigmas[i] ** (1 / 3) + sigma_down ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigmas[i]
        dt_2 = sigma_down - sigmas[i]
        x_2 = x + d * dt_1
        denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
        d_2 = util.to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
        x = x + torch.randn_like(x) * sigma_up
    return x

@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    ds = []
    for i in range(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = util.to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [util.linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x

sample_funcs = {
    "sample_euler": sample_euler,
    "sample_euler_ancestral": sample_euler_ancestral,
    "sample_heun": sample_heun,
    "sample_dpm_2": sample_dpm_2,
    "sample_dpm_2_ancestral": sample_dpm_2_ancestral,
    "sample_lms": sample_lms
}