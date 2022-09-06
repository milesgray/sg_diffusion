"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from sg.samplers.util import DiffusionSamplerWrapper
from sg.samplers.ddim import DDIMSampler
from sg.samplers.registry import register

@register("PLMS")
class PLMSSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name, **kwargs):
        kwargs["constructor"] = PLMSSampler
        super().__init__(name, **kwargs)


class PLMSSampler(DDIMSampler):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    @torch.no_grad()
    def _sampling(self, cond, shape, timesteps, **kwargs):
        x_T = kwargs.get("x_T", None)
        x0 = kwargs.get("x0", None)
        mask = kwargs.get("mask", None)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        unconditional_guidance_scale = kwargs.get("unconditional_guidance_scale", 1.)
        unconditional_conditioning = kwargs.get("unconditional_conditioning", None)
        quantize_denoised = kwargs.get("quantize_denoised", False)
        callback = kwargs.get("callback", None)
        img_callback = kwargs.get("img_callback", None)
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", None)
        log_every_t = kwargs.get("log_every_t", 100)
        verbose = kwargs.get("verbose", False)

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            x_T = torch.randn(shape, device=device)

        intermediates = {'x_inter': [x_T], 'pred_x0': [x_T]}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        if verbose: 
            print(f"Running PLMS Sampling with {total_steps} timesteps")
            iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        else:
            iterator = iter(time_range)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                x_T = img_orig * mask + (1. - mask) * x_T
            kwargs["old_eps"] = old_eps
            kwargs["t_next"] = ts_next
            outs = self.p_sample(x_T, cond, ts, index, **kwargs)
            x_T, pred_x0, e_t = outs

            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_T)
                intermediates['pred_x0'].append(pred_x0)
        img = x_T
        return img, intermediates

    @torch.no_grad()
    def p_sample(self, x, c, t, index, **kwargs):
        quantize_denoised = kwargs.get("quantize_denoised", False)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", None)
        uc_scale = kwargs.get("unconditional_guidance_scale", 1.)
        uc = kwargs.get("unconditional_conditioning", None)
        old_eps = kwargs.get("old_eps", [])
        t_next = kwargs.get("t_next", t+1)

        b, *_, device = *x.shape, x.device

        e_t = self.calculate_epsilon(x, c, t, uc, uc_scale)
        a_t, a_prev, sigma_t, sqrt_one_minus_at = self.expand_params(index, b, device)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = self.get_x_prev_and_pred_x0(x, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at, index, **kwargs)
            e_t_next = self.calculate_epsilon(x_prev, c, t_next, uc, uc_scale)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(x, e_t_prime, a_t, a_prev, sigma_t, sqrt_one_minus_at, index, **kwargs)

        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev, 
                                sigma_t, sqrt_one_minus_at, index, **kwargs):
        quantize_denoised = kwargs.get("quantize_denoised", False)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
    
        b, *_, device = *x.shape, x.device
        
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * torch.randn(x.shape, device=device) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0