import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from sg.samples.util import transfer, eps_model_fn, make_autocast_model_fn
from sg.samplers.DDIM import DDIMSampler, DDIMSamplerWrapper
from sg.samplers.registry import register

@register("PLMS2")
class PLMSSamplerWrapper(DDIMSamplerWrapper):
    def __init__(self, name, **kwargs):
        super().__init__(name, constructor=PLMS2Sampler, **kwargs)


class PLMS2Sampler(DDIMSampler):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    @torch.no_grad()
    def _sampling(self, cond, shape, timesteps,
                  x_T=None, x0=None, mask=None, 
                  temperature=1., noise_dropout=0., 
                  unconditional_guidance_scale=1., unconditional_conditioning=None,
                  quantize_denoised=False,
                  callback=None, img_callback=None, 
                  score_corrector=None, corrector_kwargs=None,
                  log_every_t=100, verbose=False,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            x_T = torch.randn(shape, device=device)

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
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

            outs = self.p_sample(x_T, cond, ts, index=index, 
                                 quantize_denoised=quantize_denoised, 
                                 temperature=temperature,
                                 noise_dropout=noise_dropout, 
                                 score_corrector=score_corrector,
                                 corrector_kwargs=corrector_kwargs,
                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                 unconditional_conditioning=unconditional_conditioning,
                                 old_eps=old_eps, t_next=ts_next)
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

        
    def plms2_step(self, x, old_eps, t_1, t_2, extra_args):
        eps_model_fn = make_eps_model_fn(self.model)
        eps = eps_model_fn(x, t_1, **extra_args)
        eps_prime = (3 * eps - old_eps[-1]) / 2
        x_new, _ = self.transfer(x, eps_prime, t_1, t_2)
        _, pred = self.transfer(x, eps, t_1, t_2)
        return x_new, eps, pred

    def pie_step(self, x, t_1, t_2, extra_args):
        eps_1 = self.calculate_epsilon(x, extra_args.get("c"), t_1, 
                                       extra_args.get("uc", None), 
                                       extra_args.get("uc_scale", 1.))
        x_1, _ = self.transfer(x, eps_1, t_1, t_2)
        eps_1 = self.calculate_epsilon(x_1, extra_args.get("c"), t_2, 
                                       extra_args.get("uc", None), 
                                       extra_args.get("uc_scale", 1.))
        eps_prime = (eps_1 + eps_2) / 2
        x_new, pred = self.transfer(x, eps_prime, t_1, t_2)
        return x_new, eps_prime, pred

    def transfer(self, x, eps, t_1, t_2):
        a_t, _, sigma_t, _ = self.expand_params(t_1, b, device)
        next_a_t, _, next_sigma_t, _ = self.expand_params(t_2, b, device)
        pred = (x - eps * append_dims(sigma_t, x.ndim)) / append_dims(a_t, x.ndim)
        x = pred * append_dims(next_a_t, x.ndim) + eps * append_dims(next_sigma_t, x.ndim)
        return x, pred

    @torch.no_grad()
    def p_sample(self, x, c, t, index, 
                quantize_denoised=False,
                temperature=1., 
                noise_dropout=0., 
                score_corrector=None, corrector_kwargs=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None,
                old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device
        a_t, a_prev, sigma_t, sqrt_one_minus_at = self.expand_params(index, b, device)

        if len(old_eps) < 1:
            x, eps, pred = self.pie_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        else:
            x, eps, pred = self.plms2_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)

        def get_x_prev_and_pred_x0(e_t, index):
            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = self.calculate_epsilon(x, c, t, unconditional_conditioning, unconditional_conditioning_scale,
                                     score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_next = self.calculate_epsilon(x_prev, c, t_next, unconditional_conditioning, unconditional_conditioning_scale,
                                     score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
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

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t




@torch.no_grad()
def plms2_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using second order
    Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        if len(old_eps) < 1:
            x, eps, pred = pie_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        else:
            x, eps, pred = plms2_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x
