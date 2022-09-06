"""SAMPLING ONLY."""

import torch
import numpy as np
from functools import partial

from sg.samplers.common import DiffusionSamplerWrapper, DiffusionSampler
from sg.samplers.registry import register

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

@register("DDIM")
class DDIMSamplerWrapper(DiffusionSamplerWrapper):
    def __init__(self, name: str, **kwargs):
        kwargs["constructor"] = DDIMSampler
        super().__init__(name, **kwargs)

class DDIMSampler(DiffusionSampler):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.num_ddpm_timesteps = model.num_timesteps

    def _register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def _make_timesteps(self, discr_method, num_timesteps, verbose=True):
        if discr_method == 'uniform':
            c = self.num_ddpm_timesteps // num_timesteps
            ddim_timesteps = np.asarray(list(range(0, self.num_ddpm_timesteps, c)))
        elif discr_method == 'quad':
            ddim_timesteps = ((np.linspace(0, np.sqrt(self.num_ddpm_timesteps * .8), num_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        steps_out = ddim_timesteps + 1
        if verbose:
            print(f'Selected timesteps for ddim sampler: {steps_out}')
        return steps_out

    def _make_sampling_parameters(self, alphacums, eta, verbose=True):
        # select alphas for computing the variance schedule
        alphas = alphacums[self.timesteps]
        alphas_prev = np.asarray([alphacums[0]] + alphacums[self.timesteps[:-1]].tolist())

        # according the the formula provided in https://arxiv.org/abs/2010.02502
        sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        if verbose:
            print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
            print(f'For the chosen value of eta, which is {eta}, '
                f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
        return sigmas, alphas, alphas_prev

    def make_schedule(self, num_steps, **kwargs): 
        verbose = kwargs.get("verbose", True)
        discretize = kwargs.get("discretize", "uniform")
        eta = kwargs.get("eta", 0)

        self.timesteps = self._make_timesteps(discr_method=discretize, 
                                              num_timesteps=num_steps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.num_ddpm_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self._register_buffer('betas', to_torch(self.model.betas))
        self._register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self._register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self._register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self._register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self._register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self._register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self._register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        sigmas, alphas, alphas_prev = self._make_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                     eta=eta,
                                                                     verbose=verbose)
        self._register_buffer('sigmas', sigmas)
        self._register_buffer('alphas', alphas)
        self._register_buffer('alphas_prev', alphas_prev)
        self._register_buffer('sqrt_one_minus_alphas', np.sqrt(1. - alphas))

        return self.timesteps
    
    def _validate_conditioning(self, conditioning, batch_size, verbose=False):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                inner = conditioning[list(conditioning.keys())[0]]
                if isinstance(inner, list):
                    inner_item = inner[0]
                    if isinstance(inner_item, tuple):                        
                        cbs = inner_item[1].shape[0]
                    elif hasattr(inner_item, "shape"):
                        cbs = inner_item.shape[0]
                    else:
                        cbs = len(inner_item)
                else:
                    cbs = inner.shape[0]
                if cbs != batch_size:
                    if verbose:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                    return False
            else:
                if conditioning.shape[0] != batch_size:
                    if verbose:
                        print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
                    return False

        return True

    @torch.no_grad()
    def sample(self, steps, batch_size, shape,
               conditioning=None,
               unconditional_guidance_scale=1., unconditional_conditioning=None,
               x_T=None, mask=None, x0=None,
               quantize_x0=False,
               temperature=1., eta=0.,
               noise_dropout=0.,
               score_corrector=None, corrector_kwargs=None,
               callback=None,
               img_callback=None,
               verbose=True, log_every_t=100,              
               **kwargs
               ):
        self._validate_conditioning(conditioning=conditioning, batch_size=batch_size, verbose=verbose)
        timesteps = self.make_schedule(num_steps=steps, eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        if verbose: print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        sampling_output = self._sampling(conditioning, size, timesteps, **kwargs)
        return sampling_output

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
            print(f"Running DDIM Sampling with {total_steps} timesteps")
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = iter(time_range)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                x_T = img_orig * mask + (1. - mask) * x_T

            outs = self.p_sample(x_T, cond, ts, index, **kwargs)
            x_T, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_T)
                intermediates['pred_x0'].append(pred_x0)
        img = x_T
        return img, intermediates

    @torch.no_grad()
    def calculate_epsilon(self, x, c, t, uc, uc_scale,
                          score_corrector=None,
                          corrector_kwargs=None):
        if uc is None or uc_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert "and" in c
                e_t_uncond = None
                e_factors = []
                if "and" in c:
                    pos_factors = c["and"] # "conjunction"
                    for (scale, factor, mask) in pos_factors:        
                        f_uc = torch.cat([uc, factor])                
                        e_t_uncond, e_i = self.model.apply_model(x_in, t_in, f_uc).chunk(2)
                        e_factors.append(mask * scale * (e_i - e_t_uncond))
                if "not" in c:
                    neg_factors = c["not"] # "negation"
                    for (scale, factor, mask) in neg_factors:
                        f_uc = torch.cat([uc, factor])                
                        e_t_uncond, e_j = self.model.apply_model(x_in, t_in, f_uc).chunk(2)
                        e_factors.append(mask * -scale * (e_j - e_t_uncond))
                e_t = e_t_uncond + uc_scale * sum(e_factors)
            else:
                c_in = torch.cat([uc, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + uc_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        return e_t

    def expand_params(self, index, batch_size, device):
        b = batch_size
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sqrt_one_minus_alphas[index], device=device)

        return a_t, a_prev, sigma_t, sqrt_one_minus_at

    @torch.no_grad()
    def p_sample(self, x, c, t, index, **kwargs):
        quantize_denoised = kwargs.get("quantize_denoised", False)
        temperature = kwargs.get("temperature", 1.)
        noise_dropout = kwargs.get("noise_dropout", 0.)
        score_corrector = kwargs.get("score_corrector", None)
        corrector_kwargs = kwargs.get("corrector_kwargs", None)
        unconditional_guidance_scale = kwargs.get("unconditional_guidance_scale", 1.)
        unconditional_conditioning = kwargs.get("unconditional_conditioning", None)

        b, *_, device = *x.shape, x.device

        e_t = self.calculate_epsilon(x, c, t, unconditional_conditioning, unconditional_conditioning_scale,
                                     score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        
        a_t, a_prev, sigma_t, sqrt_one_minus_at = self.expand_params(index, b, device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        # create noise to re-add
        noise = sigma_t * torch.randn(x.shape, device=device) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # final calculation of previous x - since we going in reverse
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    @torch.no_grad()
    def p_sample_reverse(self, x, c, t, index, 
                         quantize_denoised=False,
                         temperature=1., 
                         noise_dropout=0., 
                         score_corrector=None, corrector_kwargs=None,
                         unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        e_t = self.calculate_epsilon(x, c, t, unconditional_conditioning, unconditional_conditioning_scale,
                                     score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

        a_t, a_prev, sigma_t, sqrt_one_minus_at = self.expand_params(index, b, device)

        # current prediction for x_0
        pred_x0 = (sqrt_one_minus_at * e_t - x) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # final calculation of next x
        x_next = a_prev.sqrt() * pred_x0 + dir_xt - noise

        return x_next, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, **kwargs):
        noise = kwargs.get("noise", None)
        if noise is None:
            noise = torch.randn_like(x0)
        a_t = extract_into_tensor(torch.sqrt(self.alphas), t, x0.shape)
        sqrt_one_minus_at = extract_into_tensor(self.sqrt_one_minus_alphas, t, x0.shape)
        return (a_t * x0 +
                sqrt_one_minus_at * noise)

    @torch.no_grad()
    def encode(self, latent):
        return self.model.get_first_stage_encoding(
            self.model.encode_first_stage(x_latent)
        )
    
    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, 
               unconditional_guidance_scale=1.0, 
               unconditional_conditioning=None,
               mask=None, x0=None,
               verbose=False):

        timesteps = self.timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        if mask is not None and x0 is None:            
            x0 = self.encode(x_latent)
        
        if verbose: 
            print(f"Running DDIM Sampling with {total_steps} timesteps")
            iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        else:
            iterator = iter(time_range)
        
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)

            if mask is not None:                
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                x_T = img_orig * mask + (1. - mask) * x_T

            x_dec, _ = self.p_sample(x_dec, cond, ts, index=index, 
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning)
        return x_dec