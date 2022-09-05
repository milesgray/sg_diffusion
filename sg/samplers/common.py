import torch
from tqdm import trange


class DiffusionSamplerWrapper:
    def __init__(self, name: str, **kwargs):
        constructor = kwargs.get("constructor", DiffusionSampler)
        self.sampler = constructor(kwargs.get("model"))
        self.name = name
        self.batch_size = kwargs.get("batch_size", 1)
        self.width = kwargs.get("width", 512)
        self.height = kwargs.get("height", 512)
        self.z_channels = kwargs.get("z_channels", 4)
        self.scale = kwargs.get("scale", 7.5)
        self.use_start_code = kwargs.get("use_start_code", False)
        self.steps = kwargs.get("steps", 50)
        self.eta = kwargs.get("eta", 0)
        self.temperature = kwargs.get("temperature", 1)
        self.denoising_strength = kwargs.get("denoising_strength", 0.0)

    def to_json(self):
        return {
            "name": self.name,
            "args": {
                "batch_size": self.batch_size,
                "width": self.width,
                "height": self.height,
                "z_channels": self.z_channels,
                "scale": self.scale,
                "use_start_code": self.use_start_code,
                "steps": self.steps,
                "eta": self.eta,
                "temperature": self.temperature,
                "denoising_strength": self.denoising_strength,
            }
        }

    def sample(self, 
               conditioning: torch.Tensor=None, 
               unconditional_conditioning: torch.Tensor=None,
               start_code: torch.Tensor=None):
        shape = [self.z_channels, self.width // 8, self.height // 8]
        if self.use_start_code:
            if start_code is None:
                start_code = torch.randn((self.batch_size,) + shape)
        else:
            start_code = None
        
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            result = self.sampler.sample(steps=self.steps,
                                         conditioning=conditioning,
                                         batch_size=self.batch_size,
                                         shape=self.shape,
                                         verbose=self.verbose,
                                         unconditional_guidance_scale=self.scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         eta=self.eta,
                                         temperature=self.temperature,
                                         x_T=start_code)
            if isinstance(result, tuple):
                samples = result[0]
            else:
                samples = result
        return samples

    def sample_img(self, img, mask, 
                   conditioning=None,    
                   unconditional_conditioning=None, 
                   noise=None):
        self.sampler.make_schedule(num_steps=self.steps, eta=self.eta, verbose=False)
        
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            t_enc = int(min(self.denoising_strength, 0.999) * self.steps)
            t = torch.Tensor([t_enc] * int(img.shape[0]))

            x = self.sampler.stochastic_encode(img, t, noise=noise)
            
            samples = self.sampler.decode(x, conditioning, t_enc, 
                                          unconditional_guidance_scale=self.scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return samples

class DiffusionSampler:
    def __init__(self, model):
        self.model = model

    def make_schedule(self, num_steps, discretize="uniform", eta=0., verbose=True):
        self.timesteps = num_steps
    
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
            return torch.randn(shape)

    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None):
        return x0

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, 
               unconditional_guidance_scale=1.0, 
               unconditional_conditioning=None,
               mask=None, x0=None,
               verbose=False):

        return x_latent

class VanillaStableDiffusionSampler:
    def __init__(self, model, constructor):
        self.sampler = constructor(model)

    def sample(self, S: int,
               conditioning: torch.Tensor=None, 
               batch_size: int=1,
               shape=None,
               verbose: bool=False,
               unconditional_guidance_scale: float=1.0,
               unconditional_conditioning: torch.Tensor=None,
               eta: float=0.0,
               x_T: torch.Tensor=None):
        samples, _ = self.sampler.sample(S=S,
                                    conditioning=conditioning,
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=verbose,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    eta=eta,
                                    x_T=x_T)
        return samples



@torch.no_grad()
def sample(model, x, steps, eta, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

@torch.no_grad()
def cond_sample(model, x, steps, eta, extra_args, cond_fn, callback=None):
    """Draws guided samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            with torch.cuda.amp.autocast():
                v = model(x, ts * steps[i], **extra_args)

            pred = x * alphas[i] - v * sigmas[i]

            # Call the callback
            if callback is not None:
                callback({'x': x, 'i': i, 't': steps[i], 'v': v.detach(), 'pred': pred.detach()})

            if steps[i] < 1:
                cond_grad = cond_fn(x, ts * steps[i], pred, **extra_args).detach()
                v = v.detach() - cond_grad * (sigmas[i] / alphas[i])
            else:
                v = v.detach()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

@torch.no_grad()
def reverse_sample(model, x, steps, extra_args, callback=None):
    """Finds a starting latent that would produce the given image with DDIM
    (eta=0) sampling."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps) - 1, disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # Recombine the predicted noise and predicted denoised image in the
        # correct proportions for the next step
        x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    return x
