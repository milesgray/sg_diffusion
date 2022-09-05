import PIL
import numpy as np
import torch
from torch import autocast
from pytorch_lightning import seed_everything

from sg.samplers import make as make_sampler

class DiffusionModelManager:
    def __init__(self, checkpoint_file, model_key='model'):
        self.model = torch.load(checkpoint_file)[model_key]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(device)
        self.model.eval()
        self.z_channels = model.first_stage_model.encoder.conv_out.weight.shape[0] // 2

    def process_prompt(self, prompt, config):
        batch_size = 1 #config.get("batch_size", 1)

        sampler = self.make_sampler(config)

        uc = self.get_unconditional_embeddings(batch_size=batch_size)
        c = self.get_conditioning_embeddings(prompt, batch_size=batch_size)
        assert c.shape == uc.shape

        x = self.render(sampler, c, uc,
                         batch_size=batch_size,
                         steps=config.get("steps", 50),
                         z_channels=shape.get("z_channels", self.z_channels),
                         width=shape.get("width", 512), 
                         height=shape.get("height", 512),
                         scale=config.get("scale", 7.5),
                         seed=config.get("seed", 42),
                         use_start_code=config.get("use_start_code", False),
                         eta=config.get("eta", 0.0))

        img = self.create_image(x)

        return img

    def make_sampler(self, config):
        sampler_config = config.get("sampler", {"name": "DDIM", "args": {}})        
        
        assert "args" in sampler_config and "name" in sampler_config
        
        return make_sampler(sampler_config, args={"model": self.model})

    def get_unconditional_embeddings(batch_size=1):
        with torch.no_grad():
            return self.model.get_learned_conditioning(batch_size * [""])
    
    def get_conditioning_embeddings(prompt, batch_size=1):                
        with torch.no_grad():
            return self.model.get_learned_conditioning(batch_size * prompt)      

    def create_image(self, x):
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        img = x.cpu().squeeze().permute(1, 2, 0)
        img = img.mul(255).to(torch.uint8).numpy()
        return img

    def render(self, sampler, c, uc, 
               batch_size=1,
               width=512,
               height=512,
               z_channels=4, 
               scale=5.0, 
               use_start_code=False, 
               steps=1,
               eta=0.0,
               seed=0,
               reset_seed=True):
        if reset_seed: seed_everything(seed)
        if use_start_code:
            start_code = torch.randn((batch_size,) + shape)
        else:
            start_code = None
        shape = [z_channels, width // 8, height // 8]
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            samples_x = sampler.sample(steps=steps,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=eta,
                                        x_T=start_code)
            x = self.model.decode_first_stage(samples_x)
        
            return x    