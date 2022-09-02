import PIL
import numpy as np
import torch
from torch import autocast
from pytorch_lightning import seed_everything

from diffuson.samplers import make as make_sampler

class DiffusionModelManager:
    def __init__(self, checkpoint_file, model_key='model'):
        self.model = torch.load(checkpoint_file)[model_key]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(device)
        self.model.eval()
        self.z_channels = model.first_stage_model.encoder.conv_out.weight.shape[0] // 2

    def process_prompt(self, prompt, config):
        # extract out values from config
        shape = (config["shape"]["width"], config["shape"]["height"])
        steps = config["steps"]
        batch_size = config["batch_size"]
        scale = config["scale"]
        seed = config["seed"]
        eta = config["eta"]

        sampler_config = {"name": "DDIM", "args": {}}
        sampler_config = config["sampler"] if "sampler" in config else sampler_config
        assert "args" in sampler_config and "name" in samler_config
        sampler_config["args"]["model"] = self.model
        
        self.sampler = make_sampler(sampler_config)

        uc = self.get_unconditional_embeddings(batch_size=batch_size)
        c = self.get_conditioning_embeddings(prompt)
        assert c.shape == uc.shape

        img = self.render(c, uc=uc,
                          steps=steps,
                          shape=shape,
                          scale=scale,
                          seed=seed,
                          eta=eta)

        img = img.mul(255).to(torch.uint8)

        if format.lower() in ["pil"]:
            img = PIL.Image.fromarray(img.numpy(), 'RGB')
        elif format.lower() in ["numpy", "np"]:
            img = img.numpy()

        return img

    def get_unconditional_embeddings(batch_size=1):
        with torch.no_grad():
            uc = self.model.get_learned_conditioning(batch_size * [""])
        
        return uc
    
    def get_conditioning_embeddings(data):        
        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            return self.model.get_learned_conditioning(data)      

    def render(self, c, 
               uc=None, 
               shape=(512, 512), 
               scale=5.0, 
               start_code=None, 
               steps=1,
               eta=0.0,
               seed=0,
               reset_seed=True):
        if reset_seed: seed_everything(seed)
        
        shape = [self.z_channels, shape[0] // 8, shape[1] // 8]
        samples_x = self.sampler.sample(S=steps,
                                        conditioning=c,
                                        batch_size=c.shape[0],
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=eta,
                                        x_T=start_code)
        x = self.model.decode_first_stage(samples_x)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        img = x.cpu().permute(0, 2, 3, 1).squeeze()
        return img    