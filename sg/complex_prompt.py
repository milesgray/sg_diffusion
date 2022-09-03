#@title Complex Prompt
def get_prompt_map(prompt, model=model):
    tokenizer = model.cond_stage_model.tokenizer
    encoding = tokenizer(prompt,  truncation=True, max_length=77, return_length=True,
                                return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = encoding["input_ids"].squeeze()
    prompt_map = [tokenizer.decode(id) for id in encoding["input_ids"].squeeze()]
    return prompt_map

class ComplexPromptEmbedding:
    def __init__(self, prompt: str, scale: float=1.0, mask=None, model=model):
        self.scale = scale if scale else 1.0
        self.mask = mask if mask else 1.0
        self.prompt = prompt
        self.token_map = get_prompt_map(prompt, model=model)
        self.model = model
        self.tokenizer = model.cond_stage_model.tokenizer
        self._raw_embeddings = self._get_conditioning_embeddings()
        self.embeddings = self._raw_embeddings.clone()
        self.path = []
        self.path_embeddings = []
        self.path_history = []
        self.built = False
        self.trend_map = self._build_index_trend_map()

    def _get_conditioning_embeddings(self):
        self.model.eval()        
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            return self.model.get_learned_conditioning(self.prompt)  
    
    def _build_index_trend_map(self):
        index_token_map = defaultdict(list)
        active_hi_idx = []
        active_low_idx = []
        for i in range(self.embeddings.shape[1]):
            token_string = self.token_map[i]
            if '<|end' in token_string: continue
            hi_val,hi_idx = torch.topk(self.embeddings[0,i,:], k=1, dim=0, largest=True)
            low_val,low_idx = torch.topk(self.embeddings[0,i,:], k=1, dim=0, largest=False)    
            hi_idx = hi_idx[0].item()
            low_idx = low_idx[0].item()
            if hi_idx not in active_hi_idx:
                index_token_map[i].append(('high',token_string, hi_idx))
                active_hi_idx.append(hi_idx)
            if low_idx not in active_low_idx:        
                index_token_map[i].append(('low',token_string, low_idx))
                active_low_idx.append(low_idx)
        return index_token_map

    def get_embeddings(self, force=False, verbose=False):
        if self.built or force:
            return self.embeddings
        else:
            return self.build_embeddings(verbose=verbose)

    def build_embeddings(self, steps=1, verbose=False):
        self.built = False
        self.path_history = []
        self.path_embeddings = []
        self.embeddings = self._raw_embeddings.clone()
        for p in self.path:
            new_embeddings_list = p.apply(self, steps=steps, verbose=verbose)
            self.path_embeddings += new_embeddings_list
            new_embeddings = new_embeddings_list[-1]
            step_edist = ((self.embeddings - new_embeddings) ** 2).sqrt().mean()
            step_sdist = get_spherical_dist(self.embeddings, new_embeddings).mean()
            origin_edist = ((self._raw_embeddings - new_embeddings) ** 2).sqrt().mean()
            origin_sdist = get_spherical_dist(self._raw_embeddings, new_embeddings).mean()
            self.path_history.append({"prompt": f"{p.prompt.prompt}", 
                                      "step": {
                                          "euler_dist": step_edist, 
                                          "sphere_dist": step_sdist,
                                      },
                                      "origin": {
                                          "euler_dist": origin_edist,
                                          "sphere_sdist": origin_sdist,
                                      },
                                      "sub_prompt_history": p.prompt.path_history})
            self.embeddings = new_embeddings
        self.built = True
        return self.embeddings

    def add_transform(self, other_prompt, config, transform_cls):
        self.path.append(transform_cls(other_prompt, config))
        self.built = False

class CompositionalPromptEmbedding(ComplexPromptEmbedding):
    def __init__(self, prompt: str, scale: float=1.0, mask=None, model=model):
        super().__init__(prompt, scale=scale, 
                         mask=mask, 
                         model=model)
        self._conjunctions = []
        self._negations = []

    def get_embeddings(self, force=False, verbose=False):
        base_embeddings = super().get_embeddings(force=force, verbose=verbose)
        if len(self._conjunctions) or len(self._negations):
            composition = defaultdict(list)
            composition["and"].append((self.scale, base_embeddings, self.mask))
            
            for conj in self._conjunctions:
                composition["and"].append((conj.scale, conj.get_embeddings(verbose=verbose), conj.mask))
                if verbose: print(f"[{conj.scale}x]\tCONJUNCTION added: {conj.prompt}")
                _edist = ((self.embeddings - conj.embeddings) ** 2).sqrt().mean()
                _sdist = get_spherical_dist(self.embeddings, conj.embeddings).mean()
                self.path_history.append({"prompt": f"{conj.prompt}", 
                                          "mode": "conjunction",
                                          "euler_dist": _edist,
                                          "sphere_dist": _sdist,})
            for neg in self._negations:
                composition["not"].append((neg.scale, neg.get_embeddings(verbose=verbose), neg.mask))
                if verbose: print(f"[{neg.scale}x]\tNEGATION added: {neg.prompt}")
                _edist = ((self.embeddings - neg.embeddings) ** 2).sqrt().mean()
                _sdist = get_spherical_dist(self.embeddings, neg.embeddings).mean()
                self.path_history.append({"prompt": f"{neg.prompt}", 
                                          "mode": "negation",
                                          "euler_dist": _edist,
                                          "sphere_dist": _sdist,})
            return composition
        else:
            return base_embeddings
    
    def add_conjunction(self, prompt: Union[ComplexPromptEmbedding,str], 
                        scale: Union[float,None]=None, 
                        mask: Union[torch.Tensor,np.ndarray,None]=None) -> None:
        if isinstance(prompt, str):
            prompt = ComplexPromptEmbedding(prompt, scale=scale, mask=mask, model=self.model)            
        self._conjunctions.append(prompt)

    def add_negation(self, prompt: Union[ComplexPromptEmbedding,str], 
                     scale: Union[float,None]=None,
                     mask: Union[torch.Tensor,np.ndarray,None]=None) -> None:
        if isinstance(prompt, str):
            prompt = ComplexPromptEmbedding(prompt, scale=scale, mask=mask, model=self.model)            
        self._negations.append(prompt)

class AbstractPromptTransform:
    def __init__(self, prompt: str, config: dict):
        self.prompt = prompt
        self.config = config
        self.param_lerp_keys = config['lerp_keys'] if 'lerp_keys' in config else []
        self.step_results = []

    def apply(self, prompt_start, steps=1, verbose=False): 
        if len(self.param_lerp_keys) == 0 or \
            all([k not in self.config for k in self.param_lerp_keys]):
            steps = 1       
        for s in range(min(1,steps)):
            params = self.lerp_params(self.config, s/steps)
            batch_embedding = self.step(prompt_start, self.prompt, params, verbose=verbose)
            self.step_results.append(batch_embedding)
        return self.step_results

    def step(self, prompt_start, prompt_end, params, verbose=False):
        raise NotImplementedError

    def lerp_params(self, params, amount):
        if amount == 1:
            return params
        result = {}        
        for k,v in params.items():
            if k not in self.param_lerp_keys:
                result[k] = v
            else:
                if isinstance(v, float):
                    result[k] = v * amount
                elif isinstance(v, int):
                    result[k] = int(v * amount)
                elif isinstance(v, tuple):
                    if len(v) != 2: 
                        result[k] = v
                    elif isinstance(v[0], int) and isinstance(v[1], int):
                        v[0] = int(v[0] * amount)
                        v[1] = int(v[1] + v[1] * (1-amount))
                        result[k] = v
                    elif isinstance(v[0], float) and isinstance(v[1], float):
                        v[0] = v[0] * amount
                        v[1] = v[1] + v[1] * (1-amount)
                        result[k] = v
                    else:
                        result[k] = v
                else:
                    result[k] = v
        return result


class LerpPromptTransform(AbstractPromptTransform):
    def __init__(self, c_prompt_end: str, config: dict):
        super().__init__(c_prompt_end, config)

        self.config['magnitude'] = self.config['magnitude'] \
                if 'magnitude' in self.config else 1.0
        self.config['lerp_threshold'] = self.config['lerp_threshold'] \
                if 'lerp_threshold' in self.config else 0.995

        self.config['token_k'] = self.config['token_k'] \
                if 'token_k' in self.config else 1
        self.config['token_idxs'] = self.config['token_idxs'] \
                if 'token_idxs' in self.config else None
        self.config['token_range'] = self.config['token_range'] \
                if 'token_range' in self.config else None        
        self.config['token_largest'] = self.config['token_largest'] \
                if 'token_largest' in self.config else True

        
        self.config['embed_k'] = self.config['embed_k'] \
                if 'embed_k' in self.config else 1
        self.config['embed_idxs'] = self.config['embed_idxs'] \
                if 'embed_idxs' in self.config else None
        self.config['embed_range'] = self.config['embed_range'] \
                if 'embed_range' in self.config else None        
        self.config['embed_largest'] = self.config['embed_largest'] \
                if 'embed_largest' in self.config else True

        self.config['delta_mult'] = self.config['delta_mult'] \
                if 'delta_mult' in self.config else 1.0
        self.config['static_mult'] = self.config['static_mult'] \
                if 'static_mult' in self.config else 1.0

    def step(self, prompt_start, prompt_end, params, verbose=False):
        c_start = prompt_start.get_embeddings(force=True)
        c_end = prompt_end.get_embeddings()
        token_maps = (prompt_start.token_map, prompt_end.token_map)

        assert c_start.shape == c_end.shape

        batch_size = c_start.shape[0] if len(c_start.shape) == 3 else 1

        results = []
        for b in range(batch_size):
            results.append(self._do_step(c_start[b], c_end[b], token_maps, params,
                                         verbose=verbose))
        
        return torch.stack(results)

    def _do_step(self, c_start, c_end, token_maps, params, verbose=False):
        token_idxs = self._get_token_idxs(c_start, c_end, 
                                          token_maps,
                                          token_idxs=params['token_idxs'],
                                          token_range=params['token_range'],
                                          token_sim_k=params['token_sim_k'],
                                          token_sim_largest=params['token_sim_largest'],
                                          verbose=verbose)

        if token_idxs.shape[0] == 0:
            # no tokens selected, so no movement - still apply static multiplier
            result = c_start * params['static_mult']
        else:
            # interpolate - magnitude 1.0 means go all the way to end, 0.0 means don't move
            c_delta = self._slerp(c_start, c_end, params['magnitude'])

            # compute mask to restrict to subset and larger/smallest features to interpolate between
            mask = self._embed_topk_mask(c_delta, token_idxs,
                                        k=params['embed_k'], 
                                        embed_range=params['embed_range'],
                                        embed_idxs=params['embed_idxs'],
                                        largest=params['embed_largest'],
                                        verbose=verbose)
            mask = torch.Tensor(mask).to(c_start.device)    
            # values we want to change 
            delta = c_delta * mask.float()
            delta_max = delta.max().item()
            delta_min = delta.min().item()
            delta = blur(delta)
            delta = torch.clip(delta, delta_min, delta_max)
            # values we want to stay the same
            static = c_start * torch.logical_not(mask).float()
            # combine together with elementwise addition since there is no overlap
            result = delta * params['delta_mult'] + \
                     static * params['static_mult']
            if verbose:
                plt.close()
                c_diff = c_start.sub(result)
                width_ratio = result.shape[1]/result.shape[0]
                fig, axs = plt.subplots(5, 1, figsize=(width_ratio * 4, 5 * 4))
                axs[0].imshow(mask.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[0].axis('off')
                axs[0].set_title("Binary mask - 77 tokens x 768 embedding dimensions")
                axs[1].imshow(delta.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[1].axis('off')
                axs[1].set_title("Masked Delta Embedding - interpolated embedding after masking")
                axs[2].imshow(static.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[2].axis('off')
                axs[2].set_title("Masked Static Embedding - original embedding after masking")
                axs[3].imshow(c_diff.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[3].axis('off')
                axs[3].set_title("Changes - start - result")
                axs[4].imshow(result.cpu().numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                axs[4].axis('off')
                axs[4].set_title("Result Embedding - delta + static (after scaling applied to each)")            
                fig.tight_layout()            
                plt.draw_all()

        return result 

    def _get_range(self, S, range=None, idxs=None, verbose=False):
        range = (0,S) if range is None else range
        range_min = 0 if idxs is None else min(idxs)
        range_max = S if idxs is None else max(idxs)        
        range_start = max(min(range[0], range[1]), range_min)
        range_end = min(max(range[0], range[1]), range_max)
        return {
            "start": range_start, 
            "end": range_end
        }

    def _get_token_idxs(self, embed_start, embed_end, token_maps,
                        token_idxs=None,
                        token_range=None,
                        token_sim_k=None,
                        token_sim_largest=None,
                        verbose=False):
        T, E = embed_start.shape

        if token_sim_k is not None and token_sim_largest is not None:
            # force into 0-token_count range
            k = token_sim_k = max(min(embed_start.shape[0], token_sim_k), 0)
            if k == embed_start.shape[0]:
                # max number of tokens selected
                selected_idxs = np.array([i for i in range(k)])
            elif k == 0:
                # no tokens selected
                selected_idxs = np.array([])
            else:
                # compare token embeddings
                sim = F.cosine_similarity(embed_start, embed_end) 
                if token_sim_largest:
                    token_repeat_mask = torch.Tensor(
                        [int(p1 != p2) for (p1,p2) 
                            in zip(token_maps[0], 
                                   token_maps[1])
                            ]).cuda()
                    sim *= token_repeat_mask
                selected_idxs = torch.topk(sim, 
                                        k=k, 
                                        dim=0,
                                        largest=token_sim_largest)[1]
                selected_idxs = selected_idxs.cpu().numpy()
        elif token_range is not None:
            token_range = self._get_range(T, range=token_range, idxs=token_idxs)
            selected_idxs = [i for i in range(token_range['start'], token_range['end'])] \
                            if token_idxs is None else token_idxs
            selected_idxs = np.array(selected_idxs)
        else:
            selected_idxs = [] if token_idxs is None else token_idxs
            selected_idxs = np.array(selected_idxs)

        if verbose:            
            print(f"embed k: {token_sim_k}\tselected max: {selected_idxs.max()} min: {selected_idxs.min()}") 

        return selected_idxs

    def _embed_topk_mask(self, embeddings, token_idxs,
                         k=None, 
                         embed_range=None, 
                         embed_idxs=None,
                         largest=True,
                         verbose=False):
        T, E = embeddings.shape
        
        embed_range = self._get_range(E, range=embed_range, idxs=embed_idxs)
        
        total_embed_idxs = embed_idxs.shape[0] if embed_idxs is not None else \
                embed_range['end']-embed_range['start']
        k = k if k else total_embed_idxs
        # force into 0 - total embedding indexes range        
        k = max(min(k,total_embed_idxs), 0)        
        
        embed_idxs = embed_idxs if embed_idxs is not None else \
            [i for i in range(embed_range['start'], embed_range['end'])]
                        
        # select top/bottom k values from each token embedding specified in token_idxs
        embeddings_slice = embeddings[token_idxs, embed_range['start']:embed_range['end']]                
        selected_idxs = torch.topk(embeddings_slice, 
                                   k=k, 
                                   dim=1, 
                                   largest=largest)[1].cpu().numpy()
        # shift indexes over by the range min, to account for slice index reset
        if embed_range['start'] > 0:
            selected_idxs = [[i2+embed_range['start'] for i2 in i] for i in selected_idxs]
            selected_idxs = np.array(selected_idxs)
        if verbose:            
            print(f"embed k: {k}\tembed range: {embed_range['start']} - {embed_range['end']}\tembedding slice shape: {embeddings_slice.shape}\tselected max: {selected_idxs.max()} min: {selected_idxs.min()}") 
        # create a lookup mapping between the 0-N index that top_idxs uses to actual token index
        token_idx_lookup = {n:i for (i,n) in enumerate(token_idxs)}
        # creates a TxR size array of bools with True only when 
        # t is in token_idxs and r is in the top_idx for that token
        mask = np.array([
            [(t in token_idxs) and 
             (r in selected_idxs[token_idx_lookup[t]]) and
             (r in embed_idxs)
                for r in range(E)
            ] for t in range(T)
        ])
        return mask  

    def _slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float):
        device = v0.device
        v0 = v0.detach().cpu().numpy()
        v1 = v1.detach().cpu().numpy()
        
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > self.config['lerp_threshold']:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        v2 = torch.from_numpy(v2).to(device)

        return v2