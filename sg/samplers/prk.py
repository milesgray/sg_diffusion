import torch

from sg.samples.util import transfer, eps_model_fn, make_autocast_model_fn


@torch.no_grad()
def prk_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth-order
    Pseudo Runge-Kutta."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    for i in trange(len(steps) - 1, disable=None):
        x, _, pred = prk_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x

def prk_step(model, x, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    t_mid = (t_2 + t_1) / 2
    eps_1 = eps_model_fn(x, t_1, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = eps_model_fn(x_1, t_mid, **extra_args)
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = eps_model_fn(x_2, t_mid, **extra_args)
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = eps_model_fn(x_3, t_2, **extra_args)
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred    