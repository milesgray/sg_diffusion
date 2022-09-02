import torch

from sg.samples.util import transfer, eps_model_fn, make_autocast_model_fn



def iplms_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    if len(old_eps) == 0:
        eps_prime = eps
    elif len(old_eps) == 1:
        eps_prime = (3/2 * eps - 1/2 * old_eps[-1])
    elif len(old_eps) == 2:
        eps_prime = (23/12 * eps - 16/12 * old_eps[-1] + 5/12 * old_eps[-2])
    else:
        eps_prime = (55/24 * eps - 59/24 * old_eps[-1] + 37/24 * old_eps[-2] - 9/24 * old_eps[-3])
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


@torch.no_grad()
def iplms_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth order
    Improved Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        x, eps, pred = iplms_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if len(old_eps) >= 3:
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x