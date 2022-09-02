import torch

from sg.samples.util import transfer, eps_model_fn, make_autocast_model_fn


def plms2_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    eps_prime = (3 * eps - old_eps[-1]) / 2
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred

def pie_step(model, x, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps_1 = eps_model_fn(x, t_1, **extra_args)
    x_1, _ = transfer(x, eps_1, t_1, t_2)
    eps_2 = eps_model_fn(x_1, t_2, **extra_args)
    eps_prime = (eps_1 + eps_2) / 2
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred

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
