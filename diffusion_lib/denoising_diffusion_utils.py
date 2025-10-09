import math
import sys
from collections import namedtuple
import torch

def _custom_exception_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)


def hook_exception_ipdb():
    """Add a hook to ipdb when an exception is raised."""
    if not hasattr(_custom_exception_hook, 'origin_hook'):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    """Remove the hook to ipdb when an exception is raised."""
    assert hasattr(_custom_exception_hook, 'origin_hook')
    sys.excepthook = _custom_exception_hook.origin_hook

hook_exception_ipdb()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float = 0
    avg: float = 0
    sum: float = 0
    sum2: float = 0
    std: float = 0
    count: float = 0
    tot_count: float = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += val * val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count
        self.std = (self.sum2 / self.count - self.avg * self.avg) ** 0.5

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def patch_reshape(x, num_patches, patch_size, to_dims = 2):
    if to_dims == 2:
        return x.reshape(x.shape[0], num_patches * patch_size)
    elif to_dims == 3:
        return x.reshape(x.shape[0], num_patches, patch_size)
    else:
        raise ValueError(f'Invalid number of dimensions {to_dims}')

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def extract_patchwise(a, t_patchwise, x):
    assert x.ndim == 3, f'x must be 3D, got {x.ndim}'
    assert t_patchwise.ndim == 2, f't_patchwise must be 2D, got {t_patchwise.ndim}'
    assert t_patchwise.shape[1] == x.shape[1], f'num_patches must match, got {t_patchwise.shape[1]} and {x.shape[1]}'
    assert t_patchwise.shape[0] == x.shape[0], f'batch size must match, got {t_patchwise.shape[0]} and {x.shape[0]}'
    b, num_patches, patch_size = x.shape
    out = a.gather(-1, t_patchwise.flatten()).reshape(b, num_patches)
    return out.unsqueeze(-1).repeat(1, 1, patch_size) # [B, num_patches, patch_size]

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# TODO: make the new beta schedule from the SRM paper

def set_beta_schedule(beta_schedule, timesteps): # MODULE_ADDITION
    if beta_schedule == 'linear':
        return linear_beta_schedule(timesteps)
    elif beta_schedule == 'cosine':
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {beta_schedule}')

def set_loss_weight(objective, snr): # MODULE_ADDITION
    if objective == 'pred_noise':
        loss_weight = torch.ones_like(snr)
    elif objective == 'pred_x0':
        loss_weight = snr
    elif objective == 'pred_v':
        loss_weight = snr / (snr + 1)
    else:
        raise ValueError(f'unknown objective {objective}')
    return loss_weight
