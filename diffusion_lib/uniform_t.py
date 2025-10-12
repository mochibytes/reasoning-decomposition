import torch
import torch.nn.functional as F
from torch.distributions import Beta

# from spatial reasoning with denoising models

def recursive_sum_constrained_vector(s, d, sharpness=1.0):
    """
    recursively generate a vector of length d with elements in [0,1]
    whose sum equals s (≈ d * t̄).
    implements Algorithm 1 from Appendix C.1 (Uniform t̄ sampling).
    """
    if d == 1:
        return torch.clamp(s, 0.0, 1.0).unsqueeze(0)

    d1 = d // 2
    d2 = d - d1

    smax1 = min(s, d1)
    smax2 = min(s, d2)
    smin1 = max(0.0, s - smax2)

    # parameters for psplit(r|d) = Beta(α, β), Eq. 26
    alpha = beta = ((d - 1 - (d % 2)) ** 1.05) * sharpness + 1e-8
    beta_dist = Beta(alpha, beta)
    r = beta_dist.sample()
    s1 = smin1 + (smax1 - smin1) * r
    s2 = s - s1

    x1 = recursive_sum_constrained_vector(s1, d1, sharpness)
    x2 = recursive_sum_constrained_vector(s2, d2, sharpness)
    return torch.cat([x1, x2], dim=0)

def sample_noise_levels(batch_size, num_patches, sharpness=1.0, device='cpu'):
    """
    Implements 'Uniform t̄' noise level sampling from Sec. 3.2.1.
    Returns tensor of shape [batch_size, num_patches] with values in [0,1].
    """
    t_list = []
    for _ in range(batch_size):
        t_bar = torch.rand(1).item()  # sample mean t̄ ~ U(0,1)
        s = num_patches * t_bar
        t = recursive_sum_constrained_vector(s, num_patches, sharpness)
        t_list.append(t)
    return torch.stack(t_list).to(device)
