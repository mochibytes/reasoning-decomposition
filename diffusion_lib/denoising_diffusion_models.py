from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from einops import reduce

from tqdm.auto import tqdm

from diffusion_lib.denoising_diffusion_utils import * # MODULE_REARRANGEMENT
from diffusion_lib.denoising_diffusion_eval_metrics import * # MODULE_REARRANGEMENT
from diffusion_lib.uniform_t import sample_noise_levels

class PatchGaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        noising_scheme = 'random',
        sharpness = 1.0,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        # auto_normalize = True,
        supervise_energy_landscape = True,
        use_innerloop_opt = True,
        show_inference_tqdm = True,
        baseline = False,
        sudoku = False,
        continuous = False,
        connectivity = False,
        shortest_path = False,
        patch_baseline = False,
        energy_weight_gt = 0.2,
    ):
        # PATCHWISE_EDIT_DONE
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )
        self.patch_size = self.model.patch_size
        self.num_patches = self.model.out_dim // self.patch_size
        
        assert self.out_dim == self.num_patches * self.patch_size, 'out_dim must be divisible by patch_size'

        self.self_condition = False

        self.supervise_energy_landscape = supervise_energy_landscape
        self.use_innerloop_opt = use_innerloop_opt
        self.seq_length = seq_length
        self.objective = objective
        self.show_inference_tqdm = show_inference_tqdm
        self.num_timesteps = int(timesteps)
        self.baseline = baseline
        self.sudoku = sudoku
        self.connectivity = connectivity
        self.continuous = continuous
        self.shortest_path = shortest_path
        self.patch_baseline = patch_baseline
        self.energy_weight_gt = energy_weight_gt
        self.noising_scheme = noising_scheme
        self.sharpness = sharpness

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        betas = set_beta_schedule(beta_schedule, timesteps) # MODULE_ADDITION

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Step size for optimizing
        register_buffer('opt_step_size', betas * torch.sqrt( 1 / (1 - alphas_cumprod)))
        # register_buffer('opt_step_size', 0.25 * torch.sqrt(alphas_cumprod) * torch.sqrt(1 / alphas_cumprod -1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        loss_weight = set_loss_weight(objective, snr) # MODULE_ADDITION

        register_buffer('loss_weight', loss_weight)
        
        # whether to autonormalize (unfinished)

    def predict_start_from_noise(self, x_t, t_patchwise, noise):
        # PATCHWISE_EDIT_DONE
        if x_t.ndim != 3:
            x_t = patch_reshape(x_t, self.num_patches, self.patch_size, to_dims = 3)
        if t_patchwise.ndim != 2:
            t_patchwise = patch_reshape(t_patchwise, self.num_patches, 1, to_dims = 2)
        if noise.ndim != 3:
            noise = patch_reshape(noise, self.num_patches, self.patch_size, to_dims = 3)

        out_3d = (
            extract_patchwise(self.sqrt_recip_alphas_cumprod, t_patchwise, x_t) * x_t -
            extract_patchwise(self.sqrt_recipm1_alphas_cumprod, t_patchwise, x_t) * noise
        ) # [B, num_patches, patch_size]
        out_2d = patch_reshape(out_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        return out_2d

    def predict_noise_from_start(self, x_t, t_patchwise, x0):
        # PATCHWISE_EDIT_DONE
        if x_t.ndim != 3:
            x_t = patch_reshape(x_t, self.num_patches, self.patch_size, to_dims = 3)
        if t_patchwise.ndim != 2:
            t_patchwise = patch_reshape(t_patchwise, self.num_patches, 1, to_dims = 2)
        if x0.ndim != 3:
            x0 = patch_reshape(x0, self.num_patches, self.patch_size, to_dims = 3)

        out_3d = (
            (extract_patchwise(self.sqrt_recip_alphas_cumprod, t_patchwise, x_t) * x_t - x0) / \
            extract_patchwise(self.sqrt_recipm1_alphas_cumprod, t_patchwise, x_t)
        ) # [B, num_patches, patch_size]
        out_2d = patch_reshape(out_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        return out_2d

    def predict_v(self, x_start, t_patchwise, noise):
        # PATCHWISE_EDIT_DONE
        if x_start.ndim != 3:
            x_start = patch_reshape(x_start, self.num_patches, self.patch_size, to_dims = 3)
        if t_patchwise.ndim != 2:
            t_patchwise = patch_reshape(t_patchwise, self.num_patches, 1, to_dims = 2)
        if noise.ndim != 3:
            noise = patch_reshape(noise, self.num_patches, self.patch_size, to_dims = 3)

        out_3d = (
            extract_patchwise(self.sqrt_alphas_cumprod, t_patchwise, x_start) * noise -
            extract_patchwise(self.sqrt_one_minus_alphas_cumprod, t_patchwise, x_start) * x_start
        ) # [B, num_patches, patch_size]

        out_2d = patch_reshape(out_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        return out_2d

    def predict_start_from_v(self, x_t, t_patchwise, v):
        # PATCHWISE_EDIT_DONE
        if x_t.ndim != 3:
            x_t = patch_reshape(x_t, self.num_patches, self.patch_size, to_dims = 3)
        if t_patchwise.ndim != 2:
            t_patchwise = patch_reshape(t_patchwise, self.num_patches, 1, to_dims = 2)
        if v.ndim != 3:
            v = patch_reshape(v, self.num_patches, self.patch_size, to_dims = 3)
        out_3d = (
            extract_patchwise(self.sqrt_alphas_cumprod, t_patchwise, x_t) * x_t -
            extract_patchwise(self.sqrt_one_minus_alphas_cumprod, t_patchwise, x_t) * v
        ) # [B, num_patches, patch_size]

        out_2d = patch_reshape(out_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        return out_2d

    def q_posterior(self, x_start, x_t, t_patchwise):
        # PATCHWISE_EDIT_DONE
        if x_start.ndim != 3:
            x_start = patch_reshape(x_start, self.num_patches, self.patch_size, to_dims = 3)
        if t_patchwise.ndim != 2:
            t_patchwise = patch_reshape(t_patchwise, self.num_patches, 1, to_dims = 2)
        if x_t.ndim != 3:
            x_t = patch_reshape(x_t, self.num_patches, self.patch_size, to_dims = 3)

        posterior_mean_3d = (
            extract_patchwise(self.posterior_mean_coef1, t_patchwise, x_t) * x_start +
            extract_patchwise(self.posterior_mean_coef2, t_patchwise, x_t) * x_t
        ) # [B, num_patches, patch_size]
        posterior_variance_3d = extract_patchwise(self.posterior_variance, t_patchwise, x_t) # [B, num_patches, patch_size]
        posterior_log_variance_clipped_3d = extract_patchwise(self.posterior_log_variance_clipped, t_patchwise, x_t) # [B, num_patches, patch_size]

        posterior_mean = patch_reshape(posterior_mean_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        posterior_variance = patch_reshape(posterior_variance_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        posterior_log_variance_clipped = patch_reshape(posterior_log_variance_clipped_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]

        return posterior_mean, posterior_variance, posterior_log_variance_clipped # each is [B, num_patches * patch_size]

    def model_predictions(self, inp, x, t_patchwise, clip_x_start = False, rederive_pred_noise = False):
        # PATCHWISE_EDIT_DONE
        with torch.enable_grad():
            model_output = self.model(inp, x, t_patchwise) # [B, num_patches * patch_size] since return_energy and return_both are false

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output # [B, num_patches * patch_size], gets reshaped in predict_start_from_noise
            x_start = self.predict_start_from_noise(x, t_patchwise, pred_noise) # [B, num_patches * patch_size]
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t_patchwise, x_start) # [B, num_patches * patch_size]

        elif self.objective == 'pred_x0':
            x_start = model_output # [B, num_patches * patch_size], gets reshaped in predict_noise_from_start
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_patchwise, x_start) # [B, num_patches * patch_size]

        elif self.objective == 'pred_v':
            v = model_output # [B, num_patches * patch_size], gets reshaped in predict_start_from_v
            x_start = self.predict_start_from_v(x, t_patchwise, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_patchwise, x_start) # [B, num_patches * patch_size]

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, cond, x, t_patchwise, x_self_cond = None, clip_denoised = False):
        # PATCHWISE_EDIT_DONE
        preds = self.model_predictions(cond, x, t_patchwise, x_self_cond)
        x_start = preds.pred_x_start # [B, num_patches * patch_size]

        if clip_denoised:
            # x_start.clamp_(-6, 6)

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            x_start.clamp_(-sf, sf)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t_patchwise = t_patchwise)
        return model_mean, posterior_variance, posterior_log_variance, x_start # all are [B, num_patches * patch_size]

    @torch.no_grad()
    def p_sample(self, cond, x, t_patchwise, x_self_cond = None, clip_denoised = True, with_noise=False, scale=False):
        # PATCHWISE_EDIT_DONE
        b, *_, device = *x.shape, x.device

        if type(t_patchwise) == int:
            batched_times = torch.full((b, self.num_patches), t_patchwise, device = x.device, dtype = torch.long) # [B, num_patches]
            noise = torch.randn_like(x) if t_patchwise > 0 else 0.  # no noise if t == 0
        elif t_patchwise.shape != (b, self.num_patches):
            raise ValueError(f'Invalid shape for t_patchwise: {t_patchwise.shape}')
        else:
            batched_times = t_patchwise # [B, num_patches]
            noise = torch.randn_like(x)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t_patchwise = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)

        # Don't scale inputs by expansion factor (Do that later)
        if not scale:
            x_start = patch_reshape(x_start, self.num_patches, self.patch_size, to_dims = 3)
            model_mean = extract_patchwise(self.sqrt_alphas_cumprod, batched_times, x_start) * x_start # [B, num_patches, patch_size]
            model_mean = patch_reshape(model_mean, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]

        if with_noise:
            pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise # [B, num_patches * patch_size]  
        else:
            pred_img = model_mean #  + (0.5 * model_log_variance).exp() * noise # [B, num_patches * patch_size]

        return pred_img, x_start # both are [B, num_patches * patch_size]

    def opt_step(self, inp, img, t_patchwise, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
        # PATCHWISE_EDIT_DONE
        with torch.enable_grad():
            for i in range(step):
                energy, grad = self.model(inp, img, t_patchwise, return_both=True) # energy is [B, 1] and grad is [B, num_patches * patch_size]
                grad_3d = patch_reshape(grad, self.num_patches, self.patch_size, to_dims = 3)
                img_3d = patch_reshape(img, self.num_patches, self.patch_size, to_dims = 3)
                img_new_3d = img_3d - extract_patchwise(self.opt_step_size, t_patchwise, grad_3d) * grad_3d * sf  # / (i + 1) ** 0.5 # [B, num_patches, patch_size]
                img_new = patch_reshape(img_new_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]

                if mask is not None:
                    img_new = img_new * (1 - mask) + mask * data_cond

                if self.continuous:
                    sf = 2.0
                else:
                    sf = 1.0

                img_new_masked_3d = patch_reshape(img_new, self.num_patches, self.patch_size, to_dims = 3)
                max_val = extract_patchwise(self.sqrt_alphas_cumprod, t_patchwise, img_new_masked_3d)[0, 0, 0] * sf # scalar
                img_new_masked_3d = torch.clamp(img_new_masked_3d, -max_val, max_val)
                img_new = patch_reshape(img_new_masked_3d, self.num_patches, self.patch_size, to_dims = 2)

                energy_new = self.model(inp, img_new, t_patchwise, return_energy=True) # [B, 1]
                if len(energy_new.shape) == 2:
                    bad_step = (energy_new > energy)[:, 0]
                elif len(energy_new.shape) == 1:
                    bad_step = (energy_new > energy)
                else:
                    raise ValueError('Bad shape for energy_new!!!')

                # print("step: ", i, bad_step.float().mean())
                img_new[bad_step] = img[bad_step]

                if eval:
                    img = img_new.detach()
                else:
                    img = img_new

        return img

    @torch.no_grad()
    def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
        # PATCHWISE_EDIT_DONE
        device = self.betas.device

        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device) # [B, num_patches * patch_size]

        x_start = None


        if self.show_inference_tqdm:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))

        preds = []

        for t in iterator:
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((batch_size, self.num_patches), t, device = inp.device, dtype = torch.long) # [B, num_patches]

            cond_val = None
            if mask is not None:
                cond_val = self.q_sample(x_start = inp, t_patchwise = batched_times, noise = torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask # [B, num_patches * patch_size]

            img, x_start = self.p_sample(inp, img, batched_times, self_cond, scale=False, with_noise=self.baseline)

            if mask is not None:
                img = img * (1 - mask) + cond_val * mask

            # if t < 50:

            if self.sudoku:
                step = 20
            else:
                step = 5

            if self.use_innerloop_opt:
                if t < 1:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
                else:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)

                img = img.detach()

            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            # This clip threshold needs to be adjust to be larger for generalizations settings
            x_start_3d = patch_reshape(x_start, self.num_patches, self.patch_size, to_dims = 3)
            max_val = extract_patchwise(self.sqrt_alphas_cumprod, batched_times, x_start_3d)[0, 0, 0] * sf

            img = torch.clamp(img, -max_val, max_val) # [B, num_patches * patch_size]

            # Correctly scale output
            img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img)) # [B, num_patches * patch_size]
            preds.append(img_unscaled)

            batched_times_prev = batched_times - 1

            if t != 0:
                img_unscaled_3d = patch_reshape(img_unscaled, self.num_patches, self.patch_size, to_dims = 3)
                img_3d = extract_patchwise(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled_3d) * img_unscaled_3d
                img = patch_reshape(img_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
            # img, _, _ = self.q_posterior(img_unscaled, img, batched_times)

        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img

    @torch.no_grad()
    def sample(self, x, label, mask, batch_size = 16, return_traj=False):
        # PATCHWISE_EDIT_DONE
        # seq_length, channels = self.seq_length, self.channels
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # return sample_fn(batch_size, self.out_shape, x, label, mask, return_traj=return_traj)
        return self.p_sample_loop(batch_size, self.out_shape, x, label, mask, return_traj=return_traj)

    def q_sample(self, x_start, t_patchwise, noise=None):
        # PATCHWISE_EDIT_DONE
        if x_start.ndim != 3:
            x_start = patch_reshape(x_start, self.num_patches, self.patch_size, to_dims = 3)
        if t_patchwise.ndim != 2:
            t_patchwise = patch_reshape(t_patchwise, self.num_patches, 1, to_dims = 2)

        noise = default(noise, lambda: torch.randn_like(x_start))
        if noise.ndim != 3:
            noise = patch_reshape(noise, self.num_patches, self.patch_size, to_dims = 3)


        out_3d = (
            extract_patchwise(self.sqrt_alphas_cumprod, t_patchwise, x_start) * x_start +
            extract_patchwise(self.sqrt_one_minus_alphas_cumprod, t_patchwise, x_start) * noise
        )
        out_2d = patch_reshape(out_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        return out_2d

    def p_losses(self, inp, x_start, mask, t_patchwise, noise = None):
        # PATCHWISE_EDIT_DONE
        b, *c = x_start.shape # B, num_patches * patch_size
        noise = default(noise, lambda: torch.randn_like(x_start)) # [B, num_patches * patch_size]

        # noise sample
        x = self.q_sample(x_start = x_start, t_patchwise = t_patchwise, noise = noise) # [B, num_patches * patch_size]

        if mask is not None:
            # Mask out inputs
            x_cond = self.q_sample(x_start = inp, t_patchwise = t_patchwise, noise = torch.zeros_like(noise))
            x = x * (1 - mask) + mask * x_cond

        # predict and take gradient step

        model_out = self.model(inp, x, t_patchwise) # [B, num_patches * patch_size]

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t_patchwise, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if mask is not None:
            # Mask out targets
            model_out = model_out * (1 - mask) + mask * target # [B, num_patches * patch_size]


        loss = F.mse_loss(model_out, target, reduction = 'none') # [B, num_patches * patch_size]

        if self.shortest_path:
            mask1 = (x_start > 0)
            mask2 = torch.logical_not(mask1)
            # mask1, mask2 = mask1.float(), mask2.float()
            weight = mask1 * 10 + mask2 * 0.5
            # loss = (loss * weight) / weight.sum() * target.numel()
            loss = loss * weight

        loss_3d = patch_reshape(loss, self.num_patches, self.patch_size, to_dims = 3)
        loss_patchwise_3d = loss_3d * extract_patchwise(self.loss_weight, t_patchwise, loss_3d) # [B, num_patches, patch_size], need to reweight by patchwise t
        loss_patchwise = patch_reshape(loss_patchwise_3d, self.num_patches, self.patch_size, to_dims = 2) # [B, num_patches * patch_size]
        
        loss = reduce(loss_patchwise, 'b ... -> b (...)', 'mean') # [B]
        loss_mse = loss # [B]

        if self.supervise_energy_landscape:
            noise = torch.randn_like(x_start)
            data_sample = self.q_sample(x_start = x_start, t_patchwise = t_patchwise, noise = noise)

            if mask is not None:
                data_cond = self.q_sample(x_start = x_start, t_patchwise = t_patchwise, noise = torch.zeros_like(noise))
                data_sample = data_sample * (1 - mask) + mask * data_cond

            # Add a noise contrastive estimation term with samples drawn from the data distribution
            #noise = torch.randn_like(x_start)

            # Optimize a sample using gradient descent on energy landscape
            xmin_noise = self.q_sample(x_start = x_start, t_patchwise = t_patchwise, noise = 3.0 * noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
            else:
                data_cond = None

            if self.sudoku:
                s = x_start.size()
                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
                xmin_noise_im = (xmin_noise_im - 0.5) * 2

                xmin_noise_rescale = xmin_noise_im.view(-1, 729)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.connectivity:
                s = x_start.size()
                x_start_im = x_start.view(-1, 12, 12)
                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.shortest_path:
                x_start_list = x_start.argmax(dim=2)
                classes = x_start.size(2)
                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)

                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2

                loss_opt = torch.ones(1)

                loss_scale = 0.5
            else:

                xmin_noise = self.opt_step(inp, xmin_noise, t_patchwise, mask, data_cond, step=2, sf=1.0)
                x_start_3d = patch_reshape(x_start, self.num_patches, self.patch_size, to_dims = 3)
                xmin_3d = extract_patchwise(self.sqrt_alphas_cumprod, t_patchwise, x_start_3d) * x_start_3d
                xmin = patch_reshape(xmin_3d, self.num_patches, self.patch_size, to_dims = 2)

                loss_opt = torch.pow(xmin_noise - xmin, 2).mean()

                xmin_noise = xmin_noise.detach()
                xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t_patchwise, torch.zeros_like(xmin_noise))
                xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)

                # loss_opt = torch.ones(1)


                # rand_mask = (torch.rand(x_start.size(), device=x_start.device) < 0.2).float()

                # xmin_noise_rescale =  x_start * (1 - rand_mask) + rand_mask * x_start_noise

                # nrep = 1


                loss_scale = 0.5

            xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t_patchwise=t_patchwise, noise=noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

            # Compute energy of both distributions
            inp_concat = torch.cat([inp, inp], dim=0) # [2 * B, inp_dim] = [2 * B, num_patches * patch_size]
            x_concat = torch.cat([data_sample, xmin_noise], dim=0) # [2 * B, num_patches * patch_size]
            # x_concat = torch.cat([xmin, xmin_noise_min], dim=0)
            t_concat = torch.cat([t_patchwise, t_patchwise], dim=0) # [2 * B, num_patches]
            energy = self.model(inp_concat, x_concat, t_concat, return_energy=True) # [2 * B, 1]

            # Compute noise contrastive energy loss
            energy_real, energy_fake = torch.chunk(energy, 2, 0) # each is [B, 1]
            energy_stack = torch.cat([energy_real, energy_fake], dim=-1) # [B, 2]
            target = torch.zeros(energy_real.size(0)).to(energy_stack.device) # [B]
            loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None] # [B, 1]

            # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()

            loss = loss_mse + loss_scale * loss_energy # + 0.001 * loss_opt #[B, 1]
            return loss.mean(), (loss_mse.mean(), loss_energy.mean(), loss_opt.mean()) # constant, mean over batch
        else:
            loss = loss_mse
            return loss.mean(), (loss_mse.mean(), -1, -1)

    def forward(self, inp, target, mask, *args, **kwargs):
        # PATCHWISE_EDIT_DONE
        b, *c = target.shape
        # print(f"inp shape: {inp.shape}")
        # print(f"target shape: {target.shape}")
        device = target.device
        if len(c) == 1:
            self.out_dim = c[0]
            self.out_shape = c
        else:
            print(f"Warning: out_shape is {c}, but only the last dimension is used")
            self.out_dim = c[-1]
            self.out_shape = c
        if self.patch_baseline:
            t_patchwise = torch.randint(0, self.num_timesteps, (b,), device=device) # create tensor of size b
            t_patchwise = t_patchwise.unsqueeze(1).expand(-1, self.num_patches).to(device) # [B, num_patches]
        elif self.noising_scheme == 'random':
            t_patchwise = torch.randint(0, self.num_timesteps, (b, self.num_patches), device=device).long() # [B, num_patches]
        elif self.noising_scheme == 'uniform-t':
            t_patchwise = sample_noise_levels(b, self.num_patches, self.sharpness, device=device).long()
        else:
            raise ValueError(f'Unknown noising scheme: {self.noising_scheme}')

        return self.p_losses(inp, target, mask, t_patchwise, *args, **kwargs)
        


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        # auto_normalize = True,
        supervise_energy_landscape = True,
        use_innerloop_opt = True,
        show_inference_tqdm = True,
        baseline = False,
        sudoku = False,
        continuous = False,
        connectivity = False,
        shortest_path = False,
    ):
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )
        self.self_condition = False
        self.supervise_energy_landscape = supervise_energy_landscape
        self.use_innerloop_opt = use_innerloop_opt

        self.seq_length = seq_length
        self.objective = objective
        self.show_inference_tqdm = show_inference_tqdm
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        betas = set_beta_schedule(beta_schedule, timesteps) # MODULE_ADDITION

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.num_timesteps = int(timesteps)
        self.baseline = baseline
        self.sudoku = sudoku
        self.connectivity = connectivity
        self.continuous = continuous
        self.shortest_path = shortest_path

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Step size for optimizing
        register_buffer('opt_step_size', betas * torch.sqrt( 1 / (1 - alphas_cumprod)))
        # register_buffer('opt_step_size', 0.25 * torch.sqrt(alphas_cumprod) * torch.sqrt(1 / alphas_cumprod -1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        loss_weight = set_loss_weight(objective, snr) # MODULE_ADDITION

        register_buffer('loss_weight', loss_weight)
        # whether to autonormalize

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, inp, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        with torch.enable_grad():
            model_output = self.model(inp, x, t)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, cond, x, t, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(cond, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            # x_start.clamp_(-6, 6)

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            x_start.clamp_(-sf, sf)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, cond, x, t, x_self_cond = None, clip_denoised = True, with_noise=False, scale=False):
        b, *_, device = *x.shape, x.device

        if type(t) == int:
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        else:
            batched_times = t
            noise = torch.randn_like(x)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)

        # Don't scale inputs by expansion factor (Do that later)
        if not scale:
            model_mean = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start

        if with_noise:
            pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean #  + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
        with torch.enable_grad():
            for i in range(step):
                energy, grad = self.model(inp, img, t, return_both=True)
                img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf  # / (i + 1) ** 0.5

                if mask is not None:
                    img_new = img_new * (1 - mask) + mask * data_cond

                if self.continuous:
                    sf = 2.0
                else:
                    sf = 1.0

                max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
                img_new = torch.clamp(img_new, -max_val, max_val)

                energy_new = self.model(inp, img_new, t, return_energy=True)
                if len(energy_new.shape) == 2:
                    bad_step = (energy_new > energy)[:, 0]
                elif len(energy_new.shape) == 1:
                    bad_step = (energy_new > energy)
                else:
                    raise ValueError('Bad shape!!!')

                # print("step: ", i, bad_step.float().mean())
                img_new[bad_step] = img[bad_step]

                if eval:
                    img = img_new.detach()
                else:
                    img = img_new

        return img

    @torch.no_grad()
    def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
        device = self.betas.device

        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)

        x_start = None


        if self.show_inference_tqdm:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))

        preds = []

        for t in iterator:
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((img.shape[0],), t, device = inp.device, dtype = torch.long)

            cond_val = None
            if mask is not None:
                cond_val = self.q_sample(x_start = inp, t = batched_times, noise = torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask

            img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline)

            if mask is not None:
                img = img * (1 - mask) + cond_val * mask

            # if t < 50:

            if self.sudoku:
                step = 20
            else:
                step = 5

            if self.use_innerloop_opt:
                if t < 1:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
                else:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)

                img = img.detach()

            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            # This clip threshold needs to be adjust to be larger for generalizations settings
            max_val = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape)[0, 0] * sf

            img = torch.clamp(img, -max_val, max_val)

            # Correctly scale output
            img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
            preds.append(img_unscaled)

            batched_times_prev = batched_times - 1

            if t != 0:
                img = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled.shape) * img_unscaled
            # img, _, _ = self.q_posterior(img_unscaled, img, batched_times)

        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def sample(self, x, label, mask, batch_size = 16, return_traj=False):
        # seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(batch_size, self.out_shape, x, label, mask, return_traj=return_traj)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, inp, x_start, mask, t, noise = None):
        b, *c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        if mask is not None:
            # Mask out inputs
            x_cond = self.q_sample(x_start = inp, t = t, noise = torch.zeros_like(noise))
            x = x * (1 - mask) + mask * x_cond

        # predict and take gradient step

        model_out = self.model(inp, x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if mask is not None:
            # Mask out targets
            model_out = model_out * (1 - mask) + mask * target

        loss = F.mse_loss(model_out, target, reduction = 'none')

        if self.shortest_path:
            mask1 = (x_start > 0)
            mask2 = torch.logical_not(mask1)
            # mask1, mask2 = mask1.float(), mask2.float()
            weight = mask1 * 10 + mask2 * 0.5
            # loss = (loss * weight) / weight.sum() * target.numel()
            loss = loss * weight

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss_mse = loss

        if self.supervise_energy_landscape:
            noise = torch.randn_like(x_start)
            data_sample = self.q_sample(x_start = x_start, t = t, noise = noise)

            if mask is not None:
                data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
                data_sample = data_sample * (1 - mask) + mask * data_cond

            # Add a noise contrastive estimation term with samples drawn from the data distribution
            #noise = torch.randn_like(x_start)

            # Optimize a sample using gradient descent on energy landscape
            xmin_noise = self.q_sample(x_start = x_start, t = t, noise = 3.0 * noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
            else:
                data_cond = None

            if self.sudoku:
                s = x_start.size()
                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
                xmin_noise_im = (xmin_noise_im - 0.5) * 2

                xmin_noise_rescale = xmin_noise_im.view(-1, 729)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.connectivity:
                s = x_start.size()
                x_start_im = x_start.view(-1, 12, 12)
                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.shortest_path:
                x_start_list = x_start.argmax(dim=2)
                classes = x_start.size(2)
                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)

                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2

                loss_opt = torch.ones(1)

                loss_scale = 0.5
            else:

                xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=2, sf=1.0)
                xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                loss_opt = torch.pow(xmin_noise - xmin, 2).mean()

                xmin_noise = xmin_noise.detach()
                xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
                xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)

                # loss_opt = torch.ones(1)


                # rand_mask = (torch.rand(x_start.size(), device=x_start.device) < 0.2).float()

                # xmin_noise_rescale =  x_start * (1 - rand_mask) + rand_mask * x_start_noise

                # nrep = 1


                loss_scale = 0.5

            xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

            # Compute energy of both distributions
            inp_concat = torch.cat([inp, inp], dim=0)
            x_concat = torch.cat([data_sample, xmin_noise], dim=0)
            # x_concat = torch.cat([xmin, xmin_noise_min], dim=0)
            t_concat = torch.cat([t, t], dim=0)
            energy = self.model(inp_concat, x_concat, t_concat, return_energy=True)

            # Compute noise contrastive energy loss
            energy_real, energy_fake = torch.chunk(energy, 2, 0)
            energy_stack = torch.cat([energy_real, energy_fake], dim=-1)
            target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
            loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None]

            # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()

            loss = loss_mse + loss_scale * loss_energy # + 0.001 * loss_opt
            return loss.mean(), (loss_mse.mean(), loss_energy.mean(), loss_opt.mean())
        else:
            loss = loss_mse
            return loss.mean(), (loss_mse.mean(), -1, -1)

    def forward(self, inp, target, mask, *args, **kwargs):
        b, *c = target.shape
        device = target.device
        if len(c) == 1:
            self.out_dim = c[0]
            self.out_shape = c
        else:
            self.out_dim = c[-1]
            self.out_shape = c

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(inp, target, mask, t, *args, **kwargs)

