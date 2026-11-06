from dataclasses import dataclass, field
from typing import Any, List, Optional

from jaxtyping import Float, Int
import torch
Tensor = torch.Tensor
import torch.nn as nn
import torch.nn.functional as F

from SDS.guidance.misc import C, cleanup, parse_structured
from omegaconf import OmegaConf
from einops import rearrange
import importlib
import sys

sys.modules.setdefault("lvdm", importlib.import_module("SDS.videocrafter.lvdm"))
sys.modules.setdefault("utils.utils", importlib.import_module("SDS.videocrafter.utils.utils"))

class VideoCrafterGuidance:
    @dataclass
    class Config:
        config: str = None
        pretrained_model_name_or_path: Optional[str] = None
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None 
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True
        low_ram_vae: int = -1
        width_vid: int = 128
        height_vid: int = 80

        motion_amp_scale: float = 1.0

        fps: int = 28

        negative_prompt: str = "low motion, static statue, not moving, no motion"

        device: str = "cuda"

        rgb_as_latents: bool = False

    cfg: Config

    def __init__(self, cfg, *args, **kwargs):
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)
       
    def configure(self) -> None:
        from SDS.videocrafter.utils.utils import instantiate_from_config
        from SDS.videocrafter.scripts.evaluation.funcs import load_model_checkpoint
        from SDS.videocrafter.lvdm.models.samplers.ddim import DDIMSampler

        print("Loading VideoCrafter ...")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        config = OmegaConf.load(self.cfg.config)
        self.device = self.cfg.device
        self.model_config = config.pop("model", OmegaConf.create())
        self.model = instantiate_from_config(self.model_config)
        self.model = self.model.to(self.device)

        self.model = load_model_checkpoint(self.model, self.cfg.pretrained_model_name_or_path)
        
        self.rgb_as_latents = self.cfg.rgb_as_latents

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
            
        cleanup()
        
        self.scheduler = DDIMSampler(self.model)
        self.num_train_timesteps = self.model_config.params.timesteps
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

        self.alphas = self.model.alphas_cumprod
        self.text_encoder = self.model.cond_stage_model

        self.grad_clip_val: Optional[float] = None

        print("Loaded VideoCrafter!")

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)
    
    def get_text_embeddings(self, prompts):

        with torch.no_grad():
            prompts = prompts + [self.cfg.negative_prompt] * len(prompts)
            text_embeddings = self.text_encoder.encode(prompts)

        return text_embeddings
    
    def encode_first_stage(self, x):
        if self.model.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False

        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(x.shape[0]) < vnum
            if not mask_vae.all():
                with torch.no_grad():
                    posterior_mask = torch.cat(
                        [
                            self.model.first_stage_model.encode(
                                x[~mask_vae][i : i + 1].to(self.weights_dtype)
                            ).sample(None)
                            for i in range(x.shape[0] - vnum)
                        ],
                        dim=0,
                    )

            posterior = self.model.first_stage_model.encode(x[mask_vae].to(self.weights_dtype)).sample(None)
            encoder_posterior = torch.zeros(
                x.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            if not mask_vae.all():
                encoder_posterior[~mask_vae] = posterior_mask
            encoder_posterior[mask_vae] = posterior
        else:
            encoder_posterior = self.model.first_stage_model.encode(x.to(self.weights_dtype)).sample(None)
        results = encoder_posterior*self.model.scale_factor
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results
    
    @torch.amp.autocast(device_type="cuda", enabled=False, dtype=torch.float16)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 512"], normalize: bool = True
    ) -> Float[Tensor, "B 4 N 40 64"]:
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        batch_size, channels, num_frames, height, width = imgs.shape
        
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        latents = self.encode_first_stage(imgs)

        latents = (
            latents
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + latents.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        ) # [B, C, N, H, W]
        
        return latents.to(input_dtype)
    
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def decode_latents(self, latents):
        raise NotImplementedError

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        
        alphas_cumprod = self.alphas
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 T 40 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
        pred_rgb_512,
        num_frames,
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            cond = {'c_crossattn': [text_embeddings], 'fps': torch.full((2,), self.cfg.fps, device=latents.device)}
            noise_pred = self.model.apply_model(
                latent_model_input,
                torch.cat([t] * 2),
                cond,
                x0=None,
                temporal_length=num_frames
                )
                
        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t])
        elif self.cfg.weighting_strategy == "uniform":
            w = torch.ones_like(t, dtype=latents.dtype, device=latents.device)
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t]))
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        
        w = w.view(-1, 1, 1, 1, 1)  # (B, 1, T, 1, 1)
        
        score = noise_pred - noise # (B, C, T, H, W)
        if self.cfg.motion_amp_scale != 1.0:
            score_mean = score.mean(2, keepdim=True) # (B, C, 1, H, W)
            score = score_mean + self.cfg.motion_amp_scale*(score - score_mean)
        grad = w * score
        
        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B T C H W"],
        prompts: List[str],
        **kwargs,
    ):
        batch_size, num_frames = rgb.shape[:2]
        rgb_flat = rgb.view(batch_size*num_frames, *rgb.shape[2:]) # [B*T, C, H, W]
        
        if self.rgb_as_latents:
            latents = F.interpolate(
                rgb_flat, (40, 64), mode="bilinear", align_corners=False
            )
            latents = latents.view(batch_size, latents.shape[1], num_frames, *latents.shape[2:]) # B 4 N H W
        else:
            rgb_BTCHW_512 = F.interpolate(
                rgb_flat, (320, 512), mode="bilinear", align_corners=False
            )
            rgb_BCTHW_512 = rgb_BTCHW_512.view(batch_size, num_frames, *rgb_BTCHW_512.shape[1:]).permute(0, 2, 1, 3, 4) # B 3 N 320 512
            latents = self.encode_images(rgb_BCTHW_512) # B 4 N 40 64
        
        text_embeddings = self.get_text_embeddings(prompts)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        grad = self.compute_grad_sds(latents, text_embeddings, t, rgb_BCTHW_512, num_frames) # (B, C, T, H, W)
        grad = torch.nan_to_num(grad)
        
        # clip grad for stable training
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad        
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="none")
        loss_sds = loss_sds.reshape(batch_size, -1).mean(dim=-1)  # [B,]
        
        return {
            "loss_sds_video": loss_sds,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):

        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)