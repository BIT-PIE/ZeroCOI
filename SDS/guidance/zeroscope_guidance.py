from dataclasses import dataclass, field
from typing import Any, List, Optional

from jaxtyping import Float, Int
import torch
Tensor = torch.Tensor
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

from SDS.guidance.misc import C, cleanup, parse_structured, parse_version
from transformers import CLIPTokenizer, CLIPTextModel



class ZeroscopeGuidance:
    @dataclass
    class Config:
        pretrained_model_name_or_path: str = None
        enable_memory_efficient_attention: bool = True
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

        motion_amp_scale: float = 1.0
        negative_prompt: str = "low motion, static statue, not moving, no motion"

        device: str = "cuda"

        rgb_as_latents: bool = False

    cfg: Config

    def __init__(self, cfg, *args, **kwargs):
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self) -> None:
        print("Loading Stable Diffusion ...")

        self.device = self.cfg.device
        
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "use_safetensors": False,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        # Extra modules
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer",
            torch_dtype=self.weights_dtype
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder",
            torch_dtype=self.weights_dtype
        )
        self.text_encoder = self.text_encoder.to(self.device)


        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                print(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        cleanup()

        self.rgb_as_latents = self.cfg.rgb_as_latents
        
        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd
            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        # Extra for latents
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        print("Loaded Stable Diffusion!")

    def get_text_embeddings(self, prompts):

        with torch.no_grad():
            prompts = prompts + [self.cfg.negative_prompt] * len(prompts)
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.text_encoder.device))[0]

        return text_embeddings
    
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
    
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 576"], normalize: bool = True
    ) -> Float[Tensor, "B 4 N 40 72"]:

        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]

        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0

        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(imgs.shape[0]) < vnum
            with torch.no_grad():
                posterior_mask = torch.cat(
                    [
                        self.vae.encode(
                            imgs[~mask_vae][i : i + 1].to(self.weights_dtype)
                        ).latent_dist.sample()
                        for i in range(imgs.shape[0] - vnum)
                    ],
                    dim=0,
                )

            posterior = self.vae.encode(imgs[mask_vae].to(self.weights_dtype)).latent_dist.sample()
            posterior_full = torch.zeros(
                imgs.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            posterior_full[~mask_vae] = posterior_mask
            posterior_full[mask_vae] = posterior
            latents = posterior_full * self.vae.config.scaling_factor
        else:
            posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor

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
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        ) # [B, C, N, H, W]
        video = video.float()
        return video

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 T 40 72"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
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
                rgb_flat, (40, 72), mode="bilinear", align_corners=False
            )
            latents = latents.view(batch_size, latents.shape[1], num_frames, *latents.shape[2:]) # B 4 N H W
        else:
            rgb_BTCHW_512 = F.interpolate(
                rgb_flat, (320, 576), mode="bilinear", align_corners=False
            )
            rgb_BCTHW_512 = rgb_BTCHW_512.view(batch_size, num_frames, *rgb_BTCHW_512.shape[1:]).permute(0, 2, 1, 3, 4) # B 3 N 320 512
            latents = self.encode_images(rgb_BCTHW_512) # B 4 N 40 72
        
        text_embeddings = self.get_text_embeddings(prompts)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        grad = self.compute_grad_sds(latents, text_embeddings, t)
        grad = torch.nan_to_num(grad)
        
        # clip grad for stable training
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="none")
        loss_sds = loss_sds.reshape(batch_size, -1).mean(dim=-1)
        
        return {
            "loss_sds_video": loss_sds,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int):

        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)