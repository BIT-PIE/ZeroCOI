import os
from omegaconf import OmegaConf
import importlib

base_dir = os.path.dirname(__file__)

class MotionVideoSDS:
    
    def __init__(self, cfg_src, device):

        self.device = device

        cfg_path = os.path.join(base_dir, "configs", cfg_src)
        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.guidance_video.device = str(device)

        guidance_type = self.cfg.guidance_type_video
        module_name, class_name = guidance_type.rsplit(".", 1)
        guidance_cls = getattr(importlib.import_module(module_name), class_name)
        self.guidance_video = guidance_cls(self.cfg.guidance_video)

    
    def __call__(self, imgs, prompts, step):
        
        guidance_out = self.guidance_video(
                    imgs,    # [B T C H W]
                    prompts, # List[str], size B
                )
        loss = guidance_out["loss_sds_video"]
        
        self.guidance_video.update_step(epoch=step, global_step=step)
        
        return loss