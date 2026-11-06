import torch
from diffusion.obj_losses import *
from utils.BITE_adapter import SMAL
from utils.BITE_utils import get_verts_from_data
from utils.sdf_utils import OBJInfo
from utils.decouple_utils import apply_rt_to_motion

class Interaction:
    def __init__(self,
                 inv_transform_th,
                 device,
                 with_decay=False,
                 decay_step=50,
                 decay_scale=0.1,
                 classifiler_scale_app=50.0,
                 classifiler_scale_pene=50.0,
                 classifiler_scale_contact=50.0,
                 classifiler_scale_smooth=10.0,
                 stop_cond_from=0,
                 verbose=False):
        
        self.device = device
        self.decay_scale = decay_scale
        self.decay_step = decay_step
        self.with_decay = with_decay
        self.stop_cond_from = stop_cond_from
        self.inv_transform_th = inv_transform_th
        self.classifiler_scale_app = classifiler_scale_app
        self.classifiler_scale_pene = classifiler_scale_pene
        self.classifiler_scale_contact = classifiler_scale_contact
        self.classifiler_scale_smooth = classifiler_scale_smooth
        
        self.smal_model = SMAL(smal_model_type='39dogs_norm_newv3').to(self.device)
        self.obj_info = OBJInfo()
        self.verbose = verbose
    
    def __call__(self, x, t, p_mean_var, y=None,):
        
        if int(t[0]) < self.stop_cond_from: # when t < thresold, not guidance
            return torch.zeros_like(x)
        
        with torch.enable_grad():
            xstart_in = p_mean_var['pred_xstart'] # cleaned x0
            raw_out = p_mean_var['raw_out']
            
            verts, joints, _ = get_verts_from_data(
                                                xstart_in, y['scale_betas'], 
                                                self.smal_model, self.inv_transform_th
                                                ) # (B, N, V, 3) (B, N, J, 3) (B, N, 3)
            
            verts = apply_rt_to_motion(verts, raw_out)
            joints = apply_rt_to_motion(joints, raw_out)
            
            loss_approch = approach_loss(joints, y)
            loss_pene = obj_penetration_loss(joints, y, self.obj_info)
            loss_contact = adaptive_contact_loss(joints, y, self.obj_info)
            loss_smooth = temporal_smooth_loss(joints, y)

            combined_loss = self.classifiler_scale_app * loss_approch + \
                            self.classifiler_scale_pene * loss_pene + \
                            self.classifiler_scale_contact * loss_contact + \
                            self.classifiler_scale_smooth * loss_smooth
            
            combined_loss = combined_loss.sum()
            
            if self.verbose and int(t[0]) % 10 == 0:
                print(f'time step: {t[0]}, \
                      app_scale: {self.classifiler_scale_app}, approach guidance loss: {loss_approch.mean()}, \
                      pene_scale: {self.classifiler_scale_pene}, penetration guidance loss: {loss_pene.mean()}, \
                      contact_scale: {self.classifiler_scale_contact}, contact guidance loss: {loss_contact.mean()}, \
                      smooth_scale: {self.classifiler_scale_smooth}, smooth guidance loss: {loss_smooth.mean()}')

            grad = torch.autograd.grad(-combined_loss, x, retain_graph=False, create_graph=False)[0]
            
            if self.with_decay and t[0] < self.decay_step:
                print('decay scale: ', self.decay_scale, f'decay step: {t[0]}', self.decay_step)
                grad = grad * self.decay_scale #(1 - t / 1000)
            
            return grad


def load_condition(args, inv_transform_th):

    # combination of different condition
    if args.guidance_mode != '' and args.guidance_mode is not None:
        cond_fn = []
        guidance_mode = args.guidance_mode.split(',')
        for one in guidance_mode:
            print(f'add {one} in guidance mode.')
            if one == 'physical':
                cond_fn.append(
                    Interaction(
                        inv_transform_th,
                        device=args.device,
                        with_decay=args.with_decay,
                        decay_step=args.decay_step,
                        decay_scale=args.decay_scale,
                        classifiler_scale_app=args.classifiler_scale_app,
                        classifiler_scale_pene=args.classifiler_scale_pene,
                        classifiler_scale_contact=args.classifiler_scale_contact,
                        classifiler_scale_smooth=args.classifiler_scale_smooth,
                        verbose=args.verbose,
                    )
                )
            else:
                raise NotImplementedError(f'Guidance mode {one} not implemented.')
        
        return cond_fn
    
    return None
