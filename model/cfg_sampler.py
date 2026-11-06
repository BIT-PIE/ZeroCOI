import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cond_mode):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'
        
        # pointers to inner model
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = cond_mode

    def forward(self, x, timesteps, y=None):
        if self.cond_mode == 'text' or self.model.cond_mask_prob_noObj == 0:
            y_uncond = deepcopy(y)
            y_uncond['text_uncond'] = True
            out, raw_out = self.model(x, timesteps, y)
            out_uncond, _ = self.model(x, timesteps, y_uncond)
            return out_uncond + (y['text_guide_scale'].view(-1, 1, 1, 1) * (out - out_uncond)), raw_out
        else:
            y_text_uncond = deepcopy(y)
            y_text_uncond['text_uncond'] = True
            out_text_uncond, raw_out = self.model(x, timesteps, y_text_uncond) # no text, with obj
            
            y_obj_uncond = deepcopy(y)
            y_obj_uncond['obj_uncond'] = True
            out_obj_uncond, _ = self.model(x, timesteps, y_obj_uncond) # with text, no obj
            
            y_text_obj_uncond = deepcopy(y)
            y_text_obj_uncond['text_uncond'] = True
            y_text_obj_uncond['obj_uncond'] = True
            out_text_obj_uncond, _ = self.model(x, timesteps, y_text_obj_uncond) # no text, no obj

            return out_text_obj_uncond + \
                   (y['text_guide_scale'].view(-1, 1, 1, 1) * (out_obj_uncond - out_text_obj_uncond)) + \
                   (y['obj_guide_scale'].view(-1, 1, 1, 1) * (out_text_uncond - out_text_obj_uncond)), raw_out

