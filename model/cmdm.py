# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
import clip
import math
from .transformer import *
from utils.bps_utils import get_bps_pose_feature
from model.obj_condition import ObjectEncoder, RTDecoder
from utils.BITE_utils import get_verts_from_data
from utils.BITE_adapter import SMAL


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class CMDM(torch.nn.Module):
    def __init__(self, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', clip_version=None, sample_points_num=1024, beta_dims=38, rt_dim=4,
                 inv_transform_th=None, *args, **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim
        self.beta_dims = beta_dims
        self.rt_dim = rt_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim

        self.input_feats = self.njoints * self.nfeats

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # for text cond
        self.cond_mask_prob_noObj = kargs.get('cond_mask_prob_noObj', 0.) # for object cond
        
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.inv_transform_th = inv_transform_th
        self.smal_model = SMAL(smal_model_type='39dogs_norm_newv3')

        # --- MDM ---
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.beta_encoder = BetaEncoder(self.beta_dims, self.latent_dim)
        
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_model = self.load_and_freeze_clip(clip_version)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)
        
        # --- CMDM ---
        self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC_COND init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers,
                                                    return_intermediate=True)

        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
        
        self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.c_beta_encoder = BetaEncoder(self.beta_dims, self.latent_dim)

        self.c_RT_decoder = RTDecoder(self.latent_dim, self.rt_dim)
        
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            if 'obj' in self.cond_mode:
                self.freeze_mdm_params()
                self.sample_points_num = sample_points_num
                self.c_merge_bps_info = ObjectEncoder(
                    'bps', # data_rep
                    self.sample_points_num*3, # [bps point xyz, w/o], bps point->object distance, bps point->body joint distance, with bps point->body joint nearest label.
                    self.latent_dim,
                )

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def open_gradient(self):
        train_layer_name = [
            'c_input_process',
            'c_sequence_pos_encoder',
            'c_seqTransEncoder',
            'zero_convs',
            'c_embed_timestep',
            'c_beta_encoder',
            'c_embed_text',
            'c_merge_bps_info', 
            'c_RT_decoder',
        ]
        for name in train_layer_name:
            module = getattr(self, name)
            for p in module.parameters():
                p.requires_grad = True

    def freeze_mdm_params(self):
        fixed_layer_name = [
            'input_process',
            'sequence_pos_encoder',
            'seqTransEncoder',
            'embed_timestep',
            'beta_encoder',
            'embed_text',
            'output_process',
        ]
        for name in fixed_layer_name:
            getattr(self, name).eval()
            for p in getattr(self, name).parameters():
                p.requires_grad = False
          
    def mask_text_cond(self, cond, force_mask=False):
        # for classifier-free guidance
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def mask_object_cond(self, cond, y_flag, force_mask=False):
        bs = cond.shape[0]
        
        if force_mask: 
            return torch.zeros_like(cond)

        elif self.training and self.cond_mask_prob_noObj > 0.: # drop condition
            import pdb;pdb.set_trace()
            assert y_flag is not None
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            self.object_mask = (1.0 - mask.view(bs)).to(torch.bool)
            if y_flag.sum() > y_flag.shape[0] * (1 - self.cond_mask_prob_noObj): # if the dropped object is not sufficient.
                selected_obj_num = math.ceil(y_flag.sum() - y_flag.shape[0] * (1 - self.cond_mask_prob_noObj))
                select_obj_idx = mask.nonzero()[:selected_obj_num].squeeze()
                y_flag[select_obj_idx] = 0
            if len(cond.shape) == 4:
                return cond * (1. * y_flag.view(bs, 1, 1, 1)) # 1: with object; 0 with dropped out object
            elif len(cond.shape) == 3:
                return cond * (1. * y_flag.view(bs, 1, 1))
        else:
            self.object_mask = torch.ones(bs, device=cond.device).to(torch.bool)
            return cond 
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 25 if self.dataset in ['dogml'] else 28  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def get_bps_feature(self, x, y):
        _, world_poses, _ = get_verts_from_data(x, y['scale_betas'], self.smal_model, self.inv_transform_th) # [B, N, J, 3]
        
        bps_obj_feat = y['bps_obj_feat'].float().to(world_poses.device) # pre-compute, [B, 1, n_bps]  
        bps_to_joint_dis, bps_to_joint_idx = get_bps_pose_feature(world_poses, y) # [B, N, n_bps]
        
        bps_input_feat = torch.stack((bps_obj_feat.repeat(1, bps_to_joint_dis.shape[1], 1), bps_to_joint_dis, bps_to_joint_idx), axis=-1)

        return bps_input_feat.permute(0, 2, 3, 1) # [B, n_bps, 3, N]

    def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
        """
        Object Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, njoints, nfeats, nframes = x.shape
        emb = self.c_embed_timestep(timesteps)
        emb += self.c_beta_encoder(y['scale_betas']).unsqueeze(0) # [1, bs, d]

        force_mask = y.get('text_uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.c_embed_text(self.mask_text_cond(enc_text, force_mask=force_mask))
        
        # obj guidance
        bps_input_feat = self.get_bps_feature(x, y)
        bps_feat = self.c_merge_bps_info(bps_input_feat) # [N, bs, d]
        
        x = self.c_input_process(x)
        x += bps_feat

        # mask obj cond
        obj_id = y.get('obj_id', None)
        obj_flag = [1 if obj_i is not None else 0 for obj_i in obj_id]
        obj_flag = torch.tensor(obj_flag).to(x.device)
        noObj_force_mask = y.get('obj_uncond', False) # used for inference, always not mask                    
        x = self.mask_object_cond(x.permute(1,0,2), obj_flag, force_mask=noObj_force_mask).permute(1,0,2)
        
        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)
        xseq = self.c_sequence_pos_encoder(xseq) # [seqlen+1, bs, d]

        is_valid_mask = y['mask'].shape[-1] > 1 # Don't use mask with the generate script

        if self.arch == 'trans_enc' or not is_valid_mask:
            output = self.c_seqTransEncoder(xseq)
        elif self.arch == 'trans_enc_new_attention':
            x_mask = y['mask']
            x_mask = x_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, -1)
            time_mask = torch.ones(bs, dtype=bool).reshape(1, bs, -1).to(x_mask.device)
            aug_mask = torch.cat((time_mask, x_mask), 0).to(xseq.device) # ignore the attention for the padding elements
            output = self.c_seqTransEncoder(xseq, src_key_padding_mask=~aug_mask[:,:,0].permute(1,0))  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control
    
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        emb += self.beta_encoder(y['scale_betas']).unsqueeze(0) # [1, bs, d]

        force_mask = y.get('text_uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_text_cond(enc_text, force_mask=force_mask))

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(xseq) # [seqlen+1, bs, d]

        is_valid_mask = y['mask'].shape[-1] > 1 # Don't use mask with the generate script
        
        if self.arch == 'trans_enc' or not is_valid_mask:
            output = self.seqTransEncoder(xseq, control=control)[1:] # [seqlen, bs, d]
        elif self.arch == 'trans_enc_new_attention': 
            x_mask = y['mask'] # [bs, 1, 1, seqlen]
            x_mask = x_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, -1)
            time_mask = torch.ones(bs, dtype=bool).reshape(1, bs, -1).to(x_mask.device)
            aug_mask = torch.cat((time_mask, x_mask), 0).to(xseq.device) # [seqlen+1, bs, 1]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask[:,:,0].permute(1,0), control=control)[1:] # [seqlen, bs, d]
        
        return output

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        if 'obj' in self.cond_mode:
            control = self.cmdm_forward(x, timesteps, y)
            output = self.mdm_forward(x, timesteps, y, control)
            raw_out = self.c_RT_decoder(output)  # [bs, rt_dim]
            motion = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        else:
            control = None
            raw_out = None
            output = self.mdm_forward(x, timesteps, y, control)
            motion = self.output_process(output)

        return motion, raw_out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    

class BetaEncoder(nn.Module):
    def __init__(self, beta_dims, latent_dim):
        super().__init__()
        self.beta_dims = beta_dims
        self.latent_dim = latent_dim
        self.beta_encoder = nn.Sequential(
                    nn.Linear(in_features=self.beta_dims, out_features=self.latent_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                    )
        self.ln = nn.LayerNorm(self.latent_dim)
        
    def forward(self, x):
        x = self.beta_encoder(x)
        return self.ln(x)
