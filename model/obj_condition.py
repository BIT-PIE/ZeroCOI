import math
import torch
import torch.nn as nn

class ObjectEncoder(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        if self.data_rep in 'bps':
            self.merge_object = nn.Sequential(
                        nn.Linear(in_features=self.input_feats, out_features=self.latent_dim),
                        nn.ReLU(),
                        nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                    )
        elif self.data_rep in 'sdf': 
            self.merge_object = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape # [B, n_bps, 3, N]
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['sdf', 'bps']:
            x = self.merge_object(x)  # [seqlen, bs, d]
            return x
        else:
            raise ValueError


class RTDecoder(nn.Module):
    def __init__(self, latent_dim, rt_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = latent_dim // 2
        self.rt_dim = rt_dim

        self.decode_rt = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.rt_dim),
        )

        # angle range
        self.max_yaw_deg = 90.0
        self.max_yaw_rad = math.radians(self.max_yaw_deg)

        # translation range
        self.tx_range = 4.0
        self.ty_min = -0.5
        self.ty_max = 2.5
        self.tz_min = -6.0
        self.tz_max = 1.5

        self.ty_center = (self.ty_max + self.ty_min) / 2
        self.ty_radius = (self.ty_max - self.ty_min) / 2
        self.tz_center = (self.tz_max + self.tz_min) / 2
        self.tz_radius = (self.tz_max - self.tz_min) / 2 

    def forward(self, x):
        # x: [seqlen, bs, d]
        raw_out = self.decode_rt(x)  # [seqlen, bs, rt_dim]
        raw_out = raw_out.mean(dim=0)  # [bs, rt_dim]

        yaw_raw = raw_out[:, 0:1]
        tx_raw = raw_out[:, 1:2]
        ty_raw = raw_out[:, 2:3]
        tz_raw = raw_out[:, 3:4]

        # ----------- map to valid ranges ---------------

        # yaw ∈ [-max_yaw, max_yaw]
        yaw = torch.tanh(yaw_raw) * self.max_yaw_rad

        # tx ∈ [-tx_range, +tx_range]
        tx = torch.tanh(tx_raw) * self.tx_range
        
        # tz ∈ [tz_min, tz_max]
        tz = self.tz_center + torch.tanh(tz_raw) * self.tz_radius

        # ty ∈ [ty_min, ty_max]
        ty = self.ty_center + torch.tanh(ty_raw) * self.ty_radius

        raw_out = torch.cat([yaw, tx, ty, tz], dim=-1)  # [B,4]

        return raw_out