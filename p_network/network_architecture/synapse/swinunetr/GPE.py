import json

import torch
import torch.nn as nn
# import tinycudann as tcnn

class FrequencyEncoding(nn.Module):
    def __init__(self, L_embed=10):
        super(FrequencyEncoding, self).__init__()
        self.L_embed = L_embed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = 3*(2*L_embed + 1)

    # def forward(self, inp):
    #     res = [inp]
    #     for l in range(self.L_embed):
    #         res.append(torch.sin(2**l * inp))
    #         res.append(torch.cos(2**l * inp))
    #     return torch.cat(res, dim=1)
    
    def forward(self, inp):
        # inp: (..., 3)
        inp = inp.permute(0, 2, 3, 4, 1)
        freq = 2 ** torch.arange(self.L_embed, device=inp.device).float()  # (L_embed,)
        inp_expanded = inp.unsqueeze(-1) * freq  # (..., 3, L_embed)
        sin = torch.sin(inp_expanded)
        cos = torch.cos(inp_expanded)
        # Flatten the last two dims for sin and cos
        sin = sin.reshape(*inp.shape[:-1], -1)  # (..., 3*L_embed)
        cos = cos.reshape(*inp.shape[:-1], -1)  # (..., 3*L_embed)
        emb = torch.cat([inp, sin, cos], dim=-1)  # (..., 3 + 2*3*L_embed)
        return emb.permute(0, 4, 1, 2, 3).contiguous()


# class HashEncoding(nn.Module):
#     def __init__(self, n_input_dim = 3):
#         super(HashEncoding, self).__init__()
        
#         with open("p_network/network_architecture/synapse/unetr/config_hash.json") as f:
#         	config = json.load(f)
#         self.encoding = tcnn.Encoding(n_input_dim, config["encoding"])

#     def forward(self, inp):
#         B, C, D, H, W = inp.shape
#         inp = inp.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
#         encoded = self.encoding(inp)
#         encoded = encoded.view(B, D, H, W, encoded.shape[-1]).permute(0, 4, 1, 2, 3)
#         return encoded.to(torch.float32).contiguous()


class PosEnc(nn.Module):
    def __init__(self):
        super(PosEnc, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, pos, inp_shape):
        B = pos.shape[0]
        d, h, w = inp_shape
        
        mesh_grids = []
        
        for i in range(B):
            x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = pos[i]
            x = torch.linspace(x_lb, x_ub, d, device=self.device)
            y = torch.linspace(y_lb, y_ub, h, device=self.device)
            z = torch.linspace(z_lb, z_ub, w, device=self.device)
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            grid = torch.stack([X, Y, Z], dim=-1)
            
            mesh_grids.append(grid)
        
        mesh_grids = torch.stack(mesh_grids, dim=0).to(self.device, non_blocking=True)
        mesh_grids = mesh_grids.permute(0, 4, 1, 2, 3)
            
        return mesh_grids.contiguous()  