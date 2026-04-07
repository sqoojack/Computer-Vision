
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from diffusers.models.transformers import DiTModelOutput
from diffusers.models.attention import Attention
from einops import rearrange    # used to rearrange the tensor's dimension
from dataclasses import dataclass

@dataclass
class DiTModelOutput:
    sample: torch.Tensor

@dataclass
class MyDiTConfig:
    addition_time_embed_dim: int = 256  # it's in original unet attribute
    time_embedding_type = "learned"

# to solve not have linear_1 bug
class AddEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_1(x)

""" shape of input tensor: (B, F, C, H, W), Batch_size, Frames, Channels, Height, Width """
""" define DiT """
class MyDiT(nn.Module):
    # All of the following parameters are custom parameters
    def __init__(self, image_size=64, patch_size=8,  #  patch: small block, patch_size=8: means that every patch is 8x8 pixels
                in_channels=8,  # has 7 channels in every frame
                hidden_dim=256, num_layers=12,
                num_heads=8, mlp_ratio=4.0,   # it means that dimension of MLP is four times the hidden_dim (MLP: multi layer perception)
                use_cross_attention=True,
                addition_time_embed_dim=256):      # in resource-constrained situations, you need to forbid it
                
        super().__init__()  # ensure that nn.Module can successfully initialize
        self.config = MyDiTConfig(addition_time_embed_dim=addition_time_embed_dim)
        
        self.add_embedding = AddEmbedding(in_features=768, out_features=hidden_dim)
        
        self.image_size = image_size    # store the externally passed parameters as attributes of the class
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        self.latent_fft_post_merge = False
        self.latent_fft_ratio = 0.5
        self.optimize_latent_iter = 5
        self.optimize_latent_lr = 0.21
        self.optimize_latent_time = list(range(30, 46))
        self.record_layer_sublayer = [(2, 1), (2, 2)]
        
        # used to devide input image into small patches and map each patch to a feature space with a dimension of hidden_dim
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)    # stride(步幅) = patch_size: ensure each kernel will not overlap   
        
        self.time_embedding = nn.Sequential(    # is similar to U-Net's time embedding, used to let timesteps map into continue vector
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.time_proj = nn.Embedding(10000, hidden_dim)    # used to map discrete timesteps to a hidden_dim dimensional space, 
        self.add_time_proj = nn.Embedding(10000, hidden_dim)    # provide added_time_ids embedding capability, assume the maximum timestep is 10000
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))  # add position information for each patch
        
        self.layers = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, use_cross_attention=use_cross_attention)     # create num_layers instance of DiTBlock
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)    # layer normalization
        
        # project the final hidden_dim back to the latent dimension (align to dimension of UNet's final output)
        self.head = nn.Linear(hidden_dim, 4 * (patch_size ** 2))

        self.latent_shape = None    # used to reshape
        
        self.encoder_h_proj = nn.Linear(1024, hidden_dim, bias=False)  # 將 1024 維度投影到 256 維  (to fix bug)
        
    def forward(self, sample, timestep, encoder_hidden_states, added_time_ids, return_dict: bool = True):
        # sample.shape: [B, F, C, H, W]
        # convert timestep to tensor and embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].long().to(sample.device)
        else:
            timestep = timestep.long().to(sample.device)
            
        b, f, c, h, w = sample.shape
        # print(f"b, f, c, h, w: {b}, {f}, {c}, {h}, {w}")  # debug line
        self.latent_shape = sample.shape
        
        x = sample.reshape(b*f, c, h, w)
        x = self.patch_embed(x)     # patch embedding
        _, hidden_dim, H_patch, W_patch = x.shape
        x = x.flatten(2).transpose(1, 2)    # [B*F, hidden_dim, H_patch, W_patch] -> [B*F, hidden_dim, num_patches] -> [B*F, num_patches, hidden_dim]
        
        # 動態調整 pos_embed 的尺寸
        num_patches_current = H_patch * W_patch
        num_patches_model = self.num_patches
        H_embed = W_embed = int(math.sqrt(num_patches_model))
        
        # 假設模型的 pos_embed 是 [1, H_embed * W_embed, hidden_dim]
        pos_embed = self.pos_embed.reshape(1, H_embed, W_embed, hidden_dim).permute(0, 3, 1, 2)  # [1, hidden_dim, H_embed, W_embed]
        
        # 插值 pos_embed 以匹配當前的 H_patch 和 W_patch
        pos_embed = F.interpolate(pos_embed, size=(H_patch, W_patch), mode='bicubic', align_corners=False)  # [1, hidden_dim, H_patch, W_patch]
        
        # 重新排列 pos_embed 的維度以匹配 x 的形狀
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, H_patch * W_patch, hidden_dim)  # [1, num_patches_current, hidden_dim]
        
        # 將插值後的 pos_embed 添加到 x
        x = x + pos_embed
        
        # print(f"timestep values: {timestep}")   # debug line
        # print(f"timestep max: {timestep.max()}, min: {timestep.min()}")     # debug line
        timestep = torch.clamp(timestep, min=0)     # to debug timestep is negative
        
        t_embed = self.time_proj(timestep)  # [1, hidden_dim]
        t_embed = t_embed.to(x.dtype)
        t_embed = self.time_embedding(t_embed)  # [1, hidden_dim]
        # print(f"t_embed:  {t_embed.shape}")
        
        add_t_embed = self.add_time_proj(added_time_ids.flatten())   # process add_time
        # print(f"add_t_embed shape_1: {add_t_embed.shape}")
        add_t_embed = add_t_embed.reshape((b, -1))    # [B, ...]
        # print(f"add_t_embed shape_2: {add_t_embed.shape}")
        add_t_embed = add_t_embed.to(x.dtype)
        add_t_embed = add_t_embed.reshape((-1, 768))
        # print(f"add_t_embed shape_3: {add_t_embed.shape}")
        add_t_embed = self.add_embedding(add_t_embed)   # [B, hidden_dim]
        # print(f"add_t_embed shape_4: {add_t_embed.shape}") 
        add_t_embed = add_t_embed[:b, :]
        
        # print(f"t_embed shape:  {t_embed.shape}")
        t_embed = t_embed + add_t_embed     # [B*F, hidden_dim]
        t_embed = t_embed.repeat_interleave(f, dim=0)  # [B*F, hidden_dim]
        # print("t_embed shape after * F: ", t_embed.shape)
        # print("x shape: ", x.shape)
        x = x + t_embed.unsqueeze(1)    # similar to add time emb to hidden state in UNet
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_h_proj(encoder_hidden_states)  # [B, 256]
            print(f"encoder_hidden_states shape 1: {encoder_hidden_states.shape}")
            
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(f, dim=0)   #   [B*F, 1, 256]
            print(f"encoder_hidden_states shape before cross-attn: {encoder_hidden_states.shape}")
            
        for i, block in enumerate(self.layers):
            x = block(x, encoder_hidden_states=encoder_hidden_states, parent=self, layer_index=i)
        
        x = self.norm(x)
        
        x = self.head(x)    # let it move to original latent patch [B*F, C, H, W]
        p = self.patch_size
        hp = h // p
        wp = w // p
        
        print(f"x shape: {x.shape}")
        x = rearrange(x, '(b f) (hp wp) (c p1 p2) -> b f c (hp p1) (wp p2)', b=b, f=f, hp=hp, wp=wp, c=4, p1=p, p2=p) 
        print(f"x after rearrange shape: {x.shape}")
        if not return_dict:
            return (x,)
        
        return DiTModelOutput(sample=x)
    
    def dtype(self):
        return torch.float16
        
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio, use_cross_attention=True):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.multi_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)  # let normalize input data before attn_layer
            self.cross_attn = Attention(
                query_dim=hidden_dim,               
                cross_attention_dim=hidden_dim,     
                heads=num_heads,                     
                dim_head=hidden_dim // num_heads,   
                dropout=0.0,                     
                bias=False,                         
                only_cross_attention=False,
                # added_kv_proj_dim=hidden_dim
            )
        
        # MLP (Feed-Forward network)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        )
        
    def forward(self, x, encoder_hidden_states=None, parent=None, layer_index=None):
        h = self.attn_norm(x)
        attn_out, _ = self.multi_attn(h, h, h, need_weights=False)
        x = x + attn_out
                    
        # cross-Attention
        if self.use_cross_attention and encoder_hidden_states is not None:  
            h = self.cross_attn_norm(x)
            x = x + self.cross_attn(h, encoder_hidden_states)[0]
            
        h = self.mlp_norm(x)
        x = x + self.mlp(h)
        
        return x