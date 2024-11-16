import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=128, patch_size=8, in_channels=1, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([embed_dim, image_size // patch_size, image_size // patch_size]),
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Inicialización de parámetros
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positions, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Agregar token de clasificación
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Agregar posición embeddings
        x = x + self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ImprovedViT(nn.Module):
    def __init__(self, 
                 image_size=128,
                 patch_size=8,
                 in_channels=1,
                 num_classes=10,
                 embed_dim=256,
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.,
                 dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout * 1.5)

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer Blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Normalization y Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            self.dropout,
            nn.Linear(embed_dim // 2, num_classes)
        )
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.01
        # Normalización de entrada
        x = x / 255.0
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Clasificación usando solo el token [CLS]
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        
        return x