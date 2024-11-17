import torch
import torch.nn as nn

class ViTModel(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_layers=6, num_heads=8, dropout_rate=0.1):
        super(ViTModel, self).__init__()
        
        self.patch_size = 8  # tamaño del parche
        self.embed_dim = embed_dim
        
        # capa de embeddings de parches
        self.patch_embeddings = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # capa de Transformer
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # capa de salida
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2) 
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.mean(dim=1) 
        return self.fc(x)

