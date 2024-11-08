import torch
import torch.nn as nn
import torch.nn.functional as F

class RusticModel(nn.Module):
    def __init__(self, num_classes):
        super(RusticModel, self).__init__()
        self.red_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # Capa 1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Capa 2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Capa 3
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Capa 4
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        
        # Actualiza la capa fully connected con el tamaño de entrada correcto
        self.fc1 = nn.Linear(256 * 3 * 3, 256)  # 256*3*3 = 2304 para imágenes de 50x50
        self.dropout = nn.Dropout(0.30)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x / 255.0
        x = self.red_conv(x)
        print(f"Dimensiones después de convolución: {x.shape}")  # Añade esta línea
        x = self.flatten(x)
        print(f"Dimensiones después de flatten: {x.shape}")  # Añade esta línea
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
