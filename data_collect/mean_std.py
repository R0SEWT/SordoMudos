import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
#calcular la media y la desviacion estandard
def calcular_mean_std(data_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mean = 0.0
    var = 0.0
    total_images = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        total_images += batch_size
        mean += images.mean([0, 2, 3]) * batch_size
        var += images.var([0, 2, 3], unbiased=False) * batch_size
    mean /= total_images
    var /= total_images
    std = torch.sqrt(var)
    
    return mean.item(), std.item()


data_path = r"C:\Users\joaqu\OneDrive\Escritorio\6tociclo\procesamiento_imagenes\SordoMudos\captures"

mean, std = calcular_mean_std(data_path)
print(f"Media calculada: {mean}")
print(f"Desviación estándar calculada: {std}")
