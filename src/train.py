import os
import torch
import time
import logging
from pyinstrument import Profiler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from VitModel import ViTModel
from data_processing import process_imgs
from model import RusticModel

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__}: {end_time - start_time:.2f} segundos")
        return result
    return wrapper

@log_time
def load_and_process_data(data_directory):
    return process_imgs(data_directory)

@log_time
def convert_to_tensors(X_train, X_val, y_train, y_val):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    if X_train.dim() == 3:
        X_train = X_train.unsqueeze(1)
    else:
        X_train = X_train.permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_val = torch.tensor(X_val, dtype=torch.float32)
    if X_val.dim() == 3:
        X_val = X_val.unsqueeze(1)
    else:
        X_val = X_val.permute(0, 3, 1, 2)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    return X_train, X_val, y_train, y_val

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    profiler = Profiler()
    profiler.start()
    
    total_start_time = time.time()
    
    # Define el directorio de datos
    data_directory = os.path.join(os.getcwd(), 'Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet')
    X_train, X_val, y_train, y_val, label_mapping = load_and_process_data(data_directory)
    
    # Convertir los datos a tensores
    X_train, X_val, y_train, y_val = convert_to_tensors(X_train, X_val, y_train, y_val)
    
    logger.info(f"Tamaño de conjunto de entrenamiento: {len(X_train)}")
    logger.info(f"Tamaño de conjunto de validación: {len(X_val)}")
    logger.info(f"Diccionario de etiquetas: {label_mapping}")
    
    # Crear los DataLoaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)
    
    total_end_time = time.time()
    logger.info(f"Tiempo total de preparación: {total_end_time - total_start_time:.2f} segundos")
    
    profiler.stop()
    logger.info(profiler.output_text())

    # Inicializar el modelo
    num_classes = len(label_mapping)
    model = RusticModel(num_classes).to(device)
    # Alternativa de modelo:
    # model = ViTModel(num_classes).to(device)

    # Definir el optimizador y la función de pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Entrenamiento del modelo
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()  # Modo entrenamiento
        total_loss = 0
        total_correct = 0

        try:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass y optimización
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
            
            scheduler.step()
            
            # Calcula la precisión en el conjunto de entrenamiento
            train_accuracy = total_correct / len(train_loader.dataset)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

            model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad(): 
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = val_correct / len(val_loader.dataset)
            logger.info(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        except Exception as e:
            logger.exception(f"Error durante el entrenamiento: {e}")
            break
    
    # Guardar el modelo entrenado
    model_path = "modelo_vit.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Modelo guardado exitosamente en {model_path}")

if __name__ == "__main__":
    main()
