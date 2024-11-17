# config.py
import yaml

# Cargar el archivo config.yml una sola vez
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
