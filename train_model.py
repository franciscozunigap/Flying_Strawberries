import os
from ultralytics import YOLO

dataset_dir = os.getcwd()

# Crea las rutas completas a los archivos de modelo y conjunto de datos
model_path = os.path.join(dataset_dir, "datasets/yolo11n.pt")
data_path = os.path.join(dataset_dir, "datasets/dataset.yaml")

# Cargar el modelo
model = YOLO(model_path)

# Entrenar el modelo
results = model.train(data=data_path, epochs=50, imgsz=640, augment=True)
