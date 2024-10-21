import os
import cv2
from ultralytics import YOLO

def load_model(model_path=os.path.join(os.getcwd(), "runs/detect/train/weights/best.pt")):
    model = YOLO(model_path)
    return model

# Función para detectar frutillas en todas las imágenes de un directorio
def detect_strawberries(model, input_images_dir):
    # Extensiones de imagen comunes
    valid_extensions = (".png", ".jpg", ".jpeg")

    # Obtener todas las imágenes con extensiones válidas en el directorio de entrada
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(valid_extensions)]

    results_list = []  # Lista para almacenar los resultados

    # Realizar inferencia en cada imagen
    for image_file in image_files:
        image_path = os.path.join(input_images_dir, image_file)  # Ruta completa de la imagen
        original_image = cv2.imread(image_path)  # Leer la imagen original con OpenCV (formato BGR)

        # Realizar inferencia con el modelo
        results = model(image_path)

        # Guardar los resultados y las imágenes
        for result in results:
            detected_image = result.plot()  # Obtener la imagen con detecciones
            results_list.append((original_image, detected_image, image_file))  # Guardar original y detección

    return results_list  # Devolver lista de tuplas (imagen original, imagen detectada, nombre archivo)