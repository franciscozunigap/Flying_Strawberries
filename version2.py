from modules.detection import load_model
import cv2
import os
from ultralytics import YOLO

def extract_bounding_boxes(image_path, model):
    # Obtener las detecciones
    results = model(image_path)  # Realiza las detecciones

    # Extraer las coordenadas de las bounding boxes
    bounding_boxes = []  # Lista para guardar las coordenadas de las cajas

    # Acceder a todas las detecciones
    detections = results[0].boxes  # Detecciones de la primera imagen

    # Iterar sobre las detecciones y extraer las coordenadas de las bounding boxes
    for box in detections:
        coords = box.xyxy.cpu().numpy().astype(int)  # Coordenadas [x1, y1, x2, y2]

        # Añadir solo las coordenadas de la bounding box
        bounding_boxes.append((coords[0][0], coords[0][1], coords[0][2], coords[0][3]))

    # Retornar las coordenadas de las bounding boxes
    return bounding_boxes

def save_bounding_boxes_as_images(image, bounding_boxes, output_dir):
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre las bounding boxes y guardar las imágenes correspondientes
    for idx, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = map(int, box)  # Convertir las coordenadas a enteros

        # Recortar la región de interés (ROI) de la imagen
        roi = image[y1:y2, x1:x2]

        # Ruta completa para guardar la imagen de la bounding box
        output_path = os.path.join(output_dir, f"strawberry_{idx + 1}.png")

        # Guardar la imagen recortada
        cv2.imwrite(output_path, roi)

        print(f"Imagen recortada guardada en: {output_path}")

def classify_image_with_yolo(image_path, model):
    # Realizar la inferencia
    results = model(image_path)

    return results[0].probs.top1

# Cargar el modelo
model = load_model()  # Modelo YOLO cargado

# Ruta de la imagen a procesar
image_path = "img/input/R.png"  # Especifica tu imagen aquí
image = cv2.imread(image_path)

# Obtener las coordenadas de las bounding boxes
boxes = extract_bounding_boxes(image_path, model)
save_bounding_boxes_as_images(image, boxes, "save_detections")

classify_image_with_yolo("save_detections/strawberry_1.png", YOLO("runs/classify/train/weights/best.pt"))
