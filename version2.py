from modules.detection import load_model
import cv2
import matplotlib.pyplot as plt
import os

def extract_bounding_boxes(image_path, model):
    # Cargar la imagen
    image = cv2.imread(image_path)

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

def save_strawberry_roi(image_path, bounding_box, output_dir, output_name):
    # Cargar la imagen original
    image = cv2.imread(image_path)

    # Extraer las coordenadas de la bounding box y asegurarse de que sean enteros
    x1, y1, x2, y2 = map(int, bounding_box)  # Convertir las coordenadas a enteros

    # Recortar la región de la frutilla (ROI)
    roi = image[y1:y2, x1:x2]

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Ruta completa del archivo de salida
    output_path = os.path.join(output_dir, output_name)

    # Guardar la imagen recortada
    cv2.imwrite(output_path, roi)

    print(f"Imagen recortada guardada en: {output_path}")

model = load_model()

boxes = extract_bounding_boxes("datasets/flying_strawberries/train/images/1.png", model)
for i, box in enumerate(boxes):
    save_strawberry_roi("datasets/flying_strawberries/train/images/1.png", box, "train/madura", f"{i}.png")



