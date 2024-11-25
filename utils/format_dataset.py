import os
import cv2

# Función para encontrar bounding boxes en las imágenes de etiquetas
def find_bounding_boxes(label_img):
    # Detectar las áreas no negras (que representan objetos) como bounding boxes
    contours, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))
    return bboxes

# Función para convertir los bounding boxes al formato YOLO
def convert_to_yolo_format(bboxes, img_width, img_height):
    yolo_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        # Calcular el centro y dimensiones normalizadas
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        yolo_bboxes.append((0, x_center, y_center, norm_width, norm_height))  # Asumiendo class_id = 0 para frutillas
    return yolo_bboxes

# Función para guardar las etiquetas en formato YOLO
def save_yolo_labels(yolo_bboxes, label_path):
    with open(label_path, 'w') as f:
        for bbox in yolo_bboxes:
            class_id, x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Directorios de las imágenes y etiquetas
dataset_dir = os.getcwd()
splits = ['train', 'val', 'test']

for split in splits:
    image_dir = os.path.join(dataset_dir, "datasets", split, "images")
    label_dir = os.path.join(dataset_dir, "datasets", split, "labels")

    # Crear las carpetas de salida si no existen
    os.makedirs(label_dir, exist_ok=True)

    for img_file in os.listdir(image_dir):
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            # Leer la imagen de etiquetas correspondiente
            img_name = os.path.splitext(img_file)[0]
            label_img_path = os.path.join(label_dir, img_name + '.png')
            label_txt_path = os.path.join(label_dir, img_name + '.txt')

            # Comprobar si existe la imagen de etiquetas
            if os.path.exists(label_img_path):
                # Cargar la imagen de etiquetas en escala de grises
                label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)

                # Obtener las dimensiones de la imagen original
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[:2]

                # Encontrar bounding boxes en la imagen de etiquetas
                bboxes = find_bounding_boxes(label_img)

                # Convertir los bounding boxes al formato YOLO
                yolo_bboxes = convert_to_yolo_format(bboxes, img_width, img_height)

                # Guardar las etiquetas en formato YOLO
                save_yolo_labels(yolo_bboxes, label_txt_path)

                # Eliminar la imagen de etiquetas .png
                os.remove(label_img_path)

print("Conversión completada y eliminación de imágenes de etiquetas finalizada.")



