import matplotlib.pyplot as plt
import cv2

def extract_boxes(results):
    # Lista para guardar las coordenadas de las cajas
    bounding_boxes = []

    # Acceder a todas las detecciones
    detections = results[0].boxes

    # Iterar sobre las detecciones y extraer las coordenadas de las bounding boxes
    for box in detections:
        # Coordenadas [x1, y1, x2, y2]
        coords = box.xyxy.cpu().numpy().astype(int)

        # Añadir solo las coordenadas de la bounding box
        bounding_boxes.append((coords[0][0], coords[0][1], coords[0][2], coords[0][3]))

    return bounding_boxes

def classify_strawberry(results):
    # Obtener la clase con mayor probabilidad
    class_number = results[0].probs.top1

    # Obtener el porcentaje de confianza de la clasificación
    percent_number = results[0].probs.top1conf.item()

    return class_number, percent_number

def inference(image_path, detect_model, classify_model):
    # Leer la imagen original
    image = cv2.imread(image_path)

    # Detectar frutillas en la imagen
    detect_results = detect_model(image_path)

    # Extraer las cajas delimitadoras de las frutillas detectadas
    boxes = extract_boxes(detect_results)

    colors = {0: "#00FF00", 1: "#FF0000"}
    class_name = {0: "madura", 1: "inmadura"}

    # Convertir de BGR a RGB para matplotlib
    image_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crear una figura y un eje
    fig, ax = plt.subplots(1)

    # Mostrar la imagen
    ax.imshow(image_plt)

    # Iterar sobre cada caja delimitadora y dibujarla en la imagen
    for (x1, y1, x2, y2) in boxes:
        # Recortar la frutilla de la imagen original usando la caja delimitadora
        strawberry_crop = image[y1:y2, x1:x2]

        classify_results = classify_model(strawberry_crop)
        class_number, percent_number = classify_strawberry(classify_results)

        color = colors.get(class_number, "y")
        name = class_name.get(class_number, "No identificada")

        # Crear un rectángulo en la imagen
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Añadir el texto de la clase y el porcentaje de acierto
        plt.text(x1, y1 - 10, f'{name}, {percent_number*100:.1f}%', color=color, fontsize=6, weight='bold')

    # Mostrar la imagen con las cajas delimitadoras
    plt.axis("off")
    plt.show()