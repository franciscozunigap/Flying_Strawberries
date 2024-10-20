import matplotlib.pyplot as plt
import cv2

# Dibuja las detecciones en la imagen. NO FUNCIONA
def draw_detections(img, detecciones):
    # Recorrer cada detección y dibujarla en la imagen
    for deteccion in detecciones:
        # Extraer la caja y normalizar las coordenadas
        box = deteccion['caja']
        height, width, _ = img.shape
        ymin, xmin, ymax, xmax = (box[0] * height, box[1] * width,
                                   box[2] * height, box[3] * width)

        # Convertir coordenadas a enteros
        xmin, xmax = int(xmin), int(xmax)
        ymin, ymax = int(ymin), int(ymax)

        # Dibujar la caja de detección
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Azul

        # Extraer la puntuación y la clase
        score = deteccion['puntuacion']
        class_id = deteccion['clase']

        # Escribir la puntuación en la imagen
        label = f'Class: {class_id}, Score: {score:.2f}'
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)  # Azul

    return img

# Muestra la imagen con las detecciones.
def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
