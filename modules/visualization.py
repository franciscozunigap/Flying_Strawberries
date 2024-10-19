import matplotlib.pyplot as plt
import cv2

# Dibuja las detecciones en la imagen.
def draw_detections(img, detecciones, clases=["Frutilla"]):
    for detection in detecciones:
        # Obtener la caja delimitadora y la clase
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        score = detection['score']

        # Dibujar la caja delimitadora y la etiqueta
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{clases[class_id]}: {score:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

# Muestra la imagen con las detecciones.
def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
