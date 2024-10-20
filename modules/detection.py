import tensorflow as tf
import os

# NO FUNCIONA
def load_efficientdet_model():
    model_path = os.path.join(os.getcwd(), "models", "efficientdet-d0")
    model = tf.saved_model.load(model_path)  # Ruta al modelo
    return model

# NO FUNCIONA
def detect_fruit(img_preprocesada, modelo_deteccion, umbral=0.3):
    # Realizar la detección
    detecciones = modelo_deteccion(img_preprocesada)

    # Obtener las cajas, puntuaciones y clases de las detecciones
    boxes = detecciones['detection_boxes'][0].numpy()
    scores = detecciones['detection_scores'][0].numpy()
    classes = detecciones['detection_classes'][0].numpy().astype(int)
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Classes:", classes)


    # Filtrar detecciones según el umbral
    detecciones_filtradas = []
    for i in range(len(scores)):
        if scores[i] >= umbral:  # Solo mantener detecciones con alta puntuación
            deteccion = {
                'caja': boxes[i],
                'puntuacion': scores[i],
                'clase': classes[i]
            }
            detecciones_filtradas.append(deteccion)

    return detecciones_filtradas


