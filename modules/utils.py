import os
import matplotlib.pyplot as plt
import cv2

def display_results(results_list):
    for original_image, detected_image, _ in results_list:
        # Crear figura con dos subgráficos
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Crear dos subgráficos

        # Mostrar la imagen original (convertir de BGR a RGB)
        axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Imagen Original")
        axs[0].axis('off')  # Ocultar los ejes

        # Mostrar la imagen con detecciones
        axs[1].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Imagen con Detecciones")
        axs[1].axis('off')  # Ocultar los ejes

        plt.tight_layout()
        plt.show()

def save_predicts(results_list, output_predict_dir):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_predict_dir, exist_ok=True)

    # Guardar cada imagen detectada en el directorio de salida
    for _, detected_image, image_file in results_list:
        output_image_path = os.path.join(output_predict_dir, f"result_{image_file}")
        cv2.imwrite(output_image_path, detected_image)  # Guardar la imagen con detecciones