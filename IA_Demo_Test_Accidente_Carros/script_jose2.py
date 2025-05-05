import os
from pathlib import Path
import cv2
from ultralytics import YOLO

# Carga el modelo YOLO
model = YOLO('yolov9c.pt')  # Asegúrate de usar el modelo correcto y la ruta al archivo de pesos.

# Directorios de entrada y salida
input_dir = Path('./test')
output_dir = Path('./test_output')
output_dir.mkdir(exist_ok=True)  # Crea el directorio de salida si no existe

# Procesa todas las imágenes en el directorio de entrada
for image_path in input_dir.glob('*.jpg'):  # Asume que las imágenes son JPEGs
    # Carga la imagen
    image = cv2.imread(str(image_path))
    if image is None:
        continue  # Si por alguna razón la imagen no se puede cargar, continua con la siguiente

    # Procesa la imagen con YOLO
    results = model(image)


    for box, label, conf in zip(results.boxes.xyxy[0], results.labels[0], results.confidences[0]):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image, f'{model.names[label]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Guarda la imagen procesada en el directorio de salida
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)
    print(f"Processed and saved {output_path}")

print("Processing complete.")
