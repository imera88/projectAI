from ultralytics import YOLO
import glob
import cv2
import os  # Importa os para manejar rutas y nombres de archivos de forma más efectiva

# loading a custom model
#model = YOLO('yolov9c.pt')
model = YOLO('yolov9c.pt')


# Cambio en la forma de iterar sobre los archivos
for image_path in glob.glob("test/*.jpg"):  # Usamos * directamente para todos los archivos .jpg
    image = cv2.imread(image_path)
    # Extraer nombre del archivo sin la extensión
    imagename_withoutextension = os.path.basename(image_path).split('.')[0]
    results = model.predict(image_path, imgsz=640, conf=0.25, iou=0.45)  # Asegúrate de que los parámetros sean correctos para model.predict
    results = results[0]  
    for i in range(len(results.boxes)):
        box = results.boxes[i]
        tensor = box.xyxy[0]
        x1 = int(tensor[0].item())
        y1 = int(tensor[1].item())
        x2 = int(tensor[2].item())
        y2 = int(tensor[3].item())
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    # Guarda la imagen en el directorio de salida
    cv2.imwrite(f"test_output/{imagename_withoutextension}.jpg", image)
 