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
        if results.boxes and results.boxes.xyxy is not None:
            for box, conf, cls_idx in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                confidence = f"{conf*100:.2f}%"  # Asumiendo que 'conf' es un solo valor de confianza
                label = model.names[int(cls_idx)]  # Asumiendo que 'cls_idx' es el índice de la clase
                #detections.append({"label": label, "confidence": confidence, "box": [x1, y1, x2, y2]})
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            continue
    # Guarda la imagen en el directorio de salida
    cv2.imwrite(f"test_output/{imagename_withoutextension}.jpg", image)
 