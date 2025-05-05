from fastapi import FastAPI, File, UploadFile,HTTPException,Form
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import random
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Suponiendo que el modelo esté cargado aquí
#model = YOLO('Predicciones/yolov9c.pt')
model = YOLO('best.pt')

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...), input_id: str = Form(...), modelo: str = Form(...), input_file_name: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    input_id = input_id
    modelo = modelo
    input_file_name = input_file_name

    # we define the output folder as the input_id + _ + modelo
    output_folder = f"{input_id}_{modelo}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Guardar la imagen temporalmente si es necesario para procesamiento
    cv2.imwrite(f"{output_folder}/{input_file_name}_output.jpg", image)
    results = model([f"{output_folder}/{input_file_name}_output.jpg"])[0]  # Procesar la imagen y obtener el primer objeto de Results

    #results = model(['temp.jpg'])[0]  # Procesar la imagen y obtener el primer objeto de Results

    detections = []
    if results.boxes and results.boxes.xyxy is not None:
        for box, conf, cls_idx in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            confidence = f"{conf*100:.2f}%"  # Asumiendo que 'conf' es un solo valor de confianza
            label = model.names[int(cls_idx)]  # Asumiendo que 'cls_idx' es el índice de la clase
            detections.append({"label": label, "confidence": confidence, "box": [x1, y1, x2, y2]})
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        raise HTTPException(status_code=404, detail="No detections found")

    # sobreescribimos la imagen de output
    cv2.imwrite(f"{output_folder}/{input_file_name}_output.jpg", image)
    #cv2.imwrite('output.jpg', image)

    return JSONResponse(content={"image_url": f"{output_folder}/{input_file_name}_output.jpg", "detections": detections})
    #return JSONResponse(content={"image_url": "output.jpg", "detections": detections})




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/hello")
def hello_world():
    return {"message": f"Hola mundo. Nr {random.randint(1, 100)}"}

if __name__ == "__main__":
    uvicorn.run(app)