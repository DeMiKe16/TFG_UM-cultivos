import os
import cv2 as cv
from ultralytics import YOLO
import numpy as np

def putText(img, string, orig=(5,16), color=(255,255,255), div=2, scale=1, thickness=1):
    (x,y) = orig
    if div > 1:
        (w,h), b = cv.getTextSize(string, cv.FONT_HERSHEY_PLAIN, scale, thickness)
        img[y-h-4:y+b, x-3:x+w+3] //= div
    cv.putText(img, string, (x,y), cv.FONT_HERSHEY_PLAIN, scale, color, thickness, cv.LINE_AA)

# Carga un modelo preentrenado YOLOv8 (por ejemplo, 'yolov8n.pt')
model = YOLO('weights/best.pt')

# Carga una imagen en la que deseas realizar predicciones (reemplaza 'imagen.jpg' con la ruta de tu imagen)
image_path = 'prueba.jpg'
results = model(image_path)  # Devuelve una lista de objetos Results


# Dibuja las cajas delimitadoras en la imagen
for result in results:
    boxes = result.boxes.data  # Utiliza el atributo boxes para obtener las coordenadas de las cajas
    conteo = len(boxes)
    black_bg = np.zeros_like(results[0].orig_img)
    putText(result.orig_img, f"Naranjas: {conteo}", (10, 30))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        label = "Naranja"
        cv.rectangle(result.orig_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        putText(result.orig_img, f"{label}", (x1, y1 - 10))




# Muestra la imagen con las cajas dibujadas
cv.imshow('Imagen con cajas', result.orig_img)
cv.waitKey(0)
cv.destroyAllWindows()