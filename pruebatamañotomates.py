import os
import cv2 as cv
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation
import numpy as np
import math

def calcular_distancia_angular(fov, distance, image_width):

        # Calcular el ángulo por píxel
        angle_per_pixel = fov / image_width
        
        # Calcular la distancia angular entre los dos puntos
        distance_angle = distance * angle_per_pixel
        
        return distance_angle

def calcular_tamaño_real(tamaño_angular, distancia):
    # Convertir el tamaño angular de grados a radianes
    tamaño_angular_radianes = math.radians(tamaño_angular)
    
    # Calcular el tamaño real del objeto
    tamaño_real = 2 * distancia * math.tan(tamaño_angular_radianes / 2)
    
    return tamaño_real

def putText(img, string, orig=(5,16), color=(255,255,255), div=2, scale=1, thickness=1):
    (x,y) = orig
    if div > 1:
        (w,h), b = cv.getTextSize(string, cv.FONT_HERSHEY_PLAIN, scale, thickness)
        img[y-h-4:y+b, x-3:x+w+3] //= div
    cv.putText(img, string, (x,y), cv.FONT_HERSHEY_PLAIN, scale, color, thickness, cv.LINE_AA)

# Carga un modelo preentrenado YOLOv8 (por ejemplo, 'yolov8n.pt')
model = YOLO('entrenamiento tomates/entrenamiento 2/weights/best.pt')

# Carga una imagen en
# la que deseas realizar predicciones (reemplaza 'imagen.jpg' con la ruta de tu imagen)
image_path = 'tomate1.jpg'
results = model(image_path)  # Devuelve una lista de objetos Results

dist_obj = distance_calculation.DistanceCalculation()
dist_obj.set_args(names="Tomate", view_img=True)

distancia = 16 # cm

FOV = 67 # grados

ancho_foto = 1536

# Dibuja las cajas delimitadoras en la imagen
for result in results:
    boxes = result.boxes.data  # Utiliza el atributo boxes para obtener las coordenadas de las cajas
    conteo = len(boxes)
    black_bg = np.zeros_like(results[0].orig_img)
    putText(result.orig_img, f"Tomates: {conteo}", (10, 30))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        label = "Tomate"
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center = (int(center_x), int(center_y))
        
        side_length = x2 - x1
        radius = side_length / 2   
        
        print(radius)
        angulo = calcular_distancia_angular(FOV, radius*2, ancho_foto)
        print(angulo)
        tam_real = calcular_tamaño_real(angulo, distancia)
        putText(result.orig_img, f"{tam_real:.1f}cm", (x1, y1 - 10))
        
        
        cv.circle(result.orig_img, center, int(radius), (255, 255, 255), 2)
        cv.circle(result.orig_img, center, 5, (255, 255, 255), -1)



cv.imwrite('imagen_forma.jpg', result.orig_img)
# Muestra la imagen con las cajas dibujadas
cv.imshow('Imagen con cajas', result.orig_img)
cv.waitKey(0)
cv.destroyAllWindows()