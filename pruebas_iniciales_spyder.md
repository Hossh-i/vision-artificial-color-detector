Pruebas Iniciales del Proyecto en Spyder
Autores: Fernanda Ríos, Alonso Avello, Pablo Arias, Rivaldo Rodríguez  
Curso: Visión Artificial – Taller 2  
Archivo: pruebas_iniciales_spyder.md

Estas pruebas se realizaron antes de implementar el proyecto final para asegurar que:

- La webcam funciona
- La instalación de OpenCV es correcta
- El modelo YOLOv8 carga correctamente
- El sistema puede procesar video en un loop
- La GPU está disponible (si se usa)

---

# 🟩 1. Prueba de Webcam

Código utilizado para validar que la cámara está operativa:

```python
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    print("Webcam OK")
    cv2.imshow("Test Webcam", frame)
    cv2.waitKey(2000)   # Mostrar por 2 segundos
else:
    print("Error: no se pudo acceder a la webcam")

cap.release()
cv2.destroyAllWindows()
```

**Resultado esperado:**  
Ventana con la imagen de la cámara por 2 segundos.

---

# 🟩 2. Prueba de Loop de Video (FPS)

Esto confirma que Spyder puede refrescar la ventana correctamente.

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Loop Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

**Resultado esperado:**  
Video fluido y se cierra con tecla **q**.

---

# 🟩 3. Prueba de YOLO (carga del modelo)

Validación de que Ultralytics YOLOv8 está instalado y funciona:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print("Modelo YOLO cargado correctamente")
```

**Resultado esperado:**  
Mensaje en consola: *"Modelo YOLO cargado correctamente"*.

---

# 🟩 4. Prueba de YOLO sobre un solo frame

Para verificar que el modelo puede ejecutar detecciones:

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    results = model(frame, verbose=False)[0]
    annotated = results.plot()
    cv2.imshow("YOLO Test", annotated)
    cv2.waitKey(1500)
else:
    print("No se pudo capturar frame para YOLO")

cap.release()
cv2.destroyAllWindows()
```

**Resultado esperado:**  
Cuadro con cajas verdes alrededor de los objetos detectados.

---

# 🟩 5. Prueba de GPU (si está disponible)

```python
import torch

print("PyTorch GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre GPU:", torch.cuda.get_device_name(0))
```

**Resultado esperado:**  
- En CPU: `False`
- En GPU: `True` + nombre de la tarjeta

*(Nota: muchas laptops no permiten usar GPU desde Spyder)*

---

# 🟩 6. Notas importantes para Spyder

- **NO usar `cv2.getWindowProperty`** para cerrar ventanas (causa pantalla gris).  
- Cerrar las ventanas SIEMPRE con tecla **"q"**.  
- Spyder puede congelarse si abusas del GUI; para la demo final usar Terminal es más estable.
---
