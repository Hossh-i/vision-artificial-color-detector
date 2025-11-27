from ultralytics import YOLO
import cv2
import numpy as np
import time
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# CARGA DEL MODELO YOLO
# ---------------------------------------------------------
model = YOLO("yolov8n.pt")
print("Modelo YOLO cargado correctamente")


# ---------------------------------------------------------
# COLORES BGR PARA BOUNDING BOXES
# ---------------------------------------------------------
color_bgr = {
    "ROJO": (0, 0, 255),
    "NARANJO": (0, 128, 255),
    "AMARILLO": (0, 255, 255),
    "VERDE": (0, 255, 0),
    "AZUL": (255, 0, 0),
    "MORADO": (255, 0, 255),
    "BLANCO": (255, 255, 255),
    "NEGRO": (0, 0, 0),
    "GRIS": (180, 180, 180),
    "OTRO": (0, 255, 255)
}


# ---------------------------------------------------------
# FUNCIÓN OPTIMIZADA: COLOR DOMINANTE CON K-MEANS
# ---------------------------------------------------------
def detect_color(roi):

    # 1) Tomar región central del ROI (evita bordes/fondo)
    h0 = int(roi.shape[0] * 0.20)
    h1 = int(roi.shape[0] * 0.80)
    w0 = int(roi.shape[1] * 0.20)
    w1 = int(roi.shape[1] * 0.80)

    roi_center = roi[h0:h1, w0:w1]

    # 2) Reducir tamaño para mejorar FPS
    roi_center = cv2.resize(roi_center, (50, 50))

    # 3) Convertir a HSV
    hsv = cv2.cvtColor(roi_center, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # 4) Submuestreo (reduce carga)
    pixels = pixels[::4]

    # 5) K-means (2 clusters = color rápido y estable)
    kmeans = KMeans(n_clusters=2, n_init=5)
    kmeans.fit(pixels)

    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    h, s, v = dominant

    # 6) Reglas de clasificación
    if v < 50:
        return "NEGRO"
    if s < 40 and v > 180:
        return "BLANCO"
    if s < 40:
        return "GRIS"

    if h < 10 or h > 170:
        return "ROJO"
    if 10 <= h <= 25:
        return "NARANJO"
    if 25 < h <= 35:
        return "AMARILLO"
    if 35 < h <= 85:
        return "VERDE"
    if 85 < h <= 130:
        return "AZUL"
    if 130 < h <= 160:
        return "MORADO"

    return "OTRO"


# ---------------------------------------------------------
# CONTADORES DE COLORES
# ---------------------------------------------------------
color_counts = {
    "ROJO": 0,
    "NARANJO": 0,
    "AMARILLO": 0,
    "VERDE": 0,
    "AZUL": 0,
    "MORADO": 0,
    "BLANCO": 0,
    "NEGRO": 0,
    "GRIS": 0,
    "OTRO": 0
}


# ---------------------------------------------------------
# LOOP PRINCIPAL (YOLO + COLOR + DASHBOARD)
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLO con mayor conf (reduce ruido)
    results = model(frame, conf=0.45, verbose=False)[0]
    annotated = frame.copy()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Procesar detecciones YOLO
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # FILTRO: ignorar objetos muy pequeños (mejora FPS)
        w = x2 - x1
        h = y2 - y1
        if w * h < 3000:
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Detectar color robusto
        color_name = detect_color(roi)
        color_counts[color_name] += 1

        # Dibujar bounding box
        box_color = color_bgr[color_name]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated, color_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # -----------------------------------------------------
    # PANEL (DASHBOARD)
    # -----------------------------------------------------
    panel = np.zeros((annotated.shape[0], 250, 3), dtype=np.uint8)

    y = 30
    cv2.putText(panel, "CONTADOR", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y += 40

    for color, count in color_counts.items():
        cv2.putText(panel, f"{color}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        y += 25

    cv2.putText(panel, f"FPS: {fps:.1f}",
                (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Concatenar imagen + panel
    dashboard = np.hstack((annotated, panel))

    cv2.imshow("YOLO + Color + Dashboard", dashboard)

    # Cerrar con "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
