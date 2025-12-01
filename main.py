from ultralytics import YOLO
import cv2
import numpy as np
import time

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
# FUNCIÓN ROBUSTA: COLOR DOMINANTE POR HISTOGRAMA HSV
# ---------------------------------------------------------
def detect_color(roi):

    h0 = int(roi.shape[0] * 0.15)
    h1 = int(roi.shape[0] * 0.85)
    w0 = int(roi.shape[1] * 0.15)
    w1 = int(roi.shape[1] * 0.85)

    roi_center = roi[h0:h1, w0:w1]

    hsv = cv2.cvtColor(roi_center, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    hist = cv2.calcHist([H], [0], None, [180], [0, 180])
    h_peak = int(np.argmax(hist))

    s_mean = S.mean()
    v_mean = V.mean()

    if v_mean < 50:
        return "NEGRO"
    if s_mean < 40 and v_mean > 180:
        return "BLANCO"
    if s_mean < 40:
        return "GRIS"

    if h_peak < 10 or h_peak > 170:
        return "ROJO"
    if 10 <= h_peak <= 25:
        return "NARANJO"
    if 25 < h_peak <= 35:
        return "AMARILLO"
    if 35 < h_peak <= 85:
        return "VERDE"
    if 85 < h_peak <= 130:
        return "AZUL"
    if 130 < h_peak <= 160:
        return "MORADO"

    return "OTRO"

# ---------------------------------------------------------
# CONTADORES DE COLORES + ESTADOS DE DETECCIÓN
# ---------------------------------------------------------
color_counts = {
    "ROJO": 0, "NARANJO": 0, "AMARILLO": 0, "VERDE": 0,
    "AZUL": 0, "MORADO": 0, "BLANCO": 0, "NEGRO": 0,
    "GRIS": 0, "OTRO": 0
}

# Estado si el color está siendo visto actualmente
color_present = {c: False for c in color_counts.keys()}

# ---------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, verbose=False)[0]
    annotated = frame.copy()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    detected_colors_this_frame = set()

    # Procesar detecciones YOLO
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        color_name = detect_color(roi)
        detected_colors_this_frame.add(color_name)

        # -------------------------
        # CONTADOR REAL (1 vez por aparición)
        # -------------------------
        if not color_present[color_name]:
            color_counts[color_name] += 1
            color_present[color_name] = True

        # Dibujar bounding box
        box_color = color_bgr[color_name]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated, color_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # -----------------------------------------------------
    # Reset de colores NO detectados este frame
    # -----------------------------------------------------
    for color in color_present.keys():
        if color not in detected_colors_this_frame:
            color_present[color] = False

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

    dashboard = np.hstack((annotated, panel))

    cv2.imshow("YOLO + Color + Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
