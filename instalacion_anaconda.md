# ---------------------------------------------------------
# INSTALACIÓN DEL ENTORNO PARA PROYECTO DE VISIÓN ARTIFICIAL
# YOLOv8 + OpenCV + K-Means (Fernanda Ríos, Alonso Avello, Pablo Arias, Rivaldo Rodriguez)
# ---------------------------------------------------------

# 1. Crear un entorno nuevo en Anaconda
conda create -n vision-dl python=3.10

# 2. Activar el entorno
conda activate vision-dl

# ---------------------------------------------------------
# PAQUETES REQUERIDOS
# ---------------------------------------------------------

# 3. Instalar OpenCV (procesamiento de imágenes), Ultralytics (YOLOv8) y NumPy (matrices y cálculos)
pip install ultralytics opencv-python matplotlib numpy


# 4. Instalar scikit-learn (K-Means para color dominante)
pip install scikit-learn

# 5. Instalar SciPy (dependencia científica)
pip install scipy

# 6. Instalar FilterPy (si se usa tracking SORT en un futuro)
pip install filterpy

# ---------------------------------------------------------
# COMPROBACIONES OPCIONALES
# ---------------------------------------------------------

# Ver versión de Python
python --version

# Ver versión de OpenCV
python - << "EOF"
import cv2
print("OpenCV:", cv2.__version__)
EOF

# Ver que YOLO funciona
python - << "EOF"
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("YOLO cargado correctamente")
EOF

# Ver que la webcam funciona
python - << "EOF"
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("Webcam:", ret)
cap.release()
EOF

# ---------------------------------------------------------
# NOTAS IMPORTANTES
# ---------------------------------------------------------
# - Este entorno funciona correctamente con Spyder o VSCode.
# - Evitar usar getWindowProperty en Spyder (causa pantalla gris).
# - Se recomienda ejecutar la demo final desde terminal con: python main.py
# - Para cerrar la ventana usar la tecla "q".
