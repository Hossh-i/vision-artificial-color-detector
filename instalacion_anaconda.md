# ---------------------------------------------------------
# INSTALACIÓN DEL ENTORNO PARA PROYECTO DE VISIÓN ARTIFICIAL
# YOLOv8 + OpenCV + K-Means (Fernanda Ríos, Alonso Avello, Pablo Arias, Rivaldo Rodríguez)
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


