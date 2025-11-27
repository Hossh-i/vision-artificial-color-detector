# 🎨 vision-artificial-color-detector  
### Sistema de detección de objetos y clasificación de color en tiempo real usando YOLOv8 + K-Means

**Autores:** Fernanda Ríos, Alonso Avello, Pablo Arias, Rivaldo Rodríguez 
**Curso:** Visión Artificial – Taller 2  
**Lenguaje:** Python  
**Tecnologías:** YOLOv8, OpenCV, Scikit-Learn, HSV Color Space  

---

## 📘 Descripción General

Este proyecto implementa un sistema en tiempo real capaz de:

- Detectar objetos utilizando **YOLOv8 pre-entrenado**  
- Extraer la región de interés (**ROI**) de cada detección  
- Calcular el **color dominante** del objeto mediante K-Means en el espacio de color HSV  
- Clasificar el color en categorías:  
  **Rojo, Naranjo, Amarillo, Verde, Azul, Morado, Blanco, Gris, Negro, Otro**  
- Contar la cantidad de objetos por color  
- Mostrar un **dashboard lateral en tiempo real** con:  
  ✔ Conteo por color  
  ✔ FPS del sistema  
  ✔ Ventana principal con cajas de detección coloreadas  

Este es el proyecto desarrollado para el Taller 2 del curso de Visión Artificial.

---

## 🎯 Objetivos del Proyecto

1. Implementar detección de objetos usando YOLOv8.
2. Analizar la región donde se encuentra cada objeto.
3. Determinar el color dominante de forma robusta.
4. Construir un dashboard informativo en tiempo real.
5. Mantener un rendimiento visual estable (10–20 FPS).

---

## 🧠 Arquitectura del Sistema
Cámara → YOLOv8 → Bounding Box → ROI → HSV → K-Means → Clasificación de Color → Dashboard

- **YOLOv8** detecta objetos y entrega las coordenadas del bounding box.  
- Se extrae el **ROI** central del objeto detectado.  
- Convertimos a **HSV** (más estable ante luz).  
- Aplicamos **K-Means (k=2)** para hallar el color dominante.  
- Clasificamos según rangos de tono, saturación y valor.
- Se actualiza un **panel lateral** con conteos y FPS.

---

## 🖥 Tecnologías Utilizadas

- **Python 3.10**
- **OpenCV** (captura de video, procesamiento)
- **Ultralytics YOLOv8**
- **NumPy**
- **Scikit-Learn (K-Means)**
- Anaconda + Spyder

---

## 🛠 Instalación del Proyecto

Las instrucciones detalladas están en:

🔗 **instalacion_anaconda.md**


## ▶ Ejecución
Ejecutar en el terminal o Spyder:
python main.py
Cerrar con la tecla q.

## 🧪 Pruebas Iniciales
Las pruebas documentadas incluyen:
- Funcionamiento de webcam
- Carga de YOLO
- Test de detección en un frame
- Prueba de GPU
- Loop en tiempo real

Ver archivo:
🔗 pruebas_iniciales_spyder.md

##📊 Dashboard en Tiempo Real
El panel lateral muestra:
- Conteo acumulado de objetos por color
- FPS del sistema
- Color asignado a cada bounding box
- Visualización limpia y estable

## 📈 Métricas
| Métrica         | Valor                                  |
| --------------- | -------------------------------------- |
| FPS promedio    | 10–20 FPS según CPU                    |
| Método de color | K-Means en HSV                         |
| Detección YOLO  | Modelo YOLOv8n pre-entrenado           |
| Ruido reducido  | Filtro de objetos pequeños + conf=0.45 |

## 🎥 Video Demo (pendiente)
El video demostrará:
- Detección en tiempo real
- Color dominante correcto
- Dashboard actualizándose
- FPS estables
- Cierre con tecla q


## ✔ Conclusiones
- El sistema cumple con todos los requisitos del taller.
- YOLOv8 permite detección robusta sin entrenamiento adicional.
- El análisis de color en HSV con K-Means mejora la precisión.
- El dashboard entrega una visualización clara y útil.
- Se logró mantener un rendimiento estable en tiempo real.

## 🚀 Próximas Mejoras
- Tracking (SORT/DeepSORT) para evitar doble conteo.
- Guardar estadísticas en archivo CSV.
- Ajuste automático de color según luz ambiente.
- Implementar interfaz gráfica (PyQt5).

## 📄 Licencia
Proyecto académico. Uso libre con fines educativos.
