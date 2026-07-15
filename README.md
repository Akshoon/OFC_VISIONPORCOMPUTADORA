# OFC_VISIONPORCOMPUTADORA
# Proyecto Final: ¿Qué detector despliegas, y por qué?
### Evaluación técnica de cuatro paradigmas de visión computacional para el monitoreo de EPP en construcción activa

---

## Integrantes del Equipo
* **Martina Avila** - Ingeniería UAH
* **Iván Muñoz** - Ingeniería UAH
* **Yona Carrera** - Ingeniería UAH
* **Florencia García** - Ingeniería UAH
* **Diego Molina** - Ingeniería UAH

---

## Contexto y Misión
Una empresa constructora requiere un sistema automatizado de monitoreo de seguridad por video en tiempo real. La prioridad crítica del sistema es vigilar el uso de **Equipo de Protección Personal (EPP)** y detectar de forma infalible la violación más peligrosa: **trabajadores sin casco (`NO-Hardhat`)**.

Este proyecto evalúa cuatro paradigmas de detección sobre el mismo conjunto de datos (**Construction Site Safety Dataset - CSS-Data**, bajo licencia CC BY 4.0, compuesto de 2,801 imágenes con splits `train`/`valid`/`test` y anotaciones explícitas de control) para respaldar una decisión de despliegue en producción fundamentada con datos técnicos.

---

## Descripción de lo Implementado
El pipeline reproducible está en **`Proyecto_Final_PIDVC.ipynb`** e incluye:
1. **Frame Sampler (~1 FPS):** justificado porque los VLMs no pueden procesar video a 30 FPS continuo en una GPU T4; un muestreo a 1 FPS simula monitoreo continuo sin perder eventos relevantes en obra.
2. **Detector Cerrado (YOLOv8n):** cargado con pesos `yolov8n.pt` preentrenados en COCO y evaluado sobre el test split.
3. **Vocabulario Abierto (YOLO-World Zero-shot):** evaluado con prompts semánticos de clase (`['hardhat', 'head', 'person']`).
4. **VLM de Grounding (Florence-2-base):** ejecutado bajo la tarea `<OPEN_VOCABULARY_DETECTION>` con el prompt `'hardhat, person, head, worker without helmet'`.
5. **VLM Generativo (Moondream2):** evaluado a nivel de evento con la pregunta binaria *"Is there anyone without a helmet?"*, incluyendo una prueba de sensibilidad con 4 variantes de prompt.
6. **Video comparativo 2×2** (`video_comparativo_modelos.mp4`) que muestra lado a lado las salidas de los cuatro modelos sobre 20 imágenes del test split, con overlay de latencia/FPS por modelo.

---

## Estado real de la evaluación

Al revisar la ejecución del notebook encontramos una diferencia importante entre lo que el pipeline **mide de verdad** y lo que la tabla comparativa final **reporta**. Documentamos ambas cosas por separado para que la sección de resultados sea honesta:

### Lo que sí se ejecutó y midió empíricamente
* **Latencia/FPS de YOLOv8n, YOLO-World y Florence-2**, cronometrados con `time.time()` sobre 50 imágenes reales del test split (Moondream2 solo se cronometró sobre 1 imagen para la demo, no sobre las 50).
* El **video comparativo**, generado corriendo los cuatro modelos sobre 20 imágenes reales.
* La **prueba de sensibilidad al prompt** de Moondream2 sobre 5 imágenes con 4 variantes de pregunta (celda 19), que sí muestra el problema de negación de forma cualitativa.

### Lo que falta por ejecutar/calcular
* **El fine-tuning de YOLOv8n nunca se ejecutó.** La celda 9 solo *imprime* el comando de entrenamiento (`!yolo task=detect mode=train ...`); la línea que cargaría el modelo ya afinado (`YOLO('best.pt')`) está comentada. Todas las mediciones de YOLOv8n en este notebook corresponden a los **pesos base preentrenados en COCO**, que no conocen las clases `Hardhat`/`NO-Hardhat` del dataset. El overlay del video que dice "YOLOv8n (Fine-tuned)" es, por lo tanto, inexacto tal como está el código hoy.
* **El mAP@0.5 y el F1-Score de nivel-evento de la tabla comparativa (celda 21) están escritos a mano como listas de Python**, no calculados contra el ground truth de CSS-Data. No existe en el notebook ninguna función que compare cajas predichas vs. anotadas, ni una que compare la respuesta de Moondream2 contra una etiqueta real por imagen. Son valores de referencia/objetivo, no mediciones propias.
* El gráfico de barras de la celda 21 tampoco usa las latencias medidas (`yolo_lats`, `yolow_lats`, etc.); usa una lista fija `fps_vals = [80, 28, 2, 0.5]`.

### Números realmente medidos vs. los publicados originalmente

| Modelo | Medido en notebook (prom., 50 imágenes) | Visto en el video (por frame) |
| :--- | :---: | :---: |
| YOLOv8n (COCO base, **no** afinado) | 108.0 ms → **9.3 FPS** | 12–56 ms → 18–81 FPS (varía por *warm-up* de GPU) |
| YOLO-World | 50.4 ms → **19.9 FPS** | ~17 ms → ~58 FPS |
| Florence-2 | 514.7 ms → **1.9 FPS** | 741 ms → 1.3 FPS |
| Moondream2 | ~5075 ms (1 imagen) → ~0.2 FPS | 4726–5033 ms → ~0.2 FPS |

La diferencia entre el promedio de 50 imágenes y lo mostrado en el video se explica por *warm-up* de CUDA (la primera inferencia siempre es más lenta); el promedio sobre 50 imágenes es la cifra más representativa y **no llega a 80 FPS** como se afirmaba antes.

**Recomendación para dejar el proyecto listo para entregar:** o bien (a) ejecutar realmente el fine-tuning (`epochs=15` ya está definido) y programar el cálculo de mAP@0.5 con `model.val()` de Ultralytics más una función de F1 a nivel-evento contra las etiquetas de test, o bien (b) mantener la tabla pero rotularla explícitamente como "valores de referencia de la literatura / objetivo de diseño", no como resultado propio. Mientras eso no se resuelva, cualquier cifra de mAP o F1 en este README debe leerse como orientativa.

---

## 📊 Tabla Comparativa (mixta: medido + referencia)

| Característica / Métrica | 1. Cerrado (YOLOv8n) | 2. Vocabulario Abierto (YOLO-World) | 3. VLM de Grounding (Florence-2) | 4. VLM Generativo (Moondream2) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@0.5 (Caja)** | 0.82 *(objetivo, no calculado)* | 0.45 *(objetivo, no calculado)* | 0.51 *(objetivo, no calculado)* | N/A (sin cajas) |
| **F1-Score Nivel-Evento** | 0.84 *(objetivo, no calculado)* | 0.52 *(objetivo, no calculado)* | 0.58 *(objetivo, no calculado)* | 0.62 *(objetivo, no calculado)* |
| **Velocidad medida (50 img., GPU T4)** | **9.3 FPS** (base, no afinado) | **19.9 FPS** | **1.9 FPS** | ~0.2 FPS |
| **Resolución de Negación** | Esperado: Excelente (una vez afinado con clase `NO-Hardhat`) | Débil (confunde el prompt) | Débil (atribución pobre) | Moderada/Inconsistente (confirmado con la prueba de 4 prompts) |
| **Costo Computacional** | Muy Bajo (Edge/CPU local, una vez exportado) | Bajo-Medio | Alto (GPU requerida) | Muy Alto (GPU requerida) |
| **Licencia de Software** | AGPL-3.0 | GPL-3.0 | MIT | Apache-2.0 |

---

## El Hallazgo Pedagógico: El Problema de la Negación
La pregunta crítica de seguridad es una **negación**: *"¿Hay alguien sin casco?"*.
* **YOLOv8n (Cerrado)**, una vez afinado con la clase explícita `NO-Hardhat`, está diseñado para resolver esto directamente — pero, como se explicó arriba, ese fine-tuning todavía no se ejecutó en este notebook, así que el 84% de F1 es una meta a validar, no un resultado confirmado.
* **YOLO-World** y los **VLMs** sí muestran, de forma cualitativa (video + prueba de 4 prompts en Moondream2), que tropiezan con la negación y los atributos: la palabra *"helmet"/"hardhat"* en el prompt confunde al alineador texto-imagen, provocando que marquen a infractores como protegidos o generen falsos positivos. La celda 19 documenta cómo la misma imagen recibe respuestas contradictorias de Moondream2 según cómo se formule la pregunta — esa evidencia sí es real y reproducible.
* Nota de implementación: la función `moondream_answer_as_detection()` del video clasifica la respuesta como "alerta" si el texto contiene las palabras "yes"/"sí"/"there is"/"there are", sin analizar la negación inicial ("No, ..."). Es un heurístico simple para el overlay del video, no una medición formal — pero irónicamente reproduce el mismo problema de atribución débil que el proyecto busca demostrar.

---

## Decisión de Despliegue: ¿Cuál desplegar en producción y por qué?
**La recomendación técnica sigue apuntando a desplegar el Detector Cerrado YOLOv8n**, condicionada a completar el fine-tuning pendiente:

### Justificación Técnica y de Negocio:

1. **Precisión en Seguridad (Negación):** un falso negativo (no detectar a un trabajador sin casco) puede provocar accidentes fatales y multas severas. El diseño de YOLOv8n con clase explícita `NO-Hardhat` es la única arquitectura que puede resolver la negación de forma consistente por diseño; falta correr el entrenamiento para confirmarlo con números propios.
2. **Tiempo Real y FPS:** con los pesos base (sin afinar) YOLOv8n ya corre a ~9-80 FPS según warm-up, muy por encima de Florence-2 (~1-2 FPS) y Moondream2 (~0.2 FPS). Una vez exportado a ONNX/TensorRT debería acercarse o superar los 30 FPS necesarios para cámaras de obra en tiempo real; esto debe confirmarse con benchmarks post-optimización.
3. **Costo de Operación (Infraestructura):** YOLOv8n puede exportarse a ONNX o TensorRT y ejecutarse localmente (Edge computing) en mini-PCs baratas o CPUs estándar, mientras que mantener GPU en la nube 24/7 para VLMs cuesta cientos de dólares al mes.
4. **Análisis de Licencias:** YOLOv8n usa **AGPL-3.0**, lo cual obliga a liberar el código de cualquier software derivado si se ofrece como SaaS web. Para la constructora esto se resuelve con **despliegue local/Edge (On-Premise)**: al no distribuirse ni ofrecerse como servicio público SaaS, no se activa la cláusula Copyleft, protegiendo la propiedad intelectual del software de integración.

### Evaluación de Escenarios Alternativos:

* **¿Qué pasa si mañana nos piden detectar "guantes"?**
  YOLO-World o Florence-2 ganarían ventaja en el corto plazo por ser *zero-shot*. Sin embargo, dado su bajo desempeño esperado en negación, la solución de ingeniería correcta sigue siendo recopilar una muestra de imágenes de guantes y hacer fine-tuning incremental sobre YOLOv8n.
* **¿Qué pasa si un falso negativo de "sin casco" implica una multa millonaria?**
  El parámetro clave a optimizar es el **Recall** de la clase `NO-Hardhat`. YOLOv8n permite bajar el umbral de confianza (ej. `conf=0.25`) para capturar más sospechas de infracción, algo mucho más difícil de controlar con prompts en VLMs.

---
