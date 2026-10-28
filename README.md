# Detector de figuras geométricas para manufactura

Reconoce 19 figuras (imágenes 512×512, fondo blanco, figura negra) o responde «figura no reconocida».
Python corre en el navegador con Pyodide; no hay servidor.

## Publicar
1. Crea un repositorio y sube **todo el contenido** de esta carpeta a la raíz (incluida `figuras/`).
2. Settings → Pages → Branch `main`, carpeta `/ (root)`.
3. Abre `https://USUARIO.github.io/REPOSITORIO/`.

## Archivos
- `index.html` interfaz y pestañas · `detector.py` preprocesado, características, grafo, búsqueda
- `genetico.py` algoritmo genético · `plantillas.json` plantillas, pesos y umbral · `figuras/` imágenes modelo
- `entrenar.py` (opcional, local) regenera `plantillas.json`: `pip install numpy scipy scikit-image networkx pillow` y `python entrenar.py`

## Notas
- Se usa scikit-image en lugar de OpenCV (cubre lo mismo y está garantizado en Pyodide).
- Se excluyó el octágono (se confundía con el hexágono en las pruebas).
- Resultados sintéticos del prototipo: ver pestaña «Genético». No garantizan el mismo desempeño con fotos reales.
