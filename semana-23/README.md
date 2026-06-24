# Detección de anomalías: lo inesperado, no lo alto (Colab)

Notebook de la **Semana 23 (jueves)** de la serie *IA sin humo*.

Muestra por qué un **umbral fijo** sobre el valor crudo de una serie estacional es una fábrica de falsas alarmas, y cómo detectar sobre la **diferencia estacional** (cada día vs el mismo día de la semana pasada) + umbral robusto (mediana/MAD) encuentra lo realmente inesperado. Simulación con numpy.

Validado localmente: umbral fijo → recall 1/4 y 36 falsos positivos (picos estacionales). Diferencia estacional → recall 4/4 con pocos falsos (ecos a +7 días).

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `anomalias.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `anomalias.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
