# Calibración: que el 0.9 signifique 0.9 (Colab)

Notebook de la **Semana 8 (jueves)** de la serie *IA sin humo*.

Muestra cómo **medir** la calibración (Brier, ECE, reliability diagram) de un modelo sobreconfiado y cómo **arreglarla** con Platt (sigmoid) e isotónica — sin cambiar qué clase predice.

Validado localmente: ECE **sin calibrar ≈ 0.10 → sigmoid ≈ 0.045 → isotónica ≈ 0.02**.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `calibracion.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `calibracion.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
