# Forecast: baseline vs ML (Colab)

Notebook de la **Semana 17 (jueves)** de la serie *IA sin humo*.

Muestra cómo el **seasonal naive** (predecir = el mismo día de la semana pasada), sin entrenar nada, le gana o empata a un modelo de ML elaborado en una serie estacional. La lección: hay que vencer al baseline antes de festejar.

Validado localmente: MAE naive 2.33, seasonal naive 1.19, ML (GBR+lags) 1.28 → el baseline estacional gana.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `baseline_vs_ml.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `baseline_vs_ml.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
