# Intervalos de predicción con quantile regression (Colab)

Notebook de la **Semana 11 (jueves)** de la serie *IA sin humo*.

Muestra cómo predecir un **rango** (no solo el punto) con quantile regression: entrenar modelos para p10/p50/p90 y obtener un intervalo del 80% que **se ensancha donde hay más incertidumbre** (datos heterocedásticos).

Validado localmente: cobertura empírica del intervalo p10–p90 ≈ 77% (objetivo 80%); ancho ~1.6 donde hay poca incertidumbre vs ~6 donde hay mucha.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `forecast_intervalos.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `forecast_intervalos.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
