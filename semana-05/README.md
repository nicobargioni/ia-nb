# Backtesting honesto: walk-forward vs CV ingenuo (Colab)

Notebook de la **Semana 5 (jueves)** de la serie *IA sin humo*.

Muestra, sobre una serie de tiempo, por qué validar con K-fold **shuffle** infla el resultado (leakage temporal: el modelo entrena con el futuro) y cómo **walk-forward** (TimeSeriesSplit) da la métrica honesta. Incluye la visualización de los folds que explica la causa.

Validado localmente: CV ingenuo R² ≈ 0.91 vs walk-forward R² ≈ 0.60 (optimismo de fantasía ≈ 0.32).

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`, preinstalados):
- `backtesting_walkforward.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `backtesting_walkforward.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
