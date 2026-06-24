# Double Machine Learning (Colab)

Notebook de la **Semana 15 (jueves)** de la serie *IA sin humo*.

Implementa **Double / Debiased ML**: estimar un efecto causal usando ML para controlar confounders no lineales, sin que el ML introduzca sesgo. Residualización de Y y T contra los confounders + cross-fitting.

Validado localmente: efecto real θ=2.0; control lineal ingenuo ≈1.61 (sesgado); Double ML ≈1.98.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `double_ml.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `double_ml.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
