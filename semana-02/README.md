# Conformal prediction: incertidumbre con garantía (Colab)

Notebook de la **Semana 2 (jueves)** de la serie *IA sin humo*.

Muestra **split conformal prediction** para clasificación: cómo construir sets de predicción con **cobertura garantizada** (ej. 90%) sobre cualquier modelo, sin asumir ninguna distribución.

Validado localmente: cobertura empírica ≈ 0.90, tamaño de set promedio ≈ 1.06 clases.

## Correr
Gratis en Colab, sin API key (solo `scikit-learn` + `numpy`, preinstalados):
- `conformal_prediction.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `conformal_prediction.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
