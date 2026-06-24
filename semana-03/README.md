# Diferencias-en-diferencias: causalidad sin A/B (Colab)

Notebook de la **Semana 3 (jueves)** de la serie *IA sin humo*.

Muestra cómo estimar un **efecto causal** cuando no podés randomizar (no hay A/B): el estimador **diff-in-diff** cancela la tendencia temporal y la diferencia de base entre grupos. Incluye la forma de regresión con término de interacción.

Validado localmente: efecto real simulado = 5.0, estimación DiD ≈ 4.70 (coincide con el coeficiente de la regresión).

## Correr
Gratis en Colab, sin API key (solo `numpy` + `matplotlib`):
- `diff_in_diff.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `diff_in_diff.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
