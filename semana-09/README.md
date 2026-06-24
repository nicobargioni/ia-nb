# Propensity score matching: causalidad sin experimento (Colab)

Notebook de la **Semana 9 (jueves)** de la serie *IA sin humo*.

Muestra cómo estimar un efecto causal con datos observacionales (sin A/B) emparejando tratados y controles por su **propensity score** (probabilidad de haber sido tratado). Incluye el chequeo de solapamiento y el límite clave (solo corrige confounders observados).

Validado localmente: efecto real = 3.0, comparación ingenua = 5.2 (sesgada por el confounder), propensity matching = 3.1.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `propensity_matching.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `propensity_matching.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
