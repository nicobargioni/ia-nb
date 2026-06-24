# Counterfactuals: qué cambiar para dar vuelta la decisión (Colab)

Notebook de la **Semana 12 (jueves)** de la serie *IA sin humo*.

Muestra explicaciones **contrafactuales**: dado un solicitante rechazado por un modelo de crédito, encuentra el cambio mínimo y accionable (en features que se pueden cambiar) que da vuelta la decisión. Incluye la visualización de la frontera de decisión y los puntos contrafactuales.

Validado localmente: solicitante rechazado (p≈0.18) → "se aprobaría con +ingreso o −deuda", respetando lo inmutable (antigüedad).

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `counterfactuals.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `counterfactuals.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
