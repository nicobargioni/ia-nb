# Feature importance honesta: impurity vs permutation (Colab)

Notebook de la **Semana 6 (jueves)** de la serie *IA sin humo*.

Muestra por qué la importancia por defecto (impurity) miente —infla features de alta cardinalidad y sin señal— y cómo **permutation importance out-of-sample** es la honesta. Incluye la **trampa de las features correlacionadas** (romper una sola subestima a ambas).

Validado localmente: el `id_alta_card` sin señal recibe ~0.08 en impurity vs **~0.00** en permutation; romper señal+copia juntas tira el accuracy ~0.84 → ~0.48.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `feature_importance.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `feature_importance.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
