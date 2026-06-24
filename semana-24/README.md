# El test del baseline (Colab) — notebook de cierre

Notebook **capstone** de la **Semana 24 (jueves)** de la serie *IA sin humo*.

Resume la tesis de las 24 semanas: ningún modelo vale por su número absoluto, sino por cuánto le gana al baseline tonto, medido con la métrica correcta. Dos casos:

1. **Forecast** — seasonal-naive vs modelo lineal+estacional. El modelo gana (lift ~23%): complejidad justificada.
2. **Clasificación desbalanceada (9% positivos)** — la logística da 91% de accuracy, igual que el baseline de mayoría (lift 0). Pero AUC 0.50 → 0.72: el modelo sí separa. Lección: elegí la métrica que no te miente.

Validado localmente: forecast MAE naive 2.80 → modelo 2.17 (lift 23%); clasificación accuracy 0.91 == 0.91, AUC 0.72.

## Correr
Gratis en Colab, sin API key (`numpy` + `scikit-learn` + `matplotlib`):
- `baseline_test.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `baseline_test.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
