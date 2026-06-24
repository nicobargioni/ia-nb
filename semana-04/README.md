# Constrained decoding: JSON válido por construcción (Colab)

Notebook de la **Semana 4 (jueves)** de la serie *IA sin humo*.

Simula el mecanismo de **structured output / constrained decoding**: cómo garantizar JSON válido enmascarando, en cada paso de generación, los tokens que romperían el formato. Sin API ni modelo pesado — todo con numpy + json, para ver el mecanismo real.

Validado localmente: salidas válidas **sin restricción ≈ 9.9%** → **con restricción = 100%**.

## Correr
Gratis en Colab, sin API key (solo `numpy`, `matplotlib`, `json`):
- `structured_output.ipynb` → Abrir en Colab → Ejecutar todo.

> En producción este mecanismo lo dan Outlines, las gramáticas de llama.cpp, o el structured output de las APIs (esos requieren key). El notebook lo simula para enseñar el cómo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `structured_output.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
