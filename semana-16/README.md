# Semantic caching: ahorrar llamadas al LLM (y su trampa) (Colab)

Notebook de la **Semana 16 (jueves)** de la serie *IA sin humo*.

Muestra cómo el **semantic caching** (cachear respuestas por significado, no por texto exacto) ahorra llamadas al LLM, y el **trade-off del umbral**: muy bajo → mucho ahorro pero respuestas equivocadas; muy alto → seguro pero poco ahorro. Simulación con numpy.

Validado localmente: umbral 0.6 → 93% ahorro pero 37% hits equivocados; umbral 0.9 → 68% ahorro, 0% equivocados.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `semantic_cache.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `semantic_cache.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
