# Evaluar tu RAG: golden set y métricas de retrieval (Colab)

Notebook de la **Semana 13 (jueves)** de la serie *IA sin humo*.

Muestra cómo armar un **golden set** y medir el retrieval de un RAG: recall@k, hit@k y MRR a distintos valores de k. Y cómo esas métricas **diagnostican dónde está el problema** (recuperación vs ordenamiento).

Validado localmente: recall@k 0.20 (k=1) → 0.78 (k=20); hit@k hasta 0.93.

> Acá se mide el retrieval. La parte de generación (faithfulness, answer relevance) se mide con un LLM-judge (ej. RAGAS) y requiere API key — se menciona en el cierre.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `rag_eval.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `rag_eval.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
