# Reranking: cuánto sube el nDCG (Colab)

Notebook de la **Semana 7 (jueves)** de la serie *IA sin humo*.

Mide el efecto del **reranking** (reordenar el top-k del bi-encoder con un modelo más preciso) sobre el **nDCG@10**. Simulación honesta con numpy (sin descargar modelos): un rankeador rápido y ruidoso (bi-encoder) vs reordenar sus top-30 con uno preciso (cross-encoder).

Validado localmente: nDCG@10 **0.40 (solo bi-encoder) → 0.78 (con reranking)**.

> En producción el rankeador preciso es un cross-encoder real (sentence-transformers); la dinámica es idéntica.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `reranking_ndcg.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `reranking_ndcg.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
