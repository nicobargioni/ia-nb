# Búsqueda híbrida: dense vs BM25 (Colab)

Notebook de la **Semana 1 (jueves)** de la serie *IA sin humo*.

Demuestra, sobre un mini-catálogo de productos con códigos de modelo, por qué:
- la **búsqueda densa** (embeddings) gana con el significado pero falla con códigos exactos,
- **BM25** (léxico) clava lo exacto pero se pierde con la intención,
- la **búsqueda híbrida** (fusión RRF de ambos rankings) gana en los dos mundos.

## Correr
Gratis en Colab, sin API key (usa `sentence-transformers` multilingüe):
- `retrieval_hibrido.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
El notebook se genera de forma reproducible:
```bash
python3 build_notebook.py   # -> retrieval_hibrido.ipynb
```

## Archivos
- `retrieval_hibrido.ipynb` — el notebook (subir a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
