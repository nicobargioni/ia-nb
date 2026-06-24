# Chunking medido (Colab)

Notebook de la **Semana 19 (jueves)** de la serie *IA sin humo*.

Mide cómo la estrategia de **chunking** decide cuánta "respuesta" sobrevive intacta para que un RAG la recupere: corte fijo sin overlap vs con overlap vs consciente de la estructura. Simulación con numpy.

Validado localmente: corte fijo sin overlap deja ~80% de respuestas intactas; con overlap sube a ~100%; consciente de estructura ~98%.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `chunking.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `chunking.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
