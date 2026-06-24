# 🔦 ia-nb — Notebooks de la serie *IA sin humo*

Cuaderno por cuaderno de la serie **IA sin humo**: IA / Data Science **sin hype**, entendiendo los fundamentos. Cada notebook corre gratis en Colab (sin API key) y acompaña una publicación.

> Tesis de la serie: tirarle un LLM encima no reemplaza el criterio que dan las herramientas básicas. Acá lo mostramos con código que podés correr.

## Notebooks

| # | Tema | Pilar | Abrir |
|---|------|-------|-------|
| S1 | **Búsqueda híbrida: por qué los vectores solos pierden** (dense vs BM25, fusión RRF) | Retrieval/RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-01/retrieval_hibrido.ipynb) |
| S2 | **Conformal prediction: incertidumbre con garantía** (sets con cobertura garantizada, distribution-free) | Incertidumbre | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-02/conformal_prediction.ipynb) |
| S3 | **Diferencias-en-diferencias: causalidad sin A/B** (efecto causal sin experimento, con regresión) | Causalidad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-03/diff_in_diff.ipynb) |
| S4 | **Constrained decoding: JSON válido por construcción** (structured output, mecanismo de masking) | LLMs en prod | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-04/structured_output.ipynb) |

*(se van sumando semana a semana)*

## Cómo usar
Click en el badge **Open in Colab** → `Entorno de ejecución → Ejecutar todo`. La primera celda instala lo necesario.

## Reproducible
Cada notebook se genera con su `build_notebook.py`:
```bash
python3 build_notebook.py
```

🇦🇷 Hecho para entender, no solo para usar.
