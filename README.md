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
| S5 | **Backtesting honesto: walk-forward vs CV ingenuo** (leakage temporal medido) | Series temporales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-05/backtesting_walkforward.ipynb) |
| S6 | **Feature importance honesta: impurity vs permutation** (la trampa de las correlacionadas) | Interpretabilidad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-06/feature_importance.ipynb) |
| S7 | **Reranking medido: cuánto sube el nDCG** (retrieve generoso + rerank top-k) | Retrieval/RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-07/reranking_ndcg.ipynb) |
| S8 | **Calibración: que el 0.9 signifique 0.9** (Brier, ECE, Platt vs isotónica) | Incertidumbre | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-08/calibracion.ipynb) |
| S9 | **Propensity score matching: causalidad sin experimento** (datos observacionales) | Causalidad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-09/propensity_matching.ipynb) |
| S10 | **Costo de un LLM en producción: prompt vs RAG vs fine-tune** (break-even) | LLMs en prod | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-10/costo_llm.ipynb) |
| S11 | **Forecast con intervalos (quantile regression)** (predecir el rango, no el punto) | Series temporales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-11/forecast_intervalos.ipynb) |
| S12 | **Counterfactuals: qué cambiar para dar vuelta la decisión** (explicaciones accionables) | Interpretabilidad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-12/counterfactuals.ipynb) |
| S13 | **Evaluar un RAG: golden set y métricas de retrieval** (recall@k, hit@k, MRR) | Retrieval/RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-13/rag_eval.ipynb) |
| S14 | **La curva de sesgo-varianza** (por qué el modelo más complejo no gana) | Incertidumbre | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-14/sesgo_varianza.ipynb) |
| S15 | **Double Machine Learning** (ML para confounders sin sesgar la causalidad) | Causalidad | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-15/double_ml.ipynb) |
| S16 | **Semantic caching (y su trampa)** (ahorro de llamadas LLM vs respuestas equivocadas) | LLMs en prod | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-16/semantic_cache.ipynb) |
| S17 | **Forecast: baseline vs ML** (el seasonal naive que cuesta ganar) | Series temporales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-17/baseline_vs_ml.ipynb) |
| S18 | **Monitor de drift sobre un stream** (PSI: detectar el cambio antes de que duela) | MLOps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-18/drift_monitor.ipynb) |
| S19 | **Chunking medido** (cómo cortás decide cuánto recupera tu RAG) | Retrieval/RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicobargioni/ia-nb/blob/main/semana-19/chunking.ipynb) |

*(se van sumando semana a semana)*

## Cómo usar
Click en el badge **Open in Colab** → `Entorno de ejecución → Ejecutar todo`. La primera celda instala lo necesario.

## Reproducible
Cada notebook se genera con su `build_notebook.py`:
```bash
python3 build_notebook.py
```

🇦🇷 Hecho para entender, no solo para usar.
