# Bootstrap: intervalos de confianza para cualquier estadístico (Colab)

Notebook de la **Semana 20 (jueves)** de la serie *IA sin humo*.

Muestra el **bootstrap**: obtener un intervalo de confianza para CUALQUIER estadístico (mediana, p90, correlación, KPIs) remuestreando, sin fórmula y sin asumir distribución. Caso: la mediana de datos sesgados, con chequeo de cobertura.

Validado localmente: cobertura del IC bootstrap ≈95% (objetivo 95%); la fórmula normal-ingenua se descalibra (~98.5%).

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `bootstrap.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `bootstrap.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
