# Múltiples comparaciones / FDR (Colab)

Notebook de la **Semana 21 (jueves)** de la serie *IA sin humo*.

Muestra cómo probar muchas hipótesis infla los falsos positivos, y cómo **Benjamini-Hochberg** controla la tasa de falsos descubrimientos (FDR) — con el trade-off honesto en poder.

Validado localmente (promedio de 2000 simulaciones): FDP ingenuo ~37% → BH ~5% (objetivo); poder ingenuo ~77% → BH ~30% (el costo de corregir).

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `fdr_multiple.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `fdr_multiple.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
