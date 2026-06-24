# Monitor de drift sobre un stream (Colab)

Notebook de la **Semana 18 (jueves)** de la serie *IA sin humo*.

Muestra cómo monitorear **data drift** en producción con el **PSI** (Population Stability Index): sobre un stream donde la distribución cambia a mitad de camino, el monitor dispara la alarma exactamente cuando empieza el drift.

Validado localmente: PSI ≈0.02 antes del shift, ≈0.63 después; cruza el umbral (0.2) en el paso del cambio real.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `drift_monitor.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `drift_monitor.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
