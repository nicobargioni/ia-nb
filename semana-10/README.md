# Costo de un LLM en producción: prompt vs RAG vs fine-tune (Colab)

Notebook de la **Semana 10 (jueves)** de la serie *IA sin humo*.

Modela la economía de un LLM en producción: costo por llamada y costo acumulado de **prompt-only vs RAG vs fine-tune**, y dónde está el **break-even** (cuándo amortiza el costo fijo del fine-tune). Simulación con numpy y supuestos editables.

Validado localmente: con los supuestos de ejemplo, costo/llamada $0.00042 / $0.000345 / $0.000157; break-even fine-tune vs prompt-only ≈ 30.000 llamadas.

> Cambiá los tokens y precios por los de tu proveedor: importa la forma de la decisión, no los valores exactos.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `costo_llm.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `costo_llm.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
