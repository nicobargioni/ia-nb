# Agentes: el error compuesto y la verificación (Colab)

Notebook de la **Semana 22 (jueves)** de la serie *IA sin humo*.

Muestra la matemática del **error compuesto** en agentes (éxito de la cadena = p^n) y cómo la **verificación por paso** recupera la confiabilidad. Simulación con numpy.

Validado localmente: cadena de 10 pasos al 90%/paso → 35% sin verificación → 75-86% con verificación. A 20 pasos: 12% → 57-75%.

## Correr
Gratis en Colab, sin API key (`numpy` + `matplotlib`):
- `agente_error.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `agente_error.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
