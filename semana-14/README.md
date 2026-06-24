# La curva de sesgo-varianza (Colab)

Notebook de la **Semana 14 (jueves)** de la serie *IA sin humo*.

Muestra por qué el modelo más complejo no es el mejor: el error de entrenamiento baja siempre, pero el de test hace una **U** (sesgo-varianza). Incluye los tres regímenes (underfit/justo/overfit) y la curva de complejidad. Menciona el double descent como matiz moderno.

Validado localmente: error de test mínimo en grado ~9 (sobre datos generados), con train siempre decreciente.

## Correr
Gratis en Colab, sin API key (`scikit-learn` + `numpy`):
- `sesgo_varianza.ipynb` → Abrir en Colab → Ejecutar todo.

## Regenerar
```bash
python3 build_notebook.py
```

## Archivos
- `sesgo_varianza.ipynb` — el notebook (subido a GitHub)
- `build_notebook.py` — generador
- `post-linkedin.md` — texto del post que acompaña
