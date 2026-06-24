"""Genera costo_llm.ipynb — Semana 10 (jueves) de 'IA sin humo'.
Economía de un LLM en producción: costo/latencia de prompt-only vs RAG vs fine-tune,
y dónde está el break-even. Simulación con numpy (free), supuestos editables."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 💸 Costo de un LLM en producción: prompt vs RAG vs fine-tune""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">💸 ⚖️ 📉</div>
    <h1 style="margin:.1em 0 0;font-size:2.05em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       El <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">costo real</span> de un LLM</h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 10 · LLMs en producción</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tu PoC salió barata. Multiplicada por el volumen real, la factura despierta.
       Modelamos el costo de prompt-only vs RAG vs fine-tune y el break-even.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación

> Simulación con **supuestos de ejemplo** (cantidades de tokens y precios). Cambiá los números por los de tu proveedor y tu caso: lo que importa es la *forma* de la decisión, no los valores exactos.""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; GOLD="#e8b86a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(8,4.6),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
  "savefig.facecolor":NIGHT,"axes.edgecolor":LINE,"axes.labelcolor":INK,"text.color":INK,
  "axes.titlecolor":INK,"xtick.color":MIST,"ytick.color":MIST,"axes.grid":True,"grid.color":LINE,
  "grid.alpha":.5,"font.size":11,"legend.framealpha":0})
def intro(emoji,titulo,parrafos):
    ps="".join(f'<p style="margin:0 0 11px;color:#cfe3ef;font-size:.97em;line-height:1.62">{p}</p>' for p in parrafos)
    return HTML(f'''<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;
      background:linear-gradient(135deg,#0a1b27,#102b3c 55%,#15384b);border:1px solid rgba(79,184,232,.22);
      border-left:5px solid #4fb8e8;border-radius:12px;padding:20px 26px;margin:6px 0;color:#eef7fb;box-shadow:0 8px 30px rgba(0,0,0,.35)">
      <div style="font-size:1.3em;font-weight:800;text-transform:uppercase;margin-bottom:12px">{emoji}&nbsp;{titulo}</div>{ps}</div>''')
print("Listo.")""")

md(r"""## 1 · Las tres estrategias y su costo por llamada""")

code(r"""display(intro("🧮", "1 · Costo por llamada", [
 "<b>De qué se trata.</b> Tres formas de hacer que el LLM haga lo que querés, con perfiles de costo distintos. <b>Prompt-only:</b> metés instrucciones largas y ejemplos few-shot en CADA llamada (muchos tokens de entrada). <b>RAG:</b> instrucciones cortas + documentos recuperados (tokens medios). <b>Fine-tune:</b> el comportamiento está 'horneado', el prompt es cortísimo (pocos tokens), pero hay un costo fijo de entrenamiento por única vez.",
 "<b>Qué vas a ver.</b> El costo por llamada: fine-tune es el más barato por request (prompt corto), prompt-only el más caro. Pero fine-tune arranca con una deuda (el costo de entrenar).",
 "🧮 <b>Dónde mirar.</b> El costo por llamada NO cuenta toda la historia: falta amortizar el costo fijo del fine-tune. Eso lo vemos en el paso 2.",
]))
IN_PRICE, OUT_PRICE = 0.15e-6, 0.60e-6   # USD por token (ejemplo)
OUT = 200                                 # tokens de salida por llamada
TOK_IN = {"prompt-only":2000, "RAG":1500, "fine-tune":250}   # tokens de entrada por estrategia
UPFRONT = {"prompt-only":0.0, "RAG":0.0, "fine-tune":8.0}     # costo fijo (entrenar)
per_call = {k: TOK_IN[k]*IN_PRICE + OUT*OUT_PRICE for k in TOK_IN}
for k in TOK_IN: print(f"{k:11s} entrada={TOK_IN[k]:5d}tok  costo/llamada=${per_call[k]:.6f}  fijo=${UPFRONT[k]:.0f}")

fig,ax=plt.subplots(figsize=(6.5,4))
ax.bar(list(per_call), [v*1000 for v in per_call.values()], color=[CORAL,GOLD,BEAM])
ax.set_ylabel("costo por 1000 llamadas (USD)"); ax.set_title("Costo por llamada (sin contar el fijo)")
for i,v in enumerate(per_call.values()): ax.text(i,v*1000+.01,f"${v*1000:.2f}",ha="center",color=INK,fontweight="bold")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Por llamada, fine-tune es el más barato (prompt cortísimo) y prompt-only el más caro (arrastra los ejemplos few-shot en cada request). RAG queda en el medio. Pero quedarse con este gráfico lleva a la conclusión apurada "fine-tune es lo más barato, hagámoslo". Falta meter en la cuenta el **costo fijo** de entrenar — y eso cambia todo según el volumen.""")

md(r"""## 2 · El costo acumulado y el break-even""")

code(r"""display(intro("📈", "2 · ¿A partir de cuántas llamadas conviene?", [
 "<b>De qué se trata.</b> Sumamos el costo a lo largo de N llamadas: prompt-only y RAG crecen lineal desde cero; fine-tune arranca con su costo fijo pero crece más lento (es más barato por llamada). En algún punto las líneas se cruzan: el break-even.",
 "<b>Qué vas a ver.</b> Con pocos volúmenes, fine-tune es el MÁS caro (no amortizó el entrenamiento). Recién a partir de decenas de miles de llamadas empieza a convenir por costo. Si tu volumen es bajo, fine-tune es tirar plata.",
 "🧮 <b>Dónde mirar.</b> El cruce de las líneas. A la izquierda de ese punto, fine-tune pierde; a la derecha, gana (solo en costo).",
]))
N = np.arange(0, 80000, 100)
cost = {k: UPFRONT[k] + N*per_call[k] for k in per_call}
be = UPFRONT["fine-tune"]/(per_call["prompt-only"]-per_call["fine-tune"])
plt.figure()
for k,c in zip(cost,[CORAL,GOLD,BEAM]): plt.plot(N,cost[k],color=c,lw=2,label=k)
plt.axvline(be,color=MIST,ls="--",lw=1); plt.text(be*1.02,2,f"break-even ≈ {be:,.0f} llamadas",color=CELESTE)
plt.xlabel("número de llamadas"); plt.ylabel("costo acumulado (USD)")
plt.title("Costo acumulado: fine-tune amortiza solo a alto volumen"); plt.legend(); plt.tight_layout(); plt.show()
print(f"Break-even fine-tune vs prompt-only: ~{be:,.0f} llamadas")""")

md(r"""📝 **Lectura.** El cruce está en decenas de miles de llamadas. Por debajo de ese volumen, fine-tune es el **más caro** de los tres, porque nunca amortizó el costo de entrenar. Mucha gente fine-tunea para "ahorrar" con volúmenes donde en realidad pierde plata. El costo por llamada engaña: lo que decide es el costo *total* al volumen *real* que vas a tener.""")

md(r"""## 3 · Cierre (el costo no es toda la historia)""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> El costo de un LLM en producción es una variable de diseño, no un detalle. Y el costo por llamada miente: hay que amortizar lo fijo y proyectar al volumen real (usuarios × consultas × llamadas por consulta).",
 "<b>El asterisco grande.</b> Aun cuando fine-tune gane por costo a alto volumen, NO suele ser la respuesta: no incorpora conocimiento actualizable (eso es RAG), congela el modelo (cuando sale uno mejor, reentrenás) y necesita un dataset grande. La decisión no es solo $/llamada.",
 "<b>El criterio.</b> Empezá por lo barato y reversible (prompt → RAG); subí a fine-tune solo si el volumen Y la necesidad lo justifican. Y bajá costo con caché y prompts más cortos antes de cambiar de estrategia.",
]))
print("Costo por llamada:", {k: round(v,6) for k,v in per_call.items()})
print(f"Break-even fine-tune vs prompt-only: ~{be:,.0f} llamadas")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("costo_llm.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> costo_llm.ipynb ({len(cells)} celdas)")
