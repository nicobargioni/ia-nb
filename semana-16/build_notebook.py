"""Genera semantic_cache.ipynb — Semana 16 (jueves) de 'IA sin humo'.
Semantic caching: cachear respuestas por significado para ahorrar llamadas al LLM,
y el trade-off del umbral (ahorro vs respuestas equivocadas). Free/runnable (numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 💾 Semantic caching: ahorrar llamadas al LLM (y su trampa)""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">💾 ⚡ ⚠️</div>
    <h1 style="margin:.1em 0 0;font-size:2.0em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Semantic <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">caching</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 16 · LLMs en producción</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tus usuarios preguntan lo mismo de mil formas. Cachear por significado ahorra
       muchas llamadas — pero el umbral mal puesto sirve la respuesta equivocada.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación

> Simulamos un espacio de consultas con numpy (intenciones repetidas como vectores cercanos) para ver la dinámica del caché y el trade-off del umbral, sin depender de embeddings reales. En producción, los vectores salen de tu modelo de embeddings.""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(5)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; GOLD="#e8b86a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(8,4.4),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · El flujo de consultas y el caché""")

code(r"""display(intro("🔁", "1 · Cómo funciona el caché semántico", [
 "<b>De qué se trata.</b> Simulamos un stream de 2000 consultas. El 70% son paráfrasis de alguna de 40 'intenciones' (la misma pregunta escrita distinto → vectores cercanos); el 30% son consultas únicas. El caché funciona así: para cada consulta nueva, busca en el caché la más parecida (coseno); si supera un UMBRAL, devuelve la respuesta guardada (HIT, sin llamar al LLM); si no, llama al LLM y guarda (MISS).",
 "<b>Qué vas a ver.</b> Con un umbral razonable, una gran fracción de consultas se resuelve desde el caché → ahorro de llamadas. Pero medimos también algo crítico: los HITS EQUIVOCADOS (servir una respuesta cacheada a una consulta que en realidad era otra cosa).",
 "🧮 <b>Dónde mirar.</b> El umbral de similitud es la perilla. Y como todo umbral, tiene un trade-off — el paso 2 lo hace explícito.",
]))
dim=12; K=40
centers=rng.normal(0,1,(K,dim)); centers/=np.linalg.norm(centers,axis=1,keepdims=True)
N=2000; qs=[]; intent=[]
for _ in range(N):
    if rng.random()<0.7:
        k=int(rng.integers(K)); v=centers[k]+rng.normal(0,0.06,dim); intent.append(k)
    else:
        v=rng.normal(0,1,dim); intent.append(-1)
    qs.append(v/np.linalg.norm(v))
qs=np.array(qs); intent=np.array(intent)
def sim_cache(thr):
    cache=[]; ci=[]; hits=0; wrong=0
    for i in range(N):
        if cache:
            sims=np.array(cache)@qs[i]; j=int(np.argmax(sims))
            if sims[j]>=thr:
                hits+=1
                if ci[j]!=intent[i] or intent[i]==-1: wrong+=1
                continue
        cache.append(qs[i]); ci.append(intent[i])
    return hits/N, wrong/max(hits,1)
hr,wr=sim_cache(0.9)
print(f"Con umbral 0.90: ahorro de llamadas = {hr:.0%}, hits equivocados = {wr:.0%}")""")

md(r"""📝 **Lectura.** Con un umbral bien elegido, el caché evita una buena parte de las llamadas al LLM (ahorro directo de costo y latencia), sin servir respuestas equivocadas. Eso es plata y velocidad gratis en dominios con consultas repetitivas (soporte, FAQ, asistentes internos). Pero todo depende del umbral — y elegirlo mal tiene un costo silencioso, que vemos ahora.""")

md(r"""## 2 · El trade-off del umbral""")

code(r"""display(intro("⚖️", "2 · Ahorro vs respuestas equivocadas", [
 "<b>De qué se trata.</b> Barremos el umbral de similitud y medimos las DOS cosas a la vez: cuánto ahorramos (hit-rate) y qué porcentaje de esos hits fueron equivocados (servimos la respuesta de una consulta que en realidad era distinta).",
 "<b>Qué vas a ver.</b> Umbral bajo → ahorrás muchísimo PERO servís muchas respuestas equivocadas (el caché 'confunde' consultas parecidas pero distintas). Umbral alto → casi no te equivocás, pero ahorrás menos. El punto dulce está en el medio, y depende de tu tolerancia al error.",
 "🧮 <b>Dónde mirar.</b> Las dos curvas. Un caché agresivo (umbral bajo) que sirve respuestas equivocadas es peor que no tener caché: rompe la confianza del usuario para ahorrar unos centavos.",
]))
thrs=[0.6,0.7,0.8,0.85,0.9,0.95]
ahorro=[]; errados=[]
for t in thrs:
    hr,wr=sim_cache(t); ahorro.append(hr); errados.append(wr)
plt.figure()
plt.plot(thrs,ahorro,"o-",color=BEAM,label="ahorro (hit-rate)")
plt.plot(thrs,errados,"o-",color=CORAL,label="hits equivocados")
plt.xlabel("umbral de similitud"); plt.ylabel("proporción"); plt.ylim(0,1)
plt.title("El trade-off del umbral: ahorro vs respuestas equivocadas"); plt.legend(); plt.tight_layout(); plt.show()
for t,a,e in zip(thrs,ahorro,errados): print(f"  umbral {t}: ahorro {a:.0%}  ·  equivocados {e:.0%}")""")

md(r"""📝 **Lectura.** Acá está la trampa que nadie te cuenta del semantic caching. Con umbral bajo (0.6) ahorrás muchísimo… pero más de un tercio de los hits son **respuestas equivocadas**: el caché agarra una consulta parecida pero distinta y sirve la respuesta de otra pregunta. Con umbral alto (0.9+) los errores desaparecen, a costa de ahorrar menos.

El punto dulce no es universal: depende de cuánto te duele una respuesta equivocada. En un asistente de soporte crítico, preferís umbral alto y menos ahorro. La lección: el umbral NO se adivina, se mide contra tu propio tráfico.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Antes de buscar un modelo más barato, mirá cuánto pagás por responder lo MISMO una y otra vez. En dominios repetitivos, el semantic caching corta una parte grande de las llamadas — costo y latencia abajo, sin tocar el modelo.",
 "<b>Los cuidados.</b> (1) El umbral es delicado: medilo contra tu tráfico, no lo adivines. (2) Cuidado con el contexto: '¿cuánto cuesta?' depende de qué se hablaba antes; cachear sin contexto sirve respuestas equivocadas. (3) Invalidá el caché cuando la info cambie (precios, horarios).",
 "<b>El criterio.</b> La mejor optimización a veces no es un modelo mejor: es no llamar al modelo. Pero un caché mal calibrado que sirve respuestas equivocadas para ahorrar centavos es peor que no tenerlo. Medí el trade-off, no lo asumas.",
]))
print("Resumen: el umbral define ahorro y error a la vez. Elegilo con datos.")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("semantic_cache.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> semantic_cache.ipynb ({len(cells)} celdas)")
