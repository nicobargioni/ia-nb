"""Genera agente_error.ipynb — Semana 22 (jueves) de 'IA sin humo'.
El error compuesto de los agentes (éxito = p^n) y cómo la verificación por paso
recupera la confiabilidad. Simulación con numpy (free)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🤖 Agentes: el error compuesto (y cómo la verificación lo arregla)""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🤖 ⛓️ 🧮</div>
    <h1 style="margin:.1em 0 0;font-size:1.95em;font-weight:800;text-transform:uppercase;line-height:1.05;letter-spacing:-.02em">
       El <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">error compuesto</span> de los agentes</h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 22 · LLMs en producción</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       95% de acierto por paso suena genial. En una cadena de 10 pasos se vuelve 60%.
       Lo medimos, y vemos cómo la verificación por paso lo rescata.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
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

md(r"""## 1 · La matemática despiadada del error compuesto""")

code(r"""display(intro("⛓️", "1 · Éxito de la cadena = p elevado a n", [
 "<b>De qué se trata.</b> Un agente encadena pasos (llamar una herramienta, decidir el siguiente movimiento, etc.). Si cada paso acierta con probabilidad p y son independientes, la cadena ENTERA acierta p^n. La intuición falla acá: 95% por paso suena altísimo, pero elevado a 10 pasos se desploma.",
 "<b>Qué vas a ver.</b> El éxito de la cadena vs la cantidad de pasos, para distintas confiabilidades por paso. Aun con 95% por paso, una cadena larga termina siendo una moneda al aire o peor.",
 "🧮 <b>Dónde mirar.</b> La caída exponencial. Cada paso que agregás multiplica el riesgo de que algo se rompa en el camino.",
]))
ns=np.arange(1,21)
plt.figure()
for p,c in [(0.99,BEAM),(0.95,CELESTE),(0.90,GOLD),(0.80,CORAL)]:
    plt.plot(ns,p**ns,"o-",color=c,ms=3,label=f"{p:.0%} por paso")
plt.axhline(0.5,color=MIST,ls="--",lw=1)
plt.xlabel("pasos en la cadena"); plt.ylabel("éxito de la cadena completa"); plt.ylim(0,1.02)
plt.title("Error compuesto: el éxito por paso se evapora en la cadena"); plt.legend(); plt.tight_layout(); plt.show()
for n in [5,10,20]:
    print(f"pasos={n:2d}:  95%/paso -> {0.95**n:.0%}   90%/paso -> {0.90**n:.0%}")""")

md(r"""📝 **Lectura.** La caída es brutal: con 95% de acierto por paso, una cadena de 10 pasos termina en ~60%, y de 20 en ~36%. Con 90% por paso, 10 pasos son ~35%. Por eso muchos "agentes autónomos" impresionan en la demo (cadenas cortas, casos felices) y fallan en producción (cadenas largas, casos reales): el error no se suma, se MULTIPLICA. Y un error temprano suele descarrilar todo lo que sigue.""")

md(r"""## 2 · La verificación por paso rescata la confiabilidad""")

code(r"""display(intro("✅", "2 · Verificar cada paso cambia todo", [
 "<b>De qué se trata.</b> Si en cada paso verificás la salida (con código, un check, otro modelo) y reintentás cuando detectás un error, subís la confiabilidad efectiva de cada paso — y como la cadena es p^n, esa mejora por paso se amplifica en toda la cadena.",
 "<b>Qué vas a ver.</b> La misma cadena (90% por paso) con verificación que detecta errores con distinta efectividad. Una verificación decente convierte una cadena de 10 pasos de 35% a 75-86% de éxito.",
 "🧮 <b>Dónde mirar.</b> No bajamos la cantidad de pasos ni cambiamos el modelo: solo agregamos un control por paso. El efecto en la cadena completa es enorme.",
]))
p=0.90
def chain(n,v,retries=1):
    pe=p
    for _ in range(retries): pe=pe+(1-pe)*v*p   # detecta error (v) y reintenta
    return pe**n
plt.figure()
for v,c,lab in [(0,CORAL,"sin verificación"),(0.8,CELESTE,"verif. detecta 80%"),(0.95,BEAM,"verif. detecta 95%")]:
    plt.plot(ns,[chain(n,v) for n in ns],"o-",color=c,ms=3,label=lab)
plt.xlabel("pasos en la cadena"); plt.ylabel("éxito de la cadena"); plt.ylim(0,1.02)
plt.title("Cadena de 90%/paso: la verificación recupera la confiabilidad"); plt.legend(); plt.tight_layout(); plt.show()
for n in [10,20]:
    print(f"pasos={n}:  sin verif {chain(n,0):.0%}  ->  con verif(0.8) {chain(n,0.8):.0%}  ->  con verif(0.95) {chain(n,0.95):.0%}")""")

md(r"""📝 **Lectura.** Agregar verificación por paso (detectar el error y reintentar) sube la confiabilidad efectiva de cada paso, y como la cadena multiplica, esa mejora se amplifica: una cadena de 10 pasos pasa de 35% a 75-86%. No tocamos el modelo ni acortamos la tarea — solo pusimos un control en cada eslabón.

Esto es por qué los sistemas agénticos serios no son "dejá que el LLM haga todo solo": son cadenas con verificación, checkpoints y recuperación en cada paso. La autonomía sin control se come a sí misma por el error compuesto.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> La autonomía no es gratis: se paga en confiabilidad. Una cadena larga de pasos 'casi buenos' es una bomba de tiempo por el error compuesto. La demo con 3 pasos engaña; producción tiene 10.",
 "<b>Qué hacer.</b> (1) Menos pasos: cada uno que sacás sube la confiabilidad total. (2) Verificá cada paso con código, no con fe. (3) Checkpoints y recuperación para que un error no descarrile todo. (4) Para tareas predecibles, un workflow fijo con LLMs en pasos puntuales le gana a un agente libre.",
 "<b>El criterio.</b> Antes de darle 10 pasos a un agente, hacé la cuenta: p^n. Si no te cierra, el problema no es el modelo, es la arquitectura. A veces lo más inteligente no es el agente más autónomo, es el flujo más corto y verificado.",
]))
print(f"cadena 10 pasos @90%/paso: sin verif {0.90**10:.0%} -> con verif(0.95) {chain(10,0.95):.0%}")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("agente_error.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> agente_error.ipynb ({len(cells)} celdas)")
