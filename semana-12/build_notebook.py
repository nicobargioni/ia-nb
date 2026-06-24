"""Genera counterfactuals.ipynb — Semana 12 (jueves) de 'IA sin humo'.
Explicaciones contrafactuales: el cambio mínimo y accionable que da vuelta una
decisión del modelo. Free/runnable (sklearn+numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# ↪️ Counterfactuals: qué cambiar para dar vuelta la decisión""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">↪️ 🔁 💡</div>
    <h1 style="margin:.1em 0 0;font-size:2.05em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Explicaciones <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">contrafactuales</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 12 · Interpretabilidad</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       'Tu crédito se rechazó por el ingreso' no le sirve a nadie. 'Se habría aprobado
       con $X más' sí. Buscamos el cambio mínimo y accionable que da vuelta la decisión.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from IPython.display import display, HTML
rng = np.random.default_rng(7)
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

md(r"""## 1 · Un modelo de aprobación de crédito y un rechazo""")

code(r"""display(intro("🏦", "1 · El modelo y el caso", [
 "<b>De qué se trata.</b> Entrenamos un modelo que aprueba o rechaza créditos según ingreso, deuda y antigüedad laboral. Tomamos un solicitante REchazado y nos ponemos en su lugar: no quiere saber 'qué pesó', quiere saber 'qué hago para que me aprueben'.",
 "<b>Qué vas a ver.</b> El perfil del rechazado y su probabilidad de aprobación (baja). Una explicación de atribución (tipo SHAP) le diría 'tu deuda restó X'. Útil, pero no le dice qué hacer. Vamos por otra cosa.",
 "🧮 <b>Dónde mirar.</b> Distinguimos features ACCIONABLES (ingreso, deuda: se pueden cambiar) de INMUTABLES (no le vamos a pedir que cambie cosas que no puede). El contrafactual solo toca lo accionable.",
]))
n=4000
ingreso=rng.uniform(20,120,n); deuda=rng.uniform(0,60,n); antig=rng.uniform(0,15,n)
score=0.06*ingreso - 0.09*deuda + 0.15*antig - 3 + rng.normal(0,0.4,n)
y=(score>0).astype(int)
X=np.column_stack([ingreso,deuda,antig]); NAMES=["ingreso (k)","deuda (k)","antigüedad (años)"]
clf=LogisticRegression().fit(X,y)
p=clf.predict_proba(X)[:,1]
i=np.argmin(np.abs(p-0.18))           # un rechazado claro, no en el borde
x0=X[i].copy()
print("Solicitante RECHAZADO:")
for nme,v in zip(NAMES,x0): print(f"   {nme:20s} {v:.1f}")
print(f"   probabilidad de aprobación: {p[i]:.2f}  → RECHAZADO")""")

md(r"""📝 **Lectura.** Tenemos un solicitante rechazado con una probabilidad de aprobación clara pero baja. Una explicación de atribución le diría qué variable pesó más en el rechazo. Pero esa persona no puede "cambiar su atribución" — necesita una instrucción concreta. Ahí entra el contrafactual: en vez de explicar el pasado, propone el cambio mínimo hacia el futuro deseado.""")

md(r"""## 2 · Buscar el contrafactual accionable""")

code(r"""display(intro("↪️", "2 · El cambio mínimo que aprueba", [
 "<b>De qué se trata.</b> Buscamos, para cada feature accionable, el cambio MÁS CHICO que da vuelta la decisión (de rechazado a aprobado). Es la versión más simple de un contrafactual: '¿cuánto tendrías que subir el ingreso, o bajar la deuda, para que te aprueben?'.",
 "<b>Qué vas a ver.</b> Dos contrafactuales accionables y concretos. No tocamos la antigüedad (no se cambia a voluntad). El resultado es una instrucción que la persona entiende sin saber nada del modelo.",
 "🧮 <b>Dónde mirar.</b> Que la explicación sea accionable (qué hacer) y realista (un cambio alcanzable), no solo correcta.",
]))
def busca_cf(feat, paso, lim, signo):
    for d in np.arange(0, lim, paso):
        xc=x0.copy(); xc[feat]+=signo*d
        if clf.predict([xc])[0]==1: return d
    return None
d_ing = busca_cf(0,  0.5, 100, +1)   # subir ingreso
d_deu = busca_cf(1,  0.5, x0[1], -1) # bajar deuda
print("Tu crédito se habría APROBADO si:")
if d_ing is not None: print(f"   • subías el ingreso en +{d_ing:.0f}k  (de {x0[0]:.0f}k a {x0[0]+d_ing:.0f}k)")
if d_deu is not None: print(f"   • o bajabas la deuda en  -{d_deu:.0f}k  (de {x0[1]:.0f}k a {x0[1]-d_deu:.0f}k)")
print("   (no le pedimos cambiar la antigüedad: no es accionable a voluntad)")

# visual: frontera en el plano ingreso-deuda, con el punto y los dos contrafactuales
gi=np.linspace(20,120,120); gd=np.linspace(0,60,120); GI,GD=np.meshgrid(gi,gd)
GA=np.full(GI.shape, x0[2])
P=clf.predict_proba(np.column_stack([GI.ravel(),GD.ravel(),GA.ravel()]))[:,1].reshape(GI.shape)
plt.figure()
plt.contourf(GI,GD,P,levels=[0,0.5,1],colors=["#3a1f28","#16303f"],alpha=.7)
plt.contour(GI,GD,P,levels=[0.5],colors=[CELESTE],linewidths=2)
plt.scatter([x0[0]],[x0[1]],color=CORAL,s=80,zorder=5,label="vos (rechazado)")
if d_ing is not None: plt.scatter([x0[0]+d_ing],[x0[1]],color=BEAM,s=80,zorder=5,label="cf: +ingreso")
if d_deu is not None: plt.scatter([x0[0]],[x0[1]-d_deu],color=GOLD,s=80,zorder=5,label="cf: -deuda")
plt.xlabel("ingreso (k)"); plt.ylabel("deuda (k)"); plt.title("Contrafactuales: el cambio mínimo que cruza la frontera")
plt.legend(); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El contrafactual da una instrucción **accionable y concreta**: "te habrían aprobado con tanto más de ingreso, o tanto menos de deuda". En el gráfico se ve la frontera de decisión y los dos puntos contrafactuales — el camino más corto desde "rechazado" hasta cruzar la línea, moviendo solo lo que la persona puede cambiar.

Esto responde la pregunta que de verdad le importa al afectado ("¿qué hago?"), algo que una importancia global o una atribución no hacen. Y respeta lo que no se puede cambiar: no le pedimos modificar su antigüedad.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Para decisiones que afectan a personas (crédito, selección, salud), la mejor explicación no es 'qué pesó' sino 'qué cambiar'. El contrafactual es accionable, comprensible sin saber de modelos, y trata a la persona como alguien que puede actuar.",
 "<b>Los cuidados.</b> Un buen contrafactual es realista (sugerir algo alcanzable) y respeta lo inmutable (no pidas cambiar edad o género). Buscá el cambio mínimo y plausible, no cualquiera que cruce la frontera.",
 "<b>El criterio.</b> Elegí el tipo de explicación según QUIÉN la recibe: importancia global para entender el modelo, atribución para justificar, contrafactual para que la persona pueda actuar. La interpretabilidad útil empieza por preguntarse para quién es.",
]))
print("Contrafactuales encontrados:")
if d_ing is not None: print(f"   ingreso +{d_ing:.0f}k")
if d_deu is not None: print(f"   deuda  -{d_deu:.0f}k")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("counterfactuals.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> counterfactuals.ipynb ({len(cells)} celdas)")
