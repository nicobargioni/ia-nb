"""Genera calibracion.ipynb — Semana 8 (jueves) de 'IA sin humo'.
Calibrar probabilidades: medir ECE/Brier, leer el reliability diagram y arreglar
con Platt (sigmoid) vs isotónica. Free/runnable (sklearn)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 📊 Calibración: que el 0.9 signifique 0.9""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">📊 🎯 ✅</div>
    <h1 style="margin:.1em 0 0;font-size:2.3em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">Calibración</span> de probabilidades</h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 8 · Incertidumbre</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tu modelo dice 0.9 y acierta 0.7. Predice bien pero sus probabilidades mienten.
       Lo medimos (Brier, ECE), lo vemos (reliability diagram) y lo arreglamos.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from IPython.display import display, HTML
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; GOLD="#e8b86a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(7,5),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
  "savefig.facecolor":NIGHT,"axes.edgecolor":LINE,"axes.labelcolor":INK,"text.color":INK,
  "axes.titlecolor":INK,"xtick.color":MIST,"ytick.color":MIST,"axes.grid":True,"grid.color":LINE,
  "grid.alpha":.5,"font.size":11,"legend.framealpha":0})
def intro(emoji,titulo,parrafos):
    ps="".join(f'<p style="margin:0 0 11px;color:#cfe3ef;font-size:.97em;line-height:1.62">{p}</p>' for p in parrafos)
    return HTML(f'''<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;
      background:linear-gradient(135deg,#0a1b27,#102b3c 55%,#15384b);border:1px solid rgba(79,184,232,.22);
      border-left:5px solid #4fb8e8;border-radius:12px;padding:20px 26px;margin:6px 0;color:#eef7fb;box-shadow:0 8px 30px rgba(0,0,0,.35)">
      <div style="font-size:1.3em;font-weight:800;text-transform:uppercase;margin-bottom:12px">{emoji}&nbsp;{titulo}</div>{ps}</div>''')
def ece(p,y,bins=10):
    e=0.0; N=len(y)
    for b in range(bins):
        m=(p>b/bins)&(p<=(b+1)/bins)
        if m.sum()==0: continue
        e+=m.sum()/N*abs(p[m].mean()-y[m].mean())
    return e
print("Listo. (Brier y ECE: más bajo = mejor calibrado)")""")

md(r"""## 1 · Un modelo que predice bien pero miente en la probabilidad""")

code(r"""display(intro("🤥", "1 · El modelo sobreconfiado", [
 "<b>De qué se trata.</b> Entrenamos un Naive Bayes sobre datos con muchas features correlacionadas. Naive Bayes asume que las features son independientes; cuando no lo son (acá lo rompemos a propósito), el modelo se vuelve <b>sobreconfiado</b>: empuja sus probabilidades hacia 0 y 1. Clasifica bien, pero sus números no son fieles.",
 "<b>Qué vas a ver.</b> El reliability diagram: agrupamos las predicciones por su probabilidad y comparamos lo que el modelo DICE (eje X) contra lo que REALMENTE acierta (eje Y). La diagonal es la calibración perfecta. Y dos números: Brier y ECE.",
 "🧮 <b>Dónde mirar.</b> Si la curva se aleja de la diagonal, hay miscalibración. El ECE resume esa distancia en un solo número.",
]))
X,y = make_classification(n_samples=8000,n_informative=5,n_features=20,n_redundant=12,class_sep=0.7,random_state=0)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.4,random_state=0)
nb = GaussianNB().fit(Xtr,ytr)
p0 = nb.predict_proba(Xte)[:,1]
print(f"Accuracy: {nb.score(Xte,yte):.3f}   <- predice bien")
print(f"Brier: {brier_score_loss(yte,p0):.3f}   ECE: {ece(p0,yte):.3f}   <- pero la probabilidad miente")

frac,mean_pred = calibration_curve(yte,p0,n_bins=10)
plt.figure(); plt.plot([0,1],[0,1],"--",color=MIST,label="calibración perfecta")
plt.plot(mean_pred,frac,"o-",color=CORAL,label="sin calibrar")
plt.xlabel("probabilidad que dice el modelo"); plt.ylabel("frecuencia real de acierto")
plt.title("Reliability diagram — sin calibrar"); plt.legend(); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El modelo tiene buena accuracy —predice bien la clase— pero el reliability diagram muestra que la curva se aleja de la diagonal: cuando dice "0.9", en realidad acierta bastante menos. Es **sobreconfiado**. El ECE pone número a esa brecha. Y ojo: la accuracy NO te avisa de esto, porque acertar la clase y reportar una probabilidad fiel son dos cosas distintas. Si usaras ese 0.9 para decidir un umbral de riesgo, estarías decidiendo mal de forma sistemática.""")

md(r"""## 2 · Calibrar: Platt (sigmoid) vs isotónica""")

code(r"""display(intro("🔧", "2 · El arreglo", [
 "<b>De qué se trata.</b> Recalibramos las probabilidades con dos métodos clásicos. <b>Platt (sigmoid):</b> ajusta una curva en S; pocos parámetros, ideal con POCOS datos de calibración. <b>Isotónica:</b> ajusta una función escalonada monótona; más flexible, necesita MÁS datos o sobreajusta.",
 "<b>Qué vas a ver.</b> Las dos curvas recalibradas se pegan mucho más a la diagonal, y el ECE/Brier bajan. Importante: la calibración NO cambia qué clase predice el modelo (la accuracy queda igual); solo arregla las probabilidades.",
 "🧮 <b>Dónde mirar.</b> La caída del ECE. Y cuál método queda mejor según los datos.",
]))
results={"sin calibrar":(p0, ece(p0,yte), brier_score_loss(yte,p0))}
plt.figure(); plt.plot([0,1],[0,1],"--",color=MIST,label="perfecta")
plt.plot(mean_pred,frac,"o-",color=CORAL,label="sin calibrar")
for meth,col,lab in [("sigmoid",BEAM,"Platt (sigmoid)"),("isotonic",GOLD,"Isotónica")]:
    cc=CalibratedClassifierCV(GaussianNB(),method=meth,cv=5).fit(Xtr,ytr)
    p=cc.predict_proba(Xte)[:,1]; results[lab]=(p,ece(p,yte),brier_score_loss(yte,p))
    f,mp=calibration_curve(yte,p,n_bins=10); plt.plot(mp,f,"o-",color=col,label=lab)
plt.xlabel("probabilidad que dice el modelo"); plt.ylabel("frecuencia real de acierto")
plt.title("Reliability diagram — antes y después de calibrar"); plt.legend(); plt.tight_layout(); plt.show()
print(f"{'método':14s} {'ECE':>7} {'Brier':>8}")
for lab,(p,e,b) in results.items(): print(f"{lab:14s} {e:7.3f} {b:8.3f}")""")

md(r"""📝 **Lectura.** Las curvas calibradas se pegan a la diagonal y el ECE baja fuerte (sin calibrar ≈ 0.10 → calibrado ≈ 0.02-0.05). Ahora cuando el modelo dice 0.9, acierta cerca del 90%. Fijate que la accuracy no cambió: solo arreglamos las probabilidades, no las predicciones.

La regla práctica: **pocos datos de calibración → Platt**; **muchos datos → isotónica** (más flexible pero hambrienta de datos). Y calibrá siempre con datos que el modelo no usó para entrenar.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Si vas a USAR la probabilidad para decidir (umbral, riesgo, priorización, expected value), tiene que ser fiel. Un modelo sobreconfiado te lleva a decisiones sistemáticamente sesgadas — y la accuracy nunca te avisa.",
 "<b>Cómo.</b> Medí con Brier y ECE (no con accuracy). Mirá el reliability diagram. Calibrá con Platt o isotónica sobre un set aparte. Tres pasos, mejora directa.",
 "<b>El criterio.</b> 'Predecir la clase' y 'reportar una probabilidad confiable' son objetivos distintos. Saber cuál necesitás —y medir el correcto— es lo que separa un modelo de demo de uno que sostiene decisiones.",
]))
print("Resumen ECE:", {k: round(v[1],3) for k,v in results.items()})
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("calibracion.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> calibracion.ipynb ({len(cells)} celdas)")
