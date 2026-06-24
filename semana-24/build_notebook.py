"""Genera baseline_test.ipynb — Semana 24 (jueves, CIERRE) de 'IA sin humo'.
Capstone: el test del baseline. Antes de celebrar un modelo, comparalo contra la
versión tonta. A veces la complejidad gana (forecast); a veces el '91% de accuracy'
ya lo daba la mayoría. Y elegir la métrica correcta cambia el veredicto. Free."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🏁 El test del baseline — cierre de *IA sin humo*""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.30);margin:6px 0">
  <div style="padding:46px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.36), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(232,184,106,.18), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🏁 📏 🤔</div>
    <h1 style="margin:.1em 0 0;font-size:1.95em;font-weight:800;text-transform:uppercase;line-height:1.05;letter-spacing:-.02em">
       El test del <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">baseline</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 24 · Cierre de la serie</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:580px;margin-left:auto;margin-right:auto">
       Toda la serie en una idea: antes de celebrar un modelo, comparalo contra
       la versión tonta. Y elegí la métrica que no te miente.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(24)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; GOLD="#e8b86a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(9,4.2),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · Cuando la complejidad SÍ se gana su lugar""")

code(r"""display(intro("📈", "1 · Forecast: baseline tonto vs modelo", [
 "<b>De qué se trata.</b> Serie con tendencia y estacionalidad semanal. El baseline más tonto que existe en series: 'mañana va a ser como el mismo día de la semana pasada' (seasonal naive). Contra eso, un modelo que ajusta tendencia + estacionalidad por mínimos cuadrados. ¿Vale la pena el modelo?",
 "<b>Qué vas a ver.</b> Acá el modelo le gana al baseline de verdad: baja el error (MAE) de forma clara. Cuando hay tendencia sostenida, el naive se queda corto y el modelo lo aprovecha. Lift real → complejidad justificada.",
 "🧮 <b>Dónde mirar.</b> El % de mejora del modelo sobre el baseline. Ese número —el lift— es lo único que justifica la complejidad.",
]))
n=240; t=np.arange(n); S=7
y=50+0.08*t+6*np.sin(2*np.pi*t/S)+rng.normal(0,2.5,n)
tr,te=y[:200],y[200:]; tt=t[200:]
pred_naive=y[200-S:n-S]; mae_naive=np.mean(np.abs(te-pred_naive))
X=np.column_stack([np.ones(200),t[:200],np.sin(2*np.pi*t[:200]/S),np.cos(2*np.pi*t[:200]/S)])
beta,*_=np.linalg.lstsq(X,tr,rcond=None)
Xte=np.column_stack([np.ones(40),tt,np.sin(2*np.pi*tt/S),np.cos(2*np.pi*tt/S)]); pred_m=Xte@beta
mae_m=np.mean(np.abs(te-pred_m)); liftA=100*(mae_naive-mae_m)/mae_naive
print(f"MAE seasonal-naive: {mae_naive:.2f}")
print(f"MAE modelo        : {mae_m:.2f}")
print(f"Lift del modelo   : {liftA:.0f}%  -> la complejidad se gana su lugar")
plt.figure()
plt.plot(t[:200],tr,color=MIST,lw=1,label="histórico")
plt.plot(tt,te,color=INK,lw=1.5,label="real (test)")
plt.plot(tt,pred_naive,color=CORAL,lw=1.5,ls="--",label=f"naive (MAE {mae_naive:.1f})")
plt.plot(tt,pred_m,color=BEAM,lw=1.8,label=f"modelo (MAE {mae_m:.1f})")
plt.legend(fontsize=8); plt.title("Forecast: el modelo le gana al baseline"); plt.xlabel("día"); plt.show()""")

md(r"""📝 **Lectura.** Acá la historia es feliz: el modelo le saca ~23% al baseline. Hay una tendencia sostenida que el seasonal-naive no puede capturar (siempre repite la semana pasada, se queda atrás), y el modelo la modela. La complejidad rindió. Pero fijate que lo sabemos por una sola razón: **medimos el baseline primero**. Sin ese número, "MAE de 2.2" no significa nada.""")

md(r"""## 2 · Cuando el "91% de accuracy" no dice nada""")

code(r"""display(intro("🪤", "2 · Clasificación: la trampa del accuracy", [
 "<b>De qué se trata.</b> Problema de clasificación con clases desbalanceadas: solo ~9% de positivos (fraude, churn, enfermedad rara: casi siempre así). Entrenamos una regresión logística y reportamos su accuracy. El baseline tonto acá: predecir SIEMPRE la clase mayoritaria (todo negativo).",
 "<b>Qué vas a ver.</b> La logística da 91% de accuracy. Suena espectacular… hasta que ves que predecir 'siempre no' también da 91%. Lift sobre el baseline: cero. Pero el modelo NO es inútil: con la métrica correcta (AUC) se ve que sí separa. El problema era el accuracy, no el modelo.",
 "🧮 <b>Dónde mirar.</b> Accuracy del modelo == accuracy de la mayoría. Y abajo, el AUC: 0.72 vs 0.50. Elegir la métrica decide si el modelo 'sirve' o 'no sirve'.",
]))
nn=1500; Xb=rng.normal(0,1,(nn,4))
logit=0.9*Xb[:,0]+0.5*Xb[:,1]-3.0; p=1/(1+np.exp(-logit))
yb=(rng.uniform(size=nn)<p).astype(int)
Xtr,Xte2,ytr,yte=train_test_split(Xb,yb,test_size=.3,random_state=0)
maj=max(yte.mean(),1-yte.mean())
clf=LogisticRegression().fit(Xtr,ytr); acc=clf.score(Xte2,yte)
auc=roc_auc_score(yte,clf.predict_proba(Xte2)[:,1])
print(f"Tasa de positivos        : {yb.mean():.0%}")
print(f"Accuracy baseline (mayoría): {maj:.2f}")
print(f"Accuracy logística        : {acc:.2f}   -> lift {100*(acc-maj):.0f} puntos (!)")
print(f"AUC baseline (azar)        : 0.50")
print(f"AUC logística             : {auc:.2f}   -> ACÁ sí se ve la señal")
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.2,3.8))
a1.bar(["mayoría","logística"],[maj,acc],color=[MIST,CORAL]); a1.set_ylim(0,1)
a1.set_title("Accuracy: el modelo no agrega NADA"); a1.set_ylabel("accuracy")
a2.bar(["azar","logística"],[0.5,auc],color=[MIST,BEAM]); a2.set_ylim(0,1)
a2.set_title("AUC: el modelo SÍ separa"); a2.set_ylabel("AUC")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El mismo modelo es "inútil" o "útil" según la métrica que mires. En accuracy no le gana en nada al baseline tonto —porque con 9% de positivos, decir 'siempre no' ya acierta 91%—. Pero en AUC pasa de 0.50 (azar) a 0.72: el modelo ordena bien los casos, captura señal real. La lección doble: **(1)** siempre comparás contra el baseline antes de festejar un número, y **(2)** el baseline también te obliga a elegir la métrica correcta. El "91%" no era mentira; era irrelevante.""")

md(r"""## 3 · Cierre de la serie""")

code(r"""display(intro("🏁", "El criterio que cierra IA sin humo", [
 "<b>El test del baseline, en una frase:</b> ningún modelo —ni el más simple ni un LLM carísimo— vale por su número absoluto. Vale por cuánto le gana a la versión tonta del problema, medido con la métrica que no te miente. El lift es el rey; el número solo, no dice nada.",
 "<b>Por qué cierra la serie.</b> Durante 24 semanas vimos lo mismo desde 24 ángulos: búsqueda híbrida, conformal, diff-in-diff, calibración, backtesting, drift, agentes… En todos, la herramienta básica bien usada le marcaba el piso a la solución cara. Tirarle un LLM (o un modelo complejo) encima sin medir ese piso es fe, no ingeniería.",
 "<b>El criterio que ningún LLM te da.</b> Un modelo te da una predicción; no te dice contra qué compararla, qué métrica importa, ni si el problema valía la pena. Eso —el baseline, la métrica, la pregunta— lo ponés vos. Esa es la parte que no se automatiza, y es la que separa usar IA de entenderla.",
]))
print(f"Forecast   : naive {mae_naive:.1f} -> modelo {mae_m:.1f}  (lift {liftA:.0f}%, vale la pena)")
print(f"Clasificac.: accuracy 0.91 == baseline 0.91 (lift 0), pero AUC 0.50 -> {auc:.2f}")
print("\nGracias por leer las 24 semanas. — IA sin humo")
print("github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("baseline_test.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> baseline_test.ipynb ({len(cells)} celdas)")
