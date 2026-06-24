"""Genera conformal_prediction.ipynb — Semana 2 (jueves) de 'IA sin humo'.
Conformal prediction para clasificación: intervalos/sets con cobertura GARANTIZADA,
sin asumir distribución. Free / runnable en Colab (solo sklearn + numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🎯 Conformal prediction: incertidumbre con garantía""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🎯 📊 ✅</div>
    <h1 style="margin:.1em 0 0;font-size:2.4em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Conformal <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">prediction</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 2 · Incertidumbre</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tu modelo escupe un número y se calla el riesgo. Conformal le pone un intervalo
       con cobertura GARANTIZADA — sin asumir ninguna distribución, sobre cualquier modelo.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
np.random.seed(0)

NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; LINE="#15303f"
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

md(r"""## 1 · Un clasificador normal (y su falsa confianza)""")

code(r"""display(intro("🤖", "1 · El modelo y su 'confianza' que miente", [
 "<b>De qué se trata.</b> Entrenamos un clasificador común (Random Forest) sobre un problema de 4 clases con cierto solapamiento. La trampa: cuando el modelo dice 'clase A con 0.85 de probabilidad', ese 0.85 NO garantiza que acierte el 85% de las veces. Es confianza sin respaldo.",
 "<b>Qué vas a ver.</b> Partimos los datos en TRES: entrenamiento, calibración y test. El set de calibración es la pieza clave de conformal: es donde vamos a 'medir cuánto se equivoca' el modelo para después dar garantías.",
 "🧮 <b>Dónde mirar.</b> Por ahora solo entrenamos. La magia viene en el paso 2.",
]))
X,y = make_classification(n_samples=3000,n_classes=4,n_informative=6,n_features=12,
                          n_clusters_per_class=1,class_sep=1.1,random_state=0)
X_tr,X_tmp,y_tr,y_tmp = train_test_split(X,y,test_size=.6,random_state=0)
X_cal,X_te,y_cal,y_te = train_test_split(X_tmp,y_tmp,test_size=.5,random_state=0)
clf = RandomForestClassifier(n_estimators=300,random_state=0).fit(X_tr,y_tr)
print(f"train={len(y_tr)}  calibración={len(y_cal)}  test={len(y_te)}")
print(f"Accuracy en test: {clf.score(X_te,y_te):.3f}")""")

md(r"""📝 **Lectura.** Tenemos un modelo decente. Pero un accuracy global no te dice nada sobre la confianza de una predicción puntual. Si el modelo dice 0.6 para una fila y 0.99 para otra, ¿qué garantía tenés en cada caso? Ninguna, hasta ahora. Eso es lo que conformal viene a arreglar — y la clave es ese set de **calibración** que apartamos.""")

md(r"""## 2 · Split conformal: la garantía en 3 líneas""")

code(r"""display(intro("🎯", "2 · La calibración que da la garantía", [
 "<b>De qué se trata.</b> Definimos un 'score de no-conformidad': qué tan sorprendido está el modelo del valor verdadero. Para clasificación usamos <code>score = 1 − p(clase_verdadera)</code>. Lo calculamos en el set de calibración, donde SÍ conocemos la respuesta.",
 "<b>Qué vas a ver.</b> Tomamos el cuantil (1−α) de esos scores → un umbral <code>q̂</code>. Después, en test, el 'set de predicción' incluye TODAS las clases cuya probabilidad supere <code>1−q̂</code>. La teoría garantiza que la clase verdadera cae en el set ≥ 90% de las veces.",
 "🧮 <b>Dónde mirar.</b> Es distribution-free: no asumimos normalidad ni nada. La única condición es intercambiabilidad de los datos. Mirá el q̂ que sale abajo.",
]))
alpha = 0.10
p_cal = clf.predict_proba(X_cal)
scores = 1 - p_cal[np.arange(len(y_cal)), y_cal]
n = len(scores)
qhat = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="higher")
print(f"alpha = {alpha}  → cobertura objetivo = {1-alpha:.0%}")
print(f"Umbral conformal q̂ = {qhat:.3f}")
print("Regla: el set incluye toda clase con prob >= 1 - q̂ =", round(1-qhat,3))""")

md(r"""## 3 · ¿Cumple la garantía? Medimos en test""")

code(r"""display(intro("✅", "3 · Cobertura empírica", [
 "<b>De qué se trata.</b> Construimos los sets de predicción en el set de TEST (datos que el modelo nunca vio ni para entrenar ni para calibrar) y verificamos: ¿la clase verdadera cae adentro el 90% de las veces, como prometimos?",
 "<b>Qué vas a ver.</b> La cobertura empírica debería dar ≈ 0.90, y el tamaño promedio de los sets. Sets chicos (1 clase) = el modelo está seguro; sets grandes (2-3 clases) = el modelo duda, y conformal te lo dice honestamente.",
 "🧮 <b>Dónde mirar.</b> Compará la cobertura objetivo (línea) con la lograda (barra). Que coincidan es la garantía funcionando.",
]))
p_te = clf.predict_proba(X_te)
sets = p_te >= (1 - qhat)
covered = sets[np.arange(len(y_te)), y_te]
coverage = covered.mean()
set_sizes = sets.sum(1)
print(f"Cobertura empírica: {coverage:.3f}  (objetivo {1-alpha:.2f})")
print(f"Tamaño promedio del set: {set_sizes.mean():.2f} clases")

fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
a1.bar(["objetivo","logrado"],[1-alpha,coverage],color=[MIST,BEAM])
a1.axhline(1-alpha,color=CELESTE,ls="--",lw=1); a1.set_ylim(0,1); a1.set_title("Cobertura")
for i,v in enumerate([1-alpha,coverage]): a1.text(i,v+.02,f"{v:.2f}",ha="center",color=INK)
vals,counts=np.unique(set_sizes,return_counts=True)
a2.bar(vals,counts,color=BEAM,alpha=.85); a2.set_title("Tamaño de los sets de predicción")
a2.set_xlabel("clases en el set"); a2.set_ylabel("cantidad"); a2.set_xticks(vals)
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** La cobertura lograda cae prácticamente sobre el objetivo del 90% — **sin que hayamos asumido ninguna distribución**. Eso es lo notable: la garantía es matemática y vale para cualquier modelo (cambiá el Random Forest por una red neuronal y sigue funcionando).

Y mirá los tamaños de set: cuando el modelo está seguro, el set tiene 1 sola clase; cuando duda, el set crece a 2 o 3. Conformal **convierte la incertidumbre en algo accionable**: en vez de un 0.6 vacío, te dice 'puede ser A o C' y vos decidís si automatizás o mandás a revisión humana.""")

md(r"""## 4 · Por qué importa (y por qué casi nadie lo usa)""")

code(r"""display(intro("💡", "4 · El para qué", [
 "<b>El punto.</b> La mayoría de los pipelines reportan un punto (la clase más probable) y tiran a la basura la incertidumbre. En decisiones de negocio —aprobar un crédito, derivar un ticket, diagnosticar— el riesgo importa tanto como la predicción.",
 "<b>Conformal te da una perilla honesta:</b> elegís el nivel de cobertura que tu negocio tolera (90%, 99%) y el método ajusta el tamaño de los sets. Más exigencia → sets más grandes → más casos a revisión humana. Es una decisión de riesgo explícita, no un número escondido.",
 "<b>Por qué casi nadie lo usa:</b> no está en el tutorial de 'tu primer modelo'. Pero es de lo más útil y simple que podés sumar a un sistema en producción. Tres líneas y una garantía.",
]))
for thr in [0.05, 0.10, 0.20]:
    q = np.quantile(scores, np.ceil((n+1)*(1-thr))/n, method="higher")
    s = (p_te >= (1-q))
    cov = s[np.arange(len(y_te)),y_te].mean()
    print(f"cobertura objetivo {1-thr:.0%}  →  lograda {cov:.3f}  ·  set promedio {s.sum(1).mean():.2f} clases")""")

md(r"""📝 **Lectura.** Subir la exigencia de cobertura (de 90% a 95% a 99%) agranda los sets: el modelo tiene que 'incluir más opciones' para garantizarte que no se le escape la verdadera. Esa es exactamente la decisión de negocio que conformal hace **explícita** en vez de barrerla bajo la alfombra.

Conformal prediction es un caso perfecto de la tesis de la serie: no es un modelo más grande ni un LLM, es una **herramienta estadística sólida** que te da algo que ningún modelo escupe solo — una garantía honesta sobre su propia incertidumbre. Ahí está el criterio.

— Serie *IA sin humo* · github.com/nicobargioni/ia-nb
""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("conformal_prediction.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> conformal_prediction.ipynb ({len(cells)} celdas)")
