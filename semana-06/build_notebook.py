"""Genera feature_importance.ipynb — Semana 6 (jueves) de 'IA sin humo'.
Importancia de features: la que viene por defecto (impurity) miente; permutation
out-of-sample es honesta; y la trampa de las features correlacionadas. Free (sklearn)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🔬 Feature importance honesta: impurity vs permutation""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🔬 📊 ⚠️</div>
    <h1 style="margin:.1em 0 0;font-size:2.25em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Feature importance <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">honesta</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 6 · Interpretabilidad</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       La importancia que trae tu modelo por defecto te miente: infla variables sin señal
       y se confunde con las correlacionadas. Permutation out-of-sample es la honesta.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
rng = np.random.default_rng(2)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
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

md(r"""## 1 · Un dataset con trampas a propósito""")

code(r"""display(intro("🧪", "1 · El dataset", [
 "<b>De qué se trata.</b> Armamos un dataset donde sabemos la verdad. Tiene: una feature con SEÑAL real, una COPIA correlacionada de esa señal, una feature de ALTA CARDINALIDAD sin ninguna señal (tipo un ID con miles de valores únicos) y dos features de RUIDO puro. El target depende solo de la señal.",
 "<b>Qué vas a ver.</b> Entrenamos un Random Forest. La pregunta es: ¿qué feature importance reporta? Y sobre todo, ¿le creemos?",
 "🧮 <b>Dónde mirar.</b> La feature de alta cardinalidad y las de ruido NO deberían importar nada. La copia correlacionada y la señal son, en realidad, la misma información.",
]))
n=3000
signal = rng.normal(0,1,n)
dup    = signal + rng.normal(0,0.05,n)          # copia correlacionada
hc     = rng.integers(0,n,n).astype(float)      # alta cardinalidad, SIN señal (tipo ID)
z1, z2 = rng.normal(0,1,n), rng.normal(0,1,n)   # ruido puro
X = np.column_stack([signal,dup,hc,z1,z2]); names=["señal","copia(corr)","id_alta_card","ruido1","ruido2"]
y = (signal + rng.normal(0,0.5,n) > 0).astype(int)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.4,random_state=0)
clf = RandomForestClassifier(n_estimators=300,random_state=0).fit(Xtr,ytr)
print(f"Accuracy test: {clf.score(Xte,yte):.3f}")
print("Verdad: solo 'señal' (y su copia) importan; id_alta_card y ruidos = 0")""")

md(r"""## 2 · La importancia por defecto miente""")

code(r"""display(intro("⚠️", "2 · Importancia por defecto (impurity)", [
 "<b>De qué se trata.</b> La feature importance que trae Random Forest / XGBoost por defecto (basada en impurity) se calcula sobre los datos de entrenamiento y tiene un sesgo conocido: <b>infla las features con muchos valores distintos</b>, aunque no tengan señal.",
 "<b>Qué vas a ver.</b> La feature de alta cardinalidad (un ID inútil) recibe importancia parecida a las de ruido — y todas reciben algo, cuando deberían dar cero. El modelo se apoyó en el ID para 'memorizar' filas en entrenamiento.",
 "🧮 <b>Dónde mirar.</b> Que el id_alta_card y los ruidos no den exactamente cero ya es una señal de alarma: la métrica está contaminada por el sesgo de impurity.",
]))
imp = clf.feature_importances_
order = np.argsort(imp)
plt.figure(); plt.barh([names[i] for i in order],[imp[i] for i in order],color=CORAL)
plt.title("Importancia por defecto (impurity) — sobre datos de entrenamiento")
plt.xlabel("importancia"); plt.tight_layout(); plt.show()
for nme,v in zip(names,imp): print(f"  {nme:14s} {v:.3f}")""")

md(r"""📝 **Lectura.** Mirá el `id_alta_card` y los ruidos: reciben importancia no nula, cuando la verdad es que **no aportan nada**. Es el sesgo clásico de la importancia por impurity: premia a las features con muchos valores distintos porque le dan al árbol más puntos por dónde cortar, y mide todo sobre el set de entrenamiento, donde el modelo ya sobreajustó. Si tomaras decisiones con este ranking, invertirías en una variable que es puro ruido.""")

md(r"""## 3 · Permutation importance: la honesta""")

code(r"""display(intro("✅", "3 · Permutation importance (out-of-sample)", [
 "<b>De qué se trata.</b> Idea simple: si una feature importa, romperla debería empeorar al modelo. Barajamos esa columna —solo esa— en el set de TEST y medimos cuánto cae la métrica. Lo que no aporta, no cae al romperse.",
 "<b>Qué vas a ver.</b> Ahora el id_alta_card y los ruidos caen a ~cero, como debe ser. La señal y su copia quedan arriba. Medido out-of-sample, sin el sesgo de impurity.",
 "🧮 <b>Dónde mirar.</b> Compará con el gráfico anterior: las features inútiles desaparecen. Esa es la diferencia entre una métrica honesta y una contaminada.",
]))
r = permutation_importance(clf, Xte, yte, n_repeats=15, random_state=0)
order = np.argsort(r.importances_mean)
plt.figure(); plt.barh([names[i] for i in order],[r.importances_mean[i] for i in order],
                        xerr=[r.importances_std[i] for i in order],color=BEAM)
plt.title("Permutation importance (out-of-sample) — honesta")
plt.xlabel("caída de accuracy al romper la feature"); plt.tight_layout(); plt.show()
for nme,v in zip(names,r.importances_mean): print(f"  {nme:14s} {v:.3f}")""")

md(r"""📝 **Lectura.** Ahora sí: el `id_alta_card` y los ruidos caen a **prácticamente cero**, y la señal con su copia quedan arriba. Romper una feature inútil no empeora al modelo, así que la métrica la marca como inútil — sin el sesgo de la alta cardinalidad y medido sobre datos que el modelo no vio. Esta es la importancia en la que podés confiar para entender en qué se apoya tu modelo de verdad.""")

md(r"""## 4 · La trampa de las features correlacionadas""")

code(r"""display(intro("🔗", "4 · El asterisco: features correlacionadas", [
 "<b>De qué se trata.</b> Hay una trampa que afecta a permutation importance (y también a SHAP): cuando dos features están muy correlacionadas, romper una sola subestima a ambas. ¿Por qué? Porque el modelo, al perder una, se apoya en la copia y casi no se entera.",
 "<b>Qué vas a ver.</b> Romper 'señal' sola baja poco. Romper 'copia' sola baja poco. Pero romper LAS DOS juntas tira el modelo: ahí está la verdadera importancia conjunta, escondida cuando las mirás por separado.",
 "🧮 <b>Dónde mirar.</b> La caída individual vs la caída conjunta. La diferencia es la información compartida que ninguna métrica por-feature te muestra bien.",
]))
base = clf.score(Xte, yte)
def drop_cols(cols):
    Xp = Xte.copy()
    for c in cols: rng.shuffle(Xp[:,c])
    return base - clf.score(Xp, yte)
print(f"Caída al romper 'señal' sola:        {drop_cols([0]):.3f}")
print(f"Caída al romper 'copia' sola:        {drop_cols([1]):.3f}")
print(f"Caída al romper 'señal'+'copia':     {drop_cols([0,1]):.3f}   <- la real")
print("\nCada una sola parece modesta; juntas son TODA la señal.")""")

md(r"""📝 **Lectura.** Acá está la trampa que más confunde. Romper la señal sola, o la copia sola, baja poco — porque el modelo se apoya en la otra y apenas lo nota. Pero romper **las dos juntas** desploma el modelo: esa es su importancia real, que estaba escondida porque la información está repartida entre features correlacionadas.

La moraleja: con features correlacionadas, **no leas la importancia de cada una por separado** (ni permutation ni SHAP). Agrupalas y evalualas juntas. Y el recordatorio de siempre: importancia ≠ causalidad. Saber qué pregunta estás respondiendo —y qué trampa puede estar arruinando la respuesta— es el criterio que ninguna librería te da.
""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("feature_importance.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> feature_importance.ipynb ({len(cells)} celdas)")
