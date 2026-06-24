"""Genera baseline_vs_ml.ipynb — Semana 17 (jueves) de 'IA sin humo'.
Forecast: el baseline estacional naive le gana a un modelo ML elaborado.
Por qué hay que vencer al baseline antes de festejar. Free/runnable (sklearn+numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🥊 Forecast: baseline vs ML (y por qué el baseline gana)""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🥊 📊 📉</div>
    <h1 style="margin:.1em 0 0;font-size:2.0em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Baseline <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">vs ML</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 17 · Series temporales</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Montaste un modelo de ML para forecasting. ¿Le gana a 'lo mismo que el martes pasado'?
       Te sorprendería cuántas veces no. Y por qué eso importa.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from IPython.display import display, HTML
rng = np.random.default_rng(13)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; GOLD="#e8b86a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(8.5,4.4),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
  "savefig.facecolor":NIGHT,"axes.edgecolor":LINE,"axes.labelcolor":INK,"text.color":INK,
  "axes.titlecolor":INK,"xtick.color":MIST,"ytick.color":MIST,"axes.grid":True,"grid.color":LINE,
  "grid.alpha":.5,"font.size":11,"legend.framealpha":0})
def intro(emoji,titulo,parrafos):
    ps="".join(f'<p style="margin:0 0 11px;color:#cfe3ef;font-size:.97em;line-height:1.62">{p}</p>' for p in parrafos)
    return HTML(f'''<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;
      background:linear-gradient(135deg,#0a1b27,#102b3c 55%,#15384b);border:1px solid rgba(79,184,232,.22);
      border-left:5px solid #4fb8e8;border-radius:12px;padding:20px 26px;margin:6px 0;color:#eef7fb;box-shadow:0 8px 30px rgba(0,0,0,.35)">
      <div style="font-size:1.3em;font-weight:800;text-transform:uppercase;margin-bottom:12px">{emoji}&nbsp;{titulo}</div>{ps}</div>''')
def mae(a,b): return float(np.mean(np.abs(np.array(a)-np.array(b))))
print("Listo.")""")

md(r"""## 1 · Una serie con estacionalidad y dos baselines""")

code(r"""display(intro("📅", "1 · La serie y los baselines", [
 "<b>De qué se trata.</b> Serie diaria con tendencia suave, estacionalidad SEMANAL (los días se parecen al mismo día de la semana anterior) y ruido — como ventas, tráfico o demanda. Definimos dos baselines tontos: <b>naive</b> (mañana = hoy) y <b>seasonal naive</b> (este martes = el martes pasado).",
 "<b>Qué vas a ver.</b> Pronosticamos los últimos 120 días, un paso por vez. El seasonal naive, que no tiene un solo parámetro entrenado, ya captura casi toda la señal — porque la serie es estacional.",
 "🧮 <b>Dónde mirar.</b> La diferencia entre el naive simple y el seasonal naive: respetar la estacionalidad, sin modelo, ya mejora muchísimo. Ese es el rival a vencer.",
]))
n=730; t=np.arange(n); S=7
serie=10+0.02*t+4*np.sin(2*np.pi*t/S)+rng.normal(0,1.2,n)
test=list(range(n-120,n)); real=[serie[i] for i in test]
naive=[serie[i-1] for i in test]; snaive=[serie[i-S] for i in test]
print(f"MAE naive (mañana = hoy):           {mae(real,naive):.2f}")
print(f"MAE seasonal naive (= hace 7 días): {mae(real,snaive):.2f}  <- baseline fuerte, cero entrenamiento")""")

md(r"""📝 **Lectura.** El seasonal naive le saca una diferencia enorme al naive simple, y no entrenó nada: solo dice "este día va a ser como el mismo día de la semana pasada". En cualquier serie con estacionalidad fuerte, este baseline es brutalmente difícil de superar, porque captura gratis el patrón dominante. Ahora veamos si nuestro modelo de ML elaborado le gana.""")

md(r"""## 2 · El modelo de ML… ¿le gana?""")

code(r"""display(intro("🤖", "2 · ML con features de lag", [
 "<b>De qué se trata.</b> Entrenamos un Gradient Boosting con features de los últimos 7 días (lags) para predecir el siguiente. Es un modelo de verdad, con cientos de árboles. La pregunta: ¿justifica su complejidad venciendo al seasonal naive?",
 "<b>Qué vas a ver.</b> El spoiler incómodo: el ML puede quedar a la par o incluso PEOR que el seasonal naive. Toda esa maquinaria para no superar a 'mirá el martes pasado'.",
 "🧮 <b>Dónde mirar.</b> Las tres barras de error. Si el ML no le gana claramente al baseline, su complejidad no se justifica.",
]))
def feats(i): return [serie[i-k] for k in range(1,8)]
Xtr=np.array([feats(i) for i in range(8,n-120)]); ytr=serie[8:n-120]
m=GradientBoostingRegressor(n_estimators=200,random_state=0).fit(Xtr,ytr)
ml=[m.predict([feats(i)])[0] for i in test]
e_naive,e_snaive,e_ml=mae(real,naive),mae(real,snaive),mae(real,ml)
print(f"MAE naive          : {e_naive:.2f}")
print(f"MAE seasonal naive : {e_snaive:.2f}")
print(f"MAE modelo ML      : {e_ml:.2f}")
plt.figure(figsize=(6.5,4))
plt.bar(["naive","seasonal\nnaive","ML\n(GBR+lags)"],[e_naive,e_snaive,e_ml],color=[CORAL,BEAM,GOLD])
for i,v in enumerate([e_naive,e_snaive,e_ml]): plt.text(i,v+.02,f"{v:.2f}",ha="center",color=INK,fontweight="bold")
plt.ylabel("MAE (menor = mejor)"); plt.title("¿El ML justifica su complejidad?"); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Acá está el cachetazo de humildad. El seasonal naive —una línea de código, cero entrenamiento— queda a la par (o mejor) que el Gradient Boosting con cientos de árboles y features de lag. Toda esa maquinaria no le ganó a "mirá el mismo día de la semana pasada".

No significa que el ML sea inútil: significa que en esta serie, la estacionalidad explica casi todo, y el baseline ya la captura. El modelo complejo solo se justifica si aporta **por encima** del baseline lo suficiente como para pagar su costo de entrenarlo, mantenerlo y explicarlo.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> En forecasting, los baselines no son un trámite: son rivales durísimos. El seasonal naive captura gratis la estacionalidad, que suele ser la mayor parte de la señal. Un modelo que no le gana con claridad no es un modelo, es un gasto.",
 "<b>El error común.</b> Reportar 'mi modelo tiene MAE 1.3' sin decir que el seasonal naive tiene 1.2. Suena bien en aislamiento; comparado con el baseline, es peor. La métrica sola engaña; la métrica CONTRA el baseline informa.",
 "<b>El criterio.</b> Siempre, antes de festejar un modelo: ¿le gana al baseline tonto? Si no, el problema no necesitaba ML, o tus features no tienen señal extra. La sofisticación se mide contra lo trivial — es la espina dorsal de toda la serie.",
]))
print(f"seasonal naive={e_snaive:.2f}  vs  ML={e_ml:.2f}  → ", "el baseline aguanta" if e_snaive<=e_ml else "el ML gana")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("baseline_vs_ml.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> baseline_vs_ml.ipynb ({len(cells)} celdas)")
