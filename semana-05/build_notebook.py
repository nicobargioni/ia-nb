"""Genera backtesting_walkforward.ipynb — Semana 5 (jueves) de 'IA sin humo'.
Por qué validar una serie de tiempo con CV random (shuffle) infla el resultado,
y cómo walk-forward (TimeSeriesSplit) da la verdad. Free/runnable (sklearn+numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# ⏳ Backtesting honesto: walk-forward vs CV ingenuo""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">⏳ 📉 ⚠️</div>
    <h1 style="margin:.1em 0 0;font-size:2.3em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Backtesting <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">honesto</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 5 · Series temporales</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Validar una serie de tiempo barajando las filas te filtra el futuro y te da
       una métrica hermosa que se desploma en producción. Lo medimos.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from IPython.display import display, HTML
rng = np.random.default_rng(1)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(8.5,4.2),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · Una serie de tiempo y un modelo""")

code(r"""display(intro("📈", "1 · La serie y el modelo", [
 "<b>De qué se trata.</b> Armamos una serie con tendencia, estacionalidad y ruido autocorrelado — como casi cualquier serie real (ventas, tráfico, demanda). Entrenamos un modelo que predice el valor de hoy a partir de los últimos 5 valores (features de lag).",
 "<b>Qué vas a ver.</b> La serie graficada y el set de features. Hasta acá, nada raro. El problema aparece cuando elegimos CÓMO validar.",
 "🧮 <b>Dónde mirar.</b> Los puntos vecinos en el tiempo se parecen mucho (autocorrelación). Acordate de esto: es la grieta por la que se cuela el leakage.",
]))
n=500; t=np.arange(n)
noise=np.zeros(n)
for i in range(1,n): noise[i]=0.7*noise[i-1]+rng.normal(0,1)   # ruido AR(1)
y = 0.02*t + 3*np.sin(t/25) + noise
L=5
X = np.column_stack([np.roll(y,k) for k in range(1,L+1)])[L:]
yt = y[L:]
plt.figure(); plt.plot(t, y, color=BEAM, lw=1.3)
plt.title("La serie: tendencia + estacionalidad + ruido autocorrelado")
plt.xlabel("tiempo"); plt.ylabel("valor"); plt.tight_layout(); plt.show()
print(f"{len(yt)} observaciones · features = últimos {L} valores")""")

md(r"""## 2 · Dos formas de validar, dos verdades distintas""")

code(r"""display(intro("⚖️", "2 · CV ingenuo (shuffle) vs walk-forward", [
 "<b>De qué se trata.</b> Validamos el MISMO modelo de dos maneras. (1) CV ingenuo: K-fold con shuffle, que mezcla las filas al azar. (2) Walk-forward: TimeSeriesSplit, que siempre entrena con el pasado y valida con el futuro. Comparamos el R² que reporta cada uno.",
 "<b>Qué vas a ver.</b> El CV ingenuo reporta un R² mucho más alto. No es que el modelo sea mejor: es que con shuffle, para predecir un punto, el modelo entrenó con sus vecinos temporales (pasado Y futuro). Está interpolando entre puntos que ya conoce. Eso es leakage.",
 "🧮 <b>Dónde mirar.</b> La diferencia entre las dos barras es la 'optimismo de fantasía' que te llevás a producción si validás mal.",
]))
model = RandomForestRegressor(n_estimators=200, random_state=0)
naive = cross_val_score(model, X, yt, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
wf    = cross_val_score(model, X, yt, cv=TimeSeriesSplit(5), scoring="r2").mean()
print(f"CV ingenuo (shuffle):  R² = {naive:.3f}   <- infla")
print(f"Walk-forward:          R² = {wf:.3f}   <- honesto")
print(f"Optimismo de fantasía: {naive-wf:.3f}")

fig,ax=plt.subplots(figsize=(6.5,4))
ax.bar(["CV ingenuo\n(shuffle)","Walk-forward\n(honesto)"],[naive,wf],color=[CORAL,BEAM])
for i,v in enumerate([naive,wf]): ax.text(i,v+.01,f"{v:.2f}",ha="center",color=INK,fontweight="bold")
ax.set_ylabel("R² (validación)"); ax.set_title("Mismo modelo, dos métricas: una miente")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El mismo modelo, los mismos datos, y dos R² muy distintos. El CV con shuffle reporta mucho más alto — pero es **mentira**: al barajar, para predecir un punto el modelo entrenó con sus vecinos en el tiempo, que se parecen muchísimo (la autocorrelación). Está interpolando entre puntos que ya vio, no prediciendo el futuro. Walk-forward te da el número honesto, porque solo deja usar el pasado. Esa diferencia es exactamente la decepción que te espera en producción si validaste con shuffle.""")

md(r"""## 3 · Por qué pasa: mirá los folds""")

code(r"""display(intro("🔍", "3 · La causa, visual", [
 "<b>De qué se trata.</b> Dibujamos qué índices usa cada esquema para entrenar (azul) y validar (coral). Ahí se ve de un vistazo por qué uno filtra el futuro y el otro no.",
 "<b>Qué vas a ver.</b> En el CV ingenuo, los puntos de validación están rodeados de puntos de entrenamiento posteriores: el modelo 've el futuro'. En walk-forward, el entrenamiento siempre queda a la izquierda (pasado) y la validación a la derecha (futuro).",
 "🧮 <b>Dónde mirar.</b> El orden temporal. Esa es toda la diferencia conceptual entre una validación honesta y una que se autoengaña.",
]))
fig,(a1,a2)=plt.subplots(2,1,figsize=(9,4.2),sharex=True)
m=len(yt); idx=np.arange(m)
for f,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=0).split(idx)):
    a1.scatter(tr,[f]*len(tr),c=BEAM,s=4); a1.scatter(te,[f]*len(te),c=CORAL,s=6)
a1.set_title("CV ingenuo (shuffle): validación rodeada de futuro  →  leakage",color=CORAL); a1.set_ylabel("fold")
for f,(tr,te) in enumerate(TimeSeriesSplit(5).split(idx)):
    a2.scatter(tr,[f]*len(tr),c=BEAM,s=4); a2.scatter(te,[f]*len(te),c=CORAL,s=6)
a2.set_title("Walk-forward: entrenar=pasado, validar=futuro  →  honesto",color=BEAM)
a2.set_ylabel("fold"); a2.set_xlabel("índice temporal")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El gráfico lo dice todo. Arriba (CV ingenuo), los puntos coral de validación están **rodeados** de puntos azules de entrenamiento que vienen *después* en el tiempo: el modelo entrena con el futuro para predecir el pasado. Abajo (walk-forward), el entrenamiento siempre está a la izquierda de la validación: solo pasado. Mismo modelo, distinta forma de cortar — y eso decide si tu métrica es real o ciencia ficción.""")

md(r"""## 4 · Cierre""")

code(r"""display(intro("💡", "4 · El para qué", [
 "<b>El punto.</b> Una métrica inflada por leakage temporal es peor que no tener métrica: te da confianza falsa, te hace elegir el modelo equivocado, y el desastre llega después — cuando ya es caro.",
 "<b>La regla.</b> En series de tiempo NUNCA barajes. Validá walk-forward (entrenar pasado, validar futuro), en varios cortes. Y cuidá los features que miran al futuro: promedios sobre toda la serie, imputaciones globales, encodings con datos posteriores. Todos son la misma trampa con otra cara.",
 "<b>El criterio.</b> No es un modelo más complejo lo que arregla esto: es entender que el tiempo tiene una flecha y respetarla. La validación honesta es la que te ahorra perder plata con convicción.",
]))
print(f"CV ingenuo (shuffle):  R² = {naive:.3f}")
print(f"Walk-forward (honesto): R² = {wf:.3f}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("backtesting_walkforward.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> backtesting_walkforward.ipynb ({len(cells)} celdas)")
