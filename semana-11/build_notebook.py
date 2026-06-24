"""Genera forecast_intervalos.ipynb — Semana 11 (jueves) de 'IA sin humo'.
Intervalos de predicción con quantile regression: predecir un rango, no solo el punto,
y que el intervalo se ensanche donde hay más incertidumbre. Free/runnable (sklearn)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 📏 Intervalos de predicción con quantile regression""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">📏 📈 〰️</div>
    <h1 style="margin:.1em 0 0;font-size:2.1em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Predecir el <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">rango</span>, no el punto</h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 11 · Series temporales</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Un forecast sin intervalo es una opinión. Con quantile regression predecimos
       un rango honesto que se ensancha justo donde hay más incertidumbre.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
rng = np.random.default_rng(6)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(8.5,4.6),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · Datos con incertidumbre que cambia""")

code(r"""display(intro("🌫️", "1 · El escenario", [
 "<b>De qué se trata.</b> Generamos datos donde la incertidumbre NO es constante: el ruido crece a medida que avanzamos (heterocedástico). Es lo típico en la realidad — predecir mañana es más fácil que predecir dentro de un mes. Un intervalo de ancho fijo sería mentira; necesitamos uno que se adapte.",
 "<b>Qué vas a ver.</b> Entrenamos TRES modelos de quantile regression: uno para el percentil 10, uno para el 50 (la mediana) y uno para el 90. Entre el p10 y el p90 queda un intervalo de predicción del 80%.",
 "🧮 <b>Dónde mirar.</b> La clave es que cada modelo se entrena con una pérdida distinta (la 'pinball loss') que apunta a su percentil, no al promedio. Por eso captura la forma real de la incertidumbre.",
]))
n=3000
x = rng.uniform(0,10,n)
y = np.sin(x)*3 + 0.4*x + rng.normal(0, 0.3+0.25*x)   # ruido que crece con x
Xtr,Xte,ytr,yte = train_test_split(x.reshape(-1,1),y,test_size=.4,random_state=0)
preds={}
for a in [0.1,0.5,0.9]:
    preds[a] = GradientBoostingRegressor(loss="quantile",alpha=a,n_estimators=200,max_depth=3,random_state=0).fit(Xtr,ytr).predict(Xte)
print("Tres modelos entrenados: p10, p50 (mediana), p90.")""")

md(r"""## 2 · El intervalo que se adapta""")

code(r"""display(intro("〰️", "2 · El rango, y su cobertura", [
 "<b>De qué se trata.</b> Graficamos los datos, la mediana predicha (p50) y la banda entre p10 y p90. Y verificamos lo más importante: ¿el valor real cae dentro del intervalo el 80% de las veces, como prometimos?",
 "<b>Qué vas a ver.</b> La banda es ANGOSTA donde hay poca incertidumbre y ANCHA donde hay mucha. Eso es lo que un intervalo de ancho fijo no puede hacer. La cobertura empírica queda cerca del 80% objetivo.",
 "🧮 <b>Dónde mirar.</b> El ancho de la banda variando, y la cobertura medida. Un forecast que comunica su propia incertidumbre.",
]))
cov = np.mean((yte>=preds[0.1]) & (yte<=preds[0.9]))
xs = Xte.ravel(); o = np.argsort(xs)
plt.figure()
plt.scatter(xs, yte, s=5, color=MIST, alpha=.4, label="datos reales")
plt.fill_between(xs[o], preds[0.1][o], preds[0.9][o], color=BEAM, alpha=.25, label="intervalo p10–p90 (80%)")
plt.plot(xs[o], preds[0.5][o], color=CELESTE, lw=2, label="mediana (p50)")
plt.title(f"Intervalo de predicción adaptativo — cobertura empírica {cov:.0%}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout(); plt.show()
ancho = preds[0.9]-preds[0.1]
print(f"Cobertura p10–p90 (objetivo 80%): {cov:.1%}")
print(f"Ancho medio del intervalo donde x<3: {ancho[xs<3].mean():.2f}   donde x>7: {ancho[xs>7].mean():.2f}")""")

md(r"""📝 **Lectura.** La banda hace lo que un intervalo fijo no puede: es **angosta donde el modelo está seguro y ancha donde no**. Lo ves en los números (ancho ~1.6 al principio, ~6 al final) y en el gráfico. Y la cobertura empírica queda cerca del 80% prometido: cuando decimos "80% de probabilidad de caer acá adentro", se cumple.

Esto es predecir con honestidad. La quantile regression no asume que el error es normal ni simétrico: deja que los datos definan la forma del intervalo. Por eso captura que la incertidumbre crece con el horizonte, algo que un "media ± 2 desvíos" jamás haría.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Un solo número esconde el riesgo. El intervalo es la información que tu negocio necesita para decidir: planificás para el p10 (pesimista), esperás el p50 y no te quedás corto si pega el p90.",
 "<b>Las herramientas.</b> Quantile regression (lo que vimos: predecís percentiles directo) o conformal prediction (intervalos con cobertura garantizada, sin asumir distribución; lo vimos en otra semana). Las dos te dan rango en vez de punto.",
 "<b>El criterio.</b> Reportar solo el punto se ve 'más profesional' pero es menos riguroso: esconde lo que no sabés. Un forecast honesto comunica su incertidumbre. El rango no es el adorno, es la respuesta.",
]))
print(f"Cobertura: {cov:.1%}  ·  ancho medio: {ancho.mean():.2f}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("forecast_intervalos.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> forecast_intervalos.ipynb ({len(cells)} celdas)")
