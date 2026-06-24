"""Genera double_ml.ipynb — Semana 15 (jueves) de 'IA sin humo'.
Double / Debiased ML: estimar un efecto causal usando ML para controlar confounders
no lineales, sin que el ML sesgue el estimando. Free/runnable (sklearn)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# ➗ Double Machine Learning: ML sin que te sesgue la causalidad""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">➗ 🤖 🎯</div>
    <h1 style="margin:.1em 0 0;font-size:2.0em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Double <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">Machine Learning</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 15 · Causalidad</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tenés muchos confounders con relaciones complejas. Querés usar ML para controlarlos.
       Hecho directo, te sesga. Double ML lo arregla trabajando con residuos.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from IPython.display import display, HTML
rng = np.random.default_rng(11)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(7,4.4),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · Confounders complejos y el control que falla""")

code(r"""display(intro("🌀", "1 · El problema", [
 "<b>De qué se trata.</b> Queremos el efecto causal de un tratamiento T sobre un resultado Y. Hay 5 confounders X que afectan a AMBOS, pero con relaciones NO lineales (curvas, interacciones). Conocemos la verdad: el efecto es θ=2.",
 "<b>Qué vas a ver.</b> El enfoque ingenuo —una regresión lineal de Y sobre T controlando X de forma lineal— da un número sesgado, porque el control lineal no captura las relaciones curvas de los confounders. Controlaste, pero mal.",
 "🧮 <b>Dónde mirar.</b> La distancia entre el estimado ingenuo y el θ=2 real. El sesgo viene de modelar mal cómo los confounders afectan a Y y a T.",
]))
n=4000; THETA=2.0
X=rng.normal(0,1,(n,5))
gX=np.sin(X[:,0]) + 0.5*X[:,1]**2 + X[:,2]           # X -> T (no lineal)
T=gX + rng.normal(0,1,n)
fX=2*np.cos(X[:,1]) + 1.5*X[:,3] + X[:,0]*X[:,2]      # X -> Y (no lineal + interacción)
Y=THETA*T + fX + rng.normal(0,1,n)
naive=LinearRegression().fit(np.column_stack([T,X]),Y).coef_[0]
print(f"Efecto causal REAL: θ = {THETA}")
print(f"Estimado ingenuo (OLS con control lineal de X): {naive:.2f}  → sesgado")""")

md(r"""📝 **Lectura.** El estimado ingenuo no da 2: controlar los confounders de forma lineal no alcanza cuando su efecto real es curvo y con interacciones. El modelo "controla" X pero deja parte de su influencia sin capturar, y esa influencia residual se cuela en el coeficiente de T. La tentación es usar ML (que sí modela relaciones complejas) para controlar X — pero hacerlo directo introduce otro sesgo, el de regularización. De ahí Double ML.""")

md(r"""## 2 · Double ML: residuos + cross-fitting""")

code(r"""display(intro("➗", "2 · La doble residualización", [
 "<b>De qué se trata.</b> Double ML usa ML para los confounders, pero de forma que el sesgo se cancele. Tres pasos: (1) predecir Y a partir de X con ML y quedarse con el residuo (lo que X no explica de Y); (2) predecir T a partir de X con ML y quedarse con el residuo; (3) regresar el residuo de Y sobre el residuo de T → ese coeficiente es el efecto causal.",
 "<b>Qué vas a ver.</b> Sumamos cross-fitting (entrenar los modelos en una mitad de los datos y predecir en la otra) para evitar overfitting. El resultado recupera el θ=2 real.",
 "🧮 <b>Dónde mirar.</b> Al trabajar con residuos, la parte de X (por más no lineal que sea) ya fue removida de ambos lados; lo que queda es la relación T→Y limpia. El sesgo de regularización del ML se cancela.",
]))
idx=rng.permutation(n); a,b=idx[:n//2],idx[n//2:]
def residuos(tr,te):
    mY=GradientBoostingRegressor(random_state=0).fit(X[tr],Y[tr])
    mT=GradientBoostingRegressor(random_state=0).fit(X[tr],T[tr])
    return Y[te]-mY.predict(X[te]), T[te]-mT.predict(X[te])
rY1,rT1=residuos(a,b); rY2,rT2=residuos(b,a)
rY=np.concatenate([rY1,rY2]); rT=np.concatenate([rT1,rT2])
dml=np.sum(rT*rY)/np.sum(rT*rT)
print(f"Double ML: {dml:.2f}   (real {THETA})")
plt.figure()
plt.bar(["ingenuo","Double ML","real"],[naive,dml,THETA],color=[CORAL,BEAM,MIST])
for i,v in enumerate([naive,dml,THETA]): plt.text(i,v+.03,f"{v:.2f}",ha="center",color=INK,fontweight="bold")
plt.ylabel("efecto estimado (θ)"); plt.title("Double ML recupera el efecto; el control lineal no")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Double ML recupera el **θ≈2 real**, mientras el control lineal se quedaba sesgado. La clave: al residualizar Y y T contra X con ML flexible, sacamos toda la influencia de los confounders (lineal o no) de ambos lados. Lo que queda —regresar residuo contra residuo— es la relación causal limpia. Y el cross-fitting evita que el ML se sobreajuste y reintroduzca sesgo.

Es la unión de dos mundos que suelen ir separados: la **flexibilidad del ML** para los confounders complejos y el **rigor de la inferencia causal** para el número que te importa.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Cuando tenés muchos confounders con relaciones complejas, Double ML te deja usar ML para controlarlos sin que ese ML sesgue la estimación del efecto. Te da el número causal con su incertidumbre.",
 "<b>El límite de siempre.</b> Como toda técnica causal observacional, solo corrige los confounders que MEDISTE. Si falta uno importante, el sesgo vuelve. El ML no te salva de eso: identificar los confounders correctos sigue siendo juicio humano.",
 "<b>El criterio.</b> ML y causalidad no son lo mismo. Un modelo predictivo excelente puede dar una estimación causal pésima si lo usás directo. Double ML es el puente correcto — pero el puente no reemplaza saber qué pregunta causal estás respondiendo.",
]))
print(f"ingenuo={naive:.2f}  ·  Double ML={dml:.2f}  ·  real={THETA}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("double_ml.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> double_ml.ipynb ({len(cells)} celdas)")
