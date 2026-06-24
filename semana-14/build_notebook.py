"""Genera sesgo_varianza.ipynb — Semana 14 (jueves) de 'IA sin humo'.
La curva de sesgo-varianza: por qué el modelo más complejo no es el mejor.
Free/runnable (sklearn). Menciona double descent como matiz moderno."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 〽️ Sesgo-varianza: por qué el modelo más complejo no gana""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">〽️ 📉 📈</div>
    <h1 style="margin:.1em 0 0;font-size:2.0em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       La curva de <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">sesgo-varianza</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 14 · Incertidumbre</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Más complejo no es mejor. El error de entrenamiento siempre baja; el de test
       baja, toca fondo y vuelve a subir. Ese punto medio es el que buscás.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from IPython.display import display, HTML
rng = np.random.default_rng(0)
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
def fit(d):
    return make_pipeline(PolynomialFeatures(d), LinearRegression())
print("Listo.")""")

md(r"""## 1 · Tres modelos: poco, justo, demasiado""")

code(r"""display(intro("🎯", "1 · Underfit, justo, overfit", [
 "<b>De qué se trata.</b> Ajustamos polinomios de distinto grado (la 'complejidad' del modelo) a unos pocos datos ruidosos generados por una función conocida. Grado bajo = modelo rígido (underfit); grado alto = modelo que se retuerce para pasar por cada punto (overfit); en el medio está el justo.",
 "<b>Qué vas a ver.</b> Tres ajustes superpuestos a los datos. El de grado alto pasa casi exacto por los puntos de entrenamiento… y se dispara entre ellos. Memorizó el ruido en vez de aprender la señal.",
 "🧮 <b>Dónde mirar.</b> El modelo más complejo es el que MEJOR se ajusta a los datos que vio. Eso es justo el problema, no la virtud.",
]))
def f(x): return np.cos(1.5*x)+0.3*x
xtr = np.sort(rng.uniform(-3,3,30)); ytr = f(xtr)+rng.normal(0,0.3,30)
xg = np.linspace(-3,3,500); yg = f(xg)
plt.figure()
plt.scatter(xtr,ytr,color=MIST,s=25,zorder=5,label="datos (con ruido)")
for d,c,lab in [(1,CORAL,"grado 1 (underfit)"),(4,BEAM,"grado 4 (justo)"),(18,GOLD,"grado 18 (overfit)")]:
    m=fit(d).fit(xtr.reshape(-1,1),ytr); plt.plot(xg,m.predict(xg.reshape(-1,1)),color=c,lw=2,label=lab)
plt.ylim(-3,3); plt.title("Mismos datos, tres complejidades"); plt.legend(fontsize=9); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El grado 1 es demasiado rígido: ni siquiera captura la forma (underfit, mucho sesgo). El grado 18 pasa casi exacto por cada punto de entrenamiento pero se vuelve loco entre ellos: memorizó el ruido (overfit, mucha varianza). El grado 4 captura la señal sin perseguir el ruido. La moraleja incómoda: el modelo que mejor ajusta los datos de entrenamiento (grado 18) es el peor para datos nuevos.""")

md(r"""## 2 · La curva: train siempre baja, test hace U""")

code(r"""display(intro("〽️", "2 · El error vs la complejidad", [
 "<b>De qué se trata.</b> Para cada grado medimos el error en entrenamiento y en test (datos no vistos). El de entrenamiento baja siempre (más complejidad = ajusta mejor lo que vio). El de test cuenta otra historia.",
 "<b>Qué vas a ver.</b> El error de test BAJA hasta un punto óptimo (el balance sesgo-varianza) y después SUBE: a partir de ahí, complejidad extra solo agrega varianza, no señal. Esa U es la firma del overfitting.",
 "🧮 <b>Dónde mirar.</b> La brecha entre train y test que se abre a la derecha: ese es el overfitting medido. Y el mínimo de la curva de test es la complejidad que querés.",
]))
grados=list(range(1,20))
tr=[]; te=[]
for d in grados:
    m=fit(d).fit(xtr.reshape(-1,1),ytr)
    tr.append(mean_squared_error(ytr,m.predict(xtr.reshape(-1,1))))
    te.append(mean_squared_error(yg,np.clip(m.predict(xg.reshape(-1,1)),-10,10)))
best=grados[int(np.argmin(te))]
plt.figure()
plt.plot(grados,tr,"o-",color=CORAL,label="error de entrenamiento")
plt.plot(grados,te,"o-",color=BEAM,label="error de test")
plt.axvline(best,color=CELESTE,ls="--",lw=1); plt.text(best+.2,max(te)*0.6,f"óptimo: grado {best}",color=CELESTE)
plt.yscale("log"); plt.xlabel("complejidad (grado del polinomio)"); plt.ylabel("error (MSE, log)")
plt.title("Sesgo-varianza: el test toca fondo y vuelve a subir"); plt.legend(); plt.tight_layout(); plt.show()
print(f"Mínimo error de test en grado {best}. Train sigue bajando; test sube después.")""")

md(r"""📝 **Lectura.** Acá está el corazón del asunto. El error de **entrenamiento baja monótono**: cuanta más complejidad, mejor ajusta lo que ya vio. Pero el error de **test hace una U**: baja hasta un óptimo y después sube, porque la complejidad extra ya no captura señal, solo memoriza ruido (varianza).

Esto explica por qué "mi modelo da error casi cero en entrenamiento" no es una buena noticia: probablemente estés a la derecha del óptimo, sobreajustando. La complejidad correcta se elige mirando el error de **validación/test**, nunca el de entrenamiento.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué (y un matiz moderno)", [
 "<b>El punto.</b> Más capacidad no es mejor. Existe una complejidad óptima, y se encuentra con un set de validación, no minimizando el error de entrenamiento. Un error de train bajísimo suele ser señal de overfitting, no de éxito.",
 "<b>Cómo elegir.</b> Curva de validación (como esta), regularización, o validación cruzada (respetando el tiempo si es serie). El objetivo es generalizar, no memorizar.",
 "<b>El matiz moderno (double descent).</b> En modelos MUY sobreparametrizados (redes gigantes, muchísimos más parámetros que datos), se observó que tras el pico de la U el error vuelve a bajar — un fenómeno llamado double descent que matiza la intuición clásica. Pero para la mayoría de los modelos tabulares, la U de sesgo-varianza sigue mandando.",
]))
print(f"Óptimo: grado {best}.  El criterio: validar, no minimizar el error de entrenamiento.")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("sesgo_varianza.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> sesgo_varianza.ipynb ({len(cells)} celdas)")
