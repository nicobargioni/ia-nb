"""Genera drift_monitor.ipynb — Semana 18 (jueves) de 'IA sin humo'.
Monitorear drift en un stream: detectar con PSI cuándo la distribución cambió,
antes de que el modelo se degrade en silencio. Free/runnable (numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 📡 Monitor de drift: detectar cuándo tu modelo se vuelve obsoleto""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">📡 📈 🚨</div>
    <h1 style="margin:.1em 0 0;font-size:2.0em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Monitor de <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">drift</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 18 · MLOps</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tu modelo se degrada en silencio cuando el mundo cambia. El drift no tira error:
       hay que salir a buscarlo. Lo detectamos con PSI, antes de que duela.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(14)
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
print("Listo.")""")

md(r"""## 1 · El PSI: medir cuánto cambió una distribución""")

code(r"""display(intro("📏", "1 · Population Stability Index (PSI)", [
 "<b>De qué se trata.</b> El PSI es una métrica simple y muy usada en la industria para medir cuánto se corrió una distribución respecto de una referencia (los datos de entrenamiento). Se calcula partiendo el rango en bins y comparando qué proporción de datos cae en cada bin: referencia vs actual.",
 "<b>Qué vas a ver.</b> Reglas prácticas del PSI: <b>< 0.1</b> sin drift relevante, <b>0.1–0.25</b> drift moderado (vigilar), <b>> 0.25</b> drift fuerte (actuar). Fijamos la referencia con los datos de entrenamiento.",
 "🧮 <b>Dónde mirar.</b> El PSI es una suma sobre bins de (proporción_actual − proporción_ref) × log(actual/ref). Cero = idénticas; crece cuando la distribución se aparta.",
]))
ref = rng.normal(0,1,5000)
edges = np.quantile(ref, np.linspace(0,1,11)); edges[0]=-1e9; edges[-1]=1e9
def psi(cur):
    e = np.clip(np.histogram(ref,edges)[0]/len(ref), 1e-4, None)
    a = np.clip(np.histogram(cur,edges)[0]/len(cur), 1e-4, None)
    return float(np.sum((a-e)*np.log(a/e)))
print("Referencia: 5000 muestras de la distribución de entrenamiento.")
print(f"PSI contra una muestra de la MISMA distribución: {psi(rng.normal(0,1,400)):.3f}  (≈0, sin drift)")
print(f"PSI contra una distribución corrida (media +0.8): {psi(rng.normal(0.8,1,400)):.3f}  (drift fuerte)")""")

md(r"""📝 **Lectura.** El PSI es bajísimo cuando comparás contra datos de la misma distribución, y salta cuando la distribución se corre. Es barato de calcular, no necesita el resultado real (solo las features que llegan) y se interpreta con umbrales conocidos. Es la herramienta básica para vigilar **data drift** en producción. Ahora lo ponemos a correr sobre un stream que cambia en el medio.""")

md(r"""## 2 · El monitor en acción sobre un stream""")

code(r"""display(intro("🚨", "2 · Detectar el cambio en vivo", [
 "<b>De qué se trata.</b> Simulamos 40 ventanas temporales de datos que llegan a producción. Las primeras 20 vienen de la distribución de entrenamiento; a partir de la 20, la distribución se corre (drift de media). Calculamos el PSI de cada ventana contra la referencia y lo graficamos en el tiempo.",
 "<b>Qué vas a ver.</b> El PSI se mantiene bajo (verde) mientras los datos son los de siempre, y CRUZA el umbral de alarma justo cuando empieza el drift — sin que nadie se haya quejado todavía. Eso es detectar el problema antes de que el modelo falle visiblemente.",
 "🧮 <b>Dónde mirar.</b> El momento en que la curva cruza el umbral coincide con el cambio real. Esa es la diferencia entre enterarte por un dashboard o por un cliente enojado.",
]))
T=40; shift=20; thr=0.2
psis=[]
for s in range(T):
    mu = 0.0 if s<shift else 0.8
    psis.append(psi(rng.normal(mu,1,400)))
psis=np.array(psis); alarma=int(np.argmax(psis>thr))
plt.figure()
plt.plot(range(T),psis,"o-",color=BEAM,lw=2,label="PSI")
plt.axhline(thr,color=GOLD,ls="--",lw=1.5,label=f"umbral de alarma ({thr})")
plt.axvline(shift,color=CORAL,ls=":",lw=1.5,label="drift real (paso 20)")
plt.xlabel("ventana temporal"); plt.ylabel("PSI"); plt.title("El monitor detecta el drift apenas ocurre")
plt.legend(fontsize=9); plt.tight_layout(); plt.show()
print(f"Drift real en el paso {shift}. El monitor disparó la alarma en el paso {alarma}.")""")

md(r"""📝 **Lectura.** El monitor disparó la alarma **exactamente cuando empezó el drift**, mientras el sistema seguía respondiendo con normalidad y nadie se había quejado. Esa es toda la idea: el drift no tira un error, así que si no lo medís, te enterás tarde y caro. Un PSI que cruza el umbral es la señal para investigar y, si corresponde, reentrenar — antes de que el modelo lleve semanas decidiendo mal.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Poner un modelo en producción no es el final: es el principio del monitoreo. El drift es silencioso por definición (el sistema anda, las predicciones empeoran). Vigilarlo activamente es lo que evita la degradación invisible.",
 "<b>El stack mínimo.</b> PSI o tests de distribución (KS, chi-cuadrado) sobre las features en ventanas; monitorear también la distribución de las predicciones; y, cuando llega el resultado real, la métrica de performance (lo único que detecta concept drift). Umbrales de acción definidos de antemano.",
 "<b>El criterio.</b> 'Entrené un buen modelo' y 'tengo un buen sistema en producción' son cosas distintas. La diferencia es, en buena parte, el monitoreo. Un modelo sin detección de drift es una bomba de tiempo que anda perfecto… hasta que no.",
]))
print(f"drift real: paso {shift}  ·  alarma del monitor: paso {alarma}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("drift_monitor.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> drift_monitor.ipynb ({len(cells)} celdas)")
