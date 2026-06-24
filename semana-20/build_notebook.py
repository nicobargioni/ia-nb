"""Genera bootstrap.ipynb — Semana 20 (jueves) de 'IA sin humo'.
Bootstrap: intervalos de confianza para CUALQUIER estadístico, sin fórmula y sin
asumir distribución. Caso: la mediana de datos sesgados. Free/runnable (numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🎽 Bootstrap: intervalos de confianza para cualquier estadístico""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🎽 📊 🔁</div>
    <h1 style="margin:.1em 0 0;font-size:2.1em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       El <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">bootstrap</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 20 · Incertidumbre</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       ¿Intervalo de confianza para la mediana? ¿Para un p90, una correlación, un KPI raro?
       No hay fórmula. El bootstrap te lo da remuestreando, sin asumir nada.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(16)
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
print("Listo.")""")

md(r"""## 1 · La idea: remuestrear para ver la incertidumbre""")

code(r"""display(intro("🔁", "1 · Bootstrap en una idea", [
 "<b>De qué se trata.</b> Tenés UNA muestra y calculaste un estadístico (digamos, la mediana de los ingresos por usuario). ¿Cuánta incertidumbre tiene ese número? Para la media hay fórmula; para la mediana, un p90, una correlación o un KPI a medida, no. El bootstrap lo resuelve sin fórmula: remuestreás tu propia muestra (con reemplazo) muchas veces, calculás el estadístico en cada remuestreo, y la dispersión de esos valores ES tu incertidumbre.",
 "<b>Qué vas a ver.</b> Sobre datos sesgados (tipo ingresos), tomamos una muestra, generamos 2000 remuestreos y calculamos la mediana de cada uno. La distribución de esas medianas nos da el intervalo de confianza (percentiles 2.5 y 97.5).",
 "🧮 <b>Dónde mirar.</b> El intervalo bootstrap es asimétrico (refleja la asimetría real de los datos), algo que un 'media ± 2 desvíos' nunca capta.",
]))
poblacion = lambda n: rng.lognormal(0.5,0.8,n)   # sesgada
x = poblacion(40)
B=2000
boot_meds = np.array([np.median(rng.choice(x,len(x),replace=True)) for _ in range(B)])
lo,hi = np.percentile(boot_meds,[2.5,97.5])
print(f"Mediana de la muestra: {np.median(x):.3f}")
print(f"IC 95% por bootstrap: [{lo:.3f}, {hi:.3f}]  (asimétrico, sin asumir distribución)")
plt.figure()
plt.hist(boot_meds,bins=40,color=BEAM,alpha=.8)
plt.axvline(lo,color=GOLD,ls="--"); plt.axvline(hi,color=GOLD,ls="--")
plt.axvline(np.median(x),color=CORAL,lw=2,label="mediana de la muestra")
plt.title("Distribución bootstrap de la mediana"); plt.xlabel("mediana remuestreada"); plt.legend(); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El bootstrap te dio un intervalo de confianza para la mediana **sin ninguna fórmula** y sin asumir que los datos son normales. La distribución de las medianas remuestreadas dibuja, directamente, la incertidumbre del estadístico. Y fijate que el intervalo es asimétrico: respeta la forma sesgada de los datos, algo imposible con un "± 2 desvíos" simétrico. Esto funciona igual para un p90, una correlación, un AUC o cualquier KPI raro que se te ocurra.""")

md(r"""## 2 · ¿Funciona? Chequeamos la cobertura""")

code(r"""display(intro("✅", "2 · Cobertura: bootstrap vs fórmula ingenua", [
 "<b>De qué se trata.</b> ¿El IC al 95% del bootstrap contiene de verdad el valor real el 95% de las veces? Lo verificamos simulando muchas muestras. Y lo comparamos con la 'fórmula ingenua': aplicar la SE de la mediana que vale para datos NORMALES a datos que no lo son (un error común).",
 "<b>Qué vas a ver.</b> El bootstrap queda cerca del 95% objetivo (bien calibrado). La fórmula ingenua se descalibra (intervalos de ancho equivocado) porque su supuesto de normalidad no se cumple.",
 "🧮 <b>Dónde mirar.</b> La cobertura de cada método contra el 95%. El bootstrap acierta sin saber nada de la distribución; la fórmula falla por asumir lo que no es.",
]))
true_median=np.exp(0.5)
def coverage(method, trials=3000, n=40, B=1000):
    inside=0
    for _ in range(trials):
        s=poblacion(n)
        if method=="normal":
            se=1.253*s.std(ddof=1)/np.sqrt(n); m=np.median(s); lo,hi=m-1.96*se,m+1.96*se
        else:
            bs=np.array([np.median(rng.choice(s,n,replace=True)) for _ in range(B)]); lo,hi=np.percentile(bs,[2.5,97.5])
        inside += lo<=true_median<=hi
    return inside/trials
cov_n, cov_b = coverage("normal"), coverage("boot")
print(f"Mediana real: {true_median:.3f}")
print(f"Cobertura fórmula ingenua (objetivo 95%): {cov_n:.1%}  <- mal calibrada")
print(f"Cobertura bootstrap       (objetivo 95%): {cov_b:.1%}  <- bien calibrada")""")

md(r"""📝 **Lectura.** El bootstrap queda pegado al 95% objetivo: sus intervalos son honestos. La fórmula ingenua —usar la SE de la mediana para datos normales sobre datos que no lo son— se descalibra: da intervalos del ancho equivocado, porque asume una forma que los datos no tienen. El bootstrap no asume nada: deja que los datos hablen.

La gran ventaja no es solo que funciona mejor acá: es que funciona para **cualquier estadístico**, incluso los que no tienen fórmula. Esa generalidad es lo que lo hace tan útil y tan subestimado.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Reportar un estadístico (una mediana, un p90, una tasa de conversión, una correlación) sin intervalo esconde su incertidumbre. El bootstrap te da ese intervalo para CUALQUIER estadístico, en pocas líneas, sin asumir distribución.",
 "<b>Cuándo brilla.</b> Justo cuando no hay fórmula simple: medianas, percentiles, ratios, métricas de negocio a medida, diferencias entre grupos. Y cuando dudás de que tus datos sean normales (casi siempre).",
 "<b>El criterio.</b> Un número sin intervalo es media respuesta. El bootstrap es la navaja suiza de la incertidumbre: simple, general y honesto. No saber que existe te deja reportando puntos donde deberías reportar rangos.",
]))
print(f"bootstrap: {cov_b:.1%} de cobertura  ·  fórmula ingenua: {cov_n:.1%}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("bootstrap.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> bootstrap.ipynb ({len(cells)} celdas)")
