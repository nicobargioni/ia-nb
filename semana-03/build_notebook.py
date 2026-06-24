"""Genera diff_in_diff.ipynb — Semana 3 (jueves) de 'IA sin humo'.
Diferencias-en-diferencias: estimar un efecto causal cuando no podés hacer un A/B.
Free / runnable (solo numpy + matplotlib)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 📐 Diferencias-en-diferencias: causalidad sin A/B""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">📐 🔀 ➖</div>
    <h1 style="margin:.1em 0 0;font-size:2.3em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Diferencias-en-<span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">diferencias</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 3 · Causalidad</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Lanzaste un cambio y no pudiste hacer un A/B. ¿Cómo medís su efecto real
       sin confundirlo con tendencias que ya venían pasando? Con una resta doble.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(0)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(8,4.6),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · El escenario: un cambio sin grupo de control aleatorio""")

code(r"""display(intro("🏬", "1 · El problema", [
 "<b>De qué se trata.</b> Lanzaste algo (una promo, un rediseño, una política) en un grupo —digamos, ciertas sucursales— pero no pudiste randomizar quién lo recibía. Querés saber el efecto real sobre las ventas. Comparar 'después vs antes' del grupo tratado te confunde el efecto con la tendencia general. Comparar 'tratado vs no-tratado' te confunde con que los grupos ya eran distintos de entrada.",
 "<b>Qué vas a ver.</b> Simulamos datos donde conocemos la VERDAD (efecto real = +5). El grupo tratado ya era más alto de movida (+10) y hubo una tendencia temporal (+3) que afectó a todos. Vamos a ver cómo la comparación ingenua se equivoca y cómo diff-in-diff recupera el +5 exacto.",
 "🧮 <b>Dónde mirar.</b> Cuatro promedios: tratado/control × antes/después. Toda la magia sale de combinarlos.",
]))
n=500; TRUE_EFFECT=5.0
def gen(treated, post):
    base = 30 if treated else 20      # el grupo tratado ya era más alto
    time = 3 if post else 0           # tendencia temporal que afecta a TODOS
    eff  = TRUE_EFFECT if (treated and post) else 0
    return base + time + eff + rng.normal(0, 4, n)
cb, ca = gen(0,0), gen(0,1)   # control: antes, después
tb, ta = gen(1,0), gen(1,1)   # tratado: antes, después
print(f"Control  antes={cb.mean():.2f}  después={ca.mean():.2f}")
print(f"Tratado  antes={tb.mean():.2f}  después={ta.mean():.2f}")
print(f"(efecto real simulado = {TRUE_EFFECT})")""")

md(r"""## 2 · Las comparaciones ingenuas fallan""")

code(r"""display(intro("⚠️", "2 · Por qué lo intuitivo se equivoca", [
 "<b>De qué se trata.</b> Probamos las dos comparaciones que todos hacen primero, y vemos que las dos están sesgadas — cada una por un motivo distinto.",
 "<b>Qué vas a ver.</b> 'Tratado después − tratado antes' incluye la tendencia temporal que habría pasado igual. 'Tratado después − control después' incluye que el grupo tratado ya era más alto de entrada. Ninguna da el +5 real.",
 "🧮 <b>Dónde mirar.</b> Compará cada estimación ingenua contra el efecto real. La distancia es el sesgo.",
]))
antes_despues = ta.mean() - tb.mean()
tratado_control = ta.mean() - ca.mean()
print(f"Ingenua 'después − antes' (solo tratado): {antes_despues:.2f}   <- infla con la tendencia temporal")
print(f"Ingenua 'tratado − control' (solo después): {tratado_control:.2f}   <- infla con la diferencia de base")
print(f"Efecto real: {TRUE_EFFECT}")""")

md(r"""📝 **Lectura.** Las dos comparaciones intuitivas **mienten**, y por razones distintas. "Después − antes" sobre el grupo tratado mezcla el efecto con la **tendencia temporal** que habría ocurrido igual (el +3). "Tratado − control" en el período post mezcla el efecto con que el grupo tratado **ya partía más alto** (el +10). Cada atajo captura un sesgo. Por eso ninguna se acerca al +5 verdadero.""")

md(r"""## 3 · Diff-in-diff: la resta que cancela los dos sesgos""")

code(r"""display(intro("📐", "3 · La doble diferencia", [
 "<b>De qué se trata.</b> La idea es elegante: calculá el cambio del grupo tratado (después − antes) Y el cambio del control en el mismo período. Restá uno del otro. La tendencia temporal afecta a ambos por igual, así que se CANCELA. Lo que queda es el efecto causal limpio.",
 "<b>Qué vas a ver.</b> El estimador diff-in-diff recupera el +5 (con ruido mínimo de muestreo). El control actúa como el 'contrafactual': lo que le habría pasado al tratado si no hubiera recibido nada.",
 "🧮 <b>Dónde mirar.</b> El supuesto clave es 'tendencias paralelas': sin el tratamiento, ambos grupos habrían evolucionado igual. El gráfico de abajo lo muestra.",
]))
did = (ta.mean() - tb.mean()) - (ca.mean() - cb.mean())
print(f"Estimador Diff-in-Diff: {did:.2f}   (efecto real {TRUE_EFFECT})")

fig,ax=plt.subplots()
x=[0,1]
ax.plot(x,[cb.mean(),ca.mean()],"o-",color=MIST,lw=2,label="Control")
ax.plot(x,[tb.mean(),ta.mean()],"o-",color=BEAM,lw=2,label="Tratado (real)")
# contrafactual: tratado siguiendo la tendencia del control
cf = tb.mean() + (ca.mean()-cb.mean())
ax.plot(x,[tb.mean(),cf],"o--",color=CORAL,lw=2,label="Tratado (contrafactual)")
ax.annotate("", xy=(1,ta.mean()), xytext=(1,cf), arrowprops=dict(arrowstyle="<->",color=CELESTE,lw=2))
ax.text(1.02, (ta.mean()+cf)/2, f" efecto DiD ≈ {did:.1f}", color=CELESTE, va="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["antes","después"]); ax.set_ylabel("ventas")
ax.set_title("Diff-in-diff: el control da el contrafactual"); ax.legend(); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Diff-in-diff recupera el **+5 real**. La clave visual: la línea punteada (contrafactual) es lo que le habría pasado al grupo tratado si solo lo hubiera afectado la tendencia general — la misma pendiente que el control. La distancia entre esa línea y lo que realmente pasó es el efecto causal. El control no es un grupo "comparable" cualquiera: es nuestra mejor estimación del **qué habría pasado si no**.""")

md(r"""## 4 · La forma de regresión (mismo número, una línea)""")

code(r"""display(intro("🧮", "4 · DiD como una regresión", [
 "<b>De qué se trata.</b> En la práctica nadie resta promedios a mano: se corre una regresión con un término de interacción <code>tratado × post</code>. Su coeficiente ES el estimador diff-in-diff, y de yapa te da el error estándar para saber si es significativo.",
 "<b>Qué vas a ver.</b> Armamos la regresión <code>y = b0 + b1·tratado + b2·post + b3·(tratado·post)</code> con numpy. El coeficiente <code>b3</code> coincide con la doble diferencia de antes.",
 "🧮 <b>Dónde mirar.</b> b3 ≈ 5. Es el mismo resultado, pero escalable a muchos grupos, períodos y controles adicionales.",
]))
y = np.concatenate([cb,ca,tb,ta])
treated = np.concatenate([np.zeros(2*n), np.ones(2*n)])
post    = np.concatenate([np.zeros(n),np.ones(n),np.zeros(n),np.ones(n)])
X = np.column_stack([np.ones_like(y), treated, post, treated*post])
beta,*_ = np.linalg.lstsq(X, y, rcond=None)
print("Coeficientes de la regresión:")
for name,b in zip(["intercepto","tratado","post","tratado×post (DiD)"], beta):
    print(f"  {name:22s} {b:6.2f}")
print(f"\nEl coeficiente de interacción ({beta[3]:.2f}) = el efecto causal estimado.")""")

md(r"""📝 **Lectura.** El coeficiente de la interacción `tratado×post` da el mismo ~5: es diff-in-diff escrito como regresión. Esta forma es la que usás en serio, porque escala a muchos grupos y períodos, te da intervalos de confianza, y te deja sumar covariables.

Diff-in-diff es la tesis de la serie en estado puro: no es un modelo gigante ni un LLM, es una **idea estadística simple** —una resta doble apoyada en un supuesto explícito (tendencias paralelas)— que te deja estimar causalidad cuando no podés randomizar. El criterio está en saber *cuándo* el supuesto se sostiene, y eso lo ponés vos.

— Serie *IA sin humo* · github.com/nicobargioni/ia-nb
""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("diff_in_diff.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> diff_in_diff.ipynb ({len(cells)} celdas)")
