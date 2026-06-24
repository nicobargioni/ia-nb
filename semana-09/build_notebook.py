"""Genera propensity_matching.ipynb — Semana 9 (jueves) de 'IA sin humo'.
Propensity score matching: estimar un efecto causal con datos observacionales
emparejando tratados y controles parecidos. Free/runnable (sklearn+numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# ⚖️ Propensity score matching: causalidad sin experimento""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">⚖️ 🔗 🎯</div>
    <h1 style="margin:.1em 0 0;font-size:2.15em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Propensity score <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">matching</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 9 · Causalidad</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Sin A/B test, los tratados no son comparables a los controles. Emparejando a
       cada uno con su 'gemelo' estadístico, recuperamos el efecto causal real.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from IPython.display import display, HTML
rng = np.random.default_rng(5)
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

md(r"""## 1 · Datos observacionales con un confounder""")

code(r"""display(intro("🧪", "1 · El escenario", [
 "<b>De qué se trata.</b> Simulamos un caso típico: hay una variable Z (un confounder, por ejemplo el 'nivel de actividad' de un cliente) que influye en DOS cosas: en la probabilidad de recibir el tratamiento (los más activos tienden a recibir la promo) Y en el resultado (los más activos compran más igual). Como no randomizamos, tratados y controles NO son comparables de entrada.",
 "<b>Qué vas a ver.</b> Conocemos la verdad: el efecto causal del tratamiento es +3. Pero la comparación ingenua (promedio de tratados − promedio de controles) va a dar mucho más, porque mezcla el efecto con la diferencia de Z.",
 "🧮 <b>Dónde mirar.</b> La brecha entre la comparación ingenua y el +3 real: ese exceso es el sesgo del confounder.",
]))
n=4000; TE=3.0
Z = rng.normal(0,1,n)                       # confounder
p_treat = 1/(1+np.exp(-1.5*Z))
T = (rng.random(n) < p_treat).astype(int)   # tratamiento depende de Z
Y = TE*T + 2.0*Z + rng.normal(0,1,n)        # outcome depende de T y de Z
naive = Y[T==1].mean() - Y[T==0].mean()
print(f"Efecto causal REAL: {TE}")
print(f"Comparación ingenua (tratados − controles): {naive:.2f}   <- sesgada")
print(f"Z medio en tratados: {Z[T==1].mean():.2f}   en controles: {Z[T==0].mean():.2f}   <- no son comparables")""")

md(r"""📝 **Lectura.** La comparación ingenua da bastante más que el +3 real. ¿Por qué? Porque los tratados tienen un Z medio más alto que los controles (mirá los números): no son grupos comparables. Parte de su mayor resultado no es el tratamiento, es que ya eran "más activos". Confundir esa diferencia con el efecto es el error causal más común con datos observacionales.""")

md(r"""## 2 · El propensity score empareja a los comparables""")

code(r"""display(intro("⚖️", "2 · Estimar el propensity y emparejar", [
 "<b>De qué se trata.</b> El propensity score es la probabilidad de haber sido tratado, dadas tus características. Lo estimamos con una regresión logística sobre Z. Después, a cada tratado le buscamos el control con propensity más parecido — su 'gemelo estadístico', alguien con la misma chance de haber sido tratado pero que no lo fue.",
 "<b>Qué vas a ver.</b> La distribución del propensity en tratados vs controles (se solapan, por eso el matching es posible) y el efecto estimado tras emparejar: recupera el +3 real, limpiando el sesgo del confounder.",
 "🧮 <b>Dónde mirar.</b> El efecto matcheado vs el ingenuo vs el real. El matching cierra la brecha.",
]))
ps = LogisticRegression().fit(Z.reshape(-1,1), T).predict_proba(Z.reshape(-1,1))[:,1]
ctrl_idx = np.where(T==0)[0]; ctrl_ps = ps[ctrl_idx]
diffs = []
for i in np.where(T==1)[0]:
    j = ctrl_idx[np.argmin(np.abs(ctrl_ps - ps[i]))]   # control más cercano en propensity
    diffs.append(Y[i] - Y[j])
matched = np.mean(diffs)
print(f"Efecto real: {TE}   ingenuo: {naive:.2f}   propensity matching: {matched:.2f}")

fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
a1.hist(ps[T==1],bins=30,alpha=.6,color=BEAM,label="tratados")
a1.hist(ps[T==0],bins=30,alpha=.6,color=CORAL,label="controles")
a1.set_title("Propensity score (se solapan → matching posible)"); a1.set_xlabel("P(tratado | Z)"); a1.legend()
a2.bar(["ingenuo","matching","real"],[naive,matched,TE],color=[CORAL,BEAM,MIST])
for i,v in enumerate([naive,matched,TE]): a2.text(i,v+.05,f"{v:.2f}",ha="center",color=INK,fontweight="bold")
a2.set_title("Efecto estimado"); a2.set_ylabel("efecto")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El propensity matching recupera el **+3 real**, mientras que la comparación ingenua se quedaba en ~5. Al emparejar cada tratado con un control de igual propensity, comparamos personas con la misma "chance de haber sido tratadas" — unas lo fueron y otras no — y esa diferencia sí es atribuible al tratamiento, no al confounder.

El histograma muestra algo clave: las distribuciones de propensity de tratados y controles **se solapan**. Si no se solaparan (hay tratados sin ningún control comparable), el matching no sería válido para esa zona. Siempre chequeá el solapamiento.""")

md(r"""## 3 · Cierre (y el límite que importa)""")

code(r"""display(intro("⚠️", "3 · El para qué y el límite", [
 "<b>El para qué.</b> Cuando no podés hacer un A/B, propensity matching te da una estimación causal honesta a partir de datos observacionales: simulás el experimento emparejando comparables.",
 "<b>El límite, que es enorme.</b> Solo corrige los confounders que MEDISTE (acá, Z). Si existe una variable no observada que influye en tratamiento y resultado, el matching NO la corrige y el sesgo vuelve. No hay forma estadística de descartar ese riesgo: se argumenta con conocimiento del dominio.",
 "<b>El criterio.</b> El método es mecánico; la parte difícil —y que ningún algoritmo hace por vos— es defender que mediste los confounders que importan. Ahí está el juicio causal.",
]))
print(f"Real: {TE}   Ingenuo: {naive:.2f}   Matching: {matched:.2f}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("propensity_matching.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> propensity_matching.ipynb ({len(cells)} celdas)")
