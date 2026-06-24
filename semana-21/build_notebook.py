"""Genera fdr_multiple.ipynb — Semana 21 (jueves) de 'IA sin humo'.
Múltiples comparaciones: probar muchas hipótesis infla los falsos positivos;
Benjamini-Hochberg controla la tasa de falsos descubrimientos. Free (numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🎰 Múltiples comparaciones: por qué 'encontré algo significativo' no alcanza""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🎰 📊 🚦</div>
    <h1 style="margin:.1em 0 0;font-size:1.9em;font-weight:800;text-transform:uppercase;line-height:1.05;letter-spacing:-.02em">
       Múltiples <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">comparaciones</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 21 · Causalidad / Estadística</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Probaste 300 métricas y 30 dieron 'significativas'. ¿Cuántas son reales?
       Con muchos tests, el azar te regala falsos positivos. Lo medimos y lo corregimos.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(17)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; GOLD="#e8b86a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(7.5,4.4),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · Muchos tests, muchos falsos positivos""")

code(r"""display(intro("🎰", "1 · El problema de probar mucho", [
 "<b>De qué se trata.</b> Cada test estadístico con umbral 0.05 tiene un 5% de chance de dar 'significativo' por puro azar, aunque NO haya efecto. Si hacés un test, bien. Si hacés 300 (un dashboard con 300 métricas, un análisis de muchos subgrupos), esperás ~15 falsos positivos solo por azar. 'Encontré algo' deja de significar nada.",
 "<b>Qué vas a ver.</b> Simulamos 300 tests donde el 90% NO tiene efecto real (son nulos). Aplicamos el umbral ingenuo 0.05 y medimos la <b>FDP</b> (false discovery proportion): de los que declaramos significativos, ¿qué fracción son en realidad falsos?",
 "🧮 <b>Dónde mirar.</b> La FDP del enfoque ingenuo. Si es alta, la mayoría de tus 'descubrimientos' son ruido con cara de hallazgo.",
]))
M=300; is_null=rng.random(M)<0.9
p=np.where(is_null, rng.uniform(0,1,M), rng.beta(0.3,8,M))   # nulos U(0,1); efectos -> p chico
naive=p<0.05
fdp_naive=(naive&is_null).sum()/max(naive.sum(),1)
print(f"{M} tests, {is_null.sum()} son nulos (sin efecto real).")
print(f"Umbral ingenuo p<0.05: {naive.sum()} 'significativos', de los cuales {(naive&is_null).sum()} son FALSOS.")
print(f"→ FDP (proporción de descubrimientos falsos): {fdp_naive:.0%}")""")

md(r"""📝 **Lectura.** Una parte enorme de los 'significativos' son falsos: el umbral 0.05, aplicado a cientos de tests, deja entrar el ruido a montones. Por eso "encontré una métrica que mejoró / un subgrupo que responde distinto" no significa nada si probaste 300 cosas: con esa cantidad de tests, encontrar 'algo' está garantizado por azar. Es el mismo mecanismo del p-hacking: buscá suficiente y vas a encontrar.""")

md(r"""## 2 · Benjamini-Hochberg controla la tasa de falsos""")

code(r"""display(intro("🚦", "2 · Corregir por múltiples comparaciones", [
 "<b>De qué se trata.</b> Benjamini-Hochberg (control de FDR) ajusta los umbrales según cuántos tests hiciste, para que la PROPORCIÓN de falsos descubrimientos entre los significativos quede controlada (ej. ≤5%). Es menos conservador que Bonferroni y por eso es el estándar en campos con miles de tests (genómica, A/B a escala).",
 "<b>Qué vas a ver.</b> Promediando muchas simulaciones: el ingenuo deja una FDP alta; BH la baja al ~5% objetivo. Pero ojo —y esto es lo honesto— BH es más estricto, así que pierde algo de PODER (detecta menos efectos reales). No es magia gratis: es un trade-off.",
 "🧮 <b>Dónde mirar.</b> La FDP controlada de BH y el costo en poder. Elegís cuánto falso positivo tolerás.",
]))
def run():
    isn=rng.random(M)<0.9
    pv=np.where(isn, rng.uniform(0,1,M), rng.beta(0.3,8,M))
    nv=pv<0.05
    order=np.argsort(pv); thr=0.05*np.arange(1,M+1)/M; passed=pv[order]<=thr
    bh=np.zeros(M,bool)
    if passed.any(): bh[order[:np.max(np.where(passed)[0])+1]]=True
    fdp=lambda s:(s&isn).sum()/s.sum() if s.sum() else 0
    pw =lambda s:(s&~isn).sum()/max((~isn).sum(),1)
    return fdp(nv),fdp(bh),pw(nv),pw(bh)
R=np.array([run() for _ in range(2000)])
fdp_n,fdp_b,pw_n,pw_b=R.mean(0)
print(f"FDP promedio   ingenuo: {fdp_n:.0%}   Benjamini-Hochberg: {fdp_b:.0%}  (objetivo ≤5%)")
print(f"Poder promedio ingenuo: {pw_n:.0%}   Benjamini-Hochberg: {pw_b:.0%}")
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
a1.bar(["ingenuo","BH"],[fdp_n,fdp_b],color=[CORAL,BEAM]); a1.axhline(0.05,color=GOLD,ls="--")
a1.set_title("FDP (descubrimientos falsos)"); a1.set_ylim(0,.5)
for i,v in enumerate([fdp_n,fdp_b]): a1.text(i,v+.01,f"{v:.0%}",ha="center",color=INK,fontweight="bold")
a2.bar(["ingenuo","BH"],[pw_n,pw_b],color=[MIST,BEAM]); a2.set_title("Poder (detecta efectos reales)"); a2.set_ylim(0,1)
for i,v in enumerate([pw_n,pw_b]): a2.text(i,v+.02,f"{v:.0%}",ha="center",color=INK,fontweight="bold")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Benjamini-Hochberg baja la proporción de falsos descubrimientos al ~5% objetivo, mientras el ingenuo deja ~37%. La corrección funciona. Pero el segundo gráfico cuenta la parte honesta: BH es más estricto, así que detecta menos efectos reales (pierde poder). Ese es el trade-off real de corregir por múltiples comparaciones — no es gratis, es un intercambio entre no engañarte con falsos y no perderte hallazgos reales. Vos elegís el balance según el costo de cada error.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> 'Encontré algo significativo' es facilísimo si buscás suficiente. La pregunta honesta no es '¿hay algo con p<0.05?', es '¿cuántas cosas probé para encontrarlo?'. Con muchos tests, el azar te regala falsos positivos a montones.",
 "<b>Qué hacer.</b> Si probás más de un puñado de hipótesis, corregí: Benjamini-Hochberg (FDR) como default, Bonferroni si necesitás ser muy estricto. Y la regla de oro: definí qué vas a testear ANTES de mirar los datos.",
 "<b>El criterio.</b> Dónde aparece esto: dashboards con decenas de métricas, A/B con muchas métricas secundarias, análisis de subgrupos, feature selection masiva. En todos, sin corrección, vas a 'descubrir' patrones que son ruido. Saberlo —y corregirlo— es lo que separa un hallazgo de una casualidad disfrazada.",
]))
print(f"FDP ingenuo {fdp_n:.0%} -> BH {fdp_b:.0%}  ·  poder {pw_n:.0%} -> {pw_b:.0%}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("fdr_multiple.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> fdr_multiple.ipynb ({len(cells)} celdas)")
