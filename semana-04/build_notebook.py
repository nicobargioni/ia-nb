"""Genera structured_output.ipynb — Semana 4 (jueves) de 'IA sin humo'.
Constrained decoding / structured output: garantizar JSON válido POR CONSTRUCCIÓN,
enmascarando los tokens que romperían el formato. Simulación free/runnable (numpy+json),
sin API ni modelo pesado: demuestra el MECANISMO real."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🔒 Constrained decoding: JSON válido por construcción""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🔒 { } ✅</div>
    <h1 style="margin:.1em 0 0;font-size:2.3em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Structured <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">output</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 4 · LLMs en producción</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Pedirle JSON a un LLM y rezar que salga bien no es una estrategia. Acá vemos el
       mecanismo que lo hace IMPOSIBLE de romper — enmascarar los tokens inválidos.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación

> Este notebook **simula** el mecanismo de constrained decoding con numpy, sin API ni modelo pesado, para que veas exactamente *cómo* funciona. En producción lo dan librerías como Outlines, las gramáticas de llama.cpp, o el "structured output" de las APIs (que requieren su key).""")

code(r"""import numpy as np, json, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(0)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(7.5,4.2),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
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

md(r"""## 1 · Un mini-modelo que genera token por token""")

code(r"""display(intro("🎲", "1 · El modelo y el formato que queremos", [
 "<b>De qué se trata.</b> Un LLM genera de a un token por vez, eligiendo de una distribución sobre todo su vocabulario. Acá armamos un modelo de juguete que arma un JSON tipo <code>{\"name\": ..., \"age\": ...}</code> en 9 slots. En cada slot el modelo es 'mayormente correcto' pero deja un 25% de probabilidad repartida en tokens que romperían el formato — igual que un LLM real, que a veces se desvía.",
 "<b>Qué vas a ver.</b> El vocabulario, el esquema objetivo (qué token es válido en cada posición) y la distribución del modelo. Es el escenario donde el JSON 'casi siempre' sale bien… pero no siempre.",
 "🧮 <b>Dónde mirar.</b> Fijate que en los slots de valor (nombre, edad) hay varias opciones válidas: el modelo SÍ debe poder elegir entre ellas. Lo que no queremos es que rompa la estructura.",
]))
VOCAB = ['{','}','"name"','"age"',':',',','"Ana"','"Beto"','"Caro"','25','30','42','XX']
idx = {t:i for i,t in enumerate(VOCAB)}
SLOTS = [['{'],['"name"'],[':'],['"Ana"','"Beto"','"Caro"'],[','],['"age"'],[':'],['25','30','42'],['}']]
def dist_for(slot, leak=0.25):
    p = np.zeros(len(VOCAB)); allowed = SLOTS[slot]
    for t in allowed: p[idx[t]] += (1-leak)/len(allowed)
    dis = [t for t in VOCAB if t not in allowed]
    for t in dis: p[idx[t]] += leak/len(dis)
    return p
print("Vocabulario:", VOCAB)
print("Esquema objetivo: { \"name\": <nombre> , \"age\": <num> }")
print("El modelo acierta ~75% por slot y filtra ~25% a tokens que rompen el JSON.")""")

md(r"""## 2 · Sin restricción: el JSON se rompe seguido""")

code(r"""display(intro("⚠️", "2 · Generación SIN restricción", [
 "<b>De qué se trata.</b> Dejamos que el modelo samplee libre, como cuando le pedís JSON 'por las buenas'. Generamos miles de salidas, las parseamos con json.loads y medimos cuántas son JSON válido con los campos esperados.",
 "<b>Qué vas a ver.</b> El porcentaje de salidas válidas va a ser bajo: basta que UN slot estructural se desvíe para romper todo. Y a escala, ese porcentaje de fallo son incidentes en producción.",
 "🧮 <b>Dónde mirar.</b> El número de validez sin restricción. Spoiler: da para el susto.",
]))
def generar(constrained):
    toks = []
    for s in range(len(SLOTS)):
        p = dist_for(s).copy()
        if constrained:
            mask = np.zeros(len(VOCAB))
            for t in SLOTS[s]: mask[idx[t]] = 1.0
            p = p*mask; p = p/p.sum()
        toks.append(VOCAB[rng.choice(len(VOCAB), p=p)])
    return ''.join(toks)
def es_valido(s):
    try:
        o = json.loads(s); return isinstance(o,dict) and 'name' in o and 'age' in o
    except Exception: return False

N = 4000
sin = [generar(False) for _ in range(N)]
val_sin = np.mean([es_valido(s) for s in sin])
print(f"Salidas válidas SIN restricción: {val_sin:.1%}")
print("Ejemplos de salidas rotas:")
for s in sin:
    if not es_valido(s): print("   ", s[:50]);
    if sin.index(s) > 0 and len([x for x in sin[:sin.index(s)+1] if not es_valido(x)])>=3: break""")

md(r"""📝 **Lectura.** La validez sin restricción es baja: alcanza con que **un solo slot estructural** se desvíe para que `json.loads` explote. Mirá las salidas rotas: una llave de más, un orden cambiado, un token basura. En una demo esto se ignora; en producción, a miles de llamadas, es un incidente recurrente que llena tu código de `try/except` defensivos. Pedir el formato "por prompt" no garantiza nada.""")

md(r"""## 3 · Con restricción: imposible romperlo""")

code(r"""display(intro("🔒", "3 · Constrained decoding", [
 "<b>De qué se trata.</b> La idea es simple y poderosa: en cada slot, antes de samplear, ponemos en CERO la probabilidad de todo token que rompería el formato (lo enmascaramos) y renormalizamos. El modelo solo puede elegir entre los tokens que mantienen el JSON válido — pero sigue eligiendo libremente entre las opciones de valor.",
 "<b>Qué vas a ver.</b> La validez salta a 100%. No es que 'mejora': es que generar algo inválido se vuelve imposible por construcción. Y las salidas siguen teniendo variedad en los valores.",
 "🧮 <b>Dónde mirar.</b> Comparación de validez sin vs con restricción. Una máscara sobre la distribución, eso es todo el truco.",
]))
con = [generar(True) for _ in range(N)]
val_con = np.mean([es_valido(s) for s in con])
print(f"Salidas válidas CON restricción: {val_con:.1%}")
print("Ejemplos (todas válidas, con variedad):")
for s in con[:4]: print("   ", s)

fig,ax=plt.subplots()
ax.bar(["sin restricción","con restricción"],[val_sin,val_con],color=[CORAL,BEAM])
for i,v in enumerate([val_sin,val_con]): ax.text(i,v+.02,f"{v:.0%}",ha="center",color=INK,fontweight="bold")
ax.set_ylim(0,1.08); ax.set_title("Salidas JSON válidas"); ax.set_ylabel("% válido")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Con restricción, **el 100% de las salidas son JSON válido** — y conservan variedad en los valores (Ana/Beto/Caro, 25/30/42). No validamos a posteriori ni reintentamos: hicimos que lo inválido sea **imposible de generar**. Ese es el corazón de constrained decoding: enmascarar, en cada paso, los tokens que violarían la gramática/esquema.

Lo que simulamos a mano lo dan, sobre LLMs reales, librerías como **Outlines**, las **gramáticas GBNF de llama.cpp**, o el **structured output** nativo de las APIs (estos últimos requieren tu key). El mecanismo es exactamente este.""")

md(r"""## 4 · Cierre""")

code(r"""display(intro("💡", "4 · El para qué", [
 "<b>El punto.</b> 'Reintentá si el JSON es inválido' gasta tokens, suma latencia y no garantiza nada. Constrained decoding cambia el problema: en vez de detectar y corregir errores, los hace imposibles.",
 "<b>Cuándo usarlo.</b> Siempre que la salida del LLM alimente código: extracción a JSON, tool/function calling, clasificación con etiquetas cerradas, generación de SQL con una gramática. Ahí la robustez no es opcional.",
 "<b>El criterio.</b> La confiabilidad de un sistema no se pide por prompt, se garantiza por construcción. Es la diferencia entre una demo que anda 'casi siempre' y un sistema que no se rompe. No hace falta un modelo más grande — hace falta entender cómo decodifica.",
]))
print("Resumen:")
print(f"  sin restricción : {val_sin:.1%} válido")
print(f"  con restricción : {val_con:.1%} válido")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("structured_output.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> structured_output.ipynb ({len(cells)} celdas)")
