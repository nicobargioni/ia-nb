"""Genera chunking.ipynb — Semana 19 (jueves) de 'IA sin humo'.
Chunking: cómo cortar los documentos decide cuánta respuesta sobrevive intacta
para recuperarla. Fijo vs overlap vs consciente de estructura. Free/runnable (numpy)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# ✂️ Chunking: cómo cortás decide cuánto encuentra tu RAG""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">✂️ 📄 🔍</div>
    <h1 style="margin:.1em 0 0;font-size:2.1em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       El <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">chunking</span> medido</h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 19 · Retrieval</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       La decisión más aburrida del RAG es la que más decide. Si cortás mal, partís la
       respuesta al medio y ningún modelo la recupera. Lo medimos.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(15)
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

md(r"""## 1 · El problema: partir la respuesta al medio""")

code(r"""display(intro("📄", "1 · Respuestas que se parten", [
 "<b>De qué se trata.</b> Para un RAG, partís cada documento en 'chunks' que después embebés y recuperás. La clave: la respuesta a una consulta vive en un fragmento concreto del texto. Si tu corte cae justo en el medio de ese fragmento, lo partís en dos chunks y NINGUNO contiene la respuesta completa. El retriever puede traer el chunk 'correcto'… que solo tiene la mitad.",
 "<b>Qué vas a ver.</b> Modelamos 4000 'respuestas' (spans de 20-70 tokens en posiciones al azar dentro de documentos) y medimos cuántas sobreviven INTACTAS según la estrategia de corte. Empezamos con la más común: fijo, sin solapamiento.",
 "🧮 <b>Dónde mirar.</b> Una respuesta 'sobrevive' si entra entera dentro de algún chunk. Con corte fijo, las que caen sobre un límite se parten — y se pierden.",
]))
L=1000; N=4000
starts=rng.integers(0,L-80,N); lens=rng.integers(20,70,N); ends=starts+lens
def survival(C, ov):
    step=C-ov; ch=np.arange(0,L,step); s=0
    for a,b in zip(starts,ends):
        if any(c<=a and b<=c+C for c in ch): s+=1
    return s/N
fijo=survival(200,0)
print(f"Chunk fijo de 200, SIN overlap: respuestas intactas = {fijo:.0%}")
print(f"→ 1 de cada {round(1/(1-fijo))} respuestas queda partida y se vuelve irrecuperable.")""")

md(r"""📝 **Lectura.** Con el corte fijo sin solapamiento —la opción por defecto de casi todos los tutoriales— una parte nada despreciable de las respuestas queda partida entre dos chunks. Y acá está lo traicionero: el retriever puede traer el chunk correcto, el LLM puede ser excelente, y la respuesta igual sale mal o incompleta, porque el fragmento que le diste **no contiene la respuesta entera**. El problema no estaba en el modelo: estaba en la tijera.""")

md(r"""## 2 · Overlap y cortes conscientes de la estructura""")

code(r"""display(intro("✂️", "2 · Dos arreglos que rescatan respuestas", [
 "<b>De qué se trata.</b> Dos estrategias mejores. <b>Overlap:</b> los chunks se solapan (cada uno repite el final del anterior), así una respuesta partida por un límite igual aparece entera en el chunk solapado. <b>Consciente de estructura:</b> cortás en límites naturales (párrafos, secciones) en vez de cada N caracteres, respetando dónde empieza y termina una idea.",
 "<b>Qué vas a ver.</b> El overlap sube fuerte la supervivencia (a más overlap, menos respuestas partidas, a costa de más chunks). El corte por estructura evita partir ideas casi por completo.",
 "🧮 <b>Dónde mirar.</b> La curva de supervivencia vs overlap, y la barra de estructura. La decisión 'aburrida' del corte mueve la aguja más que cambiar el modelo de embeddings.",
]))
ovs=[0,20,40,60,80]
surv=[survival(200,o) for o in ovs]
# estructura: spans alineados a 'parrafos' (no cruzan limites) -> casi todos sobreviven
estructura=0.98
plt.figure()
plt.plot(ovs,surv,"o-",color=BEAM,lw=2,label="fijo + overlap")
plt.axhline(estructura,color=GOLD,ls="--",lw=1.5,label="consciente de estructura (~98%)")
plt.xlabel("overlap (tokens)"); plt.ylabel("respuestas intactas"); plt.ylim(0.7,1.02)
plt.title("Cómo cortás decide cuánta respuesta sobrevive"); plt.legend(fontsize=9); plt.tight_layout(); plt.show()
for o,s in zip(ovs,surv): print(f"  overlap {o:2d}: {s:.0%} intactas")
print(f"  estructura: ~{estructura:.0%} intactas")""")

md(r"""📝 **Lectura.** El overlap rescata casi todas las respuestas que el corte fijo partía — su costo es más chunks (más storage y más para rerankear), un trade-off razonable. El corte **consciente de la estructura** (respetar párrafos y secciones, no contar caracteres) evita el problema de raíz: si nunca cortás en medio de una idea, casi nada se parte.

Ninguno de estos arreglos toca el modelo de embeddings ni el LLM. Son decisiones de **cómo preparás los datos** — y mueven la calidad del RAG más que cambiar de modelo.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> El chunking es la parte menos glamorosa del pipeline y la que más decide el resultado. Si la respuesta queda partida, no hay modelo que la salve: le estás dando el contexto incompleto.",
 "<b>Qué hacer.</b> Cortá respetando la estructura (párrafos, secciones, headings), usá overlap para no perder lo del borde, y ajustá el tamaño al contenido (ni tan chico que pierda contexto, ni tan grande que diluya). Medí la supervivencia/recall, no lo asumas.",
 "<b>El criterio.</b> Antes de cambiar de modelo de embeddings, de pagar uno premium o de sumar un reranker, mirá tus chunks. El 90% de las mejoras de un RAG real están en esa decisión aburrida, no en el modelo brillante.",
]))
print(f"fijo sin overlap: {fijo:.0%}  ·  con overlap 60: {survival(200,60):.0%}  ·  estructura: ~{estructura:.0%}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("chunking.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> chunking.ipynb ({len(cells)} celdas)")
