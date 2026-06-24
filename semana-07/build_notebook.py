"""Genera reranking_ndcg.ipynb — Semana 7 (jueves) de 'IA sin humo'.
Mide el efecto del reranking (cross-encoder sobre el top-k del bi-encoder) en el nDCG.
Simulación honesta con numpy (free/runnable, sin descargar modelos)."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🎯 Reranking: cuánto sube el nDCG, medido""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🎯 📈 🔁</div>
    <h1 style="margin:.1em 0 0;font-size:2.3em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       El <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">reranking</span>, medido</h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 7 · Retrieval</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Recuperás los 30 más cercanos rápido, y reordenás esos 30 con un modelo más preciso.
       ¿Cuánto sube la calidad del top-10? Lo medimos con nDCG.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación

> Simulamos relevancias reales y dos rankeadores con distinto ruido (uno rápido y grueso, otro lento y preciso) para medir el efecto del reranking **sin descargar modelos**. En producción el rankeador preciso es un cross-encoder real (p. ej. de sentence-transformers); la dinámica es exactamente esta.""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(3)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
plt.rcParams.update({"figure.figsize":(7,4.2),"figure.facecolor":NIGHT,"axes.facecolor":NIGHT,
  "savefig.facecolor":NIGHT,"axes.edgecolor":LINE,"axes.labelcolor":INK,"text.color":INK,
  "axes.titlecolor":INK,"xtick.color":MIST,"ytick.color":MIST,"axes.grid":True,"grid.color":LINE,
  "grid.alpha":.5,"font.size":11,"legend.framealpha":0})
def intro(emoji,titulo,parrafos):
    ps="".join(f'<p style="margin:0 0 11px;color:#cfe3ef;font-size:.97em;line-height:1.62">{p}</p>' for p in parrafos)
    return HTML(f'''<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;
      background:linear-gradient(135deg,#0a1b27,#102b3c 55%,#15384b);border:1px solid rgba(79,184,232,.22);
      border-left:5px solid #4fb8e8;border-radius:12px;padding:20px 26px;margin:6px 0;color:#eef7fb;box-shadow:0 8px 30px rgba(0,0,0,.35)">
      <div style="font-size:1.3em;font-weight:800;text-transform:uppercase;margin-bottom:12px">{emoji}&nbsp;{titulo}</div>{ps}</div>''')
def dcg(rels): rels=np.asarray(rels,float); return np.sum((2**rels-1)/np.log2(np.arange(2,len(rels)+2)))
def ndcg(order_rels,k=10):
    ideal=sorted(order_rels,reverse=True); d=dcg(order_rels[:k]); i=dcg(ideal[:k]); return d/i if i>0 else 0.0
print("Listo. (nDCG: 1 = ranking perfecto, 0 = pésimo)")""")

md(r"""## 1 · El escenario: relevancia real y dos rankeadores""")

code(r"""display(intro("🧪", "1 · La simulación", [
 "<b>De qué se trata.</b> Para cada consulta hay 100 documentos candidatos, cada uno con una relevancia real (0 = nada que ver, 1-3 = cada vez más relevante; pocos son relevantes, como en la vida real). Tenemos dos rankeadores: el <b>bi-encoder</b> (rápido, lo que usás para buscar) ve la relevancia con MUCHO ruido; el <b>cross-encoder</b> (lento, preciso) la ve con poco ruido.",
 "<b>Qué vas a ver.</b> El bi-encoder rankea los 100 rápido pero impreciso. El cross-encoder sería ideal, pero es carísimo correrlo sobre los 100 (y sobre millones, imposible). La jugada del reranking: correrlo solo sobre los 30 que ya trajo el bi-encoder.",
 "🧮 <b>Dónde mirar.</b> Medimos calidad con nDCG@10: premia tener los documentos más relevantes arriba del top-10. Es la métrica estándar de ranking.",
]))
Q=300; C=100; TOPK=30
def simular():
    rel = np.where(rng.random(C)<0.08, rng.integers(1,4,C), 0).astype(float)
    if rel.max()==0: rel[rng.integers(C)]=2
    bi = rel + rng.normal(0,2.0,C)   # bi-encoder: rápido y ruidoso
    cr = rel + rng.normal(0,0.6,C)   # cross-encoder: lento y preciso
    return rel, bi, cr
print(f"{Q} consultas · {C} candidatos c/u · reranking sobre top-{TOPK}")""")

md(r"""## 2 · Solo bi-encoder: rápido pero impreciso""")

code(r"""display(intro("⚡", "2 · Ranking del bi-encoder solo", [
 "<b>De qué se trata.</b> Medimos el nDCG@10 promedio si usamos solo el bi-encoder: rankeamos los 100 candidatos por su score y nos quedamos con el orden tal cual.",
 "<b>Qué vas a ver.</b> Un nDCG decente pero lejos de 1: el bi-encoder casi nunca pierde del todo a los relevantes (suelen estar en el top-30), pero los ORDENA mal dentro del top. El mejor resultado a veces queda séptimo.",
 "🧮 <b>Dónde mirar.</b> Este número es el punto de partida. La pregunta: ¿cuánto lo mejora reordenar solo los 30 de arriba?",
]))
ndcg_bi=[]
for _ in range(Q):
    rel,bi,cr = simular()
    order = np.argsort(-bi)
    ndcg_bi.append(ndcg(rel[order]))
print(f"nDCG@10 solo bi-encoder: {np.mean(ndcg_bi):.3f}")""")

md(r"""📝 **Lectura.** El bi-encoder da un nDCG@10 medio pero claramente mejorable. Y acá está la clave que justifica todo: el bi-encoder rara vez se pierde a los documentos relevantes *del todo* —suelen entrar en el top-30—, pero los **ordena mal** dentro de esa lista. El problema no es de recall, es de orden fino. Y el orden fino es justo lo que un cross-encoder hace bien.""")

md(r"""## 3 · Con reranking: reordenar el top-30""")

code(r"""display(intro("🔁", "3 · Reranking con el cross-encoder", [
 "<b>De qué se trata.</b> Tomamos los 30 candidatos que trajo el bi-encoder y los reordenamos con el cross-encoder (preciso). El resto queda como estaba. Solo pagamos el modelo caro 30 veces por consulta, no 100 (ni millones).",
 "<b>Qué vas a ver.</b> El nDCG@10 sube fuerte: al reordenar bien los de arriba, los más relevantes suben al top-10. Misma recuperación, mejor orden.",
 "🧮 <b>Dónde mirar.</b> La diferencia entre las dos barras es la calidad que ganás agregando un solo paso de reordenamiento, sin tocar tu base ni tu modelo de embeddings.",
]))
ndcg_re=[]
rng2 = np.random.default_rng(3)  # mismas consultas
for _ in range(Q):
    rel = np.where(rng2.random(C)<0.08, rng2.integers(1,4,C), 0).astype(float)
    if rel.max()==0: rel[rng2.integers(C)]=2
    bi = rel + rng2.normal(0,2.0,C); cr = rel + rng2.normal(0,0.6,C)
    order = np.argsort(-bi)
    cand = order[:TOPK]
    rer = cand[np.argsort(-cr[cand])]
    final = np.concatenate([rer, order[TOPK:]])
    ndcg_re.append(ndcg(rel[final]))
m_bi, m_re = np.mean(ndcg_bi), np.mean(ndcg_re)
print(f"nDCG@10 solo bi-encoder : {m_bi:.3f}")
print(f"nDCG@10 con reranking   : {m_re:.3f}   (mejora {m_re-m_bi:+.3f})")

fig,ax=plt.subplots(figsize=(6,4))
ax.bar(["solo\nbi-encoder","con\nreranking"],[m_bi,m_re],color=[MIST,BEAM])
for i,v in enumerate([m_bi,m_re]): ax.text(i,v+.01,f"{v:.2f}",ha="center",color=INK,fontweight="bold")
ax.set_ylim(0,1.05); ax.set_ylabel("nDCG@10"); ax.set_title("Calidad del top-10: el reranking sube fuerte")
plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** El nDCG@10 pega un salto grande con el reranking. No cambiamos la base de datos, ni el modelo de embeddings, ni recuperamos más documentos: solo **reordenamos los 30 de arriba** con un modelo más preciso. Los relevantes que estaban en el puesto 7 u 8 suben al top-10.

Y lo barato es el truco: el cross-encoder es lento, pero corre solo sobre 30 ítems por consulta (no sobre los millones de tu corpus). Recuperás generoso con el rápido, afinás el orden con el preciso. Esa es la mejor relación costo/beneficio del retrieval.""")

md(r"""## 4 · Cierre""")

code(r"""display(intro("💡", "4 · El para qué", [
 "<b>El punto.</b> Si tu buscador semántico 'casi trae lo bueno pero mal ordenado', el problema no es el modelo de embeddings: es la falta de un reranker. Cambiar a un embedding más caro suele mover menos la aguja que agregar este paso.",
 "<b>La regla.</b> Recuperá generoso (top-30/50 con el bi-encoder rápido), reordená preciso (cross-encoder sobre esos pocos), devolvé el top-10. Medí con nDCG@k, no a ojo.",
 "<b>El criterio.</b> Entender que recuperación y ordenamiento son dos problemas distintos —y que cada uno tiene su herramienta— es lo que te deja mejorar el buscador sin gastar de más. No es un modelo más grande; es la arquitectura correcta.",
]))
print(f"nDCG@10 solo bi-encoder : {m_bi:.3f}")
print(f"nDCG@10 con reranking   : {m_re:.3f}")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("reranking_ndcg.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> reranking_ndcg.ipynb ({len(cells)} celdas)")
