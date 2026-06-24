"""Genera rag_eval.ipynb — Semana 13 (jueves) de 'IA sin humo'.
Evaluar un RAG: armar un golden set y medir el retrieval (recall@k, hit@k, MRR).
Free/runnable (numpy). La parte de generación (faithfulness) se menciona: necesita LLM-judge."""
import json
cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 📐 Evaluar tu RAG: golden set y métricas de retrieval""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">📐 ✅ 🔍</div>
    <h1 style="margin:.1em 0 0;font-size:2.05em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Evaluar tu <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">RAG</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 13 · Retrieval</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       'El chatbot anda mejor' no es una métrica. Armamos un golden set y medimos
       el retrieval (recall@k, hit@k, MRR) para mejorar con evidencia, no con onda.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, HTML
rng = np.random.default_rng(8)
NIGHT="#0a1b27"; INK="#bcdcec"; BEAM="#4fb8e8"; CELESTE="#8fc0e8"; MIST="#6f93a8"; CORAL="#e88a8a"; LINE="#15303f"
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
def recall_at_k(order,rel,k): return len(set(order[:k])&rel)/len(rel) if rel else 0.0
def hit_at_k(order,rel,k):    return 1.0 if set(order[:k])&rel else 0.0
def mrr(order,rel):
    for r,d in enumerate(order):
        if d in rel: return 1.0/(r+1)
    return 0.0
print("Listo. Métricas: recall@k, hit@k, MRR.")""")

md(r"""## 1 · El golden set: la base de toda evaluación""")

code(r"""display(intro("🗂️", "1 · Armar el golden set", [
 "<b>De qué se trata.</b> Para evaluar un RAG no alcanza con leer 3 respuestas y opinar. Necesitás un GOLDEN SET: un conjunto de consultas reales para las que sabés qué documentos SON relevantes (la verdad). Eso te deja medir objetivamente si el retrieval trae lo correcto.",
 "<b>Qué vas a ver.</b> Simulamos un golden set: 300 consultas, cada una con sus documentos relevantes marcados, y un retriever (ruidoso, como cualquiera real) que rankea los candidatos. En tu caso real, este set lo armás vos con ejemplos de tu dominio.",
 "🧮 <b>Dónde mirar.</b> El golden set es trabajo, sí. Pero es lo que convierte 'me parece que anda mejor' en un número que podés comparar entre versiones.",
]))
Q=300; C=80
def gen_eval():
    nrel = rng.integers(1,4)
    rel = set(int(d) for d in rng.choice(C, nrel, replace=False))
    truescore = np.array([2.0 if d in rel else 0.0 for d in range(C)])
    obs = truescore + rng.normal(0,1.3,C)         # retriever ruidoso
    order = list(np.argsort(-obs))
    return order, rel
eval_set = [gen_eval() for _ in range(Q)]
print(f"Golden set: {Q} consultas, {C} candidatos por consulta, con relevancia conocida.")""")

md(r"""## 2 · Medir el retrieval: recall@k, hit@k, MRR""")

code(r"""display(intro("📏", "2 · Las métricas que importan", [
 "<b>De qué se trata.</b> Medimos tres cosas a distintos valores de k (cuántos documentos recupera el sistema). <b>Recall@k:</b> qué fracción de los relevantes entró en el top-k. <b>Hit@k:</b> con qué frecuencia AL MENOS un relevante entró. <b>MRR:</b> qué tan arriba aparece el primer relevante.",
 "<b>Qué vas a ver.</b> Recall@k y hit@k suben con k (recuperás más, capturás más relevantes). El MRR no depende de k: mide la calidad del orden, no la cantidad.",
 "🧮 <b>Dónde mirar.</b> Estas métricas te dicen DÓNDE está el problema: si recall@20 es bajo, ni siquiera traés los documentos buenos (problema de embeddings/chunking). Si recall@20 es alto pero recall@3 bajo, traés lo bueno pero mal ordenado (problema de reranking).",
]))
ks=[1,3,5,10,20]
rec=[np.mean([recall_at_k(o,r,k) for o,r in eval_set]) for k in ks]
hit=[np.mean([hit_at_k(o,r,k)    for o,r in eval_set]) for k in ks]
mr = np.mean([mrr(o,r) for o,r in eval_set])
for k,rr,hh in zip(ks,rec,hit): print(f"k={k:2d}  recall@k={rr:.2f}  hit@k={hh:.2f}")
print(f"MRR (independiente de k): {mr:.2f}")
plt.figure()
plt.plot(ks,rec,"o-",color=BEAM,label="recall@k"); plt.plot(ks,hit,"o-",color=CELESTE,label="hit@k")
plt.xlabel("k (documentos recuperados)"); plt.ylabel("métrica"); plt.ylim(0,1)
plt.title("Calidad del retrieval vs cuántos documentos traés"); plt.legend(); plt.tight_layout(); plt.show()""")

md(r"""📝 **Lectura.** Ahora tenés números, no impresiones. Recall@k y hit@k suben con k: cuantos más documentos traés, más relevantes capturás (por eso conviene "recuperar generoso" y después rerankear). El MRR, constante, te habla del orden.

Lo más útil es el **diagnóstico**: si recall@20 ya es bajo, el problema es de recuperación (embeddings, chunking) y ningún reranker lo arregla. Si recall@20 es alto pero recall@3 es bajo, traés lo bueno pero mal ordenado → ahí sí un reranker rinde. La métrica no solo te dice si andás bien: te dice dónde mirar.""")

md(r"""## 3 · Cierre""")

code(r"""display(intro("💡", "3 · El para qué", [
 "<b>El punto.</b> Sin golden set y métricas, cada cambio en tu RAG es una apuesta a ciegas. Con ellos, comparás versiones objetivamente: 'el cambio subió recall@5 de 0.49 a 0.61' en vez de 'parece que anda mejor'.",
 "<b>Lo que falta (y necesita LLM-judge).</b> Acá medimos el RETRIEVAL. La otra mitad —si la RESPUESTA generada es fiel al contexto (faithfulness), si responde la pregunta (answer relevance)— se mide con un LLM como juez. Librerías como RAGAS automatizan eso, pero requieren una API key.",
 "<b>El criterio.</b> 'Si no lo podés medir, no lo podés mejorar' no es una frase de cuaderno: es la diferencia entre iterar con evidencia y dar vueltas con onda. Empezá por el golden set; es lo más aburrido y lo más importante.",
]))
print(f"recall@5={rec[2]:.2f}  hit@5={hit[2]:.2f}  MRR={mr:.2f}")
print("\n— Serie 'IA sin humo' · github.com/nicobargioni/ia-nb")""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("rag_eval.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> rag_eval.ipynb ({len(cells)} celdas)")
