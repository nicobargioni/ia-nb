"""Genera retrieval_hibrido.ipynb — Semana 1 (jueves) de la serie 'IA sin humo'.
Demuestra por qué la búsqueda densa (embeddings) sola pierde con nombres/códigos,
y cómo la búsqueda híbrida (dense + BM25, fusionada con RRF) gana en ambos mundos.
Estética 'noche de estadio' consistente con la serie. Free / runnable en Colab."""
import json

cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# 🔎 Búsqueda híbrida: por qué los vectores solos pierden""")

code(r"""from IPython.display import display, HTML
display(HTML('''
<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;border-radius:16px;overflow:hidden;
            box-shadow:0 14px 56px rgba(0,0,0,.55);border:1px solid rgba(79,184,232,.28);margin:6px 0">
  <div style="padding:44px 30px;text-align:center;color:#eef7fb;
       background:radial-gradient(120% 90% at 12% -12%, rgba(79,184,232,.34), transparent 52%),
                  radial-gradient(90% 80% at 90% 120%, rgba(38,86,116,.5), transparent 60%),
                  linear-gradient(160deg,#08161f,#0a1b27 55%,#061019)">
    <div style="font-size:2.1em;filter:drop-shadow(0 0 12px rgba(124,200,238,.7))">🔎 🧲 ⚡</div>
    <h1 style="margin:.1em 0 0;font-size:2.4em;font-weight:800;text-transform:uppercase;line-height:1;letter-spacing:-.02em">
       Búsqueda <span style="color:#4fb8e8;text-shadow:0 0 26px rgba(79,184,232,.7)">híbrida</span></h1>
    <div style="font-size:.95em;color:#7cc8ee;font-weight:700;letter-spacing:.2em;text-transform:uppercase;margin-top:10px">
       IA sin humo · Semana 1 · Retrieval</div>
    <div style="margin-top:14px;font-size:.92em;color:#bcdcec;max-width:560px;margin-left:auto;margin-right:auto">
       Tus embeddings son geniales con el significado y malísimos con los nombres y códigos.
       La solución no es un modelo más caro: es combinar dos buscadores.</div>
  </div>
</div>
<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&display=swap');</style>
'''))""")

md(r"""## 0 · Preparación""")

code(r"""!pip -q install sentence-transformers >/dev/null 2>&1
print('✅ Listo')""")

code(r"""import numpy as np, re
from collections import Counter
from sentence_transformers import SentenceTransformer
from IPython.display import display, HTML

np.random.seed(0)

def intro(emoji, titulo, parrafos):
    ps="".join(f'<p style="margin:0 0 11px;color:#cfe3ef;font-size:.97em;line-height:1.62">{p}</p>' for p in parrafos)
    return HTML(f'''<div style="font-family:Montserrat,system-ui,sans-serif;width:100%;box-sizing:border-box;
      background:linear-gradient(135deg,#0a1b27,#102b3c 55%,#15384b);border:1px solid rgba(79,184,232,.22);
      border-left:5px solid #4fb8e8;border-radius:12px;padding:20px 26px;margin:6px 0;color:#eef7fb;box-shadow:0 8px 30px rgba(0,0,0,.35)">
      <div style="font-size:1.3em;font-weight:800;text-transform:uppercase;margin-bottom:12px">{emoji}&nbsp;{titulo}</div>{ps}</div>''')

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def embed(t): return model.encode(list(t), normalize_embeddings=True, show_progress_bar=False)
print("Modelo cargado.")""")

md(r"""## 1 · El corpus: un catálogo con nombres y códigos""")

code(r"""display(intro("📦", "1 · El corpus", [
 "<b>De qué se trata.</b> Armamos un mini catálogo de productos de audio. Lo importante: cada ítem tiene un <b>código de modelo</b> (tipo X-450, ZX-900) además de su descripción. Esos códigos son justo donde la búsqueda por significado se va a tropezar.",
 "<b>Qué vas a ver.</b> 14 productos como strings. Algunos comparten palabras (auriculares, parlante), otros se distinguen sólo por el código. Es el escenario típico de un e-commerce o un soporte técnico.",
 "🧮 <b>Dónde mirar.</b> Más abajo vamos a buscar de tres formas sobre este mismo corpus y comparar quién encuentra qué.",
]))

CORPUS = [
 "Auriculares inalámbricos X-450 con cancelación activa de ruido y 30h de batería.",
 "Auriculares inalámbricos X-460 edición pro, cancelación de ruido mejorada.",
 "Parlante portátil ZX-900 resistente al agua, sonido 360 grados.",
 "Parlante de estantería HiFi para el living, madera y 120W.",
 "Micrófono USB para streaming y podcast con filtro anti-pop.",
 "Placa de sonido externa para grabación profesional, baja latencia.",
 "Auriculares con cable para estudio, respuesta plana, modelo ST-7.",
 "Cargador rápido USB-C de 65W compatible con notebooks.",
 "Cable de audio jack 3.5mm trenzado de 2 metros.",
 "Soporte de brazo articulado para micrófono de escritorio.",
 "Parlante inteligente con asistente de voz integrado.",
 "Auriculares deportivos resistentes al sudor, ajuste seguro para correr.",
 "Interfaz de audio ZX-900 para músicos, dos entradas XLR.",
 "Funda de transporte rígida para auriculares plegables.",
]
print(len(CORPUS), "productos")""")

md(r"""## 2 · Búsqueda densa (embeddings): genial con el significado""")

code(r"""display(intro("🧠", "2 · Búsqueda densa (embeddings)", [
 "<b>De qué se trata.</b> La búsqueda semántica clásica: embebemos cada producto y la consulta, y devolvemos los más parecidos por coseno. Es la que brilla cuando buscás por <i>idea</i>, no por palabra exacta.",
 "<b>Qué vas a ver.</b> Dos consultas. Con 'algo para escuchar música sin molestar a los demás' va a funcionar perfecto (entiende que = cancelación de ruido). Con el código exacto 'ZX-900' va a empezar a fallar.",
 "🧮 <b>Dónde mirar.</b> Es un producto punto consulta-vs-corpus y un orden. Mirá el segundo resultado: el código exacto no siempre queda primero.",
]))

E = embed(CORPUS)
def dense(query, k=4):
    q = embed([query])[0]
    sims = E @ q
    idx = np.argsort(-sims)[:k]
    return [(i, float(sims[i])) for i in idx]

def mostrar(query, ranking):
    print(f"🔎 '{query}'")
    for i,s in ranking: print(f"   {s:5.3f}  {CORPUS[i]}")
    print()

mostrar("algo para escuchar música sin molestar a los demás", dense("algo para escuchar música sin molestar a los demás"))
mostrar("ZX-900", dense("ZX-900"))""")

md(r"""📝 **Lectura.** Con la consulta por *significado* la búsqueda densa la rompe: entiende que "sin molestar a los demás" = cancelación de ruido, sin que aparezca esa palabra. Pero con **"ZX-900"** se nota la grieta: el embedding no "lee" el código como un token exacto, lo diluye entre productos parecidos, y el match exacto puede no quedar primero. Los modelos de embeddings son malísimos con identificadores arbitrarios (SKUs, nombres propios, versiones), porque no tienen significado semántico que capturar — son cadenas.""")

md(r"""## 3 · BM25: genial con las palabras exactas""")

code(r"""display(intro("🔤", "3 · BM25 (léxico)", [
 "<b>De qué se trata.</b> El buscador 'de toda la vida': cuenta coincidencias de palabras, pesando las raras (un código aparece en 1 solo producto, así que pesa muchísimo). No entiende significado, pero clava lo exacto.",
 "<b>Qué vas a ver.</b> Con 'ZX-900' lo pone primero sin dudar. Pero con la consulta semántica ('sin molestar a los demás') se pierde, porque ninguna de esas palabras está en los productos.",
 "🧮 <b>Dónde mirar.</b> Implementamos BM25 a mano (TF, IDF y normalización por longitud). Es álgebra simple sobre conteos de palabras.",
]))

def tok(s): return re.findall(r"[a-záéíóúñ0-9\-]+", s.lower())
docs_tok = [tok(d) for d in CORPUS]
N = len(CORPUS); avgdl = np.mean([len(d) for d in docs_tok])
df = Counter(t for d in docs_tok for t in set(d))
idf = {t: np.log(1 + (N - n + 0.5)/(n + 0.5)) for t,n in df.items()}

def bm25(query, k=4, k1=1.5, b=0.75):
    q = tok(query); scores = np.zeros(N)
    for i,d in enumerate(docs_tok):
        tf = Counter(d); dl = len(d)
        for t in q:
            if t not in tf: continue
            s = idf.get(t,0) * (tf[t]*(k1+1)) / (tf[t] + k1*(1-b+b*dl/avgdl))
            scores[i] += s
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]

mostrar("ZX-900", bm25("ZX-900"))
mostrar("algo para escuchar música sin molestar a los demás", bm25("algo para escuchar música sin molestar a los demás"))""")

md(r"""📝 **Lectura.** El espejo exacto del caso anterior. BM25 pone **"ZX-900" primero** sin dudar, porque el código es una palabra rarísima (alto IDF) que aparece en pocos productos. Pero con la consulta por significado **se queda en cero**: ninguna de esas palabras figura literalmente en el catálogo, así que no tiene de dónde agarrarse. Densa y BM25 fallan en mundos opuestos — y ahí está la oportunidad.""")

md(r"""## 4 · Búsqueda híbrida (RRF): lo mejor de los dos""")

code(r"""display(intro("🧲", "4 · Fusión híbrida con RRF", [
 "<b>De qué se trata.</b> Si cada buscador gana en un mundo distinto, combinémoslos. Reciprocal Rank Fusion (RRF) no mezcla los scores (que están en escalas distintas e incomparables): mezcla los <b>rankings</b>. Cada documento suma 1/(k+puesto) según dónde quedó en cada lista.",
 "<b>Qué vas a ver.</b> La búsqueda híbrida resuelve BIEN las DOS consultas: el código exacto y la intención semántica. Sin tocar el modelo, sin reentrenar nada.",
 "🧮 <b>Dónde mirar.</b> RRF es una suma de recíprocos de rangos. Robusto justamente porque ignora la magnitud de los scores y sólo mira el orden.",
]))

def hybrid(query, k=4, rrf_k=60):
    d = dense(query, k=N); bm = bm25(query, k=N)
    rank = {}
    for r,(i,_) in enumerate(d):  rank[i] = rank.get(i,0) + 1/(rrf_k + r)
    for r,(i,_) in enumerate(bm): rank[i] = rank.get(i,0) + 1/(rrf_k + r)
    idx = sorted(rank, key=lambda i:-rank[i])[:k]
    return [(i, rank[i]) for i in idx]

mostrar("ZX-900", hybrid("ZX-900"))
mostrar("algo para escuchar música sin molestar a los demás", hybrid("algo para escuchar música sin molestar a los demás"))""")

md(r"""📝 **Lectura.** La híbrida resuelve **las dos** consultas: clava el código exacto (gracias a BM25) y entiende la intención (gracias a los embeddings). Y lo hace con **RRF**, que fusiona *rankings* y no *scores* — clave, porque el coseno (0 a 1) y el score BM25 (sin tope) no son comparables; sumarlos directo sería un error. RRF sólo mira "¿en qué puesto quedó cada uno?", que sí es comparable.

**El para qué:** si tu buscador trae basura, antes de pagar un modelo de embeddings más caro, probá **híbrido**. Es la mejora con mejor relación costo/beneficio en retrieval del mundo real, donde siempre hay nombres, SKUs y códigos que el significado no captura.""")

md(r"""## 🏁 Cierre

| | Significado ("sin molestar") | Código exacto ("ZX-900") |
|---|---|---|
| **Densa (embeddings)** | ✅ excelente | ❌ falla |
| **BM25 (léxico)** | ❌ falla | ✅ excelente |
| **Híbrida (RRF)** | ✅ | ✅ |

Los embeddings no son la respuesta a todo. **Combinar** un buscador semántico con uno léxico —dos herramientas simples— le gana a tirarle un modelo más grande encima. Ahí está el criterio.
""")

def to_source(s): return s.splitlines(keepends=True)
nb={"cells":[({"cell_type":"markdown","metadata":{},"source":to_source(x)} if t=="markdown"
  else {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":to_source(x)}) for (t,x) in cells],
  "metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},
  "nbformat":4,"nbformat_minor":5}
with open("retrieval_hibrido.ipynb","w",encoding="utf-8") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"OK -> retrieval_hibrido.ipynb ({len(cells)} celdas)")
