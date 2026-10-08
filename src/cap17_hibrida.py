"""capitulo 17: consulta hibrida.

estudia, en numpy, como se combinan las senales de una busqueda: el filtro
estructurado sobre los metadatos, la senal lexica (BM25, del cap. 10) y una
senal densa, y la reordenacion en dos etapas. el corpus es SINTETICO y
CONTROLADO: esta construido para que lexico y denso se complementen (las
consultas semanticas usan palabras de tema reservadas que los documentos no
contienen, y cada documento lleva un termino raro unico que solo BM25 capta).
el resultado de complementariedad esta, por tanto, incorporado al generador: el
modulo verifica mecanismos, no estima cuanto gana un hibrido en datos reales.
la senal densa es tematica y sintetica (centroide del tema mas ruido), no la
salida de un codificador entrenado.

medidas de calidad: recall@k de relevancia, |R ∩ top_k| / |R|, y su version
acotada (capped recall), |R ∩ top_k| / min(|R|, k), que vale 1 cuando los k
primeros son todos relevantes aunque haya mas de k; y el MRR.

mide:
  1. complementariedad: recall acotado de BM25, denso e hibrido (RRF) en
     consultas lexicas, semanticas y su mezcla.
  2. fusion lineal: recall acotado segun el peso alpha, con normalizacion
     min-max y z-score, sobre las MISMAS consultas que (1).
  3. senal debil: la misma mezcla con una senal densa cada vez mas ruidosa.
  4. dos etapas: MRR de la recuperacion densa frente a un reordenador
     sintetico (BM25 normalizado mas 0,3 del denso) sobre los R primeros.
  5. profundidad de reordenacion: MRR segun R.
  6. filtrado: prefiltrado frente a posfiltrado ingenuo (solo k), con
     sobre-recuperacion (overfetch) y recorriendo el ranking hasta k validos.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

# un solo hilo de BLAS: las latencias del filtrado son mas estables
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

SEMILLA = 17


def anunciar() -> None:
    print("=" * 64)
    print("cap. 17: consulta hibrida")
    print("recursos: python + numpy · cpu. sin servicio, gpu ni torch.")
    print(f"semilla = {SEMILLA}")
    print("=" * 64)


def _escribir(ruta: str, nota: str, cols: str,
              filas: List[Tuple]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(f"# {nota}\n")
        fh.write(cols + "\n")
        for fila in filas:
            fh.write("  ".join(str(x) for x in fila) + "\n")
    print(f"escrito {ruta}")


def _orden(score: np.ndarray) -> np.ndarray:
    """ranking completo de mayor a menor; los empates, por indice menor."""
    return np.argsort(-score, kind="stable")


# ---------------------------------------------------------------------------
# corpus sintetico: temas (senal densa) + terminos raros (senal lexica)
# ---------------------------------------------------------------------------

class Corpus:
    """Coleccion sintetica construida para que lexico y denso se complementen.

    Cada documento pertenece a un TEMA: muestrea 8 de las 40 palabras comunes
    de su tema y lleva un TERMINO RARO unico, dos veces. Su vector denso es el
    centroide del tema mas ruido de escala `ruido` (una senal tematica
    sintetica). Las palabras 40-49 de cada tema se reservan para las consultas
    semanticas: ningun documento las contiene.
    """

    def __init__(self, n_docs: int = 2000, n_temas: int = 40,
                 dim: int = 128, ruido: float = 0.6,
                 semilla: int = SEMILLA) -> None:
        rng = np.random.default_rng(semilla)
        self.n_docs, self.n_temas = n_docs, n_temas
        self.tema = rng.integers(0, n_temas, n_docs)
        self.docs: List[List[str]] = []
        self.raros: List[str] = []
        for i in range(n_docs):
            t = self.tema[i]
            palabras = [f"t{t}_w{w}" for w in rng.integers(0, 40, 8)]
            raro = f"raro_{i}"                       # termino unico del doc
            palabras += [raro] * 2
            self.docs.append(palabras)
            self.raros.append(raro)
        centros = rng.standard_normal((n_temas, dim))
        ruido_docs = rng.standard_normal((n_docs, dim))
        self.densos = centros[self.tema] + ruido * ruido_docs
        self.densos /= np.linalg.norm(self.densos, axis=1, keepdims=True)
        self.centros = centros / np.linalg.norm(centros, axis=1, keepdims=True)
        self.anio = rng.integers(2018, 2026, n_docs)
        self._indexar_bm25()

    def _indexar_bm25(self) -> None:
        self.df: Counter = Counter()
        self.tf: List[Counter] = []
        for doc in self.docs:
            c = Counter(doc)
            self.tf.append(c)
            for term in c:
                self.df[term] += 1
        self.long = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.long_media = float(self.long.mean())

    def bm25(self, consulta: List[str], k1: float = 1.5,
             b: float = 0.75) -> np.ndarray:
        """Puntuacion BM25 de la consulta frente a cada documento (0 si no
        comparte ningun termino)."""
        score = np.zeros(self.n_docs, dtype=np.float32)
        for term in consulta:
            if term not in self.df:
                continue
            idf = math.log(1 + (self.n_docs - self.df[term] + 0.5)
                           / (self.df[term] + 0.5))
            for i, c in enumerate(self.tf):
                if term in c:
                    f = c[term]
                    denom = f + k1 * (1 - b + b * self.long[i]
                                      / self.long_media)
                    score[i] += idf * f * (k1 + 1) / denom
        return score

    def denso(self, q_vec: np.ndarray) -> np.ndarray:
        """Coseno con cada documento: las filas de self.densos ya tienen
        norma 1, y se normaliza la consulta (que no puede ser nula)."""
        nq = float(np.linalg.norm(q_vec))
        if nq == 0:
            raise ValueError("el coseno no admite una consulta nula")
        return self.densos @ (q_vec / nq)


# ---------------------------------------------------------------------------
# fusion: rangos reciprocos y combinacion lineal
# ---------------------------------------------------------------------------

def lista_top(score: np.ndarray, L: int = 100,
              excluir_ceros: bool = False) -> List[int]:
    """la lista top-L que entrega un recuperador. BM25 excluye las
    puntuaciones nulas (ningun termino comun); el denso no excluye nada: un
    coseno negativo puede estar entre los mejores."""
    orden = _orden(score)
    if excluir_ceros:
        orden = orden[score[orden] > 0]
    return orden[:L].tolist()


def rrf(listas: List[List[int]], k: int = 60) -> Dict[int, float]:
    """Fusion de rangos reciprocos: cada documento suma 1/(k + r) por cada
    lista en la que aparece, con el rango r contado desde 1. Acumula en un
    diccionario porque cada recuperador puede devolver documentos distintos."""
    fus: Dict[int, float] = {}
    for lista in listas:
        for r, doc in enumerate(lista, start=1):
            fus[doc] = fus.get(doc, 0.0) + 1.0 / (k + r)
    return fus


def a_puntuacion(fus: Dict[int, float], n: int) -> np.ndarray:
    """la fusion como vector de puntuaciones (0 para los no recuperados)."""
    s = np.zeros(n)
    for doc, v in fus.items():
        s[doc] = v
    return s


def norm01(x: np.ndarray) -> np.ndarray:
    """min-max a [0, 1]; una senal constante (max == min) da ceros."""
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def norm_z(x: np.ndarray) -> np.ndarray:
    """z-score por consulta; una senal constante da ceros."""
    sd = float(x.std())
    if sd == 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(x.mean())) / sd


def hibrido_rrf(corpus: Corpus, palabras: List[str],
                q_vec: np.ndarray, L: int = 100) -> np.ndarray:
    sb, sd = corpus.bm25(palabras), corpus.denso(q_vec)
    fus = rrf([lista_top(sb, L, excluir_ceros=True), lista_top(sd, L)])
    return a_puntuacion(fus, corpus.n_docs)


# ---------------------------------------------------------------------------
# consultas, verdad de referencia y metricas
# ---------------------------------------------------------------------------

def consultas_semanticas(corpus: Corpus, n: int, rng) -> List[Tuple]:
    """Por tema: 4 palabras reservadas del tema (ningun documento las
    contiene) y un vector cerca del centroide. Relevantes: el tema entero."""
    out = []
    for _ in range(n):
        t = int(rng.integers(0, corpus.n_temas))
        palabras = [f"t{t}_w{w}" for w in rng.integers(40, 50, 4)]
        q_vec = corpus.centros[t] + 0.3 * rng.standard_normal(
            corpus.densos.shape[1])
        rel = set(np.flatnonzero(corpus.tema == t).tolist())
        out.append((palabras, q_vec, rel))
    return out


def consultas_lexicas(corpus: Corpus, n: int, rng) -> List[Tuple]:
    """Por documento: su termino raro y dos palabras de su tema, con un
    vector cerca del centroide del tema. Relevante: ese documento."""
    out = []
    for _ in range(n):
        i = int(rng.integers(0, corpus.n_docs))
        t = corpus.tema[i]
        palabras = [corpus.raros[i]] + [f"t{t}_w{w}"
                                        for w in rng.integers(0, 40, 2)]
        q_vec = corpus.centros[t] + 0.3 * rng.standard_normal(
            corpus.densos.shape[1])
        out.append((palabras, q_vec, {i}))
    return out


def recall_acotado(score: np.ndarray, rel: set, k: int = 10) -> float:
    """|R ∩ top_k| / min(|R|, k): el recall acotado (capped recall). Con un
    solo relevante coincide con el recall; con mas de k, con la precision@k."""
    top = set(_orden(score)[:k].tolist())
    return len(top & rel) / min(len(rel), k)


def recall_rel(score: np.ndarray, rel: set, k: int = 10) -> float:
    """recall@k de relevancia: |R ∩ top_k| / |R|."""
    top = set(_orden(score)[:k].tolist())
    return len(top & rel) / len(rel)


def rr(ranking: np.ndarray, objetivo: int) -> float:
    """rango reciproco: 1 / puesto (desde 1) del relevante; 0 si no esta."""
    pos = np.flatnonzero(ranking == objetivo)
    return 1.0 / (pos[0] + 1) if len(pos) else 0.0


# ---------------------------------------------------------------------------
# 1. complementariedad y 2. fusion lineal, sobre las mismas consultas
# ---------------------------------------------------------------------------

def simular_complementariedad(corpus: Corpus, sem, lex) -> None:
    filas = []
    acum = {}
    print("\ncomplementariedad: recall acotado@10 por tipo y metodo")
    print("  consulta     BM25    denso   hibrido  | recall@10 semantico")
    for nombre, consultas in (("semantica", sem), ("lexica", lex)):
        rb = rd = rh = 0.0
        rel_d = 0.0
        for palabras, q_vec, rel in consultas:
            sb, sd = corpus.bm25(palabras), corpus.denso(q_vec)
            sh = hibrido_rrf(corpus, palabras, q_vec)
            rb += recall_acotado(sb, rel)
            rd += recall_acotado(sd, rel)
            rh += recall_acotado(sh, rel)
            rel_d += recall_rel(sd, rel)
        m = len(consultas)
        acum[nombre] = (rb / m, rd / m, rh / m)
        filas.append((nombre, round(rb / m, 4), round(rd / m, 4),
                      round(rh / m, 4)))
        print(f"  {nombre:<11}  {rb/m:.4f}  {rd/m:.4f}  {rh/m:.4f}"
              f"   | denso {rel_d/m:.4f}")
    mez = [round((acum["semantica"][j] + acum["lexica"][j]) / 2, 4)
           for j in range(3)]
    filas.append(("mixto", *mez))
    print(f"  {'mixto':<11}  {mez[0]:.4f}  {mez[1]:.4f}  {mez[2]:.4f}")
    tam = np.bincount(corpus.tema)
    print(f"  relevantes por tema: media {tam.mean():.1f}, "
          f"min {tam.min()}, max {tam.max()}")
    _escribir(os.path.join("data", "cap17_complementariedad.dat"),
              "recall acotado@10 de BM25, denso e hibrido (RRF, rangos desde 1, "
              "listas top-100) en consultas semanticas, lexicas y su mezcla "
              "(corpus sintetico)", "consulta  bm25  denso  hibrido", filas)


def simular_fusion(corpus: Corpus, sem, lex) -> None:
    """combinacion lineal (1 - alpha) * lexico + alpha * denso, con
    normalizacion min-max y z-score, sobre la mezcla de (1): alpha = 0 y
    alpha = 1 reproducen BM25 y denso."""
    consultas = sem + lex
    cache = [(corpus.bm25(p), corpus.denso(q), rel) for p, q, rel in consultas]
    filas = []
    print("\nfusion lineal: recall acotado@10 de la mezcla segun alpha")
    print("  alpha  minmax  zscore")
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        tm = tz = 0.0
        for sb, sd, rel in cache:
            tm += recall_acotado((1 - alpha) * norm01(sb)
                                 + alpha * norm01(sd), rel)
            tz += recall_acotado((1 - alpha) * norm_z(sb)
                                 + alpha * norm_z(sd), rel)
        m = len(cache)
        filas.append((alpha, round(tm / m, 4), round(tz / m, 4)))
        print(f"  {alpha:<5}  {tm/m:.4f}  {tz/m:.4f}")
    _escribir(os.path.join("data", "cap17_fusion.dat"),
              "recall acotado@10 de la combinacion lineal lexico-denso segun el "
              "peso alpha (0 = solo lexico, 1 = solo denso), con normalizacion "
              "min-max y z-score, sobre la mezcla de la complementariedad",
              "alpha  minmax  zscore", filas)


# ---------------------------------------------------------------------------
# 3. una senal densa debil: la fusion puede no ayudar
# ---------------------------------------------------------------------------

def simular_senal_debil(nq: int = 300) -> None:
    """la misma mezcla de consultas con una senal densa cada vez mas ruidosa
    (ruido de los documentos 0,6, 1,5, 3 y 6 frente a centroides de norma 1
    antes de normalizar)."""
    filas = []
    print("\nsenal debil: recall acotado@10 de la mezcla segun el ruido denso")
    print("  ruido  BM25    denso   hibrido")
    for ruido in (0.6, 1.5, 3.0, 6.0):
        corpus = Corpus(ruido=ruido)
        rng = np.random.default_rng(SEMILLA + 1)
        cons = (consultas_semanticas(corpus, nq, rng)
                + consultas_lexicas(corpus, nq, rng))
        rb = rd = rh = 0.0
        for palabras, q_vec, rel in cons:
            rb += recall_acotado(corpus.bm25(palabras), rel)
            rd += recall_acotado(corpus.denso(q_vec), rel)
            rh += recall_acotado(hibrido_rrf(corpus, palabras, q_vec), rel)
        m = len(cons)
        filas.append((ruido, round(rb / m, 4), round(rd / m, 4),
                      round(rh / m, 4)))
        print(f"  {ruido:<5}  {rb/m:.4f}  {rd/m:.4f}  {rh/m:.4f}")
    _escribir(os.path.join("data", "cap17_debil.dat"),
              "recall acotado@10 de la mezcla de consultas con BM25, denso e "
              "hibrido (RRF) segun el ruido de la senal densa sintetica",
              "ruido  bm25  denso  hibrido", filas)


# ---------------------------------------------------------------------------
# 4. dos etapas y 5. profundidad, con un reordenador sintetico
# ---------------------------------------------------------------------------

def reordenar(corpus: Corpus, palabras: List[str], q_vec: np.ndarray,
              R: int, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """primera etapa: los R primeros por la senal densa. segunda: un
    reordenador sintetico (BM25 normalizado mas 0,3 del denso) sobre esos R.
    no es un cross-encoder: no lee consulta y documento juntos."""
    if R < k:
        raise ValueError("la profundidad R debe ser al menos k")
    sd = corpus.denso(q_vec)
    cand = _orden(sd)[:R]
    sb = corpus.bm25(palabras)
    fino = norm01(sb)[cand] + 0.3 * norm01(sd)[cand]
    return cand, cand[np.argsort(-fino, kind="stable")]


def simular_dos_etapas(corpus: Corpus, lex, R: int = 50) -> None:
    m1 = m2 = 0.0
    for palabras, q_vec, rel in lex:
        objetivo = next(iter(rel))
        cand, recand = reordenar(corpus, palabras, q_vec, R)
        m1 += rr(cand, objetivo)
        m2 += rr(recand, objetivo)
    m = len(lex)
    filas = [("una_etapa_denso", round(m1 / m, 4)),
             ("dos_etapas_reordenador", round(m2 / m, 4))]
    print("\ndos etapas: MRR del denso frente al reordenador sintetico")
    print(f"  una etapa (denso, top {R}): {m1/m:.4f}")
    print(f"  dos etapas (reordenador):   {m2/m:.4f}")
    _escribir(os.path.join("data", "cap17_dos_etapas.dat"),
              "MRR de la recuperacion densa (una etapa) frente al reordenador "
              "sintetico (BM25 normalizado mas 0,3 del denso) sobre los %d "
              "primeros candidatos, consultas lexicas" % R,
              "metodo  mrr", filas)


def simular_profundidad(corpus: Corpus, lex) -> None:
    filas = []
    print("\nprofundidad de reordenacion: MRR segun R")
    print("  R     mrr     techo")
    for R in (10, 20, 50, 100, 200, 500):
        tot = techo = 0.0
        for palabras, q_vec, rel in lex:
            objetivo = next(iter(rel))
            cand, recand = reordenar(corpus, palabras, q_vec, R)
            tot += rr(recand, objetivo)
            techo += objetivo in set(cand.tolist())
        m = len(lex)
        filas.append((R, round(tot / m, 4), round(techo / m, 4)))
        print(f"  {R:<4}  {tot/m:.4f}  {techo/m:.4f}")
    _escribir(os.path.join("data", "cap17_profundidad.dat"),
              "MRR tras el reordenador sintetico segun la profundidad R de la "
              "primera etapa (denso), y fraccion de consultas cuyo relevante "
              "esta entre los R candidatos (techo)", "R  mrr  techo", filas)


# ---------------------------------------------------------------------------
# 6. filtrado: prefiltrado y tres posfiltrados
# ---------------------------------------------------------------------------

def prefiltro(densos: np.ndarray, anio: np.ndarray, q: np.ndarray,
              permitidos: np.ndarray, k: int = 10) -> np.ndarray:
    """mascara de metadatos, barrido exacto en el subconjunto e ids
    globales del subconjunto."""
    idx = np.flatnonzero(np.isin(anio, permitidos))
    local = _orden(densos[idx] @ q)[:k]
    return idx[local]


def posfiltro(densos: np.ndarray, anio: np.ndarray, q: np.ndarray,
              permitidos: np.ndarray, k: int = 10,
              overfetch: int = 10) -> List[int]:
    """barrido exacto, los `overfetch` primeros y filtro despues: con
    overfetch = k es el posfiltrado ingenuo, que puede devolver menos de k."""
    cand = _orden(densos @ q)[:overfetch]
    ok = set(permitidos.tolist())
    return [int(c) for c in cand if anio[c] in ok][:k]


def posfiltro_iterativo(densos: np.ndarray, anio: np.ndarray, q: np.ndarray,
                        permitidos: np.ndarray, k: int = 10) -> List[int]:
    """recorre el ranking exacto hasta reunir k validos: con busqueda exacta,
    el mismo top-k que el prefiltrado."""
    ok = set(permitidos.tolist())
    out: List[int] = []
    for c in _orden(densos @ q):
        if anio[c] in ok:
            out.append(int(c))
            if len(out) == k:
                break
    return out


def simular_filtrado(nq: int = 100) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = Corpus(n_docs=20000)
    qs = []
    for _ in range(nq):
        q = corpus.centros[int(rng.integers(0, corpus.n_temas))] \
            + 0.3 * rng.standard_normal(corpus.densos.shape[1])
        qs.append(q / np.linalg.norm(q))
    D, A = corpus.densos, corpus.anio
    filas = []
    print("\nfiltrado: prefiltro y posfiltros segun la selectividad (anio)")
    print("  sel(%)  pre_ms  post_ms  r_ingenuo  r_over100  r_iter  iguales")
    for anios_ok in (1, 2, 4, 8):
        permit = np.arange(2018, 2018 + anios_ok)
        sel = 100.0 * anios_ok / 8
        prefiltro(D, A, qs[0], permit)                 # calentamiento
        pre, over, tp, to = [], [], [], []
        for q in qs:                                   # el minimo por consulta
            t0 = time.perf_counter()
            pre.append(prefiltro(D, A, q, permit))
            tp.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            over.append(posfiltro(D, A, q, permit, overfetch=100))
            to.append(time.perf_counter() - t0)
        t_pre, t_pos = min(tp), min(to)
        ing = [posfiltro(D, A, q, permit, overfetch=10) for q in qs]
        it = [posfiltro_iterativo(D, A, q, permit) for q in qs]
        iguales = sum(list(p) == i for p, i in zip(pre, it))
        r_ing = float(np.mean([len(x) for x in ing]))
        r_over = float(np.mean([len(x) for x in over]))
        r_it = float(np.mean([len(x) for x in it]))
        filas.append((round(sel, 1), round(t_pre * 1e3, 3),
                      round(t_pos * 1e3, 3), round(r_ing, 2),
                      round(r_over, 2), round(r_it, 2), iguales))
        print(f"  {sel:<6.1f}  {t_pre*1e3:6.3f}  {t_pos*1e3:6.3f}  "
              f"{r_ing:9.2f}  {r_over:9.2f}  {r_it:6.2f}  {iguales}/{nq}")
    _escribir(os.path.join("data", "cap17_filtrado.dat"),
              "latencia minima de %d consultas (ms, un hilo) del prefiltrado y "
              "del posfiltrado con overfetch 100, y resultados medios (de k=10) del posfiltrado ingenuo "
              "(overfetch 10), con overfetch 100 y recorriendo hasta k validos, "
              "segun la selectividad del filtro de anio (n=20000, busqueda "
              "exacta; filtro independiente de la similitud); iguales = "
              "consultas en que el recorrido coincide con el prefiltrado" % nq,
              "selectividad  prefiltro_ms  posfiltro_ms  r_ingenuo  r_over100  "
              "r_iter  iguales", filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta lexica y el puesto que le da cada senal
# ---------------------------------------------------------------------------

def demostracion(corpus: Corpus) -> None:
    rng = np.random.default_rng(SEMILLA + 2)
    i = 7
    t = corpus.tema[i]
    palabras = [corpus.raros[i]] + [f"t{t}_w{w}"
                                    for w in rng.integers(0, 40, 2)]
    q_vec = corpus.centros[t] + 0.3 * rng.standard_normal(
        corpus.densos.shape[1])
    sb, sd = corpus.bm25(palabras), corpus.denso(q_vec)
    sh = hibrido_rrf(corpus, palabras, q_vec)
    print(f"\ndemostracion: consulta lexica del doc {i} (termino raro + tema)")
    print("  metodo   puesto del doc objetivo (desde 1)")
    for nombre, s in (("BM25", sb), ("denso", sd), ("hibrido", sh)):
        puesto = int(np.flatnonzero(_orden(s) == i)[0]) + 1
        print(f"  {nombre:<8} {puesto:>6}")
    # por que salen esos puestos: palabras de tema del buscado y de quienes
    # lo preceden en BM25, y presencia en las dos listas de quienes lo
    # adelantan en la fusion
    tema_q = palabras[1:]
    print("  palabras de tema de la consulta en el buscado:",
          sum(w in corpus.docs[i] for w in tema_q))
    antes = _orden(sb)[:int(np.flatnonzero(_orden(sb) == i)[0])]
    print("  por delante en BM25:", len(antes), "con las dos palabras de tema:",
          sum(all(w in corpus.docs[d] for w in tema_q) for d in antes))
    lb, ld = set(lista_top(sb, excluir_ceros=True)), set(lista_top(sd))
    delante = _orden(sh)[:int(np.flatnonzero(_orden(sh) == i)[0])]
    print("  por delante en el hibrido:", len(delante), "en las dos listas:",
          sum(d in lb and d in ld for d in delante))


def main() -> None:
    anunciar()
    corpus = Corpus()
    rng = np.random.default_rng(SEMILLA + 1)
    sem = consultas_semanticas(corpus, 300, rng)
    lex = consultas_lexicas(corpus, 300, rng)
    simular_complementariedad(corpus, sem, lex)
    simular_fusion(corpus, sem, lex)
    simular_senal_debil()
    simular_dos_etapas(corpus, lex)
    simular_profundidad(corpus, lex)
    simular_filtrado()
    demostracion(corpus)


if __name__ == "__main__":
    main()
