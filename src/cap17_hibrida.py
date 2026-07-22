"""capitulo 17 — consulta hibrida.

estudia, en numpy, como se combinan las senales de una busqueda real: el filtro
estructurado sobre los metadatos, la senal lexica (BM25, del cap. 10) y la senal
densa (vectores, del cap. 11-12), y la reordenacion en dos etapas. construye un
corpus sintetico disenado para que lexico y denso se complementen ---unos
documentos se encuentran por terminos exactos, otros por significado--- y mide
cinco cosas con datos reales:

  1. complementariedad: recall@10 de BM25 solo, denso solo e hibrido (RRF) sobre
     consultas lexicas (terminos raros) y semanticas (significado); el hibrido
     gana en ambas, cada senal sola falla en una.
  2. fusion: efecto del parametro de la fusion de rangos reciprocos (RRF) y del
     peso de una combinacion lineal sobre la calidad.
  3. dos etapas: calidad (MRR) de la recuperacion densa frente a la reordenacion
     posterior con un reordenador mas preciso (cross-encoder simulado), y su
     coste.
  4. prefiltrado vs posfiltrado: coste y resultados de filtrar antes o despues,
     segun la selectividad (la consulta hibrida con filtro estructurado).
  5. profundidad de reordenacion: como mejora la calidad al reordenar mas
     candidatos de la primera etapa.

es Python puro con numpy (sin servicio ni GPU). los vectores densos son
sinteticos, basados en el tema del documento, y hacen las veces de la salida de
un codificador real (cap. 12). ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from typing import List, Tuple

import numpy as np

SEMILLA = 17


def anunciar() -> None:
    print("=" * 64)
    print("cap. 17 — consulta hibrida")
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


# ---------------------------------------------------------------------------
# corpus sintetico: temas (senal semantica) + terminos raros (senal lexica)
# ---------------------------------------------------------------------------

class Corpus:
    """Coleccion sintetica disenada para que lexico y denso se complementen.

    Cada documento pertenece a un TEMA: usa palabras comunes de su tema (senal
    semantica, que el vector denso captura) y, ademas, unos pocos TERMINOS RAROS
    unicos (codigos, nombres propios) que solo BM25 capta. Asi, una consulta por
    significado favorece al denso y una por un termino exacto favorece a BM25.
    """

    def __init__(self, n_docs: int = 2000, n_temas: int = 40,
                 dim: int = 128, semilla: int = SEMILLA) -> None:
        rng = np.random.default_rng(semilla)
        self.n_docs = n_docs
        self.n_temas = n_temas
        # vocabulario: 20 palabras comunes por tema + terminos raros unicos
        self.tema = rng.integers(0, n_temas, n_docs)
        self.docs: List[List[str]] = []
        self.raros: List[str] = []
        # vocabulario amplio por tema (50): cada doc muestrea pocas (8), asi dos
        # docs del mismo tema comparten POCAS palabras exactas, y una consulta
        # semantica usa palabras del tema que los relevantes casi no contienen.
        for i in range(n_docs):
            t = self.tema[i]
            palabras = [f"t{t}_w{w}" for w in rng.integers(0, 40, 8)]
            raro = f"raro_{i}"                       # termino unico del doc
            palabras += [raro] * 2
            self.docs.append(palabras)
            self.raros.append(raro)
        # vectores densos: centroide del tema + ruido (hacen de embedding real)
        centros = rng.standard_normal((n_temas, dim))
        self.densos = (centros[self.tema]
                       + 0.6 * rng.standard_normal((n_docs, dim)))
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
        """Puntuacion BM25 de la consulta frente a cada documento."""
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
        """Similitud coseno de un vector de consulta con cada documento."""
        return self.densos @ (q_vec / (np.linalg.norm(q_vec) + 1e-12))


def _rank(score: np.ndarray) -> np.ndarray:
    """Devuelve el rango (0 = mejor) de cada documento segun su puntuacion."""
    orden = np.argsort(-score)
    rango = np.empty_like(orden)
    rango[orden] = np.arange(len(orden))
    return rango


def rrf(scores: List[np.ndarray], k: int = 60, top: int = 100) -> np.ndarray:
    """Fusion de rangos reciprocos sobre las listas top-K de cada metodo: cada
    documento suma 1/(k+rango) por cada lista en la que aparece entre los `top`
    primeros; los que no entran en una lista no puntuan por ella. Asi un metodo
    que no encuentra un documento no lo penaliza con un rango espurio."""
    n = len(scores[0])
    fus = np.zeros(n, dtype=np.float64)
    for s in scores:
        orden = np.argsort(-s)
        orden = orden[s[orden] > 0][:top]        # solo candidatos reales
        for rango, doc in enumerate(orden):
            fus[doc] += 1.0 / (k + rango)
    return fus


# ---------------------------------------------------------------------------
# consultas y verdad de referencia
# ---------------------------------------------------------------------------

def _consultas_semanticas(corpus: Corpus, n: int, rng) -> List[Tuple]:
    """Consultas por significado: unas palabras del tema y su vector. Relevantes
    = todos los documentos del tema."""
    out = []
    for _ in range(n):
        t = int(rng.integers(0, corpus.n_temas))
        # palabras reservadas del tema (40-49): sinonimos que los docs nunca
        # contienen, asi BM25 no los encuentra y solo el denso (tema) los capta
        palabras = [f"t{t}_w{w}" for w in rng.integers(40, 50, 4)]
        q_vec = corpus.centros[t] + 0.3 * rng.standard_normal(
            corpus.densos.shape[1])
        rel = set(np.where(corpus.tema == t)[0].tolist())
        out.append((palabras, q_vec, rel))
    return out


def _consultas_lexicas(corpus: Corpus, n: int, rng) -> List[Tuple]:
    """Consultas por termino exacto: el termino raro de un documento (mas
    contexto del tema). Relevante = ese documento concreto."""
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


def _recall_at_k(score: np.ndarray, rel: set, k: int = 10) -> float:
    top = set(np.argsort(-score)[:k].tolist())
    return len(top & rel) / min(len(rel), k)


# ---------------------------------------------------------------------------
# 1. complementariedad lexico / denso / hibrido
# ---------------------------------------------------------------------------

def simular_complementariedad(nq: int = 300) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = Corpus()
    sem = _consultas_semanticas(corpus, nq, rng)
    lex = _consultas_lexicas(corpus, nq, rng)
    filas = []
    acum = {"bm25": [0.0, 0.0], "denso": [0.0, 0.0], "hibrido": [0.0, 0.0]}
    print("\ncomplementariedad: recall@10 por tipo de consulta y metodo")
    print("  consulta     BM25    denso   hibrido")
    for j, (nombre, consultas) in enumerate((("semantica", sem),
                                             ("lexica", lex))):
        rb = rd = rh = 0.0
        for palabras, q_vec, rel in consultas:
            sb = corpus.bm25(palabras)
            sd = corpus.denso(q_vec)
            sh = rrf([sb, sd])
            rb += _recall_at_k(sb, rel)
            rd += _recall_at_k(sd, rel)
            rh += _recall_at_k(sh, rel)
        m = len(consultas)
        filas.append((nombre, round(rb / m, 4), round(rd / m, 4),
                      round(rh / m, 4)))
        acum["bm25"][j], acum["denso"][j], acum["hibrido"][j] = \
            rb / m, rd / m, rh / m
        print(f"  {nombre:<11}  {rb/m:.4f}  {rd/m:.4f}  {rh/m:.4f}")
    # fila mixto: media de los dos tipos (la carga real es mezcla de ambos)
    mez = [round(sum(acum[k]) / 2, 4) for k in ("bm25", "denso", "hibrido")]
    filas.append(("mixto", mez[0], mez[1], mez[2]))
    print(f"  {'mixto':<11}  {mez[0]:.4f}  {mez[1]:.4f}  {mez[2]:.4f}")
    _escribir(os.path.join("data", "cap17_complementariedad.dat"),
              "recall@10 de BM25, denso e hibrido (RRF) en consultas semanticas "
              "(significado) y lexicas (termino exacto)",
              "consulta  bm25  denso  hibrido", filas)


# ---------------------------------------------------------------------------
# 2. fusion: el parametro de RRF y el peso lineal
# ---------------------------------------------------------------------------

def simular_fusion(nq: int = 300) -> None:
    """Calidad del hibrido segun como se fusione: combinacion lineal de
    puntuaciones normalizadas con peso alpha (0 = solo lexico, 1 = solo denso).
    El optimo esta en medio: ahi esta la ganancia de fusionar."""
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = Corpus()
    consultas = (_consultas_semanticas(corpus, nq // 2, rng)
                 + _consultas_lexicas(corpus, nq // 2, rng))
    filas = []
    print("\nfusion: recall@10 combinando lexico y denso segun el peso alpha")
    print("  alpha  recall")
    for alpha in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        tot = 0.0
        for palabras, q_vec, rel in consultas:
            sb = corpus.bm25(palabras)
            sd = corpus.denso(q_vec)
            comb = (1 - alpha) * _norm01(sb) + alpha * _norm01(sd)
            tot += _recall_at_k(comb, rel)
        filas.append((alpha, round(tot / len(consultas), 4)))
        print(f"  {alpha:<5}  {tot/len(consultas):.4f}")
    _escribir(os.path.join("data", "cap17_fusion.dat"),
              "recall@10 de la combinacion lineal lexico-denso segun el peso "
              "alpha (0=solo lexico, 1=solo denso)", "alpha  recall", filas)


def _norm01(x: np.ndarray) -> np.ndarray:
    """Normaliza una puntuacion al rango [0,1] (min-max)."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)


# ---------------------------------------------------------------------------
# 3. dos etapas: reordenacion con un reordenador mas preciso
# ---------------------------------------------------------------------------

def simular_dos_etapas(nq: int = 300, R: int = 50) -> None:
    """Primera etapa: recuperacion densa (barata, coarse). Segunda: reordenar
    los R candidatos con un reordenador mas preciso (cross-encoder simulado, que
    lee consulta y documento juntos). Mide el MRR antes y despues."""
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = Corpus()
    consultas = _consultas_lexicas(corpus, nq, rng)   # caso fino: doc concreto
    mrr_1 = mrr_2 = 0.0
    for palabras, q_vec, rel in consultas:
        objetivo = next(iter(rel))
        sd = corpus.denso(q_vec)
        cand = np.argsort(-sd)[:R]                     # primera etapa: denso
        mrr_1 += _mrr(cand, objetivo)
        # reordenador: combina senal lexica fina y densa sobre los candidatos
        sb = corpus.bm25(palabras)
        fino = _norm01(sb)[cand] + 0.3 * _norm01(sd)[cand]
        recand = cand[np.argsort(-fino)]
        mrr_2 += _mrr(recand, objetivo)
    m = len(consultas)
    filas = [("una_etapa_denso", round(mrr_1 / m, 4)),
             ("dos_etapas_rerank", round(mrr_2 / m, 4))]
    print("\ndos etapas: MRR de la recuperacion densa vs tras reordenar")
    print(f"  una etapa (denso): {mrr_1/m:.4f}")
    print(f"  dos etapas (rerank): {mrr_2/m:.4f}")
    _escribir(os.path.join("data", "cap17_dos_etapas.dat"),
              "MRR de la recuperacion densa de una etapa vs la reordenacion de "
              "los %d primeros candidatos (cross-encoder simulado)" % R,
              "metodo  mrr", filas)


def _mrr(ranking: np.ndarray, objetivo: int) -> float:
    pos = np.where(ranking == objetivo)[0]
    return 1.0 / (pos[0] + 1) if len(pos) else 0.0


# ---------------------------------------------------------------------------
# 4. prefiltrado vs posfiltrado (consulta hibrida con filtro estructurado)
# ---------------------------------------------------------------------------

def simular_filtrado(nq: int = 100) -> None:
    """Coste y resultados de filtrar por metadatos (anio) antes o despues de la
    busqueda densa, segun la selectividad. El posfiltrado se queda corto cuando
    el filtro es selectivo (inanicion)."""
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = Corpus(n_docs=20000)
    qs = [corpus.centros[int(rng.integers(0, corpus.n_temas))]
          + 0.3 * rng.standard_normal(corpus.densos.shape[1])
          for _ in range(nq)]
    filas = []
    print("\nfiltrado: prefiltro vs posfiltro segun selectividad (anio)")
    print("  sel(%)  prefiltro_ms  posfiltro_ms  result_posfiltro")
    for anios_ok in (1, 2, 4, 8):
        permit = np.arange(2018, 2018 + anios_ok)
        sel = 100.0 * anios_ok / 8
        t0 = time.perf_counter()
        for q in qs:
            mask = np.isin(corpus.anio, permit)
            sim = corpus.densos[mask] @ (q / np.linalg.norm(q))
            np.argsort(-sim)[:10]
        pre = (time.perf_counter() - t0) / nq
        surv = 0
        t0 = time.perf_counter()
        for q in qs:
            sim = corpus.densos @ (q / np.linalg.norm(q))
            cand = np.argsort(-sim)[:100]
            paso = [c for c in cand if corpus.anio[c] in permit][:10]
            surv += len(paso)
        pos = (time.perf_counter() - t0) / nq
        filas.append((round(sel, 1), round(pre * 1e3, 3),
                      round(pos * 1e3, 3), round(surv / nq, 1)))
        print(f"  {sel:<6.1f}  {pre*1e3:>10.3f}  {pos*1e3:>10.3f}"
              f"  {surv/nq:>14.1f}")
    _escribir(os.path.join("data", "cap17_filtrado.dat"),
              "latencia (ms) de prefiltrado vs posfiltrado y resultados del "
              "posfiltrado segun selectividad del filtro de anio (n=20000)",
              "selectividad  prefiltro_ms  posfiltro_ms  result_posfiltro",
              filas)


# ---------------------------------------------------------------------------
# 5. profundidad de reordenacion
# ---------------------------------------------------------------------------

def simular_profundidad(nq: int = 300) -> None:
    """Como mejora el MRR al reordenar mas candidatos R de la primera etapa: mas
    profundidad, mas posibilidades de rescatar el documento correcto, a mas
    coste de reordenacion."""
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = Corpus()
    consultas = _consultas_lexicas(corpus, nq, rng)
    filas = []
    print("\nprofundidad de reordenacion: MRR segun cuantos candidatos R")
    print("  R     mrr")
    for R in (5, 10, 20, 50, 100, 200):
        tot = 0.0
        for palabras, q_vec, rel in consultas:
            objetivo = next(iter(rel))
            sd = corpus.denso(q_vec)
            cand = np.argsort(-sd)[:R]
            sb = corpus.bm25(palabras)
            fino = _norm01(sb)[cand] + 0.3 * _norm01(sd)[cand]
            recand = cand[np.argsort(-fino)]
            tot += _mrr(recand, objetivo)
        filas.append((R, round(tot / len(consultas), 4)))
        print(f"  {R:<4}  {tot/len(consultas):.4f}")
    _escribir(os.path.join("data", "cap17_profundidad.dat"),
              "MRR tras reordenar segun la profundidad R de candidatos de la "
              "primera etapa (denso)", "R  mrr", filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta hibrida y los rangos de cada senal
# ---------------------------------------------------------------------------

def demostracion() -> None:
    rng = np.random.default_rng(SEMILLA + 2)
    corpus = Corpus()
    i = 7
    t = corpus.tema[i]
    palabras = [corpus.raros[i]] + [f"t{t}_w{w}"
                                    for w in rng.integers(0, 40, 2)]
    q_vec = corpus.centros[t] + 0.3 * rng.standard_normal(
        corpus.densos.shape[1])
    sb, sd = corpus.bm25(palabras), corpus.denso(q_vec)
    sh = rrf([sb, sd])
    print(f"\ndemostracion: consulta lexica del doc {i} (termino raro + tema)")
    print("  metodo   rango_del_doc_objetivo  top1")
    for nombre, s in (("BM25", sb), ("denso", sd), ("hibrido", sh)):
        rango = int(np.where(np.argsort(-s) == i)[0][0])
        print(f"  {nombre:<8} {rango:>20}  {int(np.argmax(s))}")


def main() -> None:
    anunciar()
    simular_complementariedad()
    simular_fusion()
    simular_dos_etapas()
    simular_profundidad()
    simular_filtrado()
    demostracion()


if __name__ == "__main__":
    main()
