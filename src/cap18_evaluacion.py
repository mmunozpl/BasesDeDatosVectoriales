"""capitulo 18 — evaluacion y operacion.

cierra el libro midiendo, en numpy, como se EVALUA y se OPERA un sistema de
busqueda vectorial. mide cinco cosas, todas con datos reales y semilla fija:

  1. metricas: recall@k, MRR y nDCG sobre rankings con el MISMO recall pero
     distinto orden; cada metrica capta una faceta distinta de la calidad.
  2. recall@k: como crece el recall al ampliar k (la curva recall-profundidad).
  3. cuantizacion en produccion: recall@10 y memoria de float32, float16, int8
     y binaria; la palanca central del coste, con su efecto sobre el recall.
  4. cribado binario en dos etapas: la binaria sola pierde recall, pero como
     primer cribado seguido de reordenacion exacta lo recupera casi entero.
  5. deriva y reindexado: un indice IVF entrenado sobre una distribucion pierde
     recall cuando los datos derivan, y lo recupera al reentrenarse (reindexar).

ademas estima latencia y rendimiento (QPS) de la busqueda exacta segun el
tamano: la latencia depende de la maquina (se reporta como referencia), pero su
forma ---lineal con N--- no. es Python puro con numpy, CPU. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Tuple

import numpy as np

SEMILLA = 18


def anunciar() -> None:
    print("=" * 64)
    print("cap. 18 — evaluacion y operacion")
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
# metricas de evaluacion
# ---------------------------------------------------------------------------

def recall_at_k(ranking: List[int], relevantes: set, k: int) -> float:
    """fraccion de los relevantes que aparecen entre los k primeros."""
    top = set(ranking[:k])
    return len(top & relevantes) / min(len(relevantes), k)


def mrr(ranking: List[int], relevantes: set) -> float:
    """rango reciproco del primer acierto (1/posicion)."""
    for pos, doc in enumerate(ranking, start=1):
        if doc in relevantes:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranking: List[int], relevantes: set, k: int) -> float:
    """nDCG@k con relevancia binaria: ganancia descontada por la posicion,
    normalizada contra el orden ideal."""
    dcg = sum(1.0 / math.log2(pos + 1)
              for pos, doc in enumerate(ranking[:k], start=1)
              if doc in relevantes)
    ideal = sum(1.0 / math.log2(pos + 1)
                for pos in range(1, min(len(relevantes), k) + 1))
    return dcg / ideal if ideal else 0.0


def simular_metricas() -> None:
    """tres rankings con los MISMOS tres relevantes pero en posiciones
    distintas: igual recall@10, distinto MRR y nDCG. cada metrica mide algo."""
    relevantes = {0, 1, 2}
    casos = {
        "ideal": [0, 1, 2, 9, 8, 7, 6, 5, 4, 3],      # los 3 arriba
        "disperso": [0, 9, 8, 1, 7, 6, 2, 5, 4, 3],   # repartidos
        "tardio": [9, 8, 7, 6, 0, 1, 2, 5, 4, 3],     # mas abajo
    }
    filas = []
    print("\nmetricas: mismo recall@10, distinto MRR y nDCG")
    print("  ranking    recall@10  MRR     nDCG@10")
    for nombre, r in casos.items():
        rec = recall_at_k(r, relevantes, 10)
        m = mrr(r, relevantes)
        n = ndcg_at_k(r, relevantes, 10)
        filas.append((nombre, round(rec, 3), round(m, 3), round(n, 3)))
        print(f"  {nombre:<9}  {rec:.3f}      {m:.3f}   {n:.3f}")
    _escribir(os.path.join("data", "cap18_metricas.dat"),
              "recall@10, MRR y nDCG@10 de tres rankings con los mismos "
              "relevantes en posiciones distintas (igual recall, distinto orden)",
              "ranking  recall  mrr  ndcg", filas)


# ---------------------------------------------------------------------------
# datos sinteticos con verdad de referencia
# ---------------------------------------------------------------------------

def _coleccion(n: int, dim: int, n_temas: int = 40,
               semilla: int = SEMILLA) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((n_temas, dim)) * 3
    tema = rng.integers(0, n_temas, n)
    vecs = centros[tema] + rng.standard_normal((n, dim))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32)


def _verdad(base: np.ndarray, consultas: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(consultas @ base.T), axis=1)[:, :k]


def _recall_listas(aprox: np.ndarray, exacto: np.ndarray) -> float:
    tot = 0.0
    for a, e in zip(aprox, exacto):
        tot += len(set(a.tolist()) & set(e.tolist())) / e.shape[0]
    return tot / len(aprox)


def simular_recall_k(n: int = 20_000, dim: int = 128, nq: int = 200) -> None:
    """como crece el recall al ampliar k: se recupera con un indice aproximado
    (IVF) y se compara con la verdad exacta a distintas profundidades k."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    q = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ivf = IndiceIVF(base, nlist=128)
    filas = []
    print("\nrecall@k: como crece el recall al ampliar k (IVF, nprobe=8)")
    print("  k     recall@k")
    for k in (1, 5, 10, 20, 50, 100):
        exacto = _verdad(base, q, k)
        aprox = np.array([ivf.buscar(x, k, nprobe=8) for x in q])
        rec = _recall_listas(aprox, exacto)
        filas.append((k, round(rec, 4)))
        print(f"  {k:<4}  {rec:.4f}")
    _escribir(os.path.join("data", "cap18_recallk.dat"),
              "recall@k de un indice IVF (nprobe=8) frente a la verdad exacta, "
              "segun la profundidad k (n=%d)" % n, "k  recall", filas)


# ---------------------------------------------------------------------------
# IVF minimo (para recall@k y deriva)
# ---------------------------------------------------------------------------

def _kmeans(x: np.ndarray, k: int, iters: int = 10,
            semilla: int = SEMILLA) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)
    cent = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        cn = (cent ** 2).sum(1)
        asign = (cn[None, :] - 2.0 * (x @ cent.T)).argmin(1)
        sumas = np.zeros((k, x.shape[1]))
        np.add.at(sumas, asign, x)
        cuenta = np.bincount(asign, minlength=k)
        ok = cuenta > 0
        cent[ok] = (sumas[ok] / cuenta[ok, None]).astype(np.float32)
    return cent.astype(np.float32), asign.astype(np.int32)


class IndiceIVF:
    def __init__(self, base: np.ndarray, nlist: int = 128,
                 semilla: int = SEMILLA) -> None:
        self.base = base
        self.centroides, asign = _kmeans(base, nlist, semilla=semilla)
        self.listas = [np.where(asign == j)[0] for j in range(nlist)]

    def buscar(self, q: np.ndarray, k: int, nprobe: int = 8) -> List[int]:
        celdas = np.argsort(((self.centroides - q) ** 2).sum(1))[:nprobe]
        cand = np.concatenate([self.listas[c] for c in celdas])
        if not len(cand):
            return []
        sims = self.base[cand] @ q
        return cand[np.argsort(-sims)[:k]].tolist()


# ---------------------------------------------------------------------------
# cuantizacion en produccion
# ---------------------------------------------------------------------------

def simular_cuantizacion(n: int = 30_000, dim: int = 128,
                         nq: int = 200, k: int = 10) -> None:
    """recall@10 y memoria de float32, float16, int8 y binaria frente a la
    verdad exacta (float32). la palanca del coste en produccion."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    q = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    exacto = _verdad(base, q, k)
    filas = []
    print("\ncuantizacion en produccion: recall@10 y bytes por vector")
    print("  tipo      bytes/vec  recall@10")
    # float32 (referencia)
    filas.append(("float32", 4 * dim, 1.0))
    # float16
    b16 = base.astype(np.float16).astype(np.float32)
    rec16 = _recall_listas(np.argsort(-(q @ b16.T), axis=1)[:, :k], exacto)
    filas.append(("float16", 2 * dim, round(rec16, 4)))
    # int8 escalar
    escala = np.abs(base).max(1, keepdims=True) / 127.0
    q8 = np.round(base / (escala + 1e-12)).astype(np.int8)
    rec8 = _recall_listas(
        np.argsort(-(q @ (q8.astype(np.float32) * escala).T), axis=1)[:, :k],
        exacto)
    filas.append(("int8", dim, round(rec8, 4)))
    # binaria (signo)
    bb = (base > 0)
    recb = _recall_listas(
        np.argsort(-(q @ (bb.astype(np.float32) * 2 - 1).T), axis=1)[:, :k],
        exacto)
    filas.append(("binaria", dim // 8, round(recb, 4)))
    for f in filas:
        print(f"  {f[0]:<8}  {f[1]:<9}  {f[2]}")
    _escribir(os.path.join("data", "cap18_cuantizacion.dat"),
              "recall@10 y bytes por vector (dim %d) de float32, float16, int8 "
              "y binaria frente a la verdad exacta float32 (n=%d)" % (dim, n),
              "tipo  bytes  recall", filas)


def simular_binaria_dos_etapas(n: int = 30_000, dim: int = 128,
                               nq: int = 200, k: int = 10) -> None:
    """la binaria sola pierde recall, pero como primer cribado (top-R por
    Hamming) seguido de reordenacion exacta lo recupera casi entero."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    q = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    exacto = _verdad(base, q, k)
    bb = (base > 0).astype(np.float32) * 2 - 1          # signos {-1,+1}
    qb = (q > 0).astype(np.float32) * 2 - 1
    filas = []
    print("\ncribado binario + reordenacion exacta: recall@10 segun R")
    print("  R      recall@10")
    for R in (50, 100, 200, 500, 1000, 2000):
        res = []
        sims_bin = qb @ bb.T                            # parecido binario
        for i in range(nq):
            cand = np.argsort(-sims_bin[i])[:R]         # cribado binario
            orden = cand[np.argsort(-(base[cand] @ q[i]))[:k]]  # reordena
            res.append(orden)
        rec = _recall_listas(np.array(res), exacto)
        filas.append((R, round(rec, 4)))
        print(f"  {R:<5}  {rec:.4f}")
    _escribir(os.path.join("data", "cap18_binaria.dat"),
              "recall@10 del cribado binario (top-R por signo) seguido de "
              "reordenacion exacta, segun R (n=%d, dim=%d)" % (n, dim),
              "R  recall", filas)


# ---------------------------------------------------------------------------
# latencia y rendimiento
# ---------------------------------------------------------------------------

def simular_rendimiento(dim: int = 256, reps: int = 30) -> None:
    """latencia por consulta y rendimiento (QPS) de la busqueda exacta segun
    N. la latencia depende de la maquina; su forma ---lineal--- no."""
    rng = np.random.default_rng(SEMILLA + 1)
    filas = []
    print("\nrendimiento: latencia y QPS de la busqueda exacta segun N")
    print("  N         latencia_ms   QPS")
    for n in (1_000, 10_000, 100_000, 500_000, 1_000_000):
        base = _coleccion(n, dim)
        qs = rng.standard_normal((reps, dim)).astype(np.float32)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        t0 = time.perf_counter()
        for x in qs:
            np.argsort(-(base @ x))[:10]
        dt = (time.perf_counter() - t0) / reps
        qps = 1.0 / dt
        filas.append((n, round(dt * 1e3, 3), int(qps)))
        print(f"  {n:<9} {dt*1e3:>10.3f}   {int(qps)}")
    _escribir(os.path.join("data", "cap18_rendimiento.dat"),
              "latencia (ms) y rendimiento (QPS) de la busqueda exacta por "
              "fuerza bruta segun N (dim %d, una maquina)" % dim,
              "n  latencia_ms  qps", filas)


# ---------------------------------------------------------------------------
# deriva y reindexado
# ---------------------------------------------------------------------------

def simular_deriva(n: int = 20_000, dim: int = 128, nq: int = 200,
                   k: int = 10) -> None:
    """un IVF entrenado sobre una distribucion pierde recall cuando se insertan
    datos de temas NUEVOS (deriva), y lo recupera al reentrenarse (reindexar)."""
    rng = np.random.default_rng(SEMILLA + 1)
    # coleccion inicial: 30 temas; el IVF se entrena con ella
    base = _coleccion(n, dim, n_temas=30)
    ivf = IndiceIVF(base, nlist=128)
    base_actual = base
    filas = []
    print("\nderiva y reindexado: recall@10 al insertar temas nuevos")
    print("  fase            n        recall@10")
    fases = [("inicial", 0), ("+deriva", 1), ("++deriva", 2)]
    for nombre, paso in fases:
        if paso > 0:
            # insertar datos de temas NUEVOS (no vistos por el k-means)
            extra = _coleccion(n // 2, dim, n_temas=30,
                               semilla=SEMILLA + 50 + paso)
            base_actual = np.vstack([base_actual, extra])
            ivf.base = base_actual
            # asignar los nuevos a los centroides viejos (sin reentrenar)
            dc = ((extra[:, None, :] - ivf.centroides[None, :, :]) ** 2).sum(2) \
                if len(extra) * 128 * dim < 5e7 else None
            asign = (dc.argmin(1) if dc is not None
                     else np.array([int(((ivf.centroides - v) ** 2).sum(1)
                                        .argmin()) for v in extra]))
            base0 = len(base_actual) - len(extra)
            for j, c in enumerate(asign):
                ivf.listas[c] = np.append(ivf.listas[c], base0 + j)
        qi = base_actual[rng.choice(len(base_actual), nq)] \
            + 0.3 * rng.standard_normal((nq, dim)).astype(np.float32)
        qi /= np.linalg.norm(qi, axis=1, keepdims=True)
        exacto = _verdad(base_actual, qi, k)
        aprox = np.array([ivf.buscar(x, k, nprobe=8) for x in qi])
        rec = _recall_listas(aprox, exacto)
        filas.append((nombre, len(base_actual), round(rec, 4)))
        print(f"  {nombre:<14}  {len(base_actual):<7}  {rec:.4f}")
    # reindexar: reentrenar el k-means sobre toda la coleccion actual
    ivf_re = IndiceIVF(base_actual, nlist=128, semilla=SEMILLA + 2)
    qi = base_actual[rng.choice(len(base_actual), nq)] \
        + 0.3 * rng.standard_normal((nq, dim)).astype(np.float32)
    qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    exacto = _verdad(base_actual, qi, k)
    aprox = np.array([ivf_re.buscar(x, k, nprobe=8) for x in qi])
    rec = _recall_listas(aprox, exacto)
    filas.append(("reindexado", len(base_actual), round(rec, 4)))
    print(f"  {'reindexado':<14}  {len(base_actual):<7}  {rec:.4f}")
    _escribir(os.path.join("data", "cap18_deriva.dat"),
              "recall@10 de un IVF al insertar temas nuevos (deriva) sin "
              "reentrenar, y tras reindexar (reentrenar el k-means)",
              "fase  n  recall", filas)


# ---------------------------------------------------------------------------
# demostracion: un informe de evaluacion de una consulta
# ---------------------------------------------------------------------------

def demostracion(n: int = 20_000, dim: int = 128) -> None:
    rng = np.random.default_rng(SEMILLA + 2)
    base = _coleccion(n, dim)
    q = base[0] + 0.3 * rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    relevantes = set(np.argsort(-(base @ q))[:10].tolist())   # verdad
    ivf = IndiceIVF(base, nlist=128)
    ranking = ivf.buscar(q, 20, nprobe=8)
    print("\ndemostracion: informe de evaluacion de una consulta (IVF, nprobe=8)")
    print(f"  recall@10:  {recall_at_k(ranking, relevantes, 10):.3f}")
    print(f"  recall@20:  {recall_at_k(ranking, relevantes, 20):.3f}")
    print(f"  MRR:        {mrr(ranking, relevantes):.3f}")
    print(f"  nDCG@10:    {ndcg_at_k(ranking, relevantes, 10):.3f}")


def main() -> None:
    anunciar()
    simular_metricas()
    simular_recall_k()
    simular_cuantizacion()
    simular_binaria_dos_etapas()
    simular_rendimiento()
    simular_deriva()
    demostracion()


if __name__ == "__main__":
    main()
