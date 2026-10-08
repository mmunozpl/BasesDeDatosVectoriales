"""capitulo 18: evaluacion y operacion.

mide, en numpy y con semilla fija, como se evalua y se opera un sistema de
busqueda vectorial, sobre una coleccion sintetica en cumulos:

  1. metricas de relevancia: recall@k, precision@k, rango reciproco y nDCG
     (binario y graduado) sobre tres rankings con los mismos relevantes en
     posiciones distintas.
  2. recall de un indice aproximado (ann recall@k): la fidelidad de un IVF al
     top-k exacto segun la profundidad k y el numero de celdas exploradas.
  3. cuantizacion: ann recall@10 y bytes por vector de float32, float16, int8
     y binaria empaquetada (np.packbits; hamming por xor y conteo de bits).
  4. cribado binario en dos etapas: top-R por hamming y reordenacion exacta
     con los vectores originales.
  5. latencia y rendimiento de la busqueda exacta segun N: la latencia de una
     consulta y el rendimiento de un lote, que no es el inverso de aquella.
  6. deriva y reindexado: un IVF con centroides viejos frente a temas nuevos,
     reajustado (mas celdas exploradas) y reconstruido.

las consultas son muestras nuevas del mismo proceso generador, no vectores de
la coleccion. las calidades dependen solo de la semilla; las latencias, de la
maquina (se miden con un hilo de BLAS y la mediana de varias repeticiones).
"""

from __future__ import annotations

import os

# un hilo de BLAS: latencias comparables entre tamanos y entre pasadas
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import math
import time

import numpy as np

SEMILLA = 18
ESCALA = 1.0           # separacion de los temas: con 3 todo es trivial


def anunciar() -> None:
    print("=" * 64)
    print("cap. 18: evaluacion y operacion")
    print("recursos: python + numpy, cpu, un hilo de BLAS")
    print(f"semilla = {SEMILLA}")
    print("=" * 64)


def _escribir(ruta: str, nota: str, cols: str, filas: list[tuple]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(f"# {nota}\n")
        fh.write(cols + "\n")
        for fila in filas:
            fh.write("  ".join(str(x) for x in fila) + "\n")
    print(f"escrito {ruta}")


# ---------------------------------------------------------------------------
# 1. metricas de relevancia
# ---------------------------------------------------------------------------

def _grados(relevancia: set | dict) -> dict:
    """un conjunto es relevancia binaria (grado 1); un dict, graduada."""
    if isinstance(relevancia, dict):
        return {d: g for d, g in relevancia.items() if g > 0}
    return {d: 1 for d in relevancia}


def recall_at_k(ranking: list[int], relevantes: set, k: int) -> float:
    """recall de relevancia: |R ∩ top_k| / |R|. no admite R vacio."""
    if not relevantes:
        raise ValueError("recall@k no esta definido sin relevantes")
    return len(set(ranking[:k]) & relevantes) / len(relevantes)


def recall_acotado(ranking: list[int], relevantes: set, k: int) -> float:
    """|R ∩ top_k| / min(|R|, k): vale 1 si los k primeros son relevantes."""
    if not relevantes:
        raise ValueError("el recall acotado no esta definido sin relevantes")
    return len(set(ranking[:k]) & relevantes) / min(len(relevantes), k)


def precision_at_k(ranking: list[int], relevantes: set, k: int) -> float:
    """|R ∩ top_k| / k; los puestos que faltan cuentan como no relevantes."""
    return len(set(ranking[:k]) & relevantes) / k


def rr(ranking: list[int], relevantes: set) -> float:
    """rango reciproco: 1/posicion del primer relevante, 0 si no hay."""
    for pos, doc in enumerate(ranking, start=1):
        if doc in relevantes:
            return 1.0 / pos
    return 0.0


def mrr(rankings: list[list[int]], relevantes: list[set]) -> float:
    """MRR: la media del rango reciproco sobre las consultas."""
    return float(np.mean([rr(r, rel) for r, rel in zip(rankings, relevantes)]))


def ndcg_at_k(ranking: list[int], relevancia: set | dict, k: int,
              exponencial: bool = True) -> float:
    """nDCG@k: ganancia 2^g - 1 (o g, lineal), descontada por
    log2(posicion + 1) y normalizada por el orden ideal. con relevancia
    binaria las dos ganancias valen 1 por acierto."""
    g = _grados(relevancia)
    if not g:
        raise ValueError("nDCG no esta definido sin relevantes")

    def gan(x: float) -> float:
        return 2 ** x - 1 if exponencial else x

    dcg = sum(gan(g.get(doc, 0)) / math.log2(pos + 1)
              for pos, doc in enumerate(ranking[:k], start=1))
    ideal = sum(gan(gi) / math.log2(pos + 1)
                for pos, gi in enumerate(sorted(g.values(), reverse=True)[:k],
                                         start=1))
    return dcg / ideal


def simular_metricas() -> None:
    """tres rankings con los mismos tres relevantes en posiciones distintas:
    igual recall@10, distinto rango reciproco y nDCG (binario y graduado)."""
    binaria = {0, 1, 2}
    graduada = {0: 3, 1: 2, 2: 1}
    casos = {
        "ideal": [0, 1, 2, 9, 8, 7, 6, 5, 4, 3],      # posiciones 1, 2, 3
        "disperso": [0, 9, 8, 1, 7, 6, 2, 5, 4, 3],   # posiciones 1, 4, 7
        "tardio": [9, 8, 7, 6, 0, 1, 2, 5, 4, 3],     # posiciones 5, 6, 7
    }
    filas = []
    print("\nmetricas: mismo recall@10, distinto orden")
    print("  ranking    recall@10  prec@5  RR     nDCG@10  nDCG@10 graduado")
    for nombre, r in casos.items():
        fila = (nombre, recall_at_k(r, binaria, 10),
                precision_at_k(r, binaria, 5), rr(r, binaria),
                ndcg_at_k(r, binaria, 10), ndcg_at_k(r, graduada, 10))
        filas.append((fila[0],) + tuple(round(v, 3) for v in fila[1:]))
        print(f"  {nombre:<9}  {fila[1]:.3f}      {fila[2]:.3f}   "
              f"{fila[3]:.3f}  {fila[4]:.3f}    {fila[5]:.3f}")
    _escribir(os.path.join("data", "cap18_metricas.dat"),
              "recall@10, precision@5, rango reciproco y nDCG@10 (binario y "
              "graduado 3-2-1) de tres rankings con los mismos relevantes en "
              "posiciones distintas",
              "ranking  recall  prec5  rr  ndcg  ndcg_grad", filas)


# ---------------------------------------------------------------------------
# coleccion sintetica, consultas reservadas y verdad exacta
# ---------------------------------------------------------------------------

def coleccion(n: int, dim: int, n_temas: int = 40, semilla: int = SEMILLA,
              escala: float = ESCALA, con_tema: bool = False) -> tuple:
    """vectores en cumulos por tema, normalizados; devuelve (base, centros) y,
    con con_tema, tambien el tema de cada vector."""
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((n_temas, dim)) * escala
    tema = rng.integers(0, n_temas, n)
    vecs = centros[tema] + rng.standard_normal((n, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    if con_tema:
        return vecs.astype(np.float32), centros, tema
    return vecs.astype(np.float32), centros


def consultas(centros: np.ndarray, nq: int, rng,
              pesos: np.ndarray | None = None, con_tema: bool = False):
    """muestras nuevas del mismo proceso: no son vectores de la coleccion.
    con con_tema devuelve tambien el tema de cada consulta."""
    tema = rng.choice(len(centros), nq, p=pesos)
    q = centros[tema] + rng.standard_normal((nq, centros.shape[1]))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if con_tema:
        return q.astype(np.float32), tema
    return q.astype(np.float32)


def top_k(sims: np.ndarray, k: int) -> np.ndarray:
    """los k mayores de cada fila, ordenados: seleccion parcial O(N) con
    argpartition y orden solo de esos k."""
    k = min(k, sims.shape[-1])
    part = np.argpartition(-sims, k - 1, axis=-1)[..., :k]
    orden = np.argsort(-np.take_along_axis(sims, part, axis=-1), axis=-1,
                       kind="stable")
    return np.take_along_axis(part, orden, axis=-1)


def verdad(base: np.ndarray, qs: np.ndarray, k: int) -> np.ndarray:
    """top-k exacto por coseno (los vectores estan normalizados)."""
    return top_k(qs @ base.T, k)


def ann_recall_at_k(aprox, exacto) -> float:
    """recall de un indice aproximado: |aprox ∩ exacto| / k, con el top-k
    exacto del mismo embedding y la misma metrica como referencia."""
    return len(set(np.asarray(aprox).tolist())
               & set(np.asarray(exacto).tolist())) / len(exacto)


def ann_recall_listas(aprox, exacto) -> float:
    return float(np.mean([ann_recall_at_k(a, e)
                          for a, e in zip(aprox, exacto)]))


# ---------------------------------------------------------------------------
# IVF minimo: k-means, insercion y busqueda
# ---------------------------------------------------------------------------

def _asignar(x: np.ndarray, cent: np.ndarray) -> np.ndarray:
    """centroide mas cercano en L2 (|c|^2 - 2 x.c; |x|^2 no cambia el orden)."""
    return ((cent ** 2).sum(1)[None, :] - 2.0 * (x @ cent.T)).argmin(1)


def _kmeans(x: np.ndarray, k: int, iters: int = 10,
            semilla: int = SEMILLA) -> tuple[np.ndarray, np.ndarray]:
    """k-means de Lloyd. una celda vacia conserva su centroide anterior; la
    asignacion devuelta es la de los centroides finales."""
    if not 1 <= k <= len(x):
        raise ValueError("k-means necesita 1 <= k <= numero de puntos")
    rng = np.random.default_rng(semilla)
    cent = x[rng.choice(len(x), k, replace=False)].astype(np.float64)
    for _ in range(iters):
        asign = _asignar(x, cent)
        sumas = np.zeros((k, x.shape[1]))
        np.add.at(sumas, asign, x)
        cuenta = np.bincount(asign, minlength=k)
        ok = cuenta > 0                          # sin division por cero
        cent[ok] = sumas[ok] / cuenta[ok, None]
    return cent.astype(np.float32), _asignar(x, cent)


class IndiceIVF:
    """IVF minimo sobre vectores normalizados: celdas por k-means en L2 y
    puntuacion por producto escalar (coseno) dentro de las celdas."""

    def __init__(self, base: np.ndarray, nlist: int = 128,
                 semilla: int = SEMILLA) -> None:
        self.base = base
        self.centroides, asign = _kmeans(base, nlist, semilla=semilla)
        self.listas = [np.flatnonzero(asign == j) for j in range(nlist)]

    def insertar(self, nuevos: np.ndarray) -> None:
        """alta sin reentrenar: los nuevos van a la celda del centroide viejo
        mas cercano. la base interna se amplia antes de asignar ids."""
        b0 = len(self.base)
        self.base = np.vstack([self.base, nuevos])
        asign = _asignar(nuevos, self.centroides)
        for j in range(len(self.listas)):
            self.listas[j] = np.concatenate(
                [self.listas[j], b0 + np.flatnonzero(asign == j)])

    def candidatos(self, q: np.ndarray, nprobe: int = 8) -> np.ndarray:
        d2 = ((self.centroides - q) ** 2).sum(1)
        celdas = np.argpartition(d2, nprobe - 1)[:nprobe]
        return np.concatenate([self.listas[c] for c in celdas])

    def buscar(self, q: np.ndarray, k: int, nprobe: int = 8) -> np.ndarray:
        cand = self.candidatos(q, nprobe)
        if not len(cand):
            return cand
        return cand[top_k(self.base[cand] @ q, k)]


def _evaluar_ivf(ivf: IndiceIVF, qs: np.ndarray, k: int,
                 nprobe: int) -> tuple[float, float]:
    """ann recall@k medio y candidatos evaluados por consulta."""
    exacto = verdad(ivf.base, qs, k)
    aprox = [ivf.buscar(x, k, nprobe) for x in qs]
    evaluados = np.mean([len(ivf.candidatos(x, nprobe)) for x in qs])
    return ann_recall_listas(aprox, exacto), float(evaluados)


# ---------------------------------------------------------------------------
# 2. ann recall segun la profundidad k
# ---------------------------------------------------------------------------

def simular_recall_k(n: int = 20_000, dim: int = 128, nq: int = 200) -> None:
    """ann recall@k de un IVF (nlist 128) segun k y nprobe: no es monotono en
    k, a diferencia del recall de relevancia."""
    rng = np.random.default_rng(SEMILLA + 1)
    base, centros = coleccion(n, dim)
    qs = consultas(centros, nq, rng)
    ivf = IndiceIVF(base, nlist=128)
    ks = (1, 5, 10, 20, 50, 100)
    sondas = (2, 4, 8)
    exactos = {k: verdad(base, qs, k) for k in ks}
    filas = []
    print("\nann recall@k de un IVF (nlist 128) segun k y nprobe")
    print("  k     " + "  ".join(f"nprobe={p}" for p in sondas))
    for k in ks:
        fila = [k]
        for p in sondas:
            aprox = [ivf.buscar(x, k, p) for x in qs]
            fila.append(round(ann_recall_listas(aprox, exactos[k]), 4))
        filas.append(tuple(fila))
        print(f"  {k:<4}  " + "  ".join(f"{v:<8.4f}" for v in fila[1:]))
    _escribir(os.path.join("data", "cap18_recallk.dat"),
              "ann recall@k de un IVF (nlist 128) frente al top-k exacto, "
              "segun k y nprobe (n=%d, dim=%d, %d consultas reservadas)"
              % (n, dim, nq), "k  np2  np4  np8", filas)


# ---------------------------------------------------------------------------
# 3. cuantizacion: float16, int8 y binaria empaquetada
# ---------------------------------------------------------------------------

def cuantizar_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 simetrico por vector: 255 niveles (-127..127) y una escala
    float32 por vector. un vector nulo usa escala 1."""
    max_abs = np.abs(x).max(1, keepdims=True)
    escala = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    return np.round(x / escala).astype(np.int8), escala


def empaquetar(x: np.ndarray) -> np.ndarray:
    """signo de cada componente, ocho por byte: d/8 bytes por vector."""
    return np.packbits(x > 0, axis=1)


def hamming(qb: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """bits distintos por xor y conteo de bits; para signos s, t en {-1,1}^d,
    s.t = d - 2H, de modo que ordenar por H es ordenar por s.t."""
    return np.bitwise_count(np.bitwise_xor(bb, qb)).sum(1)


def simular_cuantizacion(n: int = 30_000, dim: int = 128, nq: int = 200,
                         k: int = 10) -> None:
    """ann recall@10 frente al top-k exacto float32 y bytes por vector
    realmente ocupados por cada representacion."""
    rng = np.random.default_rng(SEMILLA + 1)
    base, centros = coleccion(n, dim)
    qs = consultas(centros, nq, rng)
    exacto = verdad(base, qs, k)
    b16 = base.astype(np.float16)
    q8, esc = cuantizar_int8(base)
    bb = empaquetar(base)
    rec16 = ann_recall_listas(verdad(b16.astype(np.float32), qs, k), exacto)
    rec8 = ann_recall_listas(verdad(q8.astype(np.float32) * esc, qs, k), exacto)
    qb = empaquetar(qs)
    recb = ann_recall_listas(
        [np.argsort(hamming(x, bb), kind="stable")[:k] for x in qb], exacto)
    filas = [("float32", base.nbytes // n, 0, 1.0),
             ("float16", b16.nbytes // n, 0, round(rec16, 4)),
             ("int8", q8.nbytes // n, esc.nbytes // n, round(rec8, 4)),
             ("binaria", bb.nbytes // n, 0, round(recb, 4))]
    print("\ncuantizacion: ann recall@10 y bytes por vector (dim %d)" % dim)
    print("  tipo      carga  escala  recall@10")
    for f in filas:
        print(f"  {f[0]:<8}  {f[1]:<5}  {f[2]:<6}  {f[3]}")
    _escribir(os.path.join("data", "cap18_cuantizacion.dat"),
              "ann recall@10 frente al top-k exacto float32 y bytes por vector "
              "(carga y escala por vector) de float32, float16, int8 y binaria "
              "empaquetada (n=%d, dim=%d)" % (n, dim),
              "tipo  bytes  escala  recall", filas)


def simular_binaria_dos_etapas(n: int = 30_000, dim: int = 128, nq: int = 200,
                               k: int = 10) -> None:
    """cribado: los R de menor hamming; reordenacion: coseno exacto con los
    vectores originales float32, que el sistema debe conservar."""
    rng = np.random.default_rng(SEMILLA + 1)
    base, centros, tema_b = coleccion(n, dim, con_tema=True)
    qs, tema_q = consultas(centros, nq, rng, con_tema=True)
    exacto = verdad(base, qs, k)
    bb, qb = empaquetar(base), empaquetar(qs)
    dist = np.array([hamming(x, bb) for x in qb])          # (nq, n)
    filas = []
    print("\ncribado binario + reordenacion exacta: ann recall@10 segun R")
    print("  R      recall@10")
    for R in (50, 100, 200, 500, 1000, 2000):
        res = []
        for i in range(nq):
            cand = np.argpartition(dist[i], R - 1)[:R]      # cribado binario
            res.append(cand[top_k(base[cand] @ qs[i], k)])  # reordenacion
        rec = ann_recall_listas(res, exacto)
        filas.append((R, round(rec, 4)))
        print(f"  {R:<5}  {rec:.4f}")
    # el signo separa temas: fraccion del cribado (R=50) del tema de la consulta
    cand = np.argpartition(dist, 49, axis=1)[:, :50]
    print("  cribado R=50 dentro del tema de la consulta: %.4f"
          % np.mean(tema_b[cand] == tema_q[:, None]))
    _escribir(os.path.join("data", "cap18_binaria.dat"),
              "ann recall@10 del cribado binario (top-R por hamming) seguido "
              "de reordenacion exacta con los originales, segun R "
              "(n=%d, dim=%d)" % (n, dim), "R  recall", filas)


# ---------------------------------------------------------------------------
# 5. latencia y rendimiento de la busqueda exacta
# ---------------------------------------------------------------------------

def simular_rendimiento(dim: int = 256, reps: int = 31,
                        lote: int = 64) -> None:
    """latencia mediana de una consulta (serie) y rendimiento de un lote de
    consultas en un producto de matrices. en serie, QPS = 1/latencia; en lote,
    el rendimiento supera ese valor y cada consulta espera al lote entero."""
    rng = np.random.default_rng(SEMILLA + 1)
    filas = []
    print("\nrendimiento de la busqueda exacta segun N (un hilo)")
    print("  N         lat_ms   qps_serie  qps_lote")
    for n in (1_000, 10_000, 100_000, 500_000, 1_000_000):
        base, centros = coleccion(n, dim)
        qs = consultas(centros, reps + lote, rng)
        top_k(base @ qs[0], 10)                             # calentamiento
        t = []
        for x in qs[:reps]:
            t0 = time.perf_counter()
            top_k(base @ x, 10)
            t.append(time.perf_counter() - t0)
        lat = float(np.median(t))
        tl = []
        for _ in range(3):
            t0 = time.perf_counter()
            top_k(qs[reps:] @ base.T, 10)
            tl.append(time.perf_counter() - t0)
        qps_lote = lote / min(tl)
        filas.append((n, round(lat * 1e3, 3), int(1 / lat), int(qps_lote)))
        print(f"  {n:<9} {lat*1e3:7.3f}  {int(1/lat):9d}  {int(qps_lote):8d}")
        del base
    _escribir(os.path.join("data", "cap18_rendimiento.dat"),
              "busqueda exacta segun N (dim %d, un hilo): latencia mediana de "
              "una consulta (ms), su inverso (qps en serie) y rendimiento de "
              "un lote de %d consultas (qps en lote)" % (dim, lote),
              "n  latencia_ms  qps_serie  qps_lote", filas)


# ---------------------------------------------------------------------------
# 6. deriva: reajustar frente a reindexar
# ---------------------------------------------------------------------------

def simular_deriva(n: int = 20_000, dim: int = 128, nq: int = 200,
                   k: int = 10) -> None:
    """un IVF entrenado con 30 temas recibe dos lotes de temas nuevos
    asignados a los centroides viejos. se mide el ann recall@10 y los
    candidatos evaluados con nprobe 8, con mas nprobe (reajuste) y tras
    reconstruir el k-means sobre la coleccion actual (reindexado)."""
    rng = np.random.default_rng(SEMILLA + 1)
    base, centros = coleccion(n, dim, n_temas=30)
    ivf = IndiceIVF(base, nlist=128)
    temas, pesos = [centros], [float(n)]
    filas = []

    def medir(fase: str, indice: IndiceIVF, qs: np.ndarray, sondas) -> None:
        for p in sondas:
            rec, ev = _evaluar_ivf(indice, qs, k, p)
            filas.append((fase, len(indice.base), p, round(rec, 4), int(ev)))
            print(f"  {fase:<11} {len(indice.base):<7} {p:<6} {rec:.4f}  "
                  f"{int(ev)}")

    print("\nderiva: ann recall@10 y candidatos evaluados")
    print("  fase        n       nprobe recall  evaluados")
    medir("inicial", ivf, consultas(centros, nq, rng), (8,))
    for paso in (1, 2):
        extra, c_extra = coleccion(n // 2, dim, n_temas=30,
                                   semilla=SEMILLA + 50 + paso)
        ivf.insertar(extra)
        temas.append(c_extra)
        pesos.append(float(n // 2))
        # las consultas siguen la mezcla actual de temas viejos y nuevos
        p = np.concatenate([np.full(len(c), w / len(c))
                            for c, w in zip(temas, pesos)])
        qs, tema_q = consultas(np.vstack(temas), nq, rng, p / p.sum(),
                               con_tema=True)
        medir(f"deriva{paso}", ivf, qs, (8, 16, 32) if paso == 2 else (8,))
    # la media mezcla dos poblaciones: consultas de temas viejos y nuevos
    exacto = verdad(ivf.base, qs, k)
    rec = np.array([ann_recall_at_k(ivf.buscar(x, k, 8), e)
                    for x, e in zip(qs, exacto)])
    viejo = tema_q < len(centros)
    print(f"  deriva2, nprobe 8: temas viejos {viejo.sum()} consultas, recall "
          f"{rec[viejo].mean():.4f}; nuevos {(~viejo).sum()}, recall "
          f"{rec[~viejo].mean():.4f}; por debajo de 0,5: {np.sum(rec < 0.5)}")
    reindex = IndiceIVF(ivf.base, nlist=128, semilla=SEMILLA + 2)
    medir("reindexado", reindex, qs, (8,))
    # el desequilibrio de las listas senala la deriva sin calcular recall
    for fase, indice in (("deriva2", ivf), ("reindexado", reindex)):
        tam = np.array([len(lista) for lista in indice.listas])
        print(f"  celdas {fase}: media {tam.mean():.2f}, maxima {tam.max()} "
              f"({tam.max() / tam.mean():.1f} veces la media)")
    _escribir(os.path.join("data", "cap18_deriva.dat"),
              "ann recall@10 y candidatos evaluados por consulta de un IVF "
              "(nlist 128) al insertar temas nuevos sin reentrenar, con mas "
              "nprobe (reajuste) y tras reconstruir el k-means (reindexado)",
              "fase  n  nprobe  recall  evaluados", filas)


# ---------------------------------------------------------------------------
# demostracion: informe de fidelidad de un indice para una consulta
# ---------------------------------------------------------------------------

def demostracion(n: int = 20_000, dim: int = 128, k: int = 10) -> None:
    """compara la respuesta del IVF con el top-k exacto de una consulta segun
    nprobe: el solape (ann recall), el puesto que da el indice al primer
    vecino exacto y un nDCG con la relevancia graduada por el puesto exacto y
    ganancia lineal (fidelidad del orden)."""
    rng = np.random.default_rng(SEMILLA + 2)
    base, centros = coleccion(n, dim)
    q = consultas(centros, 1, rng)[0]
    exacto = verdad(base, q[None, :], k)[0]
    grados = {int(d): k - i for i, d in enumerate(exacto)}   # 10, 9, ..., 1
    ivf = IndiceIVF(base, nlist=128)
    print("\ndemostracion: fidelidad del IVF frente al top-10 exacto")
    print("  nprobe  ann recall@10  puesto del 1.o exacto  nDCG@10 graduado")
    for nprobe in (1, 2, 4):
        r = ivf.buscar(q, k, nprobe)
        pos = np.flatnonzero(r == exacto[0])
        puesto = str(int(pos[0]) + 1) if len(pos) else "ausente"
        print(f"  {nprobe:<6}  {ann_recall_at_k(r, exacto):<13.3f}  "
              f"{puesto:<21}  "
              f"{ndcg_at_k(r.tolist(), grados, k, exponencial=False):.3f}")


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
