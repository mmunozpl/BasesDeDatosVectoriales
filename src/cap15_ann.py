"""capitulo 15 — indices de vecino aproximado (ANN).

implementa, en numpy y desde cero, las cuatro familias de indices de vecino mas
proximo aproximado y mide el compromiso central de todos: recall frente a coste.
el coste se mide como NUMERO DE COMPARACIONES de distancia (determinista y
reproducible, no contaminado por el overhead de Python), ademas de la latencia.

  - Flat: fuerza bruta, exacta (recall 1), la referencia.
  - LSH: hiperplanos aleatorios; los vectores parecidos caen en el mismo cubo.
  - IVF: k-means parte el espacio en celdas; se buscan solo unas pocas (nprobe).
  - PQ: cuantizacion de producto; comprime el vector y abarata la distancia.
  - HNSW: grafo navegable jerarquico; se salta hacia el mas proximo.

mide cinco cosas:
  1. IVF: recall y comparaciones segun nprobe (celdas visitadas).
  2. LSH: recall y candidatos segun el numero de bits de la firma.
  3. PQ: recall, error y compresion segun el numero de subcuantizadores.
  4. HNSW: recall y comparaciones segun ef (anchura de la busqueda).
  5. frontera: recall frente a coste de las familias en un mismo plano.

es Python puro con numpy (sin servicio ni GPU): un motor real usa FAISS, hnswlib
o el indice nativo del motor, pero la mecanica es la que este modulo deja a la
vista. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np

SEMILLA = 15


def anunciar() -> None:
    print("=" * 64)
    print("cap. 15 — indices de vecino aproximado (ANN)")
    print("recursos: python + numpy · cpu. no usa servicio, gpu ni faiss.")
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
# datos y verdad de referencia
# ---------------------------------------------------------------------------

def _coleccion(n: int, dim: int, n_temas: int = 50,
               semilla: int = SEMILLA) -> np.ndarray:
    """n vectores normalizados, agrupados en temas (estructura realista)."""
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((n_temas, dim)) * 3
    tema = rng.integers(0, n_temas, n)
    vecs = centros[tema] + rng.standard_normal((n, dim))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32)


def _verdad(base: np.ndarray, consultas: np.ndarray, k: int) -> np.ndarray:
    """top-k exacto (por coseno = producto, vectores normalizados)."""
    sims = consultas @ base.T
    return np.argsort(-sims, axis=1)[:, :k]


def recall(aprox: List[List[int]], exacto: np.ndarray) -> float:
    """recall@k medio: fraccion de los k verdaderos que el indice recupera."""
    tot = 0.0
    for i, ap in enumerate(aprox):
        verdad = set(exacto[i].tolist())
        tot += len(verdad & set(ap)) / len(verdad)
    return tot / len(aprox)


# ---------------------------------------------------------------------------
# k-means en numpy (para IVF y PQ)
# ---------------------------------------------------------------------------

def _kmeans(x: np.ndarray, k: int, iters: int = 12,
            semilla: int = SEMILLA) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd euclideo: devuelve centroides y la asignacion de cada punto."""
    rng = np.random.default_rng(semilla)
    cent = x[rng.choice(len(x), k, replace=False)].copy()
    asign = np.zeros(len(x), dtype=np.int32)
    dim = x.shape[1]
    for _ in range(iters):
        # ||x-c||^2 = ||x||^2 - 2 x·c + ||c||^2; el argmin no depende de ||x||^2
        cn = (cent ** 2).sum(1)
        asign = (cn[None, :] - 2.0 * (x @ cent.T)).argmin(1)
        # actualizacion vectorizada de los centroides (media por celda)
        sumas = np.zeros((k, dim), dtype=np.float64)
        np.add.at(sumas, asign, x)
        cuenta = np.bincount(asign, minlength=k)
        no_vacia = cuenta > 0
        cent[no_vacia] = (sumas[no_vacia]
                          / cuenta[no_vacia, None]).astype(np.float32)
    return cent.astype(np.float32), asign.astype(np.int32)


# ---------------------------------------------------------------------------
# 1. Flat: fuerza bruta exacta
# ---------------------------------------------------------------------------

class IndiceFlat:
    def __init__(self, base: np.ndarray) -> None:
        self.base = base

    def buscar(self, q: np.ndarray, k: int) -> Tuple[List[int], int]:
        sims = self.base @ q
        idx = np.argsort(-sims)[:k]
        return idx.tolist(), len(self.base)     # compara con todos


# ---------------------------------------------------------------------------
# 2. LSH: hiperplanos aleatorios
# ---------------------------------------------------------------------------

class IndiceLSH:
    """Hashing por hiperplanos aleatorios con varias tablas.

    Cada tabla firma los vectores con el signo de su producto con `bits`
    hiperplanos; los que comparten firma caen en el mismo cubo. La busqueda
    reune los candidatos de los cubos que casan en todas las tablas y los
    reordena de forma exacta.
    """

    def __init__(self, base: np.ndarray, bits: int = 12, tablas: int = 8,
                 semilla: int = SEMILLA) -> None:
        self.base = base
        rng = np.random.default_rng(semilla)
        d = base.shape[1]
        self.planos = [rng.standard_normal((bits, d)).astype(np.float32)
                       for _ in range(tablas)]
        self.cubos: List[Dict[int, List[int]]] = []
        for p in self.planos:
            firmas = (base @ p.T > 0)            # (n, bits)
            claves = self._empaquetar(firmas)
            cubo: Dict[int, List[int]] = {}
            for i, c in enumerate(claves):
                cubo.setdefault(int(c), []).append(i)
            self.cubos.append(cubo)

    @staticmethod
    def _empaquetar(firmas: np.ndarray) -> np.ndarray:
        pesos = (1 << np.arange(firmas.shape[1])).astype(np.int64)
        return firmas.astype(np.int64) @ pesos

    def buscar(self, q: np.ndarray, k: int) -> Tuple[List[int], int]:
        cand = set()
        for p, cubo in zip(self.planos, self.cubos):
            clave = int(self._empaquetar((q @ p.T > 0)[None, :])[0])
            cand.update(cubo.get(clave, ()))
        if not cand:
            return [], 0
        cand = np.fromiter(cand, dtype=np.int64)
        sims = self.base[cand] @ q               # reordena exacto
        orden = np.argsort(-sims)[:k]
        return cand[orden].tolist(), len(cand)


# ---------------------------------------------------------------------------
# 3. IVF: listas invertidas sobre celdas de k-means
# ---------------------------------------------------------------------------

class IndiceIVF:
    """Inverted File: k-means parte el espacio en `nlist` celdas; cada vector
    se asigna a su celda. La busqueda mira solo las `nprobe` celdas cuyo
    centroide esta mas cerca de la consulta."""

    def __init__(self, base: np.ndarray, nlist: int = 256,
                 semilla: int = SEMILLA) -> None:
        self.base = base
        self.centroides, asign = _kmeans(base, nlist, semilla=semilla)
        self.listas: List[np.ndarray] = [
            np.where(asign == j)[0] for j in range(nlist)]

    def buscar(self, q: np.ndarray, k: int,
               nprobe: int = 8) -> Tuple[List[int], int]:
        dc = ((self.centroides - q) ** 2).sum(1)
        celdas = np.argsort(dc)[:nprobe]
        cand = np.concatenate([self.listas[c] for c in celdas]) \
            if len(celdas) else np.empty(0, dtype=np.int64)
        if not len(cand):
            return [], len(self.centroides)
        sims = self.base[cand] @ q
        orden = np.argsort(-sims)[:k]
        # coste: comparar con nprobe centroides + los candidatos
        return cand[orden].tolist(), len(self.centroides) + len(cand)


# ---------------------------------------------------------------------------
# 4. PQ: cuantizacion de producto
# ---------------------------------------------------------------------------

class IndicePQ:
    """Product Quantization: parte cada vector en `m` trozos y codifica cada
    trozo con uno de 256 centroides (un byte). La distancia se estima sumando,
    por trozo, distancias precomputadas a los centroides (ADC)."""

    def __init__(self, base: np.ndarray, m: int = 8,
                 semilla: int = SEMILLA) -> None:
        self.m = m
        self.dim = base.shape[1]
        self.sub = self.dim // m
        self.codebooks = []                      # m libros de 256 centroides
        self.codigos = np.empty((len(base), m), dtype=np.uint8)
        for j in range(m):
            trozo = base[:, j * self.sub:(j + 1) * self.sub]
            cent, asign = _kmeans(trozo, 256, iters=10, semilla=semilla + j)
            self.codebooks.append(cent)
            self.codigos[:, j] = asign.astype(np.uint8)

    def memoria_relativa(self) -> float:
        """bytes del codigo (m) frente a los del vector float32 (4*dim)."""
        return self.m / (4 * self.dim)

    def buscar(self, q: np.ndarray, k: int) -> Tuple[List[int], int]:
        # tabla de distancias: por trozo, q_sub a cada uno de los 256 codigos
        tabla = np.empty((self.m, 256), dtype=np.float32)
        for j in range(self.m):
            qs = q[j * self.sub:(j + 1) * self.sub]
            tabla[j] = ((self.codebooks[j] - qs) ** 2).sum(1)
        # distancia aproximada = suma de las distancias de cada trozo (ADC)
        dist = tabla[np.arange(self.m), self.codigos].sum(1)
        idx = np.argsort(dist)[:k]
        return idx.tolist(), len(self.codigos)


# ---------------------------------------------------------------------------
# 5. HNSW: grafo navegable jerarquico
# ---------------------------------------------------------------------------

class IndiceHNSW:
    """Hierarchical Navigable Small World: un grafo de vecindad por capas. Las
    capas altas son escasas y permiten saltos largos; las bajas, densas, afinan.
    La busqueda baja de capa en capa con una busqueda voraz por haz (ef)."""

    def __init__(self, base: np.ndarray, M: int = 16, efc: int = 64,
                 semilla: int = SEMILLA) -> None:
        self.base = base
        self.M = M
        self.efc = efc
        self.rng = np.random.default_rng(semilla)
        self.niveles: List[int] = []
        self.grafo: List[List[set]] = []         # grafo[nodo][capa] = vecinos
        self.entrada = 0
        self.maxnivel = 0
        self.comparaciones = 0
        for i in range(len(base)):
            self._insertar(i)

    def _nivel_aleatorio(self) -> int:
        return int(-np.log(self.rng.random()) * (1.0 / np.log(self.M)))

    def _dist(self, i: int, q: np.ndarray) -> float:
        self.comparaciones += 1
        return 1.0 - float(self.base[i] @ q)     # 1 - coseno

    def _buscar_capa(self, q: np.ndarray, entradas: List[int],
                     capa: int, ef: int) -> List[Tuple[float, int]]:
        visitados = set(entradas)
        cand = [(self._dist(e, q), e) for e in entradas]
        cand.sort()
        mejores = list(cand)
        while cand:
            d, c = cand.pop(0)
            if d > mejores[-1][0] and len(mejores) >= ef:
                break
            for v in self.grafo[c][capa]:
                if v not in visitados:
                    visitados.add(v)
                    dv = self._dist(v, q)
                    if len(mejores) < ef or dv < mejores[-1][0]:
                        cand.append((dv, v))
                        cand.sort()
                        mejores.append((dv, v))
                        mejores.sort()
                        mejores = mejores[:ef]
        return mejores

    def _seleccionar(self, candidatos: List[Tuple[float, int]],
                     m: int) -> List[Tuple[float, int]]:
        """Heuristica de seleccion de vecinos de HNSW: recorre los candidatos
        de mas cercano a mas lejano y acepta uno solo si esta mas cerca del
        punto de referencia que de cualquier vecino ya elegido. Promueve la
        diversidad de conexiones y, con ella, la navegabilidad del grafo."""
        sel: List[Tuple[float, int]] = []
        for d, c in candidatos:
            if len(sel) >= m:
                break
            bueno = True
            for _, e in sel:
                self.comparaciones += 1
                if (1.0 - float(self.base[c] @ self.base[e])) < d:
                    bueno = False
                    break
            if bueno:
                sel.append((d, c))
        return sel

    def _podar(self, v: int, capa: int, mmax: int) -> None:
        cand = sorted((1.0 - float(self.base[w] @ self.base[v]), w)
                      for w in self.grafo[v][capa])
        self.grafo[v][capa] = set(w for _, w in self._seleccionar(cand, mmax))

    def _insertar(self, i: int) -> None:
        nivel = self._nivel_aleatorio()
        self.niveles.append(nivel)
        self.grafo.append([set() for _ in range(nivel + 1)])
        if i == 0:
            self.entrada = 0
            self.maxnivel = nivel
            return
        q = self.base[i]
        ep = [self.entrada]
        # bajar por las capas por encima del nivel del nodo nuevo
        for capa in range(self.maxnivel, nivel, -1):
            ep = [self._buscar_capa(q, ep, capa, 1)[0][1]]
        # conectar en cada capa <= nivel del nodo
        for capa in range(min(nivel, self.maxnivel), -1, -1):
            cand = self._buscar_capa(q, ep, capa, self.efc)
            mmax = 2 * self.M if capa == 0 else self.M
            sel = self._seleccionar(cand, self.M)
            for _, v in sel:
                self.grafo[i][capa].add(v)
                self.grafo[v][capa].add(i)
                if len(self.grafo[v][capa]) > mmax:
                    self._podar(v, capa, mmax)
            ep = [v for _, v in cand] or ep
        if nivel > self.maxnivel:
            self.maxnivel = nivel
            self.entrada = i

    def buscar(self, q: np.ndarray, k: int,
               ef: int = 32) -> Tuple[List[int], int]:
        self.comparaciones = 0
        ep = [self.entrada]
        for capa in range(self.maxnivel, 0, -1):
            ep = [self._buscar_capa(q, ep, capa, 1)[0][1]]
        mejores = self._buscar_capa(q, ep, 0, max(ef, k))
        res = [v for _, v in mejores[:k]]
        return res, self.comparaciones


# ---------------------------------------------------------------------------
# medidas
# ---------------------------------------------------------------------------

def _latencia(indice, consultas, k, **kw
              ) -> Tuple[List[List[int]], float, int]:
    """devuelve los resultados, la latencia media (ms) y las comparaciones
    medias por consulta, en una sola pasada."""
    res, comps = [], []
    t0 = time.perf_counter()
    for q in consultas:
        r, c = indice.buscar(q, k, **kw)
        res.append(r)
        comps.append(c)
    dt = (time.perf_counter() - t0) / len(consultas)
    return res, dt * 1e3, int(np.mean(comps))


def simular_ivf(n: int = 20_000, dim: int = 128, nq: int = 200,
                k: int = 10) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    ivf = IndiceIVF(base, nlist=256)
    filas = []
    print("\nIVF (nlist=256): recall y comparaciones segun nprobe")
    print("  nprobe  recall   comparaciones  %_colec")
    for nprobe in (1, 2, 4, 8, 16, 32, 64):
        res, lat, comp = _latencia(ivf, consultas, k, nprobe=nprobe)
        rec = recall(res, exacto)
        pct = 100.0 * comp / n
        filas.append((nprobe, round(rec, 4), comp, round(pct, 2),
                      round(lat, 3)))
        print(f"  {nprobe:<6}  {rec:.4f}   {comp:<13}  {pct:.2f}")
    _escribir(os.path.join("data", "cap15_ivf.dat"),
              "IVF: recall@10, comparaciones, %% de la coleccion y latencia "
              "(ms) segun nprobe (n=%d, nlist=256)" % n,
              "nprobe  recall  comparaciones  pct  latencia_ms", filas)


def simular_lsh(n: int = 20_000, dim: int = 128, nq: int = 200,
                k: int = 10) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    filas = []
    print("\nLSH (16 tablas): recall y candidatos segun los bits de la firma")
    print("  bits  recall   candidatos  %_colec")
    for bits in (4, 6, 8, 10, 12, 14):
        lsh = IndiceLSH(base, bits=bits, tablas=16)
        res, lat, cand = _latencia(lsh, consultas, k)
        rec = recall(res, exacto)
        pct = 100.0 * cand / n
        filas.append((bits, round(rec, 4), cand, round(pct, 2),
                      round(lat, 3)))
        print(f"  {bits:<4}  {rec:.4f}   {cand:<10}  {pct:.2f}")
    _escribir(os.path.join("data", "cap15_lsh.dat"),
              "LSH: recall@10, candidatos, %% de la coleccion y latencia (ms) "
              "segun los bits de la firma (n=%d, 16 tablas)" % n,
              "bits  recall  candidatos  pct  latencia_ms", filas)


def simular_pq(n: int = 20_000, dim: int = 128, nq: int = 200,
               k: int = 10) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    filas = []
    print("\nPQ: recall y compresion segun el numero de subcuantizadores m")
    print("  m    bytes  compresion  recall")
    for m in (4, 8, 16, 32):
        pq = IndicePQ(base, m=m)
        res, lat, _ = _latencia(pq, consultas, k)
        rec = recall(res, exacto)
        comp = pq.memoria_relativa()
        filas.append((m, m, round(1.0 / comp, 1), round(rec, 4),
                      round(lat, 3)))
        print(f"  {m:<4}  {m:<5}  {1.0/comp:<10.1f}  {rec:.4f}")
    _escribir(os.path.join("data", "cap15_pq.dat"),
              "PQ: bytes por vector, factor de compresion frente a float32, "
              "recall@10 y latencia (ms) segun m (n=%d, dim=%d)" % (n, dim),
              "m  bytes  compresion  recall  latencia_ms", filas)


def simular_hnsw(n: int = 20_000, dim: int = 128, nq: int = 200,
                 k: int = 10) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    print("\nconstruyendo HNSW (n=%d, M=32)..." % n)
    hnsw = IndiceHNSW(base, M=32, efc=200)
    filas = []
    print("HNSW: recall y comparaciones segun ef")
    print("  ef    recall   comparaciones  %_colec")
    for ef in (8, 16, 32, 64, 128, 256):
        res, lat, comp = _latencia(hnsw, consultas, k, ef=ef)
        rec = recall(res, exacto)
        pct = 100.0 * comp / n
        filas.append((ef, round(rec, 4), comp, round(pct, 2), round(lat, 3)))
        print(f"  {ef:<4}  {rec:.4f}   {comp:<13}  {pct:.2f}")
    _escribir(os.path.join("data", "cap15_hnsw.dat"),
              "HNSW: recall@10, comparaciones, %% de la coleccion y latencia "
              "(ms) segun ef (n=%d, M=32)" % n,
              "ef  recall  comparaciones  pct  latencia_ms", filas)


def simular_frontera(n: int = 20_000, dim: int = 128, nq: int = 200,
                     k: int = 10) -> None:
    """recall frente a coste (%% de la coleccion comparada) de las tres
    familias en un mismo plano: la frontera del compromiso ANN."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    print("\nfrontera recall vs coste (%_colec) de las tres familias")
    ivf = IndiceIVF(base, nlist=256)
    hnsw = IndiceHNSW(base, M=32, efc=200)
    fil_ivf, fil_hnsw, fil_lsh = [], [], []
    for nprobe in (1, 2, 4, 8, 16, 32):
        res = [ivf.buscar(q, k, nprobe=nprobe) for q in consultas]
        rec = recall([r for r, _ in res], exacto)
        pct = 100.0 * np.mean([c for _, c in res]) / n
        fil_ivf.append((round(pct, 2), round(rec, 4)))
    for ef in (8, 16, 32, 64, 128, 256):
        res = [hnsw.buscar(q, k, ef=ef) for q in consultas]
        rec = recall([r for r, _ in res], exacto)
        pct = 100.0 * np.mean([c for _, c in res]) / n
        fil_hnsw.append((round(pct, 2), round(rec, 4)))
    for bits in (14, 12, 10, 8, 6, 4):
        lsh = IndiceLSH(base, bits=bits, tablas=16)
        res = [lsh.buscar(q, k) for q in consultas]
        rec = recall([r for r, _ in res], exacto)
        pct = 100.0 * np.mean([c for _, c in res]) / n
        fil_lsh.append((round(pct, 2), round(rec, 4)))
    _escribir(os.path.join("data", "cap15_frontera_ivf.dat"),
              "frontera IVF: %% de la coleccion comparada vs recall@10",
              "pct  recall", fil_ivf)
    _escribir(os.path.join("data", "cap15_frontera_hnsw.dat"),
              "frontera HNSW: %% de la coleccion comparada vs recall@10",
              "pct  recall", fil_hnsw)
    _escribir(os.path.join("data", "cap15_frontera_lsh.dat"),
              "frontera LSH: %% de la coleccion comparada vs recall@10",
              "pct  recall", fil_lsh)
    print("  IVF:", fil_ivf[-1], " HNSW:", fil_hnsw[-1], " LSH:", fil_lsh[-1])


def comparativa(n: int = 20_000, dim: int = 128, nq: int = 200,
                k: int = 10) -> None:
    """tabla resumen: recall, coste y memoria del mejor punto de cada familia."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    filas = []
    flat = IndiceFlat(base)
    res, lat, _ = _latencia(flat, consultas, k)
    filas.append(("Flat", round(recall(res, exacto), 3), 100.0,
                  round(lat, 3)))
    ivf = IndiceIVF(base, nlist=256)
    res = [ivf.buscar(q, k, nprobe=16) for q in consultas]
    pct = 100.0 * np.mean([c for _, c in res]) / n
    filas.append(("IVF", round(recall([r for r, _ in res], exacto), 3),
                  round(pct, 1), None))
    lsh = IndiceLSH(base, bits=6, tablas=16)
    res = [lsh.buscar(q, k) for q in consultas]
    pct = 100.0 * np.mean([c for _, c in res]) / n
    filas.append(("LSH", round(recall([r for r, _ in res], exacto), 3),
                  round(pct, 1), None))
    hnsw = IndiceHNSW(base, M=32, efc=200)
    res = [hnsw.buscar(q, k, ef=128) for q in consultas]
    pct = 100.0 * np.mean([c for _, c in res]) / n
    filas.append(("HNSW", round(recall([r for r, _ in res], exacto), 3),
                  round(pct, 1), None))
    print("\ncomparativa (n=%d): recall y %% de la coleccion comparada" % n)
    for f in filas:
        print(" ", f)
    _escribir(os.path.join("data", "cap15_comparativa.dat"),
              "comparativa: recall@10 y %% de la coleccion comparada por "
              "familia (n=%d, mejor punto operativo)" % n,
              "metodo  recall  pct_comparado", [f[:3] for f in filas])


def simular_rerank(n: int = 20_000, dim: int = 128, nq: int = 200,
                   k: int = 10) -> None:
    """efecto de la REORDENACION sobre PQ: recuperar los R primeros candidatos
    por PQ (aproximado) y reordenar los k finales con la distancia exacta.
    el recall@10 sube deprisa con R: PQ comprime, la reordenacion recupera."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    pq = IndicePQ(base, m=16)
    filas = []
    print("\nreordenacion sobre PQ (m=16): recall@10 segun la profundidad R")
    print("  R     recall")
    for R in (10, 20, 50, 100, 200, 500):
        res = []
        for q in consultas:
            cand, _ = pq.buscar(q, R)            # R candidatos por PQ
            cand = np.array(cand)
            orden = np.argsort(-(base[cand] @ q))[:k]   # reordena exacto
            res.append(cand[orden].tolist())
        rec = recall(res, exacto)
        filas.append((R, round(rec, 4)))
        print(f"  {R:<4}  {rec:.4f}")
    _escribir(os.path.join("data", "cap15_rerank.dat"),
              "recall@10 de PQ (m=16) tras reordenar los R primeros candidatos "
              "con la distancia exacta (n=%d)" % n,
              "R  recall", filas)


def simular_nlist(n: int = 20_000, dim: int = 128, nq: int = 200,
                  k: int = 10, nprobe: int = 8) -> None:
    """efecto de nlist en IVF, a nprobe fijo: mas celdas -> celdas mas pequenas
    -> se compara con menos vectores (mas rapido) pero baja el recall. el
    equilibrio tipico de nlist es del orden de la raiz de n."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _coleccion(n, dim)
    consultas = base[rng.choice(n, nq)] + 0.3 * rng.standard_normal(
        (nq, dim)).astype(np.float32)
    consultas /= np.linalg.norm(consultas, axis=1, keepdims=True)
    exacto = _verdad(base, consultas, k)
    filas = []
    print("\nIVF: recall y coste segun nlist (nprobe=%d fijo)" % nprobe)
    print("  nlist  recall   %_colec")
    for nlist in (16, 64, 256, 1024, 4096):
        ivf = IndiceIVF(base, nlist=nlist)
        res, lat, comp = _latencia(ivf, consultas, k, nprobe=nprobe)
        rec = recall(res, exacto)
        pct = 100.0 * comp / n
        filas.append((nlist, round(rec, 4), round(pct, 2)))
        print(f"  {nlist:<5}  {rec:.4f}   {pct:.2f}")
    _escribir(os.path.join("data", "cap15_nlist.dat"),
              "IVF: recall@10 y %% de la coleccion comparada segun nlist, a "
              "nprobe=%d fijo (n=%d)" % (nprobe, n),
              "nlist  recall  pct", filas)


def demostracion(n: int = 20_000, dim: int = 128, k: int = 15) -> None:
    """15 vecinos exactos frente a los del indice aproximado (HNSW)."""
    rng = np.random.default_rng(SEMILLA + 2)
    base = _coleccion(n, dim)
    q = base[0] + 0.3 * rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    exacto = np.argsort(-(base @ q))[:k].tolist()
    hnsw = IndiceHNSW(base, M=32, efc=200)
    aprox, _ = hnsw.buscar(q, k, ef=64)
    comunes = len(set(exacto) & set(aprox))
    print(f"\ndemostracion: {k} vecinos exacto vs aproximado (HNSW, ef=64)")
    print(f"  coincidencias: {comunes}/{k}")
    print("  rank  exacto   aproximado")
    for r in range(k):
        a = aprox[r] if r < len(aprox) else -1
        print(f"  {r:<4}  {exacto[r]:<7}  {a}")


def main() -> None:
    anunciar()
    simular_ivf()
    simular_lsh()
    simular_pq()
    simular_hnsw()
    simular_rerank()
    simular_nlist()
    simular_frontera()
    comparativa()
    demostracion()


if __name__ == "__main__":
    main()
