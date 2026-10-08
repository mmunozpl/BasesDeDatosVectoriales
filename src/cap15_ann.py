"""capitulo 15: indices de vecino aproximado (ANN).

implementa en numpy, desde cero y con fines didacticos, el nucleo de cuatro
tecnicas de busqueda de vecino aproximado y mide el compromiso que comparten:
recall frente a trabajo. no son indices de produccion: omiten, entre otras
cosas, la persistencia, la concurrencia, el borrado y las optimizaciones de
bajo nivel (SIMD, disposicion en memoria) de bibliotecas como FAISS o hnswlib.

  - Flat: barrido exacto, la referencia (recall 1 frente a su propia verdad).
  - LSH: hiperplanos aleatorios; cada bit coincide con P = 1 - angulo/pi.
  - IVF: k-means parte el espacio en celdas; se exploran solo nprobe.
  - PQ: cuantizacion de producto; comprime el vector y estima la distancia con
    tablas (ADC). sola, sigue evaluando los n codigos: barrido O(n*m).
  - HNSW: grafo navegable jerarquico con la heuristica de diversidad.

el trabajo se cuenta como EVALUACIONES por consulta de productos de dimension d
(candidatos, centroides e hiperplanos) y se expresa en porcentaje de la
coleccion. es una medida algoritmica reproducible, pero no una unidad universal
de coste: una distancia PQ por tablas, un hash, un producto denso y un salto de
grafo cuestan distinto en CPU, memoria y localidad, y el recuento depende
ademas de la variante del algoritmo. por eso se registra tambien la latencia de
esta maqueta, que depende de la maquina y del lenguaje.

la coleccion es sintetica (cumulos gaussianos, vectores normalizados) y las
consultas se reservan: se generan con la coleccion pero no se insertan. la
verdad de referencia es el barrido exacto por producto interno (igual al
coseno con norma 1), con los empates resueltos por el indice menor.

mide:
  1. LSH: probabilidad de colision por bit segun el angulo, frente a 1-angulo/pi.
  2. LSH: recall y evaluaciones segun los bits de la firma (16 tablas).
  3. IVF: recall y evaluaciones segun nprobe (nlist = 256).
  4. PQ: recall sin reordenar y compresion de los codigos segun m.
  5. HNSW: recall y evaluaciones segun ef, con y sin la heuristica de
     diversidad, y el grado medio de la capa 0.
  6. reordenacion exacta de los R primeros candidatos de PQ.
  7. nlist a nprobe fijo, separando el barrido de centroides y el de listas.
  8. evaluaciones de HNSW segun el tamano de la coleccion, a ef fijo.
  9. comparativa a igual objetivo de recall, con la latencia de la maqueta.
 10. los mismos indices sobre ruido isotropico, sin temas.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np

SEMILLA = 15
N, DIM, NQ, K = 20_000, 128, 200, 10
OBJETIVOS = (0.90, 0.95)                 # recall@10 de la comparativa


def anunciar() -> None:
    print("=" * 64)
    print("cap. 15: indices de vecino aproximado (ANN)")
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

def _coleccion(n: int, dim: int, n_temas: int = 50, escala: float = 1.0,
               semilla: int = SEMILLA, con_tema: bool = False):
    """n vectores normalizados, agrupados en temas (cumulos gaussianos). con
    escala 1, el centro del tema y el ruido de cada vector pesan lo mismo.
    con con_tema devuelve tambien el tema de cada vector."""
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((n_temas, dim)) * escala
    tema = rng.integers(0, n_temas, n)
    vecs = centros[tema] + rng.standard_normal((n, dim))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    if con_tema:
        return vecs.astype(np.float32), tema
    return vecs.astype(np.float32)


def datos(n: int = N, dim: int = DIM, nq: int = NQ,
          semilla: int = SEMILLA) -> Tuple[np.ndarray, np.ndarray]:
    """coleccion de n vectores y nq consultas reservadas de la misma
    distribucion: se generan juntas y las consultas no se insertan."""
    todo = _coleccion(n + nq, dim, semilla=semilla)
    return todo[:n], todo[n:]


def verdad(base: np.ndarray, consultas: np.ndarray, k: int) -> np.ndarray:
    """top-k exacto por producto interno (= coseno: vectores normalizados).
    el orden estable resuelve los empates por el indice menor."""
    sims = consultas @ base.T
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def recall(aprox: List[List[int]], exacto: np.ndarray) -> float:
    """recall@k medio: fraccion del top-k exacto presente en el top-k
    aproximado. no penaliza el orden dentro del conjunto."""
    tot = 0.0
    for i, ap in enumerate(aprox):
        ver = set(exacto[i].tolist())
        tot += len(ver & set(ap)) / len(ver)
    return tot / len(aprox)


def _top(puntos: np.ndarray, k: int, mayor: bool = True) -> np.ndarray:
    """indices de los k mejores con seleccion parcial O(n) y orden de los k."""
    k = min(k, len(puntos))
    clave = -puntos if mayor else puntos
    idx = np.argpartition(clave, k - 1)[:k]
    return idx[np.argsort(clave[idx], kind="stable")]


# ---------------------------------------------------------------------------
# k-means en numpy (para IVF y PQ)
# ---------------------------------------------------------------------------

def _kmeans(x: np.ndarray, k: int, iters: int = 12,
            semilla: int = SEMILLA) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd didactico: inicializacion aleatoria e iteraciones fijas, sin
    k-means++ ni criterio de convergencia. una celda que se vacia se reinicia
    en un punto elegido al azar."""
    if not 1 <= k <= len(x):
        raise ValueError(f"k = {k} exige entre 1 y {len(x)} centroides")
    rng = np.random.default_rng(semilla)
    cent = x[rng.choice(len(x), k, replace=False)].astype(np.float32)

    def asignar() -> np.ndarray:
        # ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2; el argmin no depende de ||x||
        cn = (cent ** 2).sum(1)
        return (cn[None, :] - 2.0 * (x @ cent.T)).argmin(1)

    for _ in range(iters):
        asign = asignar()
        sumas = np.zeros((k, x.shape[1]), dtype=np.float64)
        np.add.at(sumas, asign, x)
        cuenta = np.bincount(asign, minlength=k)
        llena = cuenta > 0
        cent[llena] = (sumas[llena] / cuenta[llena, None]).astype(np.float32)
        vacias = np.flatnonzero(~llena)
        if len(vacias):                          # se reinician las vacias
            cent[vacias] = x[rng.choice(len(x), len(vacias), replace=False)]
    return cent, asignar().astype(np.int32)


# ---------------------------------------------------------------------------
# 1. Flat: barrido exacto
# ---------------------------------------------------------------------------

class IndiceFlat:
    def __init__(self, base: np.ndarray) -> None:
        self.base = base

    def buscar(self, q: np.ndarray, k: int) -> Tuple[List[int], int]:
        sims = self.base @ q
        return _top(sims, k).tolist(), len(self.base)     # evalua todos


# ---------------------------------------------------------------------------
# 2. LSH: hiperplanos aleatorios
# ---------------------------------------------------------------------------

class IndiceLSH:
    """Hashing por hiperplanos aleatorios (por el origen) con varias tablas.

    Cada bit es el signo del producto con un hiperplano, y dos vectores que
    forman un angulo theta coinciden en un bit con probabilidad 1 - theta/pi.
    Exigir los `bits` de una tabla amplifica con AND (cubos selectivos);
    aceptar la coincidencia en cualquiera de las `tablas` amplifica con OR
    (recupera recall). Los candidatos se reordenan de forma exacta.
    """

    MAX_BITS = 62                                # la clave cabe en un int64

    def __init__(self, base: np.ndarray, bits: int = 12, tablas: int = 8,
                 semilla: int = SEMILLA) -> None:
        if not 1 <= bits <= self.MAX_BITS:
            raise ValueError("la clave empaqueta la firma en un int64: "
                             f"entre 1 y {self.MAX_BITS} bits")
        self.base, self.bits, self.tablas = base, bits, tablas
        rng = np.random.default_rng(semilla)
        d = base.shape[1]
        self.planos = [rng.standard_normal((bits, d)).astype(np.float32)
                       for _ in range(tablas)]
        self.cubos: List[Dict[int, List[int]]] = []
        for p in self.planos:
            claves = self._clave(base @ p.T > 0)     # (n, bits) -> n claves
            cubo: Dict[int, List[int]] = {}
            for i, c in enumerate(claves):
                cubo.setdefault(int(c), []).append(i)
            self.cubos.append(cubo)

    @staticmethod
    def _clave(firma: np.ndarray) -> np.ndarray:
        """bits -> entero. solo para firmas cortas (hasta 62 bits); una mas
        larga necesitaria varios enteros o bytes."""
        pesos = 1 << np.arange(firma.shape[-1], dtype=np.int64)
        return firma.astype(np.int64) @ pesos

    def buscar(self, q: np.ndarray, k: int) -> Tuple[List[int], int]:
        cand = set()
        for p, cubo in zip(self.planos, self.cubos):
            cand.update(cubo.get(int(self._clave(q @ p.T > 0)), ()))
        firmar = self.bits * self.tablas        # productos de la firma
        cand = np.fromiter(cand, dtype=np.int64)
        if cand.size == 0:
            return [], firmar
        sims = self.base[cand] @ q               # reordena exacto
        return cand[_top(sims, k)].tolist(), firmar + cand.size


# ---------------------------------------------------------------------------
# 3. IVF: listas invertidas sobre celdas de k-means
# ---------------------------------------------------------------------------

class IndiceIVF:
    """Inverted File: k-means parte el espacio en `nlist` celdas y cada vector
    va a la lista de su celda. La busqueda barre los centroides, elige los
    `nprobe` mas proximos y reordena de forma exacta sus listas.

    Elegir celdas por distancia euclidea y ordenar por producto interno es
    coherente porque la practica normaliza los vectores: con norma 1,
    ||a - b||^2 = 2 - 2 a.b, y euclidea, coseno y producto ordenan igual. Para
    otras metricas habria que adaptar el entrenamiento y la seleccion (para el
    coseno estricto es habitual el k-means esferico, con centroides
    normalizados)."""

    def __init__(self, base: np.ndarray, nlist: int = 256,
                 semilla: int = SEMILLA) -> None:
        self.base = base
        self.centroides, asign = _kmeans(base, nlist, semilla=semilla)
        self.listas: List[np.ndarray] = [
            np.flatnonzero(asign == j) for j in range(nlist)]

    def buscar(self, q: np.ndarray, k: int,
               nprobe: int = 8) -> Tuple[List[int], int]:
        nlist = len(self.centroides)
        dc = ((self.centroides - q) ** 2).sum(1)    # barrido de centroides
        celdas = _top(dc, min(nprobe, nlist), mayor=False)
        cand = np.concatenate([self.listas[c] for c in celdas])
        if cand.size == 0:
            return [], nlist
        sims = self.base[cand] @ q                   # reordena exacto
        # trabajo: los nlist centroides mas los candidatos de las listas
        return cand[_top(sims, k)].tolist(), nlist + cand.size


# ---------------------------------------------------------------------------
# 4. PQ: cuantizacion de producto
# ---------------------------------------------------------------------------

class IndicePQ:
    """Product Quantization: divide las dimensiones en `m` subespacios y en
    cada uno aprende con k-means un libro de `ks` centroides; cada trozo del
    vector se guarda como el indice de su centroide (un byte si ks <= 256). La
    distancia se estima sumando, por trozo, distancias precomputadas de la
    consulta a los centroides (ADC). Sola, evalua los n codigos: reduce la
    memoria y el coste de cada distancia, no el numero de candidatos."""

    def __init__(self, base: np.ndarray, m: int = 8, ks: int = 256,
                 semilla: int = SEMILLA) -> None:
        n, d = base.shape
        if d % m:
            raise ValueError(f"la dimension {d} debe ser divisible por m = {m}")
        if not 2 <= ks <= 256:
            raise ValueError("ks entre 2 y 256: cada codigo ocupa un byte")
        if n < ks:
            raise ValueError(f"entrenar {ks} centroides exige al menos {ks} "
                             "vectores")
        self.m, self.ks, self.dim, self.sub = m, ks, d, d // m
        self.codebooks = []                      # m libros de ks centroides
        self.codigos = np.empty((n, m), dtype=np.uint8)
        for j in range(m):
            trozo = base[:, j * self.sub:(j + 1) * self.sub]
            cent, asign = _kmeans(trozo, ks, iters=10, semilla=semilla + j)
            self.codebooks.append(cent)
            self.codigos[:, j] = asign.astype(np.uint8)

    def memoria_relativa(self) -> float:
        """bytes del codigo (m) frente a los del vector float32 (4*dim); no
        cuenta identificadores, libros de codigos ni otras estructuras."""
        return self.m / (4 * self.dim)

    def buscar(self, q: np.ndarray, k: int) -> Tuple[List[int], int]:
        # tabla de distancias: por trozo, q_sub a cada uno de los ks centroides
        tabla = np.empty((self.m, self.ks), dtype=np.float32)
        for j in range(self.m):
            qs = q[j * self.sub:(j + 1) * self.sub]
            tabla[j] = ((self.codebooks[j] - qs) ** 2).sum(1)
        # cada byte indexa la tabla de su trozo y se suman las m distancias
        dist = tabla[np.arange(self.m), self.codigos].sum(1)
        return _top(dist, k, mayor=False).tolist(), len(self.codigos)


# ---------------------------------------------------------------------------
# 5. HNSW: grafo navegable jerarquico
# ---------------------------------------------------------------------------

class IndiceHNSW:
    """Hierarchical Navigable Small World: un grafo de vecindad por capas. Las
    capas altas son escasas y permiten saltos largos; las bajas, densas, afinan.
    La busqueda baja de capa en capa con una busqueda voraz por haz (ef). La
    distancia es 1 - a.b, que es la distancia coseno porque la base esta
    normalizada (se comprueba al construir)."""

    def __init__(self, base: np.ndarray, M: int = 16, efc: int = 64,
                 diversidad: bool = True, semilla: int = SEMILLA) -> None:
        normas = np.linalg.norm(base, axis=1)
        if not np.allclose(normas, 1.0, atol=1e-3):
            raise ValueError("la distancia 1 - a.b exige vectores normalizados")
        self.base = base
        self.M = M
        self.efc = efc
        self.diversidad = diversidad
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
        punto de referencia que de cualquier vecino ya elegido. Favorece
        enlaces hacia regiones distintas. Sin diversidad, se toman los m mas
        proximos."""
        if not self.diversidad:
            return candidatos[:m]
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

    def grado_medio(self, capa: int = 0) -> float:
        """enlaces medios por nodo en una capa."""
        g = [len(n[capa]) for n in self.grafo if len(n) > capa]
        return float(np.mean(g))

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
              ) -> Tuple[List[List[int]], float, float]:
    """devuelve los resultados, la latencia media (ms) y las evaluaciones
    medias por consulta, en una sola pasada."""
    res, comps = [], []
    indice.buscar(consultas[0], k, **kw)          # calentamiento
    t0 = time.perf_counter()
    for q in consultas:
        r, c = indice.buscar(q, k, **kw)
        res.append(r)
        comps.append(c)
    dt = (time.perf_counter() - t0) / len(consultas)
    return res, dt * 1e3, float(np.mean(comps))


def _medir(indice, consultas, exacto, n: int = N,
           **kw) -> Tuple[float, float, float]:
    """recall@k, porcentaje de la coleccion evaluado y latencia (ms)."""
    res, lat, comp = _latencia(indice, consultas, exacto.shape[1], **kw)
    return recall(res, exacto), 100.0 * comp / n, lat


def simular_colision(dim: int = DIM, planos: int = 20_000) -> None:
    """probabilidad de que un hiperplano aleatorio no separe dos vectores
    que forman un angulo theta, frente a la formula 1 - theta/pi."""
    rng = np.random.default_rng(SEMILLA + 3)
    a = rng.standard_normal(dim)
    a /= np.linalg.norm(a)
    u = rng.standard_normal(dim)
    u -= (u @ a) * a
    u /= np.linalg.norm(u)                       # direccion ortogonal a a
    r = rng.standard_normal((planos, dim))
    filas = []
    print("\nLSH: P[mismo bit] segun el angulo (%d hiperplanos)" % planos)
    for grados in range(0, 181, 15):
        t = np.radians(grados)
        b = np.cos(t) * a + np.sin(t) * u        # angulo exacto con a
        emp = float(np.mean((r @ a > 0) == (r @ b > 0)))
        teo = 1.0 - grados / 180.0
        filas.append((grados, round(emp, 4), round(teo, 4)))
        print(f"  {grados:>3} grados  medida {emp:.4f}  1-theta/pi {teo:.4f}")
    _escribir(os.path.join("data", "cap15_colision.dat"),
              "LSH por hiperplanos: fraccion de %d hiperplanos que no separan "
              "dos vectores a angulo theta (dim=%d) frente a 1-theta/pi"
              % (planos, dim),
              "grados  medida  teorica", filas)


def geometria(base, consultas, exacto, ivf: IndiceIVF) -> None:
    """estadisticos de la coleccion que el texto usa para explicar las
    medidas: cosenos dentro y entre temas, candidatos esperados de LSH segun
    1 - (1 - p^b)^L con el angulo de cada vector, y reparto de las listas de
    IVF. no escribe ningun .dat."""
    _, tema = _coleccion(len(base) + len(consultas), base.shape[1],
                         con_tema=True)
    tb, tq = tema[:len(base)], tema[len(base):]
    s = consultas @ base.T
    mismo = tq[:, None] == tb[None, :]
    por_tema = np.bincount(tb)
    print("\ngeometria: coseno medio dentro del tema %.3f; entre temas %.3f "
          "(desviacion %.3f); vectores por tema %.0f (de %d a %d)"
          % (s[mismo].mean(), s[~mismo].mean(), s[~mismo].std(),
             por_tema.mean(), por_tema.min(), por_tema.max()))
    p = 1 - np.arccos(np.clip(s, -1, 1)) / np.pi    # coincidencia por bit
    for bits in (12, 6):
        prob = 1 - (1 - p ** bits) ** 16                 # 16 tablas
        rec = np.mean([prob[i, exacto[i]].mean() for i in range(len(s))])
        print(f"  LSH {bits} bits, 16 tablas: recall esperado {rec:.3f}; "
              f"candidatos esperados {prob.sum(1).mean():.0f} (del tema "
              f"{(prob * mismo).sum(1).mean():.0f}, de otros "
              f"{(prob * ~mismo).sum(1).mean():.0f})")
    tam = np.array([len(lista) for lista in ivf.listas])
    explor = []
    for q in consultas:
        celdas = _top(((ivf.centroides - q) ** 2).sum(1), 8, mayor=False)
        explor.append(tam[celdas].mean())
    asign = np.empty(len(base), dtype=np.int64)
    for j, lista in enumerate(ivf.listas):
        asign[lista] = j
    ocupa = [len(np.unique(asign[e])) for e in exacto]
    print(f"  IVF {len(tam)} celdas: lista media {tam.mean():.0f} (de "
          f"{tam.min()} a {tam.max()}); explorada con nprobe 8 "
          f"{np.mean(explor):.0f}; ponderada por tamano "
          f"{(tam ** 2).mean() / tam.mean():.0f}; los 10 vecinos ocupan "
          f"{np.mean(ocupa):.1f} celdas (como mucho {max(ocupa)})")


def simular_lsh(base, consultas, exacto) -> List[Tuple]:
    filas = []
    print("\nLSH (16 tablas): recall y evaluaciones segun los bits")
    print("  bits  recall   %_colec  latencia_ms")
    for bits in (2, 3, 4, 5, 6, 8, 10, 12, 14):
        lsh = IndiceLSH(base, bits=bits, tablas=16)
        rec, pct, lat = _medir(lsh, consultas, exacto)
        filas.append((bits, round(rec, 4), round(pct, 2), round(lat, 3)))
        print(f"  {bits:<4}  {rec:.4f}   {pct:6.2f}   {lat:.3f}")
    _escribir(os.path.join("data", "cap15_lsh.dat"),
              "LSH: recall@10, %% de la coleccion evaluado (candidatos y "
              "productos de la firma) y latencia (ms) segun los bits "
              "(n=%d, 16 tablas, consultas reservadas)" % len(base),
              "bits  recall  pct  latencia_ms", filas)
    return filas


def simular_ivf(base, consultas, exacto, ivf: IndiceIVF) -> List[Tuple]:
    filas = []
    print("\nIVF (nlist=256): recall y evaluaciones segun nprobe")
    print("  nprobe  recall   %_colec  latencia_ms")
    for nprobe in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        rec, pct, lat = _medir(ivf, consultas, exacto, nprobe=nprobe)
        filas.append((nprobe, round(rec, 4), round(pct, 2), round(lat, 3)))
        print(f"  {nprobe:<6}  {rec:.4f}   {pct:6.2f}   {lat:.3f}")
    tam = [len(l) for l in ivf.listas]
    print(f"  tamano de las listas: media {np.mean(tam):.1f}, "
          f"min {min(tam)}, max {max(tam)}")
    _escribir(os.path.join("data", "cap15_ivf.dat"),
              "IVF: recall@10, %% de la coleccion evaluado (centroides y "
              "listas) y latencia (ms) segun nprobe (n=%d, nlist=256)"
              % len(base),
              "nprobe  recall  pct  latencia_ms", filas)
    return filas


def simular_pq(base, consultas, exacto) -> None:
    filas = []
    print("\nPQ: recall sin reordenar y compresion de los codigos segun m")
    print("  m    bytes  compresion  recall")
    for m in (4, 8, 16, 32):
        pq = IndicePQ(base, m=m)
        rec, _, lat = _medir(pq, consultas, exacto)
        comp = 1.0 / pq.memoria_relativa()
        filas.append((m, m, round(comp, 1), round(rec, 4), round(lat, 3)))
        print(f"  {m:<4}  {m:<5}  {comp:<10.1f}  {rec:.4f}")
    _escribir(os.path.join("data", "cap15_pq.dat"),
              "PQ: bytes del codigo por vector, compresion frente a float32 "
              "(solo codigos), recall@10 sin reordenar y latencia (ms) segun m "
              "(n=%d, dim=%d)" % (len(base), base.shape[1]),
              "m  bytes  compresion  recall  latencia_ms", filas)


def simular_hnsw(base, consultas, exacto, hnsw: IndiceHNSW,
                 sin: IndiceHNSW) -> List[Tuple]:
    filas = []
    print("\nHNSW (M=32): recall y evaluaciones segun ef, con y sin "
          "la heuristica de diversidad")
    print("  ef    recall   %_colec  latencia_ms  recall_sin  %_sin")
    for ef in (10, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        rec, pct, lat = _medir(hnsw, consultas, exacto, ef=ef)
        rs, ps, _ = _medir(sin, consultas, exacto, ef=ef)
        filas.append((ef, round(rec, 4), round(pct, 2), round(lat, 3),
                      round(rs, 4), round(ps, 2)))
        print(f"  {ef:<4}  {rec:.4f}   {pct:6.2f}   {lat:8.3f}     "
              f"{rs:.4f}     {ps:6.2f}")
    g, gs = hnsw.grado_medio(0), sin.grado_medio(0)
    print(f"  grado medio de la capa 0: {g:.1f} con diversidad "
          f"({4 * g:.0f} bytes de identificadores de 32 bits), {gs:.1f} sin")
    print(f"  capas: {hnsw.maxnivel + 1}")
    _escribir(os.path.join("data", "cap15_hnsw.dat"),
              "HNSW (M=32, efc=200): recall@10, %% de la coleccion evaluado y "
              "latencia (ms) segun ef, con la heuristica de diversidad y sin "
              "ella (n=%d); grado medio de la capa 0: %.1f con, %.1f sin"
              % (len(base), g, gs),
              "ef  recall  pct  latencia_ms  recall_sin  pct_sin", filas)
    return filas


def simular_rerank(base, consultas, exacto) -> None:
    """reordenacion sobre PQ: los R primeros candidatos por la distancia PQ
    se reordenan con la distancia exacta, que exige conservar (o leer) los
    vectores originales."""
    pq = IndicePQ(base, m=16)
    k = exacto.shape[1]
    filas = []
    print("\nreordenacion sobre PQ (m=16): recall@10 segun la profundidad R")
    print("  R     recall")
    for R in (10, 20, 50, 100, 200, 500, 1000):
        res = []
        for q in consultas:
            cand, _ = pq.buscar(q, R)            # R candidatos por PQ
            cand = np.array(cand)
            res.append(cand[_top(base[cand] @ q, k)].tolist())
        rec = recall(res, exacto)
        filas.append((R, round(rec, 4)))
        print(f"  {R:<4}  {rec:.4f}")
    _escribir(os.path.join("data", "cap15_rerank.dat"),
              "recall@10 de PQ (m=16) tras reordenar los R primeros candidatos "
              "con la distancia exacta (n=%d)" % len(base),
              "R  recall", filas)


def simular_nlist(base, consultas, exacto, nprobe: int = 8) -> None:
    """nlist a nprobe fijo: mas celdas hacen listas mas cortas, pero el
    barrido lineal de los centroides crece con nlist."""
    n = len(base)
    filas = []
    print("\nIVF: recall y evaluaciones segun nlist (nprobe=%d fijo)" % nprobe)
    print("  nlist  recall   %_total  %_centroides  %_listas")
    for nlist in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        ivf = IndiceIVF(base, nlist=nlist)
        rec, pct, _ = _medir(ivf, consultas, exacto, nprobe=nprobe)
        pc = 100.0 * nlist / n
        filas.append((nlist, round(rec, 4), round(pct, 2), round(pc, 2),
                      round(pct - pc, 2)))
        print(f"  {nlist:<5}  {rec:.4f}   {pct:6.2f}   {pc:6.2f}       "
              f"{pct - pc:6.2f}")
    _escribir(os.path.join("data", "cap15_nlist.dat"),
              "IVF: recall@10 y %% de la coleccion evaluado (total, centroides "
              "y listas) segun nlist, a nprobe=%d fijo (n=%d)" % (nprobe, n),
              "nlist  recall  pct  pct_centroides  pct_listas", filas)


def simular_escala(base, consultas, hnsw: IndiceHNSW,
                   ef: int = 64) -> None:
    """evaluaciones de HNSW segun el tamano de la coleccion, a ef fijo: un
    crecimiento empirico en esta coleccion, no una garantia."""
    k = K
    filas = []
    print("\nHNSW (M=32, ef=%d): evaluaciones segun el tamano" % ef)
    print("  n       evaluaciones  %_colec  recall")
    for n in (2_500, 5_000, 10_000, 20_000):
        sub = base[:n]
        h = hnsw if n == len(base) else IndiceHNSW(sub, M=32, efc=200)
        res, _, comp = _latencia(h, consultas, k, ef=ef)
        rec = recall(res, verdad(sub, consultas, k))
        filas.append((n, round(comp, 1), round(100.0 * comp / n, 2),
                      round(rec, 4)))
        print(f"  {n:<6}  {comp:12.1f}  {100.0 * comp / n:6.2f}   {rec:.4f}")
    _escribir(os.path.join("data", "cap15_escala.dat"),
              "HNSW (M=32, efc=200, ef=%d): evaluaciones medias por consulta, "
              "%% de la coleccion y recall@10 segun n" % ef,
              "n  evaluaciones  pct  recall", filas)


def comparativa(base, consultas, exacto, lsh: List[Tuple],
                ivf: List[Tuple], hnsw: List[Tuple], ivf_ix: IndiceIVF,
                hnsw_ix: IndiceHNSW, reps: int = 5) -> None:
    """para cada objetivo de recall, la configuracion mas barata de cada
    familia que lo alcanza en su barrido (o la de mas recall, si ninguna). la
    latencia de la maqueta se repite `reps` veces y se toma la mediana."""
    k = exacto.shape[1]

    def lat(indice, **kw) -> float:
        return float(np.median([_latencia(indice, consultas, k, **kw)[1]
                                for _ in range(reps)]))

    filas = [("Flat", "-", "-", 1.0, 100.0,
              round(lat(IndiceFlat(base)), 3))]
    familias = (("LSH", "bits", lsh), ("IVF", "nprobe", ivf),
                ("HNSW", "ef", hnsw))
    medidas: Dict[Tuple[str, int], float] = {}   # una medida por configuracion
    print("\ncomparativa a igual objetivo de recall@10")
    for obj in OBJETIVOS:
        for nombre, param, barrido in familias:
            ok = [f for f in barrido if f[1] >= obj]
            f = min(ok, key=lambda x: x[2]) if ok else \
                max(barrido, key=lambda x: x[1])
            clave = (nombre, f[0])
            if clave not in medidas:
                if nombre == "LSH":
                    medidas[clave] = lat(IndiceLSH(base, bits=f[0], tablas=16))
                elif nombre == "IVF":
                    medidas[clave] = lat(ivf_ix, nprobe=f[0])
                else:
                    medidas[clave] = lat(hnsw_ix, ef=f[0])
            filas.append((nombre, obj, f"{param}={f[0]}", f[1], f[2],
                          round(medidas[clave], 3)))
    for f in filas:
        print(" ", f)
    _escribir(os.path.join("data", "cap15_comparativa.dat"),
              "comparativa: configuracion mas barata de cada familia que "
              "alcanza el objetivo de recall@10 en su barrido; %% evaluado y "
              "latencia mediana de %d pasadas (ms) de esta maqueta (n=%d)"
              % (reps, len(base)),
              "metodo  objetivo  parametro  recall  pct  latencia_ms", filas)

def simular_isotropa(nq: int = NQ) -> None:
    """los mismos indices sobre ruido isotropico (sin temas): pierden
    eficiencia, porque los vecinos tienen poco contraste con el resto, pero
    siguen devolviendo vecinos; se mide cuanto trabajo exige cada recall."""
    rng = np.random.default_rng(SEMILLA + 4)
    todo = rng.standard_normal((N + nq, DIM))
    todo = (todo / np.linalg.norm(todo, axis=1, keepdims=True)).astype(
        np.float32)
    base, consultas = todo[:N], todo[N:]
    exacto = verdad(base, consultas, K)
    s = consultas @ base.T
    print("\nruido isotropico: similitud media del vecino 1 y del 10: "
          f"{np.sort(s, 1)[:, -1].mean():.3f} y {np.sort(s, 1)[:, -10].mean():.3f}")
    filas = []
    ivf = IndiceIVF(base, nlist=256)
    for nprobe in (1, 4, 8, 16, 32, 64, 128):
        rec, pct, _ = _medir(ivf, consultas, exacto, nprobe=nprobe)
        filas.append(("IVF", nprobe, round(rec, 4), round(pct, 2)))
    for bits in (14, 12, 10, 8, 6, 4):
        rec, pct, _ = _medir(IndiceLSH(base, bits=bits, tablas=16),
                             consultas, exacto)
        filas.append(("LSH", bits, round(rec, 4), round(pct, 2)))
    hnsw = IndiceHNSW(base, M=32, efc=200)
    for ef in (16, 32, 64, 128, 256, 512):
        rec, pct, _ = _medir(hnsw, consultas, exacto, ef=ef)
        filas.append(("HNSW", ef, round(rec, 4), round(pct, 2)))
    for f in filas:
        print("  %-5s %-4s recall %.4f  %%_colec %6.2f" % f)
    # la formula de colision con los angulos medidos predice el LSH de 4 bits
    prob = 1 - (1 - (1 - np.arccos(np.clip(s, -1, 1)) / np.pi) ** 4) ** 16
    rec = np.mean([prob[i, exacto[i]].mean() for i in range(nq)])
    print(f"  LSH 4 bits predicho: recall {rec:.3f} evaluando el "
          f"{100 * (prob.sum(1).mean() + 4 * 16) / N:.1f} %")
    _escribir(os.path.join("data", "cap15_isotropa.dat"),
              "ruido isotropico (n=%d, dim=%d): recall@10 y %% de la coleccion "
              "evaluado de IVF (nlist=256, segun nprobe), LSH (16 tablas, "
              "segun bits) y HNSW (M=32, segun ef)" % (N, DIM),
              "metodo  parametro  recall  pct", filas)


def demostracion(base, consultas, hnsw: IndiceHNSW, k: int = 15) -> None:
    """15 vecinos exactos frente a los de HNSW para una consulta reservada."""
    q = consultas[0]
    exacto = verdad(base, q[None, :], k)[0].tolist()
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
    base, consultas = datos()
    exacto = verdad(base, consultas, K)
    simular_colision()
    fil_lsh = simular_lsh(base, consultas, exacto)
    ivf = IndiceIVF(base, nlist=256)
    fil_ivf = simular_ivf(base, consultas, exacto, ivf)
    geometria(base, consultas, exacto, ivf)
    simular_pq(base, consultas, exacto)
    t0 = time.perf_counter()
    hnsw = IndiceHNSW(base, M=32, efc=200)
    niv = np.array(hnsw.niveles)
    print(f"\nHNSW construido en {time.perf_counter() - t0:.0f} s; nodos "
          f"por capa: {[int((niv >= c).sum()) for c in range(niv.max() + 1)]}")
    sin = IndiceHNSW(base, M=32, efc=200, diversidad=False)
    fil_hnsw = simular_hnsw(base, consultas, exacto, hnsw, sin)
    simular_rerank(base, consultas, exacto)
    simular_nlist(base, consultas, exacto)
    simular_escala(base, consultas, hnsw)
    comparativa(base, consultas, exacto, fil_lsh, fil_ivf, fil_hnsw, ivf,
                hnsw)
    demostracion(base, consultas, hnsw)
    simular_isotropa()


if __name__ == "__main__":
    main()
