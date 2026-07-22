"""capitulo 8 — las familias NoSQL.

cada familia NoSQL optimiza un patron de acceso distinto, y este modulo lo
mide implementando versiones de juguete en memoria ---Python puro--- y
cronometrando la operacion que cada una sabe hacer bien frente a la que le
resulta cara:

  1. clave-valor: acceso por clave. una tabla hash (dict) resuelve la
     busqueda en tiempo constante; recorrer una lista crece con N. mide la
     latencia de buscar una clave segun el tamano.
  2. columnar: agregacion de una columna. el almacen por columnas lee solo
     la columna pedida; el de filas ha de recorrer la fila entera. mide la
     latencia de sumar una columna segun el numero de columnas M.
  3. grafo: recorrido de relaciones (amigos de amigos a profundidad k). la
     lista de adyacencia salta a los vecinos; emular la reunion sin indice
     obliga a recorrer todas las aristas en cada salto. mide la latencia
     segun la profundidad.

es Python puro (sin servicio ni torch, segun la tabla de recursos): basta el
interprete, con semilla fija para reproducir los numeros. los motores reales
---Redis, MongoDB, Cassandra, Neo4j--- se levantan con infra/, pero el
modelo de datos y su coste se ven ya en estas maquetas. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import bisect
import os
import random
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

SEMILLA = 8


def anunciar() -> None:
    print("=" * 64)
    print("cap. 8 — las familias NoSQL (maquetas en memoria)")
    print("recursos: python puro · cpu. no usa servicio, gpu ni torch.")
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


def medir(fn: Callable[[], None], reps: int) -> float:
    """mediana en milisegundos de repetir fn, con un calentamiento."""
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    t.sort()
    return t[len(t) // 2]


def simular_clavevalor() -> None:
    """acceso por clave: tabla hash frente a recorrido lineal."""
    filas = []
    print("\nclave-valor: latencia de buscar una clave (ms)")
    print("  N        hash      barrido")
    for n in (100, 1000, 10000, 100000):
        pares = [(f"k{i}", i) for i in range(n)]
        tabla: Dict[str, int] = dict(pares)
        objetivo = f"k{n - 1}"  # peor caso del barrido: la ultima

        def por_hash() -> None:
            _ = tabla[objetivo]

        def por_barrido() -> None:
            for k, v in pares:
                if k == objetivo:
                    break

        kv = medir(por_hash, 2000)
        sc = medir(por_barrido, 300)
        filas.append((n, round(kv, 6), round(sc, 4)))
        print(f"  {n:<7}  {kv:.6f}  {sc:.4f}")
    _escribir(os.path.join("data", "cap08_clavevalor.dat"),
              "latencia de buscar una clave: hash vs barrido (ms)",
              "n  hash  barrido", filas)


def simular_columnar() -> None:
    """agregacion de una columna: por columnas frente a por filas.

    el almacen por filas guarda cada registro junto, asi que sumar una
    columna obliga a recorrer las M columnas de cada fila; el almacen por
    columnas guarda cada columna aparte y solo lee la pedida.
    """
    rng = random.Random(SEMILLA)
    n = 100000
    filas = []
    print(f"\ncolumnar: latencia de sumar una columna ({n} filas, ms)")
    print("  M(columnas)  por_filas  por_columnas")
    for m in (2, 4, 8, 16, 32):
        por_filas = [[rng.random() for _ in range(m)] for _ in range(n)]
        por_columnas = [[fila[c] for fila in por_filas] for c in range(m)]

        def agg_filas() -> None:
            s = 0.0
            for fila in por_filas:      # lee la fila entera (M columnas)
                for v in fila:
                    s += v
            _ = s

        def agg_columnas() -> None:
            _ = sum(por_columnas[0])    # solo la columna pedida

        ff = medir(agg_filas, 15)
        cc = medir(agg_columnas, 15)
        filas.append((m, round(ff, 3), round(cc, 3)))
        print(f"  {m:<11}  {ff:8.3f}  {cc:8.3f}")
    _escribir(os.path.join("data", "cap08_columnar.dat"),
              "latencia de sumar una columna: por filas vs por columnas "
              "(%d filas, ms)" % n,
              "m  por_filas  por_columnas", filas)


def _grafo_aleatorio(n: int, grado: int,
                     rng: random.Random) -> Tuple[Dict, List]:
    """devuelve la lista de adyacencia y la lista de aristas de un grafo
    dirigido aleatorio con 'grado' aristas salientes por nodo."""
    ady: Dict[int, List[int]] = {i: [] for i in range(n)}
    aristas: List[Tuple[int, int]] = []
    for a in range(n):
        for _ in range(grado):
            b = rng.randrange(n)
            ady[a].append(b)
            aristas.append((a, b))
    return ady, aristas


def simular_grafo() -> None:
    """amigos de amigos a profundidad k: adyacencia frente a reunion.

    el grafo salta a los vecinos por la lista de adyacencia; emular una
    reunion sin indice obliga a recorrer TODAS las aristas en cada salto.
    """
    rng = random.Random(SEMILLA)
    n, grado = 3000, 6
    ady, aristas = _grafo_aleatorio(n, grado, rng)
    filas = []
    print("\ngrafo: latencia de 'amigos de amigos' a profundidad k (ms)")
    print("  k   adyacencia  reunion")
    for k in (1, 2, 3, 4):

        def por_adyacencia() -> None:
            frontera = {0}
            vistos = {0}
            for _ in range(k):
                nueva = set()
                for u in frontera:
                    for v in ady[u]:
                        if v not in vistos:
                            vistos.add(v)
                            nueva.add(v)
                frontera = nueva

        def por_reunion() -> None:
            frontera = {0}
            for _ in range(k):           # cada salto recorre las aristas
                frontera = {b for (a, b) in aristas if a in frontera}

        ga = medir(por_adyacencia, 40)
        re = medir(por_reunion, 40)
        filas.append((k, round(ga, 4), round(re, 4)))
        print(f"  {k}   {ga:9.4f}  {re:8.4f}")
    _escribir(os.path.join("data", "cap08_grafo.dat"),
              "latencia de amigos de amigos: adyacencia vs reunion "
              "(%d nodos, grado %d, ms)" % (n, grado),
              "k  adyacencia  reunion", filas)


def simular_escritura() -> None:
    """coste de escribir: anadir al final (LSM) frente a insertar ordenado.

    la columnar ancha y otros almacenes escriben rapido porque solo *anaden*
    al final de un registro (coste constante, como el log de transacciones del
    cap. 5); mantener una estructura ordenada en el sitio (un arbol B) obliga a
    desplazar datos, y el coste crece con el tamano ya almacenado.
    """
    rng = random.Random(SEMILLA)
    filas = []
    print("\nescritura: latencia de un alta (ms)")
    print("  N        anadir(LSM)  insertar_ordenado")
    for n in (1000, 10000, 100000, 1000000):
        log: List[float] = []
        ordenado = list(range(0, 2 * n, 2))  # lista ordenada de tamano n

        def anadir() -> None:
            log.append(rng.random())        # al final: O(1)

        def insertar() -> None:
            x = rng.randrange(2 * n)
            i = bisect.bisect_left(ordenado, x)
            ordenado.insert(i, x)           # desplaza la cola: O(n)
            ordenado.pop(i)                 # deshace para mantener el tamano

        la = medir(anadir, 2000)
        li = medir(insertar, 300)
        filas.append((n, round(la, 6), round(li, 5)))
        print(f"  {n:<7}  {la:.6f}     {li:.5f}")
    _escribir(os.path.join("data", "cap08_escritura.dat"),
              "latencia de un alta: anadir al final vs insertar ordenado "
              "(ms)",
              "n  anadir  insertar", filas)


def simular_cache() -> None:
    """ratio de aciertos de una cache LRU segun su tamano, bajo acceso
    sesgado (Zipf): unas pocas claves concentran casi todas las peticiones,
    asi que una cache pequena captura la mayor parte del trafico. es la razon
    por la que el patron cache-aside del clave-valor funciona.
    """
    rng = random.Random(SEMILLA)
    claves = 1000
    pesos = [1.0 / (i + 1) for i in range(claves)]  # ley de Zipf
    accesos = rng.choices(range(claves), weights=pesos, k=200000)
    filas = []
    print("\ncache: ratio de aciertos LRU segun el tamano (acceso Zipf)")
    print("  tam_cache(%)  aciertos")
    for frac in (1, 2, 5, 10, 20, 50):
        cap = max(1, claves * frac // 100)
        cache: OrderedDict = OrderedDict()
        aciertos = 0
        for k in accesos:
            if k in cache:
                aciertos += 1
                cache.move_to_end(k)        # uso reciente
            else:
                cache[k] = 1
                if len(cache) > cap:
                    cache.popitem(last=False)  # expulsa el menos reciente
        ratio = aciertos / len(accesos)
        filas.append((frac, round(ratio, 4)))
        print(f"  {frac:<11}  {ratio:.4f}")
    _escribir(os.path.join("data", "cap08_cache.dat"),
              "ratio de aciertos LRU segun el tamano de cache (% de claves), "
              "acceso Zipf",
              "frac  aciertos", filas)


def demostracion(k: int = 15) -> None:
    """15 claves al azar de un almacen clave-valor, con su valor."""
    rng = random.Random(SEMILLA + 1)
    tabla = {f"sesion:{1000 + i}": f"user-{rng.randrange(100)}"
             for i in range(500)}
    claves = rng.sample(list(tabla), k)
    print(f"\ndemostracion: {k} claves al azar con su valor")
    print("  clave           valor")
    print("  --------------  -------")
    for c in claves:
        print(f"  {c:14}  {tabla[c]}")


def main() -> None:
    anunciar()
    simular_clavevalor()
    simular_columnar()
    simular_grafo()
    simular_escritura()
    simular_cache()
    demostracion(15)


if __name__ == "__main__":
    main()
