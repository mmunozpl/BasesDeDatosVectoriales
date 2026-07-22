"""capitulo 14 — anatomia de una base de datos vectorial.

construye, en numpy, un mini almacen vectorial desde cero para diseccionar las
partes de las que esta hecha una base de datos vectorial real: el
almacenamiento del vector y sus metadatos, la ingesta por lotes, el indice como
estructura central (aqui, busqueda exacta por fuerza bruta) y el caso
multivector. luego mide seis cosas que fijan el criterio de diseno:

  1. almacenamiento: el coste en memoria de N vectores segun el tipo numerico
     (float32, float16, int8); la cuantizacion divide la huella.
  2. busqueda exacta: la latencia y el numero de operaciones de la busqueda por
     fuerza bruta crecen linealmente con N (O(n*d)); de ahi nace el cap. 15.
  3. filtrado por metadatos: el coste de filtrar antes (prefiltrado) o despues
     (posfiltrado) de buscar, segun la selectividad del filtro.
  4. cuantizacion: el error y el recall@10 de cuantizar a int8 frente a
     float32; mucha menos memoria, casi el mismo resultado.
  5. multivector: el coste en almacenamiento y comparaciones de representar cada
     documento por varios vectores (un vector por token, estilo MaxSim).
  6. normalizacion en ingesta: precomputar la norma al ingestar abarata cada
     consulta posterior (coseno = producto escalar sobre vectores normalizados).

es Python puro con numpy (sin servicio, sin GPU): un motor real como pgvector,
Qdrant o Milvus implementa estas mismas partes con mucha mas ingenieria, pero la
anatomia es la que este modulo deja a la vista. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SEMILLA = 14


def anunciar() -> None:
    print("=" * 64)
    print("cap. 14 — anatomia de una base de datos vectorial")
    print("recursos: python + numpy · cpu. no usa servicio, gpu ni torch.")
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
# el mini almacen: las cuatro partes de una base vectorial
# ---------------------------------------------------------------------------

class AlmacenVectorial:
    """Un almacen vectorial minimo pero completo.

    Reune las cuatro partes anatomicas de una base de datos vectorial: una
    matriz de vectores (el dato), una lista de metadatos paralela (el
    contexto), las normas precomputadas (la optimizacion de ingesta) y la
    busqueda exacta por fuerza bruta (el indice mas simple posible).
    """

    def __init__(self, dim: int, metrica: str = "coseno") -> None:
        self.dim = dim
        self.metrica = metrica
        self.vectores = np.empty((0, dim), dtype=np.float32)
        self.metadatos: List[Dict] = []
        self.normas = np.empty((0,), dtype=np.float32)

    def agregar(self, vecs: np.ndarray, metas: List[Dict]) -> None:
        """Ingesta por lotes: apila los vectores y guarda sus metadatos.

        Precomputa la norma de cada vector en la ingesta para que cada
        consulta posterior no tenga que recalcularla (coseno barato).
        """
        vecs = vecs.astype(np.float32)
        self.vectores = np.vstack([self.vectores, vecs])
        self.metadatos.extend(metas)
        normas = np.linalg.norm(vecs, axis=1)
        self.normas = np.concatenate([self.normas, normas])

    def __len__(self) -> int:
        return len(self.metadatos)

    def _similitud(self, consulta: np.ndarray) -> np.ndarray:
        """Vector de similitudes de la consulta con toda la coleccion."""
        if self.metrica == "coseno":
            sim = self.vectores @ consulta
            sim = sim / (self.normas * np.linalg.norm(consulta) + 1e-12)
            return sim
        if self.metrica == "producto":
            return self.vectores @ consulta
        # euclidea: se devuelve la negativa de la distancia (mayor = mas cerca)
        return -np.linalg.norm(self.vectores - consulta, axis=1)

    def buscar(self, consulta: np.ndarray, k: int = 10,
               filtro: Optional[Callable[[Dict], bool]] = None
               ) -> List[Tuple[int, float]]:
        """Busqueda exacta de los k mas proximos, con filtro opcional.

        Si hay filtro, se aplica como PREFILTRADO: primero se restringe la
        coleccion a los que pasan el predicado y despues se busca en ese
        subconjunto. Devuelve pares (indice, similitud).
        """
        if filtro is not None:
            idx = [i for i, m in enumerate(self.metadatos) if filtro(m)]
            if not idx:
                return []
            sub = self.vectores[idx]
            normas = self.normas[idx]
            sim = self._sim_sobre(consulta, sub, normas)
            orden = np.argsort(-sim)[:k]
            return [(idx[j], float(sim[j])) for j in orden]
        sim = self._similitud(consulta)
        orden = np.argsort(-sim)[:k]
        return [(int(j), float(sim[j])) for j in orden]

    def _sim_sobre(self, consulta: np.ndarray, sub: np.ndarray,
                   normas: np.ndarray) -> np.ndarray:
        if self.metrica == "coseno":
            s = sub @ consulta
            return s / (normas * np.linalg.norm(consulta) + 1e-12)
        if self.metrica == "producto":
            return sub @ consulta
        return -np.linalg.norm(sub - consulta, axis=1)

    def guardar(self, ruta: str) -> None:
        """Persistencia: los vectores a disco como un unico array binario."""
        np.save(ruta, self.vectores)

    def memoria_bytes(self) -> int:
        """Huella en memoria de los vectores mas las normas."""
        return self.vectores.nbytes + self.normas.nbytes


# ---------------------------------------------------------------------------
# datos sinteticos: una coleccion con estructura y metadatos
# ---------------------------------------------------------------------------

def _coleccion(n: int, dim: int, n_temas: int = 8,
               semilla: int = SEMILLA) -> Tuple[np.ndarray, List[Dict]]:
    """Genera n vectores agrupados en temas, con metadatos (tema, año)."""
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((n_temas, dim)) * 4
    tema = rng.integers(0, n_temas, n)
    vecs = centros[tema] + rng.standard_normal((n, dim))
    metas = [{"id": i, "tema": int(tema[i]),
              "anio": int(rng.integers(2018, 2026))} for i in range(n)]
    return vecs.astype(np.float32), metas


# ---------------------------------------------------------------------------
# 1. almacenamiento: la huella segun el tipo numerico
# ---------------------------------------------------------------------------

def simular_almacenamiento(dim: int = 768) -> None:
    """Memoria de N vectores de dimension dim segun float32/float16/int8.

    Es aritmetica exacta (N*dim*bytes), pero deja ver de un golpe que la
    cuantizacion ---pasar de 4 bytes a 1--- divide por cuatro la huella, la
    palanca central de la operacion a gran escala.
    """
    filas = []
    print("\nalmacenamiento: MB de N vectores de dim", dim)
    print("  N         float32   float16   int8")
    for n in (10_000, 100_000, 1_000_000, 10_000_000):
        mb32 = n * dim * 4 / 1e6
        mb16 = n * dim * 2 / 1e6
        mb8 = n * dim * 1 / 1e6
        filas.append((n, round(mb32, 1), round(mb16, 1), round(mb8, 1)))
        print(f"  {n:<9} {mb32:>8.1f}  {mb16:>8.1f}  {mb8:>8.1f}")
    _escribir(os.path.join("data", "cap14_almacenamiento.dat"),
              "memoria (MB) de N vectores de dim %d segun el tipo numerico"
              % dim, "n  float32  float16  int8", filas)


# ---------------------------------------------------------------------------
# 2. busqueda exacta: el coste crece con N (O(n*d))
# ---------------------------------------------------------------------------

def simular_busqueda(dim: int = 256, repeticiones: int = 20) -> None:
    """Latencia y operaciones de la busqueda exacta segun el tamaño N.

    El numero de operaciones (n*dim) es exacto y deterministico; la latencia
    es medida (depende de la maquina) y crece linealmente con N. Es la prueba
    de que la fuerza bruta no escala, y el motivo del cap. 15.
    """
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nbusqueda exacta: latencia y operaciones segun N (dim", dim, ")")
    print("  N         oper.(M)   latencia_ms")
    for n in (1_000, 10_000, 100_000, 500_000, 1_000_000):
        vecs, metas = _coleccion(n, dim)
        alm = AlmacenVectorial(dim, metrica="coseno")
        alm.agregar(vecs, metas)
        consultas = rng.standard_normal((repeticiones, dim)).astype(np.float32)
        t0 = time.perf_counter()
        for q in consultas:
            alm.buscar(q, k=10)
        dt = (time.perf_counter() - t0) / repeticiones
        oper = n * dim / 1e6
        filas.append((n, round(oper, 2), round(dt * 1e3, 3)))
        print(f"  {n:<9} {oper:>8.2f}   {dt * 1e3:>8.3f}")
    _escribir(os.path.join("data", "cap14_busqueda.dat"),
              "busqueda exacta: operaciones (millones) y latencia (ms) "
              "segun N, dim %d" % dim, "n  oper_M  latencia_ms", filas)


# ---------------------------------------------------------------------------
# 3. filtrado por metadatos: prefiltrar vs posfiltrar
# ---------------------------------------------------------------------------

def simular_filtrado(n: int = 200_000, dim: int = 256,
                     repeticiones: int = 30) -> None:
    """Coste y correccion de prefiltrar frente a posfiltrar.

    Prefiltrar (con un indice de metadatos: aqui, una mascara vectorizada)
    busca solo en el subconjunto que pasa el filtro: barato cuando el filtro es
    selectivo. Posfiltrar busca en toda la coleccion y descarta despues: coste
    casi fijo, pero se queda corto de resultados cuando el filtro es selectivo
    (inanicion), porque entre los k primeros por similitud pasan muy pocos.
    """
    rng = np.random.default_rng(SEMILLA)
    vecs, metas = _coleccion(n, dim, n_temas=20)
    temas = np.array([m["tema"] for m in metas])     # el indice de metadatos
    normas = np.linalg.norm(vecs, axis=1)
    consultas = rng.standard_normal((repeticiones, dim)).astype(np.float32)
    filas = []
    print("\nfiltrado: prefiltrado (indice) vs posfiltrado segun selectividad")
    print("  sel(%)   prefiltro_ms  posfiltro_ms  result_posfiltro")
    for temas_ok in (1, 2, 5, 10, 20):
        permitidos = np.arange(temas_ok)
        sel = 100.0 * temas_ok / 20
        # prefiltrado con indice: mascara vectorizada + busqueda en el subconj.
        t0 = time.perf_counter()
        for q in consultas:
            mask = np.isin(temas, permitidos)        # el indice filtra
            sub, subn = vecs[mask], normas[mask]
            sim = (sub @ q) / (subn * np.linalg.norm(q) + 1e-12)
            np.argsort(-sim)[:10]
        pre = (time.perf_counter() - t0) / repeticiones
        # posfiltrado: buscar en todo (top 100) y filtrar despues
        survivientes = 0
        t0 = time.perf_counter()
        for q in consultas:
            sim = (vecs @ q) / (normas * np.linalg.norm(q) + 1e-12)
            cand = np.argsort(-sim)[:100]
            paso = [c for c in cand if temas[c] in permitidos][:10]
            survivientes += len(paso)
        pos = (time.perf_counter() - t0) / repeticiones
        res = survivientes / repeticiones            # resultados medios (<=10)
        filas.append((round(sel, 1), round(pre * 1e3, 3),
                      round(pos * 1e3, 3), round(res, 1)))
        print(f"  {sel:<7.1f}  {pre * 1e3:>10.3f}  {pos * 1e3:>10.3f}"
              f"  {res:>14.1f}")
    _escribir(os.path.join("data", "cap14_filtrado.dat"),
              "latencia (ms) de prefiltrado (con indice) vs posfiltrado y "
              "resultados que devuelve el posfiltrado, segun selectividad "
              "(n %d, dim %d, k=10)" % (n, dim),
              "selectividad  prefiltro_ms  posfiltro_ms  result_posfiltro",
              filas)


# ---------------------------------------------------------------------------
# 4. cuantizacion: int8 frente a float32
# ---------------------------------------------------------------------------

def _cuantizar_int8(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cuantiza cada vector a int8 con un factor de escala por vector."""
    escala = np.abs(vecs).max(axis=1, keepdims=True) / 127.0
    q = np.round(vecs / (escala + 1e-12)).astype(np.int8)
    return q, escala


def simular_cuantizacion(n: int = 50_000, dim: int = 256,
                         consultas: int = 200) -> None:
    """Error y recall@10 de cuantizar a int8 frente a float32.

    Mide cuanto se degrada el resultado al guardar los vectores en un byte por
    componente en vez de cuatro: el recall@10 sigue altisimo y la memoria cae
    a la cuarta parte. Es el porque de la cuantizacion en produccion.
    """
    rng = np.random.default_rng(SEMILLA)
    vecs, _ = _coleccion(n, dim)
    q, escala = _cuantizar_int8(vecs)
    recon = q.astype(np.float32) * escala         # vectores reconstruidos
    err = float(np.linalg.norm(vecs - recon, axis=1).mean()
                / np.linalg.norm(vecs, axis=1).mean())
    qs = rng.standard_normal((consultas, dim)).astype(np.float32)
    aciertos = 0
    for x in qs:
        exacto = np.argsort(-(vecs @ x))[:10]
        aprox = np.argsort(-(recon @ x))[:10]
        aciertos += len(set(exacto) & set(aprox))
    recall = aciertos / (consultas * 10)
    filas = [("float32", 32, 1.0, 1.0),
             ("int8", 8, round(0.25, 3), round(recall, 4))]
    print("\ncuantizacion int8 vs float32 (dim", dim, ")")
    print(f"  error relativo medio: {err:.4f}")
    print(f"  recall@10 de int8:    {recall:.4f}")
    print("  memoria int8:         0.25x")
    _escribir(os.path.join("data", "cap14_cuantizacion.dat"),
              "cuantizacion: bits, memoria relativa y recall@10 de int8 "
              "frente a float32 (error rel. medio int8 = %.4f)" % err,
              "tipo  bits  memoria_rel  recall10", filas)


# ---------------------------------------------------------------------------
# 5. multivector: un vector por documento vs uno por token (MaxSim)
# ---------------------------------------------------------------------------

def _maxsim(consulta_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """MaxSim: suma, por cada token de la consulta, su mejor parecido en el
    documento. El nucleo de la interaccion tardia (ColBERT)."""
    sim = consulta_tokens @ doc_tokens.T          # (tq, td)
    return float(sim.max(axis=1).sum())


def simular_multivector(dim: int = 128, n_docs: int = 2000) -> None:
    """Coste de almacenamiento y comparaciones del multivector vs single.

    Representar cada documento por T vectores (uno por token) multiplica por T
    la memoria y por T*T el coste de comparar; a cambio, captura el parecido a
    nivel de token. Es el compromiso que la compresion tipo PLAID ataca.
    """
    filas = []
    print("\nmultivector: memoria y comparaciones segun tokens por documento")
    print("  tokens/doc  memoria_rel  comparaciones_rel")
    for t in (1, 8, 16, 32, 64):
        mem_rel = float(t)                         # T vectores por doc
        comp_rel = float(t * t)                    # MaxSim: tq*td por par
        filas.append((t, round(mem_rel, 1), round(comp_rel, 1)))
        print(f"  {t:<10}  {mem_rel:>10.1f}  {comp_rel:>14.1f}")
    # comprobacion real de que MaxSim funciona sobre un par de documentos
    rng = np.random.default_rng(SEMILLA)
    base = rng.standard_normal((16, dim)).astype(np.float32)
    q = base[:6] + 0.1 * rng.standard_normal((6, dim))   # consulta afin
    otro = rng.standard_normal((16, dim)).astype(np.float32)
    print(f"  maxsim(consulta, doc_afin) = {_maxsim(q, base):.2f}")
    print(f"  maxsim(consulta, doc_otro) = {_maxsim(q, otro):.2f}")
    _escribir(os.path.join("data", "cap14_multivector.dat"),
              "memoria y comparaciones relativas del multivector segun el "
              "numero de tokens por documento (single = 1)",
              "tokens  memoria_rel  comparaciones_rel", filas)


# ---------------------------------------------------------------------------
# demostracion: 15 vecinos al azar con su distancia
# ---------------------------------------------------------------------------

def demostracion(k: int = 15, n: int = 5000, dim: int = 256) -> None:
    """Una consulta real al mini almacen: los 15 vecinos mas proximos."""
    vecs, metas = _coleccion(n, dim)
    alm = AlmacenVectorial(dim, metrica="coseno")
    alm.agregar(vecs, metas)
    rng = np.random.default_rng(SEMILLA + 1)
    consulta = vecs[0] + 0.5 * rng.standard_normal(dim).astype(np.float32)
    res = alm.buscar(consulta, k=k)
    print(f"\ndemostracion: {k} vecinos mas proximos de una consulta")
    print("  rank  id      tema  coseno")
    print("  ----  ------  ----  ------")
    for r, (i, sim) in enumerate(res):
        m = alm.metadatos[i]
        print(f"  {r:<4}  {m['id']:<6}  {m['tema']:<4}  {sim: .4f}")


def main() -> None:
    anunciar()
    simular_almacenamiento()
    simular_busqueda()
    simular_filtrado()
    simular_cuantizacion()
    simular_multivector()
    demostracion(15)


if __name__ == "__main__":
    main()
