"""capitulo 14: anatomia de una base de datos vectorial.

construye, en numpy, una maqueta funcional del nucleo de un almacen vectorial
para diseccionar sus partes: el almacenamiento del vector y sus metadatos, la
ingesta por lotes, la busqueda exacta por barrido (sin indice de poda, la
referencia contra la que se miden los indices) y el caso multivector. luego
mide cinco cosas y muestra una demostracion:

  1. almacenamiento: el coste en memoria de N vectores segun el tipo numerico
     (float32, float16, int8); la cuantizacion divide la huella.
  2. busqueda exacta: el calculo de las similitudes cuesta O(n*d) y la
     seleccion de los k mejores, O(n) con argpartition; se mide la latencia.
  3. filtrado por metadatos: prefiltrar (mascara O(n) y busqueda en el
     subconjunto) frente a posfiltrar (buscar 10*k candidatos y filtrar).
  4. cuantizacion escalar simetrica a int8 (255 niveles, una escala por
     vector): error de reconstruccion y recall@10 frente a float32.
  5. multivector: memoria y parejas relativas de un vector por token, bajo el
     modelo simplificado T_q = T_d = T y la misma dimension por token.

es Python puro con numpy (sin servicio, sin GPU). un motor real comparte este
nucleo conceptual, pero anade estructuras de almacenamiento e indice,
recuperacion ante fallos, concurrencia, durabilidad, filtrado indexado y
distribucion que pueden cambiar mucho la implementacion.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SEMILLA = 14


def anunciar() -> None:
    print("=" * 64)
    print("cap. 14: anatomia de una base de datos vectorial")
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

def _topk(puntuaciones: np.ndarray, k: int) -> np.ndarray:
    """indices de los k mayores, ordenados: seleccion parcial O(n) con
    argpartition y orden solo de esos k (O(k log k))."""
    k = min(k, len(puntuaciones))
    if k == 0:
        return np.empty(0, dtype=int)
    idx = np.argpartition(-puntuaciones, k - 1)[:k]
    return idx[np.argsort(-puntuaciones[idx])]


class AlmacenVectorial:
    """Maqueta funcional del nucleo de un almacen vectorial.

    Reune la matriz de vectores (el dato), los metadatos paralelos (el
    contexto), las normas precomputadas (la optimizacion de ingesta) y la
    busqueda exacta por barrido. No tiene indice de poda, ni recuperacion
    ante fallos, ni concurrencia, ni borrado: es el nucleo, no un motor.
    """

    def __init__(self, dim: int, metrica: str = "coseno") -> None:
        self.dim = dim
        self.metrica = metrica
        self._buf = np.empty((1024, dim), dtype=np.float32)   # capacidad
        self._normas = np.empty(1024, dtype=np.float32)
        self.n = 0
        self.metadatos: List[Dict] = []

    @property
    def vectores(self) -> np.ndarray:
        return self._buf[:self.n]

    @property
    def normas(self) -> np.ndarray:
        return self._normas[:self.n]

    def agregar(self, vecs: np.ndarray, metas: List[Dict]) -> None:
        """Ingesta por lotes con crecimiento amortizado.

        Valida dimension y metadatos, rechaza vectores nulos con el coseno y
        precomputa la norma. La capacidad se dobla cuando hace falta, de modo
        que cada vector se copia un numero acotado de veces (un vstack por
        lote recopiaria toda la matriz en cada insercion).
        """
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError("dimension incompatible")
        if len(metas) != len(vecs):
            raise ValueError("un registro de metadatos por vector")
        normas = np.linalg.norm(vecs, axis=1)
        if self.metrica == "coseno" and np.any(normas == 0):
            raise ValueError("el coseno no admite vectores nulos")
        m = len(vecs)
        if self.n + m > len(self._buf):
            self._crecer(self.n + m)                 # dobla la capacidad
        self._buf[self.n:self.n + m] = vecs
        self._normas[self.n:self.n + m] = normas
        self.n += m
        self.metadatos.extend(metas)

    def _crecer(self, minimo: int) -> None:
        """Dobla la capacidad del bufer (o la lleva a minimo si no basta)."""
        cap = max(2 * len(self._buf), minimo)
        buf = np.empty((cap, self.dim), dtype=np.float32)
        buf[:self.n] = self._buf[:self.n]
        nor = np.empty(cap, dtype=np.float32)
        nor[:self.n] = self._normas[:self.n]
        self._buf, self._normas = buf, nor

    def __len__(self) -> int:
        return self.n

    def _sim_sobre(self, consulta: np.ndarray, sub: np.ndarray,
                   normas: np.ndarray) -> np.ndarray:
        """Similitudes de la consulta con las filas de sub."""
        if self.metrica == "coseno":
            nq = np.linalg.norm(consulta)
            if nq == 0:
                raise ValueError("el coseno no admite una consulta nula")
            return (sub @ consulta) / (normas * nq)
        if self.metrica == "producto":
            return sub @ consulta
        # euclidea: la negativa de la distancia (mayor = mas cerca)
        return -np.linalg.norm(sub - consulta, axis=1)

    def buscar(self, consulta: np.ndarray, k: int = 10,
               filtro: Optional[Callable[[Dict], bool]] = None
               ) -> List[Tuple[int, float]]:
        """Busqueda exacta de los k mas proximos, con filtro opcional.

        Con filtro, se evalua el predicado sobre todos los metadatos (O(n), sin
        indice de metadatos) y se busca en el subconjunto. Devuelve pares
        (identificador original, similitud).
        """
        if filtro is not None:
            idx = np.array([i for i, m in enumerate(self.metadatos)
                            if filtro(m)], dtype=int)
            if len(idx) == 0:
                return []
            sim = self._sim_sobre(consulta, self.vectores[idx],
                                  self.normas[idx])
            loc = _topk(sim, k)
            return [(int(idx[j]), float(sim[j])) for j in loc]
        sim = self._sim_sobre(consulta, self.vectores, self.normas)
        return [(int(j), float(sim[j])) for j in _topk(sim, k)]

    def guardar(self, ruta: str) -> None:
        """Serializacion minima: vectores, normas, metadatos y configuracion
        en un .npz. No es persistencia de base de datos: sin atomicidad, sin
        registro de escritura anticipada ni recuperacion tras una escritura
        parcial."""
        np.savez(ruta, vectores=self.vectores, normas=self.normas,
                 metadatos=np.array(json.dumps(self.metadatos)),
                 dim=self.dim, metrica=self.metrica)

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
    cuantizacion (pasar de 4 bytes a 1) divide por cuatro la huella, la
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
        alm.buscar(consultas[0], k=10)               # calentamiento
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

    Prefiltrar (aqui, una mascara vectorizada O(n) sobre los metadatos, sin
    indice) busca solo en el subconjunto que pasa el filtro. Posfiltrar busca
    los 10*k mejores en toda la coleccion y descarta despues: coste casi fijo,
    pero se queda corto de resultados cuando el filtro es selectivo y, como en
    estos datos, el filtro esta correlacionado con la similitud (los temas
    son cumulos: los mejores candidatos caen casi todos en un mismo tema).
    """
    rng = np.random.default_rng(SEMILLA)
    vecs, metas = _coleccion(n, dim, n_temas=20)
    temas = np.array([m["tema"] for m in metas])     # columna de metadatos
    normas = np.linalg.norm(vecs, axis=1)
    consultas = rng.standard_normal((repeticiones, dim)).astype(np.float32)
    filas = []
    _ = (vecs @ consultas[0]) / normas               # calentamiento
    print("\nfiltrado: prefiltrado vs posfiltrado (10*k) segun selectividad")
    print("  sel(%)   prefiltro_ms  posfiltro_ms  result_posfiltro")
    for temas_ok in (1, 2, 5, 10, 20):
        permitidos = np.arange(temas_ok)
        sel = 100.0 * temas_ok / 20
        # prefiltrado: mascara O(n) sobre los metadatos y busqueda en el
        # subconjunto; los indices locales se traducen a los originales
        t0 = time.perf_counter()
        for q in consultas:
            mask = np.isin(temas, permitidos)        # barrido de metadatos
            originales = np.flatnonzero(mask)
            sim = (vecs[mask] @ q) / (normas[mask] * np.linalg.norm(q))
            originales[_topk(sim, 10)]
        pre = (time.perf_counter() - t0) / repeticiones
        # posfiltrado: buscar en todo los 10*k = 100 mejores y filtrar despues
        survivientes = 0
        t0 = time.perf_counter()
        for q in consultas:
            sim = (vecs @ q) / (normas * np.linalg.norm(q))
            cand = _topk(sim, 100)
            paso = [c for c in cand if temas[c] in permitidos][:10]
            survivientes += len(paso)
        pos = (time.perf_counter() - t0) / repeticiones
        res = survivientes / repeticiones            # resultados medios (<=10)
        filas.append((round(sel, 1), round(pre * 1e3, 3),
                      round(pos * 1e3, 3), round(res, 1)))
        print(f"  {sel:<7.1f}  {pre * 1e3:>10.3f}  {pos * 1e3:>10.3f}"
              f"  {res:>14.1f}")
    _escribir(os.path.join("data", "cap14_filtrado.dat"),
              "latencia (ms) de prefiltrado (mascara) vs posfiltrado (10*k "
              "candidatos) y resultados que devuelve el posfiltrado, segun "
              "selectividad (n %d, dim %d, k=10)" % (n, dim),
              "selectividad  prefiltro_ms  posfiltro_ms  result_posfiltro",
              filas)


# ---------------------------------------------------------------------------
# 4. cuantizacion: int8 frente a float32
# ---------------------------------------------------------------------------

def _cuantizar_int8(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cuantizacion escalar simetrica a int8 con un factor por vector: los
    codigos van de -127 a 127 (255 niveles; el -128 no se usa). Un vector
    nulo recibe escala 1 y codigos cero, de forma explicita."""
    maximo = np.abs(vecs).max(axis=1, keepdims=True)
    escala = np.where(maximo > 0, maximo / 127.0, 1.0).astype(np.float32)
    q = np.round(vecs / escala).astype(np.int8)
    return q, escala


def simular_cuantizacion(n: int = 50_000, dim: int = 256,
                         consultas: int = 200) -> None:
    """Error y recall@10 de cuantizar a int8 frente a float32.

    El error es el cociente entre la norma media del error de reconstruccion,
    ||x - x_rec||, y la norma media de los vectores. La busqueda puntua con
    los codigos y la escala, q8 @ x * escala, sin materializar la coleccion
    reconstruida (numpy convierte el bloque de codigos al operar; un motor real
    usa nucleos que operan sobre los enteros). El recall@10 compara con el
    top-10 exacto por producto interno.
    """
    rng = np.random.default_rng(SEMILLA)
    vecs, _ = _coleccion(n, dim)
    q, escala = _cuantizar_int8(vecs)
    recon = q.astype(np.float32) * escala         # solo para medir el error
    err = float(np.linalg.norm(vecs - recon, axis=1).mean()
                / np.linalg.norm(vecs, axis=1).mean())
    del recon
    qs = rng.standard_normal((consultas, dim)).astype(np.float32)
    aciertos = 0
    margen, ruido, perdidos = [], [], []
    for x in qs:
        punt = vecs @ x
        exacto = _topk(punt, 10)
        aprox = _topk((q @ x) * escala[:, 0], 10)  # en el dominio cuantizado
        aciertos += len(set(exacto) & set(aprox))
        # margen entre el decimo y el undecimo exactos frente al ruido de
        # puntuacion que introduce la cuantizacion
        orden = np.sort(punt)[::-1]
        margen.append(orden[9] - orden[10])
        ruido.append(float(np.std((q @ x) * escala[:, 0] - punt)))
        perdidos += [r + 1 for r, i in enumerate(exacto) if i not in set(aprox)]
    recall = aciertos / (consultas * 10)
    # modelo del ruido uniforme: error relativo ~ (max/rms) / (127 raiz(12))
    rms = np.sqrt((vecs ** 2).mean(axis=1))
    pico = float((np.abs(vecs).max(axis=1) / rms).mean())
    mem_int8 = (q.nbytes + escala.nbytes) / vecs.nbytes   # con las escalas
    filas = [("float32", 32, 1.0, 1.0),
             ("int8", 8, round(mem_int8, 4), round(recall, 4))]
    print("\ncuantizacion int8 vs float32 (dim", dim, ")")
    print(f"  error relativo medio: {err:.4f}")
    print(f"  max/rms medio:        {pico:.2f} -> modelo "
          f"{pico / (127 * math.sqrt(12)):.4f}")
    print(f"  recall@10 de int8:    {recall:.4f}")
    print(f"  ruido de puntuacion:  {np.mean(ruido):.2f}; margen 10-11 "
          f"exacto: {np.mean(margen):.2f}")
    print("  puesto exacto de los perdidos:",
          dict(sorted(Counter(perdidos).items())))
    print(f"  memoria int8:         {mem_int8:.4f}x (codigos + escalas)")
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
    """Coste relativo del multivector frente al vector unico, bajo un modelo
    simplificado: T_q = T_d = T tokens y la misma dimension y precision por
    token que el vector unico. Entonces la memoria crece como T y las parejas
    que puntua MaxSim, como T_q*T_d = T^2 (cada pareja cuesta O(r) con r la
    dimension). Con vectores por token mas cortos o comprimidos, como en
    ColBERT, los factores reales son menores.
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
    """Una consulta al mini almacen con un vector de reserva: se genera con
    la coleccion pero no se inserta, de modo que no puede encontrarse a si
    mismo."""
    vecs, metas = _coleccion(n + 1, dim)
    alm = AlmacenVectorial(dim, metrica="coseno")
    alm.agregar(vecs[:n], metas[:n])
    consulta, tema_q = vecs[n], metas[n]["tema"]
    res = alm.buscar(consulta, k=k)
    mismos = sum(alm.metadatos[i]["tema"] == tema_q for i, _ in res)
    print(f"\ndemostracion: {k} vecinos de una consulta de reserva "
          f"(tema {tema_q}); del mismo tema: {mismos} de {k}")
    # el coseno lo fija el generador: |c|^2 / (|c|^2 + d) dentro de un tema
    centros = np.random.default_rng(SEMILLA).standard_normal((8, dim)) * 4
    centro = centros[tema_q]
    c2 = float(centro @ centro) / dim
    tema = np.array([m["tema"] for m in metas[:n]])
    cos = (vecs[:n] @ consulta) / (np.linalg.norm(vecs[:n], axis=1)
                                    * np.linalg.norm(consulta))
    print(f"  |c|^2 = {c2:.1f} d; coseno esperado {c2 / (c2 + 1):.3f}; "
          f"medio en el tema ({(tema == tema_q).sum()} vectores) "
          f"{cos[tema == tema_q].mean():.3f}; maximo fuera del tema "
          f"{cos[tema != tema_q].max():.3f}")
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
