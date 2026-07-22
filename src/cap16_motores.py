"""capitulo 16 — motores: dedicados frente a extensiones.

los motores reales (pgvector, Qdrant, Milvus, Redis, Weaviate) requieren
servidores que NO estan disponibles en este entorno, asi que este modulo no
inventa cifras de ninguno de ellos. en su lugar construye un MOTOR VECTORIAL
GENERICO desde cero ---la capa que envuelve al indice del capitulo 15 con API,
almacenamiento, metadatos y serializacion--- y mide lo que si es local y real,
que es comun a todos los motores:

  1. serializacion en la frontera cliente-servidor: tamano y coste de enviar
     vectores como JSON (texto) frente a binario (float32). por que los motores
     usan protocolos binarios.
  2. almacen de metadatos: coste de filtrar con los metadatos guardados por
     FILAS (estilo relacional ingenuo) frente a por COLUMNAS (estilo dedicado),
     segun la selectividad. la raiz de extension-vs-dedicado.
  3. desglose del coste de una consulta: filtrado + busqueda + materializacion;
     donde se va el tiempo dentro del motor.
  4. sobrecoste de la capa de motor: la busqueda cruda en el indice frente a la
     misma consulta a traves de la API completa del motor.

los productos reales se describen en el texto con su API factual (listados
ilustrativos) y tablas comparativas. es Python puro con numpy, CPU. ver
IMPLEMENTACION.md.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SEMILLA = 16


def anunciar() -> None:
    print("=" * 64)
    print("cap. 16 — motores: dedicados frente a extensiones")
    print("recursos: python + numpy · cpu. sin servidores (pgvector/Qdrant/")
    print("Milvus/Redis no disponibles); motor generico desde cero.")
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
# el motor generico: la capa comun a todos los motores vectoriales
# ---------------------------------------------------------------------------

class MotorVectorial:
    """Un motor vectorial minimo pero completo: la capa que envuelve al indice.

    Reune lo que todo motor ofrece por encima del indice del capitulo 15: una
    API uniforme (crear, upsert, query, borrar, persistir), un almacen de
    metadatos COLUMNAR para filtrar rapido, y la materializacion del resultado.
    El indice aqui es busqueda exacta (flat) por claridad; un motor real enchufa
    cualquiera de los del capitulo 15.
    """

    def __init__(self, dim: int, metrica: str = "coseno") -> None:
        self.dim = dim
        self.metrica = metrica
        self.vectores = np.empty((0, dim), dtype=np.float32)
        self.ids: List[int] = []
        self.meta_cols: Dict[str, np.ndarray] = {}   # metadatos por columnas

    def upsert(self, ids: List[int], vecs: np.ndarray,
               metas: Dict[str, np.ndarray]) -> None:
        """Inserta un lote: apila vectores y extiende las columnas de metadatos."""
        vecs = vecs.astype(np.float32)
        if self.metrica == "coseno":
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        self.vectores = np.vstack([self.vectores, vecs])
        self.ids.extend(ids)
        for col, valores in metas.items():
            prev = self.meta_cols.get(col, np.empty(0, dtype=valores.dtype))
            self.meta_cols[col] = np.concatenate([prev, valores])

    def query(self, vector: np.ndarray, k: int = 10,
              filtro: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]]
              = None) -> List[Dict]:
        """Consulta: prefiltra por metadatos (mascara columnar), busca y
        materializa los k resultados con sus metadatos."""
        if filtro is not None:
            mask = filtro(self.meta_cols)            # mascara booleana
            idx = np.where(mask)[0]
            if not len(idx):
                return []
            sims = self.vectores[idx] @ self._norm(vector)
            orden = idx[np.argsort(-sims)[:k]]
        else:
            sims = self.vectores @ self._norm(vector)
            orden = np.argsort(-sims)[:k]
        return [self._materializar(int(i)) for i in orden]

    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12) if self.metrica == "coseno" else v

    def _materializar(self, i: int) -> Dict:
        """Reune el id y los metadatos de un resultado (la fila del resultado)."""
        fila = {"id": self.ids[i]}
        for col, valores in self.meta_cols.items():
            fila[col] = valores[i].item()
        return fila

    def __len__(self) -> int:
        return len(self.ids)


def _coleccion(n: int, dim: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """n vectores en cumulos con metadatos columnares (tema, anio)."""
    rng = np.random.default_rng(SEMILLA)
    centros = rng.standard_normal((20, dim)) * 4
    tema = rng.integers(0, 20, n)
    vecs = (centros[tema] + rng.standard_normal((n, dim))).astype(np.float32)
    metas = {"tema": tema.astype(np.int32),
             "anio": rng.integers(2018, 2026, n).astype(np.int32)}
    return vecs, metas


# ---------------------------------------------------------------------------
# 1. serializacion: la frontera cliente-servidor
# ---------------------------------------------------------------------------

def simular_serializacion(dim: int = 768) -> None:
    """Tamano y coste de enviar un lote de vectores como JSON (texto) frente a
    binario (float32 crudo). Por que los motores usan protocolos binarios."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nserializacion: JSON (texto) vs binario (float32), dim", dim)
    print("  lote   json_KB  binario_KB  ratio")
    for n in (1, 10, 100, 1000):
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        t = time.perf_counter()
        js = json.dumps(vecs.tolist())
        t_json = time.perf_counter() - t
        t = time.perf_counter()
        bn = vecs.tobytes()
        t_bin = time.perf_counter() - t
        kb_json, kb_bin = len(js) / 1024, len(bn) / 1024
        filas.append((n, round(kb_json, 1), round(kb_bin, 1),
                      round(kb_json / kb_bin, 2)))
        print(f"  {n:<5}  {kb_json:>8.1f}  {kb_bin:>9.1f}  "
              f"{kb_json/kb_bin:.2f}  (t_json/t_bin~{t_json/max(t_bin,1e-9):.0f})")
    _escribir(os.path.join("data", "cap16_serializacion.dat"),
              "tamano (KB) de un lote de N vectores dim %d como JSON vs binario "
              "float32, y su ratio" % dim,
              "n  json_kb  binario_kb  ratio", filas)


# ---------------------------------------------------------------------------
# 2. almacen de metadatos: filas vs columnas
# ---------------------------------------------------------------------------

def simular_metadatos(n: int = 200_000, reps: int = 30) -> None:
    """Coste de aplicar un filtro de metadatos guardados por FILAS (lista de
    dicts, estilo relacional ingenuo) frente a por COLUMNAS (numpy, estilo
    dedicado), segun la selectividad. La raiz de extension-vs-dedicado."""
    rng = np.random.default_rng(SEMILLA)
    tema_col = rng.integers(0, 20, n).astype(np.int32)
    filas_dict = [{"tema": int(t)} for t in tema_col]     # almacen por filas
    out = []
    print("\nmetadatos: filtrado por filas (dicts) vs columnas (numpy)")
    print("  sel(%)  filas_ms  columnas_ms  aceleracion")
    for temas_ok in (1, 5, 10, 20):
        permitidos = set(range(temas_ok))
        sel = 100.0 * temas_ok / 20
        t = time.perf_counter()
        for _ in range(reps):
            _ = [i for i, m in enumerate(filas_dict)
                 if m["tema"] in permitidos]
        t_filas = (time.perf_counter() - t) / reps
        permit_arr = np.arange(temas_ok)
        t = time.perf_counter()
        for _ in range(reps):
            _ = np.where(np.isin(tema_col, permit_arr))[0]
        t_col = (time.perf_counter() - t) / reps
        out.append((round(sel, 1), round(t_filas * 1e3, 3),
                    round(t_col * 1e3, 3), round(t_filas / t_col, 1)))
        print(f"  {sel:<6.1f}  {t_filas*1e3:>8.3f}  {t_col*1e3:>10.3f}"
              f"  {t_filas/t_col:>6.1f}x")
    _escribir(os.path.join("data", "cap16_metadatos.dat"),
              "coste (ms) de filtrar metadatos por filas (dicts) vs columnas "
              "(numpy) y aceleracion, segun selectividad (n=%d)" % n,
              "selectividad  filas_ms  columnas_ms  aceleracion", out)


# ---------------------------------------------------------------------------
# 3. desglose del coste de una consulta
# ---------------------------------------------------------------------------

def simular_desglose(n: int = 200_000, dim: int = 256, reps: int = 50) -> None:
    """Donde se va el tiempo de una consulta en el motor: filtrado de
    metadatos, busqueda vectorial y materializacion del resultado."""
    rng = np.random.default_rng(SEMILLA)
    vecs, metas = _coleccion(n, dim)
    motor = MotorVectorial(dim, metrica="coseno")
    motor.upsert(list(range(n)), vecs, metas)
    consultas = rng.standard_normal((reps, dim)).astype(np.float32)
    permit = np.arange(5)                              # 25% selectividad
    t_filtro = t_busq = t_mat = 0.0
    for q in consultas:
        qn = motor._norm(q)
        t = time.perf_counter()
        mask = np.isin(motor.meta_cols["tema"], permit)
        idx = np.where(mask)[0]
        t_filtro += time.perf_counter() - t
        t = time.perf_counter()
        sims = motor.vectores[idx] @ qn
        orden = idx[np.argsort(-sims)[:10]]
        t_busq += time.perf_counter() - t
        t = time.perf_counter()
        _ = [motor._materializar(int(i)) for i in orden]
        t_mat += time.perf_counter() - t
    tot = t_filtro + t_busq + t_mat
    filas = [("filtrado", round(100 * t_filtro / tot, 1)),
             ("busqueda", round(100 * t_busq / tot, 1)),
             ("materializacion", round(100 * t_mat / tot, 1))]
    print("\ndesglose del coste de una consulta (25% selectividad)")
    for f in filas:
        print(f"  {f[0]:<16} {f[1]:>5.1f}%")
    _escribir(os.path.join("data", "cap16_desglose.dat"),
              "reparto del coste de una consulta del motor entre filtrado, "
              "busqueda y materializacion (porcentaje, n=%d)" % n,
              "etapa  porcentaje", filas)


# ---------------------------------------------------------------------------
# 4. sobrecoste de la capa de motor
# ---------------------------------------------------------------------------

def simular_sobrecoste(n: int = 100_000, dim: int = 256, reps: int = 50) -> None:
    """La busqueda cruda en el indice frente a la misma a traves de la API
    completa del motor (con normalizacion, filtro trivial y materializacion):
    el precio de la abstraccion."""
    rng = np.random.default_rng(SEMILLA)
    vecs, metas = _coleccion(n, dim)
    motor = MotorVectorial(dim, metrica="coseno")
    motor.upsert(list(range(n)), vecs, metas)
    base = motor.vectores                              # ya normalizado
    consultas = rng.standard_normal((reps, dim)).astype(np.float32)
    t = time.perf_counter()
    for q in consultas:
        qn = q / np.linalg.norm(q)
        sims = base @ qn
        _ = np.argsort(-sims)[:10]                     # busqueda cruda
    t_crudo = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for q in consultas:
        _ = motor.query(q, k=10)                       # API completa
    t_motor = (time.perf_counter() - t) / reps
    filas = [("indice_crudo", round(t_crudo * 1e3, 3)),
             ("motor_completo", round(t_motor * 1e3, 3))]
    print("\nsobrecoste de la capa de motor (n=%d)" % n)
    print(f"  indice crudo:   {t_crudo*1e3:.3f} ms")
    print(f"  motor completo: {t_motor*1e3:.3f} ms")
    print(f"  sobrecoste:     {100*(t_motor-t_crudo)/t_crudo:.0f}%")
    _escribir(os.path.join("data", "cap16_sobrecoste.dat"),
              "latencia (ms) de la busqueda cruda en el indice vs la consulta "
              "completa por la API del motor (n=%d, dim=%d)" % (n, dim),
              "capa  latencia_ms", filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta real a traves del motor
# ---------------------------------------------------------------------------

def simular_dimension(n: int = 1_000_000) -> None:
    """coste de almacenar N vectores segun la dimension, en float32 y en int8:
    el coste crece linealmente con la dimension, asi que reducirla ---p. ej. con
    embeddings Matryoshka--- abarata la factura tanto como acelera."""
    filas = []
    print("\ndimension: GB de %d vectores segun dim y tipo" % n)
    print("  dim    float32_GB  int8_GB")
    for d in (128, 256, 512, 768, 1024, 1536):
        gb32 = n * d * 4 / 1e9
        gb8 = n * d * 1 / 1e9
        filas.append((d, round(gb32, 2), round(gb8, 2)))
        print(f"  {d:<5}  {gb32:>9.2f}  {gb8:>7.2f}")
    _escribir(os.path.join("data", "cap16_dimension.dat"),
              "memoria (GB) de %d vectores segun la dimension, en float32 y "
              "int8" % n, "dim  float32_gb  int8_gb", filas)


def simular_escala(dim: int = 256, reps: int = 20) -> None:
    """latencia de una consulta por la API del motor segun el tamano de la
    coleccion: el motor escala como su indice (aqui flat, O(n)). con un indice
    aproximado del cap. 15 la curva seria mucho mas plana."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nescala: latencia del motor (indice flat) segun N")
    print("  N         latencia_ms")
    for n in (10_000, 50_000, 100_000, 200_000, 500_000):
        vecs, metas = _coleccion(n, dim)
        motor = MotorVectorial(dim, metrica="coseno")
        motor.upsert(list(range(n)), vecs, metas)
        consultas = rng.standard_normal((reps, dim)).astype(np.float32)
        t0 = time.perf_counter()
        for q in consultas:
            motor.query(q, k=10)
        lat = (time.perf_counter() - t0) / reps * 1e3
        filas.append((n, round(lat, 3)))
        print(f"  {n:<9} {lat:>8.3f}")
    _escribir(os.path.join("data", "cap16_escala.dat"),
              "latencia (ms) de una consulta por la API del motor (indice flat) "
              "segun el tamano de la coleccion (dim %d)" % dim,
              "n  latencia_ms", filas)


def demostracion(n: int = 20_000, dim: int = 256, k: int = 15) -> None:
    """Una consulta end-to-end por la API del motor, con filtro de metadatos."""
    vecs, metas = _coleccion(n, dim)
    motor = MotorVectorial(dim, metrica="coseno")
    motor.upsert(list(range(n)), vecs, metas)
    rng = np.random.default_rng(SEMILLA + 1)
    q = vecs[0] + 0.3 * rng.standard_normal(dim).astype(np.float32)
    filtro = lambda cols: cols["anio"] >= 2023        # solo recientes
    res = motor.query(q, k=k, filtro=filtro)
    print(f"\ndemostracion: {k} resultados por la API del motor (filtro anio>=2023)")
    print("  rank  id      tema  anio")
    print("  ----  ------  ----  ----")
    for r, fila in enumerate(res):
        print(f"  {r:<4}  {fila['id']:<6}  {fila['tema']:<4}  {fila['anio']}")


def main() -> None:
    anunciar()
    simular_serializacion()
    simular_metadatos()
    simular_desglose()
    simular_sobrecoste()
    simular_escala()
    simular_dimension()
    demostracion(15)


if __name__ == "__main__":
    main()
