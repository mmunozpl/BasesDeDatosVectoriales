"""capitulo 16: motores: dedicados frente a extensiones.

los motores reales (pgvector, Qdrant, Milvus, Redis, Weaviate) requieren
servidores que no estan disponibles en este entorno, asi que este modulo no
inventa cifras de ninguno de ellos. construye una maqueta funcional del nucleo
logico de un motor (MotorVectorial: API de upsert y query, metadatos por
columnas, filtrado y materializacion) y mide lo que es local y reproducible en
esa maqueta. no es un benchmark de productos: omite durabilidad, recuperacion,
transacciones, concurrencia, control de acceso, mantenimiento de indices y
distribucion, y sus cifras no se trasladan a ningun motor real.

mide seis cosas y comprueba una:

  1. serializacion: tamano y tiempo de codificar y descodificar un lote de
     vectores como JSON frente al payload binario minimo (los bytes float32).
  2. metadatos: filtrar una lista de diccionarios de Python (registro a
     registro) frente a un array de numpy por campo (vectorizado).
  3. desglose del tiempo de una consulta con filtro: filtrado, busqueda y
     materializacion, en esta maqueta.
  4. sobrecoste de la API de la maqueta frente a la busqueda cruda (producto
     matriz-vector y seleccion de los k mejores).
  5. latencia de la consulta segun el tamano de la coleccion (busqueda flat).
  6. payload vectorial bruto segun la dimension, en float32 e int8.
  7. sharding: el top-k exacto de cada fragmento basta para el top-k global.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

# un solo hilo de BLAS: los tiempos de la maqueta son mas estables y no
# dependen de cuantos nucleos tenga libres la maquina
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

SEMILLA = 16
METRICAS = ("coseno", "producto", "l2")


def anunciar() -> None:
    print("=" * 64)
    print("cap. 16: motores: dedicados frente a extensiones")
    print("recursos: python + numpy · cpu. sin servidores (pgvector/Qdrant/")
    print("Milvus/Redis no disponibles); maqueta del nucleo de un motor.")
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


def _top(puntos: np.ndarray, k: int) -> np.ndarray:
    """indices de las k puntuaciones mayores: seleccion parcial O(n) y orden
    de los k."""
    k = min(k, len(puntos))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-puntos, k - 1)[:k]
    return idx[np.argsort(-puntos[idx], kind="stable")]


# ---------------------------------------------------------------------------
# la maqueta del motor: el nucleo logico comun a los motores vectoriales
# ---------------------------------------------------------------------------

class MotorVectorial:
    """Maqueta funcional del nucleo logico de un motor vectorial.

    Permite observar la ingesta (upsert por identificador), el filtrado por
    metadatos guardados por columnas, la busqueda y la materializacion del
    resultado. La busqueda es flat (exacta); un motor real enchufa un indice
    del capitulo 15 y anade durabilidad, recuperacion, transacciones,
    concurrencia, control de acceso, gestion de esquemas, mantenimiento de
    indices, telemetria y distribucion, que aqui faltan.

    Metricas: "coseno" (normaliza al ingerir y rechaza vectores nulos),
    "producto" (producto interno sin normalizar) y "l2" (menos la distancia
    euclidea al cuadrado). En las tres, mayor puntuacion es mas proximo.
    """

    def __init__(self, dim: int, metrica: str = "coseno",
                 capacidad: int = 1024) -> None:
        if metrica not in METRICAS:
            raise ValueError(f"metrica desconocida: {metrica} {METRICAS}")
        self.dim, self.metrica = dim, metrica
        self._vec = np.empty((capacidad, dim), dtype=np.float32)
        self._cols: Dict[str, np.ndarray] = {}   # metadatos por columnas
        self.n = 0
        self.ids: List[int] = []
        self.pos_por_id: Dict[int, int] = {}      # id -> posicion

    @property
    def vectores(self) -> np.ndarray:
        return self._vec[:self.n]

    @property
    def meta_cols(self) -> Dict[str, np.ndarray]:
        return {c: v[:self.n] for c, v in self._cols.items()}

    def _preparar(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"se esperaban vectores de dimension {self.dim}")
        if self.metrica == "coseno":
            normas = np.linalg.norm(vecs, axis=1, keepdims=True)
            if np.any(normas == 0):
                raise ValueError("el coseno no admite vectores nulos")
            vecs = vecs / normas
        return vecs

    def _reservar(self, nuevos: int) -> None:
        """dobla la capacidad del bufer (vectores y columnas) si no caben:
        coste amortizado constante por fila, en lugar de recopiar todo en
        cada lote como haria np.vstack."""
        cap = len(self._vec)
        if self.n + nuevos <= cap:
            return
        while cap < self.n + nuevos:
            cap *= 2
        v = np.empty((cap, self.dim), dtype=np.float32)
        v[:self.n] = self._vec[:self.n]
        self._vec = v
        for c, col in self._cols.items():
            nueva = np.empty(cap, dtype=col.dtype)
            nueva[:self.n] = col[:self.n]
            self._cols[c] = nueva

    def _crear_columnas(self, metas: Dict[str, np.ndarray]) -> None:
        if not self._cols:
            cap = len(self._vec)
            self._cols = {c: np.empty(cap, dtype=np.asarray(v).dtype)
                          for c, v in metas.items()}

    def upsert(self, ids: List[int], vecs: np.ndarray,
               metas: Dict[str, np.ndarray]) -> None:
        """Inserta o actualiza un lote. Un id nuevo se anade al final; uno que
        ya existe reemplaza su vector y sus metadatos en su posicion."""
        vecs = self._preparar(vecs)
        if len(ids) != len(vecs) or len(set(ids)) != len(ids):
            raise ValueError("un id distinto por vector")
        if self._cols and set(metas) != set(self._cols):
            raise ValueError(f"columnas esperadas: {sorted(self._cols)}")
        for c, val in metas.items():
            if len(val) != len(ids):
                raise ValueError(f"la columna {c} no tiene un valor por id")
        self._crear_columnas(metas)              # el primer lote fija el esquema
        self._reservar(sum(i not in self.pos_por_id for i in ids))
        for j, i in enumerate(ids):
            p = self.pos_por_id.get(i)
            if p is None:                        # insercion al final
                p = self.n
                self.n += 1
                self.ids.append(i)
                self.pos_por_id[i] = p
            self._vec[p] = vecs[j]               # actualizacion en su sitio
            for c, val in metas.items():
                self._cols[c][p] = val[j]

    def _puntuar(self, base: np.ndarray, q: np.ndarray) -> np.ndarray:
        if self.metrica == "l2":
            return -((base - q) ** 2).sum(1)
        return base @ q                          # coseno (normalizado) o producto

    def query(self, vector: np.ndarray, k: int = 10,
              filtro: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]]
              = None) -> List[Dict]:
        """Estrategia fija: si hay filtro, prefiltra con la mascara columnar y
        busca en el subconjunto; si no, busca en toda la coleccion. No elige
        entre planes alternativos. Materializa los k resultados."""
        q = self._preparar(np.asarray(vector)[None, :])[0]
        if filtro is not None:
            idx = np.flatnonzero(filtro(self.meta_cols))   # mascara columnar
            if not len(idx):
                return []
            orden = idx[_top(self._puntuar(self.vectores[idx], q), k)]
        else:
            orden = _top(self._puntuar(self.vectores, q), k)
        return [self._materializar(int(i)) for i in orden]

    def _materializar(self, i: int) -> Dict:
        """Reune el id y los metadatos de un resultado (la fila del resultado)."""
        fila = {"id": self.ids[i]}
        for col, valores in self._cols.items():
            fila[col] = valores[i].item()
        return fila

    def __len__(self) -> int:
        return self.n


def _coleccion(n: int, dim: int,
               semilla: int = SEMILLA) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """n vectores en cumulos con metadatos columnares (tema, anio)."""
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((20, dim)) * 4
    tema = rng.integers(0, 20, n)
    vecs = (centros[tema] + rng.standard_normal((n, dim))).astype(np.float32)
    metas = {"tema": tema.astype(np.int32),
             "anio": rng.integers(2018, 2026, n).astype(np.int32)}
    return vecs, metas


def _motor(n: int, dim: int) -> Tuple[MotorVectorial, np.ndarray]:
    """una maqueta con n vectores ingeridos en lotes de 10 000, y la
    coleccion de la que salen (para consultas reservadas)."""
    vecs, metas = _coleccion(n, dim)
    motor = MotorVectorial(dim)
    for a in range(0, n, 10_000):
        b = min(a + 10_000, n)
        motor.upsert(list(range(a, b)), vecs[a:b],
                     {c: v[a:b] for c, v in metas.items()})
    return motor, vecs


# ---------------------------------------------------------------------------
# 1. serializacion: la frontera cliente-servidor
# ---------------------------------------------------------------------------

def simular_serializacion(dim: int = 768, reps: int = 5) -> None:
    """Tamano y tiempo (ida y vuelta) de un lote de vectores como JSON frente
    al payload binario minimo: los bytes float32 de ndarray.tobytes(). Un
    protocolo real anade dimension, tipo, orden de bytes, encuadre, ids y
    metadatos."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nserializacion: JSON frente a bytes float32, dim", dim)
    print("  lote   json_KB  binario_KB  ratio  json_ms  binario_ms")
    for n in (1, 10, 100, 1000):
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        tj, tb = [], []
        for _ in range(reps):
            t = time.perf_counter()
            js = json.dumps(vecs.tolist())
            _ = np.array(json.loads(js), dtype=np.float32)
            tj.append(time.perf_counter() - t)
            t = time.perf_counter()
            bn = vecs.tobytes()
            _ = np.frombuffer(bn, dtype=np.float32).reshape(n, dim)
            tb.append(time.perf_counter() - t)
        kb_json, kb_bin = len(js) / 1024, len(bn) / 1024
        mj, mb = 1e3 * float(np.median(tj)), 1e3 * float(np.median(tb))
        filas.append((n, round(kb_json, 1), round(kb_bin, 1),
                      round(kb_json / kb_bin, 2), round(mj, 4), round(mb, 4)))
        print(f"  {n:<5}  {kb_json:>8.1f}  {kb_bin:>9.1f}  "
              f"{kb_json/kb_bin:>5.2f}  {mj:>7.3f}  {mb:>10.4f}")
    _escribir(os.path.join("data", "cap16_serializacion.dat"),
              "tamano (KB) y tiempo mediano de ida y vuelta (ms) de un lote de "
              "N vectores dim %d como JSON frente a bytes float32" % dim,
              "n  json_kb  binario_kb  ratio  json_ms  binario_ms", filas)


# ---------------------------------------------------------------------------
# 2. metadatos: registros de Python frente a arrays por campo
# ---------------------------------------------------------------------------

def simular_metadatos(n: int = 200_000, reps: int = 30) -> None:
    """Filtrar una lista de diccionarios de Python (registro a registro)
    frente a un array de numpy por campo (operacion vectorizada), segun la
    selectividad. Ilustra localidad y vectorizacion en esta maqueta; no mide
    como guarda los metadatos ningun motor real."""
    rng = np.random.default_rng(SEMILLA)
    tema_col = rng.integers(0, 20, n).astype(np.int32)
    filas_dict = [{"tema": int(t)} for t in tema_col]     # registros
    out = []
    print("\nmetadatos: lista de dicts frente a array por campo")
    print("  sel(%)  registros_ms  columnas_ms  aceleracion")
    for temas_ok in (1, 5, 10, 20):
        permitidos = set(range(temas_ok))
        sel = 100.0 * temas_ok / 20
        permit_arr = np.arange(temas_ok)
        tf, tc = [], []
        for _ in range(reps):                          # el minimo de reps
            t = time.perf_counter()
            _ = [i for i, m in enumerate(filas_dict)
                 if m["tema"] in permitidos]
            tf.append(time.perf_counter() - t)
            t = time.perf_counter()
            _ = np.flatnonzero(np.isin(tema_col, permit_arr))
            tc.append(time.perf_counter() - t)
        t_filas, t_col = min(tf), min(tc)
        out.append((round(sel, 1), round(t_filas * 1e3, 3),
                    round(t_col * 1e3, 3), round(t_filas / t_col, 1)))
        print(f"  {sel:<6.1f}  {t_filas*1e3:>12.3f}  {t_col*1e3:>11.3f}"
              f"  {t_filas/t_col:>6.1f}x")
    _escribir(os.path.join("data", "cap16_metadatos.dat"),
              "coste minimo de %d repeticiones (ms) de filtrar una lista de "
              "dicts de Python frente a un array de numpy por campo, y "
              "aceleracion, segun selectividad (n=%d)" % (reps, n),
              "selectividad  filas_ms  columnas_ms  aceleracion", out)


# ---------------------------------------------------------------------------
# 3. desglose del tiempo de una consulta
# ---------------------------------------------------------------------------

def simular_desglose(n: int = 200_000, dim: int = 256, reps: int = 50) -> None:
    """Reparto del tiempo de una consulta con filtro (25 % de selectividad)
    en la maqueta: mascara de metadatos, busqueda en el subconjunto y
    materializacion de los 10 resultados."""
    rng = np.random.default_rng(SEMILLA)
    motor, _ = _motor(n, dim)
    consultas = rng.standard_normal((reps, dim)).astype(np.float32)
    permit = np.arange(5)                              # 25 % de selectividad
    t_filtro = t_busq = t_mat = 0.0
    for q in consultas:
        qn = motor._preparar(q[None, :])[0]
        t = time.perf_counter()
        idx = np.flatnonzero(np.isin(motor.meta_cols["tema"], permit))
        t_filtro += time.perf_counter() - t
        t = time.perf_counter()
        orden = idx[_top(motor.vectores[idx] @ qn, 10)]
        t_busq += time.perf_counter() - t
        t = time.perf_counter()
        _ = [motor._materializar(int(i)) for i in orden]
        t_mat += time.perf_counter() - t
    tot = t_filtro + t_busq + t_mat
    filas = [("filtrado", round(100 * t_filtro / tot, 1)),
             ("busqueda", round(100 * t_busq / tot, 1)),
             ("materializacion", round(100 * t_mat / tot, 1))]
    print("\ndesglose del tiempo de una consulta (25 % de selectividad)")
    for f in filas:
        print(f"  {f[0]:<16} {f[1]:>5.1f} %")
    print(f"  total medio: {1e3 * tot / reps:.3f} ms; subconjunto filtrado: "
          f"{len(idx)} vectores ({100 * len(idx) / n:.1f} %)")
    _escribir(os.path.join("data", "cap16_desglose.dat"),
              "reparto del tiempo de una consulta de la maqueta entre "
              "filtrado, busqueda y materializacion (porcentaje, n=%d)" % n,
              "etapa  porcentaje", filas)


# ---------------------------------------------------------------------------
# 4. sobrecoste de la API de la maqueta
# ---------------------------------------------------------------------------

def simular_sobrecoste(n: int = 100_000, dim: int = 256, reps: int = 50) -> None:
    """La busqueda cruda (producto matriz-vector y seleccion de los 10
    mejores) frente a la misma consulta por la API de la maqueta, que anade
    validacion, normalizacion y materializacion, sin filtro."""
    rng = np.random.default_rng(SEMILLA)
    motor, _ = _motor(n, dim)
    base = motor.vectores                              # ya normalizado
    consultas = rng.standard_normal((reps, dim)).astype(np.float32)
    motor.query(consultas[0], k=10)                    # calentamiento
    crudo, api = [], []
    for q in consultas:                                # alternadas
        t = time.perf_counter()
        _ = _top(base @ (q / np.linalg.norm(q)), 10)   # busqueda cruda
        crudo.append(time.perf_counter() - t)
        t = time.perf_counter()
        _ = motor.query(q, k=10)                       # API de la maqueta
        api.append(time.perf_counter() - t)
    t_crudo, t_motor = min(crudo), min(api)           # el minimo: menos ruido
    filas = [("indice_crudo", round(t_crudo * 1e3, 3)),
             ("motor_completo", round(t_motor * 1e3, 3))]
    print("\nsobrecoste de la API de la maqueta (n=%d)" % n)
    print(f"  busqueda cruda: {t_crudo*1e3:.3f} ms")
    print(f"  API completa:   {t_motor*1e3:.3f} ms")
    print(f"  sobrecoste:     {100*(t_motor-t_crudo)/t_crudo:.1f} %")
    _escribir(os.path.join("data", "cap16_sobrecoste.dat"),
              "latencia minima de %d consultas (ms) de la busqueda cruda frente "
              "a la API de la maqueta (n=%d, dim=%d, un hilo)" % (reps, n, dim),
              "capa  latencia_ms", filas)


# ---------------------------------------------------------------------------
# 5. escala, 6. dimension y 7. sharding
# ---------------------------------------------------------------------------

def simular_escala(dim: int = 256, reps: int = 30) -> None:
    """latencia de una consulta por la API de la maqueta segun el tamano de
    la coleccion, con busqueda flat: O(n d) del producto mas O(n) de la
    seleccion parcial. se toma el minimo de las consultas, que filtra las
    interferencias de otros procesos de la maquina."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nescala: latencia de la maqueta (busqueda flat) segun N")
    print("  N         latencia_ms")
    for n in (10_000, 50_000, 100_000, 200_000, 500_000):
        motor, _ = _motor(n, dim)
        consultas = rng.standard_normal((reps, dim)).astype(np.float32)
        motor.query(consultas[0], k=10)                # calentamiento
        tiempos = []
        for q in consultas:
            t0 = time.perf_counter()
            motor.query(q, k=10)
            tiempos.append(time.perf_counter() - t0)
        lat = float(np.min(tiempos)) * 1e3        # el minimo: menos ruido ajeno
        filas.append((n, round(lat, 3)))
        print(f"  {n:<9} {lat:>8.3f}")
    _escribir(os.path.join("data", "cap16_escala.dat"),
              "latencia minima de 30 consultas (ms) por la API de la maqueta "
              "(busqueda flat, un hilo) segun el tamano de la coleccion (dim %d)"
              % dim,
              "n  latencia_ms", filas)


def simular_dimension(n: int = 1_000_000) -> None:
    """payload vectorial bruto de N vectores segun la dimension, en float32 y
    en int8; no incluye indice, identificadores, escalas ni metadatos."""
    filas = []
    print("\ndimension: payload bruto (GB) de %d vectores" % n)
    print("  dim    float32_GB  int8_GB")
    for d in (128, 256, 512, 768, 1024, 1536):
        gb32 = n * d * 4 / 1e9
        gb8 = n * d * 1 / 1e9
        filas.append((d, round(gb32, 2), round(gb8, 2)))
        print(f"  {d:<5}  {gb32:>9.2f}  {gb8:>7.2f}")
    _escribir(os.path.join("data", "cap16_dimension.dat"),
              "payload vectorial bruto (GB decimales) de %d vectores segun la "
              "dimension, en float32 e int8" % n,
              "dim  float32_gb  int8_gb", filas)


def comprobar_shards(n: int = 100_000, dim: int = 128, fragmentos: int = 8,
                     k: int = 10, nq: int = 200) -> None:
    """busqueda exacta particionada por hash del id: fusionar el top-k exacto
    de cada fragmento reproduce el top-k global, porque cualquier elemento del
    top-k global esta en el top-k de su fragmento."""
    vecs, _ = _coleccion(n + nq, dim, semilla=SEMILLA + 3)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    base, consultas = vecs[:n], vecs[n:]
    frag = np.arange(n) % fragmentos                   # hash del id
    distintos = 0
    for q in consultas:
        s = base @ q
        glob = set(_top(s, k).tolist())
        cand = np.concatenate([np.flatnonzero(frag == f)[
            _top(s[frag == f], k)] for f in range(fragmentos)])
        fusion = set(cand[_top(s[cand], k)].tolist())
        distintos += glob != fusion
    print(f"\nsharding exacto ({fragmentos} fragmentos, k={k}): "
          f"{distintos} de {nq} consultas difieren del top-k global")


def demostracion(n: int = 20_000, dim: int = 256, k: int = 15) -> None:
    """Una consulta end-to-end por la API, con filtro de metadatos, y un
    upsert que reemplaza un id existente sin duplicarlo."""
    vecs, metas = _coleccion(n + 1, dim)
    motor = MotorVectorial(dim)
    motor.upsert(list(range(n)), vecs[:n],
                 {c: v[:n] for c, v in metas.items()})
    q = vecs[n]                                        # reservada, no indexada
    filtro = lambda cols: cols["anio"] >= 2023         # solo recientes
    res = motor.query(q, k=k, filtro=filtro)
    print(f"\ndemostracion: {k} resultados por la API (filtro anio>=2023)")
    print(f"  tema de la consulta: {int(metas['tema'][n])}")
    # cuantos pasan el filtro y como separa el generador los temas
    tq, recientes = metas["tema"][n], metas["anio"][:n] >= 2023
    del_tema = metas["tema"][:n] == tq
    cos = (vecs[:n] @ q) / (np.linalg.norm(vecs[:n], axis=1)
                            * np.linalg.norm(q))
    print(f"  publicados desde 2023: {recientes.sum()} "
          f"({100 * recientes.mean():.1f} %); del tema y recientes: "
          f"{(recientes & del_tema).sum()}")
    print(f"  coseno minimo en el tema {cos[del_tema].min():.3f}; maximo "
          f"fuera del tema {cos[~del_tema].max():.3f}")
    print("  rank  id      tema  anio")
    print("  ----  ------  ----  ----")
    for r, fila in enumerate(res):
        print(f"  {r:<4}  {fila['id']:<6}  {fila['tema']:<4}  {fila['anio']}")
    antes = len(motor)
    motor.upsert([res[0]["id"]], vecs[n][None, :],
                 {"tema": np.array([res[0]["tema"]], dtype=np.int32),
                  "anio": np.array([2025], dtype=np.int32)})
    print(f"  upsert de un id existente: {antes} -> {len(motor)} vectores; "
          f"anio ahora {motor._materializar(motor.pos_por_id[res[0]['id']])['anio']}")


def main() -> None:
    anunciar()
    simular_serializacion()
    simular_metadatos()
    simular_desglose()
    simular_sobrecoste()
    simular_escala()
    simular_dimension()
    comprobar_shards()
    demostracion()


if __name__ == "__main__":
    main()
