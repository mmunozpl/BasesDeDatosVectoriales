"""capitulo 20: el futuro hibrido, planes de ejecucion de la consulta hibrida.

mide en numpy, con datos sinteticos de respuesta conocida, propiedades de
planes concretos de filtrado y de mantenimiento. no compara productos ni
arquitecturas fisicas: los dos planes se ejecutan sobre los mismos arrays y no
hay red, serializacion ni coordinacion entre sistemas.

  1. top-k global y filtro despues (postfiltrado ingenuo) frente a busqueda
     restringida al conjunto elegible (prefiltrado exacto, que coincide con la
     referencia por construccion), con un filtro independiente de la similitud
     y con filtros correlacionados positiva y negativamente.
  2. sobremuestreo: candidatos que hay que recorrer para reunir k validos,
     media y percentil 95, frente a la esperanza 1/s y su cuantil teorico
     bajo independencia.
  3. rancidez: un indice con una fraccion fija de vectores obsoletos (una
     perturbacion abstracta del vector actual), evaluado con todas las
     consultas sobre el mismo estado.
  4. mantenimiento de un indice particionado (tipo IVF) bajo altas que
     derivan: cada cadencia de reconstruccion ve los mismos datos, altas y
     consultas, con tres semillas; recall a lo largo del flujo y coste en
     vectores procesados.
  5. demostracion: filtro estructurado, ranking vectorial y proyeccion de un
     documento, y el mismo resultado obtenido por dos sistemas que se
     comunican el conjunto elegible.

las cifras dependen solo de la semilla.
"""

from __future__ import annotations

import math
import os

import numpy as np

SEMILLA = 20


def anunciar() -> None:
    print("=" * 64)
    print("cap. 20: el futuro hibrido, planes de la consulta hibrida")
    print("recursos: python + numpy, cpu")
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


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("no se normaliza un vector nulo")
    return x / n


def _top(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(sims))
    part = np.argpartition(-sims, k - 1)[:k]
    return part[np.argsort(-sims[part], kind="stable")]


# ---------------------------------------------------------------------------
# filtros con y sin correlacion con la similitud
# ---------------------------------------------------------------------------

def escenario(rng, n: int, dim: int, sel: float) -> dict:
    """una coleccion y tres filtros de selectividad sel: uno aleatorio
    (independiente de cualquier consulta) y otro que deja pasar el sel mas
    alto de la proyeccion sobre una direccion u. las consultas cercanas a u
    tienen sus vecinos dentro del filtro (correlacion positiva) y las cercanas
    a -u, fuera (negativa)."""
    base = _norm(rng.standard_normal((n, dim)))
    u = _norm(rng.standard_normal(dim))
    proy = base @ u
    umbral = np.quantile(proy, 1 - sel)
    return {"base": base, "u": u,
            "indep": rng.random(n) < sel,
            "dir": proy >= umbral}


def consulta(rng, esc: dict, tipo: str) -> tuple[np.ndarray, np.ndarray]:
    """devuelve (consulta, filtro). tipo: indep, pos o neg."""
    dim = esc["base"].shape[1]
    if tipo == "indep":
        return _norm(rng.standard_normal(dim)), esc["indep"]
    signo = 1.0 if tipo == "pos" else -1.0
    q = _norm(signo * 0.5 * esc["u"]
              + rng.standard_normal(dim) / math.sqrt(dim))
    return q, esc["dir"]


def postfiltro_ingenuo(sims: np.ndarray, pasa: np.ndarray,
                       k: int) -> np.ndarray:
    """top-k global y filtro despues: puede devolver menos de k."""
    top = _top(sims, k)
    return top[pasa[top]]


def prefiltro_exacto(sims: np.ndarray, pasa: np.ndarray, k: int) -> np.ndarray:
    """busqueda exhaustiva en el conjunto elegible: devuelve los k mejores
    validos si hay al menos k."""
    idx = np.flatnonzero(pasa)
    return idx[_top(sims[idx], k)]


def simular_planes(n: int = 20_000, dim: int = 64, k: int = 10,
                   nq: int = 300) -> None:
    """recall del postfiltrado ingenuo respecto del top-k filtrado exacto
    (el prefiltrado exacto coincide con esa referencia por construccion)."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nplanes: recall del postfiltrado ingenuo frente al top-%d "
          "filtrado" % k)
    print("  sel    indep   pos     neg     (prefiltrado = 1 por construccion)")
    for sel in (0.5, 0.2, 0.1, 0.05, 0.01):
        esc = escenario(rng, n, dim, sel)
        fila = [sel]
        for tipo in ("indep", "pos", "neg"):
            rec = 0.0
            for _ in range(nq):
                q, pasa = consulta(rng, esc, tipo)
                sims = esc["base"] @ q
                oro = prefiltro_exacto(sims, pasa, k)
                post = postfiltro_ingenuo(sims, pasa, k)
                rec += len(set(post.tolist()) & set(oro.tolist())) / len(oro)
            fila.append(round(rec / nq, 4))
        filas.append(tuple(fila))
        print(f"  {sel:<5}  {fila[1]:.4f}  {fila[2]:.4f}  {fila[3]:.4f}")
    _escribir(os.path.join("data", "cap20_prefiltrado.dat"),
              "recall del top-%d global filtrado despues respecto del top-%d "
              "filtrado exacto, con filtro independiente y correlacionado "
              "positiva y negativamente con la similitud (n=%d)" % (k, k, n),
              "selectividad  indep  pos  neg", filas)


# ---------------------------------------------------------------------------
# sobremuestreo
# ---------------------------------------------------------------------------

def cuantil_teorico(k: int, s: float, p: float = 0.95) -> int:
    """menor m tal que P(al menos k validos entre m candidatos) >= p, si cada
    candidato pasa con probabilidad s de forma independiente."""
    m = k
    while True:
        prob = 0.0
        for j in range(k, m + 1):
            prob += math.exp(math.lgamma(m + 1) - math.lgamma(j + 1)
                             - math.lgamma(m - j + 1) + j * math.log(s)
                             + (m - j) * math.log1p(-s))
        if prob >= p:
            return m
        m += 1


def _recorrido(sims: np.ndarray, pasa: np.ndarray, k: int) -> int:
    """candidatos que hay que recorrer en orden de similitud hasta reunir k
    validos."""
    orden = np.argsort(-sims, kind="stable")
    validos = np.cumsum(pasa[orden])
    return int(np.searchsorted(validos, k)) + 1


def simular_sobremuestreo(n: int = 20_000, dim: int = 64, k: int = 10,
                          nq: int = 200) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    filas = []
    print("\nsobremuestreo: candidatos / k para reunir %d validos" % k)
    print("  sel    1/s    p95_teo  medio  p95    pos    neg")
    for sel in (0.5, 0.2, 0.1, 0.05, 0.01):
        esc = escenario(rng, n, dim, sel)
        f = {t: [] for t in ("indep", "pos", "neg")}
        for t in f:
            for _ in range(nq):
                q, pasa = consulta(rng, esc, t)
                f[t].append(_recorrido(esc["base"] @ q, pasa, k) / k)
        teo95 = cuantil_teorico(k, sel) / k
        fila = (sel, round(1 / sel, 1), round(teo95, 1),
                round(float(np.mean(f["indep"])), 1),
                round(float(np.percentile(f["indep"], 95)), 1),
                round(float(np.mean(f["pos"])), 1),
                round(float(np.mean(f["neg"])), 1))
        filas.append(fila)
        print("  " + "  ".join(f"{v:<5}" for v in fila))
    _escribir(os.path.join("data", "cap20_sobremuestreo.dat"),
              "factor de sobremuestreo (candidatos recorridos / k) para reunir "
              "%d validos: esperanza 1/s y cuantil 95 teoricos bajo "
              "independencia, media y percentil 95 medidos con filtro "
              "independiente, y medias con correlacion positiva y negativa "
              "(n=%d)" % (k, n),
              "selectividad  inversa  teo95  medio  p95  pos  neg", filas)


# ---------------------------------------------------------------------------
# rancidez: un estado fijo del indice
# ---------------------------------------------------------------------------

def simular_frescura(n: int = 20_000, dim: int = 64, k: int = 10,
                     nq: int = 300) -> None:
    """para cada fraccion, una mascara de vectores rancios y sus versiones
    viejas se generan una vez (el estado del indice); todas las consultas se
    evaluan contra ese estado. la version vieja es una perturbacion abstracta
    del vector actual, no un documento ni un codificador."""
    rng = np.random.default_rng(SEMILLA + 2)
    actual = _norm(rng.standard_normal((n, dim)))
    qs = _norm(rng.standard_normal((nq, dim)))
    oros = [set(_top(actual @ q, k).tolist()) for q in qs]
    filas = []
    print("\nrancidez: recall@%d segun la fraccion de vectores obsoletos" % k)
    print("  fraccion  recall  minimo")
    for frac in (0.0, 0.05, 0.1, 0.2, 0.4):
        rancio = rng.random(n) < frac
        indice = actual.copy()
        indice[rancio] = _norm(actual[rancio] + 1.2 * rng.standard_normal(
            (int(rancio.sum()), dim)))
        recs = [len(set(_top(indice @ q, k).tolist()) & o) / k
                for q, o in zip(qs, oros)]
        filas.append((frac, round(float(np.mean(recs)), 4),
                      round(float(np.min(recs)), 2)))
        print(f"  {frac:<8}  {filas[-1][1]:.4f}  {filas[-1][2]:.2f}")
    _escribir(os.path.join("data", "cap20_frescura.dat"),
              "recall@%d medio y minimo por consulta de un indice con una "
              "fraccion fija de vectores obsoletos, frente al top-%d del dato "
              "actual (n=%d)" % (k, k, n),
              "fraccion_rancia  recall  minimo", filas)


# ---------------------------------------------------------------------------
# mantenimiento de un indice particionado bajo deriva
# ---------------------------------------------------------------------------

def _kmeans_esferico(x: np.ndarray, nparts: int, semilla: int,
                     iters: int = 6) -> np.ndarray:
    if not 1 <= nparts <= len(x):
        raise ValueError("k-means necesita 1 <= nparts <= numero de puntos")
    rng = np.random.default_rng(semilla)
    cent = x[rng.choice(len(x), nparts, replace=False)].copy()
    for _ in range(iters):
        asign = np.argmax(x @ cent.T, axis=1)
        for c in range(nparts):
            m = asign == c
            if m.any():
                cent[c] = x[m].mean(0)
        cent = _norm(cent)
    return cent


def _flujo(semilla: int, n0: int, dim: int, altas: int, nq: int,
           cada: int) -> tuple:
    """datos comunes a todas las cadencias: base inicial, altas que derivan
    desde e1 hacia e2 y una bateria de consultas por punto de control (la mitad
    del tema viejo, la mitad cerca del centro actual de la deriva)."""
    rng = np.random.default_rng(semilla)
    e1, e2 = np.eye(dim)[0], np.eye(dim)[1]
    datos = np.empty((n0 + altas, dim))
    datos[:n0] = _norm(rng.standard_normal((n0, dim)))
    for j in range(altas):
        centro = _norm(e1 + (j / altas) * 2.0 * e2)
        datos[n0 + j] = _norm(centro + 0.35 * rng.standard_normal(dim))
    controles = {}
    for c in range(cada, altas + 1, cada):
        centro = _norm(e1 + (c / altas) * 2.0 * e2)
        viejas = _norm(rng.standard_normal((nq // 2, dim)))
        nuevas = _norm(centro + 0.35
                       * rng.standard_normal((nq - nq // 2, dim)))
        controles[c] = np.vstack([viejas, nuevas])
    return datos, controles


def _recall_ivf(datos, n, cent, asign, qs, k, nprobe) -> float:
    rec = 0.0
    for q in qs:
        oro = set(_top(datos[:n] @ q, k).tolist())
        celdas = _top(cent @ q, nprobe)
        cand = np.flatnonzero(np.isin(asign[:n], celdas))
        if len(cand):
            top = cand[_top(datos[cand] @ q, k)]
            rec += len(set(top.tolist()) & oro) / k
    return rec / len(qs)


def simular_mantenimiento(n0: int = 4000, dim: int = 64, k: int = 10,
                          altas: int = 6000, nq: int = 200, nparts: int = 32,
                          nprobe: int = 8, cada: int = 1000,
                          semillas: tuple = (20, 21, 22)) -> None:
    filas = []
    print("\nmantenimiento: recall@%d a lo largo del flujo segun la cadencia"
          % k)
    print("  cadencia  final   media   peor    coste_kvec  "
          "(media de %d semillas)" % len(semillas))
    for cad in (500, 1000, 2000, 3000, 0):          # 0: sin reconstruir
        res = []
        for s in semillas:
            datos, controles = _flujo(s, n0, dim, altas, nq, cada)
            n = n0
            cent = _kmeans_esferico(datos[:n], nparts, s)
            asign = np.empty(n0 + altas, dtype=np.int64)
            asign[:n] = np.argmax(datos[:n] @ cent.T, axis=1)
            coste, rebuilds, curva = 0, 0, []
            for j in range(1, altas + 1):
                asign[n] = int(np.argmax(cent @ datos[n]))
                n += 1
                if cad and j % cad == 0:             # reconstruir
                    rebuilds += 1
                    cent = _kmeans_esferico(datos[:n], nparts, s + rebuilds)
                    asign[:n] = np.argmax(datos[:n] @ cent.T, axis=1)
                    coste += n
                if j % cada == 0:
                    curva.append(_recall_ivf(datos, n, cent, asign,
                                             controles[j], k, nprobe))
            res.append((curva[-1], float(np.mean(curva)), min(curva), coste))
        r = np.array(res)
        m, lo, hi = r.mean(0), r.min(0), r.max(0)
        filas.append((cad, round(m[0], 4), round(lo[0], 4), round(hi[0], 4),
                      round(m[1], 4), round(m[2], 4), int(m[3])))
        print(f"  {cad if cad else 'nunca':<8}  {m[0]:.4f}  {m[1]:.4f}  "
              f"{m[2]:.4f}  {m[3]/1000:.0f}")
    nota = ("recall@%d de un indice particionado (nlist %d, nprobe %d) "
            "bajo %d altas que derivan, segun la cadencia de reconstruccion "
            "(0 = nunca): final (media, minimo y maximo de %d semillas), "
            "media y peor de los puntos de control, y vectores procesados "
            "por las reconstrucciones"
            % (k, nparts, nprobe, altas, len(semillas)))
    _escribir(os.path.join("data", "cap20_mantenimiento.dat"), nota,
              "cadencia  final  final_min  final_max  media  peor  coste",
              filas)


# ---------------------------------------------------------------------------
# demostracion: filtro, ranking y documento
# ---------------------------------------------------------------------------

def coleccion_productos(rng, n: int, dim: int) -> tuple:
    base = _norm(rng.standard_normal((n, dim)))
    categoria = rng.integers(0, 5, n)
    activo = rng.random(n) < 0.3
    documentos = [{"titulo": f"producto {i}",
                   "etiquetas": [f"e{int(t)}" for t in rng.integers(0, 50, 2)],
                   "ficha": {"talla": int(rng.integers(36, 47)),
                             "material": ["lona", "piel", "malla"][i % 3]}}
                  for i in range(n)]
    return base, categoria, activo, documentos


def consulta_hibrida(base, categoria, activo, documentos, q, cat: int,
                     k: int = 5) -> list[dict]:
    """una consulta logica: filtro estructurado, ranking vectorial en el
    conjunto elegible y proyeccion del documento de cada resultado."""
    elegibles = np.flatnonzero((categoria == cat) & activo)
    top = elegibles[_top(base[elegibles] @ q, k)]
    return [{"id": int(i), "sim": round(float(base[i] @ q), 4),
             "categoria": int(categoria[i]), "activo": bool(activo[i]),
             "titulo": documentos[i]["titulo"],
             "material": documentos[i]["ficha"]["material"]} for i in top]


def demostracion(n: int = 20_000, dim: int = 64, k: int = 5) -> None:
    rng = np.random.default_rng(SEMILLA + 3)
    base, categoria, activo, documentos = coleccion_productos(rng, n, dim)
    q = _norm(rng.standard_normal(dim))
    filas = consulta_hibrida(base, categoria, activo, documentos, q, cat=2, k=k)
    mask = (categoria == 2) & activo
    print("\ndemostracion: filtro estructurado + ranking vectorial + documento")
    print(f"  elegibles (categoria = 2 y activo): {int(mask.sum())} de {n}")
    print("  id      sim      titulo          material")
    for f in filas:
        print(f"  {f['id']:<6}  {f['sim']:+.4f}  {f['titulo']:<14}  "
              f"{f['material']}")
    # dos sistemas: el estructurado entrega los elegibles al vectorial
    elegibles = np.flatnonzero(mask)
    coordinado = elegibles[_top(base[elegibles] @ q, k)]
    ingenuo = postfiltro_ingenuo(base @ q, mask, k)
    print(f"  dos sistemas, conjunto elegible comunicado: "
          f"{len(coordinado)} de {k}, iguales: "
          f"{[f['id'] for f in filas] == coordinado.tolist()}")
    print(f"  dos sistemas, top-{k} global filtrado despues: "
          f"{len(ingenuo)} de {k}")


def main() -> None:
    anunciar()
    simular_planes()
    simular_sobremuestreo()
    simular_frescura()
    simular_mantenimiento()
    demostracion()


if __name__ == "__main__":
    main()
