"""capitulo 19: recuperacion aumentada (RAG) y memoria de agentes.

estudia, en numpy y con datos sinteticos de respuesta conocida, los mecanismos
que condicionan un RAG. no hay generador: donde un RAG real mediria la
respuesta, la practica mide si la evidencia necesaria llega al contexto.

  1. troceado: exito top-1 (la respuesta entera en el trozo recuperado) segun
     el tamano del trozo fijo, con la cobertura de los tres primeros trozos,
     frente a un troceador semantico real (cortes por percentil de distancia
     entre frases consecutivas) y una segmentacion oraculo que conoce la
     frontera de la respuesta (cota superior, no un algoritmo).
  2. el troceador semantico segun la intensidad del cambio de tema.
  3. cobertura de la evidencia: fraccion de piezas recuperadas frente a
     cobertura completa (todas las piezas en el top-k), con una bateria fija de
     consultas evaluada en todos los k.
  4. proximidad frente a conectividad: top-10 por similitud (sin la entidad de
     partida) frente a un recorrido explicito del grafo, con vectores
     independientes del grafo y con vectores correlacionados con sus
     comunidades. el recorrido alcanza la respuesta por construccion.
  5. memoria con distractores: hit@5 de un recuerdo segun cuantos recuerdos
     se han acumulado, por similitud pura y restringiendo a la sesion.
  6. relevancia y recencia sinteticas en la memoria, segun el tipo de consulta.

las cifras dependen solo de la semilla.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

SEMILLA = 19


def anunciar() -> None:
    print("=" * 64)
    print("cap. 19: recuperacion aumentada (RAG) y memoria de agentes")
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
    """indices de los k mayores, ordenados (seleccion parcial)."""
    k = min(k, len(sims))
    part = np.argpartition(-sims, k - 1)[:k]
    return part[np.argsort(-sims[part], kind="stable")]


# ---------------------------------------------------------------------------
# 1 y 2. troceado: fijo, semantico real y oraculo
# ---------------------------------------------------------------------------

def corpus_frases(rng, n_docs: int, L: int, dim: int, s: int,
                  intensidad: float = 1.5) -> tuple:
    """cada documento: L frases de su tema mas ruido; la respuesta ocupa s
    frases consecutivas desde ini, desplazadas hacia un subtema propio con la
    intensidad dada. la consulta apunta al subtema de la respuesta, de modo
    que su calidad no depende de la intensidad."""
    temas = rng.standard_normal((n_docs, dim))
    ini = rng.integers(0, L - s + 1, n_docs)
    resp = rng.standard_normal((n_docs, dim))
    frases = temas[:, None, :] + 0.4 * rng.standard_normal((n_docs, L, dim))
    for d in range(n_docs):
        frases[d, ini[d]:ini[d] + s] += intensidad * resp[d]
    q = _norm(resp + 0.3 * rng.standard_normal((n_docs, dim)))
    return frases, ini, q


def trozos_fijos(L: int, c: int) -> list[np.ndarray]:
    return [np.arange(i, min(i + c, L)) for i in range(0, L, c)]


def trocear_semantico(vecs: np.ndarray, percentil: float = 90) -> list:
    """corta donde la distancia coseno entre frases consecutivas supera el
    percentil dado de las distancias del propio documento. vecs: frases
    normalizadas. no conoce la posicion de la respuesta."""
    dist = 1.0 - (vecs[1:] * vecs[:-1]).sum(1)
    umbral = np.percentile(dist, percentil)
    cortes = np.flatnonzero(dist > umbral) + 1
    return np.split(np.arange(len(vecs)), cortes)


def trozos_oraculo(L: int, ini: int, s: int) -> list[np.ndarray]:
    """segmentacion oraculo: tres trozos contiguos, antes, la respuesta y
    despues. usa la frontera verdadera: es una cota, no un algoritmo."""
    partes = [np.arange(0, ini), np.arange(ini, ini + s), np.arange(ini + s, L)]
    return [p for p in partes if len(p)]


def _evaluar_trozos(frases, ini, s, q, segmentar) -> tuple[float, float]:
    """exito top-1: el trozo recuperado es del documento correcto y contiene
    las s frases de la respuesta. cobertura@3: las s frases estan en la union
    de los tres primeros trozos. el vector de un trozo es la media de sus
    frases (un supuesto del modelo, no de los codificadores reales)."""
    n_docs = len(frases)
    vecs, meta = [], []
    for d in range(n_docs):
        for t in segmentar(d):
            vecs.append(frases[d, t].mean(0))
            meta.append((d, t))
    vecs = _norm(np.array(vecs))
    top1 = cob3 = 0
    for d in range(n_docs):               # una consulta por documento
        idx = _top(vecs @ q[d], 3)
        respuesta = set(range(ini[d], ini[d] + s))
        dd, t = meta[idx[0]]
        top1 += dd == d and respuesta <= set(t.tolist())
        vistas = set()
        for i in idx:
            dd, t = meta[i]
            if dd == d:
                vistas |= set(t.tolist())
        cob3 += respuesta <= vistas
    return top1 / n_docs, cob3 / n_docs


def simular_troceado(n_docs: int = 400, L: int = 24, dim: int = 64,
                     s: int = 3, trials: int = 5) -> None:
    rng = np.random.default_rng(SEMILLA)
    datos = [corpus_frases(rng, n_docs, L, dim, s) for _ in range(trials)]
    sem = ora = 0.0
    for frases, ini, q in datos:
        fn = _norm(frases)
        sem += _evaluar_trozos(frases, ini, s, q,
                               lambda d: trocear_semantico(fn[d]))[0]
        ora += _evaluar_trozos(frases, ini, s, q,
                               lambda d: trozos_oraculo(L, ini[d], s))[0]
    sem, ora = sem / trials, ora / trials
    filas = []
    print("\ntroceado: exito top-1 y cobertura@3 segun el tamano del trozo fijo")
    print(f"  semantico real (percentil 90): {sem:.4f}   oraculo: {ora:.4f}")
    print("  tam   top1    cob@3")
    for c in (1, 2, 3, 4, 6, 8, 12, 24):
        t1 = c3 = 0.0
        for frases, ini, q in datos:
            a, b = _evaluar_trozos(frases, ini, s, q,
                                   lambda d: trozos_fijos(L, c))
            t1, c3 = t1 + a, c3 + b
        filas.append((c, round(t1 / trials, 4), round(c3 / trials, 4),
                       round(sem, 4), round(ora, 4)))
        print(f"  {c:<4}  {t1/trials:.4f}  {c3/trials:.4f}")
    _escribir(os.path.join("data", "cap19_chunking.dat"),
              "exito top-1 (respuesta de %d frases entera en el trozo "
              "recuperado) y cobertura con los tres primeros trozos, segun el "
              "tamano del trozo fijo; exito top-1 del troceador semantico real "
              "y de la segmentacion oraculo (n_docs=%d, L=%d)" % (s, n_docs, L),
              "tam  fijo  fijo3  semantico  oraculo", filas)


def simular_semantico(n_docs: int = 400, L: int = 24, dim: int = 64,
                      s: int = 3, trials: int = 5) -> None:
    """el troceador semantico real depende de que la respuesta sea un cambio
    de tema detectable: se barre la intensidad del desplazamiento."""
    rng = np.random.default_rng(SEMILLA + 3)
    filas = []
    print("\ntroceador semantico segun la intensidad del cambio de tema")
    print("  intens  semantico  fijo6   oraculo")
    for a in (0.25, 0.5, 0.75, 1.0, 1.5):
        r = np.zeros(3)
        for _ in range(trials):
            frases, ini, q = corpus_frases(rng, n_docs, L, dim, s, a)
            fn = _norm(frases)
            r += [_evaluar_trozos(frases, ini, s, q,
                                  lambda d: trocear_semantico(fn[d]))[0],
                  _evaluar_trozos(frases, ini, s, q,
                                  lambda d: trozos_fijos(L, 6))[0],
                  _evaluar_trozos(frases, ini, s, q,
                                  lambda d: trozos_oraculo(L, ini[d], s))[0]]
        r /= trials
        filas.append((a, round(r[0], 4), round(r[1], 4), round(r[2], 4)))
        print(f"  {a:<6}  {r[0]:.4f}     {r[1]:.4f}  {r[2]:.4f}")
    _escribir(os.path.join("data", "cap19_semantico.dat"),
              "exito top-1 del troceador semantico real, del fijo de 6 frases y "
              "del oraculo segun la intensidad del cambio de tema de la "
              "respuesta", "intensidad  semantico  fijo6  oraculo", filas)


# ---------------------------------------------------------------------------
# 3. cobertura de la evidencia: piezas recuperadas frente a todas
# ---------------------------------------------------------------------------

def _bateria_piezas(rng, base0: np.ndarray, piezas: int, nq: int) -> list:
    """consultas fijas: cada una con sus piezas de oro (filas de la base que
    se sustituyen solo para esa consulta) y su vector."""
    n, dim = base0.shape
    bat = []
    for _ in range(nq):
        centro = _norm(rng.standard_normal(dim))
        rel = rng.choice(n, piezas, replace=False)
        vrel = _norm(centro + 0.2 * rng.standard_normal((piezas, dim)))
        q = _norm(centro + 0.2 * rng.standard_normal(dim))
        bat.append((rel, vrel, q))
    return bat


def _puestos(base0: np.ndarray, rel, vrel, q) -> np.ndarray:
    """puesto (desde 1) de cada pieza: la base se restaura en cada consulta
    porque la copia se modifica solo dentro de esta funcion."""
    base = base0.copy()
    base[rel] = vrel
    sims = base @ q
    return np.array([int((sims > sims[r]).sum()) + 1 for r in rel])


def simular_cobertura(n_docs: int = 2000, dim: int = 64, nq: int = 300) -> None:
    rng = np.random.default_rng(SEMILLA)
    base0 = _norm(rng.standard_normal((n_docs, dim)))
    bat = _bateria_piezas(rng, base0, 3, nq)
    puestos = [_puestos(base0, *b) for b in bat]
    filas = []
    print("\ncobertura de la evidencia (3 piezas): misma bateria para todo k")
    print("  k     piezas  completa")
    for k in (3, 5, 10, 20, 50):
        frac = np.mean([(p <= k).mean() for p in puestos])
        comp = np.mean([(p <= k).all() for p in puestos])
        filas.append((k, round(frac, 4), round(comp, 4)))
        print(f"  {k:<4}  {frac:.4f}  {comp:.4f}")
    _escribir(os.path.join("data", "cap19_extremo.dat"),
              "fraccion media de las 3 piezas de evidencia en el top-k y "
              "cobertura completa (las 3), sobre la misma bateria de %d "
              "consultas para todos los k" % nq, "k  recall  exito", filas)
    filas = []
    print("  piezas  fraccion@10  completa@10")
    for piezas in (1, 2, 3, 5):
        bat = _bateria_piezas(rng, base0, piezas, nq)
        p10 = [(_puestos(base0, *b) <= 10) for b in bat]
        frac = float(np.mean([p.mean() for p in p10]))
        comp = float(np.mean([p.all() for p in p10]))
        filas.append((piezas, round(frac, 4), round(comp, 4)))
        print(f"  {piezas:<6}  {frac:.4f}       {comp:.4f}")
    _escribir(os.path.join("data", "cap19_piezas.dat"),
              "fraccion de piezas y cobertura completa en el top-10 segun el "
              "numero de piezas de la evidencia", "piezas  recall  exito", filas)


# ---------------------------------------------------------------------------
# 4. proximidad frente a conectividad
# ---------------------------------------------------------------------------

def grafo_comunidades(rng, n: int, n_com: int, grado: int = 4,
                      p_intra: float = 0.8) -> tuple[list, np.ndarray]:
    """cada entidad enlaza con `grado` entidades distintas de si misma; con
    probabilidad p_intra la vecina es de su comunidad."""
    com = np.arange(n) % n_com
    miembros = [np.flatnonzero(com == c) for c in range(n_com)]
    grafo = []
    for i in range(n):
        vec = set()
        while len(vec) < grado:
            if rng.random() < p_intra:
                j = int(rng.choice(miembros[com[i]]))
            else:
                j = int(rng.integers(0, n))
            if j != i:
                vec.add(j)
        grafo.append(sorted(vec))
    return grafo, com


def _bfs(grafo: list, inicio: int, prof: int) -> set:
    """nodos alcanzables desde inicio en 1..prof saltos."""
    visto, frontera, alcanzables = {inicio}, deque([(inicio, 0)]), set()
    while frontera:
        nodo, d = frontera.popleft()
        if d == prof:
            continue
        for v in grafo[nodo]:
            alcanzables.add(v)
            if v not in visto:
                visto.add(v)
                frontera.append((v, d + 1))
    return alcanzables


def simular_multisalto(n: int = 3000, dim: int = 64, n_com: int = 60,
                       nq: int = 300) -> None:
    """la respuesta z esta a 1, 2 o 3 saltos de x por un camino del grafo.
    la similitud busca entre las demas entidades (sin x); el recorrido
    alcanza z por construccion, y lo que se mide de el es cuantos candidatos
    devuelve."""
    rng = np.random.default_rng(SEMILLA)
    grafo, com = grafo_comunidades(rng, n, n_com)
    indep = _norm(rng.standard_normal((n, dim)))
    centros = rng.standard_normal((n_com, dim))
    homof = _norm(centros[com] + rng.standard_normal((n, dim)))
    caminos = []
    while len(caminos) < nq:
        x = int(rng.integers(0, n))
        z, cam = x, []
        for _ in range(3):
            z = int(rng.choice(grafo[z]))
            cam.append(z)
        if x not in cam:
            caminos.append((x, cam))
    filas = []
    print("\nproximidad frente a conectividad: exito@10 segun los saltos")
    print("  saltos  indep   homofilo  grafo   candidatos")
    for h in (1, 2, 3):
        ok_i = ok_h = ok_g = 0
        cand = []
        for x, cam in caminos:
            z = cam[h - 1]
            for vecs, nombre in ((indep, "i"), (homof, "h")):
                sims = vecs @ vecs[x]
                sims[x] = -np.inf                  # sin la entidad de partida
                acierto = z in set(_top(sims, 10).tolist())
                if nombre == "i":
                    ok_i += acierto
                else:
                    ok_h += acierto
            alc = _bfs(grafo, x, h)
            ok_g += z in alc
            cand.append(len(alc))
        filas.append((h, round(ok_i / nq, 4), round(ok_h / nq, 4),
                      round(ok_g / nq, 4), round(float(np.mean(cand)), 1)))
        print(f"  {h:<6}  {ok_i/nq:.4f}  {ok_h/nq:.4f}    {ok_g/nq:.4f}  "
              f"{np.mean(cand):.1f}")
    _escribir(os.path.join("data", "cap19_multisalto.dat"),
              "exito@10 por similitud (sin la entidad de partida) con vectores "
              "independientes del grafo y correlacionados con sus comunidades, "
              "y alcance del recorrido explicito con su numero medio de "
              "candidatos, segun los saltos (n=%d, %d comunidades)" % (n, n_com),
              "saltos  indep  homofilo  grafo  candidatos", filas)


# ---------------------------------------------------------------------------
# 5. memoria con distractores
# ---------------------------------------------------------------------------

def simular_memoria(n_max: int = 50_000, dim: int = 64, nq: int = 300,
                    sesion: int = 100) -> None:
    """una sola memoria que se acumula: con n recuerdos se consultan los n
    primeros. hit@5 del recuerdo objetivo por similitud pura y restringiendo
    la busqueda a su sesion (bloques de `sesion` recuerdos), y numero medio de
    distractores con similitud mayor que la del objetivo."""
    rng = np.random.default_rng(SEMILLA)
    mem = _norm(rng.standard_normal((n_max, dim)))
    ses = np.arange(n_max) // sesion
    filas = []
    print("\nmemoria: hit@5 segun los recuerdos acumulados")
    print("  memoria  hit@5   sesion  por_encima")
    for n in (100, 500, 2000, 10_000, 50_000):
        hit = hit_s = 0
        encima = []
        for _ in range(nq):
            t = int(rng.integers(0, n))
            q = _norm(mem[t] + 0.5 * rng.standard_normal(dim))
            sims = mem[:n] @ q
            mayores = int((sims > sims[t]).sum())
            encima.append(mayores)
            hit += mayores < 5
            en_ses = np.flatnonzero(ses[:n] == ses[t])
            hit_s += int((sims[en_ses] > sims[t]).sum()) < 5
        filas.append((n, round(hit / nq, 4), round(hit_s / nq, 4),
                      round(float(np.mean(encima)), 1)))
        print(f"  {n:<7}  {hit/nq:.4f}  {hit_s/nq:.4f}  {np.mean(encima):.1f}")
    _escribir(os.path.join("data", "cap19_memoria.dat"),
              "hit@5 del recuerdo objetivo segun los recuerdos acumulados, por "
              "similitud pura y dentro de su sesion de %d, y distractores medios "
              "con mayor similitud que el objetivo (dim=%d)" % (sesion, dim),
              "memoria  recall  sesion  encima", filas)


# ---------------------------------------------------------------------------
# 6. relevancia y recencia
# ---------------------------------------------------------------------------

def _n01(x: np.ndarray) -> np.ndarray:
    """min-max por consulta: basta para la practica, no calibra entre
    consultas."""
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def simular_recencia(n: int = 5000, dim: int = 64, nq: int = 300) -> None:
    """una memoria y dos baterias fijas de consultas: recientes (el objetivo
    esta en el ultimo 5 %) y atemporales (en cualquier punto). todos los pesos
    alpha se evaluan sobre las mismas consultas. la recencia es sintetica:
    la posicion normalizada, 1 para el ultimo recuerdo."""
    rng = np.random.default_rng(SEMILLA)
    mem = _norm(rng.standard_normal((n, dim)))
    recencia = np.arange(n) / (n - 1)
    def bateria(lo: int) -> list:
        objs = rng.integers(lo, n, nq)
        return [(int(o), _n01(mem @ _norm(mem[o] + 0.6 * rng.standard_normal(dim))))
                for o in objs]
    recientes, atemporales = bateria(n - n // 20), bateria(0)
    filas = []
    print("\nrecencia: hit@5 segun el peso de la recencia alpha")
    print("  alpha  reciente  atemporal")
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        r = [np.mean([o in set(_top((1 - alpha) * s + alpha * recencia, 5)
                               .tolist()) for o, s in bat])
             for bat in (recientes, atemporales)]
        filas.append((alpha, round(r[0], 4), round(r[1], 4)))
        print(f"  {alpha:<5}  {r[0]:.4f}    {r[1]:.4f}")
    _escribir(os.path.join("data", "cap19_recencia.dat"),
              "hit@5 segun el peso de recencia alpha (recencia sintetica por "
              "posicion), para las mismas consultas recientes y atemporales "
              "(memoria de %d)" % n, "alpha  reciente  atemporal", filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta y el puesto de cada pieza de evidencia
# ---------------------------------------------------------------------------

def demostracion(n: int = 2000, dim: int = 64, piezas: int = 3) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    base0 = _norm(rng.standard_normal((n, dim)))
    rel, vrel, q = _bateria_piezas(rng, base0, piezas, 1)[0]
    puestos = _puestos(base0, rel, vrel, q)
    print(f"\ndemostracion: una consulta con {piezas} piezas de evidencia")
    print(f"  puestos de las piezas: {sorted(puestos.tolist())}")
    print("  k    piezas  cobertura")
    for k in (3, 5, 10, 20):
        traidas = int((puestos <= k).sum())
        estado = "completa" if traidas == piezas else "incompleta"
        print(f"  {k:<4} {traidas}/{piezas}     {estado}")


def main() -> None:
    anunciar()
    simular_troceado()
    simular_semantico()
    simular_cobertura()
    simular_multisalto()
    simular_memoria()
    simular_recencia()
    demostracion()


if __name__ == "__main__":
    main()
