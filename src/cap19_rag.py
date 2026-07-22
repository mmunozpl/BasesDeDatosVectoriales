"""capitulo 19 — recuperacion aumentada (RAG) y memoria de agentes.

estudia, en numpy, el patron recuperar-generar y sus extensiones, midiendo con
datos sinteticos cinco cosas que fijan el criterio de diseno de un RAG:

  1. troceado (chunking): el exito de recuperar la respuesta completa segun el
     tamano del trozo dibuja una U invertida ---trozos pequenos parten la
     respuesta, grandes la diluyen--- frente al troceado semantico, que la
     respeta.
  2. extremo a extremo: el recall del recuperador no es el exito de la tarea;
     responder exige TODOS los fragmentos de la respuesta, no uno cualquiera.
  3. multisalto (GraphRAG): la similitud vectorial pura falla en preguntas que
     encadenan relaciones; recorrer un grafo de conocimiento las resuelve.
  4. memoria de agentes: al crecer la memoria, recuperar el recuerdo relevante
     se complica; combinar relevancia y recencia ayuda en lo reciente.
  5. fusion recencia-relevancia: el peso entre ambas senales segun el tipo de
     consulta (atemporal frente a reciente).

los vectores son sinteticos (temas mas ruido), en lugar de un codificador real,
y el "generador" se simula por la presencia de la respuesta en el contexto. es
Python puro con numpy, CPU. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
from collections import deque
from typing import List, Tuple

import numpy as np

SEMILLA = 19


def anunciar() -> None:
    print("=" * 64)
    print("cap. 19 — recuperacion aumentada (RAG) y memoria de agentes")
    print("recursos: python + numpy · cpu. sin servicio, gpu ni torch.")
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


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# 1. troceado: la U invertida del tamano de trozo
# ---------------------------------------------------------------------------

def simular_chunking(n_docs: int = 400, L: int = 24, dim: int = 64,
                     s: int = 3, trials: int = 5) -> None:
    """exito de recuperar la respuesta COMPLETA segun el tamano del trozo. la
    respuesta ocupa s frases consecutivas; trozos pequenos la parten, grandes la
    diluyen con relleno. el troceado semantico (alineado a la respuesta) la
    respeta siempre."""
    rng = np.random.default_rng(SEMILLA)
    print("\ntroceado: exito (respuesta completa recuperada) segun tamano")
    print("  tam   fijo    semantico")
    filas = []
    for c in (1, 2, 3, 4, 6, 8, 12, 24):
        exito_fijo = exito_sem = 0
        total = 0
        for _ in range(trials):
            # cada doc: tema + una respuesta (subtema distinto) en s frases
            temas = rng.standard_normal((n_docs, dim))
            ini = rng.integers(0, L - s, n_docs)        # inicio de la respuesta
            resp = rng.standard_normal((n_docs, dim)) * 1.5  # subtema respuesta
            # frases de cada doc
            frases = (temas[:, None, :] + 0.4 * rng.standard_normal(
                (n_docs, L, dim)))
            for d in range(n_docs):
                frases[d, ini[d]:ini[d] + s] += resp[d]   # marca la respuesta
            q = _norm(resp + 0.3 * rng.standard_normal((n_docs, dim)))
            # --- troceado FIJO de tamano c ---
            ef = _eval_chunks_fijo(frases, ini, s, c, q, n_docs, L)
            # --- troceado SEMANTICO: un trozo = la respuesta ---
            es = _eval_chunks_sem(frases, ini, s, q, n_docs, L)
            exito_fijo += ef
            exito_sem += es
            total += n_docs
        filas.append((c, round(exito_fijo / total, 4),
                      round(exito_sem / total, 4)))
        print(f"  {c:<4}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap19_chunking.dat"),
              "exito de recuperar la respuesta completa (s=%d frases) segun el "
              "tamano del trozo: troceado fijo vs semantico" % s,
              "tam  fijo  semantico", filas)


def _eval_chunks_fijo(frases, ini, s, c, q, n_docs, L) -> int:
    """trozos fijos de c frases; exito si el top-1 contiene toda la respuesta."""
    vecs, meta = [], []
    for d in range(n_docs):
        for inicio in range(0, L, c):
            fin = min(inicio + c, L)
            vecs.append(frases[d, inicio:fin].mean(0))
            # cuenta cuantas frases de la respuesta caen en este trozo
            cubre = len(set(range(inicio, fin)) & set(range(ini[d], ini[d] + s)))
            meta.append((d, cubre))
    vecs = _norm(np.array(vecs))
    sims = q @ vecs.T
    top = sims.argmax(1)
    exito = 0
    for d in range(n_docs):
        dd, cubre = meta[top[d]]
        if dd == d and cubre == s:           # del doc correcto y respuesta entera
            exito += 1
    return exito


def _eval_chunks_sem(frases, ini, s, q, n_docs, L) -> int:
    """troceado semantico: un trozo es exactamente la respuesta (alineado)."""
    vecs, meta = [], []
    for d in range(n_docs):
        # trozo de la respuesta
        vecs.append(frases[d, ini[d]:ini[d] + s].mean(0))
        meta.append((d, s))
        # resto del doc en otro trozo (el contexto)
        resto = np.delete(np.arange(L), np.arange(ini[d], ini[d] + s))
        vecs.append(frases[d, resto].mean(0))
        meta.append((d, 0))
    vecs = _norm(np.array(vecs))
    sims = q @ vecs.T
    top = sims.argmax(1)
    return sum(1 for d in range(n_docs)
               if meta[top[d]][0] == d and meta[top[d]][1] == s)


# ---------------------------------------------------------------------------
# 2. extremo a extremo: recall del recuperador != exito de la tarea
# ---------------------------------------------------------------------------

def simular_extremo_a_extremo(n_docs: int = 2000, dim: int = 64,
                              piezas: int = 3, nq: int = 300) -> None:
    """la respuesta necesita TODAS sus piezas (p. ej. 3 fragmentos); el recall
    del recuperador cuenta cuantas trae, pero la tarea solo triunfa si las trae
    TODAS. el exito de la tarea va por debajo del recall, y la brecha crece con
    el numero de piezas."""
    rng = np.random.default_rng(SEMILLA)
    base = _norm(rng.standard_normal((n_docs, dim)))
    filas = []
    print("\nextremo a extremo: recall del recuperador vs exito de la tarea")
    print("  k     recall  exito_tarea")
    for k in (3, 5, 10, 20, 50):
        recall_tot = exito_tot = 0.0
        for _ in range(nq):
            # la respuesta son `piezas` fragmentos relacionados (cercanos)
            centro = _norm(rng.standard_normal(dim))
            rel = rng.choice(n_docs, piezas, replace=False)
            base[rel] = _norm(centro + 0.2 * rng.standard_normal((piezas, dim)))
            q = _norm(centro + 0.2 * rng.standard_normal(dim))
            top = set(np.argsort(-(base @ q))[:k].tolist())
            traidas = len(top & set(rel.tolist()))
            recall_tot += traidas / piezas
            exito_tot += 1.0 if traidas == piezas else 0.0   # tarea: TODAS
        filas.append((k, round(recall_tot / nq, 4), round(exito_tot / nq, 4)))
        print(f"  {k:<4}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap19_extremo.dat"),
              "recall del recuperador vs exito de la tarea (que exige las %d "
              "piezas de la respuesta) segun k" % piezas,
              "k  recall  exito", filas)


# ---------------------------------------------------------------------------
# 3. multisalto: similitud vectorial vs grafo (GraphRAG)
# ---------------------------------------------------------------------------

def simular_multisalto(n: int = 3000, dim: int = 64, nq: int = 300) -> None:
    """preguntas multisalto: la respuesta Z esta a varios saltos de la entidad
    X de la pregunta, y su vector NO se parece al de la pregunta. la busqueda
    vectorial falla; recorrer el grafo de relaciones (GraphRAG) la encuentra."""
    rng = np.random.default_rng(SEMILLA)
    vecs = _norm(rng.standard_normal((n, dim)))
    # grafo de relaciones: cada entidad enlaza con unas pocas al azar
    grafo: List[List[int]] = [list(rng.choice(n, 4, replace=False))
                              for _ in range(n)]
    filas = []
    print("\nmultisalto: exito de la busqueda vectorial vs grafo (GraphRAG)")
    print("  saltos  vectorial  grafo")
    for saltos in (1, 2, 3):
        ok_vec = ok_grafo = 0
        for _ in range(nq):
            x = int(rng.integers(0, n))            # entidad de la pregunta
            z = x                                   # respuesta a `saltos` saltos
            for _ in range(saltos):
                z = int(rng.choice(grafo[z]))
            q = vecs[x]                             # la pregunta se parece a X
            # vectorial: top-10 por similitud a la pregunta
            top = set(np.argsort(-(vecs @ q))[:10].tolist())
            ok_vec += 1 if z in top else 0
            # grafo: BFS de profundidad `saltos` desde X
            alcanzables = _bfs(grafo, x, saltos)
            ok_grafo += 1 if z in alcanzables else 0
        filas.append((saltos, round(ok_vec / nq, 4), round(ok_grafo / nq, 4)))
        print(f"  {saltos:<6}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap19_multisalto.dat"),
              "exito@10 de la busqueda vectorial vs el recorrido del grafo "
              "(GraphRAG) en preguntas de 1, 2 y 3 saltos (n=%d)" % n,
              "saltos  vectorial  grafo", filas)


def _bfs(grafo: List[List[int]], inicio: int, prof: int) -> set:
    """conjunto de nodos alcanzables a exactamente <= prof saltos desde inicio."""
    visto = {inicio}
    frontera = deque([(inicio, 0)])
    alcanzables = set()
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


# ---------------------------------------------------------------------------
# 4. memoria de agentes: recall al crecer la memoria
# ---------------------------------------------------------------------------

def simular_memoria(dim: int = 64, nq: int = 300) -> None:
    """al crecer la memoria del agente, el recuerdo relevante compite con mas
    distractores y el recall@5 baja: la memoria necesita gestion (resumen,
    olvido), no solo acumular."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nmemoria de agentes: recall@5 del recuerdo relevante segun tamano")
    print("  memoria  recall@5")
    for n in (100, 500, 2000, 10000, 50000):
        rec = 0
        for _ in range(nq):
            mem = _norm(rng.standard_normal((n, dim)))
            objetivo = int(rng.integers(0, n))
            q = _norm(mem[objetivo] + 0.5 * rng.standard_normal(dim))
            top = np.argsort(-(mem @ q))[:5]
            rec += 1 if objetivo in top else 0
        filas.append((n, round(rec / nq, 4)))
        print(f"  {n:<7}  {filas[-1][1]:.4f}")
    _escribir(os.path.join("data", "cap19_memoria.dat"),
              "recall@5 del recuerdo relevante en la memoria de un agente segun "
              "el numero de recuerdos almacenados",
              "memoria  recall", filas)


def simular_recencia(n: int = 5000, dim: int = 64, nq: int = 300) -> None:
    """fusion de relevancia y recencia en la memoria: para consultas que piden
    lo RECIENTE, combinar la similitud con un peso por recencia mejora; para las
    atemporales, lo empeora. el peso optimo depende del tipo de consulta."""
    rng = np.random.default_rng(SEMILLA)
    print("\nrecencia: recall@5 segun el peso de recencia (consultas recientes)")
    print("  alpha  reciente  atemporal")
    filas = []
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        rec_r = rec_a = 0
        for _ in range(nq):
            mem = _norm(rng.standard_normal((n, dim)))
            edad = np.arange(n)[::-1] / n            # 0 = mas reciente
            recencia = 1.0 - edad                    # 1 = mas reciente
            # consulta RECIENTE: el objetivo esta entre los ultimos recuerdos
            obj_r = int(rng.integers(n - n // 20, n))
            qr = _norm(mem[obj_r] + 0.6 * rng.standard_normal(dim))
            sim = mem @ qr
            score = (1 - alpha) * _n01(sim) + alpha * recencia
            rec_r += 1 if obj_r in np.argsort(-score)[:5] else 0
            # consulta ATEMPORAL: el objetivo esta en cualquier punto
            obj_a = int(rng.integers(0, n))
            qa = _norm(mem[obj_a] + 0.6 * rng.standard_normal(dim))
            sim = mem @ qa
            score = (1 - alpha) * _n01(sim) + alpha * recencia
            rec_a += 1 if obj_a in np.argsort(-score)[:5] else 0
        filas.append((alpha, round(rec_r / nq, 4), round(rec_a / nq, 4)))
        print(f"  {alpha:<5}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap19_recencia.dat"),
              "recall@5 segun el peso de recencia alpha, para consultas "
              "recientes vs atemporales (memoria de %d)" % n,
              "alpha  reciente  atemporal", filas)


def _n01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)


# ---------------------------------------------------------------------------
# demostracion: una consulta RAG y el contexto recuperado
# ---------------------------------------------------------------------------

def demostracion(n: int = 2000, dim: int = 64, piezas: int = 3) -> None:
    """una consulta RAG: recuperar k fragmentos y ver cuantas piezas de la
    respuesta traen (la tarea triunfa solo si las trae todas)."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _norm(rng.standard_normal((n, dim)))
    centro = _norm(rng.standard_normal(dim))
    rel = rng.choice(n, piezas, replace=False)
    base[rel] = _norm(centro + 0.2 * rng.standard_normal((piezas, dim)))
    q = _norm(centro + 0.2 * rng.standard_normal(dim))
    print(f"\ndemostracion: consulta RAG (respuesta en {piezas} fragmentos)")
    print("  k    piezas_recuperadas  tarea")
    for k in (3, 5, 10, 20):
        top = set(np.argsort(-(base @ q))[:k].tolist())
        traidas = len(top & set(rel.tolist()))
        ok = "exito" if traidas == piezas else "incompleta"
        print(f"  {k:<4} {traidas}/{piezas}                 {ok}")


def main() -> None:
    anunciar()
    simular_chunking()
    simular_extremo_a_extremo()
    simular_multisalto()
    simular_memoria()
    simular_recencia()
    demostracion()


if __name__ == "__main__":
    main()
