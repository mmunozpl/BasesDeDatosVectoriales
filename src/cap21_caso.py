"""capitulo 21 — caso de estudio integral: identificacion de aves por imagen.

cierra el libro articulando todo lo anterior sobre una pieza unica: un sistema
de recuperacion de observaciones de aves donde los metadatos relacionales
FILTRAN (habitat, estacion), los embeddings VECTORIALES ordenan por similitud,
y el documento (imagen + ficha de observacion) se DEVUELVE. demuestra la tesis
del libro: el filtrado estructurado acota, la similitud ordena, y ninguno basta
por separado.

se mide en numpy, con embeddings sinteticos (cumulos por especie mas ruido, en
lugar de un ViT real, para no exigir GPU ni torch), cuatro cosas:

  1. filtro + similitud: la precision de combinar filtro y similitud frente a
     usar solo uno de los dos. ninguno basta solo; juntos aciertan.
  2. interaccion tardia (estilo ColPali): cuando la senal discriminante vive en
     una REGION pequena de la imagen (un rasgo de campo: la barra alar, el anillo
     ocular), el vector unico global la diluye y la interaccion tardia (max-sim
     sobre parches) la recupera.
  3. prefiltrado vs postfiltrado: un filtro por un habitat raro hunde el
     postfiltrado, como en el capitulo 20.
  4. desbalance de clases: las especies raras se recuperan peor que las
     frecuentes; una limitacion honesta de relevancia para la biodiversidad.

los embeddings son sinteticos; la magnitud es ilustrativa, la direccion robusta.
Python puro con numpy, CPU. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

SEMILLA = 21

# especies del corpus sintetico (taxonomia de aves, estilo CUB-200/iNaturalist)
# y su frecuencia relativa: el corpus esta desbalanceado, como el muestreo de
# campo real (unas pocas especies comunes, muchas raras).
ESPECIES = ["gorrion_comun", "mirlo_comun", "petirrojo", "abubilla", "alimoche"]
FRECUENCIA = np.array([0.55, 0.12, 0.20, 0.08, 0.05])      # suma 1.0
HABITATS = ["bosque", "humedal", "montaña", "urbano", "matorral", "costa"]


def anunciar() -> None:
    print("=" * 64)
    print("cap. 21 — caso de estudio: identificacion de aves por imagen")
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


def _corpus(n: int, dim: int, rng) -> Dict[str, np.ndarray]:
    """genera un corpus sintetico de observaciones de aves: cada observacion
    tiene un embedding (cumulo por especie mas ruido), una especie y un habitat."""
    esp = rng.choice(len(ESPECIES), n, p=FRECUENCIA)          # clase (especie)
    hab = rng.integers(0, len(HABITATS), n)                   # habitat
    centros = _norm(rng.standard_normal((len(ESPECIES), dim)))
    emb = _norm(centros[esp] + 0.9 * rng.standard_normal((n, dim)))
    return {"emb": emb, "esp": esp, "hab": hab}


# ---------------------------------------------------------------------------
# 1. la tesis del libro: filtro + similitud frente a solo uno de los dos
# ---------------------------------------------------------------------------

def simular_filtro_mas_similitud(n: int = 20000, dim: int = 64, k: int = 10,
                                 nq: int = 400) -> None:
    """precision@k de recuperar el caso util: una observacion de la MISMA
    especie Y el MISMO habitat que la consulta. tres estrategias: solo similitud
    (acierta la especie pero ignora el habitat), solo filtro (acierta el habitat
    pero no la especie) y filtro + similitud. ninguno basta solo: la similitud
    trae parecidos de cualquier habitat, el filtro trae del habitat pero sin
    parecido; juntos aciertan ambas condiciones."""
    rng = np.random.default_rng(SEMILLA)
    c = _corpus(n, dim, rng)
    emb, esp, hab = c["emb"], c["esp"], c["hab"]
    print("\nfiltro + similitud: precision@%d (misma especie Y habitat)" % k)
    print("  estrategia        precision")
    p_sim = p_fil = p_amb = 0.0
    for _ in range(nq):
        qi = int(rng.integers(0, n))
        q, qd, qs = emb[qi], esp[qi], hab[qi]
        # relevante = misma especie Y mismo habitat (el caso util)
        rel = (esp == qd) & (hab == qs)
        sims = emb @ q
        # solo similitud: top-k por parecido, ignora el habitat
        top = np.argsort(-sims)[:k]
        p_sim += rel[top].mean()
        # solo filtro: observaciones del mismo habitat, sin ordenar por parecido
        mismo = np.where(hab == qs)[0]
        muestra = rng.choice(mismo, min(k, len(mismo)), replace=False)
        p_fil += rel[muestra].mean()
        # filtro + similitud: del mismo habitat, ordenadas por parecido
        orden = mismo[np.argsort(-sims[mismo])[:k]]
        p_amb += rel[orden].mean()
    filas = [("solo_similitud", round(p_sim / nq, 4)),
             ("solo_filtro", round(p_fil / nq, 4)),
             ("filtro+similitud", round(p_amb / nq, 4))]
    for e, p in filas:
        print(f"  {e:<16}  {p:.4f}")
    _escribir(os.path.join("data", "cap21_filtro_similitud.dat"),
              "precision@%d de recuperar la misma especie Y habitat con solo "
              "similitud, solo filtro (mismo habitat) y filtro+similitud (n=%d)"
              % (k, n),
              "estrategia  precision", filas)


# ---------------------------------------------------------------------------
# 2. interaccion tardia (estilo ColPali) frente al vector unico
# ---------------------------------------------------------------------------

def simular_interaccion_tardia(n: int = 4000, dim: int = 32, parches: int = 9,
                               k: int = 10, nq: int = 300) -> None:
    """cuando la senal discriminante vive en UNA region pequena de la imagen
    (un rasgo de campo del ave), el vector unico (media de los parches) la
    diluye; la interaccion tardia (max-sim sobre parches, estilo ColPali) la
    conserva. se mide recall@k del caso relevante segun la fuerza de la senal
    local."""
    rng = np.random.default_rng(SEMILLA)
    print("\ninteraccion tardia vs vector unico: recall@%d segun senal local"
          % k)
    print("  senal_local  vector_unico  interaccion_tardia")
    filas = []
    for senal in (0.5, 1.0, 1.5, 2.0, 3.0):
        rec_u = rec_t = 0.0
        for _ in range(nq):
            # cada observacion: `parches` vectores; el fondo es ruido comun
            fondo = _norm(rng.standard_normal((n, parches, dim)))
            # marca: un parche concreto lleva el rasgo de campo discriminante
            patron = _norm(rng.standard_normal(dim))
            objetivo = int(rng.integers(0, n))
            jp = int(rng.integers(0, parches))
            fondo[objetivo, jp] = _norm(fondo[objetivo, jp] + senal * patron)
            # consulta: el rasgo discriminante (un parche)
            q = _norm(patron + 0.15 * rng.standard_normal(dim))
            # vector unico: media de parches -> diluye la senal local
            vu = _norm(fondo.mean(axis=1))
            top_u = np.argsort(-(vu @ q))[:k]
            rec_u += 1.0 if objetivo in top_u else 0.0
            # interaccion tardia: max-sim del parche mas parecido por observacion
            maxsim = (fondo @ q).max(axis=1)        # mejor parche por observacion
            top_t = np.argsort(-maxsim)[:k]
            rec_t += 1.0 if objetivo in top_t else 0.0
        filas.append((senal, round(rec_u / nq, 4), round(rec_t / nq, 4)))
        print(f"  {senal:<11}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap21_interaccion_tardia.dat"),
              "recall@%d del caso relevante segun la fuerza de la senal local: "
              "vector unico (media de parches) vs interaccion tardia (max-sim "
              "sobre parches, estilo ColPali) (n=%d, parches=%d)"
              % (k, n, parches),
              "senal_local  vector_unico  interaccion_tardia", filas)


# ---------------------------------------------------------------------------
# 3. prefiltrado vs postfiltrado por habitat
# ---------------------------------------------------------------------------

def simular_prefiltrado(n: int = 20000, dim: int = 64, k: int = 10,
                        nq: int = 300) -> None:
    """consulta: observaciones del mismo habitat, ordenadas por parecido.
    resuelta con prefiltrado (buscar solo en el habitat) frente a postfiltrado
    (k mejores globales, luego filtrar por habitat). al ser raro el habitat, el
    postfiltrado se hunde: es el fenomeno del capitulo 20 sobre el terreno."""
    rng = np.random.default_rng(SEMILLA)
    c = _corpus(n, dim, rng)
    emb = c["emb"]
    print("\nprefiltrado vs postfiltrado por habitat: recall@%d segun rareza" % k)
    print("  prevalencia_habitat  prefiltrado  postfiltrado")
    filas = []
    for prev in (0.3, 0.15, 0.07, 0.03, 0.01):
        # un habitat con prevalencia `prev` (p. ej. una region poco muestreada)
        raro = rng.random(n) < prev
        idx_raro = np.where(raro)[0]
        rec_pre = rec_post = 0.0
        for _ in range(nq):
            q = _norm(rng.standard_normal(dim))
            sims = emb @ q
            oro = idx_raro[np.argsort(-sims[idx_raro])[:k]]     # verdad
            pre = idx_raro[np.argsort(-sims[idx_raro])[:k]]     # prefiltrado
            rec_pre += len(set(pre) & set(oro)) / len(oro)
            top = np.argsort(-sims)[:k]                          # k globales
            post = [i for i in top if raro[i]]                   # postfiltrar
            rec_post += len(set(post) & set(oro)) / len(oro)
        filas.append((prev, round(rec_pre / nq, 4), round(rec_post / nq, 4)))
        print(f"  {prev:<19}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap21_prefiltrado.dat"),
              "recall@%d de la consulta (mismo habitat, por parecido) con "
              "prefiltrado vs postfiltrado segun la prevalencia del habitat (n=%d)"
              % (k, n),
              "prevalencia_habitat  prefiltrado  postfiltrado", filas)


# ---------------------------------------------------------------------------
# 4. desbalance de clases: las especies raras se recuperan peor
# ---------------------------------------------------------------------------

def simular_desbalance(n: int = 20000, dim: int = 64, k: int = 10,
                       nq: int = 400) -> None:
    """las especies raras tienen pocos ejemplos en el corpus, asi que para una
    consulta de una especie rara hay menos vecinos de la misma y la precision@k
    baja. una limitacion honesta: la recuperacion hereda el desbalance del
    corpus, y el caso de interes para la conservacion suele ser el raro."""
    rng = np.random.default_rng(SEMILLA)
    c = _corpus(n, dim, rng)
    emb, esp = c["emb"], c["esp"]
    print("\ndesbalance: precision@%d por especie (frecuente vs rara)" % k)
    print("  especie          frecuencia  precision")
    filas = []
    for d, nombre in enumerate(ESPECIES):
        idx = np.where(esp == d)[0]
        prec = 0.0
        casos = min(nq, len(idx))
        elegidos = rng.choice(idx, casos, replace=False)
        for qi in elegidos:
            sims = emb @ emb[qi]
            top = np.argsort(-sims)[:k]
            prec += (esp[top] == d).mean()
        filas.append((nombre, round(float(FRECUENCIA[d]), 3),
                      round(prec / casos, 4)))
        print(f"  {nombre:<15}  {filas[-1][1]:.3f}       {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap21_desbalance.dat"),
              "precision@%d por especie segun su frecuencia en el corpus: "
              "las especies raras se recuperan peor (n=%d)" % (k, n),
              "especie  frecuencia  precision", filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta de extremo a extremo
# ---------------------------------------------------------------------------

def demostracion(n: int = 20000, dim: int = 64, k: int = 5) -> None:
    """una consulta completa: dada una observacion, recuperar las mas parecidas
    DEL MISMO habitat (prefiltrado relacional + similitud vectorial), devolviendo
    su especie (documental). es la tesis del libro sobre una pieza unica."""
    rng = np.random.default_rng(SEMILLA + 1)
    c = _corpus(n, dim, rng)
    emb, esp, hab = c["emb"], c["esp"], c["hab"]
    qi = int(rng.integers(0, n))
    qs = int(hab[qi])
    mismo = np.where(hab == qs)[0]                    # prefiltro: mismo habitat
    orden = mismo[np.argsort(-(emb[mismo] @ emb[qi]))[:k + 1]]
    orden = [i for i in orden if i != qi][:k]         # excluir la consulta
    print("\ndemostracion: consulta (observacion en '%s')" % HABITATS[qs])
    print(f"  especie de la consulta: {ESPECIES[int(esp[qi])]}")
    print(f"  candidatos del mismo habitat: {len(mismo)} de {n}")
    print("  rango  similitud  especie          habitat")
    for r, i in enumerate(orden, start=1):
        print(f"  {r:<5}  {float(emb[i] @ emb[qi]):+.4f}    "
              f"{ESPECIES[int(esp[i])]:<15}  {HABITATS[int(hab[i])]}")


def main() -> None:
    anunciar()
    simular_filtro_mas_similitud()
    simular_interaccion_tardia()
    simular_prefiltrado()
    simular_desbalance()
    demostracion()


if __name__ == "__main__":
    main()
