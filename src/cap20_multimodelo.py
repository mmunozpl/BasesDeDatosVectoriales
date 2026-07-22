"""capitulo 20 — el futuro hibrido: la convergencia en un motor multimodelo.

capitulo prospectivo. en lugar de especular con nombres de producto, mide en
numpy los fenomenos que explican POR QUE los estratos del libro ---relacional,
documental y vectorial--- tienden a converger en un unico motor, y que costes
trae esa convergencia. cinco medidas, todas reproducibles en CPU:

  1. prefiltrado frente a postfiltrado: una consulta hibrida (filtro + vector)
     resuelta dentro del motor (prefiltrado) conserva el recall al volverse
     selectivo el filtro; resuelta en dos sistemas separados (postfiltrar lo
     que devuelve el vector) lo pierde. es el argumento a favor de integrar.
  2. el impuesto del sobremuestreo: para sobrevivir al postfiltrado, el sistema
     separado debe pedir k/selectividad candidatos; se mide ese multiplicador.
  3. frescura (consistencia vector-dato): cuando el dato cambia y su vector no
     se reindexa, el vector queda obsoleto; el recall cae con la fraccion de
     vectores rancios. es el coste de gobernanza de la convergencia.
  4. mantenimiento del indice: bajo un flujo de altas, reindexar tarde abarata
     pero degrada el recall; reindexar pronto lo conserva pero cuesta. se mide
     el compromiso (recall vs coste) segun la cadencia.
  5. convergencia inversa: un motor relacional que anade vectores y uno
     vectorial que anade filtros tienden al mismo punto; se ilustra con el coste
     de una consulta de los tres estratos resuelta en una pasada.

los datos son sinteticos y la magnitud es ilustrativa; la direccion ---el signo
de cada efecto--- es robusta. Python puro con numpy, CPU. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

SEMILLA = 20


def anunciar() -> None:
    print("=" * 64)
    print("cap. 20 — el futuro hibrido: convergencia en un motor multimodelo")
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
# 1. prefiltrado (motor integrado) vs postfiltrado (dos sistemas separados)
# ---------------------------------------------------------------------------

def simular_prefiltrado(n: int = 20000, dim: int = 64, k: int = 10,
                        nq: int = 300) -> None:
    """consulta hibrida: filtro por atributo + vecinos mas cercanos. el motor
    integrado PREFILTRA (busca solo entre los que pasan el filtro) y siempre
    devuelve k resultados validos. dos sistemas separados POSTFILTRAN (el vector
    devuelve sus k mejores y luego se descartan los que no pasan el filtro), y
    al volverse selectivo el filtro casi todos se caen: el recall se hunde."""
    rng = np.random.default_rng(SEMILLA)
    base = _norm(rng.standard_normal((n, dim)))
    print("\nprefiltrado vs postfiltrado: recall@%d segun selectividad" % k)
    print("  selectividad  prefiltrado  postfiltrado")
    filas = []
    for sel in (0.5, 0.2, 0.1, 0.05, 0.01):
        # un atributo binario marca que fraccion `sel` pasa el filtro
        pasa = rng.random(n) < sel
        idx_pasa = np.where(pasa)[0]
        rec_pre = rec_post = 0.0
        for _ in range(nq):
            q = _norm(rng.standard_normal(dim))
            sims = base @ q
            # verdad de referencia: los k mejores ENTRE los que pasan el filtro
            oro = idx_pasa[np.argsort(-sims[idx_pasa])[:k]]
            # integrado (prefiltrado): busca solo en los que pasan -> exacto
            pre = idx_pasa[np.argsort(-sims[idx_pasa])[:k]]
            rec_pre += len(set(pre) & set(oro)) / len(oro)
            # separado (postfiltrado): k mejores globales, luego filtra
            top = np.argsort(-sims)[:k]
            post = [i for i in top if pasa[i]]
            rec_post += len(set(post) & set(oro)) / len(oro)
        filas.append((sel, round(rec_pre / nq, 4), round(rec_post / nq, 4)))
        print(f"  {sel:<12}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap20_prefiltrado.dat"),
              "recall@%d de la consulta hibrida resuelta integrada "
              "(prefiltrado) vs separada (postfiltrado) segun la selectividad "
              "del filtro (n=%d)" % (k, n),
              "selectividad  prefiltrado  postfiltrado", filas)


# ---------------------------------------------------------------------------
# 2. el impuesto del sobremuestreo del postfiltrado
# ---------------------------------------------------------------------------

def simular_sobremuestreo(n: int = 20000, dim: int = 64, k: int = 10,
                          nq: int = 200) -> None:
    """para que el postfiltrado entregue k resultados validos, el sistema
    vectorial debe pedir mas candidatos (sobremuestrear) y filtrarlos despues.
    cuanto mas selectivo el filtro, mayor el multiplicador de sobremuestreo
    necesario para alcanzar k validos. se mide ese factor (candidatos / k)."""
    rng = np.random.default_rng(SEMILLA)
    base = _norm(rng.standard_normal((n, dim)))
    print("\nsobremuestreo: factor de candidatos para %d validos tras filtrar"
          % k)
    print("  selectividad  factor_medio")
    filas = []
    for sel in (0.5, 0.2, 0.1, 0.05, 0.01):
        pasa = rng.random(n) < sel
        factores = []
        for _ in range(nq):
            q = _norm(rng.standard_normal(dim))
            orden = np.argsort(-(base @ q))     # candidatos de mas a menos
            validos = 0
            for pos, i in enumerate(orden, start=1):
                if pasa[i]:
                    validos += 1
                    if validos == k:
                        factores.append(pos / k)   # candidatos vistos / k
                        break
        filas.append((sel, round(float(np.mean(factores)), 2)))
        print(f"  {sel:<12}  {filas[-1][1]:.2f}")
    _escribir(os.path.join("data", "cap20_sobremuestreo.dat"),
              "factor de sobremuestreo (candidatos / k) que el postfiltrado "
              "necesita para entregar %d resultados validos segun la "
              "selectividad del filtro (n=%d)" % (k, n),
              "selectividad  factor", filas)


# ---------------------------------------------------------------------------
# 3. frescura: consistencia del vector respecto al dato que representa
# ---------------------------------------------------------------------------

def simular_frescura(n: int = 20000, dim: int = 64, k: int = 10,
                     nq: int = 300) -> None:
    """el vector es una FOTO del dato en el momento de indexar. si el dato
    cambia y el vector no se reindexa, queda obsoleto: ya no representa lo que
    deberia. se mide el recall@k cuando una fraccion de los vectores esta rancia
    (apunta a una version vieja del dato). el recall cae con esa fraccion."""
    rng = np.random.default_rng(SEMILLA)
    actual = _norm(rng.standard_normal((n, dim)))      # dato actual
    print("\nfrescura: recall@%d segun la fraccion de vectores rancios" % k)
    print("  fraccion_rancia  recall")
    filas = []
    for frac in (0.0, 0.05, 0.1, 0.2, 0.4):
        rec = 0.0
        for _ in range(nq):
            # el indice mezcla vectores frescos y rancios (dato ya cambiado)
            rancio = rng.random(n) < frac
            indice = actual.copy()
            ruido = _norm(actual + 1.2 * rng.standard_normal((n, dim)))
            indice[rancio] = ruido[rancio]              # version vieja
            q = _norm(rng.standard_normal(dim))
            oro = np.argsort(-(actual @ q))[:k]         # verdad sobre el dato
            rec_top = np.argsort(-(indice @ q))[:k]     # lo que el indice da
            rec += len(set(rec_top) & set(oro)) / k
        filas.append((frac, round(rec / nq, 4)))
        print(f"  {frac:<15}  {filas[-1][1]:.4f}")
    _escribir(os.path.join("data", "cap20_frescura.dat"),
              "recall@%d cuando una fraccion del indice esta rancia (el dato "
              "cambio y el vector no se reindexo) (n=%d)" % (k, n),
              "fraccion_rancia  recall", filas)


# ---------------------------------------------------------------------------
# 4. mantenimiento del indice ante un flujo de altas
# ---------------------------------------------------------------------------

def _kmeans(x: np.ndarray, nparts: int, rng, iters: int = 6) -> np.ndarray:
    """kmeans esfericos: unas pocas iteraciones de Lloyd sobre datos normalizados."""
    cent = _norm(x[rng.choice(len(x), nparts, replace=False)])
    for _ in range(iters):
        asign = np.argmax(x @ cent.T, axis=1)
        for c in range(nparts):
            miembros = x[asign == c]
            if len(miembros):
                cent[c] = miembros.mean(0)
        cent = _norm(cent)
    return cent


def simular_mantenimiento(n0: int = 4000, dim: int = 64, k: int = 10,
                          altas: int = 6000, nq: int = 400,
                          nparts: int = 32, nprobe: int = 8) -> None:
    """un indice particionado (estilo IVF: centroides fijados al construir) se
    degrada al insertar datos de temas NUEVOS sin reentrenar, porque los
    centroides viejos no cubren las regiones recien pobladas. reindexar pronto
    (cadencia baja) conserva el recall pero cuesta mas reentrenos; reindexar
    tarde o nunca abarata pero deja caer el recall. se mide ambos."""
    rng = np.random.default_rng(SEMILLA)
    print("\nmantenimiento: recall@%d y coste segun la cadencia de reindexado"
          % k)
    print("  cadencia  recall  reentrenos")
    filas = []
    for cadencia in (500, 1000, 2000, 3000, 99999):
        base = _norm(rng.standard_normal((n0, dim)))      # temas iniciales
        cent = _kmeans(base, nparts, rng)
        asign = np.argmax(base @ cent.T, axis=1)
        reentrenos = 0
        desde = 0
        for j in range(altas):
            # las altas derivan: aparecen temas nuevos (centro que se desplaza)
            centro = _norm(np.array([1.0] + [0.0] * (dim - 1))
                           + (j / altas) * 2.0 *
                           np.array([0.0, 1.0] + [0.0] * (dim - 2)))
            v = _norm(centro + 0.35 * rng.standard_normal((1, dim)))
            base = np.vstack([base, v])
            asign = np.append(asign, int(np.argmax(v @ cent.T)))
            desde += 1
            if desde >= cadencia:                         # toca reindexar
                cent = _kmeans(base, nparts, rng)
                asign = np.argmax(base @ cent.T, axis=1)
                reentrenos += 1
                desde = 0
        # consultas desde la distribucion actual (incluye los temas nuevos)
        rec = 0.0
        for _ in range(nq):
            if rng.random() < 0.5:
                q = _norm(rng.standard_normal(dim))       # tema viejo
            else:
                centro = _norm(np.array([0.0, 1.0] + [0.0] * (dim - 2)))
                q = _norm(centro + 0.35 * rng.standard_normal(dim))  # nuevo
            oro = np.argsort(-(base @ q))[:k]             # verdad por fuerza bruta
            cerca = np.argsort(-(cent @ q))[:nprobe]      # particiones sondeadas
            cand = np.where(np.isin(asign, cerca))[0]
            if len(cand):
                top = cand[np.argsort(-(base[cand] @ q))[:k]]
                rec += len(set(top) & set(oro)) / k
        filas.append((cadencia if cadencia < 99999 else 0,
                      round(rec / nq, 4), reentrenos))
        print(f"  {filas[-1][0]:<8}  {filas[-1][1]:.4f}  {reentrenos}")
    _escribir(os.path.join("data", "cap20_mantenimiento.dat"),
              "recall@%d y numero de reentrenos del indice particionado segun "
              "la cadencia de reindexado, bajo %d altas que derivan a temas "
              "nuevos (cadencia 0 = nunca)" % (k, altas),
              "cadencia  recall  reentrenos", filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta de los tres estratos resuelta en una pasada
# ---------------------------------------------------------------------------

def demostracion(n: int = 20000, dim: int = 64, k: int = 5) -> None:
    """la consulta que cierra el libro: filtrar por un atributo (relacional),
    ordenar por similitud (vectorial) y devolver el documento (documental), en
    una sola pasada de un motor que integra los tres estratos."""
    rng = np.random.default_rng(SEMILLA + 1)
    base = _norm(rng.standard_normal((n, dim)))
    categoria = rng.integers(0, 5, n)        # atributo relacional
    activo = rng.random(n) < 0.3             # otro atributo relacional
    q = _norm(rng.standard_normal(dim))
    # una pasada: prefiltro (categoria==2 y activo) + vecinos + proyeccion
    mask = (categoria == 2) & activo
    idx = np.where(mask)[0]
    top = idx[np.argsort(-(base[idx] @ q))[:k]]
    print("\ndemostracion: una consulta de los tres estratos en una pasada")
    print("  filtro: categoria=2 AND activo   (prefiltrado)")
    print(f"  candidatos tras el filtro: {len(idx)} de {n}")
    print("  id      similitud  categoria  activo")
    for i in top:
        print(f"  {i:<6}  {float(base[i] @ q):+.4f}    "
              f"{int(categoria[i])}          {bool(activo[i])}")


def main() -> None:
    anunciar()
    simular_prefiltrado()
    simular_sobremuestreo()
    simular_frescura()
    simular_mantenimiento()
    demostracion()


if __name__ == "__main__":
    main()
