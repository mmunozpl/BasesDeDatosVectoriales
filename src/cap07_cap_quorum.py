"""capitulo 7: consistencia distribuida.

simula, con replicas en memoria, las tres tensiones que el teorema CAP y el
modelo de quorum hacen explicitas, y mide cada una para alimentar una grafica:

  1. consistencia: con N replicas, una escritura toca W de ellas y una lectura
     R; si R + W > N los conjuntos se solapan siempre y la lectura ve el ultimo
     valor (consistencia fuerte). por debajo del umbral aparecen lecturas
     obsoletas. se barre R + W y se mide la fraccion de lecturas obsoletas.
  2. disponibilidad: una escritura necesita W replicas vivas; al subir la
     probabilidad de fallo de cada nodo, exigir un quorum grande (consistencia
     fuerte) se vuelve indisponible antes que exigir uno pequeno. es la
     eleccion del CAP, medida.
  3. latencia: contactar un quorum es esperar a las Q replicas mas rapidas de
     N; al crecer Q, la latencia de cola crece. la consistencia cuesta espera.

es Python puro (sin servicio ni torch): basta el interprete. semilla fija
para que los numeros se reproduzcan.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

SEMILLA = 7
N = 5  # numero de replicas


def anunciar() -> None:
    print("=" * 64)
    print("cap. 7: consistencia distribuida (simulacion de quorum)")
    print("recursos: python puro · cpu. no usa servicio, gpu ni torch.")
    print(f"replicas N = {N}, semilla = {SEMILLA}")
    print("=" * 64)


def _escribir(ruta: str, cabecera: dict,
              filas: List[Tuple]) -> None:
    """vuelca una tabla a un .dat con comentario y linea de columnas."""
    os.makedirs("data", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(f"# {cabecera['nota']}\n")
        fh.write(cabecera["cols"] + "\n")
        for fila in filas:
            fh.write("  ".join(str(x) for x in fila) + "\n")
    print(f"escrito {ruta}")


def simular_consistencia(trials: int = 40000) -> None:
    """fraccion de lecturas obsoletas en funcion de R + W.

    una escritura marca W replicas con el ultimo valor; una lectura mira R
    replicas y es fresca si alcanza alguna marcada. se promedian todas las
    parejas (R, W) que comparten la misma suma R + W.
    """
    rng = random.Random(SEMILLA)
    replicas = list(range(N))
    por_suma = {}
    for r in range(1, N + 1):
        for w in range(1, N + 1):
            obsoletas = 0
            for _ in range(trials):
                escritas = set(rng.sample(replicas, w))
                leidas = rng.sample(replicas, r)
                if not any(x in escritas for x in leidas):
                    obsoletas += 1
            por_suma.setdefault(r + w, []).append(obsoletas / trials)
    filas = [(s, round(sum(v) / len(v), 4))
             for s, v in sorted(por_suma.items())]
    print("\nconsistencia: lecturas obsoletas segun R + W")
    for s, tasa in filas:
        marca = "  <- R+W > N: consistencia fuerte" if s > N else ""
        print(f"  R+W = {s:2}   obsoletas = {tasa:.3f}{marca}")
    _escribir(
        os.path.join("data", "cap07_quorum.dat"),
        {"nota": "fraccion de lecturas obsoletas segun R+W (N=%d)" % N,
         "cols": "r_mas_w  obsoletas"},
        filas)


def simular_disponibilidad(trials: int = 40000) -> None:
    """probabilidad de poder formar un quorum de escritura segun el fallo.

    cada nodo cae de forma independiente con probabilidad p; una escritura de
    nivel W tiene exito si quedan al menos W nodos vivos. se comparan tres
    niveles: ONE (W=1), QUORUM (mayoria) y ALL (W=N).
    """
    rng = random.Random(SEMILLA)
    quorum = N // 2 + 1
    niveles = [("one", 1), ("quorum", quorum), ("all", N)]
    filas = []
    print("\ndisponibilidad de escritura segun fallo por nodo p")
    print("  p      one    quorum  all")
    for i in range(0, 11):
        p = i / 20.0  # 0.00 .. 0.50
        exito = {nombre: 0 for nombre, _ in niveles}
        for _ in range(trials):
            vivos = sum(1 for _ in range(N) if rng.random() > p)
            for nombre, w in niveles:
                if vivos >= w:
                    exito[nombre] += 1
        fila = (round(p, 3),
                round(exito["one"] / trials, 4),
                round(exito["quorum"] / trials, 4),
                round(exito["all"] / trials, 4))
        filas.append(fila)
        print(f"  {fila[0]:.2f}   {fila[1]:.3f}  {fila[2]:.3f}"
              f"   {fila[3]:.3f}")
    _escribir(
        os.path.join("data", "cap07_disponibilidad.dat"),
        {"nota": "disponibilidad de escritura segun p (N=%d, "
                 "quorum=%d)" % (N, quorum),
         "cols": "p  one  quorum  all"},
        filas)


def simular_latencia(trials: int = 40000) -> None:
    """latencia de contactar un quorum de tamano Q: la Q-esima replica mas
    rapida de N. la latencia de cada replica sigue una lognormal."""
    rng = random.Random(SEMILLA)
    filas = []
    print("\nlatencia (ms) de contactar un quorum de tamano Q")
    print("  Q   mediana   p95")
    for q in range(1, N + 1):
        muestras = []
        for _ in range(trials):
            lat = sorted(rng.lognormvariate(2.0, 0.6)
                         for _ in range(N))
            muestras.append(lat[q - 1])  # esperar a las q mas rapidas
        muestras.sort()
        p50 = muestras[len(muestras) // 2]
        p95 = muestras[int(len(muestras) * 0.95)]
        filas.append((q, round(p50, 2), round(p95, 2)))
        print(f"  {q}   {p50:7.2f}  {p95:7.2f}")
    _escribir(
        os.path.join("data", "cap07_latencia.dat"),
        {"nota": "latencia ms de un quorum de tamano Q (N=%d)" % N,
         "cols": "q  p50  p95"},
        filas)


def simular_convergencia(replicas: int = 50, rondas: int = 9,
                         trials: int = 4000) -> None:
    """fraccion de replicas que conocen el ultimo valor ronda a ronda.

    modela la propagacion epidemica (gossip/anti-entropia): cada ronda, toda
    replica aun desactualizada contacta a otra al azar y, si esta ya tiene el
    valor, lo copia. la curva sube en S hasta converger: la ventana de
    inconsistencia hecha grafica.
    """
    rng = random.Random(SEMILLA)
    promedio = [0.0] * (rondas + 1)
    for _ in range(trials):
        informado = [False] * replicas
        informado[0] = True  # una replica recibio la escritura
        promedio[0] += 1.0 / replicas
        for t in range(1, rondas + 1):
            nuevos = []
            for i in range(replicas):
                if not informado[i]:
                    otro = rng.randrange(replicas)
                    if informado[otro]:
                        nuevos.append(i)
            for i in nuevos:
                informado[i] = True
            promedio[t] += sum(informado) / replicas
    filas = [(t, round(promedio[t] / trials, 4))
             for t in range(rondas + 1)]
    print(f"\nconvergencia por gossip ({replicas} replicas)")
    print("  ronda  fraccion_convergida")
    for t, frac in filas:
        print(f"  {t:<5}  {frac:.3f}")
    _escribir(
        os.path.join("data", "cap07_convergencia_t.dat"),
        {"nota": "fraccion de replicas convergidas por ronda de gossip "
                 "(%d replicas)" % replicas,
         "cols": "ronda  convergida"},
        filas)


def simular_escala_n(p: float = 0.2, trials: int = 40000) -> None:
    """disponibilidad del quorum de mayoria al crecer el numero de replicas.

    con una probabilidad de fallo por nodo fija, mas replicas hacen que la
    mayoria sea mas robusta: tolerar f fallos exige N >= 2f+1. la curva sube
    hacia 1 al crecer N, lo que explica por que los sistemas reales replican
    en numeros impares y modestos (3, 5, 7).
    """
    rng = random.Random(SEMILLA)
    filas = []
    print(f"\ndisponibilidad del quorum de mayoria con p={p}")
    print("  N    quorum  disponibilidad")
    for n in (3, 5, 7, 9, 11):
        quorum = n // 2 + 1
        exito = 0
        for _ in range(trials):
            vivos = sum(1 for _ in range(n) if rng.random() > p)
            if vivos >= quorum:
                exito += 1
        avail = round(exito / trials, 4)
        filas.append((n, quorum, avail))
        print(f"  {n:<4} {quorum:<6}  {avail:.4f}")
    _escribir(
        os.path.join("data", "cap07_escala_n.dat"),
        {"nota": "disponibilidad del quorum de mayoria segun N (p=%.2f)" % p,
         "cols": "n  quorum  disponibilidad"},
        filas)


def demostracion(k: int = 15) -> None:
    """15 lecturas bajo un nivel debil (R=W=1), con su desenlace.

    muestra que, sin solapamiento garantizado, unas lecturas ven el ultimo
    valor y otras uno obsoleto: la consistencia eventual en pequeno.
    """
    rng = random.Random(SEMILLA + 1)
    replicas = list(range(N))
    print(f"\ndemostracion: {k} lecturas con R=W=1 (consistencia eventual)")
    print("  n   escrita_en  leida_en  desenlace")
    print("  --  ----------  --------  ---------")
    frescas = 0
    for i in range(k):
        w = rng.choice(replicas)
        r = rng.choice(replicas)
        fresca = (r == w)
        frescas += int(fresca)
        estado = "fresca" if fresca else "OBSOLETA"
        print(f"  {i:<2}  rep-{w}       rep-{r}     {estado}")
    print(f"  -> {frescas}/{k} frescas; el resto vio un valor anterior")


def main() -> None:
    anunciar()
    simular_consistencia()
    simular_disponibilidad()
    simular_latencia()
    simular_convergencia()
    simular_escala_n()
    demostracion(15)


if __name__ == "__main__":
    main()
