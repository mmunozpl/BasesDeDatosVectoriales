"""capitulo 9 — mecanica de la distribucion.

implementa y mide, en Python puro, las cuatro piezas comunes a cualquier
almacen distribuido:

  1. particionado: repartir por hash modulo N reubica casi todas las claves al
     cambiar el numero de nodos; el hashing consistente (un anillo) mueve solo
     la fraccion que toca al nodo nuevo. se mide la fraccion de claves que se
     mueven al pasar de N a N+1 nodos con cada esquema.
  2. balance: el anillo desnudo reparte de forma desigual; las replicas
     virtuales (varios puntos por nodo) igualan la carga. se mide el desbalance
     segun el numero de replicas virtuales.
  3. replicacion: con factor de replicacion RF, una particion se pierde solo si
     fallan sus RF replicas a la vez. se mide la probabilidad de perdida segun
     RF bajo fallos independientes.
  4. map-reduce: el tiempo de un trabajo paralelo cae con los trabajadores,
     pero el sesgo de los datos (una clave enorme) pone un suelo. se mide el
     makespan ideal frente al sesgado segun el numero de trabajadores, sobre un
     conteo de palabras real.

es Python puro (sin servicio ni torch, segun la tabla de recursos): basta el
interprete, con semilla fija. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Dict, List, Tuple

SEMILLA = 9
ANILLO = 2 ** 32  # tamano del espacio de hash del anillo


def h(cadena: str) -> int:
    """hash estable de una cadena a un entero del anillo."""
    dig = hashlib.md5(cadena.encode("utf-8")).hexdigest()
    return int(dig, 16) % ANILLO


def anunciar() -> None:
    print("=" * 64)
    print("cap. 9 — mecanica de la distribucion")
    print("recursos: python puro · cpu. no usa servicio, gpu ni torch.")
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


class Anillo:
    """anillo de hashing consistente con replicas virtuales."""

    def __init__(self, nodos: List[str], vnodos: int = 1) -> None:
        self.vnodos = vnodos
        self.puntos: List[Tuple[int, str]] = []
        for nodo in nodos:
            for v in range(vnodos):
                self.puntos.append((h(f"{nodo}#{v}"), nodo))
        self.puntos.sort()

    def nodo(self, clave: str) -> str:
        """nodo fisico responsable de la clave: el siguiente en el anillo."""
        pos = h(clave)
        lo, hi = 0, len(self.puntos)
        while lo < hi:                      # busqueda binaria del sucesor
            mid = (lo + hi) // 2
            if self.puntos[mid][0] < pos:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(self.puntos):          # se da la vuelta al anillo
            lo = 0
        return self.puntos[lo][1]


def simular_movimiento(k: int = 20000) -> None:
    """fraccion de claves que cambian de nodo al pasar de N a N+1 nodos."""
    rng = random.Random(SEMILLA)
    claves = [f"clave-{rng.randrange(10 ** 9)}" for _ in range(k)]
    filas = []
    print("\nmovimiento al anadir un nodo (fraccion de claves movidas)")
    print("  N->N+1   modulo   anillo")
    for n in (2, 4, 8, 16, 32):
        # modulo: nodo = h(clave) % N
        mov_mod = sum(1 for c in claves
                      if h(c) % n != h(c) % (n + 1)) / k
        # anillo: nodo asignado antes y despues de anadir uno
        a1 = Anillo([f"n{i}" for i in range(n)], vnodos=50)
        a2 = Anillo([f"n{i}" for i in range(n + 1)], vnodos=50)
        mov_ani = sum(1 for c in claves
                      if a1.nodo(c) != a2.nodo(c)) / k
        filas.append((n, round(mov_mod, 4), round(mov_ani, 4)))
        print(f"  {n}->{n + 1:<5}  {mov_mod:.4f}   {mov_ani:.4f}")
    _escribir(os.path.join("data", "cap09_movimiento.dat"),
              "fraccion de claves movidas al pasar de N a N+1 nodos",
              "n  modulo  anillo", filas)


def simular_balance(k: int = 50000, nodos: int = 10) -> None:
    """desbalance de carga (max/media) segun las replicas virtuales."""
    rng = random.Random(SEMILLA)
    claves = [f"clave-{rng.randrange(10 ** 9)}" for _ in range(k)]
    nombres = [f"n{i}" for i in range(nodos)]
    filas = []
    print("\nbalance: desbalance de carga (max/media) segun vnodos")
    print("  vnodos   desbalance")
    for v in (1, 2, 5, 10, 20, 50, 100, 200):
        anillo = Anillo(nombres, vnodos=v)
        carga: Dict[str, int] = {n: 0 for n in nombres}
        for c in claves:
            carga[anillo.nodo(c)] += 1
        media = k / nodos
        desbalance = max(carga.values()) / media
        filas.append((v, round(desbalance, 4)))
        print(f"  {v:<6}   {desbalance:.4f}")
    _escribir(os.path.join("data", "cap09_balance.dat"),
              "desbalance de carga (carga_max/media) segun vnodos por nodo "
              "(%d nodos)" % nodos,
              "vnodos  desbalance", filas)


def simular_durabilidad(particiones: int = 20000, nodos: int = 20,
                        p_fallo: float = 0.3, trials: int = 30) -> None:
    """probabilidad de perder una particion segun el factor de replicacion.

    cada particion coloca sus RF replicas en RF nodos distintos al azar; un
    nodo falla con probabilidad p; la particion se pierde si fallan sus RF.
    """
    rng = random.Random(SEMILLA)
    filas = []
    print(f"\ndurabilidad: prob. de perder una particion (p_fallo={p_fallo})")
    print("  RF   prob_perdida")
    for rf in (1, 2, 3, 4, 5):
        perdidas = 0
        total = 0
        for _ in range(trials):
            caidos = {i for i in range(nodos) if rng.random() < p_fallo}
            for _ in range(particiones):
                replicas = rng.sample(range(nodos), rf)
                total += 1
                if all(r in caidos for r in replicas):
                    perdidas += 1
        prob = perdidas / total
        filas.append((rf, round(prob, 6)))
        print(f"  {rf}    {prob:.6f}")
    _escribir(os.path.join("data", "cap09_durabilidad.dat"),
              "prob. de perder una particion segun RF (p_fallo=%.2f, "
              "%d nodos)" % (p_fallo, nodos),
              "rf  prob_perdida", filas)


def simular_hotspot(k: int = 100000) -> None:
    """desbalance de carga del reparto por rango frente al de hash, cuando
    las claves estan sesgadas (p. ej. marcas de tiempo recientes).

    el reparto por rango envia todas las claves vecinas al mismo nodo, asi que
    un pico de claves (lo reciente) crea un punto caliente; el de hash las
    dispersa y reparte por igual.
    """
    rng = random.Random(SEMILLA)
    espacio = 10 ** 9
    # claves sesgadas: el 80 % cae en el 10 % alto del espacio (lo reciente)
    claves = []
    for _ in range(k):
        if rng.random() < 0.8:
            claves.append(rng.randint(int(espacio * 0.9), espacio - 1))
        else:
            claves.append(rng.randrange(espacio))
    filas = []
    print("\nhotspot: desbalance por rango vs por hash (claves sesgadas)")
    print("  N    por_rango  por_hash")
    for n in (2, 4, 8, 16):
        ancho = espacio // n
        carga_r = [0] * n
        carga_h = [0] * n
        for c in claves:
            carga_r[min(c // ancho, n - 1)] += 1
            carga_h[h(str(c)) % n] += 1
        media = k / n
        des_r = max(carga_r) / media
        des_h = max(carga_h) / media
        filas.append((n, round(des_r, 4), round(des_h, 4)))
        print(f"  {n:<3}  {des_r:8.4f}   {des_h:.4f}")
    _escribir(os.path.join("data", "cap09_hotspot.dat"),
              "desbalance (max/media) por rango vs por hash con claves "
              "sesgadas",
              "n  por_rango  por_hash", filas)


def _corpus(tokens: int, vocab: int, rng: random.Random) -> List[str]:
    """genera un texto con frecuencias de palabra sesgadas (Zipf)."""
    pesos = [1.0 / (i + 1) for i in range(vocab)]
    idx = rng.choices(range(vocab), weights=pesos, k=tokens)
    return [f"palabra{i:04d}" for i in idx]


def map_reduce_conteo(texto: List[str]) -> Dict[str, int]:
    """un map-reduce de conteo de palabras, en pequeno: map, shuffle, reduce."""
    # map: cada palabra emite (palabra, 1)
    pares = [(palabra, 1) for palabra in texto]
    # shuffle: agrupar por clave
    grupos: Dict[str, List[int]] = {}
    for clave, uno in pares:
        grupos.setdefault(clave, []).append(uno)
    # reduce: sumar cada grupo
    return {clave: sum(unos) for clave, unos in grupos.items()}


def simular_mapreduce() -> None:
    """makespan ideal frente al sesgado segun el numero de trabajadores."""
    rng = random.Random(SEMILLA)
    texto = _corpus(tokens=200000, vocab=2000, rng=rng)
    conteo = map_reduce_conteo(texto)
    total = sum(conteo.values())
    filas = []
    print("\nmap-reduce: makespan (fraccion del total) segun trabajadores")
    print("  W    ideal     sesgado")
    for w in (1, 2, 4, 8, 16, 32):
        carga = [0] * w
        for palabra, n in conteo.items():
            carga[h(palabra) % w] += n     # el reducer lo decide el hash
        ideal = (total / w) / total        # reparto perfecto
        sesgado = max(carga) / total       # lo limita el reducer mas cargado
        filas.append((w, round(ideal, 4), round(sesgado, 4)))
        print(f"  {w:<3}  {ideal:.4f}    {sesgado:.4f}")
    _escribir(os.path.join("data", "cap09_mapreduce.dat"),
              "makespan (fraccion del total) ideal vs sesgado segun W "
              "trabajadores",
              "w  ideal  sesgado", filas)
    # palabras mas frecuentes: salida real del map-reduce
    top = sorted(conteo.items(), key=lambda kv: -kv[1])[:5]
    print("  palabras mas frecuentes:", top)


def demostracion(k: int = 15) -> None:
    """15 claves al azar y el nodo del anillo que les toca."""
    rng = random.Random(SEMILLA + 1)
    anillo = Anillo([f"n{i}" for i in range(6)], vnodos=50)
    print(f"\ndemostracion: {k} claves al azar y su nodo en el anillo")
    print("  clave          nodo")
    print("  -------------  ----")
    for _ in range(k):
        c = f"clave-{rng.randrange(10 ** 6)}"
        print(f"  {c:13}  {anillo.nodo(c)}")


def main() -> None:
    anunciar()
    simular_movimiento()
    simular_hotspot()
    simular_balance()
    simular_durabilidad()
    simular_mapreduce()
    demostracion(15)


if __name__ == "__main__":
    main()
