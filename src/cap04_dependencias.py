"""capitulo 4 — diseno y normalizacion.

opera sobre dependencias funcionales: calcula el cierre de un
conjunto de atributos, halla las claves candidatas, detecta
violaciones de Boyce-Codd (BCNF) y descompone el esquema sin
perdida. ademas mide la redundancia —cuantas veces se almacena un
mismo hecho— en un esquema desnormalizado frente a su version
normalizada, y muestra 15 tuplas del esquema desnormalizado.

python puro, cpu: sin gpu, sin torch, sin servicio de base de datos (las
dependencias y relaciones viven en memoria como conjuntos). ver la tabla
de recursos de IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
import random
from itertools import combinations

SEMILLA = 20240604
Atrs = frozenset[str]
DF = tuple[Atrs, Atrs]  # (lhs, rhs)


def fd(izq: str, der: str) -> DF:
    """construye una dependencia 'a,b -> c,d' desde cadenas comodas."""
    return (frozenset(izq.split()), frozenset(der.split()))


def cierre(x: Atrs, fds: list[DF]) -> Atrs:
    """cierre de x bajo fds: todos los atributos que x determina."""
    res = set(x)
    cambio = True
    while cambio:
        cambio = False
        for izq, der in fds:
            if izq <= res and not der <= res:
                res |= der
                cambio = True
    return frozenset(res)


def es_superclave(x: Atrs, r: Atrs, fds: list[DF]) -> bool:
    """x es superclave de r si su cierre cubre todos los atributos."""
    return r <= cierre(x, fds)


def claves_candidatas(r: Atrs, fds: list[DF]) -> list[Atrs]:
    """todas las claves candidatas: superclaves minimales."""
    cand: list[Atrs] = []
    atrs = sorted(r)
    for k in range(1, len(atrs) + 1):
        for combo in combinations(atrs, k):
            x = frozenset(combo)
            if any(c <= x for c in cand):
                continue  # ya contiene una clave: no es minimal
            if es_superclave(x, r, fds):
                cand.append(x)
    return cand


def es_trivial(d: DF) -> bool:
    """una dependencia x -> y es trivial si y esta contenida en x."""
    return d[1] <= d[0]


def violaciones_bcnf(r: Atrs, fds: list[DF]) -> list[DF]:
    """dependencias no triviales x -> y con x no superclave de r."""
    viol = []
    for d in fds:
        izq, der = d
        if izq <= r and (der & r) and not es_trivial(d):
            if not es_superclave(izq, r, fds):
                viol.append((izq, der & r))
    return viol


def proyectar(fds: list[DF], s: Atrs) -> list[DF]:
    """proyecta las dependencias sobre el subconjunto de atributos s."""
    proy: list[DF] = []
    atrs = sorted(s)
    for k in range(1, len(atrs)):
        for combo in combinations(atrs, k):
            x = frozenset(combo)
            cerrado = cierre(x, fds) & s
            der = cerrado - x
            if der:
                proy.append((x, der))
    return proy


def descomponer_bcnf(r: Atrs, fds: list[DF]) -> list[Atrs]:
    """descompone r en relaciones BCNF, sin perdida (estandar)."""
    viol = violaciones_bcnf(r, fds)
    if not viol:
        return [r]
    izq, _ = viol[0]
    cerrado = cierre(izq, fds) & r
    r1 = cerrado                      # x+  (contiene a x)
    r2 = (r - cerrado) | izq          # resto + x  (interseccion = x)
    res: list[Atrs] = []
    for sub in (r1, r2):
        res += descomponer_bcnf(sub, proyectar(fds, sub))
    # elimina subconjuntos redundantes
    finales: list[Atrs] = []
    for s in res:
        if not any(s < o for o in res):
            if s not in finales:
                finales.append(s)
    return finales


def sin_perdida_binaria(r1: Atrs, r2: Atrs, fds: list[DF]) -> bool:
    """test de reunion sin perdida para una descomposicion binaria:
    la interseccion debe ser superclave de r1 o de r2."""
    comun = r1 & r2
    return (es_superclave(comun, r1, fds)
            or es_superclave(comun, r2, fds))


def demo_normalizacion() -> None:
    """ejemplo: cierre, claves, violacion BCNF y descomposicion."""
    r = frozenset({"id", "titulo", "id_autor", "autor", "pais"})
    fds = [
        fd("id", "titulo id_autor"),
        fd("id_autor", "autor pais"),
    ]
    print("esquema R =", sorted(r))
    print("dependencias:")
    for izq, der in fds:
        print(f"  {sorted(izq)} -> {sorted(der)}")
    print("cierre {id}+   =", sorted(cierre(frozenset({"id"}), fds)))
    print("cierre {id_autor}+ =",
          sorted(cierre(frozenset({"id_autor"}), fds)))
    claves = claves_candidatas(r, fds)
    print("claves candidatas:", [sorted(c) for c in claves])
    viol = violaciones_bcnf(r, fds)
    print("violaciones BCNF:",
          [f"{sorted(i)} -> {sorted(d)}" for i, d in viol])
    desc = descomponer_bcnf(r, fds)
    print("descomposicion BCNF:", [sorted(s) for s in desc])
    if len(desc) == 2:
        ok = sin_perdida_binaria(desc[0], desc[1], fds)
        print(f"reunion sin perdida: {ok}")


def medir_redundancia() -> None:
    """mide cuantas filas almacenan los datos de autor en el esquema
    desnormalizado (una por libro) frente al normalizado,
    al crecer el numero de libros con un numero fijo de autores."""
    rng = random.Random(SEMILLA)
    d_autores = 50
    tamanos = [100, 500, 1000, 5000, 20000, 100000]
    ruta = os.path.join("data", "cap04_redundancia.dat")
    os.makedirs("data", exist_ok=True)
    filas = []
    for n in tamanos:
        autores = [rng.randint(0, d_autores - 1) for _ in range(n)]
        denorm = n                       # datos de autor por libro
        norm = len(set(autores))         # datos de autor por autor
        filas.append((n, denorm, norm))
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("# filas que almacenan datos de autor por tamano N\n")
        fh.write("N  desnormalizado  normalizado\n")
        for n, de, no in filas:
            fh.write(f"{n}  {de}  {no}\n")
    print(f"\nredundancia medida en {ruta} ({d_autores} autores)")


def demostracion(k: int = 15) -> None:
    """15 tuplas al azar del esquema desnormalizado: el autor se
    repite, libro tras libro, en cada fila."""
    rng = random.Random(SEMILLA)
    paises = ["es", "uk", "us", "fr", "de"]
    autores = {a: (f"autor-{a}", rng.choice(paises)) for a in range(8)}
    filas = []
    for i in range(200):
        a = rng.randint(0, 7)
        nom, pais = autores[a]
        filas.append((i, f"obra {i:03d}", nom, pais))
    muestra = rng.sample(filas, k)
    print(f"\ndemostracion: {k} tuplas del esquema DESNORMALIZADO")
    print("  id   titulo     autor     pais")
    print("  ---  ---------  --------  ----")
    for i, t, nom, pais in muestra:
        print(f"  {i:<3}  {t:<9}  {nom:<8}  {pais}")
    print("  (autor y pais se repiten en cada libro del mismo autor)")


def main() -> None:
    print("=" * 64)
    print("cap. 4 — diseno y normalizacion")
    print("recursos: python puro · cpu. sin gpu ni servicio de bd.")
    print("=" * 64)
    demo_normalizacion()
    medir_redundancia()
    demostracion(15)


if __name__ == "__main__":
    main()
