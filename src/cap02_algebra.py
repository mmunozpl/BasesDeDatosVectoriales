"""capitulo 2: el modelo relacional.

implementa el algebra relacional sobre relaciones representadas como
conjuntos de tuplas con nombre, verifica equivalencias algebraicas (que
son la base de la optimizacion de consultas del cap. 1) y mide como el
orden de evaluacion (empujar la seleccion antes de la reunion) reduce el
tamano del resultado intermedio. la demostracion: 15 tuplas al azar de una
reunion natural.

no usa gpu, ni torch, ni servicio de base de datos: es python puro, cpu
(ver tabla de recursos en IMPLEMENTACION.md). las relaciones viven en
memoria como conjuntos, fieles a la semantica de conjuntos del modelo.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

SEMILLA = 20240602  # semilla fija: la demostracion es reproducible
Fila = tuple[Any, ...]
Pred = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class Rel:
    """una relacion: nombres de atributo y un conjunto de tuplas.

    el conjunto (no la lista) encarna la semantica del modelo: sin
    tuplas repetidas y sin orden entre ellas.
    """

    attrs: tuple[str, ...]
    rows: frozenset[Fila]

    def como_dicts(self) -> list[dict[str, Any]]:
        """devuelve las tuplas como dicts atributo->valor."""
        return [dict(zip(self.attrs, fila)) for fila in self.rows]


def rel(attrs: Sequence[str], filas: Sequence[Fila]) -> Rel:
    """constructor comodo: nombres y filas a una Rel inmutable."""
    return Rel(tuple(attrs), frozenset(filas))


# --- operadores primitivos del algebra ---

def seleccion(r: Rel, pred: Pred) -> Rel:
    """sigma: las tuplas que satisfacen el predicado."""
    filas = {f for f in r.rows if pred(dict(zip(r.attrs, f)))}
    return Rel(r.attrs, frozenset(filas))


def proyeccion(r: Rel, cols: Sequence[str]) -> Rel:
    """pi: se queda con unas columnas; el conjunto deduplica solo."""
    idx = [r.attrs.index(c) for c in cols]
    filas = {tuple(f[i] for i in idx) for f in r.rows}
    return Rel(tuple(cols), frozenset(filas))


def producto(r: Rel, s: Rel) -> Rel:
    """producto cartesiano: cada tupla de r con cada una de s."""
    attrs = r.attrs + s.attrs
    filas = {a + b for a in r.rows for b in s.rows}
    return Rel(attrs, frozenset(filas))


def union(r: Rel, s: Rel) -> Rel:
    """union; exige compatibilidad de union (mismos atributos)."""
    assert r.attrs == s.attrs, "union exige relaciones compatibles"
    return Rel(r.attrs, r.rows | s.rows)


def diferencia(r: Rel, s: Rel) -> Rel:
    """diferencia de conjuntos; misma compatibilidad de union."""
    assert r.attrs == s.attrs, "diferencia exige compatibilidad"
    return Rel(r.attrs, r.rows - s.rows)


def reunion_natural(r: Rel, s: Rel) -> Rel:
    """reunion natural: empareja por los atributos comunes y
    proyecta fuera la columna repetida."""
    comunes = [a for a in r.attrs if a in s.attrs]
    extra_s = [a for a in s.attrs if a not in comunes]
    attrs = r.attrs + tuple(extra_s)
    ir = [r.attrs.index(a) for a in comunes]
    is_ = [s.attrs.index(a) for a in comunes]
    ex = [s.attrs.index(a) for a in extra_s]
    filas = set()
    for a in r.rows:
        clave_a = tuple(a[i] for i in ir)
        for b in s.rows:
            if tuple(b[i] for i in is_) == clave_a:
                filas.add(a + tuple(b[i] for i in ex))
    return Rel(attrs, frozenset(filas))


def iguales(r: Rel, s: Rel) -> bool:
    """igualdad de relaciones sin atender al orden de las columnas:
    compara el conjunto de tuplas como conjunto de dicts."""
    ca = {frozenset(d.items()) for d in r.como_dicts()}
    cb = {frozenset(d.items()) for d in s.como_dicts()}
    return ca == cb


def verificar_equivalencias() -> None:
    """comprueba leyes del algebra sobre datos de juguete.

    cada ley es una identidad exacta entre conjuntos: si falla, el
    assert detiene el programa (arbitra error de codigo, no de dato).
    son las mismas reescrituras que el optimizador del cap. 1 aplica.
    """
    libro = rel(
        ("id", "titulo", "tema", "id_autor"),
        [
            (1, "geometria i", "geometria", 10),
            (2, "algebra i", "algebra", 11),
            (3, "redes", "redes", 10),
            (4, "logica", "logica", 12),
        ],
    )
    autor = rel(
        ("id_autor", "autor"),
        [(10, "ada"), (11, "boole"), (12, "frege"), (13, "godel")],
    )
    p: Pred = lambda t: t["tema"] == "geometria"  # solo sobre libro

    # 1) empuje de la seleccion: filtrar antes o despues de reunir.
    izq = seleccion(reunion_natural(libro, autor), p)
    der = reunion_natural(seleccion(libro, p), autor)
    assert iguales(izq, der), "falla el empuje de la seleccion"

    # 2) conmutatividad de la reunion natural (como conjuntos).
    assert iguales(
        reunion_natural(libro, autor), reunion_natural(autor, libro)
    ), "falla la conmutatividad de la reunion"

    # 3) cascada de selecciones: sigma_{p y q} = sigma_p(sigma_q()).
    q: Pred = lambda t: t["id"] > 1
    casc = seleccion(seleccion(libro, p), q)
    junta = seleccion(libro, lambda t: p(t) and q(t))
    assert iguales(casc, junta), "falla la cascada de selecciones"

    # 4) de morgan sobre el predicado de la seleccion.
    no_or = seleccion(libro, lambda t: not (p(t) or q(t)))
    y_no = seleccion(libro, lambda t: (not p(t)) and (not q(t)))
    assert iguales(no_or, y_no), "falla de morgan en la seleccion"

    # 5) distributividad de la seleccion sobre la union.
    mas = rel(
        libro.attrs,
        [(5, "calculo", "algebra", 11), (6, "grafos", "geometria", 10)],
    )
    izq5 = seleccion(union(libro, mas), p)
    der5 = union(seleccion(libro, p), seleccion(mas, p))
    assert iguales(izq5, der5), "falla distributividad sobre union"

    # 6) interseccion como R - (R - S).
    s6 = rel(
        libro.attrs,
        [(1, "geometria i", "geometria", 10), (9, "x", "x", 1)],
    )
    inter_def = Rel(libro.attrs, libro.rows & s6.rows)
    inter_id = diferencia(libro, diferencia(libro, s6))
    assert iguales(inter_def, inter_id), "falla R - (R - S)"

    print("equivalencias algebraicas: 6/6 verificadas (exactas)")


def division(r: Rel, s: Rel) -> Rel:
    """division: tuplas de r (en sus atributos no compartidos)
    asociadas a TODAS las tuplas de s. s.attrs debe estar en r.attrs."""
    assert set(s.attrs) <= set(r.attrs), "s.attrs debe estar en r.attrs"
    resto = [a for a in r.attrs if a not in s.attrs]
    ir = [r.attrs.index(a) for a in resto]
    isr = [r.attrs.index(a) for a in s.attrs]
    candidatos = {tuple(f[i] for i in ir) for f in r.rows}
    presentes = {
        (tuple(f[i] for i in ir), tuple(f[i] for i in isr))
        for f in r.rows
    }
    sv = set(s.rows)
    res = {
        c for c in candidatos
        if all((c, v) in presentes for v in sv)
    }
    return Rel(tuple(resto), frozenset(res))


def verificar_division() -> None:
    """comprueba (R / S) join S contenido en R, en datos de juguete."""
    lectura = rel(
        ("socio", "libro"),
        [
            ("ana", "l1"), ("ana", "l2"), ("ana", "l3"),
            ("bea", "l1"), ("bea", "l3"),
        ],
    )
    exigidos = rel(("libro",), [("l1",), ("l2",)])
    coc = division(lectura, exigidos)
    recompuesto = reunion_natural(coc, exigidos)
    sub = set(recompuesto.rows) <= set(lectura.rows)
    assert sub, "(R / S) join S no esta contenido en R"
    socios = sorted(s[0] for s in coc.rows)
    print(f"division: cociente = {socios}; (R/S) join S subset R: ok")


def medir_empuje() -> None:
    """mide el tamano del resultado intermedio con y sin empuje de la
    seleccion, barriendo la selectividad, y lo guarda en un .dat.

    sin empuje (reunir y luego filtrar) materializa la reunion entera,
    constante; con empuje (filtrar y luego reunir) el intermedio crece
    con la selectividad. es el argumento del optimizador, medido.
    """
    rng = random.Random(SEMILLA)
    n_r = 2000
    # cada tupla de r casa con exactamente una de s -> reunion = n_r.
    libro = rel(
        ("id", "id_autor"),
        [(i, i % 200) for i in range(n_r)],
    )
    autor = rel(
        ("id_autor", "autor"),
        [(k, f"a{k:03d}") for k in range(200)],
    )
    j = reunion_natural(libro, autor)
    total = len(j.rows)  # tamano de la reunion completa
    ruta = os.path.join("data", "cap02_pushdown.dat")
    os.makedirs("data", exist_ok=True)
    sels = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
    ids = list(range(n_r))
    with open(ruta, "w", encoding="utf-8") as fh:
        # comentario con # y, debajo, la fila de nombres SIN # para
        # que pgfplots la lea como cabecera de columnas.
        fh.write("# tuplas intermedias al evaluar sigma_R(R join S)\n")
        fh.write("selectividad  sin_empuje  con_empuje\n")
        for s in sels:
            # con empuje: filtrar primero (sigma) y luego reunir; el
            # intermedio es la reunion de las que pasan. quien pasa lo
            # decide rng (semilla fija -> reproducible).
            k = int(round(s * n_r))
            pasan = set(rng.sample(ids, k))
            filtrado = seleccion(
                libro, lambda t, pasan=pasan: t["id"] in pasan
            )
            inter_con = reunion_natural(filtrado, autor)
            sin = total  # reunir y luego filtrar (reunion entera)
            con = len(inter_con.rows)  # filtrar y luego reunir (real)
            fh.write(f"{s:<13} {sin:<11} {con}\n")
    print(f"empuje medido en {ruta} (reunion = {total} tuplas)")


def muestra_join(k: int = 15) -> None:
    """la demostracion: k tuplas al azar de una reunion natural."""
    libro = rel(
        ("id", "titulo", "id_autor"),
        [(i, f"obra {i:03d}", i % 7) for i in range(120)],
    )
    autor = rel(
        ("id_autor", "autor"),
        [(a, f"autor-{a}") for a in range(7)],
    )
    j = reunion_natural(libro, autor)
    rng = random.Random(SEMILLA)
    filas = rng.sample(sorted(j.rows), k)
    print(f"\nla demostracion: {k} tuplas al azar de la reunion natural")
    print("  " + "  ".join(j.attrs))
    print("  " + "  ".join("-" * len(a) for a in j.attrs))
    for f in filas:
        print("  " + "  ".join(str(v) for v in f))


def main() -> None:
    print("=" * 64)
    print("cap. 2: algebra relacional")
    print("recursos: python puro · cpu. sin gpu ni servicio de bd.")
    print("=" * 64)
    verificar_equivalencias()
    verificar_division()
    medir_empuje()
    muestra_join(15)


if __name__ == "__main__":
    main()
