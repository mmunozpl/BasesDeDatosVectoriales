"""capitulo 1 — la persistencia y la disciplina de gestion de datos.

demuestra, sobre una tabla relacional de juguete en postgres, las dos
ideas que el capitulo fija: (1) un sgbd separa el dato de la aplicacion
y (2) esa separacion permite cambiar el acceso fisico —de barrido
secuencial a indice b-tree— sin tocar la consulta logica. imprime el
plan de ejecucion antes y despues de crear el indice y, como demostracion,
15 observaciones reproducibles (muestreadas con la semilla fija).

no usa gpu ni torch (ver tabla de recursos en IMPLEMENTACION.md). el
unico recurso es el servicio postgres del docker-compose de infra/.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg

# corpus de juguete: un dominio bibliografico minimo. el contenido no
# importa, solo que haya filas suficientes para que el optimizador
# prefiera el indice al barrido secuencial.
TEMAS = (
    "bases de datos", "algebra", "redes", "sistemas operativos",
    "compiladores", "estadistica", "geometria", "logica",
)
N_FILAS = int(os.environ.get("CAP01_FILAS", "20000"))
SEMILLA = 20240601  # semilla fija: la demostracion es reproducible

# --- modelo de coste para la figura del punto de cruce (fig. 1.3) ---
# la curva es un modelo idealizado anclado a una corrida REAL: los
# costes del barrido y del indice son los que PostgreSQL 16.14 reporta
# en el plan (lst:seqscan/lst:idxscan; entorno: pgvector/pgvector:pg16,
# infra/docker-compose.yml). el barrido es plano (coste total medido);
# el indice se interpola lineal desde su arranque y pasa por el coste
# real en la selectividad de la consulta. el cruce, de igualar caminos.
SEQ_TOTAL = 508.00     # barrido: coste total medido (plan, lst:seqscan)
IDX_ARRANQUE = 11.14   # indice: arranque medido (plan, lst:idxscan)
IDX_CONSULTA = 229.18  # indice: total medido en la consulta (plan)
SELECTIVIDADES = (
    0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2,
)


def conexion() -> psycopg.Connection:
    """abre la conexion a postgres leyendo el entorno.

    las variables por defecto coinciden con infra/docker-compose.yml;
    se sobreescriben con PG* si el servicio vive en otro sitio.

    returns:
        conexion en autocommit, lista para ddl y consultas.
    """
    import psycopg  # perezoso: --curva no necesita el driver de BD

    conn = psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "libro"),
        user=os.environ.get("PGUSER", "libro"),
        password=os.environ.get("PGPASSWORD", "libro"),
        autocommit=True,
    )
    return conn


def anunciar_recursos() -> None:
    """declara, antes de actuar, que necesita el modulo."""
    print("=" * 64)
    print("cap. 1 — sgbd basico")
    print("recursos: postgres (servicio docker) · cpu. no usa gpu.")
    print(f"se insertaran {N_FILAS} filas de juguete.")
    print("=" * 64)


def confirmar(accion: str, umbral: int = 50_000) -> None:
    """pide confirmacion antes de una operacion costosa.

    en el cap. 1 la insercion es ligera; el guardia existe para fijar
    la convencion "avisar antes de uso" que los capitulos con gpu
    heredan. se salta con --si o con CAP_CONFIRM=1.

    args:
        accion: descripcion legible de lo que se va a ejecutar.
        umbral: tamano a partir del cual se considera costoso.
    """
    if N_FILAS < umbral:
        return
    if "--si" in sys.argv or os.environ.get("CAP_CONFIRM") == "1":
        return
    resp = input(f"[aviso] {accion}. continuar? [s/N] ").strip().lower()
    if resp not in ("s", "si", "y", "yes"):
        print("cancelado por el usuario.")
        sys.exit(0)


def progreso(hechas: int, total: int, ancho: int = 32) -> None:
    """barra de progreso en una sola linea para tareas de carga."""
    frac = hechas / total
    lleno = int(frac * ancho)
    barra = "#" * lleno + "-" * (ancho - lleno)
    print(f"\r  [{barra}] {hechas}/{total}", end="", flush=True)
    if hechas == total:
        print()


def crear_esquema(conn: psycopg.Connection) -> None:
    """crea desde cero la tabla de juguete (idempotente)."""
    conn.execute("drop table if exists libro")
    conn.execute(
        """
        create table libro (
            id      integer primary key,
            titulo  text    not null,
            anio    integer not null,
            tema    text    not null
        )
        """
    )


def poblar(conn: psycopg.Connection) -> None:
    """inserta N_FILAS filas con progreso; sin indice secundario."""
    rng = random.Random(SEMILLA)
    lote = 1000
    with conn.cursor() as cur:
        with cur.copy(
            "copy libro (id, titulo, anio, tema) from stdin"
        ) as copy:
            for i in range(1, N_FILAS + 1):
                tema = rng.choice(TEMAS)
                copy.write_row((
                    i,
                    f"obra {i:05d} sobre {tema}",
                    rng.randint(1970, 2025),
                    tema,
                ))
                if i % lote == 0 or i == N_FILAS:
                    progreso(i, N_FILAS)
    # estadisticas frescas para el optimizador
    conn.execute("analyze libro")


def plan(conn: psycopg.Connection, etiqueta: str) -> None:
    """imprime el plan de ejecucion de la misma consulta logica.

    la consulta no cambia entre llamadas; lo que cambia es el camino
    fisico que el optimizador elige: esa es la independencia fisica.

    args:
        conn: conexion abierta.
        etiqueta: rotulo para distinguir antes/despues del indice.
    """
    consulta = (
        "select id, titulo from libro "
        "where tema = %s and anio > %s"
    )
    print(f"\n--- plan de ejecucion: {etiqueta} ---")
    with conn.cursor() as cur:
        cur.execute(
            "explain (analyze, buffers, costs) " + consulta,
            ("logica", 2010),
        )
        for (linea,) in cur.fetchall():
            print("  " + linea)


def muestra_reproducible(
    conn: psycopg.Connection, k: int = 15
) -> list[tuple[Any, ...]]:
    """devuelve k observaciones reproducibles por semilla.

    el muestreo se hace en python con random.Random(SEMILLA): elige k
    ids del rango [1, N_FILAS] y los trae con WHERE id = ANY(...). la
    seleccion es asi reproducible de extremo a extremo, sin depender del
    random() del servidor —que `order by random()` no siembra—. se
    preserva el orden del muestreo (id = any no garantiza el de vuelta).

    args:
        conn: conexion abierta.
        k: numero de observaciones a inspeccionar.

    returns:
        k tuplas (id, titulo, anio, tema), en orden de muestreo.
    """
    rng = random.Random(SEMILLA)
    ids = rng.sample(range(1, N_FILAS + 1), k)
    with conn.cursor() as cur:
        cur.execute(
            "select id, titulo, anio, tema from libro "
            "where id = any(%s)",
            (ids,),
        )
        por_id = {fila[0]: fila for fila in cur.fetchall()}
    return [por_id[i] for i in ids]


def imprimir_tabla(
    filas: Iterable[Sequence[Any]], cabecera: Sequence[str]
) -> None:
    """imprime filas alineadas, sin dependencias externas."""
    filas = list(filas)
    if filas:
        cols = list(zip(cabecera, *filas))
    else:
        cols = [(c,) for c in cabecera]
    anchos = [max(len(str(v)) for v in col) for col in cols]
    sep = "  ".join("-" * w for w in anchos)
    cab = "  ".join(str(c).ljust(w) for c, w in zip(cabecera, anchos))
    print("  " + cab)
    print("  " + sep)
    for fila in filas:
        celdas = (str(v).ljust(w) for v, w in zip(fila, anchos))
        print("  " + "  ".join(celdas))


def filas_consulta() -> int:
    """cuenta las filas que cumplen el filtro de la consulta del cap.

    simula el RNG del generador (misma semilla y mismo orden que
    poblar()) para obtener el conteo de forma determinista, sin tocar
    la base de datos. es la fuente de la selectividad de la consulta.

    returns:
        filas con tema='logica' y anio>2010 (657 para N=20000).
    """
    rng = random.Random(SEMILLA)
    casos = 0
    for _ in range(N_FILAS):
        tema = rng.choice(TEMAS)          # mismo orden que poblar()
        anio = rng.randint(1970, 2025)
        if tema == "logica" and anio > 2010:
            casos += 1
    return casos


def curva_coste(ruta: str | None = None) -> str:
    """tabula el coste estimado de cada camino fisico vs selectividad.

    genera el .dat que alimenta la figura del punto de cruce (fig. 1.3)
    sin tocar la base de datos. UN solo modelo, compartido con el plan
    impreso: el indice se calibra para pasar por el coste que el plan
    reporta (IDX_CONSULTA) en la selectividad real de la consulta, que
    se deriva del generador. el barrido es plano; el cruce, de igualar.

    args:
        ruta: destino del .dat; por defecto data/cap01_coste.dat en el
            repositorio, relativo a este modulo.

    returns:
        la ruta escrita (normalizada).
    """
    if ruta is None:
        aqui = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(aqui, "..", "data", "cap01_coste.dat")
    seq = SEQ_TOTAL
    filas_q = filas_consulta()
    sel_q = filas_q / N_FILAS
    # coste por fila: ancla el modelo al coste real del plan
    idx_por_fila = (IDX_CONSULTA - IDX_ARRANQUE) / filas_q
    cruce = (seq - IDX_ARRANQUE) / (N_FILAS * idx_por_fila)
    # incluir la selectividad de la consulta: la curva pasa por ahi
    sels = sorted(set(SELECTIVIDADES) | {round(sel_q, 5)})
    lineas = [
        "# coste estimado (unidades internas del optimizador) frente a",
        "# la selectividad (fraccion de filas que pasan el filtro).",
        "# modelo anclado a una corrida real (PostgreSQL 16.14): el",
        "# indice pasa por el coste del plan en la consulta. generado",
        "# por curva_coste() en cap01_sgbd_basico.py, sin BD.",
        f"# N={N_FILAS}, sel_consulta={sel_q:.5f}, cruce={cruce:.3f}.",
        "selectividad  seq      idx",
    ]
    for sel in sels:
        idx = IDX_ARRANQUE + sel * N_FILAS * idx_por_fila
        lineas.append(f"{sel:<13.5f} {seq:7.1f} {idx:8.1f}")
    with open(ruta, "w", encoding="utf-8") as fich:
        fich.write("\n".join(lineas) + "\n")
    return os.path.normpath(ruta)


def main() -> None:
    anunciar_recursos()
    confirmar(f"insertar {N_FILAS} filas en postgres")
    inicio = time.perf_counter()
    with conexion() as conn:
        print("\n[1/4] creando esquema...")
        crear_esquema(conn)
        print("[2/4] poblando la tabla...")
        poblar(conn)

        # mismo SELECT, dos caminos fisicos: barrido vs indice.
        print("[3/4] consultando sin indice secundario...")
        plan(conn, "sin indice (barrido secuencial)")
        conn.execute(
            "create index idx_libro_tema_anio "
            "on libro (tema, anio)"
        )
        conn.execute("analyze libro")
        plan(conn, "con indice b-tree (mismo SELECT)")

        print("\n[4/4] demostracion: 15 observaciones reproducibles")
        filas = muestra_reproducible(conn, 15)
        imprimir_tabla(filas, ("id", "titulo", "anio", "tema"))

    # demostracion de figura: la curva coste/selectividad (modelo, sin BD)
    destino = curva_coste()
    print(f"\nfigura 1.3: curva de coste -> {destino}")

    print(f"\nlisto en {time.perf_counter() - inicio:.2f}s")


if __name__ == "__main__":
    # "--curva" regenera solo el .dat de la figura 1.3, sin postgres
    if "--curva" in sys.argv:
        print(f"curva de coste escrita en {curva_coste()}")
    else:
        main()
