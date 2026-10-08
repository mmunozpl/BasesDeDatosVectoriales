"""capitulo 3: SQL: del DML declarativo a las extensiones.

ejecuta sobre PostgreSQL una consulta con funcion de ventana (suma
acumulada por cliente, ordenada por tiempo) y muestra su plan con
EXPLAIN, con y sin un indice B-tree sobre (cliente, ts). con el indice
el motor lee en orden y el nodo Sort desaparece del plan: WindowAgg ->
Index Scan en vez de WindowAgg -> Sort -> Seq Scan. es un cambio
estructural verdadero, visible en el plan; pero no garantiza menor
latencia: con la tabla en memoria, ordenar es barato. imprime ademas 15
filas como demostracion reproducible.

recurso: el servicio postgres del docker-compose de infra/ (cap. 3 usa
servicio + cpu; no usa gpu ni torch). antes de
poblar la tabla avisa y pide confirmacion (--si o CAP_CONFIRM=1).
"""

from __future__ import annotations

import os
import random
import sys

import psycopg

SEMILLA = 20240603
N_DEMO = 200000  # tamano de la tabla para el plan y la demostracion


def conexion() -> psycopg.Connection:
    """abre la conexion a postgres leyendo el entorno (valores por
    defecto los de infra/docker-compose.yml)."""
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "libro"),
        user=os.environ.get("PGUSER", "libro"),
        password=os.environ.get("PGPASSWORD", "libro"),
        autocommit=True,
    )


def anunciar() -> None:
    print("=" * 64)
    print("cap. 3: SQL: funciones de ventana e indice")
    print("recursos: postgres (servicio docker) · cpu. no usa gpu.")
    print(f"tabla de demostracion: {N_DEMO} filas")
    print("=" * 64)


def confirmar(n: int, umbral: int = 100_000) -> None:
    """pide confirmacion si la tabla de demostracion es grande."""
    if n < umbral:
        return
    if "--si" in sys.argv or os.environ.get("CAP_CONFIRM") == "1":
        return
    resp = input(
        f"[aviso] se poblara una tabla de {n} filas. continuar? [s/N] "
    ).strip().lower()
    if resp not in ("s", "si", "y", "yes"):
        print("cancelado por el usuario.")
        sys.exit(0)


def progreso(hechas: int, total: int, ancho: int = 32) -> None:
    frac = hechas / total
    lleno = int(frac * ancho)
    barra = "#" * lleno + "-" * (ancho - lleno)
    print(f"\r  [{barra}] {hechas}/{total}", end="", flush=True)
    if hechas == total:
        print()


def crear_y_poblar(conn: psycopg.Connection, n: int) -> None:
    """crea desde cero la tabla evento y la puebla con n filas.

    cliente agrupa en ~n/50 grupos; ts es un entero (orden temporal);
    importe es el valor que la ventana acumula.
    """
    rng = random.Random(SEMILLA)
    conn.execute("drop table if exists evento")
    conn.execute(
        """
        create table evento (
            id      integer primary key,
            cliente integer not null,
            ts      integer not null,
            importe integer not null
        )
        """
    )
    grupos = max(1, n // 50)
    lote = 5000
    with conn.cursor() as cur:
        with cur.copy(
            "copy evento (id, cliente, ts, importe) from stdin"
        ) as cp:
            for i in range(1, n + 1):
                cp.write_row((
                    i,
                    rng.randint(0, grupos - 1),
                    rng.randint(0, 10**9),
                    rng.randint(1, 1000),
                ))
                if i % lote == 0 or i == n:
                    progreso(i, n)
    conn.execute("analyze evento")


# la consulta de ventana: suma acumulada del importe por cliente, en
# orden temporal. la columna acum obliga al motor a computar la ventana
# (a diferencia de envolverla en count(*), que el optimizador elide).
VENTANA = """
    select cliente, importe,
           sum(importe) over (partition by cliente order by ts) as acum
    from evento
"""


def demostrar_planes(conn: psycopg.Connection) -> None:
    """imprime el plan de la ventana sin y con el indice (EXPLAIN).

    sin indice el motor ordena (WindowAgg -> Sort -> Seq Scan); con el
    indice (cliente, ts) lee en orden y el Sort desaparece (WindowAgg ->
    Index Scan). el cambio es estructural; que ademas sea mas rapido
    depende de si el Sort dominaba el coste (en memoria no lo hace).
    """
    with conn.cursor() as cur:
        cur.execute("drop index if exists idx_evento_cli_ts")
        cur.execute("analyze evento")
        print("\n--- plan SIN indice: hay que ordenar ---")
        cur.execute("explain (costs off) " + VENTANA)
        for (linea,) in cur.fetchall():
            print("  " + linea)
        cur.execute(
            "create index idx_evento_cli_ts on evento (cliente, ts)"
        )
        cur.execute("analyze evento")
        print("\n--- plan CON indice (cliente, ts): sin Sort ---")
        cur.execute("explain (costs off) " + VENTANA)
        for (linea,) in cur.fetchall():
            print("  " + linea)


# para la demostracion: la misma ventana mas el id (para muestrear) y el
# numero de orden dentro del cliente (segunda funcion de ventana).
DEMOSTRACION_VENTANA = """
    select id, cliente, importe,
           sum(importe) over w as acum,
           row_number() over w as orden
    from evento
    window w as (partition by cliente order by ts)
"""


def muestra_reproducible(conn: psycopg.Connection, k: int = 15) -> None:
    """k filas reproducibles del resultado de la ventana (demostracion).

    elige k id con random.Random(SEMILLA) y los trae con WHERE id = ANY,
    preservando el orden de muestreo; no usa `order by random()`, y asi
    es reproducible de extremo a extremo. la ventana se evalua sobre
    toda la tabla y solo despues se filtra a las k filas elegidas.
    """
    rng = random.Random(SEMILLA)
    ids = rng.sample(range(1, N_DEMO + 1), k)
    consulta = (
        "select id, cliente, importe, acum, orden from ("
        + DEMOSTRACION_VENTANA + ") q where id = any(%s)"
    )
    with conn.cursor() as cur:
        cur.execute(consulta, (ids,))
        por_id = {f[0]: f for f in cur.fetchall()}
    filas = [por_id[i] for i in ids]
    print(f"\ndemostracion: {k} filas reproducibles de la ventana")
    cab = ("id", "cliente", "importe", "acum", "orden")
    anchos = [max(len(str(v)) for v in col) for col in zip(cab, *filas)]

    def linea(row: tuple) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(row, anchos))

    print("  " + linea(cab))
    print("  " + "  ".join("-" * w for w in anchos))
    for row in filas:
        print("  " + linea(row))


def main() -> None:
    anunciar()
    confirmar(N_DEMO)
    with conexion() as conn:
        print("\npoblando la tabla...")
        crear_y_poblar(conn, N_DEMO)
        demostrar_planes(conn)
        muestra_reproducible(conn, 15)


if __name__ == "__main__":
    main()
