"""capitulo 5: transacciones, concurrencia y recuperacion.

reproduce sobre PostgreSQL las anomalias clasicas de la concurrencia
---lectura sucia, lectura no repetible, fantasma y sesgo de escritura---
variando el nivel de aislamiento, y registra en una tabla comparativa
cuales aparecen en cada nivel. PostgreSQL resulta mas estricto que el
estandar ANSI: nunca permite lecturas sucias y su REPEATABLE READ ya
evita los fantasmas. Mide ademas el coste del aislamiento ---la caida de
rendimiento al endurecerlo bajo contencion--- y muestra 15 transacciones
con su desenlace como demostracion.

recurso: el servicio postgres del docker-compose de infra/ (cap. 5 usa
servicio + cpu; sin gpu ni torch). usa dos o mas conexiones a la vez para
interleaver las transacciones. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg

NIVELES = ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"]


def conexion() -> psycopg.Connection:
    """conexion en autocommit; las transacciones se abren a mano con
    BEGIN ... ISOLATION LEVEL para controlar el interleave."""
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "libro"),
        user=os.environ.get("PGUSER", "libro"),
        password=os.environ.get("PGPASSWORD", "libro"),
        autocommit=True,
    )


def reiniciar(conn: psycopg.Connection) -> None:
    """deja la tabla de cuentas en un estado conocido."""
    conn.execute("drop table if exists cuenta")
    conn.execute(
        "create table cuenta (id integer primary key, saldo integer,"
        " de_guardia boolean default true)"
    )
    conn.execute("insert into cuenta values (1, 100, true), (2, 100, true)")


def begin(conn: psycopg.Connection, iso: str) -> None:
    conn.execute(f"begin isolation level {iso}")


def no_repetible(a: psycopg.Connection, b: psycopg.Connection,
                 iso: str) -> bool:
    """lectura no repetible: a lee dos veces; b actualiza en medio."""
    reiniciar(a)
    begin(a, iso)
    v1 = a.execute("select saldo from cuenta where id=1").fetchone()[0]
    begin(b, "READ COMMITTED")
    b.execute("update cuenta set saldo=200 where id=1")
    b.execute("commit")
    v2 = a.execute("select saldo from cuenta where id=1").fetchone()[0]
    a.execute("commit")
    return v1 != v2


def fantasma(a: psycopg.Connection, b: psycopg.Connection,
             iso: str) -> bool:
    """fantasma: a cuenta filas dos veces; b inserta una que cumple."""
    reiniciar(a)
    begin(a, iso)
    n1 = a.execute(
        "select count(*) from cuenta where saldo>=100").fetchone()[0]
    begin(b, "READ COMMITTED")
    b.execute("insert into cuenta values (3, 150, true)")
    b.execute("commit")
    n2 = a.execute(
        "select count(*) from cuenta where saldo>=100").fetchone()[0]
    a.execute("commit")
    return n1 != n2


def sucia(a: psycopg.Connection, b: psycopg.Connection, iso: str) -> bool:
    """lectura sucia: a intenta leer un cambio que b no ha confirmado.
    PostgreSQL no la permite en ningun nivel."""
    reiniciar(a)
    begin(b, "READ COMMITTED")
    b.execute("update cuenta set saldo=999 where id=1")
    begin(a, iso)
    v = a.execute("select saldo from cuenta where id=1").fetchone()[0]
    a.execute("commit")
    b.execute("rollback")
    return v == 999


def sesgo_escritura(a: psycopg.Connection, b: psycopg.Connection,
                    iso: str) -> bool:
    """sesgo de escritura: dos guardias se dan de baja a la vez creyendo
    que el otro cubre; solo SERIALIZABLE lo impide (con un fallo de
    serializacion). devuelve True si la anomalia ocurre."""
    reiniciar(a)
    begin(a, iso)
    begin(b, iso)
    na = a.execute(
        "select count(*) from cuenta where de_guardia").fetchone()[0]
    nb = b.execute(
        "select count(*) from cuenta where de_guardia").fetchone()[0]
    ok = True
    if na >= 2:
        a.execute("update cuenta set de_guardia=false where id=1")
    if nb >= 2:
        b.execute("update cuenta set de_guardia=false where id=2")
    try:
        a.execute("commit")
        b.execute("commit")
    except psycopg.errors.SerializationFailure:
        ok = False  # un commit fallo: la anomalia fue impedida
        for c in (a, b):
            try:
                c.execute("rollback")
            except psycopg.Error:
                pass
    quedan = a.execute(
        "select count(*) from cuenta where de_guardia").fetchone()[0]
    return ok and quedan == 0  # anomalia: nadie de guardia


def matriz_anomalias() -> None:
    """reproduce las cuatro anomalias en los tres niveles y escribe la
    tabla comparativa en un .dat (1 = ocurre, 0 = se evita)."""
    a, b = conexion(), conexion()
    pruebas = [
        ("sucia", sucia),
        ("no_repetible", no_repetible),
        ("fantasma", fantasma),
        ("sesgo_escritura", sesgo_escritura),
    ]
    ruta = os.path.join("data", "cap05_aislamiento.dat")
    os.makedirs("data", exist_ok=True)
    print("\nmatriz de anomalias (1 = ocurre, 0 = se evita):")
    cab = "nivel  " + "  ".join(n for n, _ in pruebas)
    print("  " + cab)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("# 1 = anomalia ocurre, 0 = se evita\n")
        fh.write("nivel  " + "  ".join(n for n, _ in pruebas) + "\n")
        for iso in NIVELES:
            fila = [int(fn(a, b, iso)) for _, fn in pruebas]
            etiqueta = iso.replace(" ", "_")
            print(f"  {etiqueta:16} " + "  ".join(str(x) for x in fila))
            fh.write(f"{etiqueta}  " + "  ".join(str(x) for x in fila) + "\n")
    a.close()
    b.close()
    print(f"matriz escrita en {ruta}")


def incrementar(iso: str, vueltas: int) -> int:
    """incrementa el saldo de la cuenta 1 'vueltas' veces, reintentando
    ante fallos de serializacion. devuelve el numero de reintentos."""
    conn = conexion()
    reintentos = 0
    hechas = 0
    while hechas < vueltas:
        try:
            begin(conn, iso)
            s = conn.execute(
                "select saldo from cuenta where id=1").fetchone()[0]
            conn.execute(
                "update cuenta set saldo=%s where id=1", (s + 1,))
            conn.execute("commit")
            hechas += 1
        except psycopg.errors.SerializationFailure:
            conn.execute("rollback")
            reintentos += 1
    conn.close()
    return reintentos


def medir_coste(
    hilos: int = 4, vueltas: int = 200, reps: int = 5
) -> None:
    """mide rendimiento y reintentos bajo contencion en cada nivel.

    repite cada nivel 'reps' veces y toma la mediana, para que un pico
    del planificador no domine. es categoria B (cifras de reloj, no
    reproducibles byte a byte; el entorno se declara en el capitulo).
    """
    base = conexion()
    ruta = os.path.join("data", "cap05_coste.dat")
    filas = []
    for i, iso in enumerate(NIVELES, start=1):
        tputs, reints = [], []
        for _ in range(reps):
            reiniciar(base)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=hilos) as ex:
                r = sum(ex.map(
                    lambda _: incrementar(iso, vueltas), range(hilos)))
            dt = time.perf_counter() - t0
            tputs.append((hilos * vueltas) / dt)
            reints.append(r)
        tputs.sort()
        reints.sort()
        tput = round(tputs[len(tputs) // 2], 1)  # mediana
        reint = reints[len(reints) // 2]
        filas.append((i, iso, tput, reint))
        print(f"  {iso:16} {tput:8.1f} tx/s  {reint} reintentos")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("# coste del aislamiento bajo contencion (medianas)\n")
        fh.write("nivel  throughput  reintentos\n")
        for i, _iso, tput, reint in filas:
            fh.write(f"{i}  {tput}  {reint}\n")
    base.close()
    print(f"coste escrito en {ruta}")


def demostracion(k: int = 15) -> None:
    """15 transferencias bajo contencion real, con su desenlace.

    cada transferencia lee la cuenta y, antes de escribirla, un escritor
    rival concurrente (autocommit) modifica la misma fila. bajo READ
    COMMITTED la escritura se reaplica sobre el valor nuevo y confirma;
    bajo los niveles fuertes (repeatable read, serializable) el motor la
    aborta con un fallo de serializacion. el desenlace lo fija el nivel,
    asi que es reproducible (categoria B: la salida es de una corrida
    real, pero su forma no depende del azar del planificador).
    """
    principal = conexion()
    rival = conexion()
    reiniciar(principal)
    print(f"\ndemostracion: {k} transacciones con su desenlace")
    print("  n   nivel             desenlace")
    print("  --  ----------------  ---------")
    for i in range(k):
        iso = NIVELES[i % len(NIVELES)]
        try:
            begin(principal, iso)
            principal.execute(
                "select saldo from cuenta where id=1").fetchone()
            # rival concurrente: cambia la misma cuenta y confirma
            rival.execute("update cuenta set saldo=saldo+1 where id=1")
            principal.execute(
                "update cuenta set saldo=saldo-1 where id=1")
            principal.execute(
                "update cuenta set saldo=saldo+1 where id=2")
            principal.execute("commit")
            estado = "COMMIT"
        except psycopg.errors.SerializationFailure:
            principal.execute("rollback")
            estado = "ROLLBACK"
        print(f"  {i:<2}  {iso:16}  {estado}")
    principal.close()
    rival.close()


def main() -> None:
    print("=" * 64)
    print("cap. 5: transacciones, concurrencia y recuperacion")
    print("recursos: postgres (servicio docker) · cpu. no usa gpu.")
    print("=" * 64)
    matriz_anomalias()
    print("\ncoste del aislamiento bajo contencion:")
    medir_coste()
    demostracion(15)


if __name__ == "__main__":
    main()
