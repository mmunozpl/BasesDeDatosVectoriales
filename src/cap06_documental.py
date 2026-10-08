"""capitulo 6: limites del relacional y el movimiento NoSQL.

modela el mismo dominio ---autores con sus libros--- de dos maneras: en un
esquema relacional normalizado (PostgreSQL, dos tablas que se reunen) y en
documentos embebidos (MongoDB, un documento por autor que lleva dentro sus
libros). Mide la latencia de la misma consulta logica ---traer un autor con
todos sus libros, el agregado completo--- en ambos modelos y al crecer el
numero de libros por autor, para hacer ver el compromiso: el documento
embebido sirve el agregado de una lectura, sin reunion; el relacional reune,
pero responde mejor a las consultas que cruzan todos los autores. Tambien
muestra 15 documentos al azar como demostracion.

recurso: los servicios postgres y mongo del docker-compose de infra/ (cap. 6
usa servicio + cpu; sin gpu ni torch). ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
import time

import psycopg
from pymongo import MongoClient

PAISES = ["es", "us", "de", "fr", "it"]
TAMANOS = [1, 10, 100, 1000]  # libros por autor que se prueban


def conexion_pg() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "libro"),
        user=os.environ.get("PGUSER", "libro"),
        password=os.environ.get("PGPASSWORD", "libro"),
        autocommit=True,
    )


def cliente_mongo() -> MongoClient:
    uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri)


def poblar_relacional(conn: psycopg.Connection, k: int) -> None:
    """un autor con k libros, en dos tablas normalizadas."""
    conn.execute("drop table if exists libro")
    conn.execute("drop table if exists autor")
    conn.execute(
        "create table autor (id integer primary key, nombre text,"
        " pais text)")
    conn.execute(
        "create table libro (id integer primary key, titulo text,"
        " id_autor integer references autor(id))")
    conn.execute(
        "insert into autor values (1, 'autor-1', 'es')")
    with conn.cursor() as cur:
        cur.executemany(
            "insert into libro values (%s, %s, 1)",
            [(i, f"obra {i:04d}") for i in range(k)])
    conn.execute("create index on libro(id_autor)")


def poblar_documento(col, k: int) -> None:
    """el mismo autor con k libros, embebidos en un solo documento."""
    col.drop()
    doc = {
        "_id": 1,
        "nombre": "autor-1",
        "pais": "es",
        "libros": [{"id": i, "titulo": f"obra {i:04d}"}
                   for i in range(k)],
    }
    col.insert_one(doc)


def medir(fn, repeticiones: int = 200) -> float:
    """mediana en milisegundos de repetir la consulta (con calentamiento)."""
    fn()  # calentamiento: no se mide la primera
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    tiempos.sort()
    return tiempos[len(tiempos) // 2]


def comparar() -> None:
    """latencia del agregado (autor + sus libros) en ambos modelos."""
    conn = conexion_pg()
    cli = cliente_mongo()
    col = cli[os.environ.get("MONGO_DB", "libro")]["autor"]
    ruta = os.path.join("data", "cap06_latencia.dat")
    os.makedirs("data", exist_ok=True)
    filas = []
    print("\nlatencia del agregado 'autor + sus libros' (mediana, ms):")
    print("  k       relacional  documento")
    for k in TAMANOS:
        poblar_relacional(conn, k)
        poblar_documento(col, k)

        def join():  # reune autor con todos sus libros
            conn.execute(
                "select a.nombre, l.titulo from autor a"
                " join libro l on l.id_autor = a.id"
                " where a.id = 1").fetchall()

        def lectura():  # trae el documento entero de una
            col.find_one({"_id": 1})

        rel = medir(join)
        doc = medir(lectura)
        filas.append((k, rel, doc))
        print(f"  {k:<6}  {rel:9.2f}   {doc:8.2f}")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("# latencia del agregado completo (mediana, ms)\n")
        fh.write("k  relacional  documento\n")
        for k, rel, doc in filas:
            fh.write(f"{k}  {round(rel, 3)}  {round(doc, 3)}\n")
    conn.close()
    cli.close()
    print(f"latencias escritas en {ruta}")


def poblar_cruzada_pg(conn: psycopg.Connection, m: int) -> None:
    """m autores con un libro cada uno, repartidos entre paises.

    carga con COPY (a escala de millones, executemany es lento) e indexa
    la columna de reunion, para un relacional bien indexado.
    """
    conn.execute("drop table if exists libro")
    conn.execute("drop table if exists autor")
    conn.execute(
        "create table autor (id integer primary key, pais text)")
    conn.execute(
        "create table libro (id integer primary key,"
        " id_autor integer references autor(id))")
    with conn.cursor() as cur:
        with cur.copy("copy autor (id, pais) from stdin") as cp:
            for i in range(m):
                cp.write_row((i, PAISES[i % len(PAISES)]))
        with cur.copy("copy libro (id, id_autor) from stdin") as cp:
            for i in range(m):
                cp.write_row((i, i))
    conn.execute("create index on libro(id_autor)")
    conn.execute("analyze")


def poblar_cruzada_doc(col, m: int, lote: int = 50000) -> None:
    """m documentos, insertados por lotes para no construir una lista de
    millones en memoria de una vez."""
    col.drop()
    bloque = []
    for i in range(m):
        bloque.append({"_id": i, "pais": PAISES[i % len(PAISES)],
                       "libros": [{"id": i}]})
        if len(bloque) >= lote:
            col.insert_many(bloque)
            bloque = []
    if bloque:
        col.insert_many(bloque)


def comparar_cruzada() -> None:
    """consulta que cruza todos los autores: libros por pais. aqui el
    relacional (GROUP BY con indice) gana al documento (recorrer todo)."""
    conn = conexion_pg()
    cli = cliente_mongo()
    col = cli[os.environ.get("MONGO_DB", "libro")]["autor"]
    ruta = os.path.join("data", "cap06_cruzada.dat")
    filas = []
    print("\nlatencia de 'libros por pais' al crecer los autores (ms):")
    print("  m         relacional  documento")
    for m in (1000, 10000, 100000, 1000000):
        poblar_cruzada_pg(conn, m)
        poblar_cruzada_doc(col, m)

        def group():  # group by con indice, lado relacional
            conn.execute(
                "select a.pais, count(*) from autor a"
                " join libro l on l.id_autor = a.id"
                " group by a.pais").fetchall()

        def recorrer():  # agregacion sobre todos los documentos
            col.aggregate([
                {"$unwind": "$libros"},
                {"$group": {"_id": "$pais",
                            "n": {"$sum": 1}}}]).to_list(None)

        rel = medir(group, repeticiones=20)
        doc = medir(recorrer, repeticiones=20)
        filas.append((m, rel, doc))
        print(f"  {m:<8}  {rel:9.2f}   {doc:8.2f}")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("# latencia de 'libros por pais' (mediana, ms)\n")
        fh.write("m  relacional  documento\n")
        for m, rel, doc in filas:
            fh.write(f"{m}  {round(rel, 3)}  {round(doc, 3)}\n")
    conn.close()
    cli.close()
    print(f"latencias cruzadas escritas en {ruta}")


def demostracion(col, k: int = 15) -> None:
    """15 libros del documento embebido, muestra de su forma anidada."""
    doc = col.find_one({"_id": 1})
    libros = (doc or {}).get("libros", [])[:k]
    print(f"\ndemostracion: {len(libros)} libros embebidos en el documento")
    print("  autor    pais  libro.id  libro.titulo")
    print("  -------  ----  --------  ------------")
    for lb in libros:
        print(f"  autor-1  es    {lb['id']:<8}  {lb['titulo']}")


def main() -> None:
    print("=" * 64)
    print("cap. 6: el mismo dominio: normalizado y embebido")
    print("recursos: postgres + mongo (servicios docker) · cpu. sin gpu.")
    print("=" * 64)
    comparar()
    comparar_cruzada()
    cli = cliente_mongo()
    col = cli[os.environ.get("MONGO_DB", "libro")]["autor"]
    poblar_documento(col, 15)
    demostracion(col, 15)
    cli.close()


if __name__ == "__main__":
    main()
