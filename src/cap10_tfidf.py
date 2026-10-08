"""capitulo 10: recuperacion de informacion clasica.

construye, en Python puro, un sistema de recuperacion documental completo sobre
un corpus sintetico reducido, escrito a mano (indice invertido, modelo
booleano, modelo de espacio vectorial con TF-IDF y similitud coseno, y
medidas de evaluacion) y
mide cuatro cosas que muestran por que el modelo vectorial sucede al booleano:

  1. booleano: al encadenar terminos con AND el resultado se desploma (a menudo
     a cero) y con OR se dispara; el modelo booleano no gradua.
  2. IDF: el peso de un termino cae con el numero de documentos que lo
     contienen; lo raro informa, lo comun no.
  3. precision-recall: la curva PR del ranking por TF-IDF frente al de
     frecuencia bruta, sobre una consulta con relevancia conocida.
  4. precision@k: la precision en los primeros k resultados, TF-IDF frente a
     frecuencia bruta.

es Python puro (sin servicio ni torch): basta el interprete, con semilla
fija para la muestra de la demostracion.
"""

from __future__ import annotations

import math
import os
import random
from collections import Counter
from typing import Dict, List, Set, Tuple

SEMILLA = 10

# corpus reducido: 8 documentos de bases de datos (relevantes para la consulta)
# y 18 de otros temas (cocina, astronomia, musica), con palabras comunes que
# hacen interesante la ponderacion.
CORPUS = [
    "una base de datos relacional organiza la informacion en tablas",
    "el lenguaje sql permite consultar una base de datos",
    "un indice acelera la consulta de una tabla muy grande",
    "la clave primaria identifica cada fila de la tabla",
    "una consulta sql selecciona filas de varias tablas",
    "el modelo relacional de codd define tablas y relaciones",
    "normalizar una base de datos elimina la redundancia de datos",
    "una transaccion agrupa varias operaciones de la base de datos",
    "la receta lleva harina huevos azucar y un poco de sal",
    "para el pan se mezcla harina agua y levadura",
    "el sofrito empieza con cebolla y aceite de oliva",
    "la tarta se hornea en el horno durante cuarenta minutos",
    "una buena salsa lleva tomate ajo y aceite",
    "el postre se sirve frio y con mucho azucar",
    "las estrellas brillan en el cielo oscuro de la noche",
    "un planeta orbita alrededor de una estrella lejana",
    "la galaxia contiene muchos millones de estrellas",
    "el telescopio observa los planetas mas lejanos",
    "la luna es el satelite natural de la tierra",
    "un cometa cruza el cielo cada cierto numero de anos",
    "la guitarra tiene seis cuerdas de metal",
    "el piano produce sonido al pulsar las teclas",
    "una cancion combina una melodia y un ritmo",
    "la orquesta reune muchos instrumentos de viento y cuerda",
    "la tabla periodica ordena los elementos quimicos por numero",
    "puedes consultar tu horoscopo en la web cada manana",
]
RELEVANTES: Set[int] = set(range(8))   # los 8 de bases de datos
CONSULTA = "consulta sql sobre una base de datos con tablas e indices"


def anunciar() -> None:
    print("=" * 64)
    print("cap. 10: recuperacion de informacion clasica")
    print("recursos: python puro · cpu. no usa servicio, gpu ni torch.")
    print(f"corpus de {len(CORPUS)} documentos, semilla = {SEMILLA}")
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


def tokenizar(texto: str) -> List[str]:
    return texto.lower().split()


def indice_invertido(corpus: List[str]) -> Dict[str, Set[int]]:
    """termino -> conjunto de documentos que lo contienen."""
    idx: Dict[str, Set[int]] = {}
    for d, texto in enumerate(corpus):
        for t in set(tokenizar(texto)):
            idx.setdefault(t, set()).add(d)
    return idx


def idf(corpus: List[str]) -> Dict[str, float]:
    """idf(t) = log(N / df(t)): pesa lo raro por encima de lo comun."""
    n = len(corpus)
    idx = indice_invertido(corpus)
    return {t: math.log(n / len(docs)) for t, docs in idx.items()}


def vector_tfidf(texto: str, pesos: Dict[str, float],
                 usar_idf: bool = True) -> Dict[str, float]:
    """vector disperso del texto: tf * idf (o solo tf si usar_idf=False)."""
    tf = Counter(tokenizar(texto))
    if usar_idf:
        return {t: c * pesos.get(t, 0.0) for t, c in tf.items()}
    return {t: float(c) for t, c in tf.items()}


def coseno(a: Dict[str, float], b: Dict[str, float]) -> float:
    """similitud coseno entre dos vectores dispersos."""
    comunes = set(a) & set(b)
    num = sum(a[t] * b[t] for t in comunes)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def ranking(consulta: str, corpus: List[str], pesos: Dict[str, float],
            usar_idf: bool) -> List[int]:
    """documentos ordenados por similitud coseno con la consulta."""
    q = vector_tfidf(consulta, pesos, usar_idf)
    sims = [(d, coseno(q, vector_tfidf(texto, pesos, usar_idf)))
            for d, texto in enumerate(corpus)]
    sims.sort(key=lambda x: -x[1])
    return [d for d, s in sims if s > 0]


def simular_booleano() -> None:
    """tamano del resultado al encadenar terminos con AND y con OR."""
    idx = indice_invertido(CORPUS)
    terminos = ["datos", "base", "relacional", "consulta", "sql"]
    filas = []
    print("\nbooleano: tamano del resultado segun el numero de terminos")
    print("  k  AND  OR")
    acc_and: Set[int] = set(range(len(CORPUS)))
    acc_or: Set[int] = set()
    for k, t in enumerate(terminos, start=1):
        docs = idx.get(t, set())
        acc_and &= docs
        acc_or |= docs
        filas.append((k, len(acc_and), len(acc_or)))
        print(f"  {k}  {len(acc_and):<3}  {len(acc_or)}")
    _escribir(os.path.join("data", "cap10_booleano.dat"),
              "tamano del resultado booleano segun k terminos (AND/OR)",
              "k  and  or", filas)


def simular_idf() -> None:
    """peso idf en funcion del numero de documentos que contienen el termino."""
    n = len(CORPUS)
    idx = indice_invertido(CORPUS)
    por_df: Dict[int, float] = {}
    for t, docs in idx.items():
        df = len(docs)
        por_df[df] = math.log(n / df)
    filas = [(df, round(por_df[df], 4)) for df in sorted(por_df)]
    print("\nidf: peso segun frecuencia documental df")
    for df, v in filas:
        print(f"  df={df:<2}  idf={v:.4f}")
    _escribir(os.path.join("data", "cap10_idf.dat"),
              "idf = log(N/df) segun frecuencia documental (N=%d)" % n,
              "df  idf", filas)


def _pr_interpolada(rank: List[int],
                    relevantes: Set[int]) -> List[float]:
    """precision interpolada en 11 puntos de recall (0.0 .. 1.0)."""
    total = len(relevantes)
    hits = 0
    puntos = []
    for i, d in enumerate(rank, start=1):
        if d in relevantes:
            hits += 1
            puntos.append((hits / total, hits / i))
    interp = []
    for j in range(11):
        r = j / 10
        ps = [p for (rec, p) in puntos if rec >= r]
        interp.append(round(max(ps) if ps else 0.0, 4))
    return interp


def _ap(rank: List[int], relevantes: Set[int]) -> float:
    """precision media (average precision) de un ranking."""
    total = len(relevantes)
    hits = 0
    suma = 0.0
    for i, d in enumerate(rank, start=1):
        if d in relevantes:
            hits += 1
            suma += hits / i
    return suma / total if total else 0.0


def simular_pr() -> None:
    """curva precision-recall del ranking TF-IDF frente al de frecuencia."""
    pesos = idf(CORPUS)
    r_tfidf = ranking(CONSULTA, CORPUS, pesos, usar_idf=True)
    r_tf = ranking(CONSULTA, CORPUS, pesos, usar_idf=False)
    p_tfidf = _pr_interpolada(r_tfidf, RELEVANTES)
    p_tf = _pr_interpolada(r_tf, RELEVANTES)
    filas = [(round(j / 10, 1), p_tfidf[j], p_tf[j]) for j in range(11)]
    print("\nprecision-recall (interpolada): TF-IDF vs frecuencia bruta")
    print("  recall  tfidf   tf")
    for rec, a, b in filas:
        print(f"  {rec:.1f}     {a:.3f}  {b:.3f}")
    print(f"  AP TF-IDF = {_ap(r_tfidf, RELEVANTES):.4f} ; "
          f"AP TF = {_ap(r_tf, RELEVANTES):.4f}")
    _escribir(os.path.join("data", "cap10_pr.dat"),
              "precision interpolada vs recall: TF-IDF vs frecuencia bruta",
              "recall  tfidf  tf", filas)


def simular_pk() -> None:
    """precision@k del ranking TF-IDF frente al de frecuencia bruta."""
    pesos = idf(CORPUS)
    r_tfidf = ranking(CONSULTA, CORPUS, pesos, usar_idf=True)
    r_tf = ranking(CONSULTA, CORPUS, pesos, usar_idf=False)

    def p_at_k(rank: List[int], k: int) -> float:
        top = rank[:k]
        return sum(1 for d in top if d in RELEVANTES) / k if top else 0.0

    filas = []
    print("\nprecision@k: TF-IDF vs frecuencia bruta")
    print("  k   tfidf   tf")
    for k in range(1, 11):
        a = round(p_at_k(r_tfidf, k), 4)
        b = round(p_at_k(r_tf, k), 4)
        filas.append((k, a, b))
        print(f"  {k:<2}  {a:.3f}  {b:.3f}")
    _escribir(os.path.join("data", "cap10_pk.dat"),
              "precision@k: TF-IDF vs frecuencia bruta",
              "k  tfidf  tf", filas)


VACIAS = {"de", "la", "el", "una", "un", "en", "y", "con", "los",
          "las", "e", "muy", "mas", "del", "se", "al", "por", "sobre"}


def _stem(p: str) -> str:
    """stemming minimo: recorta plurales frecuentes."""
    for suf in ("es", "s"):
        if p.endswith(suf) and len(p) > 4:
            return p[:-len(suf)]
    return p


def _normalizar(texto: str, modo: str) -> List[str]:
    """tokeniza segun el modo: exacto, sin vacias, o con stemming."""
    palabras = texto.lower().split()
    if modo == "exacto":
        return palabras
    palabras = [p for p in palabras if p not in VACIAS]
    if modo == "stem":
        palabras = [_stem(p) for p in palabras]
    return palabras


def simular_normalizacion() -> None:
    """precision media segun el preprocesado del texto: exacto, sin palabras
    vacias y con stemming. el stemming ataca parte de la discordancia de
    terminos (plurales), pero los sinonimos siguen fuera de su alcance.
    """
    n = len(CORPUS)
    modos = [("exacto", 1), ("sin_vacias", 2), ("stem", 3)]
    filas = []
    print("\nnormalizacion: precision media (AP) segun el preprocesado")
    print("  paso  modo        AP")
    for modo, orden in modos:
        docs_tok = [_normalizar(t, modo) for t in CORPUS]
        df: Dict[str, int] = {}
        for toks in docs_tok:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        pesos = {t: math.log(n / c) for t, c in df.items()}

        def vec(toks: List[str]) -> Dict[str, float]:
            tf = Counter(toks)
            return {t: c * pesos.get(t, 0.0) for t, c in tf.items()}

        q = vec(_normalizar(CONSULTA, modo))
        sims = [(d, coseno(q, vec(toks)))
                for d, toks in enumerate(docs_tok)]
        sims.sort(key=lambda x: -x[1])
        rank = [d for d, s in sims if s > 0]
        ap = _ap(rank, RELEVANTES)
        filas.append((orden, modo, round(ap, 4)))
        print(f"  {orden}     {modo:11} {ap:.4f}")
    _escribir(os.path.join("data", "cap10_normalizacion.dat"),
              "precision media (AP) segun el preprocesado del texto",
              "paso  modo  ap", filas)


def demostracion(k: int = 15) -> None:
    """15 documentos al azar con su vector disperso TF-IDF (top 3 terminos)."""
    rng = random.Random(SEMILLA)
    pesos = idf(CORPUS)
    muestra = rng.sample(range(len(CORPUS)), k)
    print(f"\ndemostracion: {k} documentos y sus 3 terminos de mayor TF-IDF")
    print("  doc  terminos principales (peso)")
    print("  ---  -----------------------------")
    for d in sorted(muestra):
        v = vector_tfidf(CORPUS[d], pesos, usar_idf=True)
        top = sorted(v.items(), key=lambda kv: -kv[1])[:3]
        frags = ", ".join(f"{t}={p:.2f}" for t, p in top)
        print(f"  {d:<3}  {frags}")


def main() -> None:
    anunciar()
    simular_booleano()
    simular_idf()
    simular_pr()
    simular_pk()
    simular_normalizacion()
    demostracion(15)


if __name__ == "__main__":
    main()
