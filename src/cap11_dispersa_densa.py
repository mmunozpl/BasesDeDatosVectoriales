"""capitulo 11: de la representacion dispersa a la densa.

contrasta, en Python con numpy, las dos familias de representacion vectorial y
mide cinco cosas que explican por que la densa sucede a la dispersa:

  1. ocupacion: el vector one-hot y la bolsa de palabras tienen dimension enorme
     y casi toda nula; el denso tiene pocos cientos de componentes, todas no
     nulas. se mide la fraccion ocupada al crecer el vocabulario.
  2. maldicion de la dimensionalidad: en dimension alta, las distancias se
     concentran y el mas cercano deja de distinguirse del mas lejano. se mide el
     contraste (dmax-dmin)/dmin segun la dimension.
  3. casi-ortogonalidad: dos vectores aleatorios en dimension alta son casi
     perpendiculares; el coseno se concentra en cero. se mide segun la dimension.
  4. similitud capturada: en one-hot dos palabras distintas son ortogonales
     (coseno 0), no hay parecido; un embedding denso por co-ocurrencia acerca las
     palabras del mismo tema. se mide el coseno de pares afines y ajenos.
  5. compresion: unas pocas componentes densas (SVD de la co-ocurrencia) retienen
     la mayor parte de la estructura. se mide la varianza acumulada.

es Python puro con numpy (sin servicio ni torch; la GPU es opcional y aqui no se
usa, segun la tabla de recursos), con semilla fija. ver IMPLEMENTACION.md.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

SEMILLA = 11

# vocabulario por temas: palabras que tienden a coaparecer dentro de su tema.
TEMAS = {
    "tecnologia": ["base", "datos", "consulta", "tabla", "indice",
                   "servidor", "red", "codigo"],
    "cocina": ["receta", "harina", "azucar", "salsa", "horno",
               "tomate", "aceite", "postre"],
    "deporte": ["partido", "equipo", "balon", "gol", "carrera",
                "entrenar", "campo", "atleta"],
    "musica": ["guitarra", "piano", "cancion", "ritmo", "melodia",
               "concierto", "nota", "orquesta"],
}
CONECTORES = ["el", "la", "de", "una", "con", "y"]


def anunciar() -> None:
    print("=" * 64)
    print("cap. 11: de la representacion dispersa a la densa")
    print("recursos: python + numpy · cpu. la gpu es opcional; aqui no.")
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


def generar_corpus(por_tema: int = 80) -> List[List[str]]:
    """frases sinteticas: cada una mezcla palabras de un tema y un conector,
    de modo que las del mismo tema coaparecen y las de temas distintos no."""
    rng = np.random.default_rng(SEMILLA)
    corpus: List[List[str]] = []
    for palabras in TEMAS.values():
        for _ in range(por_tema):
            k = int(rng.integers(4, 7))
            frase = list(rng.choice(palabras, size=k, replace=True))
            frase.append(str(rng.choice(CONECTORES)))
            corpus.append(frase)
    return corpus


def vocabulario(corpus: List[List[str]]) -> List[str]:
    return sorted({p for frase in corpus for p in frase})


def coocurrencia(corpus: List[List[str]],
                 vocab: List[str]) -> np.ndarray:
    """matriz de co-ocurrencia: cuantas veces dos palabras salen en la misma
    frase (ventana = la frase entera)."""
    idx = {p: i for i, p in enumerate(vocab)}
    m = np.zeros((len(vocab), len(vocab)))
    for frase in corpus:
        ids = [idx[p] for p in set(frase)]
        for a in ids:
            for b in ids:
                if a != b:
                    m[a, b] += 1
    return m


def embeddings_densos(cooc: np.ndarray, dim: int = 16
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """vectores densos por SVD de la co-ocurrencia (estilo LSA). devuelve los
    vectores (V x dim) y los valores singulares."""
    m = np.log1p(cooc)                 # amortigua las cuentas grandes
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    return u[:, :dim] * s[:dim], s


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def simular_ocupacion() -> None:
    """fraccion de dimensiones no nulas: one-hot y bolsa de palabras frente al
    denso, al crecer el vocabulario."""
    filas = []
    print("\nocupacion: fraccion de dimensiones no nulas")
    print("  vocab    one-hot     bow(~10)   denso(256)")
    for v in (100, 1000, 10000, 100000):
        onehot = 1 / v                  # una sola componente activa
        bow = min(10, v) / v            # ~10 palabras distintas por doc
        denso = 256 / 256               # todas activas (dim fija 256)
        filas.append((v, round(onehot, 8), round(bow, 8),
                      round(denso, 4)))
        print(f"  {v:<7}  {onehot:.2e}  {bow:.2e}  {denso:.2f}")
    _escribir(os.path.join("data", "cap11_ocupacion.dat"),
              "fraccion de dimensiones no nulas segun el vocabulario",
              "vocab  onehot  bow  denso", filas)


def simular_concentracion(n: int = 1000, trials: int = 50) -> None:
    """maldicion de la dimensionalidad: contraste (dmax-dmin)/dmin entre las
    distancias de un punto al resto, segun la dimension."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nconcentracion de distancias: (dmax-dmin)/dmin segun dimension")
    print("  dim     contraste")
    for d in (2, 5, 10, 50, 100, 500, 1000):
        contrastes = []
        for _ in range(trials):
            pts = rng.random((n, d))
            q = rng.random(d)
            dist = np.linalg.norm(pts - q, axis=1)
            contrastes.append((dist.max() - dist.min()) / dist.min())
        c = float(np.mean(contrastes))
        filas.append((d, round(c, 4)))
        print(f"  {d:<6}  {c:.4f}")
    _escribir(os.path.join("data", "cap11_concentracion.dat"),
              "contraste (dmax-dmin)/dmin de distancias segun dimension",
              "dim  contraste", filas)


def simular_ortogonalidad(trials: int = 2000) -> None:
    """casi-ortogonalidad: coseno medio (en valor absoluto) entre vectores
    aleatorios segun la dimension; se concentra en cero al crecer."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\ncasi-ortogonalidad: |coseno| medio entre vectores aleatorios")
    print("  dim     |coseno|")
    for d in (2, 5, 10, 50, 100, 500, 1000):
        cosenos = []
        for _ in range(trials):
            a, b = rng.standard_normal(d), rng.standard_normal(d)
            cosenos.append(abs(coseno(a, b)))
        c = float(np.mean(cosenos))
        filas.append((d, round(c, 4)))
        print(f"  {d:<6}  {c:.4f}")
    _escribir(os.path.join("data", "cap11_ortogonalidad.dat"),
              "|coseno| medio entre vectores aleatorios segun dimension",
              "dim  coseno_abs", filas)


def simular_similitud() -> None:
    """coseno medio de pares de palabras afines (mismo tema) y ajenas (temas
    distintos), en el embedding denso; en one-hot ambos serian cero."""
    rng = np.random.default_rng(SEMILLA)
    corpus = generar_corpus()
    vocab = vocabulario(corpus)
    idx = {p: i for i, p in enumerate(vocab)}
    emb, _ = embeddings_densos(coocurrencia(corpus, vocab))
    tema_de = {p: t for t, ps in TEMAS.items() for p in ps}
    contenido = [p for p in vocab if p in tema_de]

    def cos_de(a: str, b: str) -> float:
        return coseno(emb[idx[a]], emb[idx[b]])

    afines, ajenas = [], []
    for _ in range(2000):
        a, b = rng.choice(contenido, 2, replace=False)
        (afines if tema_de[a] == tema_de[b] else ajenas).append(cos_de(a, b))
    filas = [("afines_denso", round(float(np.mean(afines)), 4)),
             ("ajenas_denso", round(float(np.mean(ajenas)), 4)),
             ("cualquiera_onehot", 0.0)]
    print("\nsimilitud capturada (coseno medio):")
    for et, v in filas:
        print(f"  {et:18}  {v:.4f}")
    _escribir(os.path.join("data", "cap11_similitud.dat"),
              "coseno medio de pares afines/ajenos (denso) vs one-hot",
              "caso  coseno", filas)


def simular_varianza() -> None:
    """varianza acumulada que retienen las primeras componentes del SVD de la
    co-ocurrencia: unas pocas dimensiones densas capturan casi toda la
    estructura."""
    corpus = generar_corpus()
    vocab = vocabulario(corpus)
    _, s = embeddings_densos(coocurrencia(corpus, vocab))
    var = (s ** 2)
    acum = np.cumsum(var) / var.sum()
    filas = []
    print("\ncompresion: varianza acumulada por numero de componentes")
    print("  comp   varianza_acumulada")
    for c in (1, 2, 4, 8, 16, 32):
        if c <= len(acum):
            filas.append((c, round(float(acum[c - 1]), 4)))
            print(f"  {c:<5}  {acum[c - 1]:.4f}")
    _escribir(os.path.join("data", "cap11_varianza.dat"),
              "varianza acumulada del SVD de la co-ocurrencia por componentes",
              "comp  varianza", filas)


def simular_dimension() -> None:
    """calidad del embedding segun su dimension: fraccion de palabras cuyo
    vecino mas proximo es del mismo tema, al crecer el numero de dimensiones.
    pocas dimensiones no bastan; a partir de unas pocas, la calidad se satura.
    """
    corpus = generar_corpus()
    vocab = vocabulario(corpus)
    idx = {p: i for i, p in enumerate(vocab)}
    cooc = coocurrencia(corpus, vocab)
    tema_de = {p: t for t, ps in TEMAS.items() for p in ps}
    contenido = [p for p in vocab if p in tema_de]
    filas = []
    print("\ncalidad del embedding segun la dimension")
    print("  dim   acierto_vecino")
    for d in (1, 2, 4, 8, 16, 32):
        emb, _ = embeddings_densos(cooc, dim=d)
        aciertos = 0
        for p in contenido:
            mejor, mejorc = None, -1.0
            for q in contenido:
                if q == p:
                    continue
                c = coseno(emb[idx[p]], emb[idx[q]])
                if c > mejorc:
                    mejor, mejorc = q, c
            if tema_de[p] == tema_de[mejor]:
                aciertos += 1
        tasa = aciertos / len(contenido)
        filas.append((d, round(tasa, 4)))
        print(f"  {d:<4}  {tasa:.4f}")
    _escribir(os.path.join("data", "cap11_dimension.dat"),
              "fraccion de palabras con vecino del mismo tema segun la "
              "dimension del embedding",
              "dim  acierto", filas)


def demostracion(k: int = 15) -> None:
    """15 palabras al azar y su vecino mas proximo en el espacio denso."""
    rng = np.random.default_rng(SEMILLA + 1)
    corpus = generar_corpus()
    vocab = vocabulario(corpus)
    idx = {p: i for i, p in enumerate(vocab)}
    emb, _ = embeddings_densos(coocurrencia(corpus, vocab))
    tema_de = {p: t for t, ps in TEMAS.items() for p in ps}
    contenido = [p for p in vocab if p in tema_de]
    muestra = list(rng.choice(contenido, size=k, replace=False))
    print(f"\ndemostracion: {k} palabras y su vecino mas proximo (denso)")
    print("  palabra     vecino      coseno  mismo tema?")
    print("  ----------  ----------  ------  -----------")
    for p in muestra:
        mejor, mejorc = None, -1.0
        for q in contenido:
            if q == p:
                continue
            c = coseno(emb[idx[p]], emb[idx[q]])
            if c > mejorc:
                mejor, mejorc = q, c
        igual = "si" if tema_de[p] == tema_de[mejor] else "no"
        print(f"  {p:10}  {mejor:10}  {mejorc:.3f}   {igual}")


def main() -> None:
    anunciar()
    simular_ocupacion()
    simular_concentracion()
    simular_ortogonalidad()
    simular_similitud()
    simular_varianza()
    simular_dimension()
    demostracion(15)


if __name__ == "__main__":
    main()
