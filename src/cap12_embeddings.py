"""capitulo 12: deep learning para embeddings.

implementa desde cero, en numpy, las ideas centrales del aprendizaje de
embeddings, y las mide. no usa modelos preentrenados ni GPU (se ejecuta en
CPU en segundos), porque el objetivo es entender los mecanismos, no reproducir
un modelo industrial:

  1. skip-gram con muestreo negativo (SGNS), version didactica de word2vec:
     aprende vectores de palabra prediciendo el contexto. se registran por
     epoca la perdida media y el acierto de tema, medido aparte.
  2. comparacion: la calidad del embedding aprendido frente al aleatorio y al de
     co-ocurrencia+SVD del capitulo 11.
  3. self-attention minima (Q = K = V, sin proyecciones): se mide que la salida
     de una misma palabra cambia con los vectores de su contexto, frente al
     vector unico estatico.
  4. maqueta inspirada en Matryoshka: la perdida promedia la de varios
     prefijos; se mide la calidad al truncar, frente a un embedding estandar.
  5. interaccion tardia (MaxSim): comparar por el mejor emparejamiento de tokens
     frente al vector unico promediado.

los modelos reales (Sentence-BERT, ViT, CLIP, BGE-M3) se descargan con sus
dependencias; entrenarlos pide aceleradores, y en inferencia muchos corren en
CPU, mas despacio. este modulo se queda en los mecanismos, en CPU. semilla
fija.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

SEMILLA = 12
DIM = 16

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
    print("cap. 12: deep learning para embeddings (desde cero, numpy)")
    print("recursos: python + numpy · cpu. los modelos reales (SBERT,")
    print("ViT, CLIP) no se usan; aqui solo los mecanismos.")
    print(f"semilla = {SEMILLA}, dimension = {DIM}")
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
    """frases de palabras del mismo tema (sin conectores compartidos, para que
    la senal de co-ocurrencia sea limpia y el aprendizaje, nitido)."""
    rng = np.random.default_rng(SEMILLA)
    corpus = []
    for palabras in TEMAS.values():
        for _ in range(por_tema):
            k = int(rng.integers(4, 7))
            corpus.append(list(rng.choice(palabras, size=k, replace=True)))
    return corpus


def vocabulario(corpus: List[List[str]]) -> List[str]:
    return sorted({p for f in corpus for p in f})


def sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _pares(corpus: List[List[str]], idx: Dict[str, int],
           ventana: int) -> np.ndarray:
    pares = []
    for f in corpus:
        ids = [idx[w] for w in f]
        for i, c in enumerate(ids):
            lo, hi = max(0, i - ventana), min(len(ids), i + ventana + 1)
            for j in range(lo, hi):
                if j != i:
                    pares.append((c, ids[j]))
    return np.array(pares)


def _acierto_tema(W: np.ndarray, vocab: List[str], dims: int = None
                  ) -> float:
    """fraccion de palabras de contenido cuyo vecino mas proximo (por coseno)
    es del mismo tema."""
    idx = {p: i for i, p in enumerate(vocab)}
    tema = {p: t for t, ps in TEMAS.items() for p in ps}
    cont = [p for p in vocab if p in tema]
    M = W[:, :dims] if dims else W
    aciertos = 0
    for p in cont:
        mejor, mc = None, -2.0
        for q in cont:
            if q == p:
                continue
            c = coseno(M[idx[p]], M[idx[q]])
            if c > mc:
                mejor, mc = q, c
        aciertos += int(tema[p] == tema[mejor])
    return aciertos / len(cont)


def entrenar_word2vec(corpus, vocab, dim=DIM, ventana=2, negativos=5,
                      epocas=30, lr=0.02, prefijos=None
                      ) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """skip-gram con muestreo negativo. si 'prefijos' es una lista de dims,
    entrena estilo Matryoshka: la perdida suma la de cada prefijo, de modo que
    los primeros componentes bastan por si solos.

    los gradientes se calculan sobre copias de los vectores y se aplican al
    final del par, para no corromper el calculo con vistas que se modifican.
    los negativos se muestrean de f(w)^0.75 sin el propio positivo, y su
    actualizacion acumula los indices repetidos (np.add.at).
    """
    rng = np.random.default_rng(SEMILLA)
    v = len(vocab)
    idx = {p: i for i, p in enumerate(vocab)}
    W = rng.standard_normal((v, dim)) * 0.1
    C = rng.standard_normal((v, dim)) * 0.1
    freq = np.zeros(v)
    for f in corpus:
        for w in f:
            freq[idx[w]] += 1
    p_neg = freq ** 0.75
    p_neg /= p_neg.sum()
    pares = _pares(corpus, idx, ventana)
    cortes = prefijos if prefijos else [dim]
    nc = len(cortes)
    hist = []
    for ep in range(epocas):
        rng.shuffle(pares)
        perdida = 0.0
        for c, o in pares:
            q = p_neg.copy()
            q[o] = 0.0                       # el positivo no es negativo
            neg = rng.choice(v, size=negativos, p=q / q.sum())
            vc, vo, vn = W[c].copy(), C[o].copy(), C[neg].copy()
            gW = np.zeros(dim)
            gO = np.zeros(dim)
            gN = np.zeros((negativos, dim))
            for m in cortes:                 # uno o varios prefijos
                so = sigmoide(vc[:m] @ vo[:m])
                sn = sigmoide(vn[:, :m] @ vc[:m])
                perdida += (-np.log(so + 1e-9)
                            - np.log(1 - sn + 1e-9).sum()) / nc
                gW[:m] += ((so - 1) * vo[:m]
                           + (sn[:, None] * vn[:, :m]).sum(0)) / nc
                gO[:m] += (so - 1) * vc[:m] / nc
                gN[:, :m] += (sn[:, None] * vc[:m]) / nc
            W[c] -= lr * gW
            C[o] -= lr * gO
            np.add.at(C, neg, -lr * gN)      # acumula negativos repetidos
        acc = _acierto_tema(W, vocab)
        hist.append((ep + 1, round(perdida / len(pares), 4), round(acc, 4)))
    return W, hist


def simular_entrenamiento(hist) -> None:
    """curva de aprendizaje: perdida y calidad epoca a epoca."""
    print("\nentrenamiento word2vec: perdida y calidad por epoca")
    print("  epoca  perdida  acierto")
    for ep, per, acc in hist:
        if ep % 5 == 0 or ep == 1:
            print(f"  {ep:<5}  {per:.4f}  {acc:.4f}")
    _escribir(os.path.join("data", "cap12_entrenamiento.dat"),
              "word2vec: perdida y acierto de tema por epoca",
              "epoca  perdida  acierto",
              [(ep, per, acc) for ep, per, acc in hist])


def _svd_cooc(corpus, vocab, dim=DIM) -> np.ndarray:
    idx = {p: i for i, p in enumerate(vocab)}
    m = np.zeros((len(vocab), len(vocab)))
    for f in corpus:
        ids = [idx[p] for p in set(f)]
        for a in ids:
            for b in ids:
                if a != b:
                    m[a, b] += 1
    u, s, _ = np.linalg.svd(np.log1p(m), full_matrices=False)
    return u[:, :dim] * s[:dim]


def simular_comparacion(corpus, vocab, w2v) -> None:
    """calidad del embedding: aleatorio vs co-ocurrencia (cap. 11) vs word2vec."""
    rng = np.random.default_rng(SEMILLA)
    aleatorio = rng.standard_normal((len(vocab), DIM)) * 0.1
    svd = _svd_cooc(corpus, vocab)
    filas = [("aleatorio", round(_acierto_tema(aleatorio, vocab), 4)),
             ("coocurrencia_svd", round(_acierto_tema(svd, vocab), 4)),
             ("word2vec", round(_acierto_tema(w2v, vocab), 4))]
    print("\ncomparacion de calidad (acierto de tema del vecino):")
    for et, v in filas:
        print(f"  {et:18}  {v:.4f}")
    _escribir(os.path.join("data", "cap12_comparacion.dat"),
              "acierto de tema: aleatorio vs co-ocurrencia vs word2vec",
              "metodo  acierto", filas)


def atencion(consultas, claves, valores):
    """producto escalar escalado: salida ponderada de los valores segun la
    afinidad consulta-clave. es el mecanismo del transformer."""
    dk = claves.shape[-1]
    a = consultas @ claves.T / np.sqrt(dk)                # afinidad
    a = np.exp(a - a.max(axis=-1, keepdims=True))         # softmax estable
    pesos = a / a.sum(axis=-1, keepdims=True)             # por fila
    return pesos @ valores


def simular_contexto(vocab, W) -> None:
    """la misma palabra, dos contextos: la atencion le da vectores distintos
    (contextual), mientras que el embedding estatico le da uno solo."""
    idx = {p: i for i, p in enumerate(vocab)}
    W = W - W.mean(axis=0)               # centrar realza los contrastes
    # "tabla" en contexto de tecnologia vs en contexto de cocina (ambiguo)
    pal = "tabla"
    ctx_tec = ["base", "datos", "consulta", "indice"]
    ctx_coc = ["receta", "harina", "horno", "salsa"]

    def contextual(ctx):
        sec = np.array([W[idx[w]] for w in [pal] + ctx])
        sal = atencion(sec, sec, sec)
        return sal[0]                    # vector contextual de 'pal'

    v_tec, v_coc = contextual(ctx_tec), contextual(ctx_coc)
    estatico = coseno(W[idx[pal]], W[idx[pal]])     # =1 siempre
    contextual_cos = coseno(v_tec, v_coc)
    filas = [("estatico_mismo", round(estatico, 4)),
             ("contextual_dos_ctx", round(contextual_cos, 4))]
    print("\ncontexto: coseno de 'tabla' consigo misma")
    for et, v in filas:
        print(f"  {et:20}  {v:.4f}")
    _escribir(os.path.join("data", "cap12_contexto.dat"),
              "coseno de una palabra consigo misma: estatico (1) vs "
              "contextual en dos contextos (<1)",
              "caso  coseno", filas)


def simular_matryoshka(vocab, estandar, matr) -> None:
    """calidad al truncar la dimension: embedding estandar vs Matryoshka
    (entrenado para que los primeros componentes basten)."""
    filas = []
    print("\nMatryoshka: acierto al truncar la dimension")
    print("  dim   estandar  matryoshka")
    for d in (2, 4, 8, 16):
        e = _acierto_tema(estandar, vocab, dims=d)
        m = _acierto_tema(matr, vocab, dims=d)
        filas.append((d, round(e, 4), round(m, 4)))
        print(f"  {d:<4}  {e:.4f}    {m:.4f}")
    _escribir(os.path.join("data", "cap12_matryoshka.dat"),
              "acierto de tema al truncar la dimension: estandar vs Matryoshka",
              "dim  estandar  matryoshka", filas)


def simular_maxsim(vocab, W) -> None:
    """interaccion tardia: comparar dos textos por el mejor emparejamiento de
    sus tokens (MaxSim) frente al coseno de sus vectores promediados."""
    idx = {p: i for i, p in enumerate(vocab)}
    W = W - W.mean(axis=0)               # centrar realza los contrastes

    def vecs(toks):
        return np.array([W[idx[t]] for t in toks])

    def maxsim(a, b):
        A, B = vecs(a), vecs(b)
        sim = np.array([[coseno(x, y) for y in B] for x in A])
        return float(sim.max(axis=1).mean())     # cada token, su mejor par

    def medio(a, b):
        return coseno(vecs(a).mean(0), vecs(b).mean(0))

    # un documento que SI contiene los terminos de la consulta, diluido por
    # cantidad creciente de contenido ajeno (de otros temas). MaxSim sigue
    # encontrando los terminos; el vector medio se diluye con el relleno.
    consulta = ["indice", "consulta"]
    ajeno = ["receta", "harina", "gol", "balon", "piano", "ritmo",
             "horno", "salsa", "atleta", "campo", "melodia", "nota",
             "azucar", "tomate", "carrera", "cancion"]
    filas = []
    print("\nMaxSim (interaccion tardia) vs vector medio, segun relleno ajeno")
    print("  relleno  maxsim   medio")
    for n in (0, 2, 4, 8, 16):
        doc = ["indice", "consulta"] + ajeno[:n]
        ms, md = maxsim(consulta, doc), medio(consulta, doc)
        filas.append((n, round(ms, 4), round(md, 4)))
        print(f"  {n:<7}  {ms:.4f}  {md:.4f}")
    _escribir(os.path.join("data", "cap12_maxsim.dat"),
              "similitud consulta-doc segun relleno ajeno: MaxSim "
              "(multivector) vs vector medio",
              "relleno  maxsim  medio", filas)


def demostracion(vocab, W, k: int = 15) -> None:
    """15 palabras y su vecino mas proximo en el espacio word2vec."""
    rng = np.random.default_rng(SEMILLA + 1)
    idx = {p: i for i, p in enumerate(vocab)}
    tema = {p: t for t, ps in TEMAS.items() for p in ps}
    cont = [p for p in vocab if p in tema]
    muestra = list(rng.choice(cont, size=k, replace=False))
    print(f"\ndemostracion: {k} palabras y su vecino mas proximo (word2vec)")
    print("  palabra     vecino      coseno  mismo tema?")
    print("  ----------  ----------  ------  -----------")
    for p in muestra:
        mejor, mc = None, -2.0
        for q in cont:
            if q == p:
                continue
            c = coseno(W[idx[p]], W[idx[q]])
            if c > mc:
                mejor, mc = q, c
        print(f"  {p:10}  {mejor:10}  {mc:.3f}   "
              f"{'si' if tema[p] == tema[mejor] else 'no'}")


def main() -> None:
    anunciar()
    corpus = generar_corpus()
    vocab = vocabulario(corpus)
    print("entrenando word2vec (estandar)...")
    W, hist = entrenar_word2vec(corpus, vocab)
    print("entrenando word2vec (Matryoshka)...")
    matr, _ = entrenar_word2vec(corpus, vocab, prefijos=[2, 4, 8, 16])
    simular_entrenamiento(hist)
    simular_comparacion(corpus, vocab, W)
    simular_contexto(vocab, W)
    simular_matryoshka(vocab, W, matr)
    simular_maxsim(vocab, W)
    demostracion(vocab, W, 15)


if __name__ == "__main__":
    main()
