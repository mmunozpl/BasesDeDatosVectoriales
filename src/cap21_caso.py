"""capitulo 21: caso de estudio, recuperacion de observaciones de aves.

estudia un caso de uso realista (recuperar observaciones pasadas comparables
para apoyar la identificacion de un ave) con un corpus sintetico de respuesta
conocida. los embeddings son el centro de cada especie mas ruido, no un
codificador visual real, y el habitat sigue una matriz sintetica P(habitat |
especie) con especies generalistas y especialistas y un 8 % de observaciones
fuera de su habitat habitual. ninguna cifra describe aves reales.

  1. conjuncion: precision@10 de la relevancia definida como misma especie Y
     mismo habitat, con solo similitud, solo filtro y filtro + similitud,
     promediada por especie (cada especie pesa igual).
  2. identificacion: precision@10 de la misma especie con el habitat como
     filtro duro o como prior blando (similitud + beta si coincide), para
     observaciones tipicas y excepcionales.
  3. postfiltrado ingenuo por habitat, con los habitats del corpus y con
     consultas que son observaciones (correlacionadas) o aleatorias.
  4. maqueta MaxSim: una senal concentrada en un parche, vector unico (media de
     parches) frente al maximo por parche, con 9 y 36 parches.
  5. dos etapas: vector unico para los R primeros candidatos y MaxSim para
     reordenarlos; recall de candidatos, acierto final y coste relativo.
  6. desbalance: precision@10 por especie con su soporte, su intervalo y el
     efecto de la autocoincidencia si no se excluye la consulta.

en todos los rankings la observacion usada como consulta se excluye de la
coleccion. las cifras dependen solo de la semilla.
"""

from __future__ import annotations

import os

import numpy as np

SEMILLA = 21

# especies del corpus y su frecuencia sintetica, elegida para el experimento
ESPECIES = ["gorrion_comun", "mirlo_comun", "petirrojo", "abubilla", "alimoche"]
FRECUENCIA = np.array([0.55, 0.12, 0.20, 0.08, 0.05])
HABITATS = ["bosque", "humedal", "montana", "urbano", "matorral", "costa"]
# P(habitat | especie), sintetica: el gorrion es urbano, el mirlo y el
# petirrojo forestales, la abubilla de matorral y el alimoche de montana; los
# valores pequenos son observaciones fuera del habitat habitual
P_HAB = np.array([[0.100, 0.033, 0.033, 0.600, 0.200, 0.034],
                  [0.500, 0.017, 0.017, 0.350, 0.100, 0.016],
                  [0.600, 0.017, 0.017, 0.100, 0.250, 0.016],
                  [0.150, 0.033, 0.033, 0.150, 0.600, 0.034],
                  [0.025, 0.025, 0.700, 0.025, 0.200, 0.025]])
P_HAB = P_HAB / P_HAB.sum(1, keepdims=True)
UMBRAL_TIPICO = 0.1     # habitat tipico de una especie si P(h | s) >= 0,1


def anunciar() -> None:
    print("=" * 64)
    print("cap. 21: caso de estudio, recuperacion de observaciones de aves")
    print("recursos: python + numpy, cpu")
    print(f"semilla = {SEMILLA}")
    print("=" * 64)


def _escribir(ruta: str, nota: str, cols: str, filas: list[tuple]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(f"# {nota}\n")
        fh.write(cols + "\n")
        for fila in filas:
            fh.write("  ".join(str(x) for x in fila) + "\n")
    print(f"escrito {ruta}")


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("no se normaliza un vector nulo")
    return x / n


def _top(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, int(np.isfinite(sims).sum()))
    part = np.argpartition(-sims, k - 1)[:k]
    return part[np.argsort(-sims[part], kind="stable")]


def corpus(rng, n: int = 20_000, dim: int = 64, ruido: float = 0.9) -> dict:
    """observaciones sinteticas: especie con FRECUENCIA, habitat con
    P_HAB[especie], embedding = centro de la especie mas ruido."""
    esp = rng.choice(len(ESPECIES), n, p=FRECUENCIA)
    u = rng.random(n)
    hab = (u[:, None] > np.cumsum(P_HAB[esp], axis=1)).sum(1)
    centros = _norm(rng.standard_normal((len(ESPECIES), dim)))
    emb = _norm(centros[esp] + ruido * rng.standard_normal((n, dim)))
    tipico = P_HAB[esp, hab] >= UMBRAL_TIPICO
    return {"emb": emb, "esp": esp, "hab": hab, "tipico": tipico}


def sims_sin(emb: np.ndarray, qi: int) -> np.ndarray:
    """similitudes con la observacion qi, que se excluye de su propio
    ranking."""
    s = emb @ emb[qi]
    s[qi] = -np.inf
    return s


def _consultas(rng, esp: np.ndarray, mascara: np.ndarray,
               por_especie: int) -> list[int]:
    """el mismo numero de consultas por especie (macro-promedio)."""
    out = []
    for d in range(len(ESPECIES)):
        idx = np.flatnonzero((esp == d) & mascara)
        m = min(por_especie, len(idx))
        out += rng.choice(idx, m, replace=False).tolist()
    return out


# ---------------------------------------------------------------------------
# 1. la conjuncion especie y habitat
# ---------------------------------------------------------------------------

def simular_conjuncion(k: int = 10, por_especie: int = 80) -> None:
    rng = np.random.default_rng(SEMILLA)
    c = corpus(rng)
    emb, esp, hab = c["emb"], c["esp"], c["hab"]
    filas = []
    print("\nconjuncion (misma especie Y habitat): precision@%d por "
          "especie" % k)
    print("  especie          similitud  filtro  ambos")
    res = np.zeros((len(ESPECIES), 3))
    for d in range(len(ESPECIES)):
        acc = np.zeros(3)
        qs = _consultas(rng, esp, esp == d, por_especie)
        for qi in qs:
            rel = (esp == esp[qi]) & (hab == hab[qi])
            s = sims_sin(emb, qi)
            otros = np.arange(len(esp)) != qi
            mismo = np.flatnonzero((hab == hab[qi]) & otros)
            if len(mismo) < k:
                raise ValueError("habitat con menos de k observaciones")
            acc += [rel[_top(s, k)].mean(),
                    rel[rng.choice(mismo, k, replace=False)].mean(),
                    rel[mismo[_top(s[mismo], k)]].mean()]
        res[d] = acc / len(qs)
        filas.append((ESPECIES[d],) + tuple(round(v, 4) for v in res[d]))
        print(f"  {ESPECIES[d]:<15}  {res[d][0]:.4f}     {res[d][1]:.4f}  "
              f"{res[d][2]:.4f}")
    macro = res.mean(0)
    filas.append(("macro",) + tuple(round(v, 4) for v in macro))
    print(f"  {'macro':<15}  {macro[0]:.4f}     {macro[1]:.4f}  {macro[2]:.4f}")
    _escribir(os.path.join("data", "cap21_filtro_similitud.dat"),
              "precision@%d de la relevancia misma especie y mismo habitat "
              "con solo similitud, solo filtro (muestra del mismo habitat) y "
              "filtro + similitud, por especie (%d consultas cada una, sin la "
              "propia consulta) y macro-promedio" % (k, por_especie),
              "especie  similitud  filtro  ambos", filas)


# ---------------------------------------------------------------------------
# 2. identificacion: el habitat como filtro duro o como prior blando
# ---------------------------------------------------------------------------

def simular_prior(k: int = 10, por_especie: int = 40) -> None:
    rng = np.random.default_rng(SEMILLA + 1)
    c = corpus(rng)
    emb, esp, hab, tip = c["emb"], c["esp"], c["hab"], c["tipico"]
    q_tip = _consultas(rng, esp, tip, por_especie)
    q_exc = _consultas(rng, esp, ~tip, por_especie)
    filas = []
    print("\nidentificacion (misma especie): precision@%d segun el peso beta "
          "del habitat" % k)
    print("  beta    tipicas  excepcionales")
    for beta in (0.0, 0.02, 0.05, 0.1, 0.2, np.inf):
        r = []
        for qs in (q_tip, q_exc):
            por = np.zeros(len(ESPECIES))
            cuenta = np.zeros(len(ESPECIES))
            for qi in qs:
                s = sims_sin(emb, qi)
                if np.isinf(beta):
                    s = np.where(hab == hab[qi], s, -np.inf)   # filtro duro
                else:
                    s = s + beta * (hab == hab[qi])            # prior blando
                d = esp[qi]
                por[d] += (esp[_top(s, k)] == d).mean()
                cuenta[d] += 1
            r.append(float(np.mean(por / cuenta)))
        etiqueta = "duro" if np.isinf(beta) else beta
        filas.append((etiqueta, round(r[0], 4), round(r[1], 4)))
        print(f"  {str(etiqueta):<6}  {r[0]:.4f}   {r[1]:.4f}")
    _escribir(os.path.join("data", "cap21_prior.dat"),
              "precision@%d de la misma especie (macro por especie) con el "
              "habitat como prior blando (similitud + beta si coincide) o como "
              "filtro duro, para observaciones en un habitat tipico de su "
              "especie y fuera de el" % k,
              "beta  tipicas  excepcionales", filas)


# ---------------------------------------------------------------------------
# 3. postfiltrado ingenuo con los habitats del corpus
# ---------------------------------------------------------------------------

def simular_prefiltrado(k: int = 10, nq: int = 200) -> None:
    """recall del top-k global filtrado despues respecto del top-k filtrado
    exacto (que el prefiltrado exacto reproduce por construccion), con
    consultas que son observaciones de ese habitat (sin ellas mismas) y con
    consultas aleatorias."""
    rng = np.random.default_rng(SEMILLA + 2)
    c = corpus(rng)
    emb, hab = c["emb"], c["hab"]
    n, dim = emb.shape
    filas = []
    print("\npostfiltrado ingenuo por habitat: recall frente al top-%d "
          "filtrado" % k)
    print("  habitat    prevalencia  observacion  aleatoria")
    for h in np.argsort(np.bincount(hab, minlength=len(HABITATS))):
        pasa = hab == h
        rec = []
        for tipo in ("obs", "ale"):
            acc = 0.0
            for _ in range(nq):
                if tipo == "obs":
                    qi = int(rng.choice(np.flatnonzero(pasa)))
                    s = sims_sin(emb, qi)
                else:
                    s = emb @ _norm(rng.standard_normal(dim))
                elegibles = np.flatnonzero(pasa & np.isfinite(s))
                oro = set(elegibles[_top(s[elegibles], k)].tolist())
                post = [i for i in _top(s, k) if pasa[i]]
                acc += len(set(post) & oro) / len(oro)
            rec.append(acc / nq)
        filas.append((HABITATS[h], round(float(pasa.mean()), 3),
                      round(rec[0], 4), round(rec[1], 4)))
        print(f"  {HABITATS[h]:<9}  {pasa.mean():.3f}        {rec[0]:.4f}"
              f"       {rec[1]:.4f}")
    _escribir(os.path.join("data", "cap21_prefiltrado.dat"),
              "recall del top-%d global filtrado despues frente al top-%d "
              "filtrado exacto, por habitat del corpus, con consultas que son "
              "observaciones del habitat y con consultas aleatorias" % (k, k),
              "habitat  prevalencia  observacion  aleatoria", filas)


# ---------------------------------------------------------------------------
# 4. maqueta MaxSim y 5. dos etapas
# ---------------------------------------------------------------------------

def _parches(rng, n: int, parches: int, dim: int) -> np.ndarray:
    return _norm(rng.standard_normal((n, parches, dim)))


def _consulta_local(rng, fondo: np.ndarray, senal: float) -> tuple:
    """inserta un patron en un parche de una observacion objetivo (sobre una
    copia de esa fila) y devuelve la consulta, el objetivo y la fila."""
    n, parches, dim = fondo.shape
    patron = _norm(rng.standard_normal(dim))
    obj = int(rng.integers(0, n))
    fila = fondo[obj].copy()
    jp = int(rng.integers(0, parches))
    fila[jp] = _norm(fila[jp] + senal * patron)
    q = _norm(patron + 0.15 * rng.standard_normal(dim))
    return q, obj, fila


def _puntuar(fondo, medias, q, obj, fila) -> tuple[np.ndarray, np.ndarray]:
    """puntuaciones de vector unico (media de parches) y de MaxSim (maximo
    por parche), con la fila objetivo modificada."""
    unico = medias @ q
    unico[obj] = _norm(fila.mean(0)) @ q
    maxsim = (fondo @ q).max(axis=1)
    maxsim[obj] = (fila @ q).max()
    return unico, maxsim


def simular_interaccion_tardia(n: int = 4000, dim: int = 32, k: int = 10,
                               nq: int = 300) -> None:
    rng = np.random.default_rng(SEMILLA + 3)
    filas = []
    print("\nmaqueta MaxSim: hit@%d del objetivo segun la senal local" % k)
    print("  senal  unico9  max9    unico36  max36")
    fondos = {p: _parches(rng, n, p, dim) for p in (9, 36)}
    medias = {p: _norm(f.mean(axis=1)) for p, f in fondos.items()}
    for senal in (0.5, 1.0, 1.5, 2.0, 3.0):
        fila_dat = [senal]
        for p in (9, 36):
            hu = hm = 0
            for _ in range(nq):
                q, obj, fila = _consulta_local(rng, fondos[p], senal)
                u, m = _puntuar(fondos[p], medias[p], q, obj, fila)
                hu += obj in set(_top(u, k).tolist())
                hm += obj in set(_top(m, k).tolist())
            fila_dat += [round(hu / nq, 4), round(hm / nq, 4)]
        filas.append(tuple(fila_dat))
        print("  " + "  ".join(f"{v:<6}" for v in fila_dat))
    _escribir(os.path.join("data", "cap21_interaccion_tardia.dat"),
              "hit@%d de la observacion objetivo cuya senal vive en un parche: "
              "vector unico (media de parches) frente al maximo por parche, "
              "con 9 y 36 parches (n=%d, dim=%d)" % (k, n, dim),
              "senal_local  unico9  max9  unico36  max36", filas)


def simular_dos_etapas(n: int = 4000, dim: int = 32, parches: int = 9,
                       senal: float = 1.5, k: int = 10,
                       nq: int = 300) -> None:
    """primera etapa: los R mejores por vector unico; segunda: MaxSim sobre
    esos R. el acierto final no puede superar al recall de candidatos."""
    rng = np.random.default_rng(SEMILLA + 4)
    fondo = _parches(rng, n, parches, dim)
    medias = _norm(fondo.mean(axis=1))
    lotes = [_consulta_local(rng, fondo, senal) for _ in range(nq)]
    filas = []
    print("\ndos etapas (senal %.1f, %d parches): candidatos y acierto final"
          % (senal, parches))
    print("  R      cand@R  hit@10  coste_rel")
    for R in (10, 20, 50, 100, 200, 500, 1000, n):
        cand = fin = 0
        for q, obj, fila in lotes:
            u, m = _puntuar(fondo, medias, q, obj, fila)
            c = _top(u, R)
            cand += obj in set(c.tolist())
            fin += obj in set(c[_top(m[c], k)].tolist())
        coste = (n + R * parches) / (n * parches)
        filas.append((R, round(cand / nq, 4), round(fin / nq, 4),
                      round(coste, 3)))
        print(f"  {R:<5}  {cand/nq:.4f}  {fin/nq:.4f}  {coste:.3f}")
    _escribir(os.path.join("data", "cap21_dos_etapas.dat"),
              "dos etapas: recall de candidatos del vector unico en los R "
              "primeros, hit@%d tras reordenar con MaxSim y coste relativo a "
              "MaxSim sobre toda la coleccion (senal %.1f, %d parches, n=%d)"
              % (k, senal, parches, n), "R  candidatos  final  coste", filas)


# ---------------------------------------------------------------------------
# 6. desbalance
# ---------------------------------------------------------------------------

def simular_desbalance(k: int = 10, por_especie: int = 300) -> None:
    rng = np.random.default_rng(SEMILLA + 5)
    c = corpus(rng)
    emb, esp = c["emb"], c["esp"]
    filas = []
    print("\ndesbalance: precision@%d por especie, sin la consulta" % k)
    print("  especie          soporte  precision  ic95          hit@10  "
          "con_auto")
    for d in range(len(ESPECIES)):
        qs = _consultas(rng, esp, esp == d, por_especie)
        prec, hit, auto = [], [], []
        for qi in qs:
            s = sims_sin(emb, qi)
            t = _top(s, k)
            prec.append((esp[t] == d).mean())
            hit.append(bool((esp[t] == d).any()))
            s_auto = emb @ emb[qi]                  # sin excluir la consulta
            auto.append((esp[_top(s_auto, k)] == d).mean())
        m, sd = float(np.mean(prec)), float(np.std(prec, ddof=1))
        ic = 1.96 * sd / np.sqrt(len(prec))
        soporte = int((esp == d).sum())             # observaciones de la clase
        techo = min(1.0, (soporte - 1) / k)         # sin la propia consulta
        if techo < 1.0:
            print(f"  {ESPECIES[d]}: techo de precision@{k} {techo:.2f}")
        filas.append((ESPECIES[d], soporte, round(m, 4),
                      round(m - ic, 4), round(m + ic, 4),
                      round(float(np.mean(hit)), 4),
                      round(float(np.mean(auto)), 4)))
        f = filas[-1]
        print(f"  {f[0]:<15}  {f[1]:<7}  {f[2]:.4f}     "
              f"[{f[3]:.3f}, {f[4]:.3f}]"
              f"  {f[5]:.4f}  {f[6]:.4f}")
    _escribir(os.path.join("data", "cap21_desbalance.dat"),
              "precision@%d de la misma especie por especie (sin la consulta), "
              "con su soporte en el corpus, intervalo al 95 %%, hit@%d y la "
              "precision que daria la autocoincidencia" % (k, k),
              "especie  soporte  precision  ic_bajo  ic_alto  hit  con_auto",
              filas)


# ---------------------------------------------------------------------------
# demostracion: una consulta con filtro de habitat
# ---------------------------------------------------------------------------

def demostracion(k: int = 5) -> None:
    rng = np.random.default_rng(SEMILLA + 6)
    c = corpus(rng)
    emb, esp, hab = c["emb"], c["esp"], c["hab"]
    qi = int(rng.integers(0, len(esp)))
    s = sims_sin(emb, qi)
    mismo = np.flatnonzero((hab == hab[qi]) & np.isfinite(s))
    orden = mismo[_top(s[mismo], k)]
    resultado = [{"observacion": f"obs_{int(i):05d}",
                  "imagen": f"img/obs_{int(i):05d}.jpg",
                  "sim": round(float(s[i]), 4),
                  "especie": ESPECIES[esp[i]],
                  "habitat": HABITATS[hab[i]]} for i in orden]
    print("\ndemostracion: consulta obs_%05d en '%s' (especie confirmada: %s)"
          % (qi, HABITATS[hab[qi]], ESPECIES[esp[qi]]))
    print(f"  candidatos del mismo habitat: {len(mismo)} de {len(esp)}")
    # la composicion del habitat filtrado sigue a la tabla sintetica
    h = hab[qi]
    conj = FRECUENCIA * P_HAB[:, h]                     # P(s, h)
    print(f"  el habitat reune el {100 * np.mean(hab == h):.1f} % del corpus "
          f"(tabla: {100 * conj.sum():.1f} %)")
    for j in np.argsort(-conj)[:2]:
        print(f"  {ESPECIES[j]}: {100 * np.mean(esp[hab == h] == j):.0f} % "
              f"del habitat (tabla: {100 * conj[j] / conj.sum():.0f} %)")
    print("  rango  observacion  sim      especie          habitat")
    for r, f in enumerate(resultado, start=1):
        print(f"  {r:<5}  {f['observacion']}    {f['sim']:+.4f}  "
              f"{f['especie']:<15}  {f['habitat']}")


def fuera_de_habitat() -> None:
    """fraccion de observaciones fuera de un habitat tipico de su especie:
    la esperada segun la tabla y la de los corpus de las semillas 21 a 27."""
    raro = P_HAB < UMBRAL_TIPICO
    esperada = float((FRECUENCIA[:, None] * P_HAB * raro).sum())
    medidas = [1 - corpus(np.random.default_rng(SEMILLA + i))["tipico"].mean()
               for i in range(7)]
    print(f"\nfuera del habitat tipico: esperada {100 * esperada:.1f} %; "
          f"corpus de las semillas {SEMILLA}-{SEMILLA + 6}: de "
          f"{100 * min(medidas):.1f} a {100 * max(medidas):.1f} %")


def main() -> None:
    anunciar()
    fuera_de_habitat()
    simular_conjuncion()
    simular_prior()
    simular_prefiltrado()
    simular_interaccion_tardia()
    simular_dos_etapas()
    simular_desbalance()
    demostracion()


if __name__ == "__main__":
    main()
