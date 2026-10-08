"""capitulo 13: geometria del espacio metrico.

estudia, en numpy, como se mide el parecido entre vectores y como se comporta el
espacio donde viven los embeddings. mide doce cosas que fijan el criterio para
elegir medida de parecido y entender la alta dimension:

  1. medidas y la norma: coseno, euclidea y producto interno pueden ordenar
     distinto cuando los vectores tienen normas distintas; el coseno no depende
     de la norma (vectores no nulos), la euclidea y el producto interno si.
  2. normalizacion: con base y consulta en la esfera unidad, la euclidea y el
     coseno ordenan igual.
  3. concentracion de distancias: para puntos gaussianos isotropicos, el
     contraste entre la distancia maxima y la minima cae al crecer la dimension.
  4. cosenos aleatorios: el coseno entre vectores aleatorios independientes se
     concentra en cero al crecer la dimension (casi-ortogonalidad).
  5. la cascara: casi todo el volumen de una bola de alta dimension esta pegado
     a su superficie.
  6. el volumen de la bola unidad: crece, alcanza un maximo hacia la dimension 5
     y luego se desploma hacia cero.
  7. hubness: en puntos gaussianos de alta dimension, la k-ocurrencia se vuelve
     muy asimetrica; algunos puntos son vecinos de muchisimos otros.
  8. el triangulo: la 'distancia' coseno (1 - coseno) no cumple la desigualdad
     triangular; la angular sobre la esfera si.
  9. dispersion: la dispersion relativa de las distancias (std/media) cae como
     1/raiz(d) bajo coordenadas independientes.
 10. estructura: en cumulos sinteticos bien separados el contraste se mantiene
     aunque la dimension crezca.
 11. Johnson-Lindenstrauss: distorsion media de las distancias al proyectar al
     azar a pocas dimensiones, sobre una muestra de pares.
 12. dimension lineal efectiva: componentes PCA para el 90 por ciento de la
     varianza, en datos isotropicos y en datos generados en un subespacio.

es Python puro con numpy (sin servicio ni GPU), con semilla fija; cada medida
crea su propio generador, de modo que el orden de ejecucion no cambia nada.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np

SEMILLA = 13


def anunciar() -> None:
    print("=" * 64)
    print("cap. 13: geometria del espacio metrico")
    print("recursos: python + numpy · cpu. no usa servicio, gpu ni torch.")
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


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _topk(puntuaciones: np.ndarray, k: int) -> set:
    """indices de los k mayores (los mas parecidos)."""
    return set(np.argsort(-puntuaciones)[:k].tolist())


def simular_metricas(n: int = 500, d: int = 64, k: int = 10,
                     trials: int = 200) -> None:
    """desacuerdo entre rankings de coseno, euclidea y producto interno al
    crecer la variabilidad de las normas de los vectores."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nmetricas: desacuerdo del top-k frente al coseno segun la norma")
    print("  var_norma  euclidea  prod_interno")
    for sigma in (0.0, 0.3, 0.6, 1.0, 1.5):
        des_euc, des_dot = [], []
        for _ in range(trials):
            base = rng.standard_normal((n, d))
            base /= np.linalg.norm(base, axis=1, keepdims=True)  # direccion
            escala = np.exp(rng.standard_normal(n) * sigma)      # norma
            v = base * escala[:, None]
            q = v[0]
            cos = np.array([coseno(q, x) for x in v])
            euc = -np.linalg.norm(v - q, axis=1)        # mayor = mas cerca
            dot = v @ q
            tc = _topk(cos, k + 1) - {0}
            des_euc.append(1 - len(tc & (_topk(euc, k + 1) - {0})) / k)
            des_dot.append(1 - len(tc & (_topk(dot, k + 1) - {0})) / k)
        filas.append((sigma, round(float(np.mean(des_euc)), 4),
                      round(float(np.mean(des_dot)), 4)))
        print(f"  {sigma:<9}  {filas[-1][1]:.4f}    {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap13_metricas.dat"),
              "desacuerdo del top-%d con el coseno segun variabilidad de "
              "la norma (d=%d)" % (k, d),
              "var_norma  euclidea  prod_interno", filas)


def simular_normalizacion(n: int = 500, d: int = 64, k: int = 10,
                          trials: int = 200) -> None:
    """acuerdo euclidea-coseno antes y despues de normalizar a la esfera."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nnormalizacion: acuerdo euclidea-coseno, sin y con normalizar")
    print("  var_norma  sin_norm  con_norm")
    for sigma in (0.0, 0.3, 0.6, 1.0, 1.5):
        sin, con = [], []
        for _ in range(trials):
            base = rng.standard_normal((n, d))
            base /= np.linalg.norm(base, axis=1, keepdims=True)
            escala = np.exp(rng.standard_normal(n) * sigma)
            v = base * escala[:, None]
            q = v[0]
            cos = np.array([coseno(q, x) for x in v])
            euc = -np.linalg.norm(v - q, axis=1)
            vn = v / np.linalg.norm(v, axis=1, keepdims=True)
            euc_n = -np.linalg.norm(vn - vn[0], axis=1)
            tc = _topk(cos, k + 1) - {0}
            sin.append(len(tc & (_topk(euc, k + 1) - {0})) / k)
            con.append(len(tc & (_topk(euc_n, k + 1) - {0})) / k)
        filas.append((sigma, round(float(np.mean(sin)), 4),
                      round(float(np.mean(con)), 4)))
        print(f"  {sigma:<9}  {filas[-1][1]:.4f}  {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap13_normalizacion.dat"),
              "acuerdo euclidea-coseno (top-%d) sin y con normalizar" % k,
              "var_norma  sin_norm  con_norm", filas)


def simular_concentracion(n: int = 1000, trials: int = 50) -> None:
    """contraste (dmax-dmin)/dmin de las distancias de una consulta a puntos
    gaussianos isotropicos, segun la dimension."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nconcentracion de distancias: (dmax-dmin)/dmin segun dimension")
    print("  dim     contraste")
    for d in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        c = []
        for _ in range(trials):
            pts = rng.standard_normal((n, d))
            q = rng.standard_normal(d)
            dist = np.linalg.norm(pts - q, axis=1)
            c.append((dist.max() - dist.min()) / dist.min())
        filas.append((d, round(float(np.mean(c)), 4)))
        print(f"  {d:<6}  {filas[-1][1]:.4f}")
    _escribir(os.path.join("data", "cap13_concentracion.dat"),
              "contraste (dmax-dmin)/dmin segun dimension",
              "dim  contraste", filas)


def simular_cosenos(trials: int = 4000) -> None:
    """media de |coseno| y desviacion del coseno entre vectores aleatorios
    segun la dimension: se concentra en cero (casi-ortogonalidad)."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\ncosenos aleatorios: |coseno| medio y desviacion segun dimension")
    print("  dim     |cos|_medio  desviacion")
    for d in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        cs = []
        for _ in range(trials):
            a, b = rng.standard_normal(d), rng.standard_normal(d)
            cs.append(coseno(a, b))
        cs = np.array(cs)
        filas.append((d, round(float(np.abs(cs).mean()), 4),
                      round(float(cs.std()), 4)))
        print(f"  {d:<6}  {filas[-1][1]:.4f}       {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap13_cosenos.dat"),
              "|coseno| medio y desviacion entre vectores aleatorios",
              "dim  cos_abs  desviacion", filas)


def simular_cascara() -> None:
    """fraccion del volumen de una bola que esta en la cascara exterior (el
    10 % mas externo del radio): 1 - 0.9^d. en alta dimension, casi todo."""
    filas = []
    print("\ncascara: fraccion del volumen en el 10% exterior del radio")
    print("  dim    frac_cascara")
    for d in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        frac = 1 - 0.9 ** d
        filas.append((d, round(frac, 4)))
        print(f"  {d:<5}  {frac:.4f}")
    _escribir(os.path.join("data", "cap13_cascara.dat"),
              "fraccion del volumen de la bola en el 10% exterior (1-0.9^d)",
              "dim  frac", filas)


def simular_volumen() -> None:
    """volumen de la bola unidad segun la dimension: crece, alcanza un maximo
    hacia d=5 y se desploma hacia cero."""
    filas = []
    print("\nvolumen de la bola unidad segun la dimension")
    print("  dim    volumen")
    for d in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
        vol = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
        filas.append((d, round(vol, 5)))
        print(f"  {d:<5}  {vol:.5f}")
    _escribir(os.path.join("data", "cap13_volumen.dat"),
              "volumen de la bola unidad de radio 1 segun la dimension",
              "dim  volumen", filas)


def simular_hubness(n: int = 800, k: int = 10, trials: int = 20) -> None:
    """hubness: asimetria de la distribucion de cuantas veces cada punto es
    vecino de otros (k-ocurrencia), en puntos gaussianos. crece con la
    dimension en este modelo."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nhubness: asimetria de la k-ocurrencia segun la dimension")
    print("  dim    asimetria  max_ocurr")
    for d in (2, 8, 32, 128, 512):
        asim, maxoc = [], []
        for _ in range(trials):
            pts = rng.standard_normal((n, d))
            # matriz de distancias y los k vecinos de cada punto
            ocur = np.zeros(n)
            for i in range(n):
                dist = np.linalg.norm(pts - pts[i], axis=1)
                vecinos = np.argsort(dist)[1:k + 1]
                ocur[vecinos] += 1
            m, s = ocur.mean(), ocur.std()
            asim.append(float(((ocur - m) ** 3).mean() / (s ** 3 + 1e-9)))
            maxoc.append(float(ocur.max()))
        filas.append((d, round(float(np.mean(asim)), 4),
                      round(float(np.mean(maxoc)), 1)))
        print(f"  {d:<5}  {filas[-1][1]:.4f}    {filas[-1][2]:.1f}")
    _escribir(os.path.join("data", "cap13_hubness.dat"),
              "asimetria de la k-ocurrencia (hubness) y maxima ocurrencia "
              "segun dimension (k=%d)" % k,
              "dim  asimetria  max_ocurr", filas)


def simular_triangulo(d: int = 32, trials: int = 100000) -> None:
    """fraccion de triples que violan la desigualdad triangular con la
    'distancia' coseno (1-cos) y con la distancia angular (arccos): la
    primera no es metrica, la segunda si."""
    rng = np.random.default_rng(SEMILLA)
    viol_cos, viol_ang = 0, 0
    for _ in range(trials):
        a, b, c = rng.standard_normal((3, d))
        # distancia coseno
        dab, dbc, dac = (1 - coseno(a, b), 1 - coseno(b, c),
                         1 - coseno(a, c))
        if dac > dab + dbc + 1e-9:
            viol_cos += 1
        # distancia angular (arccos del coseno)
        aab, abc, aac = (math.acos(np.clip(coseno(a, b), -1, 1)),
                         math.acos(np.clip(coseno(b, c), -1, 1)),
                         math.acos(np.clip(coseno(a, c), -1, 1)))
        if aac > aab + abc + 1e-9:
            viol_ang += 1
    filas = [("coseno_1menoscos", round(viol_cos / trials, 5)),
             ("angular_arccos", round(viol_ang / trials, 5))]
    print("\ntriangulo: fraccion de triples que violan la desigualdad")
    for et, v in filas:
        print(f"  {et:18}  {v:.5f}")
    _escribir(os.path.join("data", "cap13_triangulo.dat"),
              "fraccion de triples que violan la desigualdad triangular "
              "(d=%d)" % d,
              "distancia  violaciones", filas)


def simular_intrinseca(n: int = 2000, k: int = 10) -> None:
    """dimension lineal efectiva por PCA: cuantas componentes hacen falta para
    explicar el 90 por ciento de la varianza. los datos generados en un
    subespacio de dimension k (mas algo de ruido) necesitan unas k componentes
    por alta que sea la dimension ambiente; los isotropicos, casi todas. no es
    la dimension intrinseca de una variedad no lineal."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\ndimension lineal efectiva (componentes PCA para el 90% de varianza)")
    print("  dim     uniforme  estructurado")
    for d in (16, 32, 64, 128, 256, 512):
        uni = rng.standard_normal((n, d))
        # estructurado: vive en un subespacio de dimension k mas algo de ruido
        latente = rng.standard_normal((n, k))
        base = rng.standard_normal((k, d))
        est = latente @ base + 0.1 * rng.standard_normal((n, d))
        iu = _componentes_90(uni)
        ie = _componentes_90(est)
        filas.append((d, iu, ie))
        print(f"  {d:<6}  {iu:<8}  {ie}")
    _escribir(os.path.join("data", "cap13_intrinseca.dat"),
              "dimension lineal efectiva: componentes PCA para el 90 por ciento "
              "de la varianza en datos isotropicos vs en un subespacio",
              "dim  uniforme  estructurado", filas)


def _componentes_90(x: np.ndarray) -> int:
    """numero de componentes principales que explican el 90% de la varianza."""
    xc = x - x.mean(axis=0)
    s = np.linalg.svd(xc, compute_uv=False)
    var = s ** 2
    acum = np.cumsum(var) / var.sum()
    return int(np.searchsorted(acum, 0.90) + 1)


def simular_dispersion(n: int = 3000) -> None:
    """dispersion RELATIVA de las distancias (std/media) de un punto a los demas
    segun la dimension. con coordenadas independientes cae como 1/raiz(d): la
    distancia al cuadrado es una suma de d terminos. se excluye la distancia del
    punto consigo mismo, que es cero y sesgaria la estimacion."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\ndispersion relativa de las distancias (std/media) ~ 1/raiz(d)")
    print("  dim     std/media   1/raiz(d)   media   raiz(2d)   std")
    for d in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        pts = rng.standard_normal((n, d))
        dist = np.linalg.norm(pts[1:] - pts[0], axis=1)    # sin el propio
        rel = float(dist.std() / dist.mean())
        filas.append((d, round(rel, 4), round(1.0 / math.sqrt(d), 4)))
        # la media crece como raiz(2d) y la desviacion absoluta se estabiliza
        print(f"  {d:<6}  {filas[-1][1]:.4f}      {filas[-1][2]:.4f}"
              f"      {dist.mean():6.2f}  {math.sqrt(2 * d):6.2f}"
              f"     {dist.std():.3f}")
    _escribir(os.path.join("data", "cap13_dispersion.dat"),
              "dispersion relativa de las distancias (std/media) y la "
              "referencia 1/raiz(d), segun la dimension",
              "dim  rel  ref", filas)


def simular_estructura(n: int = 1000, trials: int = 40) -> None:
    """contraste de distancias en datos isotropicos frente a diez cumulos
    sinteticos bien separados, segun la dimension. en este modelo el contraste
    entre el cumulo propio y los demas sobrevive al crecer la dimension."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    print("\nestructura: contraste uniforme vs estructurado (cumulos)")
    print("  dim     uniforme  estructurado")
    for d in (2, 8, 32, 128, 512):
        cu, ce = [], []
        for _ in range(trials):
            # uniforme: ruido sin estructura
            pu = rng.standard_normal((n, d))
            qu = rng.standard_normal(d)
            du = np.linalg.norm(pu - qu, axis=1)
            cu.append((du.max() - du.min()) / du.min())
            # estructurado: 10 cumulos separados; la consulta, en uno
            centros = rng.standard_normal((10, d)) * 6
            etiq = rng.integers(0, 10, n)
            pe = centros[etiq] + rng.standard_normal((n, d))
            qe = centros[0] + rng.standard_normal(d)
            de = np.linalg.norm(pe - qe, axis=1)
            ce.append((de.max() - de.min()) / de.min())
        filas.append((d, round(float(np.mean(cu)), 4),
                      round(float(np.mean(ce)), 4)))
        print(f"  {d:<6}  {filas[-1][1]:.4f}    {filas[-1][2]:.4f}")
    _escribir(os.path.join("data", "cap13_estructura.dat"),
              "contraste de distancias: datos uniformes vs estructurados "
              "(cumulos) segun dimension",
              "dim  uniforme  estructurado", filas)


def simular_jl(n: int = 300, d: int = 512, trials: int = 20) -> None:
    """reduccion de dimension por proyeccion aleatoria (Johnson-Lindenstrauss):
    distorsion relativa media de las distancias de una muestra de pares al
    proyectar de d a k dimensiones. no verifica la garantia uniforme del lema."""
    rng = np.random.default_rng(SEMILLA)
    filas = []
    base = rng.standard_normal((n, d))
    # distancias originales (un subconjunto de pares)
    idx = rng.choice(n, (2000, 2))
    orig = np.linalg.norm(base[idx[:, 0]] - base[idx[:, 1]], axis=1)
    print("\nJohnson-Lindenstrauss: distorsion al reducir a k dimensiones")
    print("  k      distorsion_media")
    for k in (8, 16, 32, 64, 128, 256):
        dist = []
        for _ in range(trials):
            R = rng.standard_normal((d, k)) / math.sqrt(k)   # proyeccion
            proj = base @ R
            red = np.linalg.norm(proj[idx[:, 0]] - proj[idx[:, 1]], axis=1)
            dist.append(float(np.abs(red - orig).mean() / orig.mean()))
        filas.append((k, round(float(np.mean(dist)), 4)))
        print(f"  {k:<5}  {filas[-1][1]:.4f}")
    _escribir(os.path.join("data", "cap13_jl.dat"),
              "distorsion media de las distancias al proyectar de %d a k "
              "dimensiones (Johnson-Lindenstrauss)" % d,
              "k  distorsion", filas)


def demostracion(k: int = 15, d: int = 64) -> None:
    """15 pares de vectores al azar con sus tres medidas de parecido."""
    rng = np.random.default_rng(SEMILLA + 1)
    print(f"\ndemostracion: {k} pares de vectores y sus tres medidas")
    print("  par  coseno   euclidea  prod_int")
    print("  ---  -------  --------  --------")
    for i in range(k):
        a = rng.standard_normal(d) * np.exp(rng.standard_normal())
        b = rng.standard_normal(d) * np.exp(rng.standard_normal())
        print(f"  {i:<3}  {coseno(a, b): .4f}  "
              f"{np.linalg.norm(a - b):8.3f}  {a @ b: .3f}")


def main() -> None:
    anunciar()
    simular_metricas()
    simular_normalizacion()
    simular_concentracion()
    simular_cosenos()
    simular_cascara()
    simular_volumen()
    simular_hubness()
    simular_triangulo()
    simular_dispersion()
    simular_estructura()
    simular_jl()
    simular_intrinseca()
    demostracion(15)


if __name__ == "__main__":
    main()
