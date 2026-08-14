# Bases de datos: SQL · NoSQL · Vectorial

🇬🇧 [English](README.md) · 🇪🇸 Español

**Manuel Muñoz Plá**

[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#cómo-citar)

Manual sobre **bases de datos vectoriales**, de la coincidencia exacta del
modelo relacional a la similitud semántica del embedding, y de la teoría al
sistema en producción. Veintiún capítulos en cinco partes: *Fundamentos del
dato estructurado*, *La relajación del esquema*, *De la coincidencia a la
similitud*, *Bases de datos vectoriales* y *Arquitecturas e integración*.

Este repositorio reúne el **código reproducible** que genera cada resultado del
libro. Los ejercicios de cada capítulo, sus soluciones y los apéndices están en
la **obra completa** (papel, PDF y EPUB), que se distribuye por separado.

> 📘 **Ficha del libro** y más obras del autor:
> [manpla.net/libros/bases-datos-vectoriales](https://manpla.net/libros/bases-datos-vectoriales/)

## Contenido

```
.
├── src/                 # un módulo Python reproducible por capítulo
├── data/                # el registro de figuras: los .dat que alimentan las gráficas
├── infra/               # docker-compose de los motores (Postgres+pgvector, Mongo,
│                        #   Redis, TimescaleDB, Qdrant, Milvus) + healthcheck
└── pyproject.toml       # entorno exacto, con uv.lock y requirements.txt
```

La edición web no vive aquí: se lee en manpla.net, enlazada más abajo.

## Leer el libro

La edición web (21 capítulos, con buscador, matemáticas y bibliografía por
página) se publica en el sitio del autor:

> https://manpla.net/libros/bases-datos-vectoriales/

Cada capítulo es una página de manpla.net, con su propia navegación, y se
compone con **figuras vectoriales SVG renderizadas con el mismo pdfLaTeX del
libro**, de modo que las referencias y los números de figura, tabla y listado
son los del texto impreso.

Los amplios ejercicios de cada capítulo, sus soluciones y los apéndices no se
publican en la web: viven en la **obra completa** (papel, PDF y EPUB).

## Ejecutar el código

Cada capítulo trae un módulo reproducible en `src/capNN_*.py`, determinista con
semilla fija: al ejecutarlo regenera los `.dat` sobre los que se dibujan sus
figuras.

El `uv.lock` fija las versiones exactas con que se obtuvieron los números:
**Python 3.11.14** con `numpy` 2.3.3. La mayor parte del libro corre en **CPU**
y no necesita GPU: los índices y las ideas —HNSW, IVF, cuantización de
producto, quórumes, Map-Reduce…— están implementados desde cero, sin
dependencias pesadas.

```bash
uv run python src/cap15_ann.py        # p. ej. índices de vecino aproximado
```

Si el entorno se comparte con otros proyectos, la instalación va **aditiva**,
que no barre lo que no esté declarado aquí:

```bash
uv pip install -r <(uv export --no-hashes --no-dev)
```

Sin `uv`, con `pip`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Los capítulos que **miden sobre motores reales** necesitan los servicios, cuya
versión fija `infra/docker-compose.yml` (`pgvector/pgvector:pg16`, `mongo:7`,
`redis:7`, `qdrant/qdrant`, `milvusdb/milvus:v2.5.4`):

```bash
docker compose -f infra/docker-compose.yml up -d
bash infra/healthcheck.sh                    # confirma que los motores responden
uv run python src/cap05_concurrencia.py      # PostgreSQL (caps. 1, 3, 5)
uv run python src/cap06_documental.py        # MongoDB  (cap. 6)
```

## Reproducibilidad y datos

- Cada resultado del libro es **reproducible de extremo a extremo**: todo
  artefacto —figuras, tablas y `.dat`— se regenera ejecutando el script que lo
  produce, con semilla fija y salida determinista.
- **Medir, no proclamar.** Cada número tiene procedencia declarada: medido en
  local (registro `.dat`), sintético declarado, o citado de la literatura. Las
  gráficas se generan de forma nativa en LaTeX (`pgfplots` sobre los `.dat`,
  `TikZ` para los esquemas); las latencias se reportan por su forma, no por su
  valor absoluto (dependiente de la máquina).
- Los conjuntos de datos son públicos, de fuente canónica o repositorio abierto,
  y se citan en su primer uso.

## Capítulos

**Fundamentos del dato estructurado.** 1 Persistencia · 2 Modelo relacional ·
3 SQL · 4 Normalización · 5 Transacciones.

**La relajación del esquema.** 6 NoSQL · 7 Consistencia distribuida ·
8 Familias NoSQL (con un mapa de las especializadas: serie temporal, OLAP
columnar, NewSQL y geoespacial) · 9 Distribución.

**De la coincidencia a la similitud.** 10 Recuperación clásica · 11 Densa
frente a dispersa · 12 Embeddings neuronales (vector único y multivector de
interacción tardía, Matryoshka) · 13 Geometría métrica.

**Bases de datos vectoriales.** 14 Anatomía vectorial · 15 Índices ANN ·
16 Motores · 17 Consulta híbrida (dos etapas con reranking) · 18 Evaluación
(con cuantización float16/int8/binaria).

**Arquitecturas e integración.** 19 RAG, GraphRAG y memoria de agentes ·
20 El futuro híbrido (motor multimodelo) · 21 Caso de estudio integral.

## Licencias

Este repositorio combina dos regímenes; conviene no confundirlos:

| Directorio | Contenido | Licencia |
|---|---|---|
| `src/`, `data/`, `infra/` | Código reproducible, registro de datos e infraestructura | [MIT](src/LICENSE) — uso libre |
| — | Texto y figuras del libro (edición web) | [CC BY-NC-ND 4.0](LICENSE) — leer y compartir con atribución; sin uso comercial ni obras derivadas |

La **obra completa** —con los ejercicios de cada capítulo, sus soluciones y los
apéndices— se publica en papel, PDF y EPUB con todos los derechos reservados.

## Cómo citar

```bibtex
@book{munozpla2026basesdedatosvectoriales,
  author    = {Muñoz Plá, Manuel},
  title     = {Bases de datos: SQL · NoSQL · Vectorial},
  publisher = {qWORD.dev},
  year      = {2026},
  url       = {https://manpla.net/libros/bases-datos-vectoriales/},
}
```

Metadatos legibles por máquina: [`CITATION.cff`](CITATION.cff).
