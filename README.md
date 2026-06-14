# Proyecto de pronóstico de partidos de selecciones

> **Asignatura:** Fundamentos de Ciencia de Datos
> **Autores:** Cesar Estiven Moreno Betancur y Felipe Sandoval Ramírez
> **Entrega:** 1 — EDA y preparación de datos

Proyecto de ingeniería y ciencia de datos cuyo objetivo final es **predecir el
resultado de un partido entre dos selecciones** (gana local / empate / gana
visitante) y, a partir de ahí, estimar probabilidades de cara al Mundial 2026.

Este repositorio cubre por ahora el **EDA y la preparación de datos**. El modelo
de pronóstico se construirá en una etapa posterior.

## Qué hace

El script `main.py` ejecuta un pipeline en tres secciones:

1. **EDA inicial:** carga y valida los datasets, muestra su estructura, cuenta
   duplicados, la proporción de resultados (local/empate/visitante), la lista
   de países y otra información exploratoria.
2. **Transformación:** filtra los partidos desde el año 1990, elimina
   duplicados, unifica nombres de países, crea la variable objetivo y construye
   las variables predictoras —**todas sin fuga de información**— en un único
   dataset listo para modelar: la "forma reciente" de cada equipo, su rating
   **ELO** acumulado, el **historial directo** (head-to-head) entre ambas
   selecciones y la **importancia** del torneo.
3. **EDA posterior:** revisa el dataset final, calcula análisis con contexto
   futbolístico (ranking de selecciones por rendimiento e historial directo
   entre dos equipos) y genera **siete gráficos** en `output/figuras/`
   (distribución del objetivo, partidos por año, ventaja de local en cancha
   propia vs. neutral, poder discriminante de la feature principal, top de
   selecciones, evolución de la ventaja de local y goles por torneo).

El detalle paso a paso está en [`WORKFLOWS.md`](WORKFLOWS.md) y el origen y
diccionario de los datos en [`DATABASE.md`](DATABASE.md).

## Estructura del proyecto

```
mi-proyecto-pip/
├── main.py            # pipeline principal (las 3 secciones)
├── utils.py           # funciones auxiliares y gráficos
├── requirements.txt   # librerías con versiones acotadas
├── .gitignore
├── README.md          # este archivo
├── WORKFLOWS.md       # detalle de lo que hace el código
└── DATABASE.md        # descripción, origen y licencia de los datos
```

Al trabajar en local existen además dos carpetas que **no se versionan**
(están en `.gitignore` por ser pesadas y regenerables):

- `datasets/` — los CSV crudos de Kaggle, que colocas tú manualmente
  (ver `DATABASE.md`).
- `output/` — el `dataset_final.csv` y las figuras del EDA, que se generan
  automáticamente al ejecutar `python main.py`.

## Requisitos

- Python 3.11 (probado con 3.11.9; debería funcionar con 3.10+)
- Las librerías de `requirements.txt` (pandas, numpy, matplotlib)

## Cómo usarlo

1. Crea y activa un entorno virtual:

   ```bash
   python -m venv venv

   # Windows (cmd / PowerShell):
   venv\Scripts\activate

   # Linux / macOS:
   source venv/bin/activate
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Descarga el dataset de Kaggle
   [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
   y coloca los CSV en una carpeta `datasets/` en la raíz del proyecto
   (ver `DATABASE.md`). Verifica que los nombres coincidan exactamente con
   los esperados.

4. Ejecuta el pipeline:

   ```bash
   python main.py
   ```

Al terminar, se generan `output/dataset_final.csv` (datos listos para el
modelo) y las figuras del EDA en `output/figuras/`.

## Configuración

Todos los parámetros ajustables están reunidos como constantes al inicio de
`main.py` (un único lugar para cambiarlos):

| Constante       | Valor  | Descripción                                  |
|-----------------|--------|----------------------------------------------|
| `ANIO_INICIO`   | `1990` | Año desde el cual se conservan partidos       |
| `VENTANA_FORMA` | `5`    | Partidos previos para la forma reciente       |
| `CARPETA_DATOS` | `datasets` | Carpeta de los CSV crudos                 |
| `CARPETA_SALIDA`| `output`   | Carpeta de salida (dataset y figuras)     |

El año de inicio se fijó en **1990** porque desde esa década las métricas del
fútbol son estables y representativas del fútbol actual (ver el análisis en
`DATABASE.md`).

## Diccionario del dataset final

`output/dataset_final.csv` tiene una fila por partido. Las columnas se agrupan
por familia. **Regla de oro:** toda variable predictora responde a *"¿qué se
sabía ANTES de este partido?"*; ninguna usa información del propio partido ni de
partidos futuros (ver "Prevención de fuga" en `WORKFLOWS.md`).

### Identificadores y objetivo (no son variables del modelo)

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `match_id` | Identificador único del partido; solo sirve para unir las features. | Número de fila tras ordenar los partidos por fecha. |
| `date`, `home_team`, `away_team` | Fecha y selecciones. Describen el partido, no se usan como entrada. | Directo del dato crudo. |
| `resultado` *(objetivo)* | Lo que el modelo predecirá: `local` / `empate` / `visitante`. | Comparando `home_score` y `away_score`. |

### Contexto del partido

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `neutral` (0/1) | Si se jugó en cancha neutral (sin ventaja de local). | Del dato crudo; booleano → 0/1. |
| `importancia` (0–3) | Cuánto hay en juego: amistoso=0, clasificatorio=1, competitivo=2, mundial=3. | Mapeo del nombre del torneo (`tournament`) a una escala ordinal. |

### Forma reciente

Promedio de los **últimos 5 partidos** de cada equipo, *sin* incluir el partido
actual (`shift(1)` + `rolling(5).mean()`). El sufijo `local_`/`visit_` indica el
equipo; cada uno usa sus propios últimos 5 partidos (en cualquier cancha).

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `local_forma_gf`, `visit_forma_gf` | Pegada ofensiva: goles a favor por partido. Más = mejor. | Media de los 5 partidos anteriores. |
| `local_forma_gc`, `visit_forma_gc` | Solidez defensiva: goles en contra por partido. **Menos = mejor.** | Media de los 5 partidos anteriores. |
| `local_forma_pts`, `visit_forma_pts` | Racha de resultados: puntos por partido (3/1/0), de 0 a 3. | Media de los 5 partidos anteriores. |
| `dif_forma_pts` | Quién llega en mejor forma (positivo = el local). | `local_forma_pts − visit_forma_pts`. |

### ELO (fuerza acumulada a largo plazo)

A diferencia de la forma (solo 5 partidos), el ELO resume **toda** la trayectoria.

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `local_elo`, `visit_elo` | Nivel histórico de cada selección (toda arranca en 1500; rango típico ~1300–2100). | Rating **previo** al partido; sube al ganar y baja al perder según lo sorpresivo del resultado (con ventaja de local salvo en cancha neutral). |
| `dif_elo` | Diferencia de nivel entre ambas. Es la variable más informativa. | `local_elo − visit_elo`. |

### Historial directo (head-to-head)

Resume los enfrentamientos **previos** entre esas dos mismas selecciones, desde
la perspectiva del local actual. Si no hay historial, se rellena con valores
neutros y `h2h_n=0` avisa de que el dato es poco fiable.

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `h2h_n` | Cuántos duelos previos había entre ambas (mide la confianza, no la fuerza). | Conteo de enfrentamientos anteriores. |
| `h2h_pts_local` | Puntos promedio del local contra ese rival (0–3): ¿lo domina o le cuesta? | Media sobre los duelos previos; relleno neutro = 1.5 si no hay. |
| `h2h_dif_gol_local` | Diferencia de goles promedio a favor del local en esos duelos. | Media sobre los duelos previos; relleno neutro = 0 si no hay. |

## Control de versiones

El proyecto se versiona con Git y GitHub siguiendo este flujo: la rama `main`
contiene las versiones estables; cada mejora se desarrolla en una rama propia
(p. ej. `mejoras/auditoria-entrega-1`) con commits pequeños y mensajes
descriptivos (convención *Conventional Commits*: `feat:`, `fix:`,
`refactor:`, `docs:`, `chore:`), y se integra a `main` mediante Pull Request.
Cada entrega se marca con una etiqueta (`entrega-1`).

## Licencia de los datos

Los datos provienen de Kaggle (usuario martj42) y se usan con fines
académicos; consulta la licencia vigente en la página del dataset (ver
`DATABASE.md`). Los CSV crudos no se versionan en este repositorio.
