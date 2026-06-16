# Proyecto de pronóstico de partidos de selecciones

> **Asignatura:** Fundamentos de Ciencia de Datos
> **Autores:** Cesar Estiven Moreno Betancur y Felipe Sandoval Ramírez

Proyecto de ingeniería y ciencia de datos que **predice el resultado de un
partido entre dos selecciones** (gana local / empate / gana visitante) y, a
partir de un modelo de goles, estima el **marcador más probable** y una **matriz
de marcadores**, todo expuesto en una **web interactiva** orientada al Mundial
2026.

El proyecto cubre el ciclo completo de ciencia de datos: desde el análisis
exploratorio y la preparación de los datos, pasando por el modelado y su
evaluación, hasta un producto final navegable.

## Las tres salidas del producto

Para un partido elegido por el usuario (local, visitante, fase y tipo de cancha),
la web entrega:

1. **Probabilidades del resultado** — gana local / empate / gana visitante.
2. **Matriz de marcadores** — la probabilidad de cada marcador posible.
3. **Marcador más probable** — con un ranking de los tres más probables.

## Las cuatro etapas (y sus archivos)

1. **Datos** (`main.py` + `utils.py`): cargan los CSV de Kaggle y construyen el
   dataset final, con la variable objetivo y las features predictoras —todas
   **sin fuga de información**—: forma reciente, rating **ELO** (fórmula oficial
   de eloratings.net), **head-to-head** e **importancia** del torneo. Genera
   también los gráficos del EDA.
2. **Modelos** (`modelo.py`): entrena y evalúa los modelos con validación
   **temporal**. Un clasificador 1X2 (regresión logística, comparada contra
   gradient boosting) para las probabilidades, y un modelo **Poisson** de goles
   para el marcador y la matriz. Se mide con accuracy, log-loss y Brier.
3. **Predicción** (`prediccion.py`): el "cerebro" que reconstruye las features de
   un partido nuevo a partir del estado más reciente de cada selección y entrega
   las tres salidas. Es lo que consume la web.
4. **Web** (`app.py`): interfaz interactiva en Streamlit con identidad visual del
   Mundial 2026; el usuario elige las selecciones y ve las tres salidas.

El detalle paso a paso está en [`WORKFLOWS.md`](WORKFLOWS.md) y el origen y
diccionario de los datos en [`DATABASE.md`](DATABASE.md).

## Estructura del proyecto

```
mi-proyecto-pip/
├── main.py            # etapa 1: pipeline de datos (EDA + dataset final)
├── utils.py           # funciones auxiliares y gráficos del EDA
├── modelo.py          # etapa 2: entrenamiento y evaluación de los modelos
├── prediccion.py      # etapa 3: lógica de predicción de un partido nuevo
├── app.py             # etapa 4: web interactiva (Streamlit)
├── requirements.txt   # librerías con versiones acotadas
├── setup.cfg          # configuración de flake8 (estilo del código)
├── .gitignore
├── README.md          # este archivo
├── WORKFLOWS.md       # detalle de lo que hace el código
└── DATABASE.md        # descripción, origen y licencia de los datos
```

Al trabajar en local existen además dos carpetas que **no se versionan**
(están en `.gitignore` por ser pesadas y regenerables):

- `datasets/` — los CSV crudos de Kaggle, que colocas tú manualmente
  (ver `DATABASE.md`).
- `output/` — el `dataset_final.csv`, las figuras del EDA (`output/figuras/`) y
  las figuras de evaluación de los modelos (`output/figuras_modelo/`), que se
  generan al ejecutar `python main.py` y `python modelo.py`.

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

4. **Etapa 1 — genera el dataset** (necesario para todo lo demás):

   ```bash
   python main.py
   ```

   Crea `output/dataset_final.csv` (datos listos para el modelo) y las figuras
   del EDA en `output/figuras/`.

5. **Etapa 2 — entrena y evalúa los modelos** (opcional, para ver las métricas):

   ```bash
   python modelo.py
   ```

   Imprime las métricas de cada modelo (accuracy, log-loss, Brier) y guarda las
   figuras de evaluación en `output/figuras_modelo/`.

6. **Etapa 4 — abre la web interactiva** (el producto final):

   ```bash
   streamlit run app.py
   ```

   Se abre sola en el navegador. Elige las dos selecciones, la fase y el tipo de
   cancha, y pulsa "Predecir" para ver las tres salidas. (La web entrena los
   modelos al arrancar, así que la primera carga tarda unos segundos.)

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
| `resultado` *(objetivo)* | Lo que el clasificador predecirá: `local` / `empate` / `visitante`. | Comparando `home_score` y `away_score`. |
| `home_score`, `away_score` *(objetivo del Poisson)* | Goles de cada equipo. **No son features** (serían fuga); se usan solo como objetivo del modelo de goles. | Directo del dato crudo. |

### Contexto del partido

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `neutral` (0/1) | Si se jugó en cancha neutral (sin ventaja de local). | Del dato crudo; booleano → 0/1. |
| `importancia` (0–3) | Cuánto hay en juego: amistoso=0, competitivo=1, clasificatorio=2, mundial=3. | Mapeo del nombre del torneo (`tournament`) a una escala ordinal. |

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
Implementa la fórmula **oficial de eloratings.net** (el ranking mundial): el
cambio de rating es `K · G · (resultado − esperado)`, donde `K` depende de la
importancia del torneo (20 amistoso, 30 competitivo, 40 clasificatoria, 60
Mundial) y `G` amplifica según el margen de goles (ganar por mucho sube más, con
rendimientos decrecientes). Se valida contra el ranking real: las potencias del
top-10 coinciden con eloratings.net.

| Columna | Qué significa | Cómo se calcula |
|---|---|---|
| `local_elo`, `visit_elo` | Nivel histórico de cada selección (toda arranca en 1500; rango ~810–2220). | Rating **previo** al partido; sube al ganar y baja al perder según lo sorpresivo del resultado, el margen de goles y la importancia del torneo (con ventaja de local de 100 puntos salvo en cancha neutral). |
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

## Los modelos (etapa 2)

`modelo.py` entrena y evalúa los modelos con **validación temporal** (entrena con
partidos anteriores a 2022 y valida con los posteriores, para imitar la
predicción real del futuro). Se usan dos modelos complementarios:

| Modelo | Para qué | Resultado en validación |
|---|---|---|
| **Regresión logística** (1X2) | Probabilidades de local / empate / visitante | Accuracy ≈ 60 %, bien calibrada |
| **Gradient boosting** (1X2) | Alternativa potente, solo para comparar | Empata con la logística → se elige la logística por ser interpretable |
| **Poisson** (goles) | Marcador más probable y matriz de marcadores | MAE ≈ 1 gol; sus probabilidades 1X2 coinciden con las de la logística |

Las métricas usadas son **accuracy** (% de aciertos), **log-loss** y **Brier**
(calidad de las probabilidades, no solo del acierto). El baseline a superar es
predecir siempre "local" (≈ 48 %). El empate es la clase más difícil de predecir
(fenómeno conocido del fútbol). Para producción se usa la regresión logística por
igualar al boosting siendo interpretable y estar bien calibrada.

## Control de versiones

El proyecto se versiona con Git y GitHub siguiendo este flujo: la rama `main`
contiene las versiones estables; cada mejora se desarrolla en una rama propia
con commits pequeños y mensajes descriptivos (convención *Conventional Commits*:
`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`), y se integra a `main` mediante
Pull Request.

## Licencia de los datos

Los datos provienen de Kaggle (usuario martj42) y se usan con fines
académicos; consulta la licencia vigente en la página del dataset (ver
`DATABASE.md`). Los CSV crudos no se versionan en este repositorio.
