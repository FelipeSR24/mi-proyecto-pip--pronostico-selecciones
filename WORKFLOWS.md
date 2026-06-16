# WORKFLOWS — Descripción del funcionamiento

Este documento explica, paso a paso, qué hace el código del proyecto, desde los
CSV crudos de Kaggle hasta las tres salidas que muestra la web. El recorrido
atraviesa cuatro etapas, cada una en su archivo:

| Etapa | Archivo(s) | Qué hace | Entrada → Salida |
|---|---|---|---|
| 1 · Datos | `main.py` + `utils.py` | EDA, limpieza e ingeniería de features | `datasets/*.csv` → `dataset_final.csv` + figuras del EDA |
| 2 · Modelos | `modelo.py` | Entrena y **evalúa** los modelos | `dataset_final.csv` → métricas + figuras de evaluación |
| 3 · Predicción | `prediccion.py` | **Produce** la predicción de un partido nuevo | `dataset_final.csv` + elección del usuario → 3 salidas |
| 4 · Web | `app.py` | Interfaz interactiva (Streamlit) | elección del usuario → 3 salidas en pantalla |

Cada etapa tiene abajo su **objetivo**, su **diagrama de flujo** y el **detalle
función por función**, con el mismo nivel de profundidad. Las etapas 2 y 3 parten
del mismo `dataset_final.csv`: la 2 mide qué tan bueno es el modelo; la 3 lo pone
a producir las predicciones que consume la web.

## Vista general del flujo

Visión de extremo a extremo de las cuatro etapas, de los datos crudos a la web.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ETAPA 1 · DATOS   (main.py + utils.py)                              │
│  datasets/*.csv  ──►  EDA + limpieza + features (sin fuga)           │
│                  ──►  output/dataset_final.csv  +  7 figuras del EDA │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │  output/dataset_final.csv
                ┌───────────────────┴───────────────────┐
                ▼                                        ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  ETAPA 2 · MODELOS  (modelo.py)  │  │  ETAPA 3 · PREDICCIÓN            │
│  Validación TEMPORAL (evaluar):  │  │  (prediccion.py · "el cerebro")  │
│   • Logística 1X2 (la elegida)   │  │   • estado más reciente por      │
│   • Gradient boosting (comparar) │  │     selección (ELO, forma, h2h)  │
│   • Poisson de goles             │  │   • reconstruye las 15 features  │
│  ──► métricas + figuras_modelo   │  │   • entrena con TODO el dataset  │
└──────────────────────────────────┘  └─────────────────┬────────────────┘
   (mide qué tan bueno es el modelo)                    │  predecir_partido()
                                                        ▼
                              ┌──────────────────────────────────────────┐
                              │  ETAPA 4 · WEB   (app.py · Streamlit)    │
                              │  El usuario elige local, visitante y     │
                              │  tipo de cancha  ──►  pulsa "Predecir"   │
                              └──────────────────┬───────────────────────┘
                                                 ▼
                              ┌──────────────────────────────────────────┐
                              │  SALIDA EN PANTALLA (en este orden)      │
                              │   1. Marcador más probable (top-3)       │
                              │   2. Matriz de marcadores (heatmap 7×7)  │
                              │   3. Probabilidades  1 · X · 2           │
                              └──────────────────────────────────────────┘
```

---

# Etapa 1 — Datos (`main.py` + `utils.py`)

**Objetivo:** convertir los CSV crudos de Kaggle en un único `dataset_final.csv`
listo para modelar, con la variable objetivo y las features predictoras —todas
**sin fuga de información**—, y generar las figuras del EDA. Toda la lógica vive
en `main.py` (función `main()`, con guarda `if __name__ == "__main__"`) apoyada
en funciones de `utils.py`. Sus parámetros (año de inicio, ventana de forma,
carpetas) son constantes al inicio de `main.py`.

## Diagrama de flujo — Etapa 1

```
┌────────────────────────────────────────────────────────┐
│                   FUENTES DE ENTRADA                   │
│  • datasets/results.csv      (partidos y marcadores)   │
│  • datasets/goalscorers.csv  (goleadores)              │
│  • datasets/shootouts.csv    (tandas de penales)       │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│            1. EDA INICIAL  (eda_inicial)               │
│  • Carga con validación de archivos y columnas         │
│  • Estructura, duplicados, proporción de resultados    │
│  • Países únicos, rango de fechas, torneos, neutral    │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│            2. TRANSFORMACIÓN  (transformar)            │
│  • Filtrar años ≥ 1990 + eliminar duplicados           │
│  • Unificar nombres de equipos                         │
│  • Crear variable objetivo (resultado)                 │
│  • Asignar identificador único (match_id)              │
│            INGENIERÍA DE FEATURES (sin fuga)           │
│  • Forma reciente (shift + rolling)                    │
│  • Importancia del torneo (ordinal 0–3)                │
│  • Rating ELO oficial (usa la importancia como K)      │
│  • Head-to-head (historial directo)                    │
│  • Redondeo + recuperación de goles (objetivo Poisson) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                output/dataset_final.csv                │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│            3. EDA POSTERIOR  (eda_posterior)           │
│  • Validación del dataset final (shape, nulos, target) │
│  • Análisis con contexto futbolístico (ranking, h2h)   │
│  • 7 figuras PNG  ──►  output/figuras/                 │
└────────────────────────────────────────────────────────┘
```

## Sección 1.1 — EDA inicial (`eda_inicial()`)

Objetivo: conocer los datos crudos antes de transformarlos.

- **Carga con validación.** Se leen `results.csv`, `goalscorers.csv` y
  `shootouts.csv` con `utils.cargar_dataset()`. Si falta un archivo, el
  programa termina con un mensaje claro (no un traceback). Además,
  `utils.validar_columnas()` verifica que `results` tenga las columnas
  esperadas.
- **Estructura de cada dataset.** `utils.mostrar_estructura()` imprime la forma
  (`shape`), el `info()` (tipos y no-nulos) y los nulos por columna.
- **Duplicados.** `utils.contar_duplicados()` cuenta filas repetidas exactas
  en cada dataset (evidencia de limpieza).
- **Proporción de resultados.** `utils.proporcion_resultados()` compara
  `home_score` y `away_score` para contar cuántos partidos gana el local,
  cuántos terminan en empate y cuántos gana el visitante, con su porcentaje.
- **Países únicos.** `utils.paises_unicos()` une las columnas de local y
  visitante y cuenta en cuántos partidos aparece cada selección.
- **Información adicional.** Rango de fechas, tipos de torneo más frecuentes
  (`tournament`) y distribución de la columna `neutral` (partidos en cancha
  neutral, clave para la ventaja de local).

## Sección 1.2 — Transformación (`transformar()`)

Objetivo: producir un único dataset listo para el modelo. Qué entra: los tres
DataFrames crudos. Qué sale: `dataset_final.csv`.

1. **Filtrado temporal y limpieza** (`utils.filtrar_desde`): convierte la
   fecha, descarta partidos sin marcador, **elimina filas duplicadas exactas**
   y conserva solo los jugados desde el año 1990 (`ANIO_INICIO`). Ordena por
   fecha (imprescindible para no usar información del futuro).
2. **Unificación de nombres** (`utils.unificar_nombres`): reemplaza nombres
   antiguos por los actuales (p. ej. "Serbia and Montenegro" → "Serbia") usando
   el diccionario `MAPA_NOMBRES`.
3. **Variable objetivo** (`utils.crear_target`): crea la columna `resultado`
   con tres clases (local / empate / visitante) usando `numpy.select`.
4. **Ingeniería de forma reciente** (`utils.construir_features`): calcula la
   "forma" de cada equipo SIN fuga. Para ello: pasa los partidos a formato largo
   (una fila por equipo por partido); calcula medias móviles de goles a favor, en
   contra y puntos con `groupby().transform()` combinando `shift(1)` y
   `rolling()` (el `shift(1)` impide que el partido actual entre en su propio
   promedio); vuelve a unir esas variables separadas en local y visitante y añade
   `dif_forma_pts`. Antes asigna un `match_id` único a cada partido, que todas las
   features comparten para unirse sin ambigüedad.
5. **Features adicionales**, todas SIN fuga y unidas por `match_id`:
   - **`utils.calcular_importancia`**: traduce el torneo a la variable ordinal
     `importancia` (amistoso=0 < competitivo=1 < clasificatorio=2 < mundial=3).
   - **`utils.calcular_elo`**: recorre los partidos en orden y mantiene un rating
     ELO por selección (todas arrancan en 1500) con la fórmula oficial de
     eloratings.net: `cambio = K · G · (real − esperado)`, con `K` según la
     importancia (amistoso=20 … Mundial=60) y `G` por margen de goles. Guarda el
     rating **previo** (`local_elo`, `visit_elo`, `dif_elo`) y luego lo actualiza,
     con ventaja de local de 100 puntos salvo en cancha neutral.
   - **`utils.calcular_head_to_head`**: por par de selecciones, resume los
     enfrentamientos **previos** desde la perspectiva del local: número de duelos
     (`h2h_n`), puntos promedio (`h2h_pts_local`) y diferencia de goles promedio
     (`h2h_dif_gol_local`). Sin historial usa valores neutros.

   Como la `importancia` alimenta el `K` del ELO, se calcula **antes** de
   `calcular_elo`.
6. **Redondeo**: las features se redondean antes de guardar (ELO y diferencias a
   1 decimal, forma y head-to-head a 2) para dejar el CSV legible sin afectar al
   modelo.
7. **Recuperación de goles**: se añaden `home_score` y `away_score` cruzando por
   `match_id`. **No son features** (serían fuga); se guardan como objetivo del
   modelo Poisson de la etapa 2.
8. **Guardado**: el resultado se escribe en `output/dataset_final.csv`. Ni
   `output/` ni `datasets/` se versionan (ver `.gitignore`).

## Sección 1.3 — EDA posterior (`eda_posterior()`)

Objetivo: validar el dataset final.

- Estructura del dataset final (`shape`, `info()`, nulos), distribución de la
  variable objetivo (desbalanceo), estadísticos (`describe()`) y primeras filas.
- **Análisis con contexto futbolístico:** ranking de selecciones por rendimiento
  (`utils.ranking_rendimiento`) e historial directo entre dos selecciones
  (`utils.head_to_head`, con Brasil vs. Argentina como ejemplo).
- **Siete gráficos** en `output/figuras/`:

| Figura | Qué muestra | Para qué sirve |
|---|---|---|
| `01_distribucion_target.png` | Barras de local/empate/visitante con % | Detectar desbalance de clases antes de modelar |
| `02_partidos_por_anio.png` | Partidos por año (anota la caída de 2020) | Verificar cobertura temporal y el efecto COVID-19 |
| `03_ventaja_local.png` | Resultado en cancha propia vs. neutral | Evidencia de la ventaja de local (cae ~10 pts en neutral) |
| `04_dif_forma_vs_resultado.png` | Boxplot de `dif_forma_pts` por clase | Validar que la feature principal discrimina entre clases |
| `05_top_selecciones.png` | Top 15 selecciones por rendimiento | Dar contexto: qué selecciones dominan el periodo |
| `06_evolucion_ventaja_local.png` | % de victorias locales por año | Ver si la ventaja de local se debilita con el tiempo |
| `07_goles_por_torneo.png` | Promedio de goles por partido por torneo | Comparar qué competencias son más ofensivas |

## Prevención de fuga de información (data leakage)

Es el punto metodológico más importante de toda la etapa. Todas las variables se
calculan únicamente con partidos **anteriores** a cada partido: la forma usa
`shift(1)`, el ELO guarda el rating previo (se actualiza después), el head-to-head
solo mira duelos anteriores e `importancia` es un dato del calendario. Los goles
(`home_score`/`away_score`) se guardan solo como objetivo, nunca como entrada. Sin
esta disciplina, el modelo "vería" el resultado que intenta predecir y daría
métricas engañosamente altas.

---

# Etapa 2 — Modelos (`modelo.py`)

**Objetivo:** entrenar los modelos que predicen el partido y **evaluarlos con
honestidad** para decidir cuáles usar. Esta etapa no produce las predicciones que
ve el usuario (eso es la etapa 3): aquí se mide *qué tan bueno es cada modelo*.
Se ejecuta con `python modelo.py`.

## Diagrama de flujo — Etapa 2

```
┌────────────────────────────────────────────────────────┐
│              output/dataset_final.csv                  │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│   DIVISIÓN TEMPORAL  (separar_temporal)                │
│   • Entrenamiento: partidos < 2022   (~86%)            │
│   • Validación:    partidos ≥ 2022   (~14%)            │
│   (imita predecir el futuro; no es división aleatoria) │
└───────────────────────────┬────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ A.1 LOGÍST.  │   │ A.2 BOOSTING │   │ B · POISSON      │
│ 1X2 (escala) │   │ 1X2 (árboles)│   │ 2 reg. de goles  │
└──────┬───────┘   └──────┬───────┘   └────────┬─────────┘
       │                  │                    │ matriz_marcadores
       ▼                  ▼                    ▼ resumen_desde_matriz
┌────────────────────────────────┐   ┌──────────────────────┐
│  EVALUAR (sobre validación):   │   │ EVALUAR_POISSON:     │
│  accuracy · log-loss · Brier   │   │ MAE de goles +       │
│  vs baseline "siempre local"   │   │ accuracy 1X2 derivada│
└──────────────┬─────────────────┘   └──────────┬───────────┘
               └────────────┬───────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│  COMPARAR_MODELOS  +  9 figuras  ──► output/figuras_modelo/ │
│  Decisión: la logística (iguala al boosting, interpretable) │
└────────────────────────────────────────────────────────┘
```

## Sección 2.1 — Preparación de los datos

- **`cargar_datos`** — qué entrega: el `dataset_final.csv` cargado, con `neutral`
  convertido de booleano a 0/1 (los modelos no aceptan True/False).
- **`separar_temporal`** — qué recibe: el dataset; qué entrega: `X_train`,
  `X_val`, `y_train`, `y_val` y la lista de features. La división es **por
  fecha** (entrena con `< 2022`, valida con `≥ 2022`): predecir un partido es
  predecir el futuro, así que se evalúa con el futuro. Una división aleatoria
  inflaría las métricas porque el modelo "vería" datos futuros al entrenar. Se
  excluyen de las entradas los identificadores, el objetivo y los goles.
- **`baseline_accuracy`** — la vara mínima a superar: acertar siempre la clase
  más común ("local", ≈ 48 %). Cualquier modelo útil debe quedar por encima.

## Sección 2.2 — Los modelos

- **Modelo A.1 — Regresión logística** (`entrenar_logistica`): el clasificador
  base, interpretable. Se monta en un *pipeline* que primero **escala** las
  features con `StandardScaler` (la logística es sensible a la escala: el ELO vale
  ~1500 y la forma ~1.5) y luego entrena. Es el modelo elegido para producción.
- **Modelo A.2 — Gradient boosting** (`entrenar_boosting`): el clasificador
  potente (árboles que se corrigen en cadena). **No** necesita escalado. Su papel
  es servir de control: comprobar si la complejidad mejora el resultado. No lo
  hace (empata con la logística), lo que justifica quedarse con la simple.
- **Modelo B — Poisson** (`entrenar_poisson`, `matriz_marcadores`,
  `resumen_desde_matriz`): dos regresiones de Poisson predicen los goles
  esperados del local y del visitante (las "lambdas"). Con ellas,
  `matriz_marcadores` construye la matriz 7×7 (probabilidad de cada marcador, vía
  `np.outer` de las dos distribuciones de Poisson), y `resumen_desde_matriz`
  deriva el marcador más probable (celda máxima) y las probabilidades 1X2
  (sumando las regiones triangular inferior / diagonal / superior).

## Sección 2.3 — La evaluación

- **`evaluar`** (clasificadores): qué recibe: un modelo entrenado y la
  validación; qué entrega: un diccionario con **accuracy** (% de aciertos),
  **log-loss** (penaliza estar "seguro y equivocado") y **Brier** (calibración:
  si dice "70 %", ¿pasa el 70 %?). El detalle de cada métrica está en la cabecera
  de `modelo.py`. `brier_multiclase` adapta el Brier (que es binario) a las 3
  clases promediando uno-contra-resto.
- **`evaluar_poisson`** (Poisson): mide el **MAE de goles** (cuántos goles de
  error en promedio) y deriva una **accuracy 1X2** desde la matriz. Que esa
  accuracy coincida con la del clasificador valida que el sistema es coherente.
- **`comparar_modelos`**: imprime una tabla con las métricas de los modelos lado
  a lado para decidir cuál usar.

## Sección 2.4 — Figuras de evaluación

Se generan nueve gráficos en `output/figuras_modelo/` (funciones `graficar_*`):
matriz de confusión y curva de calibración de cada clasificador, importancia de
features (logística) y por permutación (boosting), el caso del empate por paridad,
la comparación de modelos contra el baseline, el chequeo de sobreajuste y una
matriz de marcadores de ejemplo. Son la evidencia visual para la sustentación.

## Conclusión de la etapa

Los dos clasificadores empatan (≈ 60 % de accuracy, por encima del baseline de
48 %) y el Poisson da una accuracy 1X2 coherente con ellos. **Para producción se
elige la regresión logística**: iguala al boosting siendo interpretable y está
bien calibrada (navaja de Occam con evidencia). El empate es la clase más difícil
de predecir, un fenómeno conocido del fútbol que el proyecto documenta en lugar de
ocultar.

---

# Etapa 3 — Predicción (`prediccion.py`)

**Objetivo:** predecir un partido que **aún no existe** (el que elige el usuario).
Es "el cerebro" que consume la web. El reto que resuelve: los modelos se entrenan
con partidos que ya tienen sus features calculadas, pero un partido nuevo no
tiene ELO, ni forma, ni head-to-head. Esta etapa **reconstruye esas features** a
partir del estado más reciente de cada selección y entrega las tres salidas.

## Diagrama de flujo — Etapa 3

```
┌────────────────────────────────────────────────────────┐
│              output/dataset_final.csv                  │
└──────────────┬───────────────────────┬─────────────────┘
               ▼                        ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│ estado_actual_equipos     │  │ entrenar_modelos              │
│ recorre el historial y    │  │ logística + 2 Poisson         │
│ guarda por selección:     │  │ con TODO el dataset           │
│  • ELO final              │  │ (para producir, no evaluar)   │
│  • forma últimos 5        │  └───────────────┬──────────────┘
│  • historial h2h          │                  │
└──────────────┬────────────┘                  │
               │   ┌──────────────────────────┐│
  usuario ───► │   │ local, visitante,        ││
  elige        │   │ cancha neutral,          ││
               ▼   │ importancia (fija)       ││
┌──────────────────┴──────────┐               ││
│ construir_features_partido  │◄──────────────┘│
│ arma la fila de 15 features │                │
│ del partido hipotético      │                │
└──────────────┬──────────────┘                │
               ▼                                ▼
┌────────────────────────────────────────────────────────┐
│              predecir_partido (todo-en-uno)            │
│  ├─ predecir_1x2        ──► probabilidades local/X/visit│
│  └─ predecir_marcador   ──► matriz + marcador probable  │
└────────────────────────────────────────────────────────┘
```

## Sección 3.1 — Reconstruir el estado de cada selección

- **`estado_actual_equipos`** — qué recibe: el dataset; qué entrega: un
  diccionario `{equipo: {elo, forma_gf, forma_gc, forma_pts}}` más el historial
  de duelos entre cada par. Recorre los partidos en **orden cronológico** y, con
  la misma fórmula oficial del ELO de la etapa 1 (factores K y G, ventaja de
  local, base 1500), calcula el rating **final** de cada selección (su estado más
  reciente) y la media de sus últimos `VENTANA_FORMA` (5) partidos. La subfunción
  `media_ultimos` hace ese promedio de los últimos cinco.
- **`_head_to_head_actual`** — qué recibe: el historial y dos equipos; qué
  entrega: `h2h_n`, `h2h_pts_local` y `h2h_dif_gol_local` de esos dos rivales,
  desde la perspectiva del local actual (con valores neutros si nunca se han
  enfrentado).

Se usa el estado **más reciente** (no un promedio largo): el ELO ya es
acumulativo y la forma capta el momento actual; promediar periodos largos
difuminaría esa señal. Esta decisión está respaldada por la literatura de
predicción deportiva.

## Sección 3.2 — Construir el partido hipotético

- **`construir_features_partido`** — qué recibe: el estado, los dos equipos, si
  la cancha es neutral y la importancia; qué entrega: un DataFrame de **una fila**
  con las 15 features, en el mismo orden y nombres que usó el entrenamiento. Es la
  pieza que traduce "Brasil vs. Argentina" en algo que los modelos entienden. Si
  algún equipo no tiene datos históricos, avisa con un error claro.

## Sección 3.3 — Entrenar y predecir

- **`entrenar_modelos`** — entrena el clasificador (logística) y los dos Poisson
  con **todo** el dataset. Aquí no se divide en validación: esa medición ya se
  hizo en la etapa 2; para *producir* predicciones conviene usar todos los datos.
- **`predecir_1x2`** — SALIDA 1: devuelve `{local, empate, visitante}` (suma 100 %)
  del clasificador.
- **`predecir_marcador`** — SALIDAS 2 y 3: devuelve los goles esperados de cada
  equipo, la matriz 7×7 de marcadores, el marcador más probable y su probabilidad.
- **`predecir_partido`** — función "todo en uno" que la web llama directamente:
  combina las tres salidas en un único diccionario.

El módulo se puede ejecutar solo (`python prediccion.py`) e imprime un ejemplo
(Brasil vs. Argentina) para comprobar el flujo de punta a punta.

---

# Etapa 4 — Web (`app.py`)

**Objetivo:** exponer las predicciones en una interfaz interactiva con identidad
visual del Mundial 2026. La web es una capa fina: recoge la elección del usuario,
llama a `prediccion.predecir_partido` y dibuja el resultado; toda la lógica de
cálculo vive en la etapa 3. Se ejecuta con `streamlit run app.py`.

## Diagrama de flujo — Etapa 4

```
┌────────────────────────────────────────────────────────┐
│  CARGA ÚNICA  (cargar_todo, con @st.cache_resource)    │
│  lee dataset · estado_actual_equipos · entrenar_modelos│
│  (solo la primera vez; luego queda en cache)           │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│  CONFIGURA EL PARTIDO  (recuadro)                      │
│  • Selección local      (solo las 48 del Mundial 2026) │
│  • Selección visitante  (resuelve alias de nombres)    │
│  • ☑ Cancha neutral                                    │
│  • Validación: deben ser dos equipos distintos         │
└───────────────────────────┬────────────────────────────┘
                            │  pulsa "Predecir resultado"
                            ▼
┌────────────────────────────────────────────────────────┐
│  predecir_partido(local, visitante, neutral, MUNDIAL)  │
│  (la importancia es fija: todo partido es de Mundial)  │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│  RESULTADOS DE LA PREDICCIÓN  (recuadro, en orden)     │
│   1. Marcador más probable   (podio top-3)             │
│   2. Matriz de marcadores    (heatmap Plotly 7×7)      │
│   3. Probabilidades 1·X·2    (barra horizontal)        │
│   + aviso de alcance (es un apoyo, no una certeza)     │
└────────────────────────────────────────────────────────┘
```

## Sección 4.1 — Carga y catálogo de equipos

- **`cargar_todo`** (con `@st.cache_resource`): lee `dataset_final.csv`, calcula
  el estado de cada selección (`estado_actual_equipos`) y entrena los modelos
  (`entrenar_modelos`) **una sola vez**. El cache evita rehacer ese trabajo en
  cada interacción (por eso la primera carga tarda y las siguientes son
  inmediatas).
- **`MUNDIAL_2026`** y **`ALIAS`**: la web solo ofrece las **48 selecciones del
  Mundial 2026**. `MUNDIAL_2026` mapea el nombre interno del dataset al nombre en
  español y al código de bandera; `ALIAS` (con `_canonico`) resuelve variantes de
  nombre. **`equipos_mundial`** filtra el catálogo del dataset a esas 48.

## Sección 4.2 — Configuración del usuario

Dentro de un recuadro (`st.container(border=True)`), el usuario elige la
**selección local** y la **visitante** (menús que muestran el nombre en español
vía `format_func` pero usan el nombre interno para predecir) y marca si la
**cancha es neutral**. La web exige que las dos selecciones sean **distintas**
antes de continuar. La **importancia es fija** (`IMPORTANCIA_PARTIDO` = un partido
de Mundial): no se pide al usuario porque, al ser todos partidos de Mundial,
tendría siempre el mismo valor y no cambiaría el cálculo.

## Sección 4.3 — Del clic al render

Al pulsar *Predecir resultado* se llama a `prediccion.predecir_partido`, que
devuelve las tres salidas en un diccionario. Dentro de un segundo recuadro se
dibujan, en este orden (cada sección con su encabezado numerado, `eyebrow_html`):

1. **Marcador más probable** — `top3_marcadores` ordena la matriz y `podium_html`
   muestra un podio con los tres marcadores más probables (el #1 resaltado en
   dorado, con el tipo de resultado que representa), junto con los goles esperados
   (las lambdas del Poisson) de cada selección.
2. **Matriz de marcadores** — `figura_matriz` genera un mapa de calor
   **interactivo de Plotly** (7×7, de 0-0 a 6-6) con el marcador más probable
   resaltado; se puede pasar el cursor para ver el detalle y hacer zoom.
3. **Probabilidades del resultado** — `barra_html` dibuja una barra horizontal
   segmentada `1 · X · 2` (azul/verde/rojo) con su leyenda.

Cierra con un **aviso de alcance**: son estimaciones estadísticas y el fútbol
tiene un fuerte componente de azar; es un apoyo a la lectura del partido, no una
certeza. Las funciones `_md`, `banner_html`, `pick_html` y el bloque `CSS`
componen la identidad visual (tipografía Saira Condensed + Inter y la paleta
tricolor del Mundial).
