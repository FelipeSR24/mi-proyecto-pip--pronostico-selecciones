# WORKFLOWS — Descripción del funcionamiento

Este documento explica, paso a paso, qué hace el código del proyecto. El flujo
vive en `main.py` (función `main()`, con guarda `if __name__ == "__main__"`)
y se apoya en funciones de `utils.py`.

## Vista general del flujo

```
┌────────────────────────────────────────────────────────┐
│                   FUENTES DE ENTRADA                   │
│  • datasets/results.csv                                │
│  • datasets/goalscorers.csv                            │
│  • datasets/shootouts.csv                              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     1. EDA INICIAL                     │
│  • Validación de consistencia                          │
│  • Integración y cruce de datos                        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                    RESULTS VALIDADO                    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   2. TRANSFORMACIÓN                    │
│  • Filtrar años ≥ 1990                                 │
│  • Eliminar registros duplicados                       │
│  • Unificar nombres de equipos                         │
│  • Crear variable objetivo (target)                    │
│  • Asignar identificador único (match_id)              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 INGENIERÍA DE FEATURES                 │
│                  (Evitando data leakage)               │
│  • Cálculo de forma reciente (shift + rolling)         │
│  • Cálculo de puntuación ELO                           │
│  • Historial de enfrentamientos (Head-to-Head)         │
│  • Nivel de importancia del partido                    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                output/dataset_final.csv                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                    3. EDA POSTERIOR                    │
│  • Extracción de estadísticos descriptivos             │
│  • Análisis de distribución de variables               │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  ARTEFACTOS DE SALIDA                  │
│  • 4 Figuras gráficas en formato PNG                   │
│    Ruta: (output/figuras/*.png)                        │
└────────────────────────────────────────────────────────┘
```

## Sección 1 — EDA inicial (`eda_inicial()`)

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

## Sección 2 — Transformación (`transformar()`)

Objetivo: producir un único dataset listo para el modelo.

1. **Filtrado temporal y limpieza** (`utils.filtrar_desde`): convierte la
   fecha, descarta partidos sin marcador, **elimina filas duplicadas exactas**
   y conserva solo los jugados desde el año 1990 (`ANIO_INICIO`). Ordena por
   fecha (imprescindible para no usar información del futuro).
2. **Unificación de nombres** (`utils.unificar_nombres`): reemplaza nombres
   antiguos por los actuales (p. ej. "Serbia and Montenegro" → "Serbia") usando
   el diccionario `MAPA_NOMBRES`. Puedes ampliarlo si tu EDA revela más casos.
3. **Variable objetivo** (`utils.crear_target`): crea la columna `resultado`
   con tres clases (local / empate / visitante) usando `numpy.select`.
4. **Ingeniería de variables** (`utils.construir_features`): calcula la "forma
   reciente" de cada equipo SIN fuga de información. Para ello:
   - Pasa los partidos a formato largo (una fila por equipo por partido).
   - Calcula medias móviles de goles a favor, en contra y puntos con
     `groupby().transform()` combinando `shift(1)` y `rolling()`. El `shift(1)`
     garantiza que el partido actual no entre en su propio promedio.
   - Vuelve a unir esas variables al partido, separadas en local y visitante, y
     añade `dif_forma_pts` (diferencia de forma entre ambos).
   - Descarta las primeras filas de cada equipo (sin historial todavía).

   Antes de este paso, `transformar()` asigna un `match_id` único a cada partido;
   `construir_features` lo **respeta si ya existe**, de modo que todas las
   features (forma, ELO, head-to-head, importancia) comparten el mismo
   identificador y se unen sin ambigüedad.
5. **Features adicionales** (Fase 4), todas calculadas SIN fuga (solo con
   información anterior a cada partido) y unidas al dataset por `match_id`:
   - **`utils.calcular_importancia`**: traduce el torneo a una variable ordinal
     `importancia` (amistoso=0 < clasificatorio=1 < competitivo=2 < mundial=3).
     Es un dato del propio partido, conocido antes del pitido.
   - **`utils.calcular_elo`**: recorre los partidos en orden cronológico y
     mantiene un rating ELO por selección (todas arrancan en 1500), usando la
     fórmula **oficial de eloratings.net**: `cambio = K · G · (real − esperado)`.
     El peso `K` depende de la importancia del torneo (amistoso=20 … Mundial=60)
     y `G` amplifica según el margen de goles (ganar por más mueve más el rating,
     con rendimientos decrecientes). Para cada partido guarda el rating **previo**
     de ambos equipos (`local_elo`, `visit_elo`, `dif_elo`) y luego lo actualiza,
     con ventaja de local de 100 puntos salvo en cancha neutral. Capta la fuerza
     acumulada a largo plazo; su top-10 coincide con el ranking mundial real.
   - **`utils.calcular_head_to_head`**: por par de selecciones, resume los
     enfrentamientos **previos** desde la perspectiva del local: número de
     duelos (`h2h_n`), puntos promedio (`h2h_pts_local`) y diferencia de goles
     promedio (`h2h_dif_gol_local`). Sin historial usa valores neutros.

   Como la `importancia` alimenta el `K` del ELO, se calcula y se une a los
   partidos **antes** de llamar a `calcular_elo`.
6. **Redondeo**: las features se calculan con muchos decimales que no aportan
   precisión útil; antes de guardar se redondean (ELO y diferencias a 1 decimal,
   forma y head-to-head a 2) para dejar el CSV legible sin afectar al modelo.
7. **Guardado**: el resultado se escribe en `output/dataset_final.csv`.
   Ni `output/` ni `datasets/` se versionan (ver `.gitignore`): los CSV
   crudos son pesados y se descargan de Kaggle, y la salida es regenerable
   ejecutando `python main.py`.

## Sección 3 — EDA posterior (`eda_posterior()`)

Objetivo: validar el dataset final.

- Estructura del dataset final (`shape`, `info()`, nulos).
- Distribución de la variable objetivo (para detectar desbalanceo de clases).
- Estadísticos de las variables (`describe()`).
- Primeras filas (`head()`).
- **Análisis adicionales con contexto futbolístico:**
  - Ranking de selecciones por rendimiento (puntos por partido), filtrando
    equipos con pocos partidos (`utils.ranking_rendimiento`).
  - Historial directo entre dos selecciones (`utils.head_to_head`), con
    Brasil vs. Argentina como ejemplo impreso en consola.
- **Siete gráficos** guardados en `output/figuras/`:

| Figura | Qué muestra | Para qué sirve |
|---|---|---|
| `01_distribucion_target.png` | Barras de local/empate/visitante con % | Detectar desbalance de clases antes de modelar |
| `02_partidos_por_anio.png` | Partidos por año (anota la caída de 2020) | Verificar cobertura temporal y el efecto COVID-19 |
| `03_ventaja_local.png` | Resultado en cancha propia vs. neutral | Evidencia visual de la ventaja de local (cae ~10 pts en cancha neutral) |
| `04_dif_forma_vs_resultado.png` | Boxplot de `dif_forma_pts` por clase | Validar que la feature principal discrimina entre clases |
| `05_top_selecciones.png` | Top 15 selecciones por rendimiento | Dar contexto: qué selecciones dominan el periodo |
| `06_evolucion_ventaja_local.png` | % de victorias locales por año | Ver si la ventaja de local se debilita con el tiempo |
| `07_goles_por_torneo.png` | Promedio de goles por partido por torneo | Comparar qué competencias son más ofensivas |

## Prevención de fuga de información (data leakage)

Es el punto metodológico más importante. Todas las variables de forma reciente
se calculan únicamente con partidos anteriores a la fecha de cada partido. Si
no se usara `shift(1)`, el modelo "vería" el resultado que intenta predecir y
daría métricas engañosamente altas.

La garantía se puede verificar manualmente: tomando cualquier equipo y
ordenando sus partidos por fecha, la columna `*_forma_*` de un partido
coincide con el promedio de sus `VENTANA_FORMA` partidos **anteriores**,
nunca incluye el propio partido.

La misma disciplina aplica a las features de la Fase 4: el `ELO` guarda el
rating **previo** a cada partido (se actualiza después), el `head-to-head` solo
mira duelos anteriores, e `importancia` es un dato del calendario conocido antes
de jugar. Ninguna usa el resultado que se intenta predecir.

## Columnas del dataset final

- `match_id`, `date`, `home_team`, `away_team`, `neutral`
- `resultado` (objetivo del clasificador)
- `local_forma_gf`, `local_forma_gc`, `local_forma_pts`
- `visit_forma_gf`, `visit_forma_gc`, `visit_forma_pts`
- `dif_forma_pts`
- `local_elo`, `visit_elo`, `dif_elo`
- `h2h_n`, `h2h_pts_local`, `h2h_dif_gol_local`
- `importancia`
- `home_score`, `away_score` (objetivo del modelo Poisson; **no** son features)

El significado de cada columna y cómo se calcula están en el **diccionario del
dataset** del `README.md`.

---

## Etapa 2 — Modelado (`modelo.py`)

Objetivo: entrenar modelos que predigan el partido y evaluarlos con honestidad.

- **División temporal** (`separar_temporal`): se entrena con los partidos
  anteriores a 2022 y se valida con los posteriores. Predecir es predecir el
  futuro, así que se evalúa con el futuro; una división aleatoria inflaría las
  métricas. Las features se separan del objetivo; los goles y los
  identificadores se excluyen de las entradas.
- **Modelo A.1 — Regresión logística** (`entrenar_logistica`): clasificador base,
  interpretable. Se escala con `StandardScaler` dentro de un pipeline (la
  logística es sensible a la escala de las features).
- **Modelo A.2 — Gradient boosting** (`entrenar_boosting`): clasificador potente,
  no necesita escalado. Sirve para comprobar si la complejidad mejora el
  resultado (no lo hace: empata con la logística).
- **Modelo B — Poisson** (`entrenar_poisson`): dos regresiones de Poisson (goles
  del local y del visitante). Con los goles esperados se construye la **matriz de
  marcadores** (`matriz_marcadores`) y se derivan el **marcador más probable** y
  las probabilidades 1X2 (`resumen_desde_matriz`).
- **Evaluación** (`evaluar`, `evaluar_poisson`): accuracy, log-loss y Brier para
  el clasificador; MAE de goles para el Poisson. Se generan gráficos en
  `output/figuras_modelo/` (matriz de confusión, calibración, importancia de
  features, comparación de modelos, chequeo de sobreajuste y matriz de ejemplo).

## Etapa 3 — Predicción de un partido nuevo (`prediccion.py`)

Objetivo: predecir un partido que aún no existe (el que elige el usuario).

- **`estado_actual_equipos`**: recorre el historial y calcula el estado **más
  reciente** de cada selección: su ELO final y su forma de los últimos partidos,
  además del historial directo entre cada par.
- **`construir_features_partido`**: arma la fila de 15 features del partido
  hipotético combinando el estado de los dos equipos con el contexto (cancha,
  importancia).
- **`entrenar_modelos`**: entrena el clasificador y los dos Poisson con todo el
  dataset (para producir, no para evaluar).
- **`predecir_1x2`, `predecir_marcador`, `predecir_partido`**: las tres salidas
  que consume la web.

Se usa el estado **más reciente** de cada equipo (no un promedio largo): el ELO
ya es acumulativo y la forma capta el momento actual; promediar periodos largos
difuminaría esa señal.

## Etapa 4 — Web interactiva (`app.py`)

Objetivo: exponer las predicciones en una interfaz para el Mundial 2026.

- Restringe las selecciones a las **48 del Mundial 2026** y la importancia a las
  **fases del torneo** (todas con el peso competitivo de un partido de Mundial).
- El usuario elige local, visitante, fase y tipo de cancha; al pulsar el botón,
  la web llama a `prediccion.predecir_partido` y muestra, en orden: el **marcador
  más probable** (ranking top-3), la **matriz de marcadores** (mapa de calor
  interactivo de Plotly) y las **probabilidades del resultado** (barra
  horizontal). Los modelos se entrenan una sola vez al arrancar (cache).
- Se ejecuta con `streamlit run app.py`.
