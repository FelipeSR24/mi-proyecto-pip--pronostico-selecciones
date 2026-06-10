# WORKFLOWS — Descripción del funcionamiento

Este documento explica, paso a paso, qué hace el código del proyecto. El flujo
vive en `main.py` y se apoya en funciones de `utils.py`.

## Sección 1 — EDA inicial

Objetivo: conocer los datos crudos antes de transformarlos.

- **Carga de archivos en orden.** Se leen `results.csv`, `goalscorers.csv` y
  `shootouts.csv` con `utils.cargar_dataset()`, que usa `pandas.read_csv`.
- **Estructura de cada dataset.** `utils.mostrar_estructura()` imprime la forma
  (`shape`), el `info()` (tipos y no-nulos) y los nulos por columna.
- **Proporción de resultados.** `utils.proporcion_resultados()` compara
  `home_score` y `away_score` para contar cuántos partidos gana el local, cuántos
  terminan en empate y cuántos gana el visitante, con su porcentaje.
- **Países únicos.** `utils.paises_unicos()` une las columnas de local y
  visitante y cuenta en cuántos partidos aparece cada selección.
- **Información adicional.** Rango de fechas, tipos de torneo más frecuentes
  (`tournament`) y distribución de la columna `neutral` (partidos en cancha
  neutral, clave para la ventaja de local).

## Sección 2 — Transformación

Objetivo: producir un único dataset listo para el modelo.

1. **Filtrado temporal** (`utils.filtrar_desde`): convierte la fecha, descarta
   partidos sin marcador y conserva solo los jugados desde el año 2000. Ordena
   por fecha (imprescindible para no usar información del futuro).
2. **Unificación de nombres** (`utils.unificar_nombres`): reemplaza nombres
   antiguos por los actuales (p. ej. "Serbia and Montenegro" -> "Serbia") usando
   el diccionario `MAPA_NOMBRES`. Puedes ampliarlo si tu EDA revela más casos.
3. **Variable objetivo** (`utils.crear_target`): crea la columna `resultado` con
   tres clases (local / empate / visitante) usando `numpy.select`.
4. **Ingeniería de variables** (`utils.construir_features`): calcula la "forma
   reciente" de cada equipo SIN fuga de información. Para ello:
   - Pasa los partidos a formato largo (una fila por equipo por partido).
   - Calcula medias móviles de goles a favor, en contra y puntos con
     `groupby().transform()` combinando `shift(1)` y `rolling()`. El `shift(1)`
     garantiza que el partido actual no entre en su propio promedio.
   - Vuelve a unir esas variables al partido, separadas en local y visitante, y
     añade `dif_forma_pts` (diferencia de forma entre ambos).
   - Descarta las primeras filas de cada equipo (sin historial todavía).
5. **Guardado**: el resultado se escribe en `datasets/dataset_final.csv`.

## Sección 3 — EDA posterior

Objetivo: validar el dataset final.

- Estructura del dataset final (`shape`, `info()`, nulos).
- Distribución de la variable objetivo (para detectar desbalanceo de clases).
- Estadísticos de las variables (`describe()`).
- Primeras filas (`head()`).

## Prevención de fuga de información (data leakage)

Es el punto metodológico más importante. Todas las variables de forma reciente
se calculan únicamente con partidos anteriores a la fecha de cada partido. Si no
se usara `shift(1)`, el modelo "vería" el resultado que intenta predecir y daría
métricas engañosamente altas.

## Columnas del dataset final

- `match_id`, `date`, `home_team`, `away_team`, `neutral`
- `resultado` (objetivo)
- `local_forma_gf`, `local_forma_gc`, `local_forma_pts`
- `visit_forma_gf`, `visit_forma_gc`, `visit_forma_pts`
- `dif_forma_pts`
