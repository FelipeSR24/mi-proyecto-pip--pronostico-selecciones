# WORKFLOWS — Descripción del funcionamiento

Este documento explica, paso a paso, qué hace el código del proyecto. El flujo
vive en `main.py` (función `main()`, con guarda `if __name__ == "__main__"`)
y se apoya en funciones de `utils.py`.

## Vista general del flujo

```
datasets/results.csv ─┐
datasets/goalscorers.csv ─┼─► [1. EDA inicial] ─► results validado
datasets/shootouts.csv ─┘            │
                                     ▼
                          [2. Transformación]
              filtrar ≥ 2000 ► quitar duplicados ► unificar nombres
                     ► crear target ► features de forma (shift+rolling)
                                     │
                                     ▼
                      output/dataset_final.csv
                                     │
                                     ▼
                          [3. EDA posterior]
              estadísticos + distribución + 4 figuras PNG
                        (output/figuras/*.png)
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
   y conserva solo los jugados desde el año 2000 (`START_YEAR`). Ordena por
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
5. **Guardado**: el resultado se escribe en `output/dataset_final.csv`.
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

## Columnas del dataset final

- `match_id`, `date`, `home_team`, `away_team`, `neutral`
- `resultado` (objetivo)
- `local_forma_gf`, `local_forma_gc`, `local_forma_pts`
- `visit_forma_gf`, `visit_forma_gc`, `visit_forma_pts`
- `dif_forma_pts`
