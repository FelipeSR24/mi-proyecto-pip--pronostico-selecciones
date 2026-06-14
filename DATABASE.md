# DATABASE — Origen y descripción de los datos

## De dónde salen los datos

Los datos provienen del dataset público de Kaggle **"International football
results from 1872 to 2026"**, del usuario **martj42**.

- Plataforma: Kaggle
- Identificador (slug): `martj42/international-football-results-from-1872-to-2017`
- URL: <https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017>
- Licencia: la indicada en la página del dataset (CC0 / dominio público al
  momento de la consulta; verificar en Kaggle antes de cualquier uso no
  académico).
- Contenido: resultados de partidos internacionales de selecciones masculinas.
  Incluye amistosos, clasificatorias, Mundiales, Eurocopa, Copa América y
  varios torneos más, no solo partidos de Mundial.

La extracción **no está automatizada**: los archivos CSV se descargan
manualmente desde Kaggle y se colocan en la carpeta `datasets/`. Estos CSV
crudos **no se versionan** en el repositorio (ver `.gitignore`) por su tamaño
y porque son regenerables desde la fuente.

## Archivos usados

### results.csv (principal)

Una fila por partido. Es la base del proyecto y de la variable objetivo.

| Columna     | Descripción                                              |
|-------------|----------------------------------------------------------|
| date        | Fecha del partido                                        |
| home_team   | Selección local                                          |
| away_team   | Selección visitante                                      |
| home_score  | Goles del local                                          |
| away_score  | Goles del visitante                                      |
| tournament  | Tipo de torneo (Friendly, FIFA World Cup, etc.)          |
| city        | Ciudad donde se jugó                                     |
| country     | País donde se jugó                                       |
| neutral     | Booleano: si se jugó en cancha neutral                   |

### goalscorers.csv (complementario)

Detalle de goleadores por partido. No se usa en el dataset final actual; queda
disponible para enriquecer el modelo más adelante.

| Columna     | Descripción                                |
|-------------|--------------------------------------------|
| date        | Fecha del partido                          |
| home_team   | Selección local                            |
| away_team   | Selección visitante                        |
| team        | Selección que anotó el gol                 |
| scorer      | Nombre del goleador                        |
| minute      | Minuto del gol (puede tener nulos)         |
| own_goal    | Si fue gol en propia puerta                |
| penalty     | Si fue de penal                            |

### shootouts.csv (complementario)

Tandas de penaltis. Aplica solo a partidos de eliminación empatados; no se usa
en el dataset final actual.

| Columna       | Descripción                                              |
|---------------|----------------------------------------------------------|
| date          | Fecha del partido                                        |
| home_team     | Selección local                                          |
| away_team     | Selección visitante                                      |
| winner        | Selección que ganó la tanda                              |
| first_shooter | Selección que pateó primero (tiene muchos nulos)         |

### former_names.csv (complementario, no usado)

Mapa histórico de nombres de selecciones provisto por el propio dataset de
Kaggle. No lo usa el pipeline: la unificación de nombres se hace con el
diccionario `MAPA_NOMBRES` de `utils.py`, que cubre solo los casos que
aparecen desde el año 2000 (alcance del proyecto).

| Columna    | Descripción                                  |
|------------|----------------------------------------------|
| current    | Nombre actual de la selección                |
| former     | Nombre anterior                              |
| start_date | Inicio de vigencia del nombre anterior       |
| end_date   | Fin de vigencia del nombre anterior          |

## Decisiones de alcance

- **Desde el año 1990:** se descartan los partidos anteriores. Esta fecha de
  corte se eligió con base en datos, no por intuición. Al analizar las métricas
  del fútbol por década (goles por partido, % de victorias local/empate/
  visitante), se observa que **desde 1990 son estables y casi idénticas a las
  actuales** (≈2.7 goles/partido, ≈48% victoria local), mientras que el fútbol
  previo a los años 70-80 era muy distinto (en los 1950s se marcaban ≈4 goles
  por partido). Incluir esos partidos antiguos introduciría patrones de un
  fútbol que ya no existe, restando representatividad. El año es un parámetro
  único y editable (`ANIO_INICIO` en `main.py`).
- **Nombres actuales:** se unifican nombres de selecciones que cambiaron (ver
  `MAPA_NOMBRES` en `utils.py`); por eso no se usa `former_names.csv`. La fusión
  "Serbia and Montenegro" → "Serbia" es una decisión de modelado: se asume
  continuidad deportiva de la selección para no fragmentar su historial.
- **Solo `results`:** el dataset final se construye solo a partir de `results`.
  `goalscorers` y `shootouts` se reservan como posibles mejoras futuras.

## Salida generada

`output/dataset_final.csv`: dataset transformado, con la variable objetivo y
las variables de forma reciente, listo para entrenar el modelo de pronóstico
(ver columnas y diagrama de flujo en `WORKFLOWS.md`). Las figuras del EDA se
guardan en `output/figuras/`. La carpeta `output/` se genera automáticamente
al ejecutar `python main.py` y no se versiona (es regenerable).
