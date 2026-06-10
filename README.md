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
2. **Transformación:** filtra los partidos desde el año 2000, elimina
   duplicados, unifica nombres de países, crea la variable objetivo y construye
   las variables predictoras ("forma reciente" de cada equipo, **sin fuga de
   información**) en un único dataset listo para modelar.
3. **EDA posterior:** revisa el dataset final y genera **cuatro gráficos** en
   `output/figuras/` (distribución del objetivo, partidos por año, ventaja de
   local en cancha propia vs. neutral, y poder discriminante de la feature
   principal).

El detalle paso a paso está en [`WORKFLOWS.md`](WORKFLOWS.md) y el origen y
diccionario de los datos en [`DATABASE.md`](DATABASE.md).

## Estructura del proyecto

```
mi-proyecto-pip/
├── main.py            # pipeline principal (las 3 secciones)
├── utils.py           # funciones auxiliares y gráficos
├── requirements.txt   # librerías con versiones acotadas
├── .env.example       # plantilla de configuración (main.py la lee)
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
- Las librerías de `requirements.txt` (pandas, numpy, matplotlib,
  python-dotenv)

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

4. (Opcional) Copia `.env.example` como `.env` si quieres cambiar parámetros
   sin tocar el código.

5. Ejecuta el pipeline:

   ```bash
   python main.py
   ```

Al terminar, se generan `output/dataset_final.csv` (datos listos para el
modelo) y las figuras del EDA en `output/figuras/`.

## Configuración

Los parámetros se leen del archivo `.env` (si existe) y tienen estos valores
por defecto:

| Variable      | Valor por defecto | Descripción                                  |
|---------------|-------------------|----------------------------------------------|
| `DATA_DIR`    | `datasets`        | Carpeta de los CSV crudos                     |
| `OUTPUT_DIR`  | `output`          | Carpeta de salida (dataset final y figuras)   |
| `START_YEAR`  | `2000`            | Año desde el cual se conservan partidos       |
| `FORM_WINDOW` | `5`               | Partidos previos para la forma reciente       |

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
