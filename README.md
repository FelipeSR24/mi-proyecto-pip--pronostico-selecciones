# Proyecto de pronóstico de partidos de selecciones

Proyecto de ingeniería y ciencia de datos cuyo objetivo final es **predecir el
resultado de un partido entre dos selecciones** (gana local / empate / gana
visitante) y, a partir de ahí, estimar probabilidades de cara al Mundial 2026.

Este repositorio cubre por ahora el **EDA y la preparación de datos**. El modelo
de pronóstico se construirá en una etapa posterior.

## Qué hace

El script `main.py` ejecuta un pipeline en tres secciones:

1. **EDA inicial:** carga los datasets, muestra su estructura, la proporción de
   resultados (local/empate/visitante), la lista de países y otra información
   exploratoria.
2. **Transformación:** filtra los partidos desde el año 2000, unifica nombres de
   países, crea la variable objetivo y construye las variables predictoras
   ("forma reciente" de cada equipo) en un único dataset listo para modelar.
3. **EDA posterior:** revisa el dataset final resultante.

## Estructura del proyecto

```
mi-proyecto-pip/
├── main.py            # pipeline principal (las 3 secciones)
├── utils.py           # funciones auxiliares
├── datasets/          # aquí van los CSV (los colocas tú manualmente)
│   ├── results.csv
│   ├── goalscorers.csv
│   └── shootouts.csv
├── requirements.txt   # librerías necesarias
├── .env.example       # plantilla de configuración
├── .gitignore
├── README.md          # este archivo
├── WORKFLOWS.md       # detalle de lo que hace el código
└── DATABASE.md        # descripción y origen de los datos
```

## Requisitos

- Python 3.10 o superior
- Las librerías de `requirements.txt` (pandas y numpy)

## Cómo usarlo

1. Crea y activa un entorno virtual:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   ```
2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Coloca los tres CSV en la carpeta `datasets/` (ver `DATABASE.md`).
   Verifica que los nombres coincidan exactamente con los esperados.
4. Ejecuta el pipeline:
   ```
   python main.py
   ```

Al terminar, se genera `datasets/dataset_final.csv` con los datos listos para
el modelo.

## Configuración

Los parámetros principales están como constantes al inicio de `main.py`:

- `ANIO_INICIO` = 2000 — año desde el cual se conservan partidos.
- `VENTANA_FORMA` = 5 — número de partidos previos para la forma reciente.

`.env.example` es una plantilla para cuando se automatice la extracción o se
agregue una base de datos.
