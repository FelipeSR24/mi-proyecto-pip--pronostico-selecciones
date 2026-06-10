"""
main.py
Proyecto de pronostico de partidos de seleccciones (base para el Mundial 2026).

Pipeline en 3 secciones:
  1. EDA inicial      -> explora los datos crudos
  2. Transformacion   -> arma un solo dataset listo para el modelo
  3. EDA posterior    -> revisa el dataset final

NOTA: este script NO incluye todavia el modelo de pronostico.
La extraccion de datos NO esta automatizada: tu colocas manualmente los CSV
en la carpeta 'datasets/'.
"""

import pandas as pd
import utils

# ------------------------- Configuracion -------------------------
CARPETA_DATOS = "datasets"
ANIO_INICIO = 2000        # solo partidos desde este año
VENTANA_FORMA = 5         # nro. de partidos previos para la 'forma reciente'
GUARDAR_FINAL = "datasets/dataset_final.csv"

# Nombres de los archivos. Deben coincidir EXACTAMENTE con los de tu carpeta.
# OJO: en Kaggle el archivo de penaltis se llama 'shootouts.csv' (con h).
ARCHIVO_RESULTS = "results.csv"
ARCHIVO_GOLEADORES = "goalscorers.csv"
ARCHIVO_SHOOTOUTS = "shootouts.csv"


# ============================================================
# 1. EDA INICIAL
# ============================================================
print("#" * 60)
print("# 1. EDA INICIAL")
print("#" * 60)

# 1.1 Carga de archivos en orden
results = utils.cargar_dataset(ARCHIVO_RESULTS, CARPETA_DATOS)
goalscorers = utils.cargar_dataset(ARCHIVO_GOLEADORES, CARPETA_DATOS)
shootouts = utils.cargar_dataset(ARCHIVO_SHOOTOUTS, CARPETA_DATOS)

# 1.2 Estructura de cada dataset
utils.mostrar_estructura(results, "results")
utils.mostrar_estructura(goalscorers, "goalscorers")
utils.mostrar_estructura(shootouts, "shootouts")

# 1.3 Proporcion local / empate / visitante (sobre todo el historico)
print("\n--- Proporcion de resultados (historico completo) ---")
print(utils.proporcion_resultados(results))

# 1.4 Paises unicos
print("\n--- Paises unicos y numero de partidos ---")
paises = utils.paises_unicos(results)
print(f"Total de paises distintos: {len(paises)}")
print(paises.head(20))

# 1.5 Informacion adicional
print("\n--- Informacion adicional ---")
results["date"] = pd.to_datetime(results["date"])
print(f"Rango de fechas: {results['date'].min().date()} a {results['date'].max().date()}")
print("\nTipos de torneo (top 10):")
print(results["tournament"].value_counts().head(10))
print("\nPartidos en cancha neutral:")
print(results["neutral"].value_counts())


# ============================================================
# 2. TRANSFORMACION A UN SOLO DATASET
# ============================================================
print("\n" + "#" * 60)
print("# 2. TRANSFORMACION")
print("#" * 60)

# 2.1 Filtrar desde el año de inicio y ordenar por fecha
df = utils.filtrar_desde(results, ANIO_INICIO)
print(f"Partidos desde {ANIO_INICIO}: {df.shape[0]}")

# 2.2 Unificar nombres de paises que cambiaron
df = utils.unificar_nombres(df)

# 2.3 Crear la variable objetivo (local / empate / visitante)
df = utils.crear_target(df)

# 2.4 Construir features de forma reciente (sin fuga de informacion)
dataset_final = utils.construir_features(df, VENTANA_FORMA)

# 2.5 Guardar el dataset final
dataset_final.to_csv(GUARDAR_FINAL, index=False)
print(f"Dataset final guardado en: {GUARDAR_FINAL}")
print(f"Forma del dataset final: {dataset_final.shape}")
print("Columnas:", list(dataset_final.columns))


# ============================================================
# 3. EDA POSTERIOR (sobre el dataset final)
# ============================================================
print("\n" + "#" * 60)
print("# 3. EDA POSTERIOR AL DATASET FINAL")
print("#" * 60)

utils.mostrar_estructura(dataset_final, "dataset_final")

print("\n--- Distribucion de la variable objetivo ---")
conteo = dataset_final["resultado"].value_counts()
porc = dataset_final["resultado"].value_counts(normalize=True).mul(100).round(2)
print(pd.DataFrame({"partidos": conteo, "porcentaje": porc}))

print("\n--- Estadisticos de las features ---")
print(dataset_final.describe())

print("\n--- Primeras filas ---")
print(dataset_final.head())
