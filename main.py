"""
main.py
Proyecto de pronostico de partidos de selecciones (base para el Mundial 2026).

Pipeline en 3 secciones:
  1. EDA inicial      -> explora los datos crudos
  2. Transformacion   -> arma un solo dataset listo para el modelo
  3. EDA posterior    -> revisa el dataset final (incluye graficos en PNG)

NOTA: este script NO incluye todavia el modelo de pronostico.
La extraccion de datos NO esta automatizada: tu colocas manualmente los CSV
en la carpeta 'datasets/' (ver DATABASE.md).

Configuracion: los parametros se leen de variables de entorno (archivo .env,
ver .env.example) y, si no existen, se usan los valores por defecto de abajo.
"""

import os
import sys
from pathlib import Path

import pandas as pd

import utils

# Carga opcional del archivo .env (si python-dotenv esta instalado).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # sin dotenv, simplemente se usan los valores por defecto

# ------------------------- Configuracion -------------------------
CARPETA_DATOS = Path(os.getenv("DATA_DIR", "datasets"))
ANIO_INICIO = int(os.getenv("START_YEAR", "2000"))      # solo partidos desde este anio
VENTANA_FORMA = int(os.getenv("FORM_WINDOW", "5"))      # partidos previos para la forma
CARPETA_SALIDA = Path(os.getenv("OUTPUT_DIR", "output"))
GUARDAR_FINAL = CARPETA_SALIDA / "dataset_final.csv"
CARPETA_FIGURAS = CARPETA_SALIDA / "figuras"

# Nombres de los archivos. Deben coincidir EXACTAMENTE con los de tu carpeta.
# OJO: en Kaggle el archivo de penaltis se llama 'shootouts.csv' (con h).
ARCHIVO_RESULTS = "results.csv"
ARCHIVO_GOLEADORES = "goalscorers.csv"
ARCHIVO_SHOOTOUTS = "shootouts.csv"


def seccion(titulo: str) -> None:
    """Imprime un separador de seccion."""
    print("\n" + "#" * 60)
    print(f"# {titulo}")
    print("#" * 60)


# ============================================================
# 1. EDA INICIAL
# ============================================================
def eda_inicial() -> pd.DataFrame:
    """Carga los datos crudos, imprime el EDA inicial y devuelve results."""
    seccion("1. EDA INICIAL")

    # 1.1 Carga de archivos en orden (con validacion de columnas)
    results = utils.cargar_dataset(ARCHIVO_RESULTS, CARPETA_DATOS)
    goalscorers = utils.cargar_dataset(ARCHIVO_GOLEADORES, CARPETA_DATOS)
    shootouts = utils.cargar_dataset(ARCHIVO_SHOOTOUTS, CARPETA_DATOS)
    utils.validar_columnas(results, utils.COLUMNAS_RESULTS, "results")

    # 1.2 Estructura de cada dataset
    utils.mostrar_estructura(results, "results")
    utils.mostrar_estructura(goalscorers, "goalscorers")
    utils.mostrar_estructura(shootouts, "shootouts")

    # 1.3 Duplicados (limpieza: se confirma antes de transformar)
    print("\n--- Filas duplicadas exactas ---")
    print(f"results: {utils.contar_duplicados(results)}")
    print(f"goalscorers: {utils.contar_duplicados(goalscorers)}")
    print(f"shootouts: {utils.contar_duplicados(shootouts)}")

    # 1.4 Proporcion local / empate / visitante (sobre todo el historico)
    print("\n--- Proporcion de resultados (historico completo) ---")
    print(utils.proporcion_resultados(results))

    # 1.5 Paises unicos
    print("\n--- Paises unicos y numero de partidos ---")
    paises = utils.paises_unicos(results)
    print(f"Total de paises distintos: {len(paises)}")
    print(paises.head(20))

    # 1.6 Informacion adicional
    print("\n--- Informacion adicional ---")
    results["date"] = pd.to_datetime(results["date"])
    print(f"Rango de fechas: {results['date'].min().date()} a "
          f"{results['date'].max().date()}")
    print("\nTipos de torneo (top 10):")
    print(results["tournament"].value_counts().head(10))
    print("\nPartidos en cancha neutral:")
    print(results["neutral"].value_counts())

    return results


# ============================================================
# 2. TRANSFORMACION A UN SOLO DATASET
# ============================================================
def transformar(results: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline de transformacion y guarda el dataset final."""
    seccion("2. TRANSFORMACION")

    # 2.1 Filtrar desde el anio de inicio, quitar duplicados y ordenar por fecha
    df = utils.filtrar_desde(results, ANIO_INICIO)
    print(f"Partidos desde {ANIO_INICIO}: {df.shape[0]}")

    # 2.2 Unificar nombres de paises que cambiaron
    df = utils.unificar_nombres(df)

    # 2.3 Crear la variable objetivo (local / empate / visitante)
    df = utils.crear_target(df)

    # 2.4 Construir features de forma reciente (sin fuga de informacion)
    dataset_final = utils.construir_features(df, VENTANA_FORMA)

    # 2.5 Guardar el dataset final
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    dataset_final.to_csv(GUARDAR_FINAL, index=False)
    print(f"Dataset final guardado en: {GUARDAR_FINAL}")
    print(f"Forma del dataset final: {dataset_final.shape}")
    print("Columnas:", list(dataset_final.columns))

    return dataset_final


# ============================================================
# 3. EDA POSTERIOR (sobre el dataset final)
# ============================================================
def eda_posterior(dataset_final: pd.DataFrame) -> None:
    """Valida el dataset final: estructura, distribucion, graficos."""
    seccion("3. EDA POSTERIOR AL DATASET FINAL")

    utils.mostrar_estructura(dataset_final, "dataset_final")

    print("\n--- Distribucion de la variable objetivo ---")
    conteo = dataset_final["resultado"].value_counts()
    porc = (dataset_final["resultado"]
            .value_counts(normalize=True).mul(100).round(2))
    print(pd.DataFrame({"partidos": conteo, "porcentaje": porc}))

    print("\n--- Estadisticos de las features ---")
    print(dataset_final.describe())

    print("\n--- Primeras filas ---")
    print(dataset_final.head())

    print("\n--- Graficos del EDA ---")
    utils.graficar_distribucion_target(dataset_final, CARPETA_FIGURAS)
    utils.graficar_partidos_por_anio(dataset_final, CARPETA_FIGURAS)
    utils.graficar_ventaja_local(dataset_final, CARPETA_FIGURAS)
    utils.graficar_forma_vs_resultado(dataset_final, CARPETA_FIGURAS)


def main() -> int:
    """Ejecuta el pipeline completo. Devuelve 0 si todo salio bien."""
    try:
        results = eda_inicial()
        dataset_final = transformar(results)
        eda_posterior(dataset_final)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    print("\nPipeline completado sin errores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
