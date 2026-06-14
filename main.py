"""
============================================================================
main.py  —  GUION PRINCIPAL DEL PROYECTO
============================================================================
Proyecto de pronostico de partidos de selecciones (base para el Mundial 2026).

QUE ES ESTE ARCHIVO
-------------------
Es el "director de orquesta" del proyecto: el archivo que se ejecuta con
`python main.py`. No contiene la logica detallada (esa vive en utils.py);
aqui solo se ordena, paso a paso, QUE se hace y EN QUE ORDEN.

EL PIPELINE TIENE 3 SECCIONES
-----------------------------
  1. EDA inicial    -> exploramos los datos CRUDOS para conocerlos.
  2. Transformacion -> limpiamos y construimos UN dataset listo para el modelo.
  3. EDA posterior  -> revisamos el dataset final y generamos graficos.

(EDA = Exploratory Data Analysis = Analisis Exploratorio de Datos.)

NOTA DE ALCANCE: esta entrega llega hasta preparar los datos. El modelo de
prediccion en si se construira en la siguiente etapa.

LOS DATOS NO SE DESCARGAN SOLOS: hay que colocar manualmente los CSV de Kaggle
en la carpeta 'datasets/' (ver DATABASE.md para el detalle del origen).
============================================================================
"""

# --- Librerias estandar de Python ---
import sys                # para escribir errores en la salida de error estandar
from pathlib import Path  # para manejar rutas de carpetas de forma segura

# --- Libreria de terceros ---
import pandas as pd       # la herramienta central para manejar tablas de datos

# --- Nuestro propio modulo de funciones auxiliares ---
import utils              # aqui estan TODAS las funciones que hacen el trabajo


# ===========================================================================
# CONFIGURACION DEL PROYECTO  (UNICO lugar para cambiar parametros)
# ===========================================================================
# Todos los parametros ajustables del proyecto estan reunidos aqui como
# constantes. Si necesitas cambiar algo (el anio de corte, la ventana de forma,
# las carpetas), lo haces SOLO en este bloque y todo el pipeline se adapta.
# No se usa un archivo .env porque este proyecto no maneja datos sensibles
# (credenciales, claves): son simples parametros de configuracion del codigo.

# Anio desde el cual se conservan los partidos. Se eligio 1990 porque desde esa
# decada las metricas del futbol (goles por partido, % de victorias) son
# estables y representativas del futbol actual; incluir partidos mas antiguos
# meteria patrones de un futbol distinto (ver analisis en DATABASE.md).
ANIO_INICIO = 1990

# Numero de partidos previos que se promedian para calcular la "forma reciente".
VENTANA_FORMA = 5

# Carpetas de entrada (CSV crudos) y de salida (dataset final y figuras).
CARPETA_DATOS = Path("datasets")
CARPETA_SALIDA = Path("output")
GUARDAR_FINAL = CARPETA_SALIDA / "dataset_final.csv"  # ruta del dataset final
CARPETA_FIGURAS = CARPETA_SALIDA / "figuras"          # ruta de los graficos PNG

# Nombres de los archivos de entrada. Deben coincidir EXACTAMENTE con los de
# tu carpeta 'datasets/'. OJO: en Kaggle el de penaltis es 'shootouts.csv'.
ARCHIVO_RESULTS = "results.csv"         # el principal: un partido por fila
ARCHIVO_GOLEADORES = "goalscorers.csv"  # complementario (no se usa aun)
ARCHIVO_SHOOTOUTS = "shootouts.csv"     # complementario (no se usa aun)


def seccion(titulo: str) -> None:
    """Imprime un separador visual en consola para distinguir cada seccion.

    Es puramente cosmetico: ayuda a leer la salida cuando se ejecuta el script.
    """
    print("\n" + "#" * 60)
    print(f"# {titulo}")
    print("#" * 60)


# ===========================================================================
# SECCION 1 — EDA INICIAL: conocer los datos crudos antes de tocarlos
# ===========================================================================
def eda_inicial() -> pd.DataFrame:
    """Carga los datos crudos, imprime un primer analisis y los devuelve.

    El objetivo de esta seccion es ENTENDER los datos: cuantas filas hay, que
    columnas, si hay valores nulos o duplicados, que proporcion de partidos
    gana el local, etc. Todavia NO transformamos nada.
    """
    seccion("1. EDA INICIAL")

    # 1.1 CARGA. Leemos los tres CSV. cargar_dataset() avisa con un mensaje
    #     claro si algun archivo falta (en vez de un error tecnico confuso).
    results = utils.cargar_dataset(ARCHIVO_RESULTS, CARPETA_DATOS)
    goalscorers = utils.cargar_dataset(ARCHIVO_GOLEADORES, CARPETA_DATOS)
    shootouts = utils.cargar_dataset(ARCHIVO_SHOOTOUTS, CARPETA_DATOS)
    # Verificamos que 'results' tenga las columnas que el pipeline necesita.
    utils.validar_columnas(results, utils.COLUMNAS_RESULTS, "results")

    # 1.2 ESTRUCTURA. Para cada tabla mostramos forma, tipos y nulos por columna.
    utils.mostrar_estructura(results, "results")
    utils.mostrar_estructura(goalscorers, "goalscorers")
    utils.mostrar_estructura(shootouts, "shootouts")

    # 1.3 DUPLICADOS. Confirmamos cuantas filas repetidas hay ANTES de limpiar,
    #     para dejar constancia del estado original de los datos.
    print("\n--- Filas duplicadas exactas ---")
    print(f"results: {utils.contar_duplicados(results)}")
    print(f"goalscorers: {utils.contar_duplicados(goalscorers)}")
    print(f"shootouts: {utils.contar_duplicados(shootouts)}")

    # 1.4 BALANCE DE RESULTADOS. Que tan frecuente es que gane el local, que
    #     haya empate o que gane el visitante (sobre TODO el historico).
    print("\n--- Proporcion de resultados (historico completo) ---")
    print(utils.proporcion_resultados(results))

    # 1.5 PAISES. Cuantas selecciones distintas aparecen y cuanto juega cada una.
    print("\n--- Paises unicos y numero de partidos ---")
    paises = utils.paises_unicos(results)
    print(f"Total de paises distintos: {len(paises)}")
    print(paises.head(20))

    # 1.6 CONTEXTO EXTRA. Rango de fechas, torneos mas frecuentes y cuantos
    #     partidos se jugaron en cancha neutral (importante para la ventaja local).
    print("\n--- Informacion adicional ---")
    results["date"] = pd.to_datetime(results["date"])  # texto -> fecha real
    print(f"Rango de fechas: {results['date'].min().date()} a "
          f"{results['date'].max().date()}")
    print("\nTipos de torneo (top 10):")
    print(results["tournament"].value_counts().head(10))
    print("\nPartidos en cancha neutral:")
    print(results["neutral"].value_counts())

    return results  # devolvemos results para usarlo en la siguiente seccion


# ===========================================================================
# SECCION 2 — TRANSFORMACION: de datos crudos a un dataset listo para modelar
# ===========================================================================
def transformar(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia los datos y construye el dataset final con sus variables.

    Devuelve DOS tablas:
      - dataset_final: la tabla lista para entrenar un modelo.
      - df: los partidos ya filtrados/limpios (con la columna 'tournament'),
            que reutilizamos en algunos graficos del EDA posterior.

    El orden de los pasos NO es casual: primero se limpia y se ordena por fecha,
    porque las variables de "forma reciente" dependen de ese orden temporal.
    """
    seccion("2. TRANSFORMACION")

    # 2.1 FILTRAR Y LIMPIAR. Conserva solo partidos desde ANIO_INICIO, descarta
    #     partidos sin marcador y filas duplicadas, y ordena por fecha (clave
    #     para que la "forma reciente" use solo el pasado de cada equipo).
    df = utils.filtrar_desde(results, ANIO_INICIO)
    print(f"Partidos desde {ANIO_INICIO}: {df.shape[0]}")

    # 2.2 UNIFICAR NOMBRES. Algunos paises cambiaron de nombre (p. ej.
    #     "Serbia and Montenegro" -> "Serbia"). Los unificamos ANTES de calcular
    #     la forma, para no partir el historial de una misma seleccion en dos.
    df = utils.unificar_nombres(df)

    # 2.3 VARIABLE OBJETIVO. Creamos la columna 'resultado' con 3 clases:
    #     local / empate / visitante. Es lo que el futuro modelo intentara predecir.
    df = utils.crear_target(df)

    # 2.4 IDENTIFICADOR UNICO. Asignamos un 'match_id' a cada partido ANTES de
    #     construir las features, para que TODAS (forma, ELO, head-to-head e
    #     importancia) queden alineadas por el mismo identificador al unirlas.
    df = df.reset_index(drop=True)
    df["match_id"] = range(len(df))

    # 2.5 INGENIERIA DE VARIABLES. Calculamos la "forma reciente" de cada equipo
    #     (goles y puntos de sus ultimos partidos) SIN fuga de informacion: cada
    #     partido solo usa datos de partidos ANTERIORES. Este es el paso clave.
    dataset_final = utils.construir_features(df, VENTANA_FORMA)

    # 2.6 FEATURES ADICIONALES (Fase 4). Tres senales nuevas, todas calculadas
    #     SIN fuga (solo con informacion previa a cada partido) y unidas por
    #     'match_id': fuerza ELO de cada seleccion, historial directo entre ambas
    #     e importancia del torneo. Usamos merge POR LA IZQUIERDA porque
    #     construir_features ya descarto los primeros partidos de cada equipo.
    elo = utils.calcular_elo(df)
    h2h = utils.calcular_head_to_head(df)
    importancia = utils.calcular_importancia(df)
    dataset_final = (dataset_final
                     .merge(elo, on="match_id", how="left")
                     .merge(h2h, on="match_id", how="left")
                     .merge(importancia, on="match_id", how="left"))

    # 2.7 GUARDAR. Escribimos el dataset final en disco para la siguiente etapa.
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)  # crea 'output/' si no existe
    dataset_final.to_csv(GUARDAR_FINAL, index=False)
    print(f"Dataset final guardado en: {GUARDAR_FINAL}")
    print(f"Forma del dataset final: {dataset_final.shape}")
    print("Columnas:", list(dataset_final.columns))

    return dataset_final, df


# ===========================================================================
# SECCION 3 — EDA POSTERIOR: validar el dataset final y graficar hallazgos
# ===========================================================================
def eda_posterior(dataset_final: pd.DataFrame,
                  results_filtrado: pd.DataFrame) -> None:
    """Revisa el dataset final y produce analisis y graficos del EDA.

    Aqui comprobamos que la transformacion quedo bien (sin nulos inesperados,
    con el balance de clases esperado) y extraemos hallazgos interesantes que
    sirven para entender el problema antes de modelar.
    """
    seccion("3. EDA POSTERIOR AL DATASET FINAL")

    # Estructura del resultado: forma, tipos y nulos (deberia venir limpio).
    utils.mostrar_estructura(dataset_final, "dataset_final")

    # DISTRIBUCION DEL OBJETIVO. Cuantos partidos de cada clase hay. Sirve para
    # detectar "desbalance" (si una clase domina, el modelo puede sesgarse).
    print("\n--- Distribucion de la variable objetivo ---")
    conteo = dataset_final["resultado"].value_counts()
    porc = (dataset_final["resultado"]
            .value_counts(normalize=True).mul(100).round(2))
    print(pd.DataFrame({"partidos": conteo, "porcentaje": porc}))

    # ESTADISTICOS de las variables numericas (medias, minimos, maximos...).
    print("\n--- Estadisticos de las features ---")
    print(dataset_final.describe())

    # Un vistazo a las primeras filas para ver el aspecto final de la tabla.
    print("\n--- Primeras filas ---")
    print(dataset_final.head())

    # ANALISIS ADICIONALES con contexto futbolistico.
    # IMPORTANTE: estos se calculan sobre 'results_filtrado' (TODOS los partidos
    # desde 2000), NO sobre dataset_final, porque este ultimo descarta los
    # primeros partidos de cada equipo y sesgaria levemente el ranking historico.
    print("\n--- Top 10 selecciones por rendimiento (pts por partido) ---")
    ranking = utils.ranking_rendimiento(results_filtrado, min_partidos=50)
    print(ranking.head(10).to_string(index=False))

    # Ejemplo de "head to head": historial directo entre dos selecciones.
    print("\n--- Historial directo de ejemplo: Brazil vs Argentina ---")
    print(utils.head_to_head(results_filtrado, "Brazil", "Argentina")
          .to_string(index=False))

    # GRAFICOS. Cada funcion crea un PNG en output/figuras/. El comentario al
    # lado dice que pregunta responde cada grafico.
    print("\n--- Graficos del EDA ---")
    utils.graficar_distribucion_target(dataset_final, CARPETA_FIGURAS)      # balance de clases
    utils.graficar_partidos_por_anio(dataset_final, CARPETA_FIGURAS)        # cobertura temporal
    utils.graficar_ventaja_local(dataset_final, CARPETA_FIGURAS)            # ventaja de local
    utils.graficar_forma_vs_resultado(dataset_final, CARPETA_FIGURAS)       # la feature, ¿sirve?
    utils.graficar_top_selecciones(results_filtrado, CARPETA_FIGURAS)       # potencias del periodo
    utils.graficar_evolucion_ventaja_local(dataset_final, CARPETA_FIGURAS)  # ¿cambia con el tiempo?
    utils.graficar_goles_por_torneo(results_filtrado, CARPETA_FIGURAS)      # torneos mas ofensivos


# ===========================================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ===========================================================================
def main() -> int:
    """Ejecuta el pipeline completo, en orden. Devuelve 0 si todo salio bien.

    Envolvemos las 3 secciones en un try/except para que, si falta un archivo
    o una columna, el usuario reciba un mensaje claro en vez de un error tecnico.
    El valor devuelto (0 = exito, 1 = error) es util para automatizaciones.
    """
    try:
        results = eda_inicial()                               # seccion 1
        dataset_final, results_filtrado = transformar(results)  # seccion 2
        eda_posterior(dataset_final, results_filtrado)        # seccion 3
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    print("\nPipeline completado sin errores.")
    return 0


# Esta condicion hace que main() se ejecute SOLO si corremos este archivo
# directamente (python main.py), y NO si alguien lo importa desde otro script.
if __name__ == "__main__":
    raise SystemExit(main())  # termina el programa con el codigo que devuelve main()
