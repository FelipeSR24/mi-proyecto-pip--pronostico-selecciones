"""
utils.py
Funciones auxiliares para el proyecto de pronostico de partidos de selecciones.

Mantiene la logica reutilizable fuera de main.py para que el flujo principal
se lea facil. Incluye:
  - Carga y validacion de datasets.
  - Funciones de EDA (estructura, proporciones, paises, duplicados).
  - Transformaciones (filtrado, unificacion de nombres, target, features).
  - Graficos del EDA (se guardan como PNG en la carpeta de figuras).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo guarda los PNG a disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Nombres de paises que cambiaron de denominacion y que pueden aparecer en
# partidos desde el anio 2000 (p. ej. "FR Yugoslavia" jugo hasta 2003).
# Se unifican al nombre actual para no fragmentar el historial de una misma
# seleccion. NOTA: "Serbia and Montenegro" -> "Serbia" es una decision de
# modelado (se asume continuidad deportiva); esta documentada en DATABASE.md.
MAPA_NOMBRES: dict[str, str] = {
    "Serbia and Montenegro": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Yugoslavia": "Serbia",
    "Czechia": "Czech Republic",
}

# Columnas minimas que debe tener results.csv para que el pipeline funcione.
COLUMNAS_RESULTS: list[str] = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


# ------------------------------------------------------------------
# Carga y validacion
# ------------------------------------------------------------------
def cargar_dataset(nombre_archivo: str, carpeta: str | Path = "datasets") -> pd.DataFrame:
    """Carga un CSV de la carpeta de datos y devuelve un DataFrame.

    Lanza FileNotFoundError con un mensaje orientador si el archivo no esta,
    en lugar de un traceback crudo de pandas.
    """
    ruta = Path(carpeta) / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro '{ruta}'. Descarga los CSV de Kaggle "
            f"(ver DATABASE.md) y colocalos en la carpeta '{carpeta}/'."
        )
    return pd.read_csv(ruta)


def validar_columnas(df: pd.DataFrame, esperadas: list[str], nombre: str) -> None:
    """Verifica que el DataFrame tenga las columnas esperadas."""
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Al dataset '{nombre}' le faltan las columnas {faltantes}. "
            "Verifica que descargaste la version correcta desde Kaggle."
        )


# ------------------------------------------------------------------
# EDA
# ------------------------------------------------------------------
def mostrar_estructura(df: pd.DataFrame, nombre: str) -> None:
    """Imprime un resumen de estructura de un DataFrame."""
    print(f"\n=== Estructura: {nombre} ===")
    print(f"Filas y columnas: {df.shape}")
    df.info()
    print("Nulos por columna:")
    print(df.isnull().sum())


def contar_duplicados(df: pd.DataFrame, claves: list[str] | None = None) -> int:
    """Cuenta filas duplicadas (opcionalmente segun un subconjunto de claves)."""
    return int(df.duplicated(subset=claves).sum())


def proporcion_resultados(results: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con la proporcion de local / empate / visitante."""
    local = (results["home_score"] > results["away_score"]).sum()
    empate = (results["home_score"] == results["away_score"]).sum()
    visitante = (results["home_score"] < results["away_score"]).sum()
    total = local + empate + visitante

    resumen = pd.DataFrame({
        "resultado": ["local", "empate", "visitante"],
        "partidos": [local, empate, visitante],
    })
    resumen["porcentaje"] = (resumen["partidos"] / total * 100).round(2)
    return resumen


def paises_unicos(results: pd.DataFrame) -> pd.DataFrame:
    """Devuelve cada pais y cuantos partidos jugo (como local + visitante)."""
    serie = pd.concat([results["home_team"], results["away_team"]])
    conteo = serie.value_counts()
    return conteo.rename_axis("pais").reset_index(name="partidos")


# ------------------------------------------------------------------
# Transformacion
# ------------------------------------------------------------------
def filtrar_desde(results: pd.DataFrame, anio: int = 2000) -> pd.DataFrame:
    """
    Convierte la columna de fecha, descarta partidos sin marcador y filas
    duplicadas exactas, y conserva solo los partidos desde el anio indicado.
    Devuelve ordenado por fecha (imprescindible para las ventanas moviles).
    """
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results = results.dropna(subset=["home_score", "away_score"])
    results = results.drop_duplicates()
    results = results[results["date"].dt.year >= anio]
    return results.sort_values("date").reset_index(drop=True)


def unificar_nombres(results: pd.DataFrame,
                     mapa: dict[str, str] = MAPA_NOMBRES) -> pd.DataFrame:
    """Reemplaza los nombres antiguos de selecciones por el nombre actual."""
    results = results.copy()
    results["home_team"] = results["home_team"].replace(mapa)
    results["away_team"] = results["away_team"].replace(mapa)
    return results


def crear_target(results: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo de 3 clases comparando los goles:
      local     -> gano el equipo de casa
      empate    -> mismo marcador
      visitante -> gano el visitante
    """
    results = results.copy()
    condiciones = [
        results["home_score"] > results["away_score"],
        results["home_score"] == results["away_score"],
    ]
    opciones = ["local", "empate"]
    results["resultado"] = np.select(condiciones, opciones, default="visitante")
    return results


def construir_features(results: pd.DataFrame, ventana: int = 5) -> pd.DataFrame:
    """
    Construye, para cada partido, la 'forma reciente' de cada equipo SIN fuga
    de informacion (cada partido solo usa datos de partidos anteriores).

    Pasos:
      1. Pasa los partidos a formato largo: una fila por equipo por partido.
      2. Calcula medias moviles de goles a favor, en contra y puntos, usando
         shift(1) para que el partido actual NO entre en su propio promedio.
      3. Vuelve a unir esas features al partido, separadas en local y visitante.

    Devuelve el dataset final listo para entrenar un modelo.
    """
    results = results.copy()
    results["match_id"] = range(len(results))

    # 1) Formato largo: dos filas por partido (una por equipo)
    local = pd.DataFrame({
        "match_id": results["match_id"],
        "date": results["date"],
        "equipo": results["home_team"],
        "gf": results["home_score"],
        "gc": results["away_score"],
        "es_local": 1,
    })
    visitante = pd.DataFrame({
        "match_id": results["match_id"],
        "date": results["date"],
        "equipo": results["away_team"],
        "gf": results["away_score"],
        "gc": results["home_score"],
        "es_local": 0,
    })
    largo = pd.concat([local, visitante], ignore_index=True)

    # puntos obtenidos en ese partido: 3 si gana, 1 si empata, 0 si pierde
    largo["pts"] = np.select(
        [largo["gf"] > largo["gc"], largo["gf"] == largo["gc"]],
        [3, 1],
        default=0,
    )

    # 2) Medias moviles desplazadas (shift(1) -> sin fuga de informacion)
    largo = largo.sort_values(["equipo", "date"]).reset_index(drop=True)
    g = largo.groupby("equipo")
    largo["forma_gf"] = g["gf"].transform(
        lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
    largo["forma_gc"] = g["gc"].transform(
        lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
    largo["forma_pts"] = g["pts"].transform(
        lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())

    # 3) Separar features de local y visitante y unirlas de vuelta al partido
    feats = largo[["match_id", "es_local", "forma_gf", "forma_gc", "forma_pts"]]

    local_feats = (feats[feats["es_local"] == 1]
                   .drop(columns="es_local")
                   .rename(columns={
                       "forma_gf": "local_forma_gf",
                       "forma_gc": "local_forma_gc",
                       "forma_pts": "local_forma_pts"}))
    visit_feats = (feats[feats["es_local"] == 0]
                   .drop(columns="es_local")
                   .rename(columns={
                       "forma_gf": "visit_forma_gf",
                       "forma_gc": "visit_forma_gc",
                       "forma_pts": "visit_forma_pts"}))

    final = results[["match_id", "date", "home_team", "away_team",
                     "neutral", "resultado"]].copy()
    final = final.merge(local_feats, on="match_id")
    final = final.merge(visit_feats, on="match_id")

    # feature de diferencia: util y muy informativa para el modelo
    final["dif_forma_pts"] = final["local_forma_pts"] - final["visit_forma_pts"]

    # las primeras filas de cada equipo no tienen historial -> se descartan
    final = final.dropna().reset_index(drop=True)
    return final


# ------------------------------------------------------------------
# Graficos del EDA (se guardan como PNG)
# ------------------------------------------------------------------
def _guardar(fig: plt.Figure, carpeta: str | Path, nombre: str) -> Path:
    """Guarda una figura en la carpeta indicada y la cierra."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")
    return ruta


def graficar_distribucion_target(df: pd.DataFrame,
                                 carpeta: str | Path) -> Path:
    """Barra con la distribucion de la variable objetivo (desbalance)."""
    conteo = df["resultado"].value_counts().reindex(
        ["local", "empate", "visitante"])
    fig, ax = plt.subplots(figsize=(6, 4))
    conteo.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_title("Distribucion de la variable objetivo")
    ax.set_xlabel("resultado")
    ax.set_ylabel("partidos")
    for i, v in enumerate(conteo):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    return _guardar(fig, carpeta, "01_distribucion_target.png")


def graficar_partidos_por_anio(df: pd.DataFrame, carpeta: str | Path) -> Path:
    """Linea con el numero de partidos por anio (cobertura temporal)."""
    por_anio = df["date"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    por_anio.plot(ax=ax, marker="o", color="#264653")
    ax.set_title("Partidos por anio en el dataset final")
    ax.set_xlabel("anio")
    ax.set_ylabel("partidos")
    ax.grid(alpha=0.3)
    return _guardar(fig, carpeta, "02_partidos_por_anio.png")


def graficar_ventaja_local(df: pd.DataFrame, carpeta: str | Path) -> Path:
    """
    Compara la proporcion local/empate/visitante en cancha propia vs neutral.
    Es la evidencia visual de la 'ventaja de local'.
    """
    tabla = (df.groupby("neutral")["resultado"]
               .value_counts(normalize=True)
               .mul(100)
               .rename("porcentaje")
               .reset_index()
               .pivot(index="neutral", columns="resultado",
                      values="porcentaje")
               [["local", "empate", "visitante"]])
    tabla.index = tabla.index.map({False: "Cancha propia", True: "Cancha neutral"})
    fig, ax = plt.subplots(figsize=(7, 4))
    tabla.plot(kind="bar", ax=ax,
               color=["#2a9d8f", "#e9c46a", "#e76f51"], rot=0)
    ax.set_title("Resultado segun tipo de cancha (ventaja de local)")
    ax.set_ylabel("% de partidos")
    ax.legend(title="resultado")
    return _guardar(fig, carpeta, "03_ventaja_local.png")


def graficar_forma_vs_resultado(df: pd.DataFrame, carpeta: str | Path) -> Path:
    """
    Boxplot de dif_forma_pts segun el resultado: valida visualmente que la
    feature principal discrimina entre clases.
    """
    orden = ["local", "empate", "visitante"]
    datos = [df.loc[df["resultado"] == r, "dif_forma_pts"] for r in orden]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(datos, tick_labels=orden, showfliers=False)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("dif_forma_pts segun el resultado del partido")
    ax.set_xlabel("resultado")
    ax.set_ylabel("dif_forma_pts (local - visitante)")
    return _guardar(fig, carpeta, "04_dif_forma_vs_resultado.png")
