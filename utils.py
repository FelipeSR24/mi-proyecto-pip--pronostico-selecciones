"""
utils.py
Funciones auxiliares para el proyecto de pronostico de partidos de seleccciones.
Mantiene la logica reutilizable fuera de main.py para que el flujo principal
se lea facil.
"""

import pandas as pd
import numpy as np


# Nombres de paises que cambiaron despues del año 2000.
# Se unifican al nombre actual para no fragmentar el historial de una misma
# seleccion. Puedes agregar mas casos si tu EDA revela otros nombres repetidos.
MAPA_NOMBRES = {
    "Serbia and Montenegro": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Yugoslavia": "Serbia",
    "Czechia": "Czech Republic",
}


def cargar_dataset(nombre_archivo, carpeta="datasets"):
    """Carga un CSV de la carpeta de datos y devuelve un DataFrame."""
    ruta = f"{carpeta}/{nombre_archivo}"
    return pd.read_csv(ruta)


def mostrar_estructura(df, nombre):
    """Imprime un resumen de estructura de un DataFrame."""
    print(f"\n=== Estructura: {nombre} ===")
    print(f"Filas y columnas: {df.shape}")
    df.info()
    print("Nulos por columna:")
    print(df.isnull().sum())


def proporcion_resultados(results):
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


def paises_unicos(results):
    """Devuelve un DataFrame con cada pais y cuantos partidos jugo (local + visitante)."""
    serie = pd.concat([results["home_team"], results["away_team"]])
    conteo = serie.value_counts()
    return conteo.rename_axis("pais").reset_index(name="partidos")


def filtrar_desde(results, anio=2000):
    """
    Convierte la columna de fecha, descarta partidos sin marcador y conserva
    solo los partidos desde el año indicado. Devuelve ordenado por fecha.
    """
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results = results.dropna(subset=["home_score", "away_score"])
    results = results[results["date"].dt.year >= anio]
    return results.sort_values("date").reset_index(drop=True)


def unificar_nombres(results, mapa=MAPA_NOMBRES):
    """Reemplaza los nombres antiguos de seleccciones por el nombre actual."""
    results = results.copy()
    results["home_team"] = results["home_team"].replace(mapa)
    results["away_team"] = results["away_team"].replace(mapa)
    return results


def crear_target(results):
    """
    Crea la variable objetivo de 3 clases comparando los goles:
    local  -> gano el equipo de casa
    empate -> mismo marcador
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


def construir_features(results, ventana=5):
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
