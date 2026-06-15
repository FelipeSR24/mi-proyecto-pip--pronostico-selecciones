"""
============================================================================
prediccion.py  —  LOGICA DE PREDICCION PARA UN PARTIDO NUEVO
============================================================================
Este modulo es el "cerebro" que usara la pagina web. Mientras modelo.py
ENTRENA y EVALUA los modelos sobre partidos historicos, este archivo se encarga
de PREDECIR un partido que aun no se ha jugado, a partir de lo que elige el
usuario: equipo local, equipo visitante, si la cancha es neutral y la
importancia del partido.

EL RETO QUE RESUELVE
--------------------
Los modelos se entrenan con partidos que YA tienen sus features calculadas. Pero
un partido nuevo (el que el usuario quiere consultar) no existe en el dataset:
no tiene ELO, ni forma, ni head-to-head. Por eso aqui primero RECONSTRUIMOS las
features de ese partido hipotetico usando el ESTADO MAS RECIENTE de cada
seleccion (su ultimo ELO, su forma de los ultimos partidos, su historial
directo), y luego se las pasamos a los modelos.

QUE ENTREGA (las tres salidas de la web)
----------------------------------------
  1. predecir_1x2()      -> probabilidades de local / empate / visitante.
  2. predecir_marcador() -> matriz de marcadores y marcador mas probable.

DECISION METODOLOGICA (respaldada por la literatura): se usa el estado MAS
RECIENTE de cada equipo. El ELO ya es acumulativo (incluye toda la historia
ponderada hacia lo reciente); la forma es el promedio de sus ultimos partidos
reales. Promediar periodos largos difuminaria la forma actual, que es justo lo
que se quiere captar.
============================================================================
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import poisson

import utils


# ===========================================================================
# CONFIGURACION
# ===========================================================================
DATASET = Path("output/dataset_final.csv")
VENTANA_FORMA = 5          # mismos ultimos N partidos que en el entrenamiento
MAX_GOLES_MATRIZ = 6       # la matriz cubre de 0-0 hasta 6-6

# Features que reciben los modelos (mismas que en modelo.py, en el mismo orden).
COLUMNAS_NO_FEATURE = [
    "match_id", "date", "home_team", "away_team", "resultado",
    "home_score", "away_score",
]


# ===========================================================================
# PASO 1 — CALCULAR EL "ESTADO ACTUAL" DE CADA SELECCION
# ===========================================================================
def estado_actual_equipos(dataset: pd.DataFrame) -> dict:
    """Calcula el ELO final y la forma reciente de cada seleccion.

    QUE RECIBE: el dataset final (con goles, elo previo, etc.), ordenable por
      fecha. Lo importante es que tenga date, home_team, away_team, home_score,
      away_score, neutral e importancia.
    QUE ENTREGA: un diccionario {equipo: {...}} con, por cada seleccion:
      - elo: su rating ELO tras su ultimo partido (estado mas reciente).
      - forma_gf / forma_gc / forma_pts: el promedio de sus ultimos
        VENTANA_FORMA partidos (goles a favor, en contra y puntos).
    Tambien guarda el historial completo de marcadores entre cada par de
    equipos, para reconstruir el head-to-head despues.

    COMO LO CALCULA (recorriendo los partidos en orden cronologico, igual que se
    construyeron las features, para que el estado sea coherente con lo aprendido):
      - ELO: arranca en ELO_BASE y se actualiza partido a partido con la misma
        formula oficial de utils (factor K por importancia, factor G por margen,
        ventaja de local 100 salvo en cancha neutral).
      - Forma: se guardan los ultimos VENTANA_FORMA resultados de cada equipo.
    """
    df = dataset.sort_values(["date", "match_id"]).copy()

    ratings: dict[str, float] = defaultdict(lambda: utils.ELO_BASE)
    # Para la forma guardamos listas de los ultimos partidos de cada equipo.
    ult_gf: dict[str, list] = defaultdict(list)
    ult_gc: dict[str, list] = defaultdict(list)
    ult_pts: dict[str, list] = defaultdict(list)
    # Historial directo: para cada par (sin importar orden) la lista de duelos.
    historial: dict[frozenset, list] = defaultdict(list)

    for fila in df.itertuples(index=False):
        local, visit = fila.home_team, fila.away_team
        gl, gv = int(fila.home_score), int(fila.away_score)

        # --- Actualizar ELO con la formula oficial de utils ---
        r_local, r_visit = ratings[local], ratings[visit]
        bonus = 0.0 if fila.neutral else utils.ELO_VENTAJA_LOCAL
        esperado = 1.0 / (1.0 + 10 ** (-(r_local + bonus - r_visit) / 400))
        if gl > gv:
            real = 1.0
        elif gl == gv:
            real = 0.5
        else:
            real = 0.0
        k = utils.ELO_K_POR_IMPORTANCIA.get(fila.importancia, 20.0)
        g = utils._factor_g(gl - gv)
        cambio = k * g * (real - esperado)
        ratings[local] = r_local + cambio
        ratings[visit] = r_visit - cambio

        # --- Actualizar la forma (puntos y goles) de cada equipo ---
        pts_local = 3 if gl > gv else (1 if gl == gv else 0)
        pts_visit = 3 if gv > gl else (1 if gv == gl else 0)
        ult_gf[local].append(gl)
        ult_gc[local].append(gv)
        ult_pts[local].append(pts_local)
        ult_gf[visit].append(gv)
        ult_gc[visit].append(gl)
        ult_pts[visit].append(pts_visit)

        # --- Guardar el duelo para el head-to-head ---
        historial[frozenset((local, visit))].append((local, gl, gv))

    # Construir el resumen final por equipo (promediando sus ultimos N partidos).
    def media_ultimos(lista):
        ultimos = lista[-VENTANA_FORMA:]            # los ultimos N
        return sum(ultimos) / len(ultimos) if ultimos else 0.0

    estado = {}
    for equipo in ratings:
        estado[equipo] = {
            "elo": ratings[equipo],
            "forma_gf": media_ultimos(ult_gf[equipo]),
            "forma_gc": media_ultimos(ult_gc[equipo]),
            "forma_pts": media_ultimos(ult_pts[equipo]),
        }
    return {"equipos": estado, "historial": historial}


def _head_to_head_actual(historial: dict, local: str, visit: str) -> dict:
    """Resume el historial directo entre dos equipos (perspectiva del local).

    QUE ENTREGA: h2h_n (cuantos duelos previos), h2h_pts_local (puntos promedio
    del local en esos duelos) y h2h_dif_gol_local (diferencia de goles promedio).
    Si no hay historial, devuelve valores neutros (igual que en el entrenamiento).
    """
    previos = historial.get(frozenset((local, visit)), [])
    if not previos:
        return {"h2h_n": 0, "h2h_pts_local": 1.5, "h2h_dif_gol_local": 0.0}
    puntos, difs = [], []
    for h_local, h_gf, h_gc in previos:
        # Reorientar el marcador a la perspectiva del local ACTUAL.
        gl, gr = (h_gf, h_gc) if h_local == local else (h_gc, h_gf)
        puntos.append(3 if gl > gr else (1 if gl == gr else 0))
        difs.append(gl - gr)
    return {
        "h2h_n": len(previos),
        "h2h_pts_local": sum(puntos) / len(puntos),
        "h2h_dif_gol_local": sum(difs) / len(difs),
    }


# ===========================================================================
# PASO 2 — CONSTRUIR LAS FEATURES DE UN PARTIDO NUEVO
# ===========================================================================
def construir_features_partido(estado: dict, local: str, visit: str,
                               neutral: bool, importancia: int) -> pd.DataFrame:
    """Arma la fila de features de un partido hipotetico.

    QUE RECIBE: el estado actual (de estado_actual_equipos), los dos equipos, si
      es cancha neutral y la importancia (0-3) que elige el usuario.
    QUE ENTREGA: un DataFrame de UNA fila con las 15 features, en el mismo orden
      y con los mismos nombres que uso el entrenamiento, listo para los modelos.
    """
    eq = estado["equipos"]
    if local not in eq or visit not in eq:
        faltan = [t for t in (local, visit) if t not in eq]
        raise ValueError(f"Sin datos historicos para: {faltan}")

    el, ev = eq[local], eq[visit]
    h2h = _head_to_head_actual(estado["historial"], local, visit)

    fila = {
        "neutral": int(neutral),
        "local_forma_gf": el["forma_gf"],
        "local_forma_gc": el["forma_gc"],
        "local_forma_pts": el["forma_pts"],
        "visit_forma_gf": ev["forma_gf"],
        "visit_forma_gc": ev["forma_gc"],
        "visit_forma_pts": ev["forma_pts"],
        "dif_forma_pts": el["forma_pts"] - ev["forma_pts"],
        "local_elo": el["elo"],
        "visit_elo": ev["elo"],
        "dif_elo": el["elo"] - ev["elo"],
        "h2h_n": h2h["h2h_n"],
        "h2h_pts_local": h2h["h2h_pts_local"],
        "h2h_dif_gol_local": h2h["h2h_dif_gol_local"],
        "importancia": importancia,
    }
    return pd.DataFrame([fila])


# ===========================================================================
# PASO 3 — ENTRENAR LOS MODELOS (los mismos de modelo.py)
# ===========================================================================
def entrenar_modelos(dataset: pd.DataFrame) -> dict:
    """Entrena los modelos que usara la web: clasificador 1X2 y dos Poisson.

    QUE ENTREGA: un diccionario con el modelo de clasificacion (logistica) y los
      dos modelos Poisson (goles del local y del visitante). Se entrenan con
      TODO el dataset (no dividimos en validacion aqui: eso ya se hizo en
      modelo.py para evaluar; para producir predicciones usamos todos los datos).
    """
    df = dataset.copy()
    df["neutral"] = df["neutral"].astype(int)
    features = [c for c in df.columns if c not in COLUMNAS_NO_FEATURE]
    X = df[features]

    # Clasificador 1X2 (logistica, la elegida por ser interpretable y calibrada).
    clasificador = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=1000))
    clasificador.fit(X, df["resultado"])

    # Dos regresiones Poisson: goles del local y del visitante.
    poisson_local = make_pipeline(StandardScaler(), PoissonRegressor(max_iter=1000))
    poisson_local.fit(X, df["home_score"])
    poisson_visit = make_pipeline(StandardScaler(), PoissonRegressor(max_iter=1000))
    poisson_visit.fit(X, df["away_score"])

    return {"clasificador": clasificador, "poisson_local": poisson_local,
            "poisson_visit": poisson_visit, "features": features}


# ===========================================================================
# PASO 4 — LAS TRES SALIDAS PARA LA WEB
# ===========================================================================
def predecir_1x2(modelos: dict, fila_features: pd.DataFrame) -> dict:
    """SALIDA 1: probabilidades de local / empate / visitante.

    QUE RECIBE: los modelos entrenados y la fila de features del partido.
    QUE ENTREGA: un diccionario {'local': %, 'empate': %, 'visitante': %} que
      suma 100. Sale directo del clasificador (regresion logistica).
    """
    clf = modelos["clasificador"]
    X = fila_features[modelos["features"]]
    proba = clf.predict_proba(X)[0]               # vector de 3 probabilidades
    return {clase: float(p) for clase, p in zip(clf.classes_, proba)}


def predecir_marcador(modelos: dict, fila_features: pd.DataFrame) -> dict:
    """SALIDAS 2 y 3: matriz de marcadores y marcador mas probable.

    QUE RECIBE: los modelos entrenados y la fila de features del partido.
    QUE ENTREGA: un diccionario con:
      - lambda_local / lambda_visit: goles esperados de cada equipo.
      - matriz: matriz (7x7) con la probabilidad de cada marcador (0-0 .. 6-6).
      - marcador_probable: la tupla (i, j) de mayor probabilidad.
      - prob_marcador: la probabilidad de ese marcador.
    """
    X = fila_features[modelos["features"]]
    lam_l = float(modelos["poisson_local"].predict(X)[0])   # goles esperados local
    lam_v = float(modelos["poisson_visit"].predict(X)[0])   # goles esperados visit

    # Matriz: P(local=i) * P(visit=j) para cada combinacion de goles.
    goles = np.arange(MAX_GOLES_MATRIZ + 1)
    p_local = poisson.pmf(goles, lam_l)
    p_visit = poisson.pmf(goles, lam_v)
    matriz = np.outer(p_local, p_visit)

    # Marcador mas probable: la celda de mayor probabilidad.
    i, j = np.unravel_index(np.argmax(matriz), matriz.shape)
    return {
        "lambda_local": lam_l,
        "lambda_visit": lam_v,
        "matriz": matriz,
        "marcador_probable": (int(i), int(j)),
        "prob_marcador": float(matriz[i, j]),
    }


def predecir_partido(modelos: dict, estado: dict, local: str, visit: str,
                     neutral: bool, importancia: int) -> dict:
    """Funcion 'todo en uno' que la web puede llamar directamente.

    QUE RECIBE: modelos, estado actual, los dos equipos, cancha neutral e
      importancia (lo que el usuario elige en la web).
    QUE ENTREGA: un diccionario con las TRES salidas juntas: las probabilidades
      1X2, la matriz de marcadores y el marcador mas probable.
    """
    fila = construir_features_partido(estado, local, visit, neutral, importancia)
    prob_1x2 = predecir_1x2(modelos, fila)
    marcador = predecir_marcador(modelos, fila)
    return {"probabilidades_1x2": prob_1x2, "marcador": marcador}


# ===========================================================================
# DEMOSTRACION: prediccion de un partido de ejemplo
# ===========================================================================
def main() -> int:
    """Ejemplo de uso de todo el flujo (se ejecuta con: python prediccion.py)."""
    if not DATASET.exists():
        print(f"No se encontro {DATASET}. Ejecuta antes 'python main.py'.")
        return 1

    print("Cargando dataset y entrenando modelos (puede tardar unos segundos)...")
    dataset = pd.read_csv(DATASET, parse_dates=["date"])
    estado = estado_actual_equipos(dataset)
    modelos = entrenar_modelos(dataset)

    # Partido de ejemplo: Brazil (local) vs Argentina (visitante), no neutral,
    # importancia competitiva (2).
    local, visit, neutral, imp = "Brazil", "Argentina", False, 2
    print(f"\nPrediccion de ejemplo: {local} (local) vs {visit} (visitante)")
    print(f"Cancha neutral: {neutral} | Importancia: {imp}")

    r = predecir_partido(modelos, estado, local, visit, neutral, imp)

    print("\n--- SALIDA 1: Probabilidades 1X2 ---")
    for clase, p in r["probabilidades_1x2"].items():
        print(f"  {clase:<10}: {p*100:.1f}%")

    m = r["marcador"]
    print("\n--- SALIDA 2 y 3: Marcador ---")
    print(f"  Goles esperados: {local} {m['lambda_local']:.2f}, "
          f"{visit} {m['lambda_visit']:.2f}")
    print(f"  Marcador mas probable: {m['marcador_probable'][0]}-"
          f"{m['marcador_probable'][1]} ({m['prob_marcador']*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
