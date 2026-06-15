"""
============================================================================
modelo.py  —  ENTRENAMIENTO Y EVALUACION DE LOS MODELOS DE PREDICCION
============================================================================
Esta es la etapa de MODELADO del proyecto. Toma el dataset ya preparado
(output/dataset_final.csv) y entrena modelos que predicen el resultado de un
partido entre dos selecciones.

QUE PRODUCE (las tres salidas de la web):
  - Modelo A: clasificador 1X2 -> probabilidades de local / empate / visitante.
      A.1 Regresion Logistica -> modelo base, interpretable (el ELEGIDO).
      A.2 Gradient Boosting   -> modelo potente, para comparar.
      (Ambos hacen lo MISMO; se comparan y se elige uno. Ganan resultados muy
       parecidos, asi que se usa la logistica por ser interpretable.)
  - Modelo B: Poisson -> goles esperados de cada equipo, de los que se derivan
      la matriz de marcadores y el marcador mas probable.

METODOLOGIA CLAVE:
  - Validacion TEMPORAL: se entrena con los años antiguos (< 2022) y se valida
    con los recientes (>= 2022). Predecir un partido es predecir el futuro, asi
    que evaluamos con el futuro. Una division aleatoria daria metricas
    enganosamente altas (el modelo "veria" el futuro al entrenar).
  - El baseline a superar: predecir siempre "local" acierta ~48% (la clase mas
    comun). Un modelo util tiene que superarlo.

GLOSARIO DE METRICAS (que entra, como se calcula, que sale):

  * ACCURACY (clasificador). Que ENTRA: la clase predicha y la real de cada
    partido de validacion. COMO se calcula: porcentaje de partidos en que la
    clase predicha coincide con la real. Que SALE: un % (mas alto = mejor).
    Limitacion: solo mira si acerto la clase, no la calidad de la probabilidad.

  * LOG-LOSS (clasificador). Que ENTRA: las 3 probabilidades predichas y la
    clase real. COMO: penaliza con el logaritmo estar "muy seguro y equivocado"
    (si das 5% a lo que ocurre, el castigo es enorme). Que SALE: un numero >= 0
    (mas BAJO = mejor). Mide la calidad de las PROBABILIDADES, no solo el acierto.

  * BRIER (clasificador). Que ENTRA: las 3 probabilidades y la clase real (como
    1/0 por clase). COMO: error CUADRATICO medio entre la probabilidad dada y lo
    que paso (1 si era esa clase, 0 si no), promediado sobre las 3 clases. Que
    SALE: un numero entre 0 (perfecto) y 2 (pesimo). Mide CALIBRACION: si dices
    "70%", ¿pasa el 70%?

  * MAE de goles (Poisson). Que ENTRA: los goles esperados (prediccion) y los
    goles reales. COMO: promedio del valor absoluto de la diferencia. Que SALE:
    "cuantos goles nos equivocamos en promedio" (mas bajo = mejor).

Se ejecuta con:  python modelo.py
============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: guarda los PNG a disco
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.linear_model import PoissonRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler, label_binarize  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix,  # noqa: E402
                             classification_report, brier_score_loss,
                             mean_absolute_error)
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from scipy.stats import poisson  # noqa: E402


# ===========================================================================
# CONFIGURACION
# ===========================================================================
DATASET = Path("output/dataset_final.csv")
ANIO_CORTE_VALIDACION = 2022   # entrenar con < 2022, validar con >= 2022
CARPETA_FIGURAS = Path("output/figuras_modelo")  # graficos de evaluacion
BASELINE_REF = 0.4751  # baseline de referencia (predecir siempre 'local') para graficos

# Las columnas que el modelo usa como ENTRADA (features). Excluimos los
# identificadores (match_id, date, equipos), la variable objetivo (resultado) y
# los goles (home_score/away_score son el resultado del propio partido: usarlos
# como entrada seria fuga de informacion; solo sirven para el modelo Poisson).
COLUMNAS_NO_FEATURE = [
    "match_id", "date", "home_team", "away_team", "resultado",
    "home_score", "away_score",
]


def cargar_datos() -> pd.DataFrame:
    """Carga el dataset final y convierte 'neutral' (booleano) a 0/1.

    QUE ENTREGA: el DataFrame listo, con 'neutral' como numero (0 o 1) para que
    los modelos puedan usarlo (no aceptan True/False directamente).
    """
    if not DATASET.exists():
        raise FileNotFoundError(
            f"No se encontro '{DATASET}'. Ejecuta antes 'python main.py' "
            "para generar el dataset final."
        )
    df = pd.read_csv(DATASET, parse_dates=["date"])
    # CORRECCION 3 del plan: 'neutral' booleano -> entero 0/1.
    df["neutral"] = df["neutral"].astype(int)
    return df


def separar_temporal(df: pd.DataFrame):
    """Divide en entrenamiento (años antiguos) y validacion (años recientes).

    QUE RECIBE: el DataFrame completo.
    QUE ENTREGA: X_train, X_val, y_train, y_val (features y objetivo de cada
      conjunto). La division es POR FECHA, no aleatoria: imitamos la situacion
      real de predecir partidos futuros con informacion del pasado.
    """
    features = [c for c in df.columns if c not in COLUMNAS_NO_FEATURE]
    entrena = df[df["date"].dt.year < ANIO_CORTE_VALIDACION]
    valida = df[df["date"].dt.year >= ANIO_CORTE_VALIDACION]

    X_train, y_train = entrena[features], entrena["resultado"]
    X_val, y_val = valida[features], valida["resultado"]

    print(f"Entrenamiento: {len(X_train):,} partidos (< {ANIO_CORTE_VALIDACION})")
    print(f"Validacion:    {len(X_val):,} partidos (>= {ANIO_CORTE_VALIDACION})")
    print(f"Features usadas ({len(features)}): {features}")
    return X_train, X_val, y_train, y_val, features


def baseline_accuracy(y_train: pd.Series, y_val: pd.Series) -> float:
    """Calcula el baseline: acertar siempre con la clase mas comun del entrenamiento.

    Es la vara minima: cualquier modelo util debe superar esto. En este dataset
    la clase mas comun es 'local' (~48%).
    """
    clase_mas_comun = y_train.value_counts().idxmax()
    acc = (y_val == clase_mas_comun).mean()
    print(f"\nBaseline (predecir siempre '{clase_mas_comun}'): "
          f"{acc*100:.2f}% de acierto en validacion")
    return acc


def evaluar(modelo, X_val, y_val, nombre: str) -> dict:
    """Evalua un modelo ya entrenado e imprime las metricas clave.

    QUE MIDE:
      - accuracy: % de aciertos de la clase (comparar contra el baseline).
      - log-loss: que tan buenas son las PROBABILIDADES (mas bajo = mejor).
      - matriz de confusion: en que acierta y en que se confunde.
    QUE ENTREGA: un diccionario con las metricas (para comparar modelos luego).
    """
    pred = modelo.predict(X_val)               # clase predicha
    proba = modelo.predict_proba(X_val)        # probabilidades de cada clase

    acc = accuracy_score(y_val, pred)
    ll = log_loss(y_val, proba, labels=modelo.classes_)
    brier = brier_multiclase(y_val, proba, modelo.classes_)

    print(f"\n{'='*60}\nRESULTADOS — {nombre}\n{'='*60}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Log-loss:  {ll:.4f}  (mas bajo = probabilidades mas fiables)")
    print(f"Brier:     {brier:.4f}  (mas bajo = probabilidades mejor calibradas)")
    print("\nMatriz de confusion (filas=real, columnas=predicho):")
    etiquetas = sorted(y_val.unique())
    cm = confusion_matrix(y_val, pred, labels=etiquetas)
    cm_df = pd.DataFrame(cm, index=[f"real_{e}" for e in etiquetas],
                         columns=[f"pred_{e}" for e in etiquetas])
    print(cm_df.to_string())
    print("\nReporte por clase (precision/recall/f1):")
    print(classification_report(y_val, pred, zero_division=0))

    # Graficos de evaluacion para este modelo
    graficar_matriz_confusion(y_val, pred, etiquetas, nombre)
    graficar_calibracion(y_val, proba, modelo.classes_, nombre)

    return {"nombre": nombre, "accuracy": acc, "log_loss": ll, "brier": brier}


def brier_multiclase(y_val, proba, clases) -> float:
    """Brier score para problemas de 3 clases (promedio de los Brier por clase).

    QUE MIDE: que tan CALIBRADAS estan las probabilidades. Compara la
    probabilidad que el modelo dio a cada clase contra lo que de verdad paso
    (1 si era esa clase, 0 si no), y promedia el error al cuadrado. Va de 0
    (perfecto) a 2 (pesimo); mas bajo es mejor. A diferencia del accuracy, mide
    la calidad de las PROBABILIDADES, no solo si acierta la clase ganadora.

    Como el Brier de sklearn es para 2 clases, lo aplicamos a cada clase por
    separado (one-vs-rest) y promediamos.
    """
    # Convertimos y_val en columnas 0/1, una por clase (one-hot).
    y_bin = label_binarize(y_val, classes=list(clases))
    briers = []
    for i in range(len(clases)):
        briers.append(brier_score_loss(y_bin[:, i], proba[:, i]))
    return float(np.mean(briers))


# ===========================================================================
# GRAFICOS DE EVALUACION
# ===========================================================================
def _guardar(fig, nombre_archivo: str) -> Path:
    """Guarda una figura como PNG en la carpeta de figuras del modelo."""
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_FIGURAS / nombre_archivo
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")
    return ruta


def graficar_matriz_confusion(y_val, pred, etiquetas, nombre: str) -> Path:
    """Mapa de calor de la matriz de confusion (en que acierta y en que falla).

    Las filas son el resultado REAL y las columnas el PREDICHO. La diagonal son
    los aciertos; fuera de la diagonal, los errores. Colores mas intensos =
    mas partidos en esa celda.
    """
    cm = confusion_matrix(y_val, pred, labels=etiquetas)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas)
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion — {nombre}")
    # Escribimos el numero en cada celda, en blanco o negro segun el fondo.
    umbral = cm.max() / 2
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    archivo = f"confusion_{nombre.split()[0].lower().replace('.', '')}.png"
    return _guardar(fig, archivo)


def graficar_calibracion(y_val, proba, clases, nombre: str) -> Path:
    """Curva de calibracion: cuando el modelo dice 'X%', ¿pasa el X%?

    Para cada clase, agrupa las predicciones por nivel de probabilidad y compara
    la probabilidad PREDICHA (eje X) con la frecuencia REAL observada (eje Y).
    La linea diagonal es la calibracion perfecta: cuanto mas cerca de ella, mas
    fiables son las probabilidades. Es la metrica visual clave para 'apostar'.
    """
    y_bin = label_binarize(y_val, classes=list(clases))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Calibracion perfecta")
    colores = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for i, clase in enumerate(clases):
        # frac_real: frecuencia real; media_pred: probabilidad media predicha
        frac_real, media_pred = calibration_curve(
            y_bin[:, i], proba[:, i], n_bins=10, strategy="quantile")
        ax.plot(media_pred, frac_real, marker="o",
                color=colores[i % 3], label=f"{clase}")
    ax.set_xlabel("Probabilidad predicha por el modelo")
    ax.set_ylabel("Frecuencia real observada")
    ax.set_title(f"Curva de calibracion — {nombre}")
    ax.legend()
    ax.grid(alpha=0.3)
    archivo = f"calibracion_{nombre.split()[0].lower().replace('.', '')}.png"
    return _guardar(fig, archivo)


def graficar_importancia_features(modelo, features, nombre: str) -> Path:
    """Barras horizontales con el peso de cada feature en la Regresion Logistica.

    Usa los coeficientes de la clase 'local'. Como las features estan escaladas,
    los pesos son comparables: positivo empuja hacia victoria local, negativo en
    contra; mas grande (en valor absoluto) = mas influyente.
    """
    clf = modelo.named_steps["logisticregression"]
    clases = list(clf.classes_)
    idx_local = clases.index("local")
    coefs = pd.Series(clf.coef_[idx_local], index=features)
    coefs = coefs.sort_values()  # de menor a mayor para barras horizontales

    fig, ax = plt.subplots(figsize=(8, 6))
    colores = ["#e76f51" if c < 0 else "#2a9d8f" for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colores)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title(f"Peso de cada feature (clase 'local') — {nombre}")
    ax.set_xlabel("Coeficiente (+ favorece local · − en contra)")
    return _guardar(fig, "importancia_features_logistica.png")


def graficar_empate_por_paridad(df: pd.DataFrame) -> Path:
    """Muestra por que el empate es dificil: su % segun lo parejo del partido.

    Agrupa los partidos por la diferencia de ELO (que tan parejos son) y dibuja
    el % de cada resultado. Se ve que el empate nunca llega a ser la opcion mas
    probable, ni siquiera en los partidos mas parejos.
    """
    d = df.copy()
    d["dif_elo_abs"] = d["dif_elo"].abs()
    d["bin"] = pd.cut(d["dif_elo_abs"], bins=[0, 50, 150, 300, 600, 2000],
                      labels=["muy\nparejo", "parejo", "medio",
                              "disparejo", "muy\ndisparejo"])
    tab = (d.groupby("bin", observed=True)["resultado"]
           .value_counts(normalize=True).mul(100)
           .unstack()[["local", "empate", "visitante"]])
    fig, ax = plt.subplots(figsize=(8, 5))
    tab.plot(kind="bar", ax=ax, rot=0,
             color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_title("Por que el empate es dificil: % de resultado segun la paridad")
    ax.set_xlabel("Que tan parejo es el partido (por diferencia de ELO)")
    ax.set_ylabel("Porcentaje de partidos (%)")
    ax.legend(title="Resultado")
    return _guardar(fig, "empate_por_paridad.png")


def graficar_importancia_permutacion(modelo, X_val, y_val, features,
                                     nombre: str) -> Path:
    """Importancia de features del boosting por el metodo de PERMUTACION.

    El boosting no tiene 'pesos' como la logistica. Para medir que feature
    importa, se 'desordena' (permuta) una feature al azar y se mide cuanto
    EMPEORA el accuracy: si empeora mucho, esa feature era importante. Se repite
    varias veces (n_repeats) para promediar. Mas barra = mas importante.
    """
    imp = permutation_importance(modelo, X_val, y_val, n_repeats=5,
                                 random_state=42, scoring="accuracy")
    serie = pd.Series(imp.importances_mean, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(serie.index, serie.values, color="#264653")
    ax.set_title(f"Importancia por permutacion — {nombre}")
    ax.set_xlabel("Caida de accuracy al desordenar la feature (mas = mas importante)")
    return _guardar(fig, "importancia_permutacion_boosting.png")


def graficar_comparacion_modelos(resultados: list) -> Path:
    """Barras comparando accuracy de los modelos contra el baseline.

    Muestra de un vistazo que los dos modelos quedan casi iguales y ambos
    superan al baseline (predecir siempre 'local').
    """
    nombres = [r["nombre"] for r in resultados]
    accs = [r["accuracy"] * 100 for r in resultados]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(nombres, accs, color=["#2a9d8f", "#e76f51"])
    ax.axhline(BASELINE_REF * 100, color="grey", linestyle="--",
               label=f"Baseline ({BASELINE_REF*100:.1f}%)")
    ax.set_ylabel("Accuracy en validacion (%)")
    ax.set_title("Comparacion de modelos vs baseline")
    ax.set_ylim(40, 65)
    for b, a in zip(barras, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.2f}%",
                ha="center", fontsize=9)
    ax.legend()
    return _guardar(fig, "comparacion_modelos.png")


def graficar_sobreajuste(modelo, X_train, y_train, X_val, y_val,
                         nombre: str) -> Path:
    """Compara accuracy en entrenamiento vs validacion (chequeo de sobreajuste).

    Si el modelo 'memoriza' (sobreajuste), iria mucho mejor en entrenamiento que
    en validacion. Barras parecidas = el modelo generaliza bien.
    """
    acc_train = modelo.score(X_train, y_train) * 100
    acc_val = modelo.score(X_val, y_val) * 100
    fig, ax = plt.subplots(figsize=(6, 4.5))
    barras = ax.bar(["Entrenamiento", "Validacion"], [acc_train, acc_val],
                    color=["#e9c46a", "#2a9d8f"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Chequeo de sobreajuste — {nombre}")
    ax.set_ylim(40, 70)
    for b, a in zip(barras, [acc_train, acc_val]):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.1f}%",
                ha="center", fontsize=9)
    return _guardar(fig, "sobreajuste_boosting.png")


# ===========================================================================
# MODELO A.1 — REGRESION LOGISTICA (multinomial)
# ===========================================================================
def entrenar_logistica(X_train, y_train):
    """Entrena el clasificador base: Regresion Logistica multinomial.

    POR QUE este modelo primero: es simple, interpretable y produce
    probabilidades naturales y bien calibradas. Sirve de referencia honesta:
    si un modelo mas complejo no lo supera, no vale la pena su complejidad.

    DETALLE IMPORTANTE (correccion 2 del plan): la logistica es sensible a la
    ESCALA de las features (el ELO vale ~1500 y la forma ~1.5; sin escalar, el
    modelo se distorsiona). Por eso usamos un PIPELINE que primero estandariza
    (StandardScaler: deja cada feature con media 0 y desviacion 1) y luego
    entrena. El escalado se aprende SOLO con datos de entrenamiento y se aplica
    igual a validacion, sin fuga de informacion.

    QUE ENTREGA: el modelo (pipeline) ya entrenado, listo para predecir.
    """
    modelo = make_pipeline(
        StandardScaler(),                          # paso 1: escalar
        LogisticRegression(max_iter=1000),         # paso 2: regresion logistica
    )
    modelo.fit(X_train, y_train)
    return modelo


# ===========================================================================
# MODELO A.2 — GRADIENT BOOSTING
# ===========================================================================
def entrenar_boosting(X_train, y_train):
    """Entrena el clasificador potente: Gradient Boosting (arboles).

    POR QUE este modelo: es el que GANA los benchmarks de prediccion de futbol
    en la literatura. En vez de una formula lineal, construye muchos arboles de
    decision pequenos, cada uno corrigiendo los errores del anterior. Asi captura
    relaciones NO lineales e interacciones entre features que la regresion
    logistica no puede (p. ej. "el ELO importa mas cuando el partido es de
    Mundial"). Usamos HistGradientBoosting de scikit-learn, una version rapida.

    DIFERENCIA CLAVE CON LA LOGISTICA: NO necesita escalar las features. Los
    arboles deciden por umbrales ("¿dif_elo > 200?"), donde la magnitud del
    numero da igual. Por eso aqui no usamos StandardScaler.

    QUE ENTREGA: el modelo ya entrenado, listo para predecir las 3 probabilidades.

    Nota sobre hiperparametros: usamos valores moderados (profundidad y numero de
    arboles limitados) para que el modelo generalice y no se "sobreajuste"
    memorizando el pasado. random_state fija la aleatoriedad para que el
    resultado sea reproducible.
    """
    modelo = HistGradientBoostingClassifier(
        max_depth=4,            # arboles poco profundos -> evita sobreajuste
        learning_rate=0.05,     # aprende despacio -> mas estable
        max_iter=300,           # numero de arboles
        l2_regularization=1.0,  # penaliza la complejidad
        random_state=42,        # reproducibilidad
    )
    modelo.fit(X_train, y_train)
    return modelo


def comparar_modelos(resultados: list) -> None:
    """Imprime una tabla comparativa de los modelos evaluados.

    Recibe la lista de diccionarios que devuelve evaluar() y los muestra juntos
    para ver de un vistazo cual gana en cada metrica.
    """
    print(f"\n{'='*60}\nCOMPARACION DE MODELOS\n{'='*60}")
    tabla = pd.DataFrame(resultados)
    tabla["accuracy"] = (tabla["accuracy"] * 100).round(2)
    tabla["log_loss"] = tabla["log_loss"].round(4)
    tabla["brier"] = tabla["brier"].round(4)
    tabla = tabla.rename(columns={"accuracy": "accuracy_%"})
    print(tabla.to_string(index=False))
    print("\nGuia: accuracy mas ALTO mejor; log-loss y Brier mas BAJOS mejor.")


# ===========================================================================
# MODELO B — POISSON (para marcador mas probable y matriz de marcadores)
# ===========================================================================
# Mientras el Modelo A predice directamente quien gana (local/empate/visitante),
# el Modelo B predice CUANTOS GOLES marca cada equipo. De esos goles esperados
# se derivan las otras dos salidas de la web: el marcador mas probable y la
# matriz con la probabilidad de cada marcador.
#
# COMO FUNCIONA, en dos partes:
#   1) Dos regresiones de Poisson (una para los goles del local, otra para los
#      del visitante). Cada una aprende a estimar el numero ESPERADO de goles
#      (lambda) de ese equipo, en funcion de las features del partido.
#   2) Con lambda_local y lambda_visit, la probabilidad de un marcador exacto
#      (i, j) es P(local marca i) x P(visitante marca j), usando la formula de
#      la distribucion de Poisson. Eso llena la "matriz de marcadores".

MAX_GOLES_MATRIZ = 6  # la matriz cubre marcadores de 0-0 hasta 6-6


def entrenar_poisson(X_train, goles_train):
    """Entrena UNA regresion de Poisson para predecir goles esperados.

    QUE RECIBE:
      X_train      -> las features del partido (mismas que los clasificadores).
      goles_train  -> el numero de goles que marco el equipo (variable objetivo).
    QUE ENTREGA: un modelo entrenado que, dado un partido, estima cuantos goles
      marcara ese equipo (un numero >= 0, llamado 'lambda').

    POR QUE POISSON: la distribucion de Poisson es el estandar para contar
    eventos raros e independientes en un intervalo (goles en un partido). Es el
    metodo clasico de la literatura para modelar goles de futbol.

    Igual que la logistica, escalamos las features primero (PoissonRegressor
    tambien es sensible a la escala), dentro de un pipeline sin fuga.
    """
    modelo = make_pipeline(
        StandardScaler(),
        PoissonRegressor(max_iter=1000),
    )
    modelo.fit(X_train, goles_train)
    return modelo


def matriz_marcadores(lambda_local: float, lambda_visit: float,
                      max_goles: int = MAX_GOLES_MATRIZ) -> np.ndarray:
    """Construye la matriz de probabilidad de cada marcador.

    QUE RECIBE: los goles esperados del local y del visitante (lambdas).
    QUE ENTREGA: una matriz (max_goles+1 x max_goles+1) donde la celda [i, j] es
      la probabilidad de que el partido termine con el local marcando i goles y
      el visitante j. Se asume independencia: P(i, j) = P(local=i) * P(visit=j).
    """
    # poisson.pmf(k, lambda) = probabilidad de marcar exactamente k goles.
    goles = np.arange(max_goles + 1)
    p_local = poisson.pmf(goles, lambda_local)   # vector: P(local marca 0,1,2..)
    p_visit = poisson.pmf(goles, lambda_visit)   # vector: P(visit marca 0,1,2..)
    # Producto externo: cada combinacion (i, j). Filas = goles local, col = visit.
    return np.outer(p_local, p_visit)


def resumen_desde_matriz(matriz: np.ndarray) -> dict:
    """Deriva las salidas de la web a partir de la matriz de marcadores.

    QUE ENTREGA un diccionario con:
      - marcador_probable: el (i, j) de mayor probabilidad (la celda mas alta).
      - prob_marcador: esa probabilidad.
      - p_local / p_empate / p_visit: probabilidades 1X2, sumando las celdas
        por debajo / en / por encima de la diagonal de la matriz.
    """
    # Marcador mas probable: la posicion de la celda con mayor valor.
    i, j = np.unravel_index(np.argmax(matriz), matriz.shape)
    # Probabilidades 1X2 sumando regiones de la matriz:
    p_local = np.tril(matriz, -1).sum()   # debajo de la diagonal: local marca mas
    p_empate = np.trace(matriz)           # diagonal: mismos goles
    p_visit = np.triu(matriz, 1).sum()    # encima de la diagonal: visitante mas
    return {
        "marcador_probable": (int(i), int(j)),
        "prob_marcador": float(matriz[i, j]),
        "p_local": float(p_local),
        "p_empate": float(p_empate),
        "p_visit": float(p_visit),
    }


def evaluar_poisson(modelo_local, modelo_visit, X_val, goles_local_val,
                    goles_visit_val, y_val) -> dict:
    """Evalua el modelo Poisson de dos formas complementarias.

    1) Precision de los goles: error absoluto medio (MAE) entre los goles
       predichos y los reales (¿cuantos goles nos equivocamos en promedio?).
    2) Coherencia 1X2: a partir de los lambdas se derivan probabilidades
       local/empate/visitante; se mide su accuracy y se compara con el Modelo A.
    """
    # Goles esperados (lambda) para cada partido de validacion.
    lam_local = modelo_local.predict(X_val)
    lam_visit = modelo_visit.predict(X_val)

    mae_local = mean_absolute_error(goles_local_val, lam_local)
    mae_visit = mean_absolute_error(goles_visit_val, lam_visit)

    # Para cada partido, derivar el resultado 1X2 mas probable desde su matriz.
    pred_1x2 = []
    for ll, lv in zip(lam_local, lam_visit):
        r = resumen_desde_matriz(matriz_marcadores(ll, lv))
        probs = {"local": r["p_local"], "empate": r["p_empate"],
                 "visitante": r["p_visit"]}
        pred_1x2.append(max(probs, key=probs.get))
    acc_1x2 = (np.array(pred_1x2) == y_val.values).mean()

    print(f"\n{'='*60}\nRESULTADOS — B. Modelo Poisson\n{'='*60}")
    print(f"MAE goles local:     {mae_local:.3f}  (goles de error en promedio)")
    print(f"MAE goles visitante: {mae_visit:.3f}")
    print(f"Accuracy 1X2 derivada de los goles: {acc_1x2*100:.2f}%")
    print("(comparar con el Modelo A: deberia dar parecido si es coherente)")
    return {"nombre": "B. Poisson", "mae_local": mae_local,
            "mae_visit": mae_visit, "accuracy_1x2": acc_1x2}


def graficar_matriz_ejemplo(modelo_local, modelo_visit, X_val, fila_idx: int,
                            equipo_local: str, equipo_visit: str) -> Path:
    """Dibuja la matriz de marcadores de UN partido de ejemplo (mapa de calor).

    Sirve para ilustrar como se vera esta salida en la web: una grilla donde
    cada celda es la probabilidad de un marcador, con el mas probable resaltado.

    Orientacion tipo PLANO CARTESIANO: el eje X (goles del visitante) crece de
    izquierda a derecha y el eje Y (goles del local) crece de ABAJO hacia arriba,
    con el origen (0-0) en la esquina inferior izquierda. Los ejes llevan el
    nombre real de cada equipo.
    """
    fila = X_val.iloc[[fila_idx]]
    lam_l = float(modelo_local.predict(fila)[0])
    lam_v = float(modelo_visit.predict(fila)[0])
    matriz = matriz_marcadores(lam_l, lam_v) * 100  # a porcentaje
    res = resumen_desde_matriz(matriz / 100)
    mi, mj = res["marcador_probable"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # origin="lower" pone la fila 0 (cero goles del local) ABAJO, como en un
    # plano cartesiano (en vez del comportamiento por defecto, que la pone arriba).
    im = ax.imshow(matriz, cmap="YlOrRd", origin="lower")
    ax.set_xlabel(f"Goles de {equipo_visit} (visitante)")
    ax.set_ylabel(f"Goles de {equipo_local} (local)")
    ax.set_title(f"Matriz de marcadores — {equipo_local} vs {equipo_visit}\n"
                 f"(goles esperados: {equipo_local} {lam_l:.2f}, "
                 f"{equipo_visit} {lam_v:.2f})")
    ax.set_xticks(range(MAX_GOLES_MATRIZ + 1))
    ax.set_yticks(range(MAX_GOLES_MATRIZ + 1))
    for i in range(MAX_GOLES_MATRIZ + 1):
        for j in range(MAX_GOLES_MATRIZ + 1):
            # Resaltar el marcador mas probable con texto azul y negrita.
            color = "blue" if (i == mi and j == mj) else "black"
            peso = "bold" if (i == mi and j == mj) else "normal"
            if matriz[i, j] >= 1.0:  # solo etiquetar celdas con >=1%
                ax.text(j, i, f"{matriz[i, j]:.0f}", ha="center", va="center",
                        color=color, fontweight=peso, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probabilidad (%)")
    return _guardar(fig, "matriz_marcadores_ejemplo.png")


def main() -> int:
    """Ejecuta el flujo de modelado: clasificadores A.1 y A.2, con comparacion."""
    print("#" * 60)
    print("# ENTRENAMIENTO DE MODELOS — predccion de partidos")
    print("#" * 60)

    df = cargar_datos()
    X_train, X_val, y_train, y_val, features = separar_temporal(df)
    baseline_accuracy(y_train, y_val)

    resultados = []

    # --- Modelo A.1: Regresion Logistica (base, interpretable) ---
    log_reg = entrenar_logistica(X_train, y_train)
    resultados.append(evaluar(log_reg, X_val, y_val, "A.1 Regresion Logistica"))

    # --- Modelo A.2: Gradient Boosting (potente) ---
    boosting = entrenar_boosting(X_train, y_train)
    resultados.append(evaluar(boosting, X_val, y_val, "A.2 Gradient Boosting"))

    # --- Comparacion de ambos ---
    comparar_modelos(resultados)

    # --- Modelo B: Poisson (goles -> marcador y matriz) ---
    # Separamos los goles reales con la MISMA division temporal que el resto.
    es_train = df["date"].dt.year < ANIO_CORTE_VALIDACION
    gl_train = df.loc[es_train, "home_score"]
    gv_train = df.loc[es_train, "away_score"]
    gl_val = df.loc[~es_train, "home_score"]
    gv_val = df.loc[~es_train, "away_score"]

    poisson_local = entrenar_poisson(X_train, gl_train)   # goles del local
    poisson_visit = entrenar_poisson(X_train, gv_train)   # goles del visitante
    evaluar_poisson(poisson_local, poisson_visit, X_val, gl_val, gv_val, y_val)

    # Graficos explicativos adicionales
    print("\n--- Graficos explicativos ---")
    # Sobre la logistica (interpretable): pesos de features y el caso del empate.
    graficar_importancia_features(log_reg, features, "A.1 Regresion Logistica")
    graficar_empate_por_paridad(df)
    # Sobre el boosting: importancia por permutacion y chequeo de sobreajuste.
    graficar_importancia_permutacion(boosting, X_val, y_val, features,
                                     "A.2 Gradient Boosting")
    graficar_sobreajuste(boosting, X_train, y_train, X_val, y_val,
                         "A.2 Gradient Boosting")
    # Comparacion global de los dos modelos contra el baseline.
    graficar_comparacion_modelos(resultados)
    # Sobre el Poisson: matriz de marcadores de un partido de ejemplo.
    fila0 = df.loc[~es_train].iloc[0]
    graficar_matriz_ejemplo(poisson_local, poisson_visit, X_val, 0,
                            fila0["home_team"], fila0["away_team"])

    print("\nListo. Modelos A (1X2) y B (Poisson) entrenados y evaluados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
