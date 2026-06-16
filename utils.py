"""
============================================================================
utils.py  —  CAJA DE HERRAMIENTAS DEL PROYECTO
============================================================================
Aqui viven TODAS las funciones que hacen el trabajo real. main.py solo las
llama en orden; la logica detallada esta aqui. Separar el codigo asi (un
"director" y una "caja de herramientas") hace que cada archivo sea mas corto,
mas facil de leer y de probar.

Las funciones estan agrupadas en cuatro bloques:
  1. Carga y validacion   -> leer los CSV y verificar que esten bien.
  2. EDA                   -> describir y resumir los datos.
  3. Transformacion        -> limpiar y construir las variables del modelo.
  4. Graficos del EDA       -> generar las figuras PNG.

Convencion: las funciones que empiezan con guion bajo (p. ej. _guardar) son
"privadas", es decir, de uso interno de este archivo.
============================================================================
"""

from collections import defaultdict
from pathlib import Path

import matplotlib

# El backend 'Agg' debe fijarse ANTES de importar pyplot. 'Agg' dibuja a un
# archivo de imagen sin abrir ninguna ventana, lo que permite generar los PNG
# en un servidor o en cualquier entorno sin pantalla.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (va despues de use() a proposito)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# DICCIONARIO DE NOMBRES DE PAISES
# ---------------------------------------------------------------------------
# Algunas selecciones cambiaron de nombre. Para no tratar a un mismo equipo
# como si fueran dos distintos, traducimos los nombres antiguos al actual.
# Solo incluimos casos que pueden aparecer desde el anio 2000 (el alcance del
# proyecto). La fusion "Serbia and Montenegro" -> "Serbia" es una decision de
# modelado: asumimos continuidad deportiva (documentado en DATABASE.md).
MAPA_NOMBRES: dict[str, str] = {
    "Serbia and Montenegro": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Yugoslavia": "Serbia",
    "Czechia": "Czech Republic",
}
# en validar_columnas() para detectar a tiempo un archivo incorrecto o cambiado.
COLUMNAS_RESULTS: list[str] = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


# ===========================================================================
# BLOQUE 1 — CARGA Y VALIDACION DE DATOS
# ===========================================================================
def cargar_dataset(nombre_archivo: str, carpeta: str | Path = "datasets") -> pd.DataFrame:
    """Lee un archivo CSV de la carpeta de datos y lo devuelve como DataFrame.

    Si el archivo no existe, lanza un error con un mensaje CLARO que le dice al
    usuario que debe descargar los CSV de Kaggle, en vez de soltar un error
    tecnico de pandas que seria mas dificil de entender.
    """
    ruta = Path(carpeta) / nombre_archivo  # arma la ruta de forma segura
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro '{ruta}'. Descarga los CSV de Kaggle "
            f"(ver DATABASE.md) y colocalos en la carpeta '{carpeta}/'."
        )
    return pd.read_csv(ruta)


def validar_columnas(df: pd.DataFrame, esperadas: list[str], nombre: str) -> None:
    """Comprueba que el DataFrame tenga todas las columnas necesarias.

    Es una "red de seguridad": si Kaggle cambia el formato del archivo o se
    descarga el equivocado, el pipeline avisa de inmediato y de forma clara.
    """
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Al dataset '{nombre}' le faltan las columnas {faltantes}. "
            "Verifica que descargaste la version correcta desde Kaggle."
        )


# ===========================================================================
# BLOQUE 2 — FUNCIONES DE EDA (describir y resumir los datos)
# ===========================================================================
def mostrar_estructura(df: pd.DataFrame, nombre: str) -> None:
    """Imprime un resumen de estructura de un DataFrame.

    Muestra cuantas filas y columnas tiene, los tipos de dato (.info()) y
    cuantos valores nulos hay por columna. Es lo primero que se mira en un EDA.
    """
    print(f"\n=== Estructura: {nombre} ===")
    print(f"Filas y columnas: {df.shape}")
    df.info()
    print("Nulos por columna:")
    print(df.isnull().sum())


def contar_duplicados(df: pd.DataFrame, claves: list[str] | None = None) -> int:
    """Cuenta filas duplicadas. Si se pasan 'claves', busca duplicados solo en
    esas columnas; si no, considera la fila completa."""
    return int(df.duplicated(subset=claves).sum())


def proporcion_resultados(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula que porcentaje de partidos gana el local, termina en empate o
    gana el visitante. Da una primera idea de la 'ventaja de local'."""
    # Comparamos goles del local vs visitante y contamos cada caso.
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
    """Lista cada seleccion y en cuantos partidos aparece (como local o
    visitante), ordenadas de mayor a menor numero de partidos."""
    # Juntamos las columnas de local y visitante en una sola serie y contamos.
    serie = pd.concat([results["home_team"], results["away_team"]])
    conteo = serie.value_counts()
    return conteo.rename_axis("pais").reset_index(name="partidos")


def ranking_rendimiento(df: pd.DataFrame,
                        min_partidos: int = 50) -> pd.DataFrame:
    """
    Calcula el rendimiento de cada seleccion en PUNTOS POR PARTIDO (un promedio
    que premia ganar). Suma lo que hizo el equipo tanto de local como de
    visitante. Filtra equipos con muy pocos partidos, porque con pocos datos el
    promedio no es fiable (un equipo con 2 partidos ganados tendria 3.0 perfecto).
    """
    # Puntos del equipo LOCAL en cada partido, segun el resultado.
    local = pd.DataFrame({
        "equipo": df["home_team"],
        "pts": (df["resultado"].map({"local": 3, "empate": 1, "visitante": 0})),
    })
    # Puntos del equipo VISITANTE (ojo: para el, "visitante" significa victoria).
    visit = pd.DataFrame({
        "equipo": df["away_team"],
        "pts": (df["resultado"].map({"local": 0, "empate": 1, "visitante": 3})),
    })
    # Unimos ambas perspectivas para tener todos los partidos de cada equipo.
    largo = pd.concat([local, visit], ignore_index=True)

    # Agrupamos por equipo: contamos partidos y sumamos puntos.
    agg = (largo.groupby("equipo")["pts"]
           .agg(partidos="count", pts_totales="sum")
           .reset_index())
    # Promedio de puntos por partido (la metrica que ordena el ranking).
    agg["pts_por_partido"] = (agg["pts_totales"] / agg["partidos"]).round(3)
    # Descartamos equipos con muy pocos partidos.
    agg = agg[agg["partidos"] >= min_partidos]
    # Ordenamos de mejor a peor rendimiento.
    return agg.sort_values("pts_por_partido", ascending=False).reset_index(
        drop=True)


def head_to_head(df: pd.DataFrame, equipo_a: str,
                 equipo_b: str) -> pd.DataFrame:
    """
    Historial directo ("cara a cara") entre dos selecciones. Cuenta, desde la
    perspectiva de `equipo_a`, cuantas veces gano, empato y perdio contra
    `equipo_b`, sin importar quien jugo de local.
    """
    # Mascara booleana: seleccionamos los partidos donde se enfrentaron estos
    # dos equipos, en cualquiera de los dos ordenes (A vs B o B vs A).
    mask = (((df["home_team"] == equipo_a) & (df["away_team"] == equipo_b)) |
            ((df["home_team"] == equipo_b) & (df["away_team"] == equipo_a)))
    sub = df[mask]

    # Recorremos esos partidos y clasificamos cada uno desde la vista de A.
    victorias = empates = derrotas = 0
    for _, fila in sub.iterrows():
        if fila["resultado"] == "empate":
            empates += 1
        # A gano si: era local y gano el local, O era visitante y gano el visitante.
        elif ((fila["home_team"] == equipo_a and fila["resultado"] == "local")
              or (fila["away_team"] == equipo_a
                  and fila["resultado"] == "visitante")):
            victorias += 1
        else:
            derrotas += 1

    # Devolvemos un resumen ordenado en forma de tabla.
    return pd.DataFrame({
        "metrica": ["partidos", f"victorias_{equipo_a}", "empates",
                    f"victorias_{equipo_b}"],
        "valor": [len(sub), victorias, empates, derrotas],
    })


# ===========================================================================
# BLOQUE 3 — TRANSFORMACION (limpiar y construir las variables del modelo)
# ===========================================================================
def filtrar_desde(results: pd.DataFrame, anio: int = 1970) -> pd.DataFrame:
    """
    Limpia y filtra los partidos. Devuelve solo los jugados desde 'anio',
    ordenados por fecha. El ORDEN de los pasos importa:
    """
    results = results.copy()  # copiamos para no modificar el DataFrame original
    # 1) Convertimos la fecha de texto a un tipo fecha real (permite ordenar y
    #    extraer el anio mas adelante).
    results["date"] = pd.to_datetime(results["date"])
    # 2) Quitamos partidos sin marcador: no sirven para nuestro objetivo.
    results = results.dropna(subset=["home_score", "away_score"])
    # 3) Eliminamos filas duplicadas exactas (registros repetidos).
    results = results.drop_duplicates()
    # 4) Nos quedamos solo con los partidos desde el anio indicado.
    results = results[results["date"].dt.year >= anio]
    # 5) Ordenamos por fecha. Esto es IMPRESCINDIBLE: la "forma reciente" se
    #    calcula recorriendo los partidos en orden cronologico.
    return results.sort_values("date").reset_index(drop=True)


def unificar_nombres(results: pd.DataFrame,
                     mapa: dict[str, str] = MAPA_NOMBRES) -> pd.DataFrame:
    """Reemplaza los nombres antiguos de selecciones por su nombre actual,
    usando el diccionario MAPA_NOMBRES. Asi un mismo equipo no aparece partido
    en dos identidades distintas (p. ej. 'Czechia' y 'Czech Republic')."""
    # Trabajamos sobre una COPIA para no alterar el DataFrame original que nos
    # pasaron (buena practica: una funcion no debe tener "efectos secundarios"
    # sobre los datos de quien la llama). 'results' pasa a apuntar a la copia.
    results = results.copy()
    # .replace(mapa) cambia cada valor que sea una clave del diccionario por su
    # valor correspondiente; los demas nombres quedan intactos.
    results["home_team"] = results["home_team"].replace(mapa)
    results["away_team"] = results["away_team"].replace(mapa)
    return results


def crear_target(results: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'resultado', que es la VARIABLE OBJETIVO: lo que el futuro
    modelo intentara predecir. Tiene 3 clases segun los goles:
      local     -> el equipo de casa marco mas goles
      empate    -> mismo numero de goles
      visitante -> el visitante marco mas goles
    """
    results = results.copy()  # copia defensiva (no modificar el original recibido)
    # np.select evalua una lista de condiciones EN ORDEN y asigna el primer
    # valor cuya condicion sea verdadera; si ninguna lo es, usa 'default'.
    condiciones = [
        results["home_score"] > results["away_score"],   # gana local
        results["home_score"] == results["away_score"],  # empate
    ]
    opciones = ["local", "empate"]
    # Si no gana el local ni hay empate, por descarte gano el visitante.
    results["resultado"] = np.select(condiciones, opciones, default="visitante")
    return results


def construir_features(results: pd.DataFrame, ventana: int = 5) -> pd.DataFrame:
    """
    Construye, para cada partido, la 'forma reciente' de cada equipo SIN fuga
    de informacion (cada partido solo usa datos de partidos ANTERIORES).

    Esta es la funcion mas importante del proyecto. La "forma reciente" resume
    como venia jugando un equipo (goles a favor, en contra y puntos de sus
    ultimos partidos) ANTES de cada partido. La regla de oro es: al describir un
    partido, jamas usar informacion de ese mismo partido ni de partidos futuros,
    porque el modelo no la tendria disponible en la vida real.

    Se hace en 3 pasos:
      1. Pasar a "formato largo": una fila por equipo por partido.
      2. Calcular medias moviles desplazadas (shift + rolling).
      3. Volver a unir las variables al partido, separadas en local y visitante.
    """
    results = results.copy()  # copia defensiva (no modificar el original recibido)
    # Damos a cada partido un identificador unico para poder reagrupar luego.
    # Si quien llama YA asigno 'match_id' (p. ej. main.py, para compartir el
    # mismo id entre todas las features: forma, ELO, head-to-head, importancia),
    # lo respetamos; si no, lo creamos aqui. Asi todo queda alineado por match_id.
    if "match_id" not in results.columns:
        results["match_id"] = range(len(results))

    # ----- PASO 1: FORMATO LARGO -----
    # Un partido tiene DOS equipos. Para analizar la trayectoria de cada equipo,
    # "desdoblamos" cada partido en dos filas: una vista desde el local y otra
    # desde el visitante. 'gf' = goles a favor, 'gc' = goles en contra.
    local = pd.DataFrame({
        "match_id": results["match_id"],
        "date": results["date"],
        "equipo": results["home_team"],
        "gf": results["home_score"],   # el local marca su home_score
        "gc": results["away_score"],   # y recibe el away_score
        "es_local": 1,
    })
    visitante = pd.DataFrame({
        "match_id": results["match_id"],
        "date": results["date"],
        "equipo": results["away_team"],
        "gf": results["away_score"],   # el visitante marca su away_score
        "gc": results["home_score"],   # y recibe el home_score
        "es_local": 0,
    })
    # Apilamos ambas vistas: ahora hay una fila por (equipo, partido).
    largo = pd.concat([local, visitante], ignore_index=True)

    # Puntos que se lleva el equipo en ESE partido: 3 si gana, 1 si empata, 0 si pierde.
    largo["pts"] = np.select(
        [largo["gf"] > largo["gc"], largo["gf"] == largo["gc"]],
        [3, 1],
        default=0,
    )

    # ----- PASO 2: "FORMA RECIENTE" SIN MIRAR EL FUTURO -----
    # Queremos que cada fila tenga el promedio de los ULTIMOS partidos del
    # equipo, pero SIN incluir el partido actual (eso seria "hacer trampa":
    # el modelo no puede saber el resultado de hoy para predecir hoy).
    #
    # Primero ordenamos los partidos de cada equipo del mas viejo al mas nuevo.
    largo = largo.sort_values(["equipo", "date"]).reset_index(drop=True)

    # Agrupamos por equipo para que los calculos de un equipo nunca se mezclen
    # con los de otro (la "forma" de Brasil no debe usar partidos de Peru).
    g = largo.groupby("equipo")

    # Para cada equipo aplicamos DOS operaciones encadenadas a sus goles/puntos:
    #
    #   1) .shift(1)  -> DESPLAZA los valores una fila hacia abajo. Asi, la fila
    #                    de "hoy" pasa a mostrar el dato de "ayer". Es el truco
    #                    que evita la fuga de informacion: el partido actual
    #                    queda excluido de su propio promedio.
    #
    #   2) .rolling(ventana).mean() -> hace el promedio de una "ventana
    #                    deslizante" de los ultimos 'ventana' partidos (por
    #                    defecto 5). min_periods=1 permite calcular aunque el
    #                    equipo tenga menos de 5 partidos jugados al inicio.
    #
    # EJEMPLO con los puntos de un equipo (ventana=3):
    #   Partidos en orden:        [3, 0, 1, 3, 3]   (pts de cada partido)
    #   Tras shift(1):            [NaN, 3, 0, 1, 3] (corre todo una posicion)
    #   Tras rolling(3).mean():   [NaN, 3, 1.5, 1.33, 2.33]
    #   -> En el 4o partido, su "forma" es 1.33 = promedio de los 3 anteriores
    #      (3, 0, 1), SIN contar el partido 4. Exactamente lo que queremos.
    #
    # Usamos .transform() porque devuelve un resultado del mismo tamano que la
    # tabla original, listo para asignarlo como una columna nueva.
    def forma_reciente(serie: pd.Series) -> pd.Series:
        """Promedio de los 'ventana' valores ANTERIORES (sin el actual)."""
        return serie.shift(1).rolling(ventana, min_periods=1).mean()

    largo["forma_gf"] = g["gf"].transform(forma_reciente)    # goles a favor
    largo["forma_gc"] = g["gc"].transform(forma_reciente)    # goles en contra
    largo["forma_pts"] = g["pts"].transform(forma_reciente)  # puntos

    # ----- PASO 3: VOLVER A UNIR AL PARTIDO -----
    # Nos quedamos con el id, el rol y las tres variables de forma.
    feats = largo[["match_id", "es_local", "forma_gf", "forma_gc", "forma_pts"]]

    # Separamos las filas del local y las del visitante, y renombramos las
    # columnas con prefijo para distinguirlas al volver a juntarlas.
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

    # Partimos de las columnas base del partido y le "pegamos" (merge) las
    # variables de forma del local y del visitante usando el match_id.
    final = results[["match_id", "date", "home_team", "away_team",
                     "neutral", "resultado"]].copy()
    final = final.merge(local_feats, on="match_id")
    final = final.merge(visit_feats, on="match_id")

    # Variable de DIFERENCIA: cuanto mejor (o peor) llega el local respecto al
    # visitante. Suele ser muy informativa para el modelo: si es positiva, el
    # local llega en mejor forma.
    final["dif_forma_pts"] = final["local_forma_pts"] - final["visit_forma_pts"]

    # Los PRIMEROS partidos de cada equipo no tienen historial previo (su forma
    # queda vacia), asi que se descartan. Es ~1% de las filas.
    final = final.dropna().reset_index(drop=True)
    return final


# ---------------------------------------------------------------------------
# FEATURES ADICIONALES (Fase 4): importancia del torneo, ELO y head-to-head.
# Las tres reciben los partidos limpios CON columna 'match_id' y devuelven un
# frame delgado [match_id, ...features] para unirlo al dataset por ese id, igual
# que se hace con la forma reciente. Las tres respetan la regla de oro: cada
# partido solo usa informacion conocida ANTES de jugarlo (sin fuga).
# ---------------------------------------------------------------------------

# Escala ORDINAL de importancia del partido. Es un dato del propio partido (el
# torneo en que se juega), conocido ANTES del pitido inicial, asi que NO mete
# fuga de informacion. A mayor numero, mas hay en juego.
IMPORTANCIA_AMISTOSO = 0        # amistosos (sin nada en juego)
IMPORTANCIA_COMPETITIVO = 1  # eliminatorias / clasificatorios
IMPORTANCIA_CLASIFICATORIO = 2     # otros torneos oficiales (continentales, etc.)
IMPORTANCIA_MUNDIAL = 3         # fase final de la Copa del Mundo


def _clasificar_torneo(torneo: str) -> int:
    """Traduce el nombre del torneo a un nivel ORDINAL de importancia (0 a 3).

    Funcion interna. Compara el nombre del torneo (en minusculas) con palabras
    clave. El ORDEN de las comprobaciones IMPORTA: 'FIFA World Cup qualification'
    contiene tanto 'qualif' como 'world cup' y debe contar como clasificatorio
    (no como fase final), por eso 'qualif' se evalua ANTES que 'world cup'.
    """
    t = str(torneo).lower()
    if "friendly" in t:
        return IMPORTANCIA_AMISTOSO
    if "qualif" in t:
        return IMPORTANCIA_CLASIFICATORIO
    if "world cup" in t:
        return IMPORTANCIA_MUNDIAL
    return IMPORTANCIA_COMPETITIVO


def calcular_importancia(results: pd.DataFrame) -> pd.DataFrame:
    """Devuelve [match_id, importancia]: que tanto esta en juego en cada partido.

    QUE hace: convierte la columna 'tournament' en una variable ordinal
    'importancia' (amistoso=0 < clasificatorio=1 < competitivo=2 < mundial=3).
    POR QUE: las selecciones rinden distinto segun lo que este en juego (un
    amistoso no se afronta igual que un Mundial) y esa senal no estaba en el
    dataset. No hay fuga: el torneo se conoce antes de jugar el partido.
    """
    results = results.copy()  # no mutar el DataFrame de entrada
    importancia = results["tournament"].apply(_clasificar_torneo)
    return pd.DataFrame({
        "match_id": results["match_id"],
        "importancia": importancia,
    })


# ---------------------------------------------------------------------------
# Constantes oficiales del ELO (segun eloratings.net, el ranking mundial).
# ---------------------------------------------------------------------------
# Ventaja de jugar en casa, expresada en puntos ELO. eloratings.net usa 100:
# es como si el local llegara con 100 puntos extra al calcular su expectativa.
# NO se aplica en cancha neutral (ahi nadie tiene ventaja de campo).
ELO_VENTAJA_LOCAL = 100.0

# Rating inicial de una seleccion que aun no tiene historial (valor estandar).
ELO_BASE = 1500.0

# Peso K por TIPO de torneo (cuanto puede cambiar el rating tras un partido).
# Cuanto mas importante el partido, mayor el peso. Son los valores oficiales de
# eloratings.net, indexados por la escala de importancia (0 a 3) que ya
# calculamos con _clasificar_torneo. A mayor importancia, mayor K.
ELO_K_POR_IMPORTANCIA = {
    IMPORTANCIA_AMISTOSO: 20.0,       # amistoso: el resultado pesa poco
    IMPORTANCIA_CLASIFICATORIO: 40.0,  # clasificatorias
    IMPORTANCIA_COMPETITIVO: 30.0,     # otros torneos oficiales
    IMPORTANCIA_MUNDIAL: 60.0,         # fase final del Mundial: pesa mucho
}


def _factor_g(diferencia_goles: int) -> float:
    """Factor G: cuanto AMPLIFICA el marcador el cambio de rating (margen de gol).

    Recibe: la diferencia de goles del partido (en valor absoluto).
    Entrega: un multiplicador G segun la formula OFICIAL de eloratings.net.

    Idea: ganar por mucho demuestra mas superioridad que ganar por poco, asi que
    mueve mas el rating. Pero con "rendimientos decrecientes": pasar de ganar por
    1 a por 2 importa mas que pasar de 6 a 7. Formula oficial:
       G = 1                  si la diferencia es 0 o 1 gol
       G = 1.5                si la diferencia es 2 goles
       G = 1.75               si la diferencia es 3 goles
       G = 1.75 + (N-3)/8     si la diferencia es 4 o mas goles (N = diferencia)
    """
    n = abs(diferencia_goles)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 1.75
    return 1.75 + (n - 3) / 8.0


def calcular_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula el rating ELO de cada seleccion (version OFICIAL eloratings.net).

    QUE RECIBE: el DataFrame de partidos 'results', que debe tener las columnas
      match_id, date, home_team, away_team, home_score, away_score, neutral
      e importancia (la escala 0-3 calculada por calcular_importancia).

    QUE ENTREGA: un DataFrame con una fila por partido y las columnas
      [match_id, local_elo, visit_elo, dif_elo], donde cada ELO es el rating que
      el equipo tenia ANTES de ese partido. 'dif_elo' = local_elo - visit_elo.

    QUE ES EL ELO: un puntaje de fuerza (heredado del ajedrez) que sube al ganar
    y baja al perder. Su gracia es que PONDERA AL RIVAL: ganarle a un equipo
    fuerte sube mas que ganarle a uno debil. Todas las selecciones arrancan en
    1500. Captura la fuerza ACUMULADA a largo plazo, la senal mas predictiva en
    selecciones (complementa a la "forma reciente", que solo mira 5 partidos).

    COMO FUNCIONA (para cada partido, en orden cronologico):
      1. Se lee el rating PREVIO de ambos equipos (eso es la feature que se guarda).
      2. Se calcula la expectativa de victoria del local con la formula logistica,
         sumando la ventaja de local (100 puntos) salvo en cancha neutral.
      3. Tras ver el resultado real, se actualiza el rating:
             nuevo = previo + K * G * (resultado_real - esperado)
         donde K depende de la importancia del torneo (amistoso=20 ... mundial=60)
         y G amplifica segun el margen de goles (ver _factor_g). Es de SUMA CERO:
         lo que gana un equipo lo pierde el otro.

    SIN FUGA DE INFORMACION: el rating se GUARDA antes de actualizarlo, asi que el
    ELO de un partido solo refleja lo ocurrido en partidos ANTERIORES.
    """
    results = results.copy()
    # Recorremos del partido mas antiguo al mas reciente (match_id desempata de
    # forma estable). defaultdict hace que un equipo nuevo arranque en ELO_BASE.
    orden = results.sort_values(["date", "match_id"])
    ratings: dict[str, float] = defaultdict(lambda: ELO_BASE)

    filas = []
    for fila in orden.itertuples(index=False):
        r_local = ratings[fila.home_team]   # rating previo del local
        r_visit = ratings[fila.away_team]   # rating previo del visitante

        # --- Paso 1: guardar el rating PREVIO (la feature, sin fuga) ---
        filas.append({
            "match_id": fila.match_id,
            "local_elo": r_local,
            "visit_elo": r_visit,
            "dif_elo": r_local - r_visit,
        })

        # --- Paso 2: expectativa de victoria del local ---
        # Se suma la ventaja de local salvo en cancha neutral. La formula
        # logistica convierte la diferencia de rating en una probabilidad (0 a 1).
        bonus = 0.0 if fila.neutral else ELO_VENTAJA_LOCAL
        esperado_local = 1.0 / (1.0 + 10 ** (-(r_local + bonus - r_visit) / 400))

        # --- Paso 3: resultado real desde la perspectiva del local ---
        if fila.home_score > fila.away_score:
            real_local = 1.0     # gano el local
        elif fila.home_score == fila.away_score:
            real_local = 0.5     # empate
        else:
            real_local = 0.0     # gano el visitante

        # --- Paso 4: actualizar el rating con K (importancia) y G (margen) ---
        k = ELO_K_POR_IMPORTANCIA.get(fila.importancia, 20.0)
        g = _factor_g(fila.home_score - fila.away_score)
        cambio = k * g * (real_local - esperado_local)
        ratings[fila.home_team] = r_local + cambio   # suma cero:
        ratings[fila.away_team] = r_visit - cambio   # lo que sube uno, baja el otro

    return pd.DataFrame(filas)


def calcular_head_to_head(results: pd.DataFrame) -> pd.DataFrame:
    """Devuelve [match_id, h2h_n, h2h_pts_local, h2h_dif_gol_local]: el historial directo.

    QUE hace: para cada partido mira los enfrentamientos PREVIOS entre esas dos
    mismas selecciones y resume como le fue al equipo LOCAL contra ese rival:
      h2h_n             -> cuantos duelos previos existian (mide la confianza).
      h2h_pts_local     -> puntos promedio del local en esos duelos (0 a 3).
      h2h_dif_gol_local -> diferencia de goles promedio a favor del local.
    POR QUE: algunos cruces tienen una historia marcada (un equipo que casi
    siempre le gana a otro) que ni la forma ni el ELO capturan del todo.

    SIN FUGA: recorremos en orden cronologico y solo usamos duelos ANTERIORES al
    partido actual; el partido se añade al historial DESPUES de calcular sus
    features. Como el 28% de los cruces tiene un solo enfrentamiento, cuando no
    hay historial se rellena con valores neutros (1.5 puntos, 0 de diferencia) y
    h2h_n=0 le avisa al modelo de que esa informacion es poco fiable.
    """
    results = results.copy()
    orden = results.sort_values(["date", "match_id"])
    # Historial por PAR de selecciones. Usamos frozenset para que el par sea el
    # mismo sin importar quien jugo de local ({A,B} == {B,A}).
    historial: dict[frozenset, list] = defaultdict(list)

    filas = []
    for fila in orden.itertuples(index=False):
        local, visitante = fila.home_team, fila.away_team
        previos = historial[frozenset((local, visitante))]

        if previos:
            puntos, difs = [], []
            for h_local, h_gf, h_gc in previos:
                # Reorientamos cada duelo previo a la perspectiva del LOCAL actual:
                # gl = goles del local de hoy en aquel duelo, gr = los del rival.
                gl, gr = (h_gf, h_gc) if h_local == local else (h_gc, h_gf)
                puntos.append(3 if gl > gr else (1 if gl == gr else 0))
                difs.append(gl - gr)
            h2h_n = len(previos)
            h2h_pts = sum(puntos) / h2h_n
            h2h_dif = sum(difs) / h2h_n
        else:
            # Sin enfrentamientos previos: valores neutros (h2h_n=0 lo señala).
            h2h_n, h2h_pts, h2h_dif = 0, 1.5, 0.0

        filas.append({
            "match_id": fila.match_id,
            "h2h_n": h2h_n,
            "h2h_pts_local": h2h_pts,
            "h2h_dif_gol_local": h2h_dif,
        })

        # Añadimos ESTE partido al historial del par, para los FUTUROS (no el actual).
        # Guardamos quien fue local y el marcador para poder reorientarlo despues.
        historial[frozenset((local, visitante))].append(
            (local, fila.home_score, fila.away_score)
        )

    return pd.DataFrame(filas)


# ===========================================================================
# BLOQUE 4 — GRAFICOS DEL EDA
# ===========================================================================
# Cada funcion 'graficar_*' crea una figura y la guarda como PNG en la carpeta
# de salida. El patron es siempre el mismo: preparar los datos -> dibujar ->
# poner titulo y etiquetas -> guardar. La funcion _guardar centraliza el ultimo
# paso para no repetir codigo.
def _guardar(fig: plt.Figure, carpeta: str | Path, nombre: str) -> Path:
    """Guarda una figura como PNG en la carpeta indicada y la cierra.

    Funcion interna (empieza con _). 'tight_layout' ajusta los margenes para que
    no se corten los textos, y 'plt.close' libera memoria tras guardar.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)  # crea la carpeta si no existe
    ruta = carpeta / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)  # dpi = resolucion de la imagen
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")
    return ruta


def graficar_distribucion_target(df: pd.DataFrame,
                                 carpeta: str | Path) -> Path:
    """Barra con la distribucion de la variable objetivo (desbalance)."""
    # value_counts() ordena por frecuencia; reindex fuerza el orden fijo
    # local/empate/visitante para que el grafico siempre se vea igual.
    conteo = df["resultado"].value_counts().reindex(
        ["local", "empate", "visitante"])
    total = conteo.sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    conteo.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e9c46a", "#e76f51"],
                rot=0)
    ax.set_title("Distribución de la variable objetivo")
    ax.set_xlabel("Resultado")
    ax.set_ylabel("Número de partidos")
    for i, v in enumerate(conteo):
        ax.text(i, v, f"{v:,}\n({v / total * 100:.1f}%)",
                ha="center", va="bottom")
    ax.margins(y=0.15)
    return _guardar(fig, carpeta, "01_distribucion_target.png")


def graficar_partidos_por_anio(df: pd.DataFrame, carpeta: str | Path) -> Path:
    """Linea con el numero de partidos por anio (cobertura temporal)."""
    por_anio = df["date"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    por_anio.plot(ax=ax, marker="o", color="#264653")
    ax.set_title("Partidos por año en el dataset final")
    ax.set_xlabel("Año")
    ax.set_ylabel("Número de partidos")
    ax.grid(alpha=0.3)
    if 2020 in por_anio.index:
        ax.annotate("COVID-19", xy=(2020, por_anio[2020]),
                    xytext=(2020, por_anio[2020] + 250),
                    ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="grey"))
    return _guardar(fig, carpeta, "02_partidos_por_anio.png")


def graficar_ventaja_local(df: pd.DataFrame, carpeta: str | Path) -> Path:
    """
    Compara la proporcion local/empate/visitante en cancha propia vs neutral.
    Es la evidencia visual de la 'ventaja de local'.
    """
    # Construimos una tabla con el % de cada resultado, separado por tipo de
    # cancha. La cadena de operaciones hace, paso a paso:
    #   1) groupby("neutral")              -> separa partidos en cancha propia / neutral
    #   2) .value_counts(normalize=True)   -> dentro de cada grupo, la PROPORCION
    #                                         (0 a 1) de local / empate / visitante
    #   3) .mul(100)                       -> pasa la proporcion a porcentaje
    #   4) .rename(...) / .reset_index()   -> da nombre a la columna y aplana el indice
    #   5) .pivot(...)                     -> reordena a una tabla: filas = tipo de
    #                                         cancha, columnas = resultado, celdas = %
    tabla = (df.groupby("neutral")["resultado"]
               .value_counts(normalize=True)
               .mul(100)
               .rename("porcentaje")
               .reset_index()
               .pivot(index="neutral", columns="resultado",
                      values="porcentaje"))
    tabla = tabla[["local", "empate", "visitante"]]  # orden fijo de columnas
    # La columna 'neutral' es booleana (True/False); la traducimos a etiquetas
    # legibles para el eje del grafico.
    tabla.index = tabla.index.map({False: "Cancha propia",
                                   True: "Cancha neutral"})
    fig, ax = plt.subplots(figsize=(7, 4))
    tabla.plot(kind="bar", ax=ax,
               color=["#2a9d8f", "#e9c46a", "#e76f51"], rot=0)
    ax.set_title("Resultado según el tipo de cancha (ventaja de local)")
    ax.set_xlabel("Tipo de cancha")
    ax.set_ylabel("Porcentaje de partidos (%)")
    ax.legend(title="Resultado")
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
    ax.set_title("Diferencia de forma según el resultado del partido")
    ax.set_xlabel("Resultado")
    ax.set_ylabel("dif_forma_pts (forma local − forma visitante)")
    return _guardar(fig, carpeta, "04_dif_forma_vs_resultado.png")


def graficar_top_selecciones(df: pd.DataFrame, carpeta: str | Path,
                             n: int = 15, min_partidos: int = 50) -> Path:
    """
    Barras horizontales con las selecciones de mayor rendimiento (puntos por
    partido), considerando solo equipos con al menos `min_partidos` jugados.
    Da contexto futbolistico: quienes son las potencias del periodo.
    """
    tabla = ranking_rendimiento(df, min_partidos=min_partidos).head(n)
    tabla = tabla.iloc[::-1]  # el mejor queda arriba en barras horizontales
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(tabla["equipo"], tabla["pts_por_partido"], color="#2a9d8f")
    ax.set_title(f"Top {n} selecciones por rendimiento "
                 f"(mín. {min_partidos} partidos, desde 1990)")
    ax.set_xlabel("Puntos por partido (3 victoria · 1 empate · 0 derrota)")
    ax.set_ylabel("Selección")
    for i, v in enumerate(tabla["pts_por_partido"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.margins(x=0.1)
    return _guardar(fig, carpeta, "05_top_selecciones.png")


def graficar_evolucion_ventaja_local(df: pd.DataFrame,
                                     carpeta: str | Path) -> Path:
    """
    Linea con el % de victorias locales por anio (solo partidos NO neutrales).
    Permite ver si la ventaja de local se ha debilitado con el tiempo.
    """
    # Nos quedamos solo con partidos NO neutrales (donde hay un local real).
    # El '~' invierte el booleano: ~neutral = "no es cancha neutral".
    propia = df[~df["neutral"].astype(bool)].copy()
    propia["anio"] = propia["date"].dt.year
    # Por cada anio, calculamos el % de partidos que gano el local. La lambda
    # convierte la columna 'resultado' en True/False (¿fue 'local'?) y .mean()
    # de una serie booleana da justamente la proporcion de True (x100 = %).
    pct_local = (propia.groupby("anio")["resultado"]
                 .apply(lambda s: (s == "local").mean() * 100))
    fig, ax = plt.subplots(figsize=(8, 4))
    pct_local.plot(ax=ax, marker="o", color="#e76f51",
                   label="Victorias locales por año")
    ax.axhline(pct_local.mean(), color="grey", linestyle="--",
               linewidth=0.8, label=f"Promedio: {pct_local.mean():.1f}%")
    ax.set_title("Evolución de la ventaja de local (cancha propia)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Victorias locales (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _guardar(fig, carpeta, "06_evolucion_ventaja_local.png")


def graficar_goles_por_torneo(df_results: pd.DataFrame,
                              carpeta: str | Path, n: int = 10) -> Path:
    """
    Barras con el promedio de goles por partido en los torneos mas frecuentes.
    Usa results crudo (filtrado >= anio) porque el tipo de torneo no pasa al
    dataset final. Responde: en que competencias se anotan mas goles.
    """
    d = df_results.copy()
    d["total_goles"] = d["home_score"] + d["away_score"]
    top_torneos = d["tournament"].value_counts().head(n).index
    sub = d[d["tournament"].isin(top_torneos)]
    promedio = sub.groupby("tournament")["total_goles"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(promedio.index, promedio.values, color="#264653")
    ax.set_title("Promedio de goles por partido según el torneo")
    ax.set_xlabel("Goles por partido (local + visitante)")
    ax.set_ylabel("Torneo")
    for i, v in enumerate(promedio.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.margins(x=0.1)
    return _guardar(fig, carpeta, "07_goles_por_torneo.png")
