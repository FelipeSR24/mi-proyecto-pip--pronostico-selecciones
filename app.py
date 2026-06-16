"""
============================================================================
app.py  —  PAGINA WEB INTERACTIVA (Streamlit)
============================================================================
Es la "cara" del proyecto: la pagina con la que interactua el usuario. Permite
elegir el equipo local, el visitante, si la cancha es neutral y la importancia
del partido, y muestra las TRES salidas del modelo:

  1. Probabilidades 1X2 (gana local / empate / gana visitante).
  2. Matriz de marcadores (probabilidad de cada marcador posible).
  3. Marcador mas probable.

Esta web NO contiene la logica de prediccion: solo recoge lo que elige el
usuario, se lo pasa a las funciones de 'prediccion.py' (el "cerebro") y dibuja
el resultado. Asi la interfaz y los calculos quedan separados y ordenados.

COMO EJECUTARLA (en tu maquina, con el entorno activado):
    streamlit run app.py
Se abre sola en el navegador. Necesita 'output/dataset_final.csv' (genera con
'python main.py' si no existe).
============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import prediccion


# ===========================================================================
# CARGA INICIAL (se hace UNA sola vez y se guarda en cache)
# ===========================================================================
# @st.cache_resource hace que el dataset y los modelos se carguen/entrenen solo
# la PRIMERA vez. Sin esto, Streamlit reentrenaria en cada interaccion (lento).
@st.cache_resource
def cargar_todo():
    """Carga el dataset, calcula el estado de los equipos y entrena los modelos.

    QUE ENTREGA: el estado actual de cada seleccion, los modelos entrenados y la
      lista ordenada de equipos disponibles para los menus.
    """
    dataset = pd.read_csv(prediccion.DATASET, parse_dates=["date"])
    estado = prediccion.estado_actual_equipos(dataset)
    modelos = prediccion.entrenar_modelos(dataset)
    equipos = sorted(estado["equipos"].keys())
    return estado, modelos, equipos


# Texto legible para la importancia (lo que ve el usuario -> el numero interno).
OPCIONES_IMPORTANCIA = {
    "Amistoso": prediccion.utils.IMPORTANCIA_AMISTOSO,
    "Clasificatorio / Eliminatoria": prediccion.utils.IMPORTANCIA_CLASIFICATORIO,
    "Torneo oficial (continental, etc.)": prediccion.utils.IMPORTANCIA_COMPETITIVO,
    "Mundial (fase final)": prediccion.utils.IMPORTANCIA_MUNDIAL,
}


def dibujar_matriz(matriz: np.ndarray, local: str, visit: str,
                   marcador: tuple) -> plt.Figure:
    """Dibuja la matriz de marcadores como mapa de calor (orientacion cartesiana).

    El eje X (goles del visitante) crece a la derecha y el eje Y (goles del
    local) crece hacia arriba; el origen 0-0 queda abajo a la izquierda. El
    marcador mas probable se resalta en azul.
    """
    matriz_pct = matriz * 100
    mi, mj = marcador
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matriz_pct, cmap="YlOrRd", origin="lower")
    ax.set_xlabel(f"Goles de {visit} (visitante)")
    ax.set_ylabel(f"Goles de {local} (local)")
    ax.set_title("Probabilidad de cada marcador (%)")
    n = matriz.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    for i in range(n):
        for j in range(n):
            if matriz_pct[i, j] >= 1.0:   # solo etiquetar celdas con >= 1%
                resaltar = (i == mi and j == mj)
                ax.text(j, i, f"{matriz_pct[i, j]:.0f}", ha="center",
                        va="center", fontsize=8,
                        color="blue" if resaltar else "black",
                        fontweight="bold" if resaltar else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probabilidad (%)")
    fig.tight_layout()
    return fig


# ===========================================================================
# INTERFAZ DE LA PAGINA
# ===========================================================================
def main():
    st.set_page_config(page_title="Prediccion de partidos", page_icon="⚽",
                       layout="centered")

    st.title("⚽ Predicción de partidos de selecciones")
    st.caption("Proyecto académico — Fundamentos de Ciencia de Datos · "
               "Cesar Moreno y Felipe Sandoval")

    # Aviso honesto sobre el alcance (acordado): no es una bola de cristal.
    st.info("Esta herramienta estima probabilidades **con base en datos "
            "históricos** (forma reciente, fuerza ELO e historial directo). "
            "El fútbol tiene un fuerte componente de azar: las cifras son un "
            "apoyo a la decisión, no una certeza. Proyecto con fines académicos.")

    estado, modelos, equipos = cargar_todo()

    # --- Entradas del usuario ---
    st.subheader("Configura el partido")
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Equipo local", equipos,
                             index=equipos.index("Brazil")
                             if "Brazil" in equipos else 0)
    with col2:
        visit = st.selectbox("Equipo visitante", equipos,
                             index=equipos.index("Argentina")
                             if "Argentina" in equipos else 1)

    col3, col4 = st.columns(2)
    with col3:
        neutral = st.checkbox("Cancha neutral (ninguno juega en casa)")
    with col4:
        etiqueta_imp = st.selectbox("Importancia del partido",
                                    list(OPCIONES_IMPORTANCIA.keys()),
                                    index=2)
    importancia = OPCIONES_IMPORTANCIA[etiqueta_imp]

    # Validacion simple: no permitir el mismo equipo en ambos lados.
    if local == visit:
        st.warning("Elige dos selecciones distintas.")
        st.stop()

    if not st.button("Predecir resultado", type="primary"):
        st.stop()

    # --- Calculo de la prediccion ---
    try:
        r = prediccion.predecir_partido(modelos, estado, local, visit,
                                        neutral, importancia)
    except ValueError as exc:
        st.error(f"No se pudo predecir: {exc}")
        st.stop()

    prob = r["probabilidades_1x2"]
    marc = r["marcador"]

    # --- SALIDA 1: probabilidades 1X2 ---
    st.subheader("1. Probabilidades del resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {local}", f"{prob['local']*100:.1f}%")
    c2.metric("Empate", f"{prob['empate']*100:.1f}%")
    c3.metric(f"Gana {visit}", f"{prob['visitante']*100:.1f}%")
    # Barra visual de las tres probabilidades.
    st.bar_chart(pd.DataFrame({
        "Probabilidad (%)": {
            f"Gana {local}": prob["local"] * 100,
            "Empate": prob["empate"] * 100,
            f"Gana {visit}": prob["visitante"] * 100,
        }
    }))

    # --- SALIDA 3: marcador mas probable ---
    st.subheader("2. Marcador más probable")
    mi, mj = marc["marcador_probable"]
    st.markdown(f"### {local} {mi} – {mj} {visit}")
    st.caption(f"Probabilidad de ese marcador exacto: "
               f"{marc['prob_marcador']*100:.1f}%  ·  "
               f"Goles esperados: {local} {marc['lambda_local']:.2f}, "
               f"{visit} {marc['lambda_visit']:.2f}")

    # --- SALIDA 2: matriz de marcadores ---
    st.subheader("3. Matriz de marcadores")
    fig = dibujar_matriz(marc["matriz"], local, visit,
                         marc["marcador_probable"])
    st.pyplot(fig)
    st.caption("Cada celda es la probabilidad de ese marcador exacto. La celda "
               "en azul es el marcador más probable.")


if __name__ == "__main__":
    main()
