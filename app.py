"""
============================================================================
app.py  —  PAGINA WEB INTERACTIVA (Streamlit) · MUNDIAL FIFA 2026
============================================================================
Es la "cara" del proyecto: la pagina con la que interactua el usuario. Permite
elegir la seleccion local, la visitante y si la cancha es neutral, y muestra
las TRES salidas del modelo en el orden:

  1. Marcador mas probable      -> ranking de los 3 marcadores mas probables.
  2. Matriz de marcadores        -> mapa de calor interactivo (Plotly).
  3. Probabilidades del resultado-> barra horizontal segmentada (1 / X / 2).

Esta web NO contiene la logica de prediccion: solo recoge lo que elige el
usuario, se lo pasa a las funciones de 'prediccion.py' (el "cerebro") y dibuja
el resultado. Asi la interfaz y los calculos quedan separados y ordenados.

RESTRICCIONES DE DOMINIO (Mundial 2026)
---------------------------------------
- Solo se pueden elegir las 48 selecciones clasificadas al Mundial 2026.
- La "importancia" es fija (un partido de Mundial): no se pide al usuario
  porque todas las fases tendrian el mismo peso en el modelo.

COMO EJECUTARLA (en tu maquina, con el entorno activado):
    streamlit run app.py
Se abre sola en el navegador. Necesita 'output/dataset_final.csv' (genera con
'python main.py' si no existe).
============================================================================
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import prediccion


# ===========================================================================
# DOMINIO: LAS 48 SELECCIONES DEL MUNDIAL 2026
# ===========================================================================
# Clave = nombre en ingles (como suele venir en el dataset de Kaggle).
# "esp" = nombre en espanol para mostrar; "iso" = codigo de bandera (flagcdn).
MUNDIAL_2026 = {
    # --- Anfitriones ---
    "Canada":                 {"esp": "Canadá",              "iso": "ca"},
    "Mexico":                 {"esp": "México",              "iso": "mx"},
    "United States":          {"esp": "Estados Unidos",      "iso": "us"},
    # --- AFC (Asia) ---
    "Australia":              {"esp": "Australia",           "iso": "au"},
    "Iran":                   {"esp": "Irán",                "iso": "ir"},
    "Iraq":                   {"esp": "Irak",                "iso": "iq"},
    "Japan":                  {"esp": "Japón",               "iso": "jp"},
    "Jordan":                 {"esp": "Jordania",            "iso": "jo"},
    "Qatar":                  {"esp": "Catar",               "iso": "qa"},
    "Saudi Arabia":           {"esp": "Arabia Saudita",      "iso": "sa"},
    "South Korea":            {"esp": "Corea del Sur",       "iso": "kr"},
    "Uzbekistan":             {"esp": "Uzbekistán",          "iso": "uz"},
    # --- CAF (Africa) ---
    "Algeria":                {"esp": "Argelia",             "iso": "dz"},
    "Cape Verde":             {"esp": "Cabo Verde",          "iso": "cv"},
    "DR Congo":               {"esp": "RD del Congo",        "iso": "cd"},
    "Egypt":                  {"esp": "Egipto",              "iso": "eg"},
    "Ghana":                  {"esp": "Ghana",               "iso": "gh"},
    "Ivory Coast":            {"esp": "Costa de Marfil",     "iso": "ci"},
    "Morocco":                {"esp": "Marruecos",           "iso": "ma"},
    "Senegal":                {"esp": "Senegal",             "iso": "sn"},
    "South Africa":           {"esp": "Sudáfrica",           "iso": "za"},
    "Tunisia":                {"esp": "Túnez",               "iso": "tn"},
    # --- Concacaf (no anfitriones) ---
    "Curaçao":                {"esp": "Curazao",             "iso": "cw"},
    "Haiti":                  {"esp": "Haití",               "iso": "ht"},
    "Panama":                 {"esp": "Panamá",              "iso": "pa"},
    # --- CONMEBOL (Sudamerica) ---
    "Argentina":              {"esp": "Argentina",           "iso": "ar"},
    "Brazil":                 {"esp": "Brasil",              "iso": "br"},
    "Colombia":               {"esp": "Colombia",            "iso": "co"},
    "Ecuador":                {"esp": "Ecuador",             "iso": "ec"},
    "Paraguay":               {"esp": "Paraguay",            "iso": "py"},
    "Uruguay":                {"esp": "Uruguay",             "iso": "uy"},
    # --- OFC (Oceania) ---
    "New Zealand":            {"esp": "Nueva Zelanda",       "iso": "nz"},
    # --- UEFA (Europa) ---
    "Austria":                {"esp": "Austria",             "iso": "at"},
    "Belgium":                {"esp": "Bélgica",             "iso": "be"},
    "Bosnia and Herzegovina": {"esp": "Bosnia y Herzegovina", "iso": "ba"},
    "Croatia":                {"esp": "Croacia",             "iso": "hr"},
    "Czech Republic":         {"esp": "República Checa",     "iso": "cz"},
    "England":                {"esp": "Inglaterra",          "iso": "gb-eng"},
    "France":                 {"esp": "Francia",             "iso": "fr"},
    "Germany":                {"esp": "Alemania",            "iso": "de"},
    "Netherlands":            {"esp": "Países Bajos",        "iso": "nl"},
    "Norway":                 {"esp": "Noruega",             "iso": "no"},
    "Portugal":               {"esp": "Portugal",            "iso": "pt"},
    "Scotland":               {"esp": "Escocia",             "iso": "gb-sct"},
    "Spain":                  {"esp": "España",              "iso": "es"},
    "Sweden":                 {"esp": "Suecia",              "iso": "se"},
    "Switzerland":            {"esp": "Suiza",               "iso": "ch"},
    "Turkey":                 {"esp": "Turquía",             "iso": "tr"},
}

# Variantes de nomenclatura que algunos datasets usan -> nombre canonico de arriba.
# (No afecta lo que se pasa al modelo: solo sirve para resolver nombre y bandera.)
ALIAS = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Curacao": "Curaçao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

# En el Mundial 2026 todos los partidos tienen el mismo peso competitivo para el
# modelo (un partido de Copa del Mundo), asi que la importancia es fija. No se
# pide al usuario elegir "fase" porque no cambiaria el calculo: el modelo
# recibiria el mismo valor en todos los casos.
IMPORTANCIA_PARTIDO = prediccion.utils.IMPORTANCIA_MUNDIAL


def _canonico(nombre: str) -> str:
    """Devuelve el nombre canonico (resolviendo alias) de una seleccion."""
    return ALIAS.get(nombre, nombre)


def equipos_mundial(equipos: list) -> list:
    """Filtra la lista de equipos del dataset y deja solo las 48 del Mundial 2026.

    QUE RECIBE: la lista de nombres de equipo disponibles en el dataset.
    QUE ENTREGA: una lista de diccionarios {raw, canonico, esp, iso} ordenada por
      el nombre en espanol. 'raw' conserva el nombre EXACTO del dataset (es el que
      se pasa al modelo); 'esp'/'iso' se usan solo para mostrar.
    """
    out = []
    for raw in equipos:
        can = _canonico(raw)
        if can in MUNDIAL_2026:
            datos = MUNDIAL_2026[can]
            out.append({"raw": raw, "canonico": can,
                        "esp": datos["esp"], "iso": datos["iso"]})
    out.sort(key=lambda d: d["esp"])
    return out


# ===========================================================================
# CARGA INICIAL (se hace UNA sola vez y se guarda en cache)
# ===========================================================================
@st.cache_resource
def cargar_todo():
    """Carga el dataset, calcula el estado de los equipos y entrena los modelos."""
    dataset = pd.read_csv(prediccion.DATASET, parse_dates=["date"])
    estado = prediccion.estado_actual_equipos(dataset)
    modelos = prediccion.entrenar_modelos(dataset)
    equipos = sorted(estado["equipos"].keys())
    return estado, modelos, equipos


# ===========================================================================
# CALCULO AUXILIAR: TOP-3 MARCADORES MAS PROBABLES
# ===========================================================================
def top3_marcadores(matriz) -> list:
    """Extrae de la matriz los tres marcadores con mayor probabilidad.

    QUE ENTREGA: lista de 3 dicts {gl, gv, prob} ordenada de mayor a menor.
    """
    pares = []
    n = matriz.shape[0]
    for i in range(n):
        for j in range(n):
            pares.append((i, j, float(matriz[i, j])))
    pares.sort(key=lambda t: t[2], reverse=True)
    return [{"gl": i, "gv": j, "prob": p} for i, j, p in pares[:3]]


# ===========================================================================
# ESTILOS (identidad visual Mundial 2026)
# ===========================================================================
# Paleta inspirada en la marca del torneo: azul, verde y rojo de los anfitriones,
# con un dorado del emblema para resaltar lo mas probable.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Saira+Condensed:wght@500;600;700;800&display=swap');

:root{
  --azul:#2A398D; --azul-2:#3D52C9; --verde:#3CAC3B; --rojo:#E61D25;
  --dorado:#E8A93C; --tinta:#0E1430; --tinta-2:#1C2557;
  --papel:#F4F6FC; --nube:#FFFFFF; --gris:#5B6478; --linea:#E4E7F2;
}

.stApp{ background:var(--papel); }
html, body, [class*="css"]{ font-family:'Inter',system-ui,sans-serif; }
.block-container{ padding-top:3.2rem; padding-bottom:3rem; max-width:880px; }
h1,h2,h3,h4{ font-family:'Saira Condensed',sans-serif; letter-spacing:.2px; color:var(--tinta); }

/* Boton primario */
.stButton>button{
  font-family:'Saira Condensed',sans-serif; font-weight:700; font-size:1.05rem;
  text-transform:uppercase; letter-spacing:1px;
  background:var(--azul); color:#fff; border:0; border-radius:12px;
  padding:.7rem 1.4rem; width:100%;
  box-shadow:0 8px 20px rgba(42,57,141,.28);
  transition:transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton>button:hover{ background:var(--azul-2); transform:translateY(-2px); box-shadow:0 12px 26px rgba(42,57,141,.34); }
.stButton>button:active{ transform:translateY(0); }
.stSelectbox label, .stCheckbox label{ font-family:'Inter',sans-serif; font-weight:600; color:var(--tinta); }

/* ====== TARJETA DE EQUIPO SELECCIONADO (bandera bajo el menú) ====== */
.wc-pick{ display:flex; align-items:center; gap:10px; margin-top:8px;
  padding:8px 12px; border-radius:12px; background:var(--nube);
  border:1px solid var(--linea); border-left-width:4px;
  box-shadow:0 2px 8px rgba(14,20,48,.05); }
.wc-pick--local{ border-left-color:var(--azul); }
.wc-pick--visit{ border-left-color:var(--rojo); }
.wc-pick__flag{ width:34px; height:25px; object-fit:cover; border-radius:4px; box-shadow:0 1px 4px rgba(0,0,0,.2); }
/* En la tarjeta pick la bandera es mas pequena que en el banner principal. */
.wc-pick .wc-flag-wrap{ width:38px; height:28px; border-radius:5px; }
.wc-pick .wc-flag{ width:38px; height:28px; border-radius:5px; }
.wc-pick .wc-flag-wrap::after{ font-size:.62rem; }
.wc-pick__name{ font-family:'Saira Condensed',sans-serif; font-weight:700; font-size:1.12rem; color:var(--tinta); line-height:1; }
.wc-pick__role{ margin-left:auto; font-family:'Saira Condensed',sans-serif; font-weight:600;
  font-size:.66rem; letter-spacing:1.2px; text-transform:uppercase; color:var(--gris); }

/* ====== HERO ====== */
.wc-hero{ position:relative; overflow:hidden; border-radius:20px; margin:.2rem 0 1.4rem;
  background:linear-gradient(135deg,#0E1430 0%, #1C2557 58%, #25307a 100%); color:#fff; }
.wc-hero__stripe{ height:8px; width:100%;
  background:linear-gradient(90deg,var(--azul) 0 33.3%,var(--verde) 33.3% 66.6%,var(--rojo) 66.6% 100%); }
.wc-hero__body{ padding:30px 30px 32px; position:relative; z-index:2; }
.wc-hero__kicker{ font-family:'Saira Condensed',sans-serif; font-weight:600; font-size:.82rem;
  letter-spacing:2.5px; text-transform:uppercase; color:#AEB9EC; margin-bottom:8px; }
.wc-hero__title{ font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:3rem;
  line-height:.98; text-transform:uppercase; margin:0; color:#fff; }
.wc-hero__sub{ font-family:'Inter',sans-serif; font-size:1.02rem; color:#D5DBF2; margin:.55rem 0 0; max-width:42ch; }
.wc-hero__chip{ display:inline-block; margin-top:16px; font-size:.78rem; color:#C9D1F0;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16);
  padding:6px 12px; border-radius:999px; }
.wc-hero__mark{ position:absolute; right:6px; top:-34px; font-family:'Saira Condensed',sans-serif;
  font-weight:800; font-size:13rem; line-height:1; color:rgba(255,255,255,.09); z-index:1; user-select:none; }

/* ====== BANNER PARTIDO ====== */
.wc-banner{ display:flex; align-items:stretch; border-radius:18px; overflow:hidden;
  border:1px solid var(--linea); background:var(--nube); box-shadow:0 10px 30px rgba(14,20,48,.08); }
.wc-banner__team{ flex:1; display:flex; flex-direction:column; align-items:center; gap:8px;
  padding:22px 14px; text-align:center; }
.wc-banner__team--local{ background:linear-gradient(180deg,rgba(42,57,141,.07),transparent); }
.wc-banner__team--visit{ background:linear-gradient(180deg,rgba(230,29,37,.07),transparent); }
.wc-flag{ width:62px; height:46px; object-fit:cover; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,.18); position:relative; z-index:1; }
/* Respaldo de bandera: si la imagen no carga, se ve este recuadro con el codigo. */
.wc-flag-wrap{ position:relative; display:inline-flex; align-items:center; justify-content:center;
  width:62px; height:46px; border-radius:6px; background:linear-gradient(135deg,#2A398D,#3D52C9);
  box-shadow:0 2px 8px rgba(0,0,0,.18); overflow:hidden; }
.wc-flag-wrap::after{ content:attr(data-code); position:absolute; font-family:'Saira Condensed',sans-serif;
  font-weight:700; font-size:.8rem; color:rgba(255,255,255,.92); letter-spacing:.5px; z-index:0; }
.wc-flag-wrap .wc-flag{ box-shadow:none; }
.wc-banner__name{ font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:1.45rem;
  text-transform:uppercase; color:var(--tinta); line-height:1; }
.wc-banner__role{ font-family:'Saira Condensed',sans-serif; font-weight:600; font-size:.72rem;
  letter-spacing:1.5px; text-transform:uppercase; }
.wc-banner__role--local{ color:var(--azul); }
.wc-banner__role--visit{ color:var(--rojo); }
.wc-banner__vs{ display:flex; align-items:center; justify-content:center; padding:0 8px;
  font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:1.05rem; color:var(--gris); background:var(--papel); }
.wc-meta{ text-align:center; color:var(--gris); font-size:.86rem; margin:.7rem 0 0; font-weight:500; }
.wc-meta b{ color:var(--tinta); font-weight:700; }

/* ====== EYEBROW de seccion ====== */
.wc-eyebrow{ display:flex; align-items:center; gap:12px; margin:2.1rem 0 .9rem; }
.wc-eyebrow__num{ font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:1rem;
  color:#fff; background:var(--azul); width:30px; height:30px; border-radius:9px;
  display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
.wc-eyebrow__txt{ font-family:'Saira Condensed',sans-serif; font-weight:700; font-size:1.5rem;
  text-transform:uppercase; color:var(--tinta); line-height:1; }
.wc-eyebrow__rule{ flex:1; height:3px; border-radius:3px;
  background:linear-gradient(90deg,var(--azul),var(--verde),var(--rojo)); opacity:.5; }

/* ====== PODIO TOP-3 ====== */
.wc-podium{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }
.wc-card{ position:relative; background:var(--nube); border:1px solid var(--linea);
  border-radius:16px; padding:20px 16px 16px; box-shadow:0 6px 18px rgba(14,20,48,.06);
  transition:transform .18s ease, box-shadow .18s ease; overflow:hidden; }
.wc-card:hover{ transform:translateY(-4px); box-shadow:0 14px 30px rgba(14,20,48,.12); }
.wc-card__edge{ position:absolute; inset:0 0 auto 0; height:5px; background:linear-gradient(90deg,var(--azul),var(--azul-2)); }
.wc-card--gold{ border-color:rgba(232,169,60,.55); box-shadow:0 12px 30px rgba(232,169,60,.20); background:linear-gradient(180deg,#FFFCF4,#FFFFFF); }
.wc-card--gold .wc-card__edge{ height:6px; background:linear-gradient(90deg,#E8A93C,#F6C760); }
.wc-card__top{ display:flex; align-items:center; justify-content:space-between; min-height:22px; margin-bottom:8px; }
.wc-card__rank{ font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:1.35rem; color:var(--gris); line-height:1; }
.wc-card--gold .wc-card__rank{ color:var(--dorado); }
.wc-card__badge{ font-family:'Saira Condensed',sans-serif; font-weight:700; font-size:.62rem;
  text-transform:uppercase; letter-spacing:1px; color:#9A6A12; background:#FBEFCF; padding:4px 8px; border-radius:999px; }
.wc-card__score{ font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:2.5rem; color:var(--tinta); line-height:1; letter-spacing:1px; }
.wc-card--gold .wc-card__score{ font-size:2.9rem; }
.wc-card__teams{ font-size:.74rem; color:var(--gris); margin:.4rem 0 .7rem; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.wc-card__prob{ font-family:'Saira Condensed',sans-serif; font-weight:700; font-size:1.05rem; color:var(--azul); }
.wc-card--gold .wc-card__prob{ color:#B07A14; }
.wc-card__mini{ height:6px; border-radius:999px; background:var(--papel); margin-top:9px; overflow:hidden; }
.wc-card__mini>span{ display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--azul),var(--azul-2)); }
.wc-card--gold .wc-card__mini>span{ background:linear-gradient(90deg,#E8A93C,#F6C760); }

/* ====== BARRA 1X2 ====== */
.wc-bar{ display:flex; width:100%; height:62px; border-radius:14px; overflow:hidden;
  box-shadow:0 6px 18px rgba(14,20,48,.10); border:1px solid var(--linea); }
.wc-bar__seg{ display:flex; align-items:center; justify-content:center;
  font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:1.15rem; color:#fff;
  min-width:0; transition:flex-basis .4s ease; }
.wc-bar__seg--local{ background:var(--azul); }
.wc-bar__seg--draw{ background:var(--verde); }
.wc-bar__seg--visit{ background:var(--rojo); }
.wc-legend{ display:flex; flex-wrap:wrap; gap:10px 24px; margin-top:14px; }
.wc-legend__item{ display:flex; align-items:center; gap:8px; font-size:.94rem; color:var(--tinta); }
.wc-legend__dot{ width:13px; height:13px; border-radius:4px; flex:0 0 auto; }
.wc-legend__dot--local{ background:var(--azul); }
.wc-legend__dot--draw{ background:var(--verde); }
.wc-legend__dot--visit{ background:var(--rojo); }
.wc-legend__item b{ font-family:'Saira Condensed',sans-serif; font-weight:700; }

/* ====== Nota / aviso ====== */
.wc-note{ background:#EEF1FB; border-left:4px solid var(--azul); border-radius:10px;
  padding:12px 16px; font-size:.88rem; color:#34406B; margin:.4rem 0 0; line-height:1.5; }

/* ====== Encabezado de seccion (Configura el partido / Resultados) ====== */
.wc-results-head{ display:flex; align-items:center; gap:14px; margin:1.8rem 0 .9rem; }
.wc-results-head__txt{ font-family:'Saira Condensed',sans-serif; font-weight:800; font-size:1.7rem;
  text-transform:uppercase; color:var(--tinta); line-height:1; white-space:nowrap; }
.wc-results-head__rule{ flex:1; height:4px; border-radius:4px;
  background:linear-gradient(90deg,var(--azul),var(--verde),var(--rojo)); }

/* ====== Recuadro nativo de st.container(border=True) ====== */
/* Engloba tanto la configuracion como los resultados; lo alineamos a la
   identidad visual (fondo blanco, esquinas redondeadas y sombra suave). */
div[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--nube);
  border-radius:20px !important; box-shadow:0 10px 30px rgba(14,20,48,.07); }

/* ====== Responsive (movil) ====== */
@media (max-width:640px){
  .wc-hero__title{ font-size:2.2rem; }
  .wc-hero__mark{ font-size:8rem; top:-14px; }
  .wc-podium{ grid-template-columns:1fr; }
  .wc-banner__name{ font-size:1.1rem; }
  .wc-flag{ width:52px; height:39px; }
  .wc-eyebrow__txt{ font-size:1.2rem; }
  .wc-card--gold .wc-card__score{ font-size:2.6rem; }
}
"""

HERO_HTML = """
<div class="wc-hero">
  <span class="wc-hero__mark">26</span>
  <div class="wc-hero__stripe"></div>
  <div class="wc-hero__body">
    <div class="wc-hero__kicker">FIFA World Cup 26 · Canadá · México · Estados Unidos</div>
    <h1 class="wc-hero__title">Pronóstico de partidos</h1>
    <p class="wc-hero__sub">Modelo estadístico de resultados para las 48 selecciones del Mundial 2026.</p>
    <div class="wc-hero__chip">Proyecto académico · Fundamentos de Ciencia de Datos · César Moreno y Felipe Sandoval</div>
  </div>
</div>"""


def _md(html: str):
    """Inserta HTML colapsando los saltos de linea (evita que Streamlit lo trate
    como bloque de codigo por la indentacion)."""
    st.markdown("".join(linea.strip() for linea in html.strip().splitlines()),
                unsafe_allow_html=True)


def _flag(iso: str, esp: str) -> str:
    """Etiqueta <img> de la bandera con respaldo elegante si no carga.

    La bandera se sirve desde un CDN externo. Si no carga (sin internet o fallo
    del servicio), en lugar de un icono roto se muestra el recuadro con un fondo
    neutro y el codigo del pais, para que la interfaz no se vea defectuosa.
    """
    if not iso:
        return ""
    url = f"https://flagcdn.com/w80/{iso}.png"
    codigo = iso.split("-")[-1].upper()[:3]
    # El <img> va sobre un <span> con fondo: si la imagen falla, queda el fondo
    # con el codigo del pais (data-code) en vez de un icono roto.
    return (f'<span class="wc-flag-wrap" data-code="{codigo}">'
            f'<img class="wc-flag" src="{url}" alt="{esp}" '
            f'''onerror="this.style.display='none'">'''
            f'</span>')


def banner_html(loc: dict, vis: dict) -> str:
    """Franja superior estilo transmisión: bandera + nombre + rol de cada equipo."""
    return f"""
<div class="wc-banner">
  <div class="wc-banner__team wc-banner__team--local">
    {_flag(loc['iso'], loc['esp'])}
    <div class="wc-banner__name">{loc['esp']}</div>
    <div class="wc-banner__role wc-banner__role--local">Local</div>
  </div>
  <div class="wc-banner__vs">VS</div>
  <div class="wc-banner__team wc-banner__team--visit">
    {_flag(vis['iso'], vis['esp'])}
    <div class="wc-banner__name">{vis['esp']}</div>
    <div class="wc-banner__role wc-banner__role--visit">Visitante</div>
  </div>
</div>"""


def eyebrow_html(num: int, txt: str) -> str:
    """Encabezado numerado de seccion con la regla tricolor."""
    return (f'<div class="wc-eyebrow"><div class="wc-eyebrow__num">{num}</div>'
            f'<div class="wc-eyebrow__txt">{txt}</div>'
            f'<div class="wc-eyebrow__rule"></div></div>')


def pick_html(info: dict, rol: str) -> str:
    """Tarjeta con la bandera y el nombre del equipo seleccionado (bajo el menú).

    'rol' es 'local' o 'visitante' y define el color del acento lateral.
    """
    mod = "wc-pick--local" if rol == "local" else "wc-pick--visit"
    etiqueta = "Local" if rol == "local" else "Visitante"
    return (f'<div class="wc-pick {mod}">{_flag(info["iso"], info["esp"])}'
            f'<span class="wc-pick__name">{info["esp"]}</span>'
            f'<span class="wc-pick__role">{etiqueta}</span></div>')


def podium_html(top3: list, loc: dict, vis: dict) -> str:
    """Tres tarjetas con los marcadores mas probables; la #1 resaltada en dorado."""
    pmax = top3[0]["prob"] or 1e-9
    tarjetas = []
    for k, t in enumerate(top3):
        oro = (k == 0)
        clase = "wc-card wc-card--gold" if oro else "wc-card"
        badge = '<div class="wc-card__badge">Más probable</div>' if oro else '<div></div>'
        ancho = max(6.0, t["prob"] / pmax * 100)
        # Etiqueta del tipo de resultado que representa el marcador.
        if t["gl"] > t["gv"]:
            tipo = f"Gana {loc['esp']}"
        elif t["gl"] < t["gv"]:
            tipo = f"Gana {vis['esp']}"
        else:
            tipo = "Empate"
        tarjetas.append(f"""
  <div class="{clase}">
    <div class="wc-card__edge"></div>
    <div class="wc-card__top"><div class="wc-card__rank">#{k + 1}</div>{badge}</div>
    <div class="wc-card__score">{t['gl']} – {t['gv']}</div>
    <div class="wc-card__teams">{tipo}</div>
    <div class="wc-card__prob">{t['prob'] * 100:.1f}%</div>
    <div class="wc-card__mini"><span style="width:{ancho:.1f}%"></span></div>
  </div>""")
    return '<div class="wc-podium">' + "".join(tarjetas) + '</div>'


def barra_html(prob: dict, loc: dict, vis: dict) -> str:
    """Barra horizontal segmentada (local / empate / visitante) + leyenda."""
    pl, pe, pv = prob["local"] * 100, prob["empate"] * 100, prob["visitante"] * 100

    def seg(clase, w):
        """Crea un segmento de la barra con ancho proporcional al porcentaje w.

        Muestra el numero dentro solo si el segmento es ancho (>=12%); si es muy
        estrecho, lo deja sin texto para que no se vea apretado.
        """
        txt = f"{w:.1f}%" if w >= 12 else ""
        return f'<div class="wc-bar__seg {clase}" style="flex:0 0 {w:.2f}%">{txt}</div>'

    barra = ('<div class="wc-bar">'
             + seg("wc-bar__seg--local", pl)
             + seg("wc-bar__seg--draw", pe)
             + seg("wc-bar__seg--visit", pv)
             + '</div>')
    leyenda = f"""
<div class="wc-legend">
  <div class="wc-legend__item"><span class="wc-legend__dot wc-legend__dot--local"></span> Gana {loc['esp']} <b>{pl:.1f}%</b></div>
  <div class="wc-legend__item"><span class="wc-legend__dot wc-legend__dot--draw"></span> Empate <b>{pe:.1f}%</b></div>
  <div class="wc-legend__item"><span class="wc-legend__dot wc-legend__dot--visit"></span> Gana {vis['esp']} <b>{pv:.1f}%</b></div>
</div>"""
    return barra + leyenda


# ===========================================================================
# MATRIZ DE MARCADORES INTERACTIVA (Plotly)
# ===========================================================================
def figura_matriz(matriz, local_esp: str, visit_esp: str, marcador: tuple):
    """Mapa de calor interactivo de la matriz de marcadores.

    Eje X = goles del visitante, eje Y = goles del local; el origen 0-0 queda
    abajo a la izquierda. El marcador mas probable se enmarca en dorado.
    """
    pct = matriz * 100           # pasamos de probabilidad (0-1) a porcentaje
    n = matriz.shape[0]          # tamano de la matriz (7: marcadores 0 a 6)
    mi, mj = marcador            # fila/columna del marcador mas probable
    zmax = float(pct.max()) or 1.0   # mayor probabilidad (para decidir color de texto)
    ejes = list(range(n))        # valores de los ejes: 0,1,2,...,6

    # Escala de color: de un azul muy claro (poca probabilidad) al azul del
    # Mundial (mucha). Cada par [posicion, color] marca un punto del degradado.
    escala = [[0.0, "#EEF0FB"], [0.25, "#C2CAEC"],
              [0.55, "#7C8AD4"], [1.0, "#2A398D"]]

    # Heatmap = la cuadricula de colores. z son los valores (las probabilidades),
    # x/y las posiciones; xgap/ygap dejan un hueco entre celdas; hovertemplate es
    # el texto que aparece al pasar el cursor (%{y}, %{x}, %{z} son los valores
    # de esa celda). El <extra></extra> oculta una etiqueta extra de Plotly.
    fig = go.Figure(go.Heatmap(
        z=pct, x=ejes, y=ejes, colorscale=escala, zmin=0, xgap=3, ygap=3,
        hovertemplate=(f"{local_esp} %{{y}} – %{{x}} {visit_esp}"
                       "<br>Probabilidad: %{z:.1f}%<extra></extra>"),
        colorbar=dict(title=dict(text="Prob. (%)", side="right"),
                      thickness=14, outlinewidth=0, ticksuffix="%"),
    ))

    # Escribimos el numero (porcentaje) dentro de cada celda, solo si es >= 1%
    # (las celdas casi imposibles se dejan vacias para no saturar).
    anotaciones = []
    for i in range(n):
        for j in range(n):
            if pct[i, j] >= 1.0:
                resaltar = (i == mi and j == mj)        # ¿es el marcador top?
                claro = pct[i, j] > 0.5 * zmax          # ¿fondo oscuro? -> texto blanco
                anotaciones.append(dict(
                    x=j, y=i, text=f"{pct[i, j]:.0f}", showarrow=False,
                    font=dict(family="Saira Condensed, sans-serif",
                              size=15 if resaltar else 12,
                              color="#FFFFFF" if claro else "#1A2350")))
    fig.update_layout(annotations=anotaciones)

    # Marco dorado en el marcador mas probable.
    fig.add_shape(type="rect", x0=mj - 0.5, x1=mj + 0.5, y0=mi - 0.5, y1=mi + 0.5,
                  line=dict(color="#E8A93C", width=3), fillcolor="rgba(0,0,0,0)")

    # Ejes con etiquetas enteras explícitas (0..6); automargin evita que el
    # título se solape con los números.
    eje = dict(tickmode="array", tickvals=ejes, ticktext=[str(k) for k in ejes],
               showgrid=False, zeroline=False, ticks="outside", ticklen=4,
               tickcolor="#C2CAEC", color="#3A4163", automargin=True)
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0E1430", size=13),
        xaxis=dict(title=dict(text=f"Goles de {visit_esp} (visitante)", standoff=10),
                   **eje),
        yaxis=dict(title=dict(text=f"Goles de {local_esp} (local)", standoff=10),
                   **eje),
    )
    return fig


# ===========================================================================
# INTERFAZ DE LA PAGINA
# ===========================================================================
def main():
    """Dibuja la pagina completa y orquesta el flujo de la web, en este orden:

    1. Configura la pagina e inyecta los estilos (CSS) y la cabecera (hero).
    2. Carga datos y modelos (una sola vez, en cache) y arma el menu de equipos.
    3. Recoge lo que elige el usuario (local, visitante y tipo de cancha).
    4. Al pulsar el boton, llama a prediccion.predecir_partido y muestra las tres
       salidas: marcador mas probable (top-3), matriz interactiva y barra 1X2.
    """
    st.set_page_config(page_title="Pronóstico · Mundial 2026", page_icon="🏆",
                       layout="centered")
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

    _md(HERO_HTML)

    estado, modelos, equipos_raw = cargar_todo()
    info = equipos_mundial(equipos_raw)
    if len(info) < 2:
        st.error("El dataset no contiene suficientes selecciones del Mundial 2026. "
                 "Genera 'output/dataset_final.csv' con 'python main.py' y vuelve a intentarlo.")
        st.stop()

    por_raw = {d["raw"]: d for d in info}
    opciones = [d["raw"] for d in info]

    # --- Entradas del usuario ---
    # Encabezado de la seccion de configuracion (mismo estilo que el de
    # resultados, para que se vean como dos bloques hermanos).
    _md('<div class="wc-results-head">'
        '<div class="wc-results-head__txt">Configura el partido</div>'
        '<div class="wc-results-head__rule"></div></div>')

    # st.container(border=True) crea un recuadro nativo que ENGLOBA los menus y
    # el check (a diferencia del HTML inyectado, este si puede contener widgets).
    with st.container(border=True):
        # st.columns(2) divide la fila en dos columnas (local y visitante al lado).
        c1, c2 = st.columns(2)
        with c1:
            # index = cual sale preseleccionado al abrir (Brasil si esta).
            idx_l = opciones.index("Brazil") if "Brazil" in opciones else 0
            # selectbox = menu desplegable. 'opciones' son los nombres internos
            # del dataset; format_func los traduce al espanol que ve el usuario
            # (sin cambiar el valor real que se usa para predecir).
            local_raw = st.selectbox("Selección local", opciones, index=idx_l,
                                     format_func=lambda r: por_raw[r]["esp"])
            _md(pick_html(por_raw[local_raw], "local"))   # bandera+nombre
        with c2:
            # Por defecto el visitante es Argentina (o el segundo de la lista).
            if "Argentina" in opciones:
                idx_v = opciones.index("Argentina")
            else:
                idx_v = 1 if len(opciones) > 1 else 0
            visit_raw = st.selectbox("Selección visitante", opciones, index=idx_v,
                                     format_func=lambda r: por_raw[r]["esp"])
            _md(pick_html(por_raw[visit_raw], "visitante"))

        # Tipo de cancha (la importancia es fija para el Mundial, no se pregunta).
        # El espaciador separa el check de las tarjetas de equipo de arriba.
        _md('<div style="height:22px"></div>')
        neutral = st.checkbox(
            "Cancha neutral (sede compartida, sin localía real)", value=True,
            help="En el Mundial 2026 los partidos se juegan en sede neutral. "
                 "Desactívalo solo si la selección local es anfitriona y juega en su país.")
    importancia = IMPORTANCIA_PARTIDO   # valor fijo de partido de Mundial

    # Validacion: no tiene sentido un equipo contra si mismo. st.stop() corta
    # aqui la ejecucion (no dibuja nada mas hasta que el usuario lo corrija).
    if local_raw == visit_raw:
        _md('<div class="wc-note">Elige dos selecciones distintas para generar el pronóstico.</div>')
        st.stop()

    # Estado inicial: mientras el usuario NO haya pulsado el boton, mostramos una
    # nota y paramos. st.button devuelve True solo en el momento del clic.
    if not st.button("Predecir resultado", type="primary"):
        _md('<div class="wc-note">Configura el partido y pulsa '
            '<b>Predecir resultado</b> para ver el marcador más probable, '
            'la matriz de marcadores y las probabilidades del resultado.</div>')
        st.stop()

    # --- Calculo de la prediccion ---
    # Aqui se llama al "cerebro" (prediccion.py) con lo que eligio el usuario.
    # Si algun equipo no tuviera datos, se captura el error y se avisa.
    try:
        r = prediccion.predecir_partido(modelos, estado, local_raw, visit_raw,
                                        neutral, importancia)
    except ValueError as exc:
        st.error(f"No se pudo generar el pronóstico: {exc}")
        st.stop()

    prob = r["probabilidades_1x2"]    # dict con local/empate/visitante
    marc = r["marcador"]              # dict con matriz, lambdas y marcador top
    loc, vis = por_raw[local_raw], por_raw[visit_raw]   # datos de cada equipo

    # ===================== SECCION DE RESULTADOS =====================
    # Encabezado que separa visualmente "lo que configuro el usuario" de "los
    # resultados del modelo".
    _md('<div class="wc-results-head">'
        '<div class="wc-results-head__txt">Resultados de la predicción</div>'
        '<div class="wc-results-head__rule"></div></div>')

    # st.container(border=True) engloba TODAS las salidas en un solo recuadro
    # nativo (puede contener el grafico Plotly, a diferencia de un div HTML).
    with st.container(border=True):
        # Franja del partido + contexto (sin la fase, que ya no se selecciona).
        _md(banner_html(loc, vis))
        sede = "Cancha neutral" if neutral else f"Localía de {loc['esp']}"
        _md(f'<div class="wc-meta">Mundial 2026 &nbsp;·&nbsp; {sede}</div>')

        # --- SECCION 1: marcador mas probable (ranking top-3) ---
        _md(eyebrow_html(1, "Marcador más probable"))
        top3 = top3_marcadores(marc["matriz"])
        _md(podium_html(top3, loc, vis))
        _md(f'<div class="wc-meta" style="text-align:left;margin-top:16px">'
            f'Goles esperados (promedio del modelo): '
            f'<b>{loc["esp"]} {marc["lambda_local"]:.2f}</b> · '
            f'<b>{vis["esp"]} {marc["lambda_visit"]:.2f}</b></div>')

        # --- SECCION 2: matriz de marcadores (interactiva) ---
        _md(eyebrow_html(2, "Matriz de marcadores"))
        st.caption("Probabilidad de cada marcador exacto. Pasa el cursor para ver "
                   "el detalle y usa la rueda del ratón para acercarte. El recuadro "
                   "dorado señala el marcador más probable.")
        fig = figura_matriz(marc["matriz"], loc["esp"], vis["esp"],
                            marc["marcador_probable"])
        st.plotly_chart(fig, use_container_width=True, theme=None,
                        config={"displaylogo": False, "scrollZoom": True,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d"]})

        # --- SECCION 3: probabilidades del resultado (barra segmentada) ---
        _md(eyebrow_html(3, "Probabilidades del resultado"))
        _md(barra_html(prob, loc, vis))

    # Aviso honesto sobre el alcance.
    _md('<div class="wc-note" style="margin-top:2rem">Las cifras son estimaciones '
        'estadísticas basadas en datos históricos (forma reciente, fuerza ELO e '
        'historial directo). El fútbol tiene un alto componente de azar: úsalas '
        'como apoyo a la lectura del partido, no como una certeza.</div>')


if __name__ == "__main__":
    main()
