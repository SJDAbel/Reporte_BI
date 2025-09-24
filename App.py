from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Tuple
from streamlit_echarts import st_echarts, JsCode
from streamlit_extras.metric_cards import style_metric_cards


import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_echarts import st_echarts, JsCode


from core.fun_varios import (
    cargar_excel, anexar_sectores, resumen_match_sectores
)

from tabs import Ejecutivo, Retomas




# =========================
# Configuración general
# =========================

st.set_page_config(page_title="Reporte Comercial", layout="wide")

# Paleta corporativa sugerida
COLOR_ROJO = "#D32F2F"
COLOR_GRIS = "#9E9E9E"
COLOR_FONDO = "#FFFFFF"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [COLOR_ROJO, COLOR_GRIS, "#455A64"]

# ========================================
# ----- Sidebar: carga de datos  -----
# ========================================

st.sidebar.header("Datos de entrada")

# Carga

dco = pd.DataFrame()
sect = pd.DataFrame()

up_coloc = st.sidebar.file_uploader("📄 Colocaciones (.xlsx)", type=["xlsx","xls"], key="coloc")
up_sect  = st.sidebar.file_uploader("📁 Sectores (.xlsx)",     type=["xlsx","xls"], key="sect")

try:
    # Colocaciones
    dco = cargar_excel(up_coloc, modo="colocaciones")

    # Sectores
    sect = cargar_excel(up_sect, modo="sectores")
    
except Exception as e:
    st.error("Cargar archivo de colocaciones para mostrar la información")
    st.error(f"Error al cargar el archivo: {e}")
    

if dco.empty or sect.empty:
    st.info("Carga un Excel con columnas como 'FechaOperacion', 'NetoConfirmado', 'TasaNominalMensualPorc', 'Moneda', 'CodigoLiquidacion', 'Ejecutivo', 'TipoOperacion', 'RUC/RazonSocial'.")
    st.info("Carga un Excel con columnas como 'RUC' y 'Sector'")
    st.stop()

# Vincular
df = anexar_sectores(dco, sect) 

m = resumen_match_sectores(df)
with st.sidebar.expander("📊 Vinculación RUCPagador ↔ RUC (sectores)"):
    st.write(f"Filas: {m['total_rows']:,}")
    st.write(f"Con sector: {m['con_sector']:,}  (**{m['match_pct']:.1f}%**)")
    if m["faltantes_top"]:
        st.write("Ejemplos sin sector (depurar maestro):")
        for r in m["faltantes_top"]:
            st.write("•", " - ".join(str(v) for v in r.values()))
            
# =========================
# Estructura de Filtros
# =========================

st.sidebar.markdown("---")

tc_pen_por_usd = st.sidebar.number_input("TC Compra USD→PEN", min_value=0.0001, value=3.75, step=0.01,
                                         help="Tipo de cambio aplicado para conversión de dólares (USD) a soles (PEN).")

# Fecha
if "FechaOperacion" in df:
    fmax = pd.to_datetime(df["FechaOperacion"].max()).normalize()
    # primer día del mes de fmax
    mes_ini = fmax.replace(day=1)

    fi_default = mes_ini.date()
    ff_default = fmax.date()
else:
    fi_default = date(2020, 1, 1)
    ff_default = date.today()
    
fi, ff = st.sidebar.date_input("Rango de fechas (FechaOperacion)", (fi_default, ff_default))
fi = pd.to_datetime(fi); ff = pd.to_datetime(ff)

# Moneda
monedas = sorted(df["Moneda"].dropna().astype(str).unique().tolist()) if "Moneda" in df else []
sel_monedas = st.sidebar.multiselect("Moneda", monedas, default=monedas)

# Ejecutivo
ejecutivos = sorted(df["Ejecutivo"].dropna().astype(str).unique().tolist()) if "Ejecutivo" in df else []
sel_ejec = st.sidebar.multiselect("Ejecutivo", ejecutivos, default=ejecutivos)

# Tipo Operación
tipos_validos = ["Factoring","Confirming"]
sel_tipos = st.sidebar.multiselect("Tipo de operación", tipos_validos, default=tipos_validos)

if st.sidebar.button("🔄 Reset filtros"):
    st.experimental_rerun()

# Filtro global aplicado
mask = (df["FechaOperacion"] >= fi) & (df["FechaOperacion"] <= ff)
if sel_monedas: mask &= df["Moneda"].isin(sel_monedas)
if sel_ejec:    mask &= df["Ejecutivo"].isin(sel_ejec)
if sel_tipos:   mask &= df["TipoOperacion"].isin(sel_tipos)
df_f = df[mask].copy()


# =========================
# ----- Pestañas -----
# =========================
tab1, tab2 = st.tabs(["👤 Ejecutivo", "🔁 Retomas"])

with tab1:
    Ejecutivo.render(
        df=df, df_f=df_f, fi=fi, ff=ff,
        sel_monedas=sel_monedas, sel_ejec=sel_ejec, sel_tipos=sel_tipos,
        tc_pen_por_usd=tc_pen_por_usd
    )

with tab2:
    # Retomas NO debe verse afectada por los filtros de fechas/ejecutivo/moneda (salvo que quieras lo contrario)
    Retomas.render_retomas_modo(
        df=df, ff=ff, 
        tc_pen_por_usd=tc_pen_por_usd
    )
    
