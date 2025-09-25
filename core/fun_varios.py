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
from typing import List, Dict

# =========================
# Utilidades y Funciones Generales
# =========================

# Funciones para calcular ventanas de tiempo y formateo. 
# [Lo que hace es ventanas de tiempo de meses calendario completos, a partir de una fecha de referencia ff]
def month_windows_from(ff: pd.Timestamp, months_back: int = 1):
    """
    Devuelve ((m0_start, m0_end), (m1_start, m1_end))
    m0 = mes calendario de 'ff'
    m1 = mes inmediatamente anterior.
    """
    ff = pd.to_datetime(ff)
    m0_start = ff.replace(day=1).normalize()
    m0_end   = (m0_start + pd.offsets.MonthEnd(1))
    m1_end   = (m0_start - pd.Timedelta(days=1))
    m1_start = m1_end.replace(day=1).normalize()
    return (m0_start, m0_end), (m1_start, m1_end)

#Funcion para caluclar porcentaje de variacion.
def pct_delta(cur: float, prev: float) -> str:
    if prev is None or prev == 0 or pd.isna(prev):
        return "0.0%"
    return f"{(cur - prev) / prev:+.1%}"

#Funcion para formatear dinero con separador de miles y 2 decimales.
def fmt_money(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "-"

#Funcion para formatear tasa como porcentaje.
def tasa_a_pct_str(x) -> str: 
    """0.02 -> '2.00%'""" 
    try: 
        if pd.isna(x): return "-" 
        return f"{float(x):,.2f}%" 
    except Exception: 
        return "-"

#Funcion para formatear una columna de dataframe como fecha.
def _to_datetime_robusto(s: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Convierte serie a datetime manejando strings/espacios/NaT."""
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    s = s.astype(str).str.strip().replace({"": None, "nan": None})
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)

#Funcion para cargar excel y normalizar columnas.
@st.cache_data(show_spinner=True, ttl=300)
def cargar_excel(file_or_path, sheet_name: Optional[str] = None, modo: str = "colocaciones") -> pd.DataFrame:
    """
    modo:
      - 'colocaciones' (default): aplica normalización de tu archivo principal
      - 'sectores'    : espera columnas RUC, PAGADOR, SECTOR, GRUPO ECO (o variantes)
      - 'auto'        : detecta según columnas presentes
    """
    xls = pd.ExcelFile(file_or_path)
    hoja = sheet_name or (xls.sheet_names[0] if xls.sheet_names else None)
    df = pd.read_excel(xls, sheet_name=hoja)

    # --------- AUTO-DETECCIÓN ----------
    if modo == "auto":
        if "FechaOperacion" in df.columns:
            modo = "colocaciones"
        elif any(c.upper() == "RUC" for c in df.columns):
            modo = "sectores"
        else:
            raise ValueError("No se reconoce el formato del Excel (colocaciones o sectores).")

    # --------- COLOCACIONES ----------
    if modo == "colocaciones":
        cols_esperadas = [
            "FechaOperacion", "NetoConfirmado", "TasaNominalMensualPorc", "Moneda","TipoPago",
            "CodigoLiquidacion", "Ejecutivo", "TipoOperacion",
            "RUCCliente", "RazonSocialCliente",
            "RUCPagador", "RazonSocialPagador"
        ]
        faltan = [c for c in cols_esperadas if c not in df.columns]
        if faltan:
            st.warning(f"Columnas útiles no encontradas (la app funcionará con lo disponible): {faltan}")

        if "FechaOperacion" in df.columns:
            df["FechaOperacion"] = _to_datetime_robusto(df["FechaOperacion"], dayfirst=False)
            df["Periodo"] = df["FechaOperacion"].dt.to_period("M").astype(str)

        for c in ["NetoConfirmado", "TasaNominalMensualPorc"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "TipoOperacion" in df.columns:
            df["TipoOperacion"] = df["TipoOperacion"].astype(str).str.strip().str.title()

        ordered = [c for c in cols_esperadas if c in df.columns] + [c for c in df.columns if c not in cols_esperadas]
        return df[ordered]

    # --------- SECTORES ----------
    if modo == "sectores":
        # Renombrar a nombres estándar
        ren = {
            "RUC": "RUC",
            "PAGADOR": "Pagador",
            "SECTOR": "SectorEconomico",
            "GRUPO ECO": "GrupoEconomico",
            "GRUPO_ECO": "GrupoEconomico",
            "GRUPO": "GrupoEconomico",
        }
        df = df.rename(columns={c: ren.get(c, c) for c in df.columns})

        # Validación mínima
        if "RUC" not in df.columns:
            raise ValueError("El archivo de sectores debe tener columna 'RUC'.")
        # Normalizar RUC y quedarse con columnas clave
        keep = [c for c in ["RUC", "Pagador", "SectorEconomico", "GrupoEconomico"] if c in df.columns]
        df = df[keep].drop_duplicates(subset=["RUC"], keep="last").reset_index(drop=True)
        # Completar vacíos con etiqueta estándar
        for c in ["SectorEconomico", "GrupoEconomico"]:
            if c in df.columns:
                df[c] = df[c].fillna("SIN SECTOR")
        return df

    raise ValueError(f"modo no soportado: {modo}")

#Funcion para normalizar RUC a 11 dígitos (string).
def _to_str11(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\D", "", regex=True)  # deja solo dígitos
         .replace({"": np.nan})
         .map(lambda x: x.zfill(11))          # 11 dígitos
    )

#Funcion para anexar sectores a colocaciones.
def anexar_sectores(df_coloc: pd.DataFrame, df_sect: pd.DataFrame) -> pd.DataFrame:
    """
    Une el maestro de sectores a las colocaciones:
      df_coloc['RUCPagador'] ↔ df_sect['RUC']
    Crea/actualiza columnas: SectorEconomico, GrupoEconomico.
    """
    if "RUCPagador" not in df_coloc.columns:
        raise ValueError("El archivo de colocaciones debe tener la columna 'RUCPagador'.")
    
    if "RUC" not in df_sect.columns:
        raise ValueError("El archivo de sectores debe tener la columna 'RUC'.")

    d = df_coloc.copy()
    d["RUCPagador_key"] = _to_str11(d["RUCPagador"])
    s = df_sect.copy()
    s["RUC_key"] = _to_str11(s["RUC"])

    merged = d.merge(s, left_on="RUCPagador_key", right_on="RUC_key", how="left")

    # Limpieza
    merged = merged.drop(columns=[c for c in ["RUCPagador_key","RUC_key"] if c in merged.columns])

    # Completa faltantes de sector/grupo
    for c in ["SECTOR","GRUPO ECO."]:
        if c in merged.columns:
            merged[c] = merged[c].fillna("SIN SECTOR")

    return merged

#Funcion para obtener métricas de enlazado RUCPagador ↔ RUC.
def resumen_match_sectores(df_merged: pd.DataFrame) -> dict:
    """Devuelve métricas de enlazado RUCPagador ↔ RUC (para mostrar en la UI)."""
    total = len(df_merged)
    con_sector = int(df_merged["SectorEconomico"].notna().sum()) if "SectorEconomico" in df_merged.columns else 0
    ratio = con_sector / total * 100 if total else 0.0
    # Top pagadores sin sector (para depurar)
    faltantes = []
    if "SectorEconomico" in df_merged.columns:
        miss = df_merged[df_merged["SectorEconomico"].eq("SIN SECTOR")]
        if not miss.empty:
            cols = [c for c in ["RUCPagador","RazonSocialPagador"] if c in miss.columns]
            faltantes = (miss[cols].drop_duplicates().head(10).to_dict("records"))
    return {"total_rows": total, "con_sector": con_sector, "match_pct": ratio, "faltantes_top": faltantes}

#Funcion para agregar columna MontoNetoPEN. [convierte USD a PEN usando tc]
def add_monto_pen(df: pd.DataFrame, tc: float) -> pd.DataFrame:
    """Agrega MontoNetoPEN a partir de NetoConfirmado y Moneda."""
    if df.empty: return df
    d = df.copy()
    d["NetoConfirmado"] = pd.to_numeric(d["NetoConfirmado"], errors="coerce")
    d["Moneda"] = d["Moneda"].astype(str)
    factor = d["Moneda"].map({"PEN": 1.0, "USD": float(tc)}).fillna(1.0)
    d["MontoNetoPEN"] = d["NetoConfirmado"] * factor
    return d

#Variable de mapa de meses en español.
MESES_ES = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
    7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
}

#Funcion para obtener los últimos n meses partir del ultimo dia de la data.
def ventanas_previas(ff: pd.Timestamp) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Devuelve [(label, start, end)] para:
      - Mes actual (desde el 1 hasta ff, aunque el mes no esté completo)
      - Mes anterior (calendario completo)
      - Mes anterior al anterior (calendario completo)
    """
    ff = pd.to_datetime(ff)

    # Mes actual (inicio = 1ro del mes; fin = ff para no incluir días futuros)
    m0_start = ff.replace(day=1).normalize()
    m0_end   = ff.normalize()  # parcial hasta 'ff' (no MonthEnd)

    # Mes -1 (calendario completo)
    m1_end   = m0_start - pd.Timedelta(days=1)
    m1_start = m1_end.replace(day=1).normalize()

    # Mes -2 (calendario completo)
    m2_end   = m1_start - pd.Timedelta(days=1)
    m2_start = m2_end.replace(day=1).normalize()

    ventanas = [
        (f"{MESES_ES[m0_start.month]} {m0_start.year}", m0_start, m0_end),
        (f"{MESES_ES[m1_start.month]} {m1_start.year}", m1_start, m1_end),
        (f"{MESES_ES[m2_start.month]} {m2_start.year}", m2_start, m2_end),
    ]
    return ventanas

#Funcion para obtener los últimos n meses calendario completos a partir de un dataframe con columna de fechas.
def recurrentes_3p(
    df_base: pd.DataFrame,
    ventanas: List[Tuple[str, pd.Timestamp, pd.Timestamp]],
    modo: str = "Factoring_cedente",
    tc: float = 3.75,
    monedas: Optional[List[str]] = None,
    ejecutivos: Optional[List[str]] = None,
) -> pd.DataFrame:
    if len(ventanas) < 3:
        return pd.DataFrame()

    ventanas = list(reversed(ventanas[-3:]))    
    
    # Filtros GLOBALES (independientes de fechas)
    d = df_base.copy()
    if monedas:
        d = d[d["Moneda"].isin(monedas)]
    if ejecutivos:
        d = d[d["Ejecutivo"].isin(ejecutivos)]

    # Modo / columnas id
    if  modo == "Factoring_cedente":
        filtro_tipo = "Factoring";  id_cols = ["RUCCliente", "RazonSocialCliente"]
        
    elif modo == "Factoring_pagador":
        filtro_tipo = "Factoring";  id_cols = ["RUCPagador", "RazonSocialPagador"]
        
    else:
        filtro_tipo = "Confirming"; id_cols = ["RUCPagador", "RazonSocialPagador"]

    labels = [lbl for (lbl, _, _) in ventanas]
    frames = []
    for label, fi_k, ff_k in ventanas:
        d_k = d[
            (d["TipoOperacion"] == filtro_tipo) &
            (d["FechaOperacion"] >= fi_k) &
            (d["FechaOperacion"] <= ff_k)
        ].copy()

        if d_k.empty:
            frames.append(pd.DataFrame(columns=id_cols + [label]))
            continue

        # Conversión según TC actual
        d_k = add_monto_pen(d_k, tc)

        g = (d_k.groupby(id_cols, as_index=False)["MontoNetoPEN"]
                .sum()
                .rename(columns={"MontoNetoPEN": label}))
        frames.append(g)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=id_cols, how="inner")  # presentes en TODOS

    if out.empty:
        return out

    # Tendencia con los ÚLTIMOS 3 labels (sean nombres de mes o P-3/P-2/P-1)
    last3 = labels[-3:]
    def _trend(r):
        v = [r.get(col, np.nan) for col in last3]
        if any(pd.isna(v)):
            return "→"
        return "↗" if (v[0] < v[1] < v[2]) else ("↘" if (v[0] > v[1] > v[2]) else "→")

    out["Tendencia"] = out.apply(_trend, axis=1)

    # Formato a todas las columnas de periodo
    for col in labels:
        if col in out.columns:
            out[col] = out[col].map(fmt_money)

    return out


def detectar_inactivas_n_meses(
    df: pd.DataFrame,
    ventanas: List[Tuple[str, pd.Timestamp, pd.Timestamp]],
    modo: str = "Factoring_cedente",
    n: int = 3,
    tc: float = 3.75,
) -> pd.DataFrame:
    """
    Devuelve entidades que NO colocan en los últimos n meses (todos en 0),
    pero sí tuvieron actividad histórica antes del primer mes de la ventana.
    Columnas de salida:
      - id entidad
      - InactivoUltimosN (True)
      - TuvoActividadHistorica (True/False)
      - FechaUltHistorica, MontoUltHistoricaPEN (si existe)
    """
    if len(ventanas) < n:
        return pd.DataFrame()

    if modo == "Factoring_cedente":
        filtro_tipo = "Factoring";  id_cols = ["RUCCliente", "RazonSocialCliente"]
    elif modo == "Factoring_pagador":
        filtro_tipo = "Factoring";  id_cols = ["RUCPagador", "RazonSocialPagador"]
    else:
        filtro_tipo = "Confirming"; id_cols = ["RUCPagador", "RazonSocialPagador"]

    # Últimos n meses
    labels = [lbl for (lbl, _, _) in ventanas][-n:]
    rangos = ventanas[-n:]
    d = df.copy()
    d = d[d["TipoOperacion"] == filtro_tipo]
    d = add_monto_pen(d, tc)

    # Suma por mes
    frames = []
    for label, fi_k, ff_k in rangos:
        dd = d[(d["FechaOperacion"] >= fi_k) & (d["FechaOperacion"] <= ff_k)].copy()
        g = (dd.groupby(id_cols, as_index=False)["MontoNetoPEN"]
               .sum()
               .rename(columns={"MontoNetoPEN": f"{label}_num"}))
        frames.append(g)

    # Outer para no perder entidades
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=id_cols, how="outer")
    out.fillna(0.0, inplace=True)

    # Inactividad en todos los últimos n meses
    mes_cols = [f"{lbl}_num" for lbl in labels]
    out["InactivoUltimosN"] = (out[mes_cols].sum(axis=1) == 0)

    if out.empty:
        return out

    # Actividad histórica (antes del inicio del primer mes evaluado)
    fi_primero = rangos[0][1]  # fecha inicio del primer mes considerado
    hist = d[d["FechaOperacion"] < fi_primero].copy()
    if hist.empty:
        out["TuvoActividadHistorica"] = False
        out["FechaUltHistorica"] = pd.NaT
        out["MontoUltHistoricaPEN"] = np.nan
        return out[out["InactivoUltimosN"]].reset_index(drop=True)

    hist = hist.sort_values("FechaOperacion")
    idx = hist.groupby(id_cols)["FechaOperacion"].idxmax()
    ult_hist = (hist.loc[idx, id_cols + ["FechaOperacion", "MontoNetoPEN"]]
                    .rename(columns={"FechaOperacion": "FechaUltHistorica",
                                     "MontoNetoPEN": "MontoUltHistoricaPEN"}))

    out = out.merge(ult_hist, on=id_cols, how="left")
    out["TuvoActividadHistorica"] = out["FechaUltHistorica"].notna()

    # Devolver solo los inactivos en últimos n meses
    return out[out["InactivoUltimosN"]].reset_index(drop=True)

def detectar_2_meses_consecutivos_negativos(
    df: pd.DataFrame,
    ventanas: List[Tuple[str, pd.Timestamp, pd.Timestamp]],
    modo: str = "Factoring_cedente",
    tc: float = 3.75,
    adjuntar_ultimo_detalle: bool = True,
) -> pd.DataFrame:
    """
    Devuelve entidades que colocaron en los ÚLTIMOS 2 MESES consecutivos (P-1 y P) pero
    con tendencia negativa en el último tramo (Mes_P < Mes_P-1).
    Columnas:
      - id entidad
      - Mes_P-1_num, Mes_P_num
      - GapRetomaPEN (P - P-1)
      - RatioRetoma (P / P-1)
      - (opcionales) FechaUltOp, MontoUltOpPEN, EjecutivoUlt
    """
    if len(ventanas) < 2:
        return pd.DataFrame()

    if modo == "Factoring_cedente":
        filtro_tipo = "Factoring";  id_cols = ["RUCCliente", "RazonSocialCliente"]
    elif modo == "Factoring_pagador":
        filtro_tipo = "Factoring";  id_cols = ["RUCPagador", "RazonSocialPagador"]
    else:
        filtro_tipo = "Confirming"; id_cols = ["RUCPagador", "RazonSocialPagador"]

    # Tomar últimos 2 meses
    labels = [lbl for (lbl, _, _) in ventanas]
    L1_label, L0_label = labels[-2], labels[-1]  # P-1 , P
    (_, fi_L1, ff_L1) = ventanas[-2]
    (_, fi_L0, ff_L0) = ventanas[-1]

    d = df.copy()
    d = d[d["TipoOperacion"] == filtro_tipo]
    d = add_monto_pen(d, tc)

    def _agg(fi, ff, alias):
        dd = d[(d["FechaOperacion"] >= fi) & (d["FechaOperacion"] <= ff)]
        return (dd.groupby(id_cols, as_index=False)["MontoNetoPEN"]
                  .sum()
                  .rename(columns={"MontoNetoPEN": f"{alias}_num"}))

    g_L1 = _agg(fi_L1, ff_L1, L1_label)
    g_L0 = _agg(fi_L0, ff_L0, L0_label)

    out = g_L1.merge(g_L0, on=id_cols, how="inner")  # deben haber colocado en ambos meses
    if out.empty:
        return out

    # Tendencia negativa en el último tramo (P < P-1)
    out = out[out[f"{L0_label}_num"] < out[f"{L1_label}_num"]].copy()
    if out.empty:
        return out

    out["GapRetomaPEN"] = out[f"{L0_label}_num"] - out[f"{L1_label}_num"]
    out["RatioRetoma"]  = np.where(out[f"{L1_label}_num"] > 0,
                                   out[f"{L0_label}_num"] / out[f"{L1_label}_num"],
                                   np.nan)

    if adjuntar_ultimo_detalle:
        # Reutiliza tu helper de últimos datos por entidad (si ya lo tienes definido)
        ult = ultimo_detalle_por_entidad(df, modo=modo, tc=tc)
        out = out.merge(ult, on=id_cols, how="left")

    # Orden sugerido
    cols = id_cols + [f"{L1_label}_num", f"{L0_label}_num", "GapRetomaPEN", "RatioRetoma"]
    if adjuntar_ultimo_detalle:
        cols += ["FechaUltOp", "MontoUltOpPEN", "EjecutivoUlt"]
    return out[cols].sort_values(by=["GapRetomaPEN"]).reset_index(drop=True)

def ultimo_detalle_por_entidad(
    df: pd.DataFrame,
    modo: str = "Factoring_cedente",
    tc: float = 3.75
) -> pd.DataFrame:
    """
    Devuelve por entidad (cedente/pagador) el último registro:
      - FechaUltOp
      - MontoUltOpPEN
      - EjecutivoUlt
    """
    if modo == "Factoring_cedente":
        id_cols = ["RUCCliente", "RazonSocialCliente"]
    elif modo == "Factoring_pagador":
        id_cols = ["RUCPagador", "RazonSocialPagador"]
    else:  # Confirming_pagador
        id_cols = ["RUCPagador", "RazonSocialPagador"]

    d = df.copy()
    d = add_monto_pen(d, tc)
    d = d.sort_values("FechaOperacion")  # asegura idxmax por último

    # Índice del último registro por entidad
    idx = d.groupby(id_cols)["FechaOperacion"].idxmax()

    ult = (d.loc[idx, id_cols + ["FechaOperacion", "MontoNetoPEN", "Ejecutivo"]]
             .rename(columns={
                 "FechaOperacion": "FechaUltOp",
                 "MontoNetoPEN": "MontoUltOpPEN",
                 "Ejecutivo": "EjecutivoUlt"
             })
             .reset_index(drop=True))
    return ult


def resumir_por_operacion(df_f: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidación a nivel 'operación' (CodigoLiquidacion):
      - MontoOperacion = sum(NetoConfirmado)
      - TasaPondOp     = sum(tasa * Neto) / sum(Neto)
      - FechaOperacionRep = min(FechaOperacion)
    """
    if df_f.empty or "CodigoLiquidacion" not in df_f.columns:
        return df_f.iloc[0:0]

    base = df_f.copy()
    base["NetoConfirmado"] = pd.to_numeric(base["NetoConfirmado"], errors="coerce")
    base["TasaNominalMensualPorc"] = pd.to_numeric(base["TasaNominalMensualPorc"], errors="coerce")

    grp = base.groupby(["CodigoLiquidacion", "Ejecutivo", "Moneda", "TipoOperacion"], as_index=False).agg(
        MontoOperacion=("NetoConfirmado", "sum"),
        FechaOperacionRep=("FechaOperacion", "min")
    )

    # Tasa ponderada por operación
    # Usamos groupby-apply para cada CodigoLiquidacion
    tasas = (
        base.groupby("CodigoLiquidacion")["TasaNominalMensualPorc"]
            .agg(TasaPondOp=lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan,
                _nunique="nunique")     # validación opcional
            .reset_index()
    )

    tasas = tasas.drop(columns="_nunique")
    
    grp = grp.merge(tasas, on="CodigoLiquidacion", how="left")
    return grp

def kpis_ejecutivo(df_ops: pd.DataFrame, tc: float) -> dict:
    """
    KPIs desde nivel operación:
      - Monto por moneda
      - Monto total PEN
      - # Operaciones
      - Ticket promedio PEN (promedio de monto operación convertido)
      - Ticket por moneda (promedio por Moneda)
      - Tasa ponderada global (peso: monto de la operación)
    """
    if df_ops.empty:
        return dict(monto_por_moneda={}, monto_total_pen=0.0, ops=0, ticket_pen=0.0, ticket_por_moneda={}, tasa_pond=np.nan)

    m_pm = df_ops.groupby("Moneda", as_index=False)["MontoOperacion"].sum().set_index("Moneda")["MontoOperacion"].to_dict()

    factor = df_ops["Moneda"].map({"PEN": 1.0, "USD": float(tc)}).fillna(1.0)
    monto_pen = float((df_ops["MontoOperacion"] * factor).sum())
    w = df_ops["MontoOperacion"] * factor
    
    n_ops = df_ops["CodigoLiquidacion"].nunique()
    ticket_pen = float((df_ops["MontoOperacion"] * factor).mean() if n_ops else 0.0)

    ticket_pm = df_ops.groupby("Moneda")["MontoOperacion"].mean().to_dict()

    
    tasa_pond = ((df_ops["TasaPondOp"] * w).sum() / w.sum() ) if w.sum() else np.nan

    return dict(
        monto_por_moneda=m_pm,
        monto_total_pen=monto_pen,
        ops=int(n_ops),
        ticket_pen=ticket_pen,
        ticket_por_moneda={k: float(v) for k, v in ticket_pm.items()},
        tasa_pond=tasa_pond
    )



def tabla_top(df_f: pd.DataFrame, by_cols: List[str], label_cols: List[str], tc: float, N: int = 10) -> pd.DataFrame:
    """Top N por suma de NetoConfirmado convertido a PEN."""
    if df_f.empty: return df_f.iloc[0:0]
    d = add_monto_pen(df_f, tc)
    g = (d.groupby(by_cols, as_index=False)
           .agg(MontoPEN=("MontoNetoPEN","sum"), Ops=("CodigoLiquidacion","nunique"))
           .sort_values("MontoPEN", ascending=False)
           .head(N))
    ver = g.copy()
    ver["MontoPEN"] = ver["MontoPEN"].map(fmt_money)
    return ver[label_cols + ["MontoPEN","Ops"]]


# Resumen de negativos
def resumen_neg(df_tab: pd.DataFrame, titulo: str, max_items: int = 50) -> str:
    """
    Lista entidades en tendencia ↘ e incluye:
      - Última operación (fecha y monto)
      - Último ejecutivo a cargo
      - Gap de retoma (si está disponible)
    """
    if df_tab is None or df_tab.empty or "Tendencia" not in df_tab.columns:
        return f"<p><strong>{titulo}</strong>: no se encontraron con tendencia negativa.</p>"

    neg = df_tab[df_tab["Tendencia"] == "↘"].copy()
    if neg.empty:
        return f"<p><strong>{titulo}</strong>: no se encontraron con tendencia negativa.</p>"

    # Columnas opcionales
    gap_col = "GapRetomaPEN" if "GapRetomaPEN" in neg.columns else None
    fecha_col = "FechaUltOp"  # provendrá del merge previo en la UI
    monto_col = "MontoUltOpPEN"
    eje_col   = "EjecutivoUlt"

    items = []
    for _, r in neg.head(max_items).iterrows():
        if {"RazonSocialCliente","RUCCliente"}.issubset(neg.columns):
            nombre = f"{r['RazonSocialCliente']} ({r['RUCCliente']})"
        elif {"RazonSocialPagador","RUCPagador"}.issubset(neg.columns):
            nombre = f"{r['RazonSocialPagador']} ({r['RUCPagador']})"
        else:
            nombre = "Entidad"

        fecha_txt = r.get(fecha_col, pd.NaT)
        fecha_txt = "" if pd.isna(fecha_txt) else pd.to_datetime(fecha_txt).date().isoformat()
        monto_txt = r.get(monto_col, np.nan)
        monto_txt = "-" if pd.isna(monto_txt) else fmt_money(monto_txt)
        eje_txt   = r.get(eje_col, "") or "-"

        gap_txt = ""
        if gap_col:
            g = r.get(gap_col, np.nan)
            if not pd.isna(g):
                gap_txt = f" | Gap retoma: {fmt_money(g)} PEN"

        items.append(f"<li>{nombre} — Última op: {fecha_txt} por {monto_txt} PEN — Ejecutivo: {eje_txt}{gap_txt}</li>")

    return f"<p><strong>{titulo} con tendencia negativa ({len(neg)}):</strong></p><ul>{''.join(items)}</ul>"

def semanal_coloc_vs_cobros(df: pd.DataFrame, tc: float) -> pd.DataFrame:
    """
    Suma por Ejecutivo y Semana ISO:
      - ColocoPEN: sum(MontoNetoPEN) de colocaciones
      - CobroPEN : sum(MontoNetoPEN) de cobros (TipoPago no nulo)
      - GapPEN   : ColocoPEN - CobroPEN  (si <0, cobró más de lo que colocó)
      - CumpleRetoma: ColocoPEN >= CobroPEN
    """
    if df.empty:
        return df.iloc[0:0]

    d = add_monto_pen(df, tc)
    d["SemanaISO"] = d["FechaOperacion"].dt.isocalendar().week.astype(int)
    d["AnioISO"]   = d["FechaOperacion"].dt.isocalendar().year.astype(int)

    # Flags
    d["es_coloc"] = d["TipoOperacion"].isin(["Factoring", "Confirming"])
    d["es_cobro"] = d["TipoPago"].notna() & d["TipoPago"].astype(str).str.strip().ne("")

    # Agregados
    g_col = (d[d["es_coloc"]]
             .groupby(["AnioISO","SemanaISO","Ejecutivo"], as_index=False)["MontoNetoPEN"].sum()
             .rename(columns={"MontoNetoPEN": "ColocoPEN"}))

    g_cob = (d[d["es_cobro"]]
             .groupby(["AnioISO","SemanaISO","Ejecutivo"], as_index=False)["MontoNetoPEN"].sum()
             .rename(columns={"MontoNetoPEN": "CobroPEN"}))

    out = g_col.merge(g_cob, on=["AnioISO","SemanaISO","Ejecutivo"], how="outer").fillna(0.0)
    out["GapPEN"] = out["ColocoPEN"] - out["CobroPEN"]
    out["CumpleRetoma"] = out["ColocoPEN"] >= out["CobroPEN"]

    # Orden y formatos amigables (para UI puedes re-formatear)
    out = out.sort_values(["AnioISO","SemanaISO","Ejecutivo"]).reset_index(drop=True)
    return out

######################

def _id_cols_by_modo(modo: str):
    if modo == "Factoring_cedente":
        return "Factoring", ["RUCCliente", "RazonSocialCliente"]
    elif modo == "Factoring_pagador":
        return "Factoring", ["RUCPagador", "RazonSocialPagador"]
    else:  # Confirming_pagador
        return "Confirming", ["RUCPagador", "RazonSocialPagador"]
    
def _ultimo_detalle(df: pd.DataFrame, modo: str, tc: float) -> pd.DataFrame:
    tipo, id_cols = _id_cols_by_modo(modo)
    d = add_monto_pen(df, tc)
    d = d[d["TipoOperacion"].isin(["Factoring","Confirming"])]
    d = d.sort_values("FechaOperacion")
    idx = d.groupby(id_cols)["FechaOperacion"].idxmax()
    return (d.loc[idx, id_cols + ["FechaOperacion","MontoNetoPEN","Ejecutivo"]]
              .rename(columns={"FechaOperacion":"FechaUltOp",
                               "MontoNetoPEN":"MontoUltOpPEN",
                               "Ejecutivo":"EjecutivoUlt"}))
    
def _montos_3m(df: pd.DataFrame,
               ventanas: List[Tuple[str,pd.Timestamp,pd.Timestamp]],
               modo: str,
               tc: float) -> pd.DataFrame:
    """Outer-merge para tener TODOS los que operaron en al menos 1 mes (0 cuando no)."""
    tipo, id_cols = _id_cols_by_modo(modo)
    # asegurar orden cronológico: P-2, P-1, P
    ventanas3 = list(reversed(ventanas[-3:]))
    frames = []
    d = add_monto_pen(df, tc)
    for label, fi_k, ff_k in ventanas3:
        dk = d[(d["TipoOperacion"] == tipo) &
               (d["FechaOperacion"] >= fi_k) &
               (d["FechaOperacion"] <= ff_k)].copy()
        if dk.empty:
            frames.append(pd.DataFrame(columns=id_cols + [f"{label}_num"]))
            continue
        g = (dk.groupby(id_cols, as_index=False)["MontoNetoPEN"]
                .sum()
                .rename(columns={"MontoNetoPEN": f"{label}_num"}))
        frames.append(g)
    out = None
    for f in frames:
        out = f if out is None else out.merge(f, on=id_cols, how="outer")
    out = out.fillna(0.0)
    return out, [f"{v[0]}_num" for v in ventanas3], [v[0] for v in ventanas3]

def _tendencia_3(v0, v1, v2) -> str:
    if any(pd.isna(x) for x in (v0,v1,v2)):
        return "→"
    return "↗" if (v0 < v1 < v2) else ("↘" if (v0 > v1 > v2) else "→")

def clasificar_situacion_3m(df: pd.DataFrame,
                            ventanas: List[Tuple[str,pd.Timestamp,pd.Timestamp]],
                            modo: str,
                            tc: float) -> pd.DataFrame:
    """
    Devuelve una tabla con:
      - flags por mes (P-2, P-1, P) 1/0 si colocó (>0)
      - Situacion (Activo 3/3, Dejó de colocar en P, Activo 2 últimos, Intermitente, Solo 1/3)
      - Tendencia para los que tienen 3/3
      - Último ejecutivo, última fecha y monto
    """
    montos, num_cols, labels = _montos_3m(df, ventanas, modo, tc)  # num_cols y labels en orden P-2,P-1,P
    tipo, id_cols = _id_cols_by_modo(modo)

    Pm2, Pm1, P = num_cols  # columnas numéricas
    Lm2, Lm1, L = labels    # nombres "bonitos"

    res = montos.copy()

    # ---- Flags de actividad (vectorizado)
    a2 = (res[Pm2] > 0).astype(int)
    a1 = (res[Pm1] > 0).astype(int)
    a0 = (res[P]   > 0).astype(int)

    res[f"{Lm2}_act"] = a2
    res[f"{Lm1}_act"] = a1
    res[f"{L}_act"]   = a0

    s = a2 + a1 + a0
    res["SumaMeses"] = s

    # ---- Situación (vectorizado con np.select)
    cond_activo_33     = (s == 3)
    cond_dejo_en_P     = (s == 2) & (a2.eq(1) & a1.eq(1) & a0.eq(0))
    cond_act_ult_2     = (s == 2) & (a2.eq(0) & a1.eq(1) & a0.eq(1))
    cond_intermitente  = (s == 2) & (a2.eq(1) & a1.eq(0) & a0.eq(1))
    cond_solo_P        = (s == 1) & (a0.eq(1))
    cond_solo_P1       = (s == 1) & (a1.eq(1))
    cond_solo_P2       = (s == 1) & (a2.eq(1))
    cond_sin_act       = (s == 0)

    res["Situacion"] = np.select(
        [
            cond_activo_33, cond_dejo_en_P, cond_act_ult_2, cond_intermitente,
            cond_solo_P, cond_solo_P1, cond_solo_P2, cond_sin_act
        ],
        [
            "Activo tres meses", f"Dejó de colocar en {L}", "Activo últimos 2 meses", "Intermitente",
            f"Activo solo en {L}", f"Activo solo en {Lm1}", f"Activo solo en {Lm2}", "Sin actividad 3M"
        ],
        default="Sin actividad 3M"
    )

    # ---- Tendencia solo para 3/3 (vectorizado)
    # ↗ si P-2 < P-1 < P; ↘ si P-2 > P-1 > P; → en otro caso
    tendencia_neutral = np.full(len(res), "—", dtype=object)
    crec = (res[Pm2] < res[Pm1]) & (res[Pm1] < res[P])
    decr = (res[Pm2] > res[Pm1]) & (res[Pm1] > res[P])
    tendencia_33 = np.where(crec, "↗", np.where(decr, "↘", "→"))
    res["Tendencia"] = np.where(cond_activo_33, tendencia_33, tendencia_neutral)

    # ---- Último detalle (merge por id)
    ult = _ultimo_detalle(df, modo, tc)
    res = res.merge(ult, on=id_cols, how="left")

    # ---- Columnas bonitas de montos
    res[Lm2] = res[Pm2].map(fmt_money)
    res[Lm1] = res[Pm1].map(fmt_money)
    res[L]   = res[P].map(fmt_money)

    # ---- Orden sugerido (devuelve solo las columnas existentes)
    cols = (id_cols +
            [Lm2, Lm1, L, "Tendencia", "Situacion", "FechaUltOp", "EjecutivoUlt", "MontoUltOpPEN",
             Pm2, Pm1, P, f"{Lm2}_act", f"{Lm1}_act", f"{L}_act", "SumaMeses"])
    cols = [c for c in cols if c in res.columns]
    return res.loc[:, cols]


def kpis_retoma(df_sit: pd.DataFrame, labels: List[str]) -> Dict[str, float]:
    """
    KPIs agregados a partir de la tabla de situaciones 'df_sit' generada por
    la nueva 'clasificar_situacion_3m'.

    labels: [Lm2, Lm1, L] en orden cronológico (P-2, P-1, P)
    Situacion posibles (texto exacto):
      - "Activo tres meses"
      - f"Dejó de colocar en {L}"
      - "Activo últimos 2 meses"
      - "Intermitente"
      - f"Activo solo en {L}"
      - f"Activo solo en {Lm1}"
      - f"Activo solo en {Lm2}"
      - "Sin actividad 3M"
    """
    Lm2, Lm1, L = labels  # P-2, P-1, P
    # Base: entidades con actividad en algún mes
    total_con_algo = int((df_sit.get("SumaMeses", 0) > 0).sum())

    # Conteos por categoría (coincidencia exacta con las nuevas etiquetas)
    activo_33       = int((df_sit["Situacion"] == "Activo tres meses").sum())
    dejo_en_P       = int((df_sit["Situacion"] == f"Dejó de colocar en {L}").sum())
    act_ultimos_2   = int((df_sit["Situacion"] == "Activo últimos 2 meses").sum())
    intermitentes   = int((df_sit["Situacion"] == "Intermitente").sum())
    solo_P          = int((df_sit["Situacion"] == f"Activo solo en {L}").sum())
    solo_P1         = int((df_sit["Situacion"] == f"Activo solo en {Lm1}").sum())
    solo_P2         = int((df_sit["Situacion"] == f"Activo solo en {Lm2}").sum())
    sin_act         = int((df_sit["Situacion"] == "Sin actividad 3M").sum())

    # Porcentajes (sobre total_con_algo si corresponde)
    pct_activo_33 = (activo_33 / total_con_algo) if total_con_algo else 0.0
    pct_dejo_en_P = (dejo_en_P / total_con_algo) if total_con_algo else 0.0
    pct_act_ult2  = (act_ultimos_2 / total_con_algo) if total_con_algo else 0.0
    pct_intermit  = (intermitentes / total_con_algo) if total_con_algo else 0.0
    pct_solo_1_de_3 = ((solo_P + solo_P1 + solo_P2) / total_con_algo) if total_con_algo else 0.0

    return {
        "total_con_algo": total_con_algo,
        "activo_tres_meses": activo_33,
        "pct_activo_tres_meses": pct_activo_33,
        "dejo_en_P": dejo_en_P,
        "pct_dejo_en_P": pct_dejo_en_P,
        "activo_ultimos_2": act_ultimos_2,
        "pct_activo_ultimos_2": pct_act_ult2,
        "intermitentes": intermitentes,
        "pct_intermitentes": pct_intermit,
        "solo_P": solo_P,
        "solo_P1": solo_P1,
        "solo_P2": solo_P2,
        "solo_1_de_3": (solo_P + solo_P1 + solo_P2),
        "pct_solo_1_de_3": pct_solo_1_de_3,
        "sin_actividad_3m": sin_act,
    }


def coloc_vs_cobros_7d_dual(df: pd.DataFrame, tc: float) -> pd.DataFrame:
    """
    Analiza SOLO los últimos 7 días respecto a la fecha máxima de la data,
    separando correctamente las fuentes de fecha:
      - Colocación   -> FechaOperacion
      - Cobranza     -> FechaConfirmado

    Devuelve un resumen por Ejecutivo con:
      ColocoPEN, CobroPEN, GapPEN, CumpleRetoma, CumplimientoPct
      + columnas formateadas: Coloco, Cobro, GapFmt, PctFmt, Estado, Icon

    NOTA: No filtra globalmente el DF. Calcula d_coloc (por FechaOperacion)
          y d_cobro (por FechaConfirmado) por separado para evitar sesgos.
    """
    if df.empty:
        return df.iloc[0:0]

    # Asegura monto en PEN
    d = add_monto_pen(df, tc).copy()

    # Asegura datetime (por si acaso vienen como string)
    for col in ["FechaOperacion", "FechaConfirmado"]:
        if col in d.columns:
            d[col] = pd.to_datetime(d[col], errors="coerce")

    # Fecha de referencia = max entre FechaOperacion y FechaConfirmado
    fechas_max = []
    if "FechaOperacion" in d.columns:
        fechas_max.append(d["FechaOperacion"].max())
    if "FechaConfirmado" in d.columns:
        fechas_max.append(d["FechaConfirmado"].max())
    fechas_max = [x for x in fechas_max if pd.notna(x)]
    if not fechas_max:
        return df.iloc[0:0]

    max_date = max(fechas_max)
    cutoff   = max_date - pd.Timedelta(days=7)

    # --- Colocaciones (filtra por FechaOperacion, sin tocar cobranzas) ---
    coloc_mask = (
        d["TipoOperacion"].isin(["Factoring", "Confirming"]) &
        d["FechaOperacion"].between(cutoff, max_date, inclusive="both")
    )
    d_coloc = d.loc[coloc_mask, ["Ejecutivo", "MontoNetoPEN"]].copy()
    g_col = (d_coloc.groupby("Ejecutivo", as_index=False)["MontoNetoPEN"]
                   .sum()
                   .rename(columns={"MontoNetoPEN": "ColocoPEN"}))

    # --- Cobranzas (filtra por FechaConfirmado, sin tocar colocaciones) ---
    cobro_mask = (
        d["FechaConfirmado"].notna() &
        d["FechaConfirmado"].between(cutoff, max_date, inclusive="both")
    )
    d_cobro = d.loc[cobro_mask, ["Ejecutivo", "MontoNetoPEN"]].copy()
    g_cob = (d_cobro.groupby("Ejecutivo", as_index=False)["MontoNetoPEN"]
                    .sum()
                    .rename(columns={"MontoNetoPEN": "CobroPEN"}))

    # --- Merge y métricas ---
    out = g_col.merge(g_cob, on="Ejecutivo", how="outer").fillna(0.0)
    if out.empty:
        return out

    out["ColocoPEN"] = out["ColocoPEN"].astype(float)
    out["CobroPEN"]  = out["CobroPEN"].astype(float)
    out["GapPEN"]    = out["ColocoPEN"] - out["CobroPEN"]
    out["CumpleRetoma"]    = out["ColocoPEN"] >= out["CobroPEN"]
    out["CumplimientoPct"] = np.where(out["CobroPEN"] > 0,
                                      out["ColocoPEN"] / out["CobroPEN"],
                                      np.nan)

    # --- Formatos visuales rápidos ---
    out["Estado"] = np.where(out["CumpleRetoma"], "OK", "FALTA")
    out["Icon"]   = np.where(out["CumpleRetoma"], "✅", "⚠️")
    out["Coloco"] = out["ColocoPEN"].map(fmt_money)
    out["Cobro"]  = out["CobroPEN"].map(fmt_money)
    out["GapFmt"] = out["GapPEN"].map(fmt_money)
    out["PctFmt"] = out["CumplimientoPct"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")

    # Ordena mostrando primero los que NO cumplen (para accionar)
    out = out.sort_values(by=["CumpleRetoma", "GapPEN"], ascending=[True, True]).reset_index(drop=True)
    return out

def diario_coloc_cobro(
    df: pd.DataFrame,
    tc: float,
    fecha_sel: pd.Timestamp,
    por: str = "auto",
):
    """
    Dos tablas para una fecha única (en SOLES):
      A) Por empresa (+ EjecutivoUltDia): ColocoPEN / CobroPEN / GapPEN
      B) Por ejecutivo: ColocoPEN / CobroPEN / GapPEN / CumplimientoPct

    Reglas:
      - Colocación: FechaOperacion (Factoring/Confirming)
      - Cobranza  : FechaConfirmado (TipoPago no vacío)
      - Montos en PEN: MontoNetoPEN (desde NetoConfirmado x TC)
    """
    if df.empty:
        return df.iloc[0:0], df.iloc[0:0]

    d = add_monto_pen(df, tc).copy()

    # Asegura datetime y fecha seleccionada
    for c in ["FechaOperacion", "FechaConfirmado"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    fsel = pd.to_datetime(fecha_sel).date()

    # Máscaras
    es_coloc = (
        d["TipoOperacion"].isin(["Factoring", "Confirming"])
        & (d["FechaOperacion"].dt.date == fsel)
    )
    es_cobro = (
        d["TipoPago"].astype(str).str.strip().ne("")
        & d["FechaConfirmado"].notna()
        & (d["FechaConfirmado"].dt.date == fsel)
    )

    # Columnas de empresa
    if por == "cliente" or (por == "auto" and "RUCCliente" in d.columns):
        id_cols = ["RUCCliente", "RazonSocialCliente"]
    else:
        id_cols = ["RUCPagador", "RazonSocialPagador"]

    # ---------- A) Por empresa ----------
    g_col_emp = (
        d.loc[es_coloc, id_cols + ["MontoNetoPEN"]]
          .groupby(id_cols, as_index=False)["MontoNetoPEN"].sum()
          .rename(columns={"MontoNetoPEN": "ColocoPEN"})
    )
    g_cob_emp = (
        d.loc[es_cobro, id_cols + ["MontoNetoPEN"]]
          .groupby(id_cols, as_index=False)["MontoNetoPEN"].sum()
          .rename(columns={"MontoNetoPEN": "CobroPEN"})
    )
    emp = g_col_emp.merge(g_cob_emp, on=id_cols, how="outer").fillna(0.0)
    emp["GapPEN"] = emp["ColocoPEN"] - emp["CobroPEN"]

    # Ejecutivo del día por empresa (último registro del día usando FechaRef)
    d_dia = d.loc[es_coloc | es_cobro, id_cols + ["Ejecutivo", "FechaOperacion", "FechaConfirmado"]].copy()
    d_dia["FechaRef"] = d_dia["FechaOperacion"].combine_first(d_dia["FechaConfirmado"])
    ultimos = (
        d_dia.sort_values("FechaRef")
             .drop_duplicates(subset=id_cols, keep="last")
             .rename(columns={"Ejecutivo": "EjecutivoUltDia"})
             .drop(columns=["FechaOperacion", "FechaConfirmado", "FechaRef"])
    )
    emp = emp.merge(ultimos, on=id_cols, how="left")

    # Vista amigable
    emp_view = emp.copy()
    emp_view["Coloco"] = emp_view["ColocoPEN"].map(fmt_money)
    emp_view["Cobro"]  = emp_view["CobroPEN"].map(fmt_money)
    emp_view["Gap"]    = emp_view["GapPEN"].map(fmt_money)

    # Orden: Ejecutivo, empresa, montos bonitos + numéricos al final
    cols_emp = id_cols.copy()
    if "EjecutivoUltDia" in emp_view.columns:
        cols_emp = ["EjecutivoUltDia"] + cols_emp
    cols_emp = cols_emp + ["Coloco", "Cobro", "Gap", "ColocoPEN", "CobroPEN", "GapPEN"]

    emp_view = (
        emp_view.loc[:, [c for c in cols_emp if c in emp_view.columns]]
                .sort_values(["GapPEN", "CobroPEN"], ascending=[True, False])
                .reset_index(drop=True)
    )

    # ---------- B) Por ejecutivo ----------
    g_col_eje = (
        d.loc[es_coloc, ["Ejecutivo", "MontoNetoPEN"]]
          .groupby("Ejecutivo", as_index=False)["MontoNetoPEN"].sum()
          .rename(columns={"MontoNetoPEN": "ColocoPEN"})
    )
    g_cob_eje = (
        d.loc[es_cobro, ["Ejecutivo", "MontoNetoPEN"]]
          .groupby("Ejecutivo", as_index=False)["MontoNetoPEN"].sum()
          .rename(columns={"MontoNetoPEN": "CobroPEN"})
    )
    eje = g_col_eje.merge(g_cob_eje, on="Ejecutivo", how="outer").fillna(0.0)
    eje["GapPEN"] = eje["ColocoPEN"] - eje["CobroPEN"]
    eje["CumplimientoPct"] = np.where(eje["CobroPEN"] > 0, eje["ColocoPEN"] / eje["CobroPEN"], np.nan)

    eje_view = eje.copy()
    eje_view["Coloco"] = eje_view["ColocoPEN"].map(fmt_money)
    eje_view["Cobro"]  = eje_view["CobroPEN"].map(fmt_money)
    eje_view["Gap"]    = eje_view["GapPEN"].map(fmt_money)
    eje_view["Pct"]    = eje_view["CumplimientoPct"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")

    eje_view = (
        eje_view.loc[:, ["Ejecutivo", "Coloco", "Cobro", "Gap", "Pct",
                         "ColocoPEN", "CobroPEN", "GapPEN", "CumplimientoPct"]]
                .sort_values("Ejecutivo")
                .reset_index(drop=True)
    )

    return emp_view, eje_view


def rango_coloc_cobro(df: pd.DataFrame, tc: float, fi, ff) -> pd.DataFrame:
    """
    Resumen diario por rango [fi, ff] (inclusive), en SOLES:
      Fecha | Cobro | Coloco | Gap  (Gap = Coloco - Cobro)

    - Coloco: suma por FechaOperacion (Factoring/Confirming)
    - Cobro : suma por FechaConfirmado (TipoPago no vacío)
    """
    if df.empty:
        return pd.DataFrame(columns=["Fecha","Cobro","Coloco","Gap"])

    d = add_monto_pen(df, tc).copy()

    # asegurar datetimes
    d["FechaOperacion"]  = pd.to_datetime(d["FechaOperacion"], errors="coerce")
    d["FechaConfirmado"] = pd.to_datetime(d["FechaConfirmado"], errors="coerce")

    # normalizar límites
    fi = pd.to_datetime(fi).normalize()
    ff = pd.to_datetime(ff).normalize()

    # --- Colocaciones por día (FechaOperacion) ---
    coloc_mask = (
        d["TipoOperacion"].isin(["Factoring","Confirming"]) &
        d["FechaOperacion"].between(fi, ff, inclusive="both")
    )
    g_col = (d.loc[coloc_mask, ["FechaOperacion","MontoNetoPEN"]]
               .assign(Fecha=lambda x: x["FechaOperacion"].dt.normalize())
               .groupby("Fecha", as_index=False)["MontoNetoPEN"].sum()
               .rename(columns={"MontoNetoPEN":"Coloco"}))

    # --- Cobranzas por día (FechaConfirmado) ---
    cobro_mask = (
        d["FechaConfirmado"].between(fi, ff, inclusive="both") &
        d["TipoPago"].astype(str).str.strip().ne("")
    )
    g_cob = (d.loc[cobro_mask, ["FechaConfirmado","MontoNetoPEN"]]
               .assign(Fecha=lambda x: x["FechaConfirmado"].dt.normalize())
               .groupby("Fecha", as_index=False)["MontoNetoPEN"].sum()
               .rename(columns={"MontoNetoPEN":"Cobro"}))

    # --- Unir, completar días faltantes y calcular Gap ---
    out = pd.merge(g_cob, g_col, on="Fecha", how="outer").fillna(0.0)
    # si quieres asegurar todos los días del rango:
    all_days = pd.DataFrame({"Fecha": pd.date_range(fi, ff, freq="D")})
    out = all_days.merge(out, on="Fecha", how="left").fillna(0.0)

    out["Gap"] = out["Coloco"] - out["Cobro"]
    return out.sort_values("Fecha").reset_index(drop=True)