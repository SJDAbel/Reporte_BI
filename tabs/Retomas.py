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

import streamlit as st
import pandas as pd
from core.fun_varios import clasificar_situacion_3m, kpis_retoma, ventanas_previas, diario_coloc_cobro, fmt_money, rango_coloc_cobro

def render_retomas_modo(df: pd.DataFrame, tc_pen_por_usd: float, ff):
    # Selección del modo
    modo_map = {
        "Cedente Factoring": "Factoring_cedente",
        "Pagador Factoring": "Factoring_pagador",
        "Pagador Confirming": "Confirming_pagador"
    }
    modo_sel = st.selectbox("Selecciona la vista", list(modo_map.keys()))
    modo = modo_map[modo_sel]
    
    
    # Construir ventanas (mes actual parcial + 2 previos)
    ventanas = ventanas_previas(ff)

    # Clasificación
    df_sit = clasificar_situacion_3m(df, ventanas, modo=modo, tc=tc_pen_por_usd)
    labels = [v[0] for v in reversed(ventanas[-3:])]
    k = kpis_retoma(df_sit,labels)
    
    # Detectar columnas ID
    if "RUCCliente" in df_sit.columns:
        id_cols = ["RUCCliente", "RazonSocialCliente"]
    else:
        id_cols = ["RUCPagador", "RazonSocialPagador"]

    # --- UI ---
    st.subheader(f"Análisis de retomas: {modo_sel}")
    tab_kpi, tab_resumen, tab_detalle = st.tabs(["ColocaciónVsCobranza", "Resumen", "Detalle"])

    with tab_kpi:
        st.markdown("### Resumen diario por rango (Cobro vs Coloco)")

        # Fechas disponibles combinando ambas fuentes
        fechas_disp = pd.concat([df["FechaOperacion"].dropna(), df["FechaConfirmado"].dropna()], axis=0)
        fechas_disp = pd.to_datetime(fechas_disp, errors="coerce").dropna().dt.date
        if len(fechas_disp) == 0:
            st.info("No hay fechas disponibles en la data.")
        else:
            min_f, max_f = fechas_disp.min(), fechas_disp.max()
            fi_ff = st.date_input(
                "Rango de fechas",
                value=(max_f - pd.Timedelta(days=6), max_f),  # por defecto últimos 7 días
                min_value=min_f,
                max_value=max_f,
                key="rango_cobro_coloco"
            )
            # soporta selección simple o rango
            if isinstance(fi_ff, tuple) and len(fi_ff) == 2:
                fi_sel, ff_sel = fi_ff
            else:
                fi_sel = ff_sel = fi_ff

            df_rango = rango_coloc_cobro(df, tc_pen_por_usd, fi_sel, ff_sel)

            # Totales
            total_cobro  = df_rango["Cobro"].sum()
            total_coloco = df_rango["Coloco"].sum()
            total_gap    = df_rango["Gap"].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Cobro (PEN)",  fmt_money(total_cobro))
            c2.metric("Coloco (PEN)", fmt_money(total_coloco))
            c3.metric("Gap (Coloco - Cobro)", fmt_money(total_gap))

            # Tabla formateada
            mostrar = df_rango.copy()
            mostrar["Fecha"]  = mostrar["Fecha"].dt.date
            mostrar["Cobro"]  = mostrar["Cobro"].map(fmt_money)
            mostrar["Coloco"] = mostrar["Coloco"].map(fmt_money)
            mostrar["Gap"]    = mostrar["Gap"].map(fmt_money)

            st.dataframe(
                mostrar[["Fecha","Cobro","Coloco","Gap"]],
                use_container_width=True,
                hide_index=True,
                height=380
            )

            # (Opcional) Descargar CSV con valores numéricos
            csv = df_rango.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descargar CSV (numérico)",
                data=csv,
                file_name=f"rango_cobro_coloco_{fi_sel}_{ff_sel}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            st.markdown("### Diario: Colocación vs Cobranza por fecha")

            # Fechas disponibles (de FechaOperacion y FechaConfirmado)
            fechas_disp = pd.concat([
                df["FechaOperacion"].dropna(),
                df["FechaConfirmado"].dropna()
            ], axis=0)
            fechas_disp = pd.to_datetime(fechas_disp, errors="coerce").dropna().dt.date.unique()
            fechas_disp = sorted(fechas_disp)

            if len(fechas_disp) == 0:
                st.info("No hay fechas disponibles en la data para vista diaria.")
            else:
                fecha_default = fechas_disp[-1]  # la más reciente
                fecha_sel = st.date_input(
                    "Selecciona fecha",
                    value=fecha_default,
                    min_value=fechas_disp[0],
                    max_value=fechas_disp[-1],
                    key="retomas_fecha_unica"
                )

                from core.fun_varios import diario_coloc_cobro
                emp_view, eje_view = diario_coloc_cobro(df, tc_pen_por_usd, fecha_sel, por="auto")

                # --- Tabla Ejecutivos ---
                st.markdown("#### Resumen por Ejecutivo")
                eje_show = eje_view.loc[:, ["Ejecutivo","Coloco","Cobro","Gap","Pct"]]
                st.dataframe(eje_show, use_container_width=True, hide_index=True, height=380)

                total_coloco_eje = eje_view["ColocoPEN"].sum()
                total_cobro_eje  = eje_view["CobroPEN"].sum()
                st.markdown(
                    f"**Total Ejecutivos** → Colocó: {fmt_money(total_coloco_eje)} | Cobró: {fmt_money(total_cobro_eje)}"
                )

                # --- Tabla Empresas ---
                st.markdown("#### Empresas (colocó/cobró en el día)")
                emp_show = emp_view.loc[:, [c for c in emp_view.columns if c not in ("ColocoPEN","CobroPEN","GapPEN")]]
                st.dataframe(emp_show, use_container_width=True, hide_index=True, height=380)

                total_coloco_emp = emp_view["ColocoPEN"].sum()
                total_cobro_emp  = emp_view["CobroPEN"].sum()
                st.markdown(
                    f"**Total Empresas** → Colocó: {fmt_money(total_coloco_emp)} | Cobró: {fmt_money(total_cobro_emp)}"
                )







    with tab_resumen:
        
        st.subheader("Resumen general")
        Lm2, Lm1, L = labels  # P-2, P-1, P (orden cronológico)
        total      = k.get("total_con_algo", 0)
        activos33  = k.get("activo_tres_meses", k.get("total_activos_3m", 0))
        pct33      = k.get("pct_activo_tres_meses", k.get("pct_activos_3m", 0))
        dejoP      = k.get("dejo_en_P", 0)
        actUlt2    = k.get("activo_ultimos_2", k.get("act_ultimos_2", 0))
        intermit   = k.get("intermitentes", 0)
        solo1de3   = k.get("solo_1_de_3", k.get("solo_1_de_3", 0))

        def fmt_pct(x):
            try:
                return f"{float(x):.0%}"
            except Exception:
                return "—"

        c0, c1, c2, c3 = st.columns(4)
        c0.metric("Activos ultimos 3 meses", f"{activos33}")
        c1.metric(f"Dejó de colocar en {L}", f"{dejoP}")
        c2.metric("Activos últimos 2", f"{actUlt2}")
        c3.metric("Intermitentes", f"{intermit}")
        
        sit_opts = sorted(df_sit["Situacion"].unique())
        sit_sel = st.multiselect("Filtrar por situación", options=sit_opts, default=None)
        df_res = df_sit if not sit_sel else df_sit[df_sit["Situacion"].isin(sit_sel)]

        cols_view = id_cols + [*labels, "Tendencia", "Situacion", "FechaUltOp", "EjecutivoUlt"]
        cols_view = [c for c in cols_view if c in df_res.columns]

        st.dataframe(df_res.loc[:, cols_view].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab_detalle:
        st.subheader("Detalle por Situación y Ejecutivo")

        # Detecta columnas ID (cedente o pagador)
        if "RUCCliente" in df_sit.columns:
            id_cols = ["RUCCliente", "RazonSocialCliente"]
        else:
            id_cols = ["RUCPagador", "RazonSocialPagador"]

        # Orden sugerido de situaciones (usa solo las que existan)
        sit_order = [
            "Activo tres meses",
            f"Dejó de colocar en {labels[2]}",   # {L} = mes más reciente
            "Activo últimos 2 meses",
            "Intermitente",
            f"Activo solo en {labels[2]}",
            f"Activo solo en {labels[1]}",
            f"Activo solo en {labels[0]}",
            "Sin actividad 3M",
        ]
        situaciones = [s for s in sit_order if s in df_sit["Situacion"].unique()]

        # Filtro rápido por situación (opcional)
        sit_sel = st.multiselect("Filtrar situaciones", options=situaciones, default=situaciones)

        # Búsqueda por RUC/Nombre (opcional)
        q = st.text_input("Buscar por RUC o Razón social").strip().lower()

        for sit in sit_sel:
            df_s = df_sit[df_sit["Situacion"] == sit].copy()

            # ➜ Para recurrentes 3/3, mostrar SOLO tendencia negativa
            if sit == "Activo tres meses":
                df_s = df_s[df_s["Tendencia"] == "↘"]

            # Filtro de texto
            if q:
                mask_q = (
                    df_s[id_cols[0]].astype(str).str.contains(q, case=False, na=False) |
                    df_s[id_cols[1]].astype(str).str.lower().str.contains(q, na=False)
                )
                df_s = df_s[mask_q]

            # Título por situación (anota la condición especial)
            titulo = f"{sit}"
            if sit == "Activo tres meses":
                titulo += " (solo tendencia ↘)"

            st.markdown(f"### {titulo}  \n**Total:** {len(df_s)}")

            if df_s.empty:
                st.info("Sin registros para esta situación con el filtro actual.")
                continue

            # Sublistas por EjecutivoUlt (incluye '—' si no hay asignado)
            for eje, df_e in df_s.groupby(df_s["EjecutivoUlt"].fillna("—"), dropna=False):
                with st.expander(f"{eje} — {len(df_e)}"):
                    # Columnas a mostrar (IDs + 3 meses + tendencia + última op)
                    view_cols = id_cols + [labels[0], labels[1], labels[2], "Tendencia", "FechaUltOp", "MontoUltOpPEN"]
                    view_cols = [c for c in view_cols if c in df_e.columns]

                    st.dataframe(
                        df_e.loc[:, view_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                        height=min(360, 80 + 28*len(df_e))
                    )

                    # Botón de descarga del subset del ejecutivo
                    csv = df_e.loc[:, view_cols].to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label=f"Descargar CSV — {sit} — {eje}",
                        data=csv,
                        file_name=f"{sit} - {eje}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
