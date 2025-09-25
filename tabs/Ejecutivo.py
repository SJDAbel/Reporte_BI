
# ------------------------------------------------------------
# Pestaña Ejecutivo –  (Factoring/Confirming)
# ------------------------------------------------------------
# - Filtros: fechas, moneda, ejecutivo, tipo de cambio y tipo de operación (Factoring/Confirming)
# - KPIs del período actual (Colocación PEN, Operaciones, Ticket PEN, Tasa Ponderada)
# - Comparación con P-1, P-2, P-3 (ventanas contiguas, misma duración)
# - Calendario diario (ECharts) en PEN
# - Barras por periodo y tipo de operación (Actual vs previos)
# - Top 10 Pagador Factoring, Top 10 Pagador Confirming, Top 10 Cedente Factoring
# - Recurrentes (presentes en P-3, P-2, P-1): Cedentes Factoring y Pagadores Confirming
# ------------------------------------------------------------

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
    add_monto_pen, resumir_por_operacion, kpis_ejecutivo,
    ventanas_previas, fmt_money, tasa_a_pct_str,month_windows_from,
    pct_delta,tabla_top,recurrentes_3p
)


    
# =========================
# UI – Pestaña Ejecutivo
# =========================

def render(df, df_f, fi, ff, sel_monedas, sel_ejec, sel_tipos, tc_pen_por_usd: float):
    st.title(f"Analisis de Cartera")

    # Nivel operación sobre el período actual filtrado
    ops_cur = resumir_por_operacion(df_f)
    
    # Dataset comparación períodos (Actual vs P-1, P-2, P-3)
    # Ventanas previas P-3, P-2, P-1 (misma duración)
    v_prev = ventanas_previas(ff)

    # Dataset para barras: Actual + P-1, P-2, P-3 por TipoOperacion (en PEN)
    comp_rows = []



    for label, fi_k, ff_k in v_prev:
        d_k = df[(df["FechaOperacion"] >= fi_k) & (df["FechaOperacion"] <= ff_k)].copy()
        if sel_monedas: d_k = d_k[d_k["Moneda"].isin(sel_monedas)]
        if sel_ejec:    d_k = d_k[d_k["Ejecutivo"].isin(sel_ejec)]
        if sel_tipos:   d_k = d_k[d_k["TipoOperacion"].isin(sel_tipos)]
        
        d_k_pen = add_monto_pen(d_k, tc_pen_por_usd)
        assert "MontoNetoPEN" in d_k_pen.columns, f"No se creó MontoNetoPEN (previo {label}). Columnas: {d_k_pen.columns.tolist()}"
        gk = (d_k_pen.groupby("TipoOperacion", as_index=False)["MontoNetoPEN"].sum().assign(Periodo=label))
        comp_rows.append(gk)

    comp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame(columns=["TipoOperacion","MontoNetoPEN","Periodo"])

    #####

    # === KPIs (mes actual vs mes anterior) ==========================
    df_k = df.copy()
    if sel_monedas: df_k = df_k[df_k["Moneda"].isin(sel_monedas)]
    if sel_ejec:    df_k = df_k[df_k["Ejecutivo"].isin(sel_ejec)]
    if sel_tipos:   df_k = df_k[df_k["TipoOperacion"].isin(sel_tipos)]

    # Ventanas: mes de 'ff' y mes anterior
    (m0_s, m0_e), (m1_s, m1_e) = month_windows_from(ff)

    cur_m   = df_k[(df_k["FechaOperacion"] >= m0_s) & (df_k["FechaOperacion"] <= m0_e)].copy()
    prev_m  = df_k[(df_k["FechaOperacion"] >= m1_s) & (df_k["FechaOperacion"] <= m1_e)].copy()

    ops_cur_m  = resumir_por_operacion(cur_m)
    ops_prev_m = resumir_por_operacion(prev_m)

    k_cur_m  = kpis_ejecutivo(ops_cur_m, tc_pen_por_usd)
    k_prev_m = kpis_ejecutivo(ops_prev_m, tc_pen_por_usd)

    # Valores actuales
    val_coloc = k_cur_m["monto_total_pen"]
    val_ops   = k_cur_m["ops"]
    val_tick  = k_cur_m["ticket_pen"]
    val_tasa  = k_cur_m["tasa_pond"]

    # Valores previos
    prev_coloc = k_prev_m["monto_total_pen"]
    prev_ops   = k_prev_m["ops"]
    prev_tick  = k_prev_m["ticket_pen"]

    st.subheader("Indicadores claves")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monto total", fmt_money(val_coloc), pct_delta(val_coloc, prev_coloc))
    c2.metric("Operaciones", f"{val_ops:,}",     pct_delta(val_ops,   prev_ops))
    c3.metric("Ticket promedio", fmt_money(val_tick), pct_delta(val_tick, prev_tick))
    c4.metric("Tasa nominal ponderada", tasa_a_pct_str(val_tasa), None)

    # Estilo Metric Cards (streamlit-extras)
    style_metric_cards(
        background_color="#f6f6f6d8",
        border_left_color="#D32F2F",
        border_color="#e0e0e0",
        box_shadow=True
    )


    #####
    # === Calendario diario (ECharts) últimos 6 meses, IGNORANDO el filtro de fechas ===
    

    st.subheader("Calendario de colocación total soles – Últimos 6 meses")

    # 1) Partimos del DF completo (sin filtrar por fechas)
    df_cal = df.copy()

    # 2) Aplicamos SOLO filtros de moneda / ejecutivo / tipos
    if sel_monedas:
        df_cal = df_cal[df_cal["Moneda"].isin(sel_monedas)]
    if sel_ejec:
        df_cal = df_cal[df_cal["Ejecutivo"].isin(sel_ejec)]
    if sel_tipos:
        df_cal = df_cal[df_cal["TipoOperacion"].isin(sel_tipos)]

    # 3) Convertimos a PEN (garantizado aunque esté vacío)
    df_cal_pen = add_monto_pen(df_cal, tc_pen_por_usd)

    if df_cal_pen.empty:
        st.info("No hay datos para construir el calendario.")
    else:
        # 4) Fijamos SIEMPRE últimos 12 meses respecto a la fecha máxima disponible
        fecha_max = df_cal_pen["FechaOperacion"].max()
        fecha_min = fecha_max - pd.DateOffset(months=6)

        # 5) Tomamos ese recorte de 12 meses (independiente del filtro de fechas de la UI)
        m3 = df_cal_pen[(df_cal_pen["FechaOperacion"] >= fecha_min) &
                        (df_cal_pen["FechaOperacion"] <= fecha_max)].copy()

        if m3.empty:
            st.info("No hay datos en los últimos 6 meses (en el subset por moneda/ejecutivo/tipo).")
        else:
            # 6) Agregación diaria + redondeo 2 decimales
            day = (m3.groupby(m3["FechaOperacion"].dt.date)["MontoNetoPEN"]
                    .sum()
                    .reset_index()
                    .rename(columns={"FechaOperacion": "Fecha", "MontoNetoPEN": "Monto"}))

            data_heat = [
                [pd.to_datetime(d).strftime("%Y-%m-%d"), round(float(v), 2)]
                for d, v in zip(day["Fecha"], day["Monto"])
            ]

            vmax = float(np.percentile(day["Monto"], 95)) if len(day) else 1.0
            if vmax <= 0:
                vmax = float(day["Monto"].max() or 1)

            options = {
                "tooltip": {
                    "position": "top"
                },
                "visualMap": {
                    "min": 0, "max": vmax, "type": "continuous",
                    "orient": "horizontal", "left": "center", "bottom": 10,
                    "inRange": {"color": ["#f5f5f5a9", "#D32F2F"]}
                },
                "calendar": {
                    "range": [fecha_min.strftime("%Y-%m-%d"), fecha_max.strftime("%Y-%m-%d")],
                    "cellSize": [30, 30],
                    "yearLabel": {"show": True, "margin": 50, "fontSize": 20, "color": "#777"},
                    "splitLine": {"show": True, "lineStyle": {"color":"#8c8b8b","width": 2}},
                    "monthLabel": {"nameMap": ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"], "margin": 20, "fontSize": 20, "color": "#777"},
                    "dayLabel":   {"nameMap": ["Dom","Lun","Mar","Mié","Jue","Vie","Sáb"], "firstDay": 1, "margin": 10, "fontSize": 20, "color": "#777"},
                },
                "series": [{
                    "type": "heatmap",
                    "coordinateSystem": "calendar",
                    "data": data_heat
                }]
            }

            st_echarts(options=options, height="360px", key="cal_ultimos12m_sin_filtro_fecha")

                       
        # Barras por período y tipo de operación
        st.markdown("---")
        st.markdown("### Colocación por período y tipo de operación (PEN)")
        if comp.empty:
            st.info("Sin datos para comparar períodos.")
        else:
            fig = px.bar(
                comp, x="Periodo", y="MontoNetoPEN", color="TipoOperacion",
                barmode="group", title="",
            )
            fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
            fig.update_layout(margin=dict(t=60, r=20, b=20, l=20))
            st.plotly_chart(fig, use_container_width=True)
            
        # === KPIs por moneda ===
        st.markdown("---")
        st.markdown("### KPIs por moneda")

        monedas_presentes = ops_cur["Moneda"].dropna().astype(str).unique().tolist()
        for m in ["PEN", "USD"]:
            if m not in monedas_presentes:
                continue

            ops_m = ops_cur[ops_cur["Moneda"] == m].copy()

            # Montos en moneda nativa
            monto_m = float(ops_m["MontoOperacion"].sum())
            ops_n   = int(ops_m["CodigoLiquidacion"].nunique())
            ticket_m = float(ops_m["MontoOperacion"].mean() if ops_n else 0.0)

            # Tasa promedio ponderada por operación (peso: monto en la misma moneda)
            w_native = ops_m["MontoOperacion"]
            tasa_m = float((ops_m["TasaPondOp"] * w_native).sum() / w_native.sum()) if w_native.sum() else np.nan

            # Equivalente a PEN (para referencia)
            factor = 1.0 if m == "PEN" else float(tc_pen_por_usd)
            monto_pen_equiv = monto_m * factor
            ticket_pen_equiv = ticket_m * factor

            st.caption(f"**{m}**  (equivalente: {fmt_money(monto_pen_equiv)} PEN; ticket: {fmt_money(ticket_pen_equiv)} PEN)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Colocación ({m})", fmt_money(monto_m))
            c2.metric("Operaciones", f"{ops_n:,}")
            c3.metric(f"Ticket Prom. ({m})", fmt_money(ticket_m))
            c4.metric("Tasa Nominal Ponderada", tasa_a_pct_str(tasa_m))
            st.markdown("")  # pequeño espacio


        st.markdown("---")
        st.markdown("### TOP 10 (PEN)")

        cA, cB, cC = st.columns(3)
        with cA:
            st.caption("**Top 10 Pagador – Factoring**")
            tf = tabla_top(df_f[df_f["TipoOperacion"]=="Factoring"],
                        by_cols=["RUCPagador","RazonSocialPagador"],
                        label_cols=["RUCPagador","RazonSocialPagador"], tc=tc_pen_por_usd, N=10)
            st.dataframe(tf, use_container_width=True, height=330, hide_index=True)

        with cB:
            st.caption("**Top 10 Pagador – Confirming**")
            tcg = tabla_top(df_f[df_f["TipoOperacion"]=="Confirming"],
                            by_cols=["RUCPagador","RazonSocialPagador"],
                            label_cols=["RUCPagador","RazonSocialPagador"], tc=tc_pen_por_usd, N=10)
            st.dataframe(tcg, use_container_width=True, height=330, hide_index=True)

        with cC:
            st.caption("**Top 10 Cedente**")
            tcf = tabla_top(df_f[df_f["TipoOperacion"]=="Factoring"],
                            by_cols=["RUCCliente","RazonSocialCliente"],
                            label_cols=["RUCCliente","RazonSocialCliente"], tc=tc_pen_por_usd, N=10)
            st.dataframe(tcf, use_container_width=True, height=330, hide_index=True)

        st.markdown("---")
        st.markdown("### Recurrentes en los 3 últimos periodos (continuos)")

        v3 = ventanas_previas(ff)
        colR1, colR2 = st.columns(2)

        with colR1:
            st.caption("**Cedentes**")
            rec_ced = recurrentes_3p(
                df,
                v3,
                modo="Factoring_cedente",
                tc=tc_pen_por_usd,
                monedas=sel_monedas,
                ejecutivos=sel_ejec,
            )
            st.dataframe(rec_ced, use_container_width=True, height=320, hide_index=True)

        with colR2:
            st.caption("**Pagadores – Confirming**")
            rec_pag = recurrentes_3p(
                df,
                v3,
                modo="Confirming_pagador",
                tc=tc_pen_por_usd,
                monedas=sel_monedas,
                ejecutivos=sel_ejec,
            )
            st.dataframe(rec_pag, use_container_width=True, hide_index=True,height=320)

    # =========================
    # (Opcional) Descargas del período actual
    # =========================
    st.markdown("---")
    st.subheader("Descargar datos filtrados (período actual)")
    colD1, colD2 = st.columns(2)
    with colD1:
        csv = df_f.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSV filtrado", data=csv, file_name="ejecutivo_filtrado.csv", mime="text/csv")
    with colD2:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df_f.to_excel(writer, index=False, sheet_name="Datos")
        st.download_button("⬇️ Excel filtrado", data=out.getvalue(),
                        file_name="ejecutivo_filtrado.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("---")