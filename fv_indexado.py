# fv_indexado.py
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def coste_excedentes_indexado(df_excedentes: pd.DataFrame,
                              df_mercado: pd.DataFrame,
                              cg_fv: float):
    """
    Ingresos FV indexado por QH:
      ingreso_qh = (excedentes_kWh / 1000) * (OMIE_qh - CG_FV)
    Devuelve: (ingresos_totales €, detalle QH)
    """
    if df_excedentes is None or len(df_excedentes) == 0:
        return 0.0, pd.DataFrame()

    # --- Excedentes: asegurar columnas y tz Europe/Madrid ---
    df_exc = df_excedentes.copy()
    df_exc["datetime"] = pd.to_datetime(df_exc["datetime"], errors="coerce").round("S")
    df_exc["excedentes_kWh"] = pd.to_numeric(df_exc["excedentes_kWh"], errors="coerce").fillna(0.0)
    df_exc = df_exc.dropna(subset=["datetime"]).sort_values("datetime")

    # Localiza a Europe/Madrid (naive -> tz-aware)
    try:
        dt_exc = df_exc["datetime"].dt.tz_localize("Europe/Madrid", ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        try:
            dt_exc = df_exc["datetime"].dt.tz_localize("Europe/Madrid", ambiguous=False, nonexistent="shift_forward")
        except Exception:
            tmp = df_exc["datetime"].dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="shift_forward")
            dt_exc = tmp.fillna(method="bfill")
    df_exc["datetime"] = dt_exc

    # --- Mercado: asegurar tz Europe/Madrid ---
    mk = df_mercado.copy()
    mk_idx = pd.DatetimeIndex(mk.index)
    if mk_idx.tz is None:
        mk_idx = mk_idx.tz_localize("Europe/Madrid")
    else:
        mk_idx = mk_idx.tz_convert("Europe/Madrid")
    mk.index = mk_idx

    # Localizar columna OMIE
    omie_col = None
    for c in mk.columns:
        if str(c).strip().lower() in ("omie", "precio omie", "precio_omie"):
            omie_col = c
            break
    if omie_col is None:
        raise KeyError("No se encontró columna OMIE en la base de mercado.")

    # --- Alineación temporal: merge_asof (permite duplicados a la izquierda) ---
    left = df_exc[["datetime", "excedentes_kWh"]].sort_values("datetime")
    right = mk.reset_index().rename(columns={mk.index.name or "index": "datetime"}).sort_values("datetime")
    merged = pd.merge_asof(left, right, on="datetime", direction="backward")

    # --- Cálculo de ingresos ---
    merged["OMIE_€MWh"] = pd.to_numeric(merged[omie_col], errors="coerce").fillna(0.0)
    merged["CG_FV"] = float(cg_fv or 0.0)
    merged["excedentes_MWh"] = merged["excedentes_kWh"] / 1000.0
    merged["precio_final_€MWh"] = merged["OMIE_€MWh"] - merged["CG_FV"]
    merged["ingreso_€"] = merged["excedentes_MWh"] * merged["precio_final_€MWh"]

    total_ingresos = float(merged["ingreso_€"].sum())
    return total_ingresos, merged
