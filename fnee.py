# fnee.py
import pandas as pd
import streamlit as st

FNEE_EUR_MWH = 1.429  # €/MWh

@st.cache_data(show_spinner=False)
def coste_fnee(df_consumo: pd.DataFrame, inicio: str | None = None, fin: str | None = None):
    """
    Calcula el coste del FNEE:
    - df_consumo: DataFrame con columnas ['datetime','consumo'] (kWh/15min).
    - inicio/fin opcionales en 'YYYY-MM-DD' (fin exclusivo).
    Devuelve: (coste_total_eur, energia_total_MWh)
    """
    df = df_consumo.copy()

    # Filtro temporal opcional (si el datetime ya está tz-aware, funciona igual)
    if inicio is not None:
        df = df[df["datetime"] >= pd.to_datetime(inicio)]
    if fin is not None:
        df = df[df["datetime"] < pd.to_datetime(fin)]

    energia_mwh = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0).sum() / 1000.0
    total_eur = float(energia_mwh * FNEE_EUR_MWH)
    return total_eur, float(energia_mwh)
