# fv.py
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def coste_excedentes_fijo(
    df_excedentes: pd.DataFrame,
    precio_excedentes: float,
    inicio: str | None = None,
    fin: str | None = None,
):
    """
    Calcula los ingresos por venta de excedentes FV en modalidad FIJA.
    - df_excedentes: DataFrame con columnas ['datetime', <col_excedente_en_kWh>]
    - precio_excedentes: precio de compra de excedentes (€/MWh o €/kWh)
    Retorna: (ingresos_eur, energia_excedente_MWh, precio_eur_MWh_usado)
    """
    if df_excedentes is None or len(df_excedentes) == 0:
        return 0.0, 0.0, float(precio_excedentes or 0)

    df = df_excedentes.copy()

    # Recorte opcional por fechas (si lo necesitas más adelante)
    if inicio is not None:
        df = df[df["datetime"] >= pd.to_datetime(inicio)]
    if fin is not None:
        df = df[df["datetime"] < pd.to_datetime(fin)]

    # Detecta automáticamente la columna de excedentes (numérica distinta de 'datetime')
    cand = [c for c in df.columns if c.lower() != "datetime"]
    if not cand:
        return 0.0, 0.0, float(precio_excedentes or 0)
    # coge la primera columna numérica
    for c in cand:
        if pd.api.types.is_numeric_dtype(df[c]):
            col_exc = c
            break
    else:
        col_exc = cand[0]

    energia_mwh = pd.to_numeric(df[col_exc], errors="coerce").fillna(0.0).sum() / 1000.0

    # Unidades del precio: si parece €/kWh (típico valores < 5), convierto a €/MWh
    p = float(precio_excedentes or 0.0)
    precio_eur_mwh = p * 1000.0 if p > 0 and p < 5 else p

    ingresos = float(energia_mwh * precio_eur_mwh)
    return ingresos, float(energia_mwh), float(precio_eur_mwh)
