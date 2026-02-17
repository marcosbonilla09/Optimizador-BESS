# fijo.py
import pandas as pd
import streamlit as st

TZ = "Europe/Madrid"

def _col_periodos_por_tarifa(tarifa: str) -> str:
    t = (tarifa or "").replace(" ", "").upper()
    return "periodos_20td" if t == "2.0TD" else "periodos_no20td"

def _num_periodos_energia(tarifa: str) -> int:
    return 3 if (tarifa or "").replace(" ", "").upper() == "2.0TD" else 6

def _normaliza_precios(precios_te: dict, n_per: int) -> dict:
    """
    Convierte {"Precio P1": v1, "Precio P2": v2, ...} → {"P1": v1, "P2": v2, ...}
    Rellena con 0 los periodos que falten hasta n_per.
    """
    out = {}
    for i in range(1, n_per + 1):
        out[f"P{i}"] = float(precios_te.get(f"Precio P{i}", 0.0))
    return out

@st.cache_data(show_spinner=False)
def coste_fijo_energia(
    df_consumo: pd.DataFrame,     # columnas: ['datetime','consumo'] (kWh/15min)
    df_mercado: pd.DataFrame,     # índice datetime con columnas 'periodos_20td' y 'periodos_no20td'
    tarifa: str,                  # '2.0TD','3.0TD','6.1TD',...
    precios_te: dict,             # {"Precio P1": €/MWh, "Precio P2": €/MWh, ...}
    inicio: str | None = None,
    fin: str | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Coste fijo por QH:
       coste_qh = precio_periodo(Pi)[€/MWh] * (consumo_qh[kWh] / 1000)
    Devuelve:
       - coste_total (€)
       - detalle por QH (index datetime)
       - resumen por periodo
    """
    # Alinear consumo con mercado (reutiliza función robusta con DST)
    from index_pt import attach_market_to_consumo, _ensure_tz

    df = attach_market_to_consumo(df_consumo, df_mercado)
    if len(df) != len(df_consumo):
        st.error(f"Fallo de alineación: {len(df_consumo)} → {len(df)} filas")
        st.stop()


    # Recorte opcional
    if inicio:
        df = df[df.index >= _ensure_tz(inicio)]
    if fin:
        df = df[df.index < _ensure_tz(fin)]

    # Periodos y nº de periodos de energía
    col_p = _col_periodos_por_tarifa(tarifa)
    n_per = _num_periodos_energia(tarifa)

    # 1) Intenta como numérico
    per_num = pd.to_numeric(df[col_p], errors="coerce")

    # 2) Donde no sea numérico (p.ej. 'P3'), extrae el dígito
    mask_na = per_num.isna()
    if mask_na.any():
        extra = pd.to_numeric(
            df.loc[mask_na, col_p].astype(str).str.upper().str.extract(r"(\d+)")[0],
            errors="coerce")
    per_num.loc[mask_na] = extra

    # 3) Normaliza rango válido (1..n_per)
    per_num = per_num.fillna(1).astype(int).clip(1, n_per)
    per_lbl = "P" + per_num.astype(str)


    # Consumo MWh
    cons_mwh = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0) / 1000.0

    # Mapa de precios por periodo
    precios_map = pd.Series(_normaliza_precios(precios_te, n_per), dtype="float64")
    precio_qh = per_lbl.map(precios_map).fillna(0.0)

        # Coste por QH
    coste_qh = precio_qh * cons_mwh

    # === DETALLE (nombres ASCII, sin símbolos) ===
    detalle = pd.DataFrame(
    {
        "Periodo": per_lbl.values,
        "consumo_MWh": cons_mwh.values,
        "precio_eur_MWh": precio_qh.values,
        "coste_qh_eur": coste_qh.values,
    },
    index=df.index,)

    # === RESUMEN POR PERIODO ===
    from pandas.api.types import CategoricalDtype
    orden = CategoricalDtype(categories=[f"P{i}" for i in range(1, n_per + 1)], ordered=True)

    resumen = (
        detalle.groupby("Periodo", as_index=False)
        .agg(
        energia_MWh=("consumo_MWh", "sum"),
        coste_eur=("coste_qh_eur", "sum"),)
        .assign(
        precio_medio_eur_MWh=lambda d: d["coste_eur"]
        / d["energia_MWh"].replace(0, pd.NA)))

    # ordenar P1..Pn de forma natural
    resumen["Periodo"] = resumen["Periodo"].astype(orden)
    resumen = resumen.sort_values("Periodo").reset_index(drop=True)

    total = float(detalle["coste_qh_eur"].sum())
    return total, detalle, resumen



