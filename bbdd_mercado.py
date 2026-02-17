from pathlib import Path
import pandas as pd
import streamlit as st

TZ = "Europe/Madrid"

MARKET_PATH = Path("data/Datos mercado2024-25.xlsx")

def _localize_eu_madrid(dt_series: pd.Series) -> pd.Series:
    """Localiza a Europe/Madrid gestionando DST (horas duplicadas / inexistentes)."""
    dt = pd.to_datetime(dt_series, errors="coerce").dt.round("S")
    # 1) Intento estándar: inferir la ocurrencia correcta en el cambio de hora
    try:
        return dt.dt.tz_localize("Europe/Madrid", ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        # 2) Si sigue fallando, fuerza la 2ª ocurrencia (horario de invierno)
        try:
            return dt.dt.tz_localize("Europe/Madrid", ambiguous=False, nonexistent="shift_forward")
        except Exception:
            # 3) Último recurso: marcar ambiguos como NaT y rellenar hacia delante
            tmp = dt.dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="shift_forward")
            return tmp.fillna(method="bfill")


@st.cache_data(show_spinner=False)
def load_market_db(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el Excel de mercado en: {path}")
    usecols = "A,B,C,D,E,F,G,H,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC"
    names = [
        "datetime","omie",
        "perdidas_2p0TD","perdidas_3p0TD","perdidas_6p1TD","perdidas_6p2TD","perdidas_6p3TD","perdidas_6p4TD",
        "ssaa_20td","periodos_no20td","periodos_20td",
        "pagos_cap_20td","pagos_cap_30td","pagos_cap_6x",
        "fnee","om_os",
        "peajes_cargos_20td","peajes_cargos_30td","peajes_cargos_61td","peajes_cargos_62td","peajes_cargos_63td","peajes_cargos_64td","ssaa30td","ssaa6xtd"
    ]
    df = pd.read_excel(path, usecols=usecols, header=0)
    df.columns = names

    # Localizar a tz con manejo de DST
    df["datetime"] = _localize_eu_madrid(df["datetime"])

    # A numérico todas menos los 'periodos_*' que pueden ser 'P1','P2',...
    cols_no_convertir = {"periodos_20td", "periodos_no20td"}
    for c in df.columns:
        if c == "datetime" or c in cols_no_convertir:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")


    # --- Limpieza/agrupación por timestamp ---
    # Columnas no numéricas que NO se deben promediar
    cols_periodos = ["periodos_20td", "periodos_no20td"]

    # 1) Numéricas -> media
    num_cols = [c for c in df.columns if c not in ["datetime"] + cols_periodos]
    df_num = (
        df.dropna(subset=["datetime"])
        .groupby("datetime", as_index=False)[num_cols].mean())

    # 2) Periodos -> primer valor (o podrías usar 'mode' si prefieres)
    df_per = (
        df.dropna(subset=["datetime"])
        .groupby("datetime", as_index=False)[cols_periodos].first())

    # 3) Unir y ordenar
    df = (
        df_num.merge(df_per, on="datetime", how="left")
            .set_index("datetime").sort_index())
    
    return df


def ensure_market_loaded():
    if "market" not in st.session_state:
        st.session_state["market"] = load_market_db(MARKET_PATH)


def attach_market_to_consumo(df_consumo: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Une consumo con columnas de mercado por datetime.
    Si el mercado es horario y el consumo es 15-min, se hace forward-fill para cada hora.
    """
    c = df_consumo.copy()
    if "datetime" in c.columns:
        c = c.set_index("datetime")
    # asegura tz
    if c.index.tz is None:
        c.index = c.index.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        c.index = c.index.tz_convert(TZ)

    aligned_market = market.reindex(c.index, method="ffill")
    return c.join(aligned_market, how="left")

def coste_energia_en_rango(df_costo: pd.DataFrame, inicio, fin, precio_col="omie") -> float:
    """
    Devuelve la suma de consumo*precio entre 'inicio' (incluido) y 'fin' (excluido).
    'inicio' y 'fin' pueden ser str con fecha/hora.
    """
    t0 = pd.to_datetime(inicio)
    t1 = pd.to_datetime(fin)
    if t0.tzinfo is None: t0 = t0.tz_localize(TZ)
    else: t0 = t0.tz_convert(TZ)
    if t1.tzinfo is None: t1 = t1.tz_localize(TZ)
    else: t1 = t1.tz_convert(TZ)

    df_sel = df_costo.loc[(df_costo.index >= t0) & (df_costo.index < t1)]
    return float((df_sel["consumo"] * df_sel[precio_col]).sum())
