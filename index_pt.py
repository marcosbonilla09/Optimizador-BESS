import pandas as pd
import streamlit as st

TZ = "Europe/Madrid"

def _ensure_tz(ts):
    ts = pd.to_datetime(ts)
    return ts.tz_localize(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)

# index_pt.py
@st.cache_data(show_spinner=False)
def attach_market_to_consumo(df_consumo: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Alinea el mercado a la rejilla del consumo SIN cambiar el nº de filas,
    usando merge_asof (ffill temporal) para tolerar duplicados en el consumo
    (p. ej., cambio de hora).
    """
    if market is None or len(market) == 0:
        raise ValueError("Market viene vacío o None.")

    # --- 1) Rejilla del CONSUMO -> tz-aware Europe/Madrid ---
    grid = pd.to_datetime(df_consumo["datetime"], errors="coerce").round("S")
    try:
        grid = grid.dt.tz_localize("Europe/Madrid", ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        try:
            grid = grid.dt.tz_localize("Europe/Madrid", ambiguous=False, nonexistent="shift_forward")
        except Exception:
            tmp = grid.dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="shift_forward")
            grid = tmp.fillna(method="bfill")

    mask_ok = ~grid.isna()
    grid = grid[mask_ok]
    c = df_consumo.loc[mask_ok].copy()

    # Quita columna datetime; usaremos 'grid' como índice final
    if "datetime" in c.columns:
        c = c.drop(columns=["datetime"])

    # --- 2) Asegura mercado en misma zona horaria ---
    mk = market.copy()
    mk_idx = pd.DatetimeIndex(mk.index)
    if mk_idx.tz is None:
        mk_idx = mk_idx.tz_localize("Europe/Madrid")
    else:
        mk_idx = mk_idx.tz_convert("Europe/Madrid")
    mk.index = mk_idx

    # --- 3) merge_asof: ffill temporal a la rejilla del consumo ---
    # a) Prepara dataframes ordenados por tiempo
    left = pd.DataFrame({"dt": grid, "_row": range(len(grid))}).sort_values("dt")
    right = mk.reset_index().rename(columns={mk.index.name or "index": "dt"}).sort_values("dt")

    # b) Alinea: para cada dt del consumo, toma el último dt del mercado <= dt
    aligned = pd.merge_asof(left, right, on="dt", direction="backward")

    # c) Restaura orden original y quita columnas auxiliares
    aligned = aligned.set_index("_row").loc[range(len(grid))].drop(columns=["dt"])

    # --- 4) Construye salida con EXACTAMENTE las mismas filas que consumo ---
    out = c.copy()
    for col in aligned.columns:
        out[col] = aligned[col].values

    # Índice final = grid (puede tener duplicados, y está bien)
    out.index = grid

    # Guardia
    if len(out) != len(df_consumo.loc[mask_ok]):
        raise RuntimeError(f"attach_market_to_consumo cambió filas: {len(df_consumo.loc[mask_ok])} → {len(out)}")

    return out


def _col_perdidas_por_tarifa(tarifa: str) -> str:
    """Devuelve el nombre de columna de pérdidas según la tarifa."""
    t = (tarifa or "").replace(" ", "").upper()
    if t == "2.0TD": return "perdidas_2p0TD"
    if t == "3.0TD": return "perdidas_3p0TD"
    # 6.XTD → escoger la específica si existe
    if t in ("6.1TD","61TD"): return "perdidas_6p1TD"
    if t in ("6.2TD","62TD"): return "perdidas_6p2TD"
    if t in ("6.3TD","63TD"): return "perdidas_6p3TD"
    if t in ("6.4TD","64TD"): return "perdidas_6p4TD"
    # Genérico “resto”: usa 6.1TD como fallback
    return "perdidas_6p1TD"

def _col_ssaa_por_tarifa(tarifa: str) -> str:
    """Devuelve el nombre de columna de ssaa según la tarifa."""
    t = (tarifa or "").replace(" ", "").upper()
    if t == "2.0TD": return "ssaa_20td"
    if t == "3.0TD": return "ssaa30td"
    # 6.XTD → escoger la específica si existe
    if t in ("6.1TD","61TD"): return "ssaa6xtd"
    if t in ("6.2TD","62TD"): return "ssaa6xtd"
    if t in ("6.3TD","63TD"): return "ssaa6xtd"
    if t in ("6.4TD","64TD"): return "ssaa6xtd"
    # Genérico “resto”: usa 6.1TD como fallback
    return "ssaa6xtd"

def _leer_desvios_y_cg() -> tuple[float, float]:
    """
    Busca desvíos y CG en session_state:
    - Preferencia: st.session_state['indexado_pt'] = {'desvios': x, 'CG': y}
    - Alternativas: st.session_state['comer']['desvios'] / ['CG'] o llaves sueltas.
    """
    des = None; cg = None
    if isinstance(st.session_state.get("indexado_pt"), dict):
        des = st.session_state["indexado_pt"].get("desvios", des)
        cg  = st.session_state["indexado_pt"].get("CG", cg)
    if isinstance(st.session_state.get("comer"), dict):
        des = st.session_state["comer"].get("desvios", des)
        cg  = st.session_state["comer"].get("CG", cg)
    des = float(des) if des is not None else 0.0
    cg  = float(cg)  if cg  is not None else 0.0
    return des, cg

def coste_passthrough(
    df_consumo: pd.DataFrame,
    df_mercado: pd.DataFrame,
    tarifa: str,
    desvios: float,
    cg: float,
    inicio: str | None = None,
    fin: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Calcula el coste total y devuelve también un detalle por intervalo con la fórmula:
      [(OMIE+SSAA+desvios)*(1+Pérdidas)+CG] * Energía_qh
    - df_consumo: cols ['datetime','consumo'] en kWh por 15 min
    - df_mercado: index datetime con columnas: 'omie', 'ssaa', pérdidas correspondientes
    """
    df = attach_market_to_consumo(df_consumo, df_mercado)

    # Recorte temporal opcional
    if inicio:
        df = df[df.index >= _ensure_tz(inicio)]
    if fin:
        df = df[df.index < _ensure_tz(fin)]

    # Columnas base
    perd_col = _col_perdidas_por_tarifa(tarifa)
    omie = pd.to_numeric(df["omie"], errors="coerce").fillna(0.0)
    ssaa_col = _col_ssaa_por_tarifa(tarifa)
    ssaa = pd.to_numeric(df[ssaa_col], errors="coerce").fillna(0.0)
    cons = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0)
    cons = cons / 1000.0   # ← conversión kWh → MWh

    # Pérdidas (acepta % o fracción)
    perd = pd.to_numeric(df[perd_col], errors="coerce").fillna(0.0)
    perd = perd.where(perd <= 1.0, perd / 100.0)

    # Fórmula PT
    precio_unit = ((omie + ssaa + float(desvios)) * (1.0 + perd)) + float(cg)
    coste_qh = precio_unit * cons

    detalle = pd.DataFrame(
        {
            "OMIE": omie,
            "SSAA": ssaa,
            "Desvíos": float(desvios),
            "Pérdidas(fracción)": perd,
            "CG": float(cg),
            "Precio_unitario(€/MWh)": precio_unit,
            "consumo(MWh)": cons,
            "coste_intervalo(€)": coste_qh,
        },
        index=df.index,
    )

    return float(coste_qh.sum()), detalle


def _col_peajes_por_tarifa(tarifa: str) -> str:
    t = (tarifa or "").replace(" ", "").upper()
    if t == "2.0TD": return "peajes_cargos_20td"
    if t == "3.0TD": return "peajes_cargos_30td"
    if t in ("6.1TD","61TD"): return "peajes_cargos_61td"
    if t in ("6.2TD","62TD"): return "peajes_cargos_62td"
    if t in ("6.3TD","63TD"): return "peajes_cargos_63td"
    if t in ("6.4TD","64TD"): return "peajes_cargos_64td"
    # fallback razonable
    return "peajes_cargos_61td"

@st.cache_data(show_spinner=False)
def coste_atr_energia(
    df_consumo: pd.DataFrame,   # columnas: ['datetime','consumo'] en kWh/15min
    df_mercado: pd.DataFrame,   # index datetime con columnas peajes_cargos_*
    tarifa: str,
    inicio: str | None = None,
    fin: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Calcula ATR energía por QH: Consumo(MWh) * Peajes_y_cargos_tarifa(€/MWh)
    Devuelve (coste_total, df_detalle).
    """
    # Alinear consumo con mercado (manejo DST ya dentro de attach_market_to_consumo)
    from index_pt import attach_market_to_consumo  # si está en este mismo archivo, quita el import
    df = attach_market_to_consumo(df_consumo, df_mercado)

    # Recorte temporal opcional
    if inicio:
        t0 = pd.to_datetime(inicio)
        t0 = t0.tz_localize(TZ) if t0.tzinfo is None else t0.tz_convert(TZ)
        df = df[df.index >= t0]
    if fin:
        t1 = pd.to_datetime(fin)
        t1 = t1.tz_localize(TZ) if t1.tzinfo is None else t1.tz_convert(TZ)
        df = df[df.index < t1]

    # Columnas
    col_peajes = _col_peajes_por_tarifa(tarifa)
    peajes = pd.to_numeric(df[col_peajes], errors="coerce").fillna(0.0)   # €/MWh
    cons_mwh = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0) / 1000.0  # kWh → MWh

    coste_qh = cons_mwh * peajes  # € por QH

    detalle = pd.DataFrame(
        {
            "consumo(MWh)": cons_mwh,
            "peajes_y_cargos(€/MWh)": peajes,
            "coste_ATR_qh(€)": coste_qh,
        },
        index=df.index,
    )

    return float(coste_qh.sum()), detalle, peajes
