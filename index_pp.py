# index_pp.py
import pandas as pd
import streamlit as st

TZ = "Europe/Madrid"

def _col_periodos_por_tarifa(tarifa: str) -> str:
    t = (tarifa or "").replace(" ", "").upper()
    return "periodos_20td" if t == "2.0TD" else "periodos_no20td"

def _num_periodos_energia(tarifa: str) -> int:
    return 3 if (tarifa or "").replace(" ", "").upper() == "2.0TD" else 6

def _normalize_coef(dic: dict, n_per: int, prefix: str) -> dict:
    """
    Devuelve un dict normalizado {"P1": v1, ...} a partir de:
      - "Precio Ai P1" / "Precio Ci P1"  (lo que usas en Paso 3)
      - "A P1" / "C P1"
      - "Ai P1" / "Ci P1"
      - "A1" / "C1"
      - "P1"
    """
    # normaliza claves a minúsculas y sin espacios repetidos
    norm = {}
    for k, v in (dic or {}).items():
        if k is None:
            continue
        kk = " ".join(str(k).strip().split()).lower()  # ej. "precio ai p1"
        norm[kk] = v

    label = {"A": "ai", "C": "ci"}[prefix]  # "ai" o "ci"

    out = {}
    for i in range(1, n_per + 1):
        candidates = [
            f"precio {label} p{i}",  # "precio ai p1" / "precio ci p1"  ✅
            f"{prefix.lower()} p{i}",  # "a p1" / "c p1"
            f"{label} p{i}",           # "ai p1" / "ci p1"
            f"{prefix.lower()}{i}",    # "a1" / "c1"
            f"p{i}",                   # "p1"
        ]
        val = 0.0
        for kk in candidates:
            if kk in norm and norm[kk] is not None:
                try:
                    val = float(norm[kk])
                    break
                except Exception:
                    pass
        out[f"P{i}"] = val
    return out



def _get_omie_col(df_like) -> str:
    for c in df_like.columns:
        lc = str(c).strip().lower()
        if lc in ("omie", "precio_omie", "precio omie"):
            return c
    raise KeyError("No encuentro columna OMIE en la base de mercado.")

@st.cache_data(show_spinner=False)
def coste_indexado_pp(
    df_consumo: pd.DataFrame,
    df_mercado: pd.DataFrame,
    tarifa: str,
    coef_A: dict,   # {"A P1":..., "A P2":...} o {"P1":...}
    coef_C: dict,   # {"C P1":..., "C P2":...} o {"P1":...}
    modo_omie: str = "mensual",   # "mensual" o "horario"
    inicio: str | None = None,
    fin: str | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Precio por QH:  P_qh = A_Pi + C_Pi * OMIE_ref

    - Si modo_omie == "mensual": OMIE_ref es el promedio mensual de OMIE (€/MWh)
      del mes correspondiente a cada QH (comportamiento antiguo).
    - Si modo_omie == "horario": OMIE_ref es el precio OMIE horario/QH para
      cada instante del consumo.
    """

    # 1) Alinear consumo + mercado a rejilla del consumo
    from index_pt import attach_market_to_consumo, _ensure_tz
    df = attach_market_to_consumo(df_consumo, df_mercado)

    # 2) Recorte temporal opcional
    if inicio:
        df = df[df.index >= _ensure_tz(inicio)]
    if fin:
        df = df[df.index < _ensure_tz(fin)]

    modo = (modo_omie or "mensual").strip().lower()

    # 3) Construir serie OMIE por QH según el modo elegido
    try:
        # Intentamos primero leer la columna OMIE directamente del df ya alineado
        omie_col_df = _get_omie_col(df)
    except Exception:
        omie_col_df = None

    if modo.startswith("hor"):  # OMIE horario / QH
        if omie_col_df is not None:
            mk = df.copy()
            idx = pd.DatetimeIndex(mk.index)
            if idx.tz is None:
                idx = idx.tz_localize(TZ)
            else:
                idx = idx.tz_convert(TZ)
            mk.index = idx
            omie_qh = pd.to_numeric(mk[omie_col_df], errors="coerce")
        else:
            # Fallback: usar df_mercado y reindexar a la rejilla del consumo
            omie_col = _get_omie_col(df_mercado)
            mk = df_mercado.copy()
            idx = pd.DatetimeIndex(mk.index)
            if idx.tz is None:
                idx = idx.tz_localize(TZ)
            else:
                idx = idx.tz_convert(TZ)
            mk.index = idx
            omie_qh = pd.to_numeric(
                mk[omie_col].reindex(df.index.tz_convert(TZ)).ffill().bfill(),
                errors="coerce",
            )
    else:
        # OMIE mensual (promedio de todos los QH/horas del mes)
        omie_col = _get_omie_col(df_mercado)
        mk = df_mercado.copy()
        idx = pd.DatetimeIndex(mk.index)
        if idx.tz is None:
            idx = idx.tz_localize(TZ)
        else:
            idx = idx.tz_convert(TZ)
        mk.index = idx
        omie_mens = mk[omie_col].groupby(mk.index.to_period("M")).mean()

        # Serie con OMIE mensual para cada QH del consumo
        mes_qh = df.index.tz_convert(TZ).to_period("M")
        omie_qh = mes_qh.map(omie_mens)

    omie_qh = pd.to_numeric(omie_qh, errors="coerce").fillna(0.0)

    # 4) Periodos según tarifa
    col_p = _col_periodos_por_tarifa(tarifa)
    n_per = _num_periodos_energia(tarifa)

    # soporta 1/2/3 o "P1"/"P2"/"P3"
    per_num = pd.to_numeric(df[col_p], errors="coerce")
    mask_na = per_num.isna()
    if mask_na.any():
        extra = pd.to_numeric(
            df.loc[mask_na, col_p].astype(str).str.upper().str.extract(r"(\d+)")[0],
            errors="coerce"
        )
        per_num.loc[mask_na] = extra
    per_num = per_num.fillna(1).astype(int).clip(1, n_per)
    per_lbl = "P" + per_num.astype(str)

    # 5) Coeficientes por periodo
    Amap = pd.Series(_normalize_coef(coef_A, n_per, "A"), dtype="float64")
    Cmap = pd.Series(_normalize_coef(coef_C, n_per, "C"), dtype="float64")
    A_qh = per_lbl.map(Amap).fillna(0.0)
    C_qh = per_lbl.map(Cmap).fillna(0.0)

    # 6) Consumo MWh y precio/coste
    cons_mwh = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0) / 1000.0
    precio_qh = A_qh + C_qh * omie_qh
    coste_qh = precio_qh * cons_mwh

    # Detalle por QH
    detalle = pd.DataFrame(
        {
            "Periodo": per_lbl.values,
            "consumo_MWh": cons_mwh.values,
            "OMIE_ref": omie_qh.values,  # OMIE horario o mensual según modo_omie
            "A": A_qh.values,
            "C": C_qh.values,
            "precio_eur_MWh": precio_qh.values,
            "coste_qh_eur": coste_qh.values,
        },
        index=df.index,
    )

    # Resumen por periodo
    resumen = (
        detalle.groupby("Periodo", as_index=False)
        .agg(
            energia_MWh=("consumo_MWh", "sum"),
            coste_eur=("coste_qh_eur", "sum"),
        )
        .assign(
            precio_medio_eur_MWh=lambda d: d["coste_eur"]
            / d["energia_MWh"].replace(0, pd.NA)
        )
        .sort_values(
            "Periodo",
            key=lambda s: s.astype(str).str.extract(r"(\d+)")[0].astype(int),
        )
        .reset_index(drop=True)
    )

    total = float(coste_qh.sum())
    return total, detalle, resumen

