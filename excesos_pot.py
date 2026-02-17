# excesos.py
import pandas as pd
import streamlit as st

TZ = "Europe/Madrid"

def _col_periodos_por_tarifa(tarifa: str) -> str:
    t = (tarifa or "").replace(" ", "").upper()
    return "periodos_20td" if t == "2.0TD" else "periodos_no20td"

def _num_periodos_potencia(tarifa: str) -> int:
    """
    Nº de periodos de POTENCIA por tarifa.
    - 2.0TD: 2 periodos (P1,P2) para POTENCIA
    - Resto (3.0TD, 6.xTD): 6 periodos
    """
    t = (tarifa or "").replace(" ", "").upper()
    return 2 if t == "2.0TD" else 6

def _normaliza_potencias(dic: dict, n_per: int) -> dict:
    """
    Convierte dict de potencias de sesión a {"P1": kW, "P2": kW, ...}
    Acepta claves como:
      - "P1", "P2", ...
      - "Potencia P1", "Potencia contratada P1", etc.
    Rellena con 0.0 si faltan.
    """
    norm = {}
    for k, v in (dic or {}).items():
        if k is None:
            continue
        kk = " ".join(str(k).strip().split()).lower()
        norm[kk] = v

    out = {}
    for i in range(1, n_per + 1):
        candidates = [
            f"p{i}", f"potencia p{i}", f"potencia contratada p{i}",
            f"p{i} (kW)", f"pot. p{i}", f"p{i} kw",
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

@st.cache_data(show_spinner=False)
def prepara_base_excesos(
    df_consumo: pd.DataFrame,     # columnas: ['datetime','consumo'] (kWh/15min)
    df_mercado: pd.DataFrame,     # índice datetime con 'periodos_20td' / 'periodos_no20td'
    tarifa: str,
    potencias_dict: dict,         # potencias contratadas introducidas en Paso 1
):
    """
    Devuelve un DataFrame indexado por datetime (tz Europe/Madrid) con:
      - consumo_kWh
      - maximetro_kW  (consumo_kWh * 4)
      - periodo_label (P1..Pn según tarifa y BBDD mercado)
      - pot_contratada_qh_kW (vector por QH según periodo y potencias contratadas)
      - contador_tipo ('contador4,5' si P6 <= 50 kW, si no 'contador1,2,3')
    """
    # 1) Alinear consumo con mercado preservando rejilla del consumo
    from index_pt import attach_market_to_consumo
    df = attach_market_to_consumo(df_consumo, df_mercado)

    # 2) Consumo y maxímetro
    consumo_kwh = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0)
    maximetro_kw = consumo_kwh * 4.0  # kWh/15' -> kW

    # 3) Periodo por QH según tarifa
    col_p = _col_periodos_por_tarifa(tarifa)
    n_per = _num_periodos_potencia(tarifa)

    # Permite 1..n o 'P1'..'Pn'
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

    # 4) Potencias contratadas normalizadas {"P1":kW,...}
    pot_map = pd.Series(_normaliza_potencias(potencias_dict, n_per), dtype="float64")
    pot_qh = per_lbl.map(pot_map).fillna(0.0)

    # 5) Clasificación de contador según P6
    p6 = float(pot_map.get("P6", 0.0)) if n_per == 6 else float(pot_map.get("P2", 0.0))
    contador_tipo = "contador4,5" if p6 <= 50.0 else "contador1,2,3"

    # 6) DataFrame final
    out = pd.DataFrame(
        {
            "consumo_kWh": consumo_kwh.values,
            "maximetro_kW": maximetro_kw.values,
            "periodo_label": per_lbl.values,
            "pot_contratada_qh_kW": pot_qh.values,
        },
        index=df.index,
    )
    # Metadato útil
    out.attrs["contador_tipo"] = contador_tipo
    out.attrs["tarifa"] = tarifa
    out.attrs["n_periodos_potencia"] = n_per

    return out

# --- Cálculo excesos contadores 1,2,3 (no 2.0TD) ---
import numpy as np

# Coeficientes por tarifa y periodo (usa . como separador decimal)
COEF_EXCESOS_123 = {
    "3.0TD": {"P1": 3.361213, "P2": 1.776545, "P3": 0.563477,  "P4": 0.430844, "P5": 0.12188,  "P6": 0.12188},
    "6.1TD": {"P1": 3.332942, "P2": 1.762138, "P3": 0.661311,  "P4": 0.465989, "P5": 0.009852, "P6": 0.008771},
    "6.2TD": {"P1": 3.292963, "P2": 1.867568, "P3": 0.491659134,"P4": 0.299574, "P5": 0.011746, "P6": 0.010432},
    "6.3TD": {"P1": 3.099043, "P2": 1.867297, "P3": 0.608332844,"P4": 0.396461, "P5": 0.013019, "P6": 0.01146},
    "6.4TD": {"P1": 2.73262,  "P2": 1.633705, "P3": 0.396743633,"P4": 0.275773, "P5": 0.008201, "P6": 0.005465},
}

@st.cache_data(show_spinner=False)
def calcula_excesos_cont_123(base_excesos: pd.DataFrame, tarifa: str):
    """
    Contadores 1,2,3 (no 2.0TD).
    Lógica:
      1) d_qh = max(0, maximetro_kW - pot_contratada_qh_kW)
      2) D_qh = d_qh ** 2
      3) Por (Periodo, Mes): S_mes = sqrt( sum(D_qh) )
      4) Por Periodo: S = sum(S_mes de todos los meses)
      5) coste_Pi = coef(tarifa, Pi) * S[Pi]
    """
    if base_excesos is None or len(base_excesos) == 0:
        return 0.0, pd.DataFrame()

    t = (tarifa or "").upper().replace(" ", "")
    if t == "2.0TD":
        return 0.0, pd.DataFrame()

    if base_excesos.attrs.get("contador_tipo") != "contador1,2,3":
        return 0.0, pd.DataFrame()

    periodos = [f"P{i}" for i in range(1, 7)]

    max_kW = pd.to_numeric(base_excesos["maximetro_kW"], errors="coerce").fillna(0.0)
    pot_kW = pd.to_numeric(base_excesos["pot_contratada_qh_kW"], errors="coerce").fillna(0.0)
    per_lbl = base_excesos["periodo_label"].astype(str)
    mes_lbl = base_excesos.index.month  # funciona con índices tz-aware o naïve

    # 1-2) Exceso positivo y al cuadrado
    d_qh = np.maximum(max_kW - pot_kW, 0.0)
    D_qh = d_qh ** 2

    # 3) Sumatorio por (Periodo, Mes) y raíz
    sum_pm = pd.Series(D_qh).groupby([per_lbl, mes_lbl]).sum()
    S_mes = np.sqrt(sum_pm)

    # 4) Suma mensual -> S por periodo
    S = S_mes.groupby(level=0).sum()
    S = S.reindex(periodos).fillna(0.0)

    # 5) Coeficientes y coste
    coefes = COEF_EXCESOS_123.get(tarifa, COEF_EXCESOS_123.get(t, {}))
    coef_series = pd.Series({p: float(coefes.get(p, 0.0)) for p in periodos}, dtype="float64")
    coste_p = coef_series * S

    df_res = pd.DataFrame({
        "Periodo": periodos,
        "S_sum_raiz": S.values,  # S = sum_m sqrt(sum_qh d^2)
        "coef": coef_series.values,
        "coste_€": coste_p.values
    })
    total = float(coste_p.sum())
    return total, df_res

# --- Cálculo excesos contadores 4,5 (solo 3.0TD y 6.1TD) ---
COEF_EXCESOS_45 = {
    "3.0TD": {"P1": 0.168944, "P2": 0.089294, "P3": 0.028322, "P4": 0.021656, "P5": 0.006126, "P6": 0.006126},
    "6.1TD": {"P1": 0.27254,  "P2": 0.144093,"P3": 0.054076,"P4": 0.038105,"P5": 0.000806,"P6": 0.000717},
}

@st.cache_data(show_spinner=False)
def calcula_excesos_cont_45(base_excesos: pd.DataFrame, tarifa: str):
    """
    Contadores 4,5:
      1) diferencia_qh = max(0, maxímetro_kW - potencia_contratada_qh_kW)
      2) S_mes,Pi = sum(diferencia_qh) por (Periodo, Mes)
      3) coste_mes,Pi = S_mes,Pi * dias_del_mes * coef(tarifa, Pi)
      4) Resultado por Pi = sum_meses(coste_mes,Pi)
    Devuelve (total, dataframe con desglose por periodo).
    """
    if base_excesos is None or len(base_excesos) == 0:
        return 0.0, pd.DataFrame()

    t = (tarifa or "").upper().replace(" ", "")
    if t not in ("3.0TD", "6.1TD"):
        # Por especificación, solo estas dos tienen excesos para 4,5
        return 0.0, pd.DataFrame()

    if base_excesos.attrs.get("contador_tipo") != "contador4,5":
        return 0.0, pd.DataFrame()

    periodos = [f"P{i}" for i in range(1, 6 + 1)]

    # 1) Diferencia positiva por QH
    max_kW = pd.to_numeric(base_excesos["maximetro_kW"], errors="coerce").fillna(0.0)
    pot_qh = pd.to_numeric(base_excesos["pot_contratada_qh_kW"], errors="coerce").fillna(0.0)
    per_lbl = base_excesos["periodo_label"].astype(str)
    mes = base_excesos.index.month
    dias_mes = base_excesos.index.to_series().dt.days_in_month.values

    diff_pos = (max_kW - pot_qh).clip(lower=0.0)

    # 2) Sumatorio por (Periodo, Mes)
    sum_pm = pd.DataFrame({
        "Periodo": per_lbl.values,
        "Mes": mes.values,
        "diff": diff_pos.values,
        "dias_mes": dias_mes,
    }).groupby(["Periodo", "Mes"], as_index=True)["diff"].sum()

    # 3) Multiplicar por días del mes y coeficientes
    #    (rehacemos un DF para tener días por índice)
    df_pm = sum_pm.to_frame("S_mes")
    # recuperamos días del mes por índice (Periodo, Mes)
    # construimos un mapeo de días:
    dias_map = (
        pd.Series(dias_mes, index=pd.MultiIndex.from_arrays([per_lbl.values, mes.values]))
        .groupby(level=[0,1]).first()
    )
    df_pm["dias_mes"] = dias_map
    coefes = COEF_EXCESOS_45.get(t, {})
    df_pm["coef"] = df_pm.index.get_level_values(0).map(lambda p: float(coefes.get(p, 0.0)))

    df_pm["coste_mes"] = df_pm["S_mes"] * df_pm["dias_mes"] * df_pm["coef"]

    # 4) Acumular por periodo
    coste_por_periodo = df_pm.groupby(level=0)["coste_mes"].sum()
    coste_por_periodo = coste_por_periodo.reindex(periodos).fillna(0.0)

    df_res = pd.DataFrame({
        "Periodo": periodos,
        "coef": [float(coefes.get(p, 0.0)) for p in periodos],
        "coste_€": coste_por_periodo.values,
    })

    total = float(coste_por_periodo.sum())
    return total, df_res


