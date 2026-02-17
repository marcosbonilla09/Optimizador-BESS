import pandas as pd
import streamlit as st

def _coef_potencia_por_tarifa(tarifa: str) -> dict:
    """Devuelve los coeficientes €/kW/año de la tabla según tarifa."""
    t = (tarifa or "").replace(" ", "").upper()

    if t == "2.0TD":
        return {"P1": 26.931, "P2": 0.698}
    if t == "3.0TD":
        return {"P1": 19.658, "P2": 10.252, "P3": 4.263, "P4": 3.682, "P5": 2.328, "P6": 1.356}
    if t == "6.1TD":
        return {"P1": 28.792, "P2": 15.078, "P3": 6.559, "P4": 5.172, "P5": 1.933, "P6": 0.916}
    if t == "6.2TD":
        return {"P1": 19.629, "P2": 10.932, "P3": 3.575, "P4": 2.606, "P5": 1.153, "P6": 0.554}
    if t == "6.3TD":
        return {"P1": 13.200, "P2": 7.708, "P3": 2.994, "P4": 2.256, "P5": 0.921, "P6": 0.441}
    if t == "6.4TD":
        return {"P1": 7.768, "P2": 4.530, "P3": 1.385, "P4": 1.094, "P5": 0.448, "P6": 0.210}

    # Tarifa no reconocida
    return {}

def coste_atr_potencia(tarifa: str, potencias: dict) -> tuple[float, pd.DataFrame]:
    """
    Calcula el ATR del término de potencia anual (€/año):
        suma(Pi * coef_i)
    Parámetros:
      - tarifa: str (2.0TD, 3.0TD, 6.1TD, ...)
      - potencias: dict { "P1": kW, "P2": kW, ... }
    Devuelve:
      - coste_total (€)
      - tabla detalle (DataFrame)
    """
    coef = _coef_potencia_por_tarifa(tarifa)
    if not coef:
        raise ValueError(f"Tarifa no reconocida: {tarifa}")

    detalle = []
    for periodo, valor in coef.items():
        pot = float(potencias.get(periodo, 0))
        coste = pot * valor
        detalle.append({"Periodo": periodo, "Potencia (kW)": pot,
                        "Coef €/kW·año": valor, "Coste (€)": coste})

    df = pd.DataFrame(detalle)
    total = float(df["Coste (€)"].sum())

    return total, df
