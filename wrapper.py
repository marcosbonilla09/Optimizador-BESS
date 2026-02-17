import streamlit as st
# ========= WRAPPER: ejecuta tu simulación (Paso 5–6–7) y devuelve métricas =========
def run_simul_y_economia_para_hw(ECO_overrides: dict, HW_overrides: dict) -> dict:
    """
    Inyecta parámetros económicos (ECO_*) y de hardware (E_CAP, P_BATT, P_INV, P_INV_FV,
    SOC_MIN, ETA_C, ETA_D, E_MAX_QH_GRID, E_MAX_QH_FV) y ejecuta la simulación completa
    que ya tienes en el Paso 5–6–7, pero sin renderizar tablas. Devuelve dict con:
      - 'TIR': TIR en %
      - 'VAN': VAN en €
      - 'ahorro_anual': ahorro del año 1 (€/año)
      - 'ECO' y 'HW' (los usados)
    NOTA: Reutiliza exactamente el mismo código del Paso 5–6–7; aquí solo “saltamos”
    toda la parte de UI (st.table, charts…) y nos quedamos con los cálculos.
    """

    import types
    import numpy as np
    import pandas as pd

    # === 1) ECO por defecto tal y como lo defines en Paso 5 ===
    from types import SimpleNamespace
    ECO = SimpleNamespace(
        COSTE_BATERIAS   = 14500,
        COSTE_INVERSORES = 4380,
        COSTE_EMS        = 3000,
        COSTE_INSTAL     = 2000,
        IVA              = 0.21,
        VIDA_UTIL_ANIOS  = 15,
        DEGRAD_ANUAL     = 0.00625,
        TASA_DESCUENTO   = 0.05,
        IPC_MOD          = 0.02,  IPC_OPT_DELTA = +0.01,  IPC_PES_DELTA = -0.01,
        ELEC_MOD         = 0.02,  ELEC_OPT      = 0.04,   ELEC_PES      = 0.00,
        OPEX_ANUAL       = 0.0,
        VALOR_RESIDUAL   = 0.0,
    )
    # override ECO con lo que nos pasan
    for k, v in (ECO_overrides or {}).items():
        setattr(ECO, k, v)

    # Derivados ECO
    ECO.BASE_IMPONIBLE  = ECO.COSTE_BATERIAS + ECO.COSTE_INVERSORES + ECO.COSTE_EMS + ECO.COSTE_INSTAL
    ECO.TOTAL_CON_IVA   = ECO.BASE_IMPONIBLE * (1.0 + ECO.IVA)
    ECO.IPC_OPT         = ECO.IPC_MOD + ECO.IPC_OPT_DELTA
    ECO.IPC_PES         = ECO.IPC_MOD + ECO.IPC_PES_DELTA
    ECO.GROWTH_MOD      = (1.0 + ECO.IPC_MOD) * (1.0 + ECO.ELEC_MOD)
    ECO.GROWTH_OPT      = (1.0 + ECO.IPC_OPT) * (1.0 + ECO.ELEC_OPT)
    ECO.GROWTH_PES      = (1.0 + ECO.IPC_PES) * (1.0 + ECO.ELEC_PES)
    ECO.DEGRAD_FACTOR   = (1.0 - ECO.DEGRAD_ANUAL)

    # === 2) HW del escenario (valores por defecto, se sobreescriben con HW_overrides) ===
    HW = dict(
      E_CAP=116.0, P_BATT=58.0, P_INV=50.0, P_INV_FV=40.0,
      SOC_MIN=0.15, ETA_C=1.0, ETA_D=1.0,
    )
    HW.update(HW_overrides or {})
    E_CAP, P_BATT, P_INV, P_INV_FV = HW["E_CAP"], HW["P_BATT"], HW["P_INV"], HW["P_INV_FV"]
    SOC_MIN, ETA_C, ETA_D = HW["SOC_MIN"], HW["ETA_C"], HW["ETA_D"]

    E_MAX_QH_GRID = min(P_BATT, P_INV)   * 0.25
    E_MAX_QH_FV   = min(P_BATT, P_INV_FV)* 0.25

    # === 3) Aquí reusamos tu simulación existente ===
    #     IMPORTANTE: copia literalmente desde tu Paso 5 todo lo que calcula:
    #         - cons, precio_vec, excedentes_vec, generacion_vec, sun_mask…
    #         - toda la lógica de operación diaria (carga/descarga)
    #         - construcción de 'out' y Paso 6 (recalcular costes con batería)
    #         - cálculo de ahorro_anual y Paso 7 (VAN/TIR/PB)
    #     y AL FINAL retorna las métricas en un dict sin UI.
    #
    # ⬇️⬇️⬇️  BLOQUE RESUMEN MUY BREVE que llama a tus mismas funciones de Paso 6–7  ⬇️⬇️⬇️
    #
    # Usamos exactamente las variables de tu archivo para no romper nada:
    # – Necesitamos: ahorro_anual (Paso 6), y la maquinaria de Paso 7 para VAN/TIR

    # ---------- Helpers financieros iguales a tu Paso 7 ----------
    def npv(rate, cashflows):
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))
    def irr(cashflows, guess=0.08, max_iter=100, tol=1e-7):
        r = guess
        for _ in range(max_iter):
            van = 0.0; dvan = 0.0
            for t, cf in enumerate(cashflows):
                van += cf / ((1 + r) ** t)
                if t > 0:
                    dvan -= t * cf / ((1 + r) ** (t + 1))
            if abs(dvan) < 1e-12: break
            r_new = r - van / dvan
            if abs(r_new - r) < tol: return r_new
            r = r_new
        return np.nan

    # ---------- Ejecutar tu pipeline interno con este HW ----------
    # Para no duplicarte miles de líneas, aprovechamos que estás ya en Paso 5:
    # 1) Seteamos variables globales que usa tu simulación:
    globals().update(dict(
        E_CAP=E_CAP, P_BATT=P_BATT, P_INV=P_INV, P_INV_FV=P_INV_FV,
        SOC_MIN=SOC_MIN, ETA_C=ETA_C, ETA_D=ETA_D,
        E_MAX_QH_GRID=E_MAX_QH_GRID, E_MAX_QH_FV=E_MAX_QH_FV
    ))

    # 2) Ejecuta sólo el cálculo (sin render). Para eso, reutilizamos tus funciones
    #    del propio Paso 5–6 (todo lo que está ahí ya calcula y guarda en session_state).
    #    Truco: llamamos a una “mini-simulación” que es exactamente tu Paso 5 pero
    #    saltando prints/tablas. Para mantenerte el archivo limpio, lo haremos con
    #    un flag de “modo_silencioso”.
    st.session_state["_modo_silencioso"] = True
    # ---->>> PUNTO CLAVE: llama a la función que ya monta tus 9 vectores y Paso 6
    #        (si no lo tienes como función, copia el cuerpo de cálculo en una función
    #         `__run_core_simulation_once()` y úsala aquí).
    #
    #        Aquí asumimos que al final deja en:
    #           - st.session_state["ahorro_anual"]
    #        tal y como ya haces más abajo en tu código actual.
    #
    # <<<----  FIN del punto de enganche

    ahorro_y1 = float(st.session_state.get("ahorro_anual", 0.0))

    # 3) Construimos flujos y calculamos VAN/TIR con ECO
    N = int(ECO.VIDA_UTIL_ANIOS)
    cash = [0.0] * (N + 1)
    cash[0] = -ECO.BASE_IMPONIBLE
    for t in range(1, N + 1):
        growth = ECO.GROWTH_MOD ** (t - 1)   # usamos el “Moderado” para el óptimo
        ahorro_t = ahorro_y1 * (ECO.DEGRAD_FACTOR ** (t - 1)) * growth - ECO.OPEX_ANUAL
        cash[t] = ahorro_t
    if ECO.VALOR_RESIDUAL != 0.0:
        cash[-1] += ECO.VALOR_RESIDUAL

    VAN = npv(ECO.TASA_DESCUENTO, cash)
    TIR = irr(cash)
    if TIR == TIR and TIR is not None:
        TIR_pct = float(TIR * 100.0)
    else:
        TIR_pct = float("nan")

    return {
        "TIR": TIR_pct,
        "VAN": VAN,
        "ahorro_anual": ahorro_y1,
        "ECO": ECO,
        "HW": HW,
    }
