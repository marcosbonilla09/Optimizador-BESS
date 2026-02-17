from __future__ import annotations

import streamlit as st
import pandas as pd
import calendar
import numpy as np
import streamlit.components.v1 as components

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- Aceleradores globales ---
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# Numba opcional (no rompe si no está instalada)
try:
    from numba import njit
    NUMBA_ON = True
except Exception:
    NUMBA_ON = False

def _jit(*args, **kwargs):
    if NUMBA_ON:
        return njit(*args, **kwargs)
    def deco(fn):  # fallback no-op
        return fn
    return deco

def fmt_eur(valor, dec=2):
    """Formatea número con coma decimal y punto de miles, estilo europeo."""
    return f"{valor:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


st.set_page_config(page_title="Optimizador BESS – TFG Marcos", layout="centered")

# === LOGIN  ===
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    st.markdown(
        """
        <style>
        .login-wrapper {
            margin: auto;
            max-width: 380px;   /* MÁS ESTRECHO */
            padding-top: 30px;
        }

        .login-header {
            background-color: #0a1a3a;
            padding: 14px;
            border-radius: 14px;
            text-align: center;
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            margin-bottom: 22px;
        }

        .login-button {
            background-color: #0a1a3a;
            color: white;
            padding: 8px 20px;
            border-radius: 8px;
            border: 1px solid #0a1a3a;
            cursor: pointer;
            font-size: 0.9rem;
        }

        .login-button:hover {
            background-color: #10224d;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)

    # Título dentro de la caja azul
    st.markdown(
        '<div class="login-header">Acceso optimizador BESS</div>',
        unsafe_allow_html=True
    )

    usuario = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    # ---- BOTÓN PEQUEÑO A LA IZQUIERDA ----
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        entrar = st.button("Entrar", key="login_btn", help="Iniciar sesión")

    try:
        USUARIO = st.secrets["USER"]
        PASSWORD = st.secrets["PASS"]
    except Exception:
        # Fallback para desarrollo local si no hay secrets
        USUARIO = "msenergy"
        PASSWORD = "2026"

    if entrar:
        if usuario == USUARIO and password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos")

    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

check_login()

# =========================
# NAV LATERAL (POST-LOGIN)
# =========================
if "page" not in st.session_state:
    st.session_state["page"] = "Optimizador"

with st.sidebar:
    st.markdown("### Menú")
    st.session_state["page"] = st.radio(
        "Selecciona una funcionalidad:",
        ["Optimizador", "Evaluador de soluciones"],
        index=0 if st.session_state["page"] == "Optimizador" else 1,
        key="page_radio",)

from types import SimpleNamespace

def _prepare_base_signals_from_session_state():
    """
    Calcula y deja listos los vectores globales que usa simular_bess_milp():
    cons, load_vec, generacion_vec, excedentes_vec, precio_vec, atr_vec,
    pot_contratada, precio_venta_fv_vec, base_dt_index
    """
    import numpy as np
    import pandas as pd
    import streamlit as st

    # Necesario porque simular_bess_milp() usa estas variables como globales
    global cons, load_vec, generacion_vec, excedentes_vec, precio_vec, atr_vec
    global pot_contratada, precio_venta_fv_vec, base_dt_index

    # ---------- Señales base necesarias del Paso 1–4 ----------
    if "consumo" not in st.session_state:
        st.error("Falta el consumo del Paso 1.")
        st.stop()
    if "precio_qh_eur_mwh" not in st.session_state:
        st.error("Falta el precio por QH del Paso 4.")
        st.stop()
    if "market" not in st.session_state:
        st.error("Falta la base de mercado en memoria (Paso 4).")
        st.stop()

    # Consumo alineado
    df_c = st.session_state["consumo"].copy()
    df_c["datetime"] = to_naive_utc_index(df_c["datetime"])
    df_c = df_c.dropna(subset=["datetime"]).copy()

    cons = pd.to_numeric(df_c["consumo"], errors="coerce").fillna(0.0).values
    base_dt_index = pd.DatetimeIndex(df_c["datetime"])
    n_slots = len(base_dt_index)

    # Precio €/MWh por QH alineado
    s_pre = st.session_state["precio_qh_eur_mwh"]
    if isinstance(s_pre, pd.Series):
        s_pre = s_pre.rename("precio_eur_mwh").copy()
        s_pre.index = to_naive_utc_index(s_pre.index)
    else:
        s_pre = s_pre.squeeze().rename("precio_eur_mwh")
        s_pre.index = to_naive_utc_index(s_pre.index)

    precios_uni = s_pre.to_frame().loc[~s_pre.index.duplicated(keep="last")]
    precio_vec = (
        df_c.merge(precios_uni, left_on="datetime", right_index=True, how="left")["precio_eur_mwh"]
        .ffill().bfill().to_numpy()
    )

    # ATR energía €/MWh por QH
    s_atr = st.session_state.get("atr_qh_eur_mwh", None)
    if s_atr is not None:
        if isinstance(s_atr, pd.Series):
            s_atr = s_atr.rename("atr_qh_eur_mwh").copy()
            s_atr.index = to_naive_utc_index(s_atr.index)
        else:
            s_atr = s_atr.squeeze().rename("atr_qh_eur_mwh")
            s_atr.index = to_naive_utc_index(s_atr.index)

        atr_uni = s_atr.to_frame().loc[~s_atr.index.duplicated(keep="last")]
        atr_vec = (
            df_c.merge(atr_uni, left_on="datetime", right_index=True, how="left")["atr_qh_eur_mwh"]
            .fillna(0.0).to_numpy()
        )
    else:
        atr_vec = np.zeros(n_slots)

    # Excedentes / Generación (si hay FV)
    if st.session_state.get("fv") == "Sí" and "excedentes" in st.session_state and "fecha_exc" in st.session_state:
        df_exc = pd.DataFrame({
            "datetime": to_naive_utc_index(st.session_state["fecha_exc"]),
            "exc": pd.to_numeric(st.session_state["excedentes"], errors="coerce")
        }).dropna()
        excedentes_vec = df_c[["datetime"]].merge(df_exc, on="datetime", how="left")["exc"].fillna(0.0).to_numpy()
    else:
        excedentes_vec = np.zeros(n_slots)

    if st.session_state.get("fv") == "Sí" and "generacion" in st.session_state and "fecha_gen" in st.session_state:
        df_gen = pd.DataFrame({
            "datetime": to_naive_utc_index(st.session_state["fecha_gen"]),
            "gen": pd.to_numeric(st.session_state["generacion"], errors="coerce")
        }).dropna()
        generacion_vec = df_c[["datetime"]].merge(df_gen, on="datetime", how="left")["gen"].fillna(0.0).to_numpy()
    else:
        generacion_vec = np.zeros(n_slots)

    autoconsumo_sin_bess = np.maximum(0.0, generacion_vec - excedentes_vec)
    load_vec = cons + autoconsumo_sin_bess

    # Potencias contratadas por QH
    pot_dict = st.session_state.get("potencias") or {}
    tarifa = (st.session_state.get("tarifa") or "3.0TD").replace(" ", "").upper()
    df_aux = st.session_state.get("consumo_con_mercado")

    if df_aux is None:
        pot_contratada = np.full(n_slots, float(pot_dict.get("P1", 0.0)))
    else:
        dfp_left = df_c.copy()
        dfp_left["datetime"] = to_naive_utc_index(dfp_left["datetime"])
        df_aux = df_aux.copy()

        if "datetime" in df_aux.columns:
            df_aux["datetime"] = to_naive_utc_index(df_aux["datetime"])
            dfp = dfp_left.set_index("datetime").join(df_aux.set_index("datetime"), how="left")
        else:
            df_aux.index = to_naive_utc_index(df_aux.index)
            dfp = dfp_left.set_index("datetime").join(df_aux, how="left")

        col_p = "periodos_20td" if tarifa == "2.0TD" else "periodos_no20td"
        per_num = pd.to_numeric(dfp[col_p], errors="coerce")
        miss = per_num.isna()
        if miss.any():
            per_num.loc[miss] = pd.to_numeric(
                dfp.loc[miss, col_p].astype(str).str.upper().str.extract(r"(\d+)")[0],
                errors="coerce"
            )
        n_per = 3 if tarifa == "2.0TD" else 6
        per_num = per_num.fillna(1).astype(int).clip(1, n_per)
        per_lbl = "P" + per_num.astype(str)
        pot_contratada = per_lbl.map(lambda p: float(pot_dict.get(p, 0.0))).values

    # Precio de venta FV
    if st.session_state.get("fv") == "Sí":
        modo_fv = str(st.session_state.get("modalidad_fv", ""))
        if modo_fv == "Precio fijo":
            precio_fv = float(st.session_state.get("precio_fv", {}).get("Precio FV", 0.0))
            precio_venta_fv_vec = np.full(n_slots, precio_fv, dtype=float)
        elif modo_fv == "Indexado":
            cg_fv = float(st.session_state.get("cg_fv", {}).get("Costes Gestion FV", 0.0))
            pv = (precio_vec - cg_fv)
            pv[pv < 0] = 0.0
            precio_venta_fv_vec = pv
        else:
            precio_venta_fv_vec = np.zeros(n_slots)
    else:
        precio_venta_fv_vec = np.zeros(n_slots)

    return n_slots

def _eta_to_frac(x):
    try:
        xv = float(x)
        if xv > 1.0: xv /= 100.0
        return max(0.0, min(1.0, xv))
    except Exception:
        return 1.0

def _fv_power(x):
    try:
        xv = float(x)
        return xv if np.isfinite(xv) and xv > 0 else 0.0
    except Exception:
        return 0.0

def _build_combo_namespace(inv_rows, counts):
    """
    'Inversor sintético' para que tu evaluar_combinacion no cambie:
    suma potencias (P_inv, P_fv), suma precio, y usa eta_i mínima (cuello de botella).
    """
    total_P_inv = 0.0
    total_P_fv  = 0.0
    total_precio = 0.0
    eta_list = []
    name_parts = []

    for row, c in zip(inv_rows, counts):
        if c <= 0:
            continue
        p = float(row.P_inv)
        pfv = _fv_power(getattr(row, "P_fv", 0.0))
        precio = float(getattr(row, "precio", 0.0))
        total_P_inv += c * p
        total_P_fv  += c * pfv
        total_precio += c * precio
        eta_list.append(_eta_to_frac(getattr(row, "eta_i", 1.0)))
        name_parts.append(f"{c}x {row.marca} {row.modelo}")

    if not eta_list:
        eta_list = [1.0]

    return SimpleNamespace(
    marca=" + ".join(name_parts) if name_parts else "(mix)",
    modelo="(mix)",
    P_inv=float(total_P_inv),
    P_fv=float(total_P_fv),
    precio=float(total_precio),
    eta_i=float(min(eta_list)),
    Vmin=None, Vmax=None,
    fase=getattr(inv_rows[0], "fase", None) if inv_rows else None,
    n_units=int(sum(counts)),)

def as_eta(x):
    if pd.isna(x): return 1.0
    x = float(x)
    if x > 1.0: x /= 100.0
    return float(np.clip(x, 0.0, 1.0))

from ortools.linear_solver import pywraplp

def simular_bess_milp(HW):
    """
    Optimización LP (GLOP) en 2 etapas (lexicográfica):
      - Etapa 1: minimizar excesos de potencia (ponderados por mes caro/medio/barato).
      - Etapa 2: minimizar coste energético manteniendo excesos mínimos.

    Degradación (Opción A):
      - NO hay coste de degradación en la función objetivo.
      - Se impone un presupuesto anual de ciclos equivalentes (EFC) con:
            sum(descarga_kWh) <= E_CAP * N_CICLOS_EQ_MAX
    """

    # Datos base del año (ya calculados al inicio del Paso 5)
    global cons, load_vec, generacion_vec, excedentes_vec, precio_vec, atr_vec, pot_contratada, precio_venta_fv_vec, base_dt_index
    T = len(base_dt_index)

    # --- Información mensual para la penalización de excesos (aprox. lineal) ---
    meses = base_dt_index.month.values
    meses_unicos = sorted(set(int(m) for m in meses))

    # Grupos de meses según tu tabla
    meses_caros        = {1, 2, 7, 12}         # Ene, Feb, Jul, Dic
    meses_intermedios  = {3, 6, 8, 9, 11}      # Mar, Jun, Ago, Sep, Nov
    meses_baratos      = {4, 5, 10}            # Abr, May, Oct

    # Antes eran "€/kW" simplificados; ahora los usamos como PESOS (prioridad/riesgo), no como euros reales.
    coef_exceso_mes = {}
    for m in meses_unicos:
        if m in meses_caros:
            coef_exceso_mes[m] = 1.72724817
        elif m in meses_intermedios:
            coef_exceso_mes[m] = 0.2980005
        elif m in meses_baratos:
            coef_exceso_mes[m] = 0.19320267
        else:
            coef_exceso_mes[m] = 0.0

    # --- Hardware / parámetros ---
    E_CAP    = float(HW["E_CAP"])
    P_BATT   = float(HW["P_BATT"])
    P_INV    = float(HW["P_INV"])
    P_INV_FV = float(HW.get("P_INV_FV", 0.0))
    SOC_MIN  = float(HW["SOC_MIN"])
    ETA_C    = float(HW["ETA_C"])
    ETA_D    = float(HW["ETA_D"])

    # (Compat) antes tenías C_DEG_KWH en la FO; ahora NO se usa
    _C_DEG_KWH_UNUSED = float(HW.get("C_DEG_KWH", 0.0))

    # ✅ Degradación (Opción A): presupuesto anual de ciclos equivalentes (EFC)
    N_CICLOS_EQ_MAX = float(HW.get("N_CICLOS_EQ_MAX", 1e9))  # [ciclos eq/año]; si no se define, no limita
    EPS_EXCESOS = float(HW.get("EPS_EXCESOS", 1e-6))          # tolerancia para fijar excesos entre etapas

    E_MIN       = SOC_MIN * E_CAP
    E_MAX_QH    = min(P_BATT, P_INV) * 0.25    # [kWh/QH]
    E_MAX_QH_FV = min(P_BATT, P_INV_FV) * 0.25 if P_INV_FV > 0 else E_MAX_QH

    # ----- Crear solver LP -----
    solver = pywraplp.Solver.CreateSolver("GLOP")  # LP continuo (rápido)
    if solver is None:
        raise RuntimeError("No se pudo crear el solver OR-Tools (GLOP).")

    # ----- Variables -----
    e        = [solver.NumVar(E_MIN, E_CAP, f"e_{t}") for t in range(T)]
    c_grid   = [solver.NumVar(0.0, E_MAX_QH, f"c_grid_{t}") for t in range(T)]
    c_pv     = [solver.NumVar(0.0, E_MAX_QH_FV, f"c_pv_{t}") for t in range(T)]
    d        = [solver.NumVar(0.0, E_MAX_QH, f"d_{t}") for t in range(T)]
    g_load   = [solver.NumVar(0.0, solver.infinity(), f"g_load_{t}") for t in range(T)]
    pv_load  = [solver.NumVar(0.0, solver.infinity(), f"pv_load_{t}") for t in range(T)]
    pv_export = [solver.NumVar(0.0, solver.infinity(), f"pv_export_{t}") for t in range(T)]
    p_grid   = [solver.NumVar(0.0, solver.infinity(), f"p_grid_{t}") for t in range(T)]
    z_mes    = {m: solver.NumVar(0.0, solver.infinity(), f"z_exceso_mes_{m}") for m in meses_unicos}

    # ----- Restricciones -----

    # 1) Dinámica de batería
    # t = 0: e_0 = E_MIN + eta_c*(c0) - d0/eta_d
    cons0 = solver.Constraint(E_MIN, E_MIN)
    cons0.SetCoefficient(e[0], 1.0)
    cons0.SetCoefficient(c_grid[0], -ETA_C)
    cons0.SetCoefficient(c_pv[0], -ETA_C)
    cons0.SetCoefficient(d[0], 1.0 / ETA_D)

    for t in range(1, T):
        ct = solver.Constraint(0.0, 0.0)
        # e_t - e_{t-1} - eta_c*(c_grid+c_pv) + d/eta_d = 0
        ct.SetCoefficient(e[t], 1.0)
        ct.SetCoefficient(e[t - 1], -1.0)
        ct.SetCoefficient(c_grid[t], -ETA_C)
        ct.SetCoefficient(c_pv[t], -ETA_C)
        ct.SetCoefficient(d[t], 1.0 / ETA_D)

    # 1b) Condición SOC final (evita vaciar la batería al final del año)
    ct_end = solver.Constraint(0.0, solver.infinity())  # e[T-1] - e[0] >= 0
    ct_end.SetCoefficient(e[T - 1], 1.0)
    ct_end.SetCoefficient(e[0], -1.0)

    # 2) Balance de carga: load_t = g_load_t + pv_load_t + d_t
    for t in range(T):
        ct = solver.Constraint(float(load_vec[t]), float(load_vec[t]))  # igualdad
        ct.SetCoefficient(g_load[t], 1.0)
        ct.SetCoefficient(pv_load[t], 1.0)
        ct.SetCoefficient(d[t], 1.0)

    # 2b) Definición de potencia importada de red [kW] en cada QH:
    # p_grid_t = 4 * (g_load_t + c_grid_t)
    for t in range(T):
        ct_pg = solver.Constraint(0.0, 0.0)  # igualdad
        ct_pg.SetCoefficient(p_grid[t], 1.0)
        ct_pg.SetCoefficient(g_load[t], -4.0)
        ct_pg.SetCoefficient(c_grid[t], -4.0)

    # 3) Balance FV: gen_t = pv_load_t + c_pv_t + pv_export_t
    for t in range(T):
        ct = solver.Constraint(float(generacion_vec[t]), float(generacion_vec[t]))
        ct.SetCoefficient(pv_load[t], 1.0)
        ct.SetCoefficient(c_pv[t], 1.0)
        ct.SetCoefficient(pv_export[t], 1.0)

    # 3b) Carga batería desde FV limitada a excedentes sin BESS
    for t in range(T):
        ct = solver.Constraint(0.0, float(excedentes_vec[t]))
        ct.SetCoefficient(c_pv[t], 1.0)

    # 3c) Evitar simultaneidad carga/descarga (relajación física)
    # d_t + c_grid_t + c_pv_t <= E_MAX_QH
    for t in range(T):
        ct_sync = solver.Constraint(-solver.infinity(), E_MAX_QH)
        ct_sync.SetCoefficient(d[t], 1.0)
        ct_sync.SetCoefficient(c_grid[t], 1.0)
        ct_sync.SetCoefficient(c_pv[t], 1.0)

    # 3d) El vertido FV no puede superar los excedentes reales sin BESS
    for t in range(T):
        ct_pvexp = solver.Constraint(0.0, float(excedentes_vec[t]))
        ct_pvexp.SetCoefficient(pv_export[t], 1.0)

    # 4) Potencia contratada: la batería NO puede aumentar la potencia de red respecto a:
    #    - la potencia contratada si el consumo base está por debajo
    #    - el propio consumo base si ya está por encima (no empeorar excesos)
    for t in range(T):
        P_cons_t  = 4.0 * float(cons[t])          # kW sin BESS
        P_contr_t = float(pot_contratada[t])      # kW contratada (SIN tolerancias)
        P_cap_t   = max(P_cons_t, P_contr_t)      # techo "no empeorar"

        ct = solver.Constraint(-solver.infinity(), float(P_cap_t))
        ct.SetCoefficient(p_grid[t], 1.0)

    # 4b) Enlace con exceso mensual simplificado:
    # z_mes[m] >= p_grid_t - Pcontr_t
    for t in range(T):
        m = int(meses[t])
        P_lim_t = float(pot_contratada[t])

        ct_z = solver.Constraint(-solver.infinity(), P_lim_t)  # p_grid_t - z_mes[m] <= P_lim_t
        ct_z.SetCoefficient(p_grid[t], 1.0)
        ct_z.SetCoefficient(z_mes[m], -1.0)

    # 5) Potencia máx. batería (carga total y descarga)
    for t in range(T):
        # c_grid + c_pv <= E_MAX_QH
        ct_c = solver.Constraint(-solver.infinity(), E_MAX_QH)
        ct_c.SetCoefficient(c_grid[t], 1.0)
        ct_c.SetCoefficient(c_pv[t], 1.0)
        # d_t <= E_MAX_QH ya está en la cota superior de d[t]

    # ============================================================
    # ✅ Degradación: limitar ciclos equivalentes anuales
    #     EFC ≈ sum(descarga_kWh) / E_CAP
    #     => sum(descarga_kWh) <= E_CAP * N_CICLOS_EQ_MAX
    # ============================================================
    if np.isfinite(N_CICLOS_EQ_MAX) and N_CICLOS_EQ_MAX > 0:
        ct_ciclos = solver.Constraint(-solver.infinity(), float(E_CAP) * float(N_CICLOS_EQ_MAX))
        for t in range(T):
            ct_ciclos.SetCoefficient(d[t], 1.0)

    # ============================================================
    # ✅ Optimización en 2 etapas (lexicográfica)
    # ============================================================
    objective = solver.Objective()
    objective.SetMinimization()

    # ---- Etapa 1: minimizar excesos (ponderación por mes) ----
    objective.Clear()
    for m, z in z_mes.items():
        w = float(coef_exceso_mes.get(m, 0.0))
        if w > 0.0:
            objective.SetCoefficient(z, w)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Etapa 1 (excesos) no encontró óptimo (status={status}).")

    Z_star = float(objective.Value())

    # Fijar excesos mínimos encontrados (con una tolerancia muy pequeña)
    ct_fixZ = solver.Constraint(-solver.infinity(), Z_star + float(EPS_EXCESOS))
    for m, z in z_mes.items():
        w = float(coef_exceso_mes.get(m, 0.0))
        if w > 0.0:
            ct_fixZ.SetCoefficient(z, w)

    # ---- Etapa 2: minimizar coste de energía (sin degradación, sin coste de excesos) ----
    objective.Clear()
    modalidad = st.session_state.get("modalidad")

    for t in range(T):
        # Precio base término energía (OMIE + desvíos + CG, etc.)
        precio_compra = float(precio_vec[t])  # €/MWh

        # Indexado pass through: añadir ATR energía [€/MWh] por QH
        if modalidad == "Indexado pass through":
            precio_compra += float(atr_vec[t])

        # Pasar a €/kWh
        coef_buy  = precio_compra / 1000.0
        coef_sell = float(precio_venta_fv_vec[t]) / 1000.0

        # Coste por energía comprada de red (directa a consumo + carga batería)
        objective.SetCoefficient(g_load[t], coef_buy)
        objective.SetCoefficient(c_grid[t], coef_buy)

        # Ingreso por excedentes FV
        objective.SetCoefficient(pv_export[t], -coef_sell)

    objective.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Etapa 2 (energía) no encontró óptimo (status={status}).")

    # ----- Reconstruir DataFrame con el mismo formato que simular_bess -----
    carga_red_kWh    = np.array([c_grid[t].solution_value() for t in range(T)])
    carga_exc_kWh    = np.array([c_pv[t].solution_value() for t in range(T)])
    descarga_kWh     = np.array([d[t].solution_value() for t in range(T)])
    energia_kWh      = np.array([e[t].solution_value() for t in range(T)])
    soc_pu           = energia_kWh / E_CAP
    cons_red_pro_kWh = np.array([g_load[t].solution_value() + c_grid[t].solution_value() for t in range(T)])
    vertido_kWh      = np.array([pv_export[t].solution_value() for t in range(T)])
    ingreso_vertido_eur = vertido_kWh * (precio_venta_fv_vec / 1000.0)
    maximetro_kW     = cons_red_pro_kWh * 4.0

    out = pd.DataFrame({
        "datetime": base_dt_index.values,
        "load_kWh": load_vec,
        "cons_red_pro_kWh": cons_red_pro_kWh,
        "excedentes_kWh": excedentes_vec,  # base para referencia
        "precio_eur_mwh": precio_vec,
        "carga_red_kWh": carga_red_kWh,
        "carga_exc_kWh": carga_exc_kWh,
        "carga_kWh": carga_red_kWh + carga_exc_kWh,
        "descarga_kWh": descarga_kWh,
        "energia_almacenada_kWh": energia_kWh,
        "soc_pu": soc_pu,
        "maximetro_kW": maximetro_kW,
        "vertido_kWh": vertido_kWh,
        "ingreso_vertido_eur": ingreso_vertido_eur,
        "pot_contratada_kW": pot_contratada,
        "fv_gen_kWh": generacion_vec,
        "autoconsumo_kWh": np.maximum(0.0, generacion_vec - vertido_kWh - carga_exc_kWh),
    })
    return out

def to_naive_utc_index(x):
    idx = pd.DatetimeIndex(pd.to_datetime(x, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def npv(rate, cashflows):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def irr(cashflows, guess=0.08, max_iter=100, tol=1e-7):
    r = guess
    for _ in range(max_iter):
        van = 0.0; dvan = 0.0
        for t, cf in enumerate(cashflows):
            van += cf / ((1 + r) ** t)
            if t > 0: dvan -= t * cf / ((1 + r) ** (t + 1))
        if abs(dvan) < 1e-12: break
        r_new = r - van / dvan
        if abs(r_new - r) < tol: return r_new
        r = r_new
    return np.nan

def evaluar_economia(ECO: SimpleNamespace, df_sim: pd.DataFrame, ctx: dict) -> dict:
    """
    Replica la valoración del Paso 4 pero con el consumo post-batería:
    - Consumo: df_sim['cons_red_pro_kWh']
    - Maxímetro/excesos: a partir de ese consumo
    - Vertidos FV: df_sim['vertido_kWh']
    - TE neto: igual criterio que en Paso 4
    Devuelve ahorro, VAN, TIR y desglose 'det_bess'. (La 'situación inicial' ya está en session_state.)
    """
    # ------------- Entradas post-batería -------------
    df_consumo_bess = (
        df_sim[["datetime", "cons_red_pro_kWh"]]
        .rename(columns={"cons_red_pro_kWh": "consumo"})
        .copy()
    )
    df_vertido_bess = (
        df_sim[["datetime", "vertido_kWh"]]
        .rename(columns={"vertido_kWh": "excedentes_kWh"})
        .copy()
    )

    tarifa    = ctx.get("tarifa", "3.0TD")
    modalidad = ctx.get("modalidad")
    market    = ctx["market"]  # viene del Paso 4

    # ------------- Costes regulados y TE (después) -------------
    # ATR energía (se recalcula con el nuevo consumo)
    from index_pt import _leer_desvios_y_cg, coste_passthrough, coste_atr_energia
    atr_eur_bess, _, _ = coste_atr_energia(df_consumo=df_consumo_bess, df_mercado=market, tarifa=tarifa)

    # --- TE (Después): EXACTO al Paso 4, cambiando solo el consumo por red_pro ---
    s_pre = ctx.get("precio_qh_eur_mwh")
    if s_pre is None:
        te_total_bess = 0.0
    else:
        # 1) Serie de precios a DataFrame
        if isinstance(s_pre, pd.Series):
            pre_df = s_pre.rename("precio_eur_mwh").to_frame()
        else:
            pre_df = pd.Series(s_pre, name="precio_eur_mwh").to_frame()

        # 2) Unificar zonas horarias → NAIVE UTC (igual que la simulación)
        pre_df.index = to_naive_utc_index(pre_df.index)
        df_consumo_bess["datetime"] = to_naive_utc_index(df_consumo_bess["datetime"])

        # 3) Alinear por datetime y multiplicar (kWh / 1000) * €/MWh
        df_b = df_consumo_bess.merge(pre_df, left_on="datetime", right_index=True, how="left")\
                            .dropna(subset=["precio_eur_mwh"])
        df_b["coste_qh"] = (pd.to_numeric(df_b["consumo"], errors="coerce")/1000.0) * \
                            pd.to_numeric(df_b["precio_eur_mwh"], errors="coerce")
        te_total_bess = float(df_b["coste_qh"].sum())

    # Igual que en Paso 4: PT = TE puro; resto = TE neto (total - ATR energía)
    modalidad = ctx.get("modalidad")
    if modalidad == "Indexado pass through":
        te_neto_bess = te_total_bess
    else:
        te_neto_bess = te_total_bess - float(atr_eur_bess)

    # ATR potencia (no cambia con la batería: usa el mismo valor del Paso 4)
    atr_pot_bess = float(ctx.get("atr_total_potencia", 0.0))

    # FNEE (se recalcula con el nuevo consumo)
    from fnee import coste_fnee
    fnee_bess, _ = coste_fnee(df_consumo_bess)

    # ------------- Excesos de potencia (después) -------------
    from excesos_pot import prepara_base_excesos, calcula_excesos_cont_123, calcula_excesos_cont_45
    try:
        base_excesos_bess = prepara_base_excesos(
            df_consumo=df_consumo_bess,
            df_mercado=market,
            tarifa=tarifa,
            potencias_dict=ctx.get("potencias", st.session_state.get("potencias", {}))
        )
        tipo_cont_bess = base_excesos_bess.attrs.get("contador_tipo")
        if tarifa != "2.0TD":
            if tipo_cont_bess == "contador1,2,3":
                exc_bess, _ = calcula_excesos_cont_123(base_excesos_bess, tarifa=tarifa)
            else:
                exc_bess, _ = calcula_excesos_cont_45(base_excesos_bess, tarifa=tarifa)
        else:
            exc_bess = 0.0
    except Exception:
        exc_bess = 0.0

    # ------------- Ingresos FV (después) -------------
    ingresos_fv_bess = 0.0
    if ctx.get("fv_flag"):
        modo_fv = ctx.get("modalidad_fv")
        if modo_fv == "Precio fijo":
            from fv_fijo import coste_excedentes_fijo
            ingresos_fv_bess, _, _ = coste_excedentes_fijo(
                df_excedentes=df_vertido_bess,
                precio_excedentes=ctx.get("precio_fv", st.session_state.get("precio_fv", 0.0))
            )
        elif modo_fv == "Indexado":
            from fv_indexado import coste_excedentes_indexado
            ingresos_fv_bess, _ = coste_excedentes_indexado(
                df_excedentes=df_vertido_bess,
                df_mercado=market,
                cg_fv=ctx.get("cg_fv", st.session_state.get("cg_fv", 0.0))
            )
    # 🔒 Normalizar signo: siempre positivo (los restamos más abajo)
    try:
        ingresos_fv_bess = float(ingresos_fv_bess)
        if ingresos_fv_bess < 0:
            ingresos_fv_bess = -ingresos_fv_bess
    except Exception:
        ingresos_fv_bess = 0.0

    # ------------- Impuesto eléctrico e importe total (después) -------------
    # Mismas bases que en Paso 4: TE_neto + ATR_e + ATR_p + Excesos + FNEE − Ingresos_FV
    iee_pct = 5.11269 / 100.0
    base_iee_bess = (float(te_neto_bess) + float(atr_eur_bess) + float(atr_pot_bess) + float(exc_bess) + float(fnee_bess)) - float(ingresos_fv_bess)
    iee_bess = base_iee_bess * iee_pct
    total_bess = base_iee_bess + iee_bess

    # ------------- Ahorro vs situación inicial (Paso 4) -------------
    # En el Paso 4 ya guardaste el total (te_neto0 + ATR_e0 + ATR_p0 + Excesos0 + FNEE0 − FV0 + IEE0)
    total_base = float(ctx.get("costes_iniciales_total", 0.0))
    ahorro_anual = total_base - float(total_bess)

    # ------------- VAN / TIR -------------
    # cash[0] = inversión (sin IVA si así lo usas en tu base) / resto: ahorros con degradación, IPC, etc.
    cash = [0.0] * (int(ECO.VIDA_UTIL_ANIOS) + 1)
    cash[0] = -float(ECO.BASE_IMPONIBLE)
    for t in range(1, len(cash)):
        growth = (1 + float(ECO.IPC_MOD)) * (1 + float(ECO.ELEC_MOD))
        ahorro_t = (float(ahorro_anual) * ((1 - float(ECO.DEGRAD_ANUAL)) ** (t - 1)) * (growth ** (t - 1))) - float(ECO.OPEX_ANUAL)
        cash[t] = ahorro_t
    cash[-1] += float(ECO.VALOR_RESIDUAL)

    # Usa tus utilidades npv/irr ya definidas en el script
    VAN = npv(float(ECO.TASA_DESCUENTO), cash)
    TIR = irr(cash)
    TIR_pct = float(TIR * 100) if np.isfinite(TIR) else np.nan

    # ------------- Desglose para auditoría -------------
    det_bess = dict(
        TE=te_neto_bess,                # NETO, mismo criterio que Paso 4
        ATR_energia=atr_eur_bess,
        ATR_potencia=atr_pot_bess,
        Excesos=exc_bess,
        FNEE=fnee_bess,
        IEE=iee_bess,
        Ingresos_FV=-ingresos_fv_bess,  # signo negativo en la base
        Total=total_bess
    )

    return {
        "ahorro_anual": float(ahorro_anual),
        "VAN": float(VAN),
        "TIR": TIR_pct,
        "det_bess": det_bess
    }

# ---------- Evaluar UNA combinación (batería b_row + n inversores inv_row) ----------
def evaluar_combinacion(b_row, inv_row, n_inv: int, ctx: dict, return_sim: bool = False) -> dict:
    # ECO (igual que ahora)
    ECO = SimpleNamespace(
        COSTE_BATERIAS   = float(b_row.precio),
        COSTE_INVERSORES = float(inv_row.precio),
        COSTE_EMS        = 3000.0,
        COSTE_INSTAL     = 2000.0,
        IVA              = 0.21,
        VIDA_UTIL_ANIOS  = 10,
        DEGRAD_ANUAL     = 0.00625,
        TASA_DESCUENTO   = 0.05,
        IPC_MOD          = 0.029,
        IPC_OPT_DELTA    = +0.005,
        IPC_PES_DELTA    = -0.01,
        ELEC_MOD         = 0.01,
        ELEC_OPT         = 0.02,
        ELEC_PES         = 0.00,
        OPEX_ANUAL       = 0.0,
        VALOR_RESIDUAL   = 0.0,
    )
    ECO.BASE_IMPONIBLE = ECO.COSTE_BATERIAS + ECO.COSTE_INVERSORES + ECO.COSTE_EMS + ECO.COSTE_INSTAL
    ECO.TOTAL_CON_IVA  = ECO.BASE_IMPONIBLE * (1 + ECO.IVA)

    N_CICLOS_VIDA = 8000.0
    coste_degrad_kWh = ECO.BASE_IMPONIBLE / (float(b_row.cap_kWh) * N_CICLOS_VIDA) 

    # HW (igual que ahora)
    DoD = float(pd.to_numeric(b_row.DoD, errors="coerce"))
    if not np.isfinite(DoD): raise ValueError("DoD vacío o inválido")
    if DoD > 1.0: DoD /= 100.0
    DoD = float(np.clip(DoD, 0.0, 1.0))
    SOC_MIN = 1.0 - DoD

    eta_b = as_eta(b_row.eta_b)
    eta_i = as_eta(inv_row.eta_i)
    eta   = min(eta_b, eta_i)

    HW = dict(
        E_CAP=float(b_row.cap_kWh),
        SOC_MIN=SOC_MIN,
        P_BATT=float(b_row.P_batt),
        P_INV=float(inv_row.P_inv),
        P_INV_FV=float(getattr(inv_row, "P_fv", 0.0)),
        ETA_C=float(eta), ETA_D=float(eta),
    )
    HW["C_DEG_KWH"] = float(coste_degrad_kWh)
    for k in ["E_CAP","P_BATT","P_INV","ETA_C","ETA_D","SOC_MIN"]:
        if not np.isfinite(HW[k]):
            raise ValueError(f"HW inválido: {k}={HW[k]}")
    if HW["E_CAP"] <= 0:      raise ValueError("Capacidad E_CAP debe ser > 0")
    if HW["P_BATT"] <= 0:     raise ValueError("Potencia batería P_BATT debe ser > 0")
    if HW["P_INV"]  <= 0:     raise ValueError("Potencia inversor P_INV debe ser > 0")
    if not (0 <= HW["SOC_MIN"] < 1): raise ValueError(f"SOC_MIN fuera de [0,1): {HW['SOC_MIN']}")
    if not (0 < HW["ETA_C"] <= 1 and 0 < HW["ETA_D"] <= 1):
        raise ValueError(f"Eficiencias fuera de (0,1]: ETA_C={HW['ETA_C']}, ETA_D={HW['ETA_D']}")

    # Simulación + economía (igual que ahora)
    df_sim = simular_bess_milp(HW)
    eco = evaluar_economia(ECO, df_sim, ctx)

    # --- resumen ligero para comparar combinaciones (sin devolver el DF completo) --- BORRRRAAAR!!!!
    try:
        kWh_carga_red = float(np.sum(df_sim["carga_red_kWh"]))
        kWh_descarga  = float(np.sum(df_sim["descarga_kWh"]))
        ciclos_eq     = kWh_descarga / max(1e-9, HW["E_CAP"])
        p_buy  = float(np.average(df_sim.loc[df_sim["carga_red_kWh"]>0, "precio_eur_mwh"])) if (df_sim["carga_red_kWh"]>0).any() else np.nan
        p_sell = float(np.average(df_sim.loc[df_sim["descarga_kWh"]>0, "precio_eur_mwh"])) if (df_sim["descarga_kWh"]>0).any() else np.nan
        spread = (p_sell - p_buy) if (np.isfinite(p_buy) and np.isfinite(p_sell)) else np.nan
    except Exception:
        kWh_carga_red = kWh_descarga = ciclos_eq = p_buy = p_sell = spread = np.nan
    diag = dict(
        capex=float(ECO.BASE_IMPONIBLE),
        ahorro=float(eco["ahorro_anual"]),
        payback=(float(ECO.BASE_IMPONIBLE)/float(eco["ahorro_anual"]) if float(eco["ahorro_anual"])>0 else np.inf),
        kWh_carga_red=kWh_carga_red, kWh_descarga=kWh_descarga, ciclos_eq=ciclos_eq,
        p_buy=p_buy, p_sell=p_sell, spread=spread,
        diff_Pinv_Pbatt=abs(float(HW["P_INV"]) - float(HW["P_BATT"])),)

    # ⚡️ Durante la búsqueda NO devolvemos el DataFrame enorme salvo que se pida
    out = dict(eco)
    out.update({
    "ECO": ECO,
    "HW": HW,
    "bat_marca": b_row.marca,
    "bat_modelo": b_row.modelo,
    "bat_mods_S": int(pd.to_numeric(getattr(b_row, "mods_S", 1), errors="coerce") or 1),
    "bat_mods_P": int(pd.to_numeric(getattr(b_row, "mods_P", 1), errors="coerce") or 1),
    "inv_marca": inv_row.marca,
    "inv_modelo": inv_row.modelo,
    "n_inv": n_inv,
    "diag": diag
    })

    if return_sim:
        out["SIM"] = df_sim
    return out

def build_ctx_from_session():
    return dict(
        market=st.session_state.get("market"),
        tarifa=st.session_state.get("tarifa", "3.0TD"),
        modalidad=st.session_state.get("modalidad"),
        pvpc_df=st.session_state.get("pvpc_df"),
        precios_te=st.session_state.get("precios_te", {}),
        precios_Ai=st.session_state.get("precios_Ai", {}),
        precios_Ci=st.session_state.get("precios_Ci", {}),
        potencias=st.session_state.get("potencias", {}),

        fv_flag=(st.session_state.get("fv") == "Sí"),
        modalidad_fv=st.session_state.get("modalidad_fv"),
        precio_fv=float(st.session_state.get("precio_fv", {}).get("Precio FV", 0.0)),
        cg_fv=float(st.session_state.get("cg_fv", {}).get("Costes Gestion FV", 0.0)),

        costes_iniciales_total=float(st.session_state.get("costes_iniciales_total", 0.0)),
        atr_total_energia=float(st.session_state.get("atr_total_energia", 0.0)),
        atr_total_potencia=float(st.session_state.get("atr_total_potencia", 0.0)),

        precio_qh_eur_mwh=st.session_state.get("precio_qh_eur_mwh"),
    )

def render_optimizador():
    import pandas as pd
    # --- Header bonito (hero) centrado en caja azul ---
    TITLE = "Optimizador de sistemas de almacenamiento"

    st.markdown(f"""<div style="background:#0f1c3f;border:1px solid #0a1933;border-radius:16px;padding:26px 28px;
            text-align:center;margin: 8px 0 18px 0;box-shadow: 0 6px 18px rgba(15,28,63,0.15);"><h1 style="
            color:#ffffff;margin:0;font-weight:800;letter-spacing:0.3px;font-size:calc(22px + 0.9vw);line-height:1.15;
            ">{TITLE}</h1></div>""",unsafe_allow_html=True,)

    # --- Stepper (línea de progreso de pasos) ---
    def render_stepper(total_steps=4, current_step=None):
        cur = int(st.session_state.get("step", 1) if current_step is None else current_step)
        cur = max(1, min(total_steps, cur))  # clamp

        # CSS del stepper
        st.markdown("""<style>.stepper-wrap{margin:12px 0 22px 0; display:flex; justify-content:center;}.stepper{display:flex; align-items:center; gap:14px;}
        .stepper .dot{width:26px;height:26px;border-radius:50%;background:#e6ecff;border:2px solid #b8c7ff;color:#4a5d9d;
            display:flex;align-items:center;justify-content:center;font-weight:700;}.stepper .dot.active{ background:#2b64ff; border-color:#1d47c7; color:#fff; }
        .stepper .bar{width:64px;height:6px;border-radius:6px;background:#e6ecff;border:1px solid #b8c7ff;}
        .stepper .bar.active{ background:#2b64ff; border-color:#1d47c7; }@media (max-width: 540px){
            .stepper .bar{width:44px;}}</style>""", unsafe_allow_html=True)

        # HTML dinámico según el paso actual
        parts = []
        for i in range(1, total_steps + 1):
            parts.append(f'<div class="dot {"active" if i <= cur else ""}">{i}</div>')
            if i < total_steps:
                parts.append(f'<div class="bar {"active" if i < cur else ""}"></div>')
        html = '<div class="stepper-wrap"><div class="stepper">' + "".join(parts) + "</div></div>"
        st.markdown(html, unsafe_allow_html=True)

    render_stepper(total_steps=4)


    if "step" not in st.session_state:
        st.session_state.step = 1

    # -------- Paso 1: Consumo --------
    if st.session_state.step == 1:
        st.header("Paso 1 · Datos iniciales - consumo de red")
        tarifa = st.radio("Selecciona tu tarifa:", ["2.0TD", "3.0TD", "6.1TD", "6.2TD", "6.3TD", "6.4TD"], horizontal=True)
        st.session_state["tarifa"] = tarifa

        instalacion = st.radio("Selecciona tu tipo de intalación eléctrica:", ["Trifásica", "Monofásica"], horizontal=True)
        st.session_state["instalacion"] = instalacion
        
        st.subheader("Potencias contratadas (kW)")
        
        # Carga valores previos si ya existen (para que no se pierdan al recargar)
        p_prev = st.session_state.get("potencias", {"P1":0.0,"P2":0.0,"P3":0.0,"P4":0.0,"P5":0.0,"P6":0.0})

        if tarifa == "2.0TD":
            c1, c2, = st.columns(2)
            P1 = c1.number_input("P1", min_value=0.0, step=0.1, value=float(p_prev.get("P1", 0.0)), key="pot_P1")
            P2 = c2.number_input("P2", min_value=0.0, step=0.1, value=float(p_prev.get("P2", 0.0)), key="pot_P2")


            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["potencias"] = {"P1": P1, "P2": P2}

        else:
            c1, c2, c3 = st.columns(3)
            P1 = c1.number_input("P1", min_value=0.0, step=0.1, value=float(p_prev.get("P1", 0.0)), key="pot_P1")
            P2 = c2.number_input("P2", min_value=0.0, step=0.1, value=float(p_prev.get("P2", 0.0)), key="pot_P2")
            P3 = c3.number_input("P3", min_value=0.0, step=0.1, value=float(p_prev.get("P3", 0.0)), key="pot_P3")

            c4, c5, c6 = st.columns(3)
            P4 = c4.number_input("P4", min_value=0.0, step=0.1, value=float(p_prev.get("P4", 0.0)), key="pot_P4")
            P5 = c5.number_input("P5", min_value=0.0, step=0.1, value=float(p_prev.get("P5", 0.0)), key="pot_P5")
            P6 = c6.number_input("P6", min_value=0.0, step=0.1, value=float(p_prev.get("P6", 0.0)), key="pot_P6")

            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["potencias"] = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6}
            
        st.subheader("Carga el consumo cuarto-horario anual de red")
        st.info("Los archivos deben estar en formato .xlsx o .xls y **contener dos columnas**:\n "
            "1) Fecha [formato: yyyy-mm-dd hh:mm:ss]\n"
            "2) Consumo de red [kWh]\n")
        archivo = st.file_uploader("Elige un archivo .xlsx o .xls")

        # Variables df1 = fechas, df2 = consumos
        if archivo is not None:
            try:
                df_raw = pd.read_excel(archivo)
                # Tomamos las dos primeras columnas: fecha, consumo
                fecha_col = df_raw.columns[0]
                consumo_col = df_raw.columns[1]
            
                fechas = pd.to_datetime(df_raw[fecha_col], errors='coerce')
                consumos = pd.to_numeric(df_raw[consumo_col], errors='coerce')
                base = pd.DataFrame({"datetime": fechas, "consumo": consumos}).dropna()
        
                st.session_state["consumo"] = base
                st.session_state["df1"] = base["datetime"]
                st.session_state["df2"] = base["consumo"]
            
                st.caption("Datos leidos correctamente.")

            except Exception as e:
                st.error(f"No pude leer o interpretar el Excel. Detalle: {e}")
                st.stop()

            fechas = st.session_state["df1"]
            consumos = st.session_state["df2"]

        from bbdd_mercado import attach_market_to_consumo

        if "consumo" in st.session_state and "market" in st.session_state:
            df_costes = attach_market_to_consumo(st.session_state["consumo"], st.session_state["market"])
            st.session_state["consumo_con_mercado"] = df_costes  # ya alineado por datetime


        c1, c2 = st.columns(2)
        c1.button("Continuar »", use_container_width=True,
                disabled=("consumo" not in st.session_state),
                on_click=lambda: st.session_state.update(step=2))

    # -------- Paso 2: FV sí/no + ficheros --------
    elif st.session_state.step == 2:
        st.header("Paso 2 · ¿La instalación dispone de FV?")
        fv = st.radio("Selecciona una opción:", ["No", "Sí"], horizontal=True)
        st.session_state["fv"] = fv

        if fv == "Sí":
            st.subheader("Sube las curvas cuarto-horarias anuales de la FV")
            st.info("Los archivo debe ser en formato .xlsx o .xls , **contener dos columnas**:\n "
            "1) Fecha [formato: yyyy-mm-dd hh:mm:ss]\n"
            "2) Excedentes/Generación [kWh]\n")
            up_exc = st.file_uploader("Excedentes FV (Excel/CSV)", type=["csv","xlsx","xls"], key="exc")
            up_gen = st.file_uploader("Generación FV (Excel/CSV)", type=["csv","xlsx","xls"], key="auto")

            if up_exc is not None:
                try:
                    df_exc = pd.read_excel(up_exc)
                    # Tomamos la segunda columna
                    excedente_col = df_exc.columns[1]
                    fecha_exc_col = df_exc.columns[0]

                    excedentes = pd.to_numeric(df_exc[excedente_col], errors='coerce')
                    fecha_exc = pd.to_datetime(df_exc[fecha_exc_col], errors='coerce')      
        
                    st.session_state["excedentes"] = excedentes
                    st.session_state["fecha_exc"] = fecha_exc
                    st.caption("Datos leidos correctamente.")
                except Exception as e:
                    st.error(f"No pude leer o interpretar el Excel. Detalle: {e}")

            if up_gen is not None:
                try:
                    df_gen = pd.read_excel(up_gen)
                    # Tomamos la segunda columna
                    generacion_col = df_gen.columns[1]
                    fecha_gen_col = df_gen.columns[0]
                    generacion = pd.to_numeric(df_gen[generacion_col], errors='coerce')
                    fecha_gen = pd.to_datetime(df_gen[fecha_gen_col], errors='coerce')      
        
                    st.session_state["generacion"] = generacion
                    st.session_state["fecha_gen"] = fecha_gen
                    st.caption("Datos leidos correctamente.")
                except Exception as e:
                    st.error(f"No pude leer o interpretar el Excel. Detalle: {e}")

        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=1))

        listo = (fv == "No") or ("excedentes" in st.session_state and "generacion" in st.session_state)
        c2.button("Continuar»", use_container_width=True, disabled=not listo,
                on_click=lambda: st.session_state.update(step=3))
        
    #---- Paso 3 : Datos de contrato electrico-----
    elif st.session_state.step == 3:
        st.header("Paso 3 · Datos del contrato eléctrico")
        
        tarifa = st.session_state.get("tarifa", "(sin seleccionar)")

        #----seleccion modalidad contrato----
        if tarifa == "2.0TD":
            modalidad = st.radio("Selecciona la modalidad de tu contrato eléctrico:", ["Precio fijo", "PVPC", "Indexado pass through"], horizontal=True)
            st.session_state["modalidad"] = modalidad
            # --- PVPC: carga automática desde ruta local ---
            PVPC_PATH = "data/PVPC_QH.xlsx"

            @st.cache_data(show_spinner=False)
            def load_pvpc_excel(path: str) -> pd.DataFrame:
                df_raw = pd.read_excel(path)
                fecha_col  = df_raw.columns[0]   # Columna fecha/hora
                precio_col = df_raw.columns[1]   # Columna precio [€/MWh] por QH
                df = pd.DataFrame({
                "datetime": pd.to_datetime(df_raw[fecha_col], errors="coerce"),
                "precio_eur_mwh": pd.to_numeric(df_raw[precio_col], errors="coerce"),
                }).dropna()
                # Redondear/ajustar a cuartos exactos y quitar tz
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                # Si vienen segundos/milisegundos, llevamos al inicio del cuarto
                df["datetime"] = df["datetime"].dt.floor("15min")
                # Si viniera con zona horaria, la quitamos (naive) para matchear con consumo
                if hasattr(df["datetime"].dt, "tz"):
                    df["datetime"] = df["datetime"].dt.tz_convert(None) if df["datetime"].dt.tz is not None else df["datetime"]

                return df

            if modalidad == "PVPC":
                try:
                    st.session_state["pvpc_df"] = load_pvpc_excel(PVPC_PATH)
                except Exception as e:
                    st.error(f"No se pudo leer el PVPC en {PVPC_PATH}. Detalle: {e}")

        else:
            modalidad = st.radio("Selecciona la modalidad de tu contrato eléctrico:", ["Precio fijo", "Indexado pass pool", "Indexado pass through"], horizontal=True)
            st.session_state["modalidad"] = modalidad
        
        st.info("Todos los precios deben estar en unidades de **€/MWh**")

        if modalidad == "Precio fijo" and st.session_state["tarifa"] == "2.0TD":
            c1, c2, c3 = st.columns(3)
            PrecioP1 = c1.number_input("Precio P1", min_value=0.0, step=0.01, key="Precio P1")
            PrecioP2 = c2.number_input("Precio P2", min_value=0.0, step=0.01, key="Precio P2")
            PrecioP3 = c3.number_input("Precio P3", min_value=0.0, step=0.01, key="Precio P3")

            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["precios_te"] = {"Precio P1": PrecioP1, "Precio P2": PrecioP2,"Precio P3": PrecioP3}
        
        elif modalidad == "Precio fijo" and st.session_state["tarifa"] != "2.0TD":
            c1, c2, c3 = st.columns(3)
            PrecioP1 = c1.number_input("Precio P1", min_value=0.0, step=0.01, key="Precio P1")
            PrecioP2 = c2.number_input("Precio P2", min_value=0.0, step=0.01, key="Precio P2")
            PrecioP3 = c3.number_input("Precio P3", min_value=0.0, step=0.01, key="Precio P3")

            c4, c5, c6 = st.columns(3)
            PrecioP4 = c4.number_input("Precio P4", min_value=0.0, step=0.01, key="Precio P4")
            PrecioP5 = c5.number_input("Precio P5", min_value=0.0, step=0.01, key="Precio P5")
            PrecioP6 = c6.number_input("Precio P6", min_value=0.0, step=0.01, key="Precio P6")

            st.session_state["precios_te"] = {"Precio P1": PrecioP1, "Precio P2": PrecioP2,"Precio P3": PrecioP3,"Precio P4": PrecioP4,"Precio P5": PrecioP5,"Precio P6": PrecioP6}

        elif modalidad == "Indexado pass pool":
            # 1) Tipo de OMIE para el contrato
            st.info("Selecciona cómo se indexa tu contrato al OMIE.")
            tipo_omie = st.radio(
                "Base de OMIE para el término de energía:",
                ["Horario", "Mensual"],
                index=0,
                horizontal=True,
                key="pp_omie_tipo_radio",
            )

            # Guardamos un valor limpio para usar luego en el Paso 4
            st.session_state["pp_omie_tipo"] = (
                "horario" if tipo_omie == "Horario" else "mensual"
            )

            # 2) Coeficientes fijos Ai
            st.info("Introduce el coeficiente fijo (Ai)")
            c1, c2, c3 = st.columns(3)
            PrecioAiP1 = c1.number_input("Precio Ai P1", min_value=0.0, step=0.01, key="Precio Ai P1")
            PrecioAiP2 = c2.number_input("Precio Ai P2", min_value=0.0, step=0.01, key="Precio Ai P2")
            PrecioAiP3 = c3.number_input("Precio Ai P3", min_value=0.0, step=0.01, key="Precio Ai P3")

            c4, c5, c6 = st.columns(3)
            PrecioAiP4 = c4.number_input("Precio Ai P4", min_value=0.0, step=0.01, key="Precio Ai P4")
            PrecioAiP5 = c5.number_input("Precio Ai P5", min_value=0.0, step=0.01, key="Precio Ai P5")
            PrecioAiP6 = c6.number_input("Precio Ai P6", min_value=0.0, step=0.01, key="Precio Ai P6")

            st.session_state["precios_Ai"] = {
                "Precio Ai P1": PrecioAiP1,
                "Precio Ai P2": PrecioAiP2,
                "Precio Ai P3": PrecioAiP3,
                "Precio Ai P4": PrecioAiP4,
                "Precio Ai P5": PrecioAiP5,
                "Precio Ai P6": PrecioAiP6,
            }

            # 3) Coeficientes variables Ci
            st.info("Introduce el coeficiente del término variable (Ci)")
            c1, c2, c3 = st.columns(3)
            PrecioCiP1 = c1.number_input("Precio Ci P1", min_value=0.0, step=0.01, key="Precio Ci P1")
            PrecioCiP2 = c2.number_input("Precio Ci P2", min_value=0.0, step=0.01, key="Precio Ci P2")
            PrecioCiP3 = c3.number_input("Precio Ci P3", min_value=0.0, step=0.01, key="Precio Ci P3")

            c4, c5, c6 = st.columns(3)
            PrecioCiP4 = c4.number_input("Precio Ci P4", min_value=0.0, step=0.01, key="Precio Ci P4")
            PrecioCiP5 = c5.number_input("Precio Ci P5", min_value=0.0, step=0.01, key="Precio Ci P5")
            PrecioCiP6 = c6.number_input("Precio Ci P6", min_value=0.0, step=0.01, key="Precio Ci P6")

            st.session_state["precios_Ci"] = {
                "Precio Ci P1": PrecioCiP1,
                "Precio Ci P2": PrecioCiP2,
                "Precio Ci P3": PrecioCiP3,
                "Precio Ci P4": PrecioCiP4,
                "Precio Ci P5": PrecioCiP5,
                "Precio Ci P6": PrecioCiP6,
            }

        elif modalidad == "Indexado pass through":
            st.info("Ingresa los desvíos de tu comercializadora juntamente con los costes de gestión. Ten en cuenta todos los extras que pueden incluirse, como primas de riesgo o constantes. En caso de que tu comercializadora no fije el valor de los desvíos debes poner 0,328 €/MWh")

            c1, c2 = st.columns(2)
            Desvíos = c1.number_input("Desvíos [€/MWh]", min_value=0.0, step=0.1, key="desvios")
            CG = c2.number_input("CG [€/MWh]", min_value=0.0, step=0.1, key="CG")

            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["comer"] = {"desvios": Desvíos, "CG": CG}

        #----- Si hay FV modalidad compra excedentes----
        if st.session_state["fv"] == "Sí":
            modalidad_fv = st.radio("Selecciona la modalidad de tu contrato fotovoltaico:", ["Precio fijo", "Indexado"], horizontal=True)
            st.session_state["modalidad_fv"] = modalidad_fv
            st.info("En caso de **no disponer de compensación de excedentes** se debe **seleccionar modalidad precio fijo 0 €/MWh**")
            
            if modalidad_fv == "Precio fijo":
                PrecioFV = st.number_input("Precio FV", min_value=0.0, step=0.01, key="Precio FV")
                st.session_state["precio_fv"] = {"Precio FV": PrecioFV}

            if modalidad_fv == "Indexado":
                CG_fv = st.number_input("Costes Gestion FV", min_value=0.0, step=0.01, key="Costes Gestion FV")
                st.session_state["cg_fv"] = {"Costes Gestion FV": CG_fv}

        # precio Potencia
        modalidad_pot = st.radio("Selecciona tu termino de potencia:", ["BOE", "No BOE"], horizontal=True)
        st.session_state["modalidad_pot"] = modalidad_pot

        if modalidad_pot == "No BOE" and tarifa == "2.0TD":
            c1, c2 = st.columns(2)
            precio_pot_p1 = c1.number_input("Precio potencia P1 [€/kW año]", min_value=0.0, step=0.1, key="precio_pot_p1")
            precio_pot_p2 = c2.number_input("Precio potencia P2 [€/kW año]", min_value=0.0, step=0.1, key="precio_pot_p2")
            st.session_state["precio_pot"] = {"precio_pot_p1": precio_pot_p1, "recio_pot_p2": precio_pot_p2}

        elif modalidad_pot == "No BOE" and tarifa != "2.0TD":
            c1, c2, c3 = st.columns(3)
            PreciopotP1 = c1.number_input("Precio potencia P1", min_value=0.0, step=0.01, key="Precio potencia P1")
            PreciopotP2 = c2.number_input("Precio potencia P2", min_value=0.0, step=0.01, key="Precio potencia P2")
            PreciopotP3 = c3.number_input("Precio potencia P3", min_value=0.0, step=0.01, key="Precio potencia P3")

            c4, c5, c6 = st.columns(3)
            PreciopotP4 = c4.number_input("Precio potencia P4", min_value=0.0, step=0.01, key="Precio potencia P4")
            PreciopotP5 = c5.number_input("Precio potencia P5", min_value=0.0, step=0.01, key="Precio potencia P5")
            PreciopotP6 = c6.number_input("Precio potencia P6", min_value=0.0, step=0.01, key="Precio potencia P6")

            st.session_state["precio_pot"] = {"Precio potencia P1": PreciopotP1, "Precio potencia P2": PreciopotP2,"Precio potencia P3": PreciopotP3,"Precio potencia P4": PreciopotP4,"Precio potencia P5": PreciopotP5,"Precio potencia P6": PreciopotP6}
    
        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=2))
        c2.button("Generar resumen situación inicial»", use_container_width=True, on_click=lambda: st.session_state.update(step=4))

        
    # ------Paso 4 : Resumen ------
    elif st.session_state.step == 4:
        st.header("Paso 4 · Resumen situación inicial")

        # -------- helpers de estilo --------
        def _styler_base():
            return [
            {"selector":"th", "props":"background:#eef4ff; color:#1a2b4b; font-weight:700;"},
            {"selector":"tbody tr:nth-child(even)", "props":"background:#fafbff;"},
            {"selector":"td, th", "props":"border:1px solid #e9edf5; padding:6px 10px;"},]

        def style_tabla(df, bold_first_col=True, fmt_map=None, highlight_total_label="Total"):
            stl = df.style.set_table_styles(_styler_base())
            if bold_first_col and len(df.columns)>0:
                stl = stl.set_properties(subset=pd.IndexSlice[:, [df.columns[0]]],
                                    **{"font-weight":"700"})
            if fmt_map:
                stl = stl.format(fmt_map)

        # 💡 Resaltar la fila "Total": fondo y negrita
            def _row_style(s):
                if str(s.iloc[0]).strip().lower() == str(highlight_total_label).lower():
                    return ["font-weight:700; background-color:#fff7cc;"] * len(s)
                return [""] * len(s)
            stl = stl.apply(_row_style, axis=1)

            return stl.hide(axis="index")

        # Cargar base de mercado desde disco, sin UI
        from bbdd_mercado import ensure_market_loaded
        try:
            ensure_market_loaded()
        except Exception as e:
            st.session_state["market_error"] = f"{type(e).__name__}: {e}"

        # --- Tarifa ---
        tarifa = st.session_state.get("tarifa", "(sin seleccionar)")

        # --- Potencias contratadas (kW) - compacto ---
        st.subheader("Potencias contratadas (kW)")
        pot = st.session_state.get("potencias", {"P1":0.0,"P2":0.0,"P3":0.0,"P4":0.0,"P5":0.0,"P6":0.0})

        if tarifa == "2.0TD":
            df_pot = pd.DataFrame(
            {"Periodo": ["P1","P2"],
            "Potencia [kW]": [pot.get("P1",0.0), pot.get("P2",0.0)]})
        else:
            df_pot = pd.DataFrame(
            {"Periodo": ["P1","P2","P3","P4","P5","P6"],
            "Potencia [kW]": [pot.get("P1",0.0), pot.get("P2",0.0), pot.get("P3",0.0),
                            pot.get("P4",0.0), pot.get("P5",0.0), pot.get("P6",0.0)]})

        def _fmt_kw(v):
            try:
                return f"{float(v):,.1f}".replace(",", ".")
            except Exception:
                return v

        st.table(style_tabla(df_pot, fmt_map={"Potencia [kW]": _fmt_kw}))

        # ---- Energia ---
        # consumo anual
        total_anual = float(st.session_state["consumo"]["consumo"].sum())
        st.subheader("Resumen consumo anual")
        st.metric("Consumo total del año", f"{total_anual:,.0f} kWh".replace(",", "."))

        # consumo mensual
        c = st.session_state["consumo"].copy()  # DataFrame con columnas: datetime, consumo
        c["mes"] = c["datetime"].dt.month
        cons_m = c.groupby("mes")["consumo"].sum()

        meses_es = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

        fv = st.session_state.get("fv", "No")

        if fv != "Sí":
            meses_presentes = sorted(cons_m.index.tolist())
            df_tabla = pd.DataFrame({"Mes": [meses_es[m] for m in meses_presentes],"Consumo [kWh]": cons_m.reindex(meses_presentes).values})
            df_tabla["Consumo [kWh]"] = df_tabla["Consumo [kWh]"].round(0).astype(int)

            tot_cons = df_tabla["Consumo [kWh]"].sum()
            df_total = pd.DataFrame([{"Mes": "Total","Consumo [kWh]":tot_cons}])

        else:
            df_e = pd.DataFrame({"datetime": pd.to_datetime(st.session_state["fecha_exc"], errors="coerce"),"value":    pd.to_numeric(st.session_state["excedentes"], errors="coerce")}).dropna()
            df_g = pd.DataFrame({"datetime": pd.to_datetime(st.session_state["fecha_gen"], errors="coerce"),"value":    pd.to_numeric(st.session_state["generacion"], errors="coerce")}).dropna()

            df_e["mes"] = df_e["datetime"].dt.month
            df_g["mes"] = df_g["datetime"].dt.month

            exc_m = df_e.groupby("mes")["value"].sum()
            gen_m = df_g.groupby("mes")["value"].sum()

            # Índice común (meses presentes en cualquiera)
            meses_presentes = sorted(set(cons_m.index) | set(gen_m.index) | set(exc_m.index))

            df_tabla = pd.DataFrame({"Mes": [meses_es[m] for m in meses_presentes],"Consumo [kWh]": cons_m.reindex(meses_presentes).fillna(0).values,"Generación FV [kWh]": gen_m.reindex(meses_presentes).fillna(0).values,"Excedentes [kWh]": exc_m.reindex(meses_presentes).fillna(0).values,})

            # Redondeo
            for col in ["Consumo [kWh]", "Generación FV [kWh]", "Excedentes [kWh]"]:
                df_tabla[col] = df_tabla[col].round(0).astype(int)

            # % Autoconsumo = (Generación-Excedentes) / (Consumo red + Generación - excedentes)
            autoconsumo_m = df_tabla["Generación FV [kWh]"] - df_tabla["Excedentes [kWh]"]
            denom = df_tabla["Consumo [kWh]"] + autoconsumo_m
            df_tabla["% Autoconsumo"] = (100 * autoconsumo_m / denom.where(denom != 0, 1)).fillna(0)

            # Fila Total (sumas y % con totales)
            tot_cons = df_tabla["Consumo [kWh]"].sum()
            tot_gen  = df_tabla["Generación FV [kWh]"].sum()
            tot_exc  = df_tabla["Excedentes [kWh]"].sum()
            tot_pct  = 100 * ((tot_gen-tot_exc) / (tot_cons + tot_gen - tot_exc)) if (tot_cons + tot_gen - tot_exc) > 0 else 0.0

            df_total = pd.DataFrame([{"Mes": "Total","Consumo [kWh]": tot_cons,"Generación FV [kWh]": tot_gen,"Excedentes [kWh]": tot_exc,"% Autoconsumo": tot_pct,}])

            # Redondeos bonitos
            for col in ["Consumo [kWh]", "Generación FV [kWh]", "Excedentes [kWh]"]:
                df_tabla[col] = df_tabla[col].round(0).astype(int)
                df_total[col] = int(round(df_total[col].iloc[0], 0))
            df_tabla["% Autoconsumo"] = df_tabla["% Autoconsumo"].round(1)
            df_total["% Autoconsumo"] = round(float(df_total["% Autoconsumo"]), 1)

        # Mostrar
        tabla_final = pd.concat([df_tabla, df_total], ignore_index=True)
        # Mostrar (formateo suave)
        tabla_final_fmt = tabla_final.copy()

        # formateo miles sin decimales en columnas kWh
        cols_kwh = [c for c in tabla_final_fmt.columns if "kWh" in c]
        for col in cols_kwh:
            tabla_final_fmt[col] = tabla_final_fmt[col].map(lambda v: f"{int(v):,}".replace(",", "."))

        if "% Autoconsumo" in tabla_final_fmt.columns:
            tabla_final_fmt["% Autoconsumo"] = tabla_final_fmt["% Autoconsumo"].map(lambda x: f"{x:.1f} %")

        fmt_cols = {c: (lambda v: f"{int(v):,}".replace(",", ".")) for c in tabla_final.columns if "kWh" in c}
        if "% Autoconsumo" in tabla_final.columns:
            fmt_cols["% Autoconsumo"] = lambda v: f"{v:.1f} %"

        st.table(style_tabla(tabla_final, fmt_map=fmt_cols, highlight_total_label="Total"))

        import altair as alt

        st.caption("Grafica de consumo mensual (kWh)")

        orden_meses = list(range(1,13))
        meses_es   = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

        # Consumo en orden fijo
        serie_cons = cons_m.reindex(orden_meses).fillna(0)

        # ¿Hay autoconsumo mensual? (Autoconsumo = Generación - Excedentes)
        serie_auto = None
        if st.session_state.get("fv") == "Sí":
            try:
                serie_gen = gen_m.reindex(orden_meses).fillna(0)
                serie_exc = exc_m.reindex(orden_meses).fillna(0)
                serie_auto = (serie_gen - serie_exc).clip(lower=0)  # autoconsumo ≥ 0
            except NameError:
                pass

        df_m = pd.DataFrame({"MesNum": orden_meses,
                        "Mes": [meses_es[m] for m in orden_meses],
                        "Consumo": serie_cons.values})

        if serie_auto is not None:
            df_m["Autoconsumo"] = serie_auto.values

        # Pasar a formato largo para Altair
        df_long = df_m.melt(id_vars=["MesNum","Mes"], var_name="Concepto", value_name="kWh")
        df_long = df_long[df_long["Concepto"].notna()]  # por si no hay generación

        colores = alt.Scale(
        domain=["Consumo","Autoconsumo"],
        range=["#9ec9ff", "#ffc48a"])

        chart = (
            alt.Chart(df_long)
            .mark_bar()
            .encode(
                x=alt.X("Mes:N", sort=[meses_es[m] for m in orden_meses], title="Mes"),
                y=alt.Y("kWh:Q", title="kWh"),
                color=alt.Color("Concepto:N", scale=colores, title=""),
                tooltip=[alt.Tooltip("Concepto:N"), alt.Tooltip("Mes:N"), alt.Tooltip("kWh:Q", format=",.0f")])
            .properties(height=320))

        st.altair_chart(chart, use_container_width=True)

        #Costes
        from index_pt import _leer_desvios_y_cg, coste_passthrough

        if "market" not in st.session_state:
            st.error("Falta la base de mercado. " +
                f"Detalle: {st.session_state.get('market_error','')}")
            st.stop()

        #ATR Energia
        from index_pt import coste_atr_energia
        atr_total, atr_detalle, atr_qh_eur_mwh = coste_atr_energia(
        df_consumo=st.session_state["consumo"],
        df_mercado=st.session_state["market"],
        tarifa=tarifa,)

        st.session_state["atr_total_energia"] = float(atr_total)
        st.session_state["atr_qh_eur_mwh"] = atr_qh_eur_mwh

        #ATR Potencia
        from ATR_Potencia import coste_atr_potencia

        potencias = st.session_state.get("potencias", {})

        modalidad_pot = st.session_state.get("modalidad_pot")

        if modalidad_pot == "BOE":
            try:
                atr_pot_total, atr_pot_detalle = coste_atr_potencia(tarifa, potencias)
            except Exception as e:
                st.error(f"No se pudo calcular el ATR de potencia: {e}")

            st.session_state["atr_total_potencia"] = float(atr_pot_total)

        else:
            precios_user = st.session_state.get("precio_pot", {}) or {}

            # Construir diccionario coef €/kW·año por periodo, según la tarifa y los nombres guardados en Paso 3
            t = (tarifa or "").replace(" ", "").upper()
            if t == "2.0TD":
                # En Paso 3 guardaste claves en minúsculas y hay un typo en P2 ("recio_pot_p2"), los atendemos todos
                cP1 = precios_user.get("precio_pot_p1") or precios_user.get("Precio potencia P1") or 0.0
                cP2 = precios_user.get("precio_pot_p2") or precios_user.get("Precio potencia P2") or precios_user.get("recio_pot_p2") or 0.0
                coef = {"P1": float(cP1), "P2": float(cP2)}
                periodos = ["P1", "P2"]
            else:
                # Para 3.0TD y 6.XTD guardaste: "Precio potencia P1"... "Precio potencia P6"
                coef = {
                f"P{i}": float(precios_user.get(f"Precio potencia P{i}", 0.0))
                for i in range(1, 7)}
                periodos = [f"P{i}" for i in range(1, 7)]

            # Comprobar que hay al menos un precio informado
            if all(v == 0.0 for v in coef.values()):
                st.warning("No hay precios de potencia informados en el Paso 3 (No BOE).")
                atr_pot_total, atr_pot_detalle = 0.0, pd.DataFrame(columns=["Periodo","Potencia (kW)","Coef €/kW·año","Coste (€)"])
            else:
                # Misma estructura de detalle que el cálculo BOE
                detalle = []
                for p in periodos:
                    pot_kW = float(potencias.get(p, 0.0))
                    c = float(coef.get(p, 0.0))
                    coste = pot_kW * c
                    detalle.append({
                    "Periodo": p,
                    "Potencia (kW)": pot_kW,
                    "Coef €/kW·año": c,
                    "Coste (€)": coste})
                atr_pot_detalle = pd.DataFrame(detalle)
                atr_pot_total = float(atr_pot_detalle["Coste (€)"].sum())

        # Guardar y mostrar
        st.session_state["atr_total_potencia"] = float(atr_pot_total)

        #Index PT
        if st.session_state.get("modalidad") == "Indexado pass through":
            if "market" not in st.session_state:
                st.error("Falta la base de mercado. Revisa la ruta del Excel en bbdd_mercado.py.")
                st.stop()

            tarifa = st.session_state.get("tarifa", "3.0TD")
            des, cg = _leer_desvios_y_cg()

            coste_total, detalle = coste_passthrough(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=tarifa,
                desvios=des,
                cg=cg,)

            st.session_state["coste_mercado_te"] = float(coste_total)

            # --- Dentro de "Indexado pass through" (PT) ---
            coste_total, detalle = coste_passthrough(
            df_consumo=st.session_state["consumo"],
            df_mercado=st.session_state["market"],
            tarifa=tarifa,
            desvios=des,
            cg=cg,)
            # Guarda precio €/MWh por QH (columna "Precio_unitario(€/MWh)")
            st.session_state["precio_qh_eur_mwh"] = detalle["Precio_unitario(€/MWh)"].rename("precio_eur_mwh")

        #PVPC
        if st.session_state.get("modalidad") == "PVPC":
            if "consumo" not in st.session_state:
                st.error("Falta el consumo para calcular PVPC.")
            elif "pvpc_df" not in st.session_state:
                st.error("Falta la tabla PVPC (Paso 3).")
            else:
                df_c = st.session_state["consumo"].copy()   # columnas: datetime, consumo [kWh]
                df_p = st.session_state["pvpc_df"].copy()   # columnas: datetime, precio_eur_mwh

                df = df_c.merge(df_p, on="datetime", how="left")
                if df["precio_eur_mwh"].isna().any():
                    n = int(df["precio_eur_mwh"].isna().sum())
                    st.warning(f"{n} registros de consumo sin precio PVPC. Se omiten del coste.")
                    df = df.dropna(subset=["precio_eur_mwh"])

                df["coste_qh"] = (df["consumo"] / 1000.0) * df["precio_eur_mwh"]  # kWh→MWh
                coste_pvpc_total = float(df["coste_qh"].sum())
                atr_eur = float(st.session_state.get("atr_total_energia", 0.0))
                total_pvpc_neto = coste_pvpc_total - atr_eur

            st.session_state["coste_mercado_te"] = float(total_pvpc_neto)

            # --- Dentro de "PVPC" (después de fusionar consumo con df_p) ---
            # df tiene columnas: ['datetime','consumo','precio_eur_mwh','coste_qh'] si mantienes ese orden
            # Guarda serie precio por QH alineada al consumo
            st.session_state["precio_qh_eur_mwh"] = df.set_index("datetime")["precio_eur_mwh"].rename("precio_eur_mwh")

        #Fijo
        from fijo import coste_fijo_energia

        if st.session_state.get("modalidad") == "Precio fijo":
            if "market" not in st.session_state or "consumo" not in st.session_state:
                st.error("Faltan datos: consumo o base de mercado.")
                st.stop()

            precios_te = st.session_state.get("precios_te", {}) 
            if not precios_te:
                st.warning("Introduce los precios por periodo en el Paso 3.")
            else:
                total_fijo, det_fijo, res_fijo = coste_fijo_energia(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=st.session_state["tarifa"],
                precios_te=precios_te,)

                atr_eur = float(st.session_state.get("atr_total_energia", 0.0))
                total_fijo_neto = total_fijo - atr_eur
                total_txt = fmt_eur(total_fijo_neto, 2) if 'fmt_eur' in globals() else f"{total_fijo_neto:,.2f}"

            st.session_state["coste_mercado_te"] = float(total_fijo_neto)

            total_fijo, det_fijo, res_fijo = coste_fijo_energia(
            df_consumo=st.session_state["consumo"],
            df_mercado=st.session_state["market"],
            tarifa=st.session_state["tarifa"],
            precios_te=precios_te,)
            # Guarda precio €/MWh por QH
            st.session_state["precio_qh_eur_mwh"] = det_fijo["precio_eur_MWh"].rename("precio_eur_mwh")


        #Index PP
        from index_pp import coste_indexado_pp

        if st.session_state.get("modalidad") == "Indexado pass pool":
            if "market" not in st.session_state or "consumo" not in st.session_state:
                st.error("Faltan datos de mercado o consumo.")
                st.stop()

            A = st.session_state.get("precios_Ai", {})
            C = st.session_state.get("precios_Ci", {})

            # Modo OMIE elegido en el Paso 3 (por defecto, mensual para compatibilidad)
            modo_omie = st.session_state.get("pp_omie_tipo", "mensual")

            if not A or not C:
                st.warning("Introduce los coeficientes Ai y Ci en el Paso 3.")
            else:
                total_pp, det_pp, res_pp = coste_indexado_pp(
                    df_consumo=st.session_state["consumo"],
                    df_mercado=st.session_state["market"],
                    tarifa=st.session_state["tarifa"],
                    coef_A=A,
                    coef_C=C,
                    modo_omie=modo_omie,
                )

                atr_eur = float(st.session_state.get("atr_total_energia", 0.0))
                total_pp_neto = total_pp - atr_eur
                total_txt = fmt_eur(total_pp_neto, 2)

            st.session_state["coste_mercado_te"] = float(total_pp_neto)

            # Segunda llamada para guardar el precio €/MWh por QH
            total_pp, det_pp, res_pp = coste_indexado_pp(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=st.session_state["tarifa"],
                coef_A=A,
                coef_C=C,
                modo_omie=modo_omie,
            )
            st.session_state["precio_qh_eur_mwh"] = det_pp["precio_eur_MWh"].rename("precio_eur_mwh")

        #FNEE
        from fnee import coste_fnee

        if "consumo" not in st.session_state:
            st.error("Falta cargar el consumo.")
        else:
            fnee_total, fnee_mwh = coste_fnee(st.session_state["consumo"])

            st.session_state["fnee_total"] = float(fnee_total)


        #FV Fijo
        # --- Excedentes FV en modalidad FIJA ---
        from fv_fijo import coste_excedentes_fijo

        tiene_fv = (st.session_state.get("fv") == "Sí")                 # <- tu flag real del Paso 2
        modo_fv  = st.session_state.get("modalidad_fv")                 # <- guardado en el Paso 3

        if tiene_fv and (str(modo_fv) == "Precio fijo"):
        # Construye el DataFrame esperado por la función (datetime + columna numérica)
            if "excedentes" in st.session_state and "fecha_exc" in st.session_state:
                df_exc = pd.DataFrame({
                "datetime": pd.to_datetime(st.session_state["fecha_exc"], errors="coerce"),
                "excedentes_kWh": pd.to_numeric(st.session_state["excedentes"], errors="coerce"),
                }).dropna()

                # Precio de excedentes tal y como lo guardas en el Paso 3
                precio_exc = float(st.session_state.get("precio_fv", {}).get("Precio FV", 0.0))

                ingresos_exc, energia_exc_mwh, precio_mwh_usado = coste_excedentes_fijo(
                df_excedentes=df_exc,
                precio_excedentes=precio_exc,)
                st.session_state["ingresos_fv"] = float(ingresos_exc)  

            else:
                st.warning("Faltan las curvas de excedentes FV del Paso 2.")

        # --- Excedentes FV en modalidad INDEXADA ---
        from fv_indexado import coste_excedentes_indexado

        if tiene_fv and (str(modo_fv) == "Indexado"):
            if "excedentes" in st.session_state and "fecha_exc" in st.session_state:
                # Construir DataFrame de excedentes (mismo formato que paso 2)
                df_exc = pd.DataFrame({
                "datetime": pd.to_datetime(st.session_state["fecha_exc"], errors="coerce"),
                "excedentes_kWh": pd.to_numeric(st.session_state["excedentes"], errors="coerce")
                }).dropna()

                cg_fv = float(st.session_state.get("cg_fv", {}).get("Costes Gestion FV", 0.0))

                ingresos_indexado, det_fv = coste_excedentes_indexado(
                df_excedentes=df_exc,
                df_mercado=st.session_state["market"],
                cg_fv=cg_fv)

                st.session_state["ingresos_fv"] = float(ingresos_indexado) 

            else:
                st.warning("Faltan las curvas de excedentes FV del Paso 2.")

        # Excesos de Potencia
        from excesos_pot import prepara_base_excesos

        if "consumo" not in st.session_state or "market" not in st.session_state:
            st.error("Faltan datos de consumo o base de mercado.")
        else:
            tarifa = st.session_state.get("tarifa", "3.0TD")
            potencias = st.session_state.get("potencias", {})

            try:
                base_excesos = prepara_base_excesos(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=tarifa,
                potencias_dict=potencias)
                st.session_state["base_excesos"] = base_excesos

            except Exception as e:
                st.error(f"Error al preparar base de excesos: {e}")

        # Ex_pot contadores 1,2,3
        # ---- Cálculo excesos (contadores 1,2,3) ----
        from excesos_pot import calcula_excesos_cont_123 , calcula_excesos_cont_45

        if st.session_state.get("tarifa", "") != "2.0TD":
            tipo_cont = base_excesos.attrs.get("contador_tipo")
            if tipo_cont == "contador1,2,3":
                try:
                    total_exceso, res_exceso = calcula_excesos_cont_123(
                        base_excesos=base_excesos,
                        tarifa=st.session_state.get("tarifa", "3.0TD"))

                    # Formateo de columnas para mostrar
                    # Formateo de columnas para mostrar (usa nombres reales devueltos por la función)
                    res_exceso_fmt = (
                    res_exceso
                    .assign(
                    S_sum_raiz=lambda d: d["S_sum_raiz"].round(3),
                    coef=lambda d: d["coef"].round(6),
                    **{"coste_€": lambda d: d["coste_€"].round(2)}))

                except Exception as e:
                    st.error(f"No se pudo calcular el exceso (1,2,3): {e}")

                st.session_state["excesos_total"] = float(total_exceso)

            elif tipo_cont == "contador4,5":
                try:
                    total_exceso_45, res_exceso_45 = calcula_excesos_cont_45(
                    base_excesos=base_excesos,
                    tarifa=st.session_state.get("tarifa", "3.0TD"),)

                    res_exceso_45_fmt = res_exceso_45.assign(
                    coef=lambda d: d["coef"].round(6),
                    **{"coste_€": lambda d: d["coste_€"].round(2)})

                except Exception as e:
                    st.error(f"No se pudo calcular el exceso (4,5): {e}")

                st.session_state["excesos_total"] = float(total_exceso_45)

        # --- Impuesto Especial sobre la Electricidad (IEE) ---
        iee_pct = 5.11269 / 100.0

        base_iee = sum([
        float(st.session_state.get("coste_mercado_te", 0.0)),   # Mercado – término energía
        float(st.session_state.get("atr_total_energia", 0.0)),  # ATR energía
        float(st.session_state.get("atr_total_potencia", 0.0)), # ATR potencia
        float(st.session_state.get("excesos_total", 0.0)),      # Excesos de potencia
        float(st.session_state.get("fnee_total", 0.0)),         # FNEE
        ]) - float(st.session_state.get("ingresos_fv", 0.0))         # <— restar FV (ingreso)

        iee_total = base_iee * iee_pct

        st.session_state["iee_total"] = float(iee_total)

        # --- Resumen de costes anuales (tipo factura) ---
        st.subheader("Resumen de costes anuales")

        base_iee = sum([
        float(st.session_state.get("coste_mercado_te", 0.0)),
        float(st.session_state.get("atr_total_energia", 0.0)),
        float(st.session_state.get("atr_total_potencia", 0.0)),
        float(st.session_state.get("excesos_total", 0.0)),
        float(st.session_state.get("fnee_total", 0.0)),]) - float(st.session_state.get("ingresos_fv", 0.0))

        iee_total = float(st.session_state.get("iee_total", 0.0))
        total_costes = base_iee + iee_total
        st.session_state["costes_iniciales_total"] = float(total_costes)


        lineas = [
        ("Mercado – término energía",  float(st.session_state.get("coste_mercado_te", 0.0))),
        ("ATR energía",                float(st.session_state.get("atr_total_energia", 0.0))),
        ("ATR potencia",               float(st.session_state.get("atr_total_potencia", 0.0))),
        ("Excesos de potencia",        float(st.session_state.get("excesos_total", 0.0))),
        ("FNEE",                       float(st.session_state.get("fnee_total", 0.0))),
        ("Venta excedentes",      -float(st.session_state.get("ingresos_fv", 0.0))),
        ("Base IEE",                    base_iee),
        ("Impuesto electricidad (5,1127%)", iee_total),]

        df_factura = pd.DataFrame(lineas, columns=["Concepto", "Importe (€)"])

        # formateo euros con tu helper fmt_eur()
        fmt_eur_map = {"Importe (€)": lambda v: f"{fmt_eur(v, 2)} €"}
        st.table(style_tabla(df_factura, fmt_map=fmt_eur_map))

        # “caja” de total
        st.markdown(
        f"""
        <div style="
        background:#f6f9ff;border:1px solid #dfe7fb;padding:12px 16px;
        border-radius:10px; display:flex; justify-content:space-between; align-items:center;">
        <div style="font-weight:700;color:#1a2b4b;">TOTAL COSTE ELECTRICIDAD ANUAL</div>
        <div style="font-weight:800;font-size:1.15rem;color:#0f1c3f;">{fmt_eur(total_costes, 2)} €</div>
        </div>
        """,
        unsafe_allow_html=True,)

        st.caption("Distribución del coste anual por conceptos")

        # Construir dataframe de costes (mismos conceptos que en tu tabla)
        df_costes = pd.DataFrame([
        ("Mercado – término energía",  float(st.session_state.get("coste_mercado_te", 0.0))),
        ("ATR energía",                float(st.session_state.get("atr_total_energia", 0.0))),
        ("ATR potencia",               float(st.session_state.get("atr_total_potencia", 0.0))),
        ("Excesos de potencia",        float(st.session_state.get("excesos_total", 0.0))),
        ("FNEE",                       float(st.session_state.get("fnee_total", 0.0))),
        ("Impuesto electricidad",      float(st.session_state.get("iee_total", 0.0))),
        # Nota: “Ingresos FV (restan)” suele ser negativo; lo dejamos fuera del quesito para no distorsionar el %.
        ], columns=["Concepto","Importe"])

        df_costes_pos = df_costes[df_costes["Importe"] > 0].copy()
        suma_pos = df_costes_pos["Importe"].sum()
        if suma_pos > 0:
            df_costes_pos["%"] = df_costes_pos["Importe"] / suma_pos

            pie = (
                alt.Chart(df_costes_pos)
                .mark_arc(innerRadius=60)   # donut
                .encode(
                    theta=alt.Theta("Importe:Q", stack=True),
                    color=alt.Color("Concepto:N", legend=alt.Legend(title="Concepto")),
                    tooltip=[
                    alt.Tooltip("Concepto:N"),
                    alt.Tooltip("Importe:Q", format=",.2f"),
                    alt.Tooltip("%:Q", title="% del total", format=".1%")]).properties(height=320))
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("No hay costes positivos para graficar.")

        st.caption("Nota: los *Ingresos FV* (si existen) reducen el total y por ser negativos no se incluyen en el gráfico.")

        st.markdown("""<style>/* Botón primario (azul) mejorado */.stButton > button[kind="primary"]{background: linear-gradient(180deg,#1b2e57,#0f1c3f);
        color:#fff;font-weight:800;font-size:1rem;border:0;border-radius:12px;padding:10px 16px;box-shadow:0 6px 16px rgba(43,100,255,.35);
        letter-spacing:.3px;transition:all .15s ease-in-out;}.stButton > button[kind="primary"]:hover{filter:brightness(1.05);box-shadow:0 8px 20px rgba(43,100,255,.45);
        }.stButton > button[kind="primary"]:active{transform:translateY(1px);
        }.stButton > button[kind="primary"]:focus{outline:3px solid rgba(43,100,255,.35);
        outline-offset:2px;}</style>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=3))
        c2.button("Generar propuesta»", use_container_width=True, type="primary", on_click=lambda: st.session_state.update(step=5))

    # ------Paso 5 : Propuesta ------
    elif st.session_state.step == 5:
        st.header("Sistema de almacenamiento propuesto")
        # --- Ocultar barra de pasos SOLO en Paso 5 ---
        st.markdown("""
        <style>
            .stepper-wrap, .stepper-wrap * {
                display: none !important;
                visibility: hidden !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        import traceback

        # ======================= OPTIMIZADOR COMPLETO (Paso 5) =======================
        import numpy as np, pandas as pd
        from types import SimpleNamespace

        # ---------- Helpers UI tabla ----------
        def _styler_base():
            return [
                {"selector":"th", "props":"background:#eef4ff; color:#1a2b4b; font-weight:700;"},
                {"selector":"tbody tr:nth-child(even)", "props":"background:#fafbff;"},
                {"selector":"td, th", "props":"border:1px solid #e9edf5; padding:6px 10px;"},
            ]
        def style_tabla(df, bold_first_col=True, fmt_map=None, highlight_total_label="Total"):
            stl = df.style.set_table_styles(_styler_base())
            if bold_first_col and len(df.columns) > 0:
                stl = stl.set_properties(subset=pd.IndexSlice[:, [df.columns[0]]], **{"font-weight":"700"})
            if fmt_map: stl = stl.format(fmt_map)
            def _row_style(s):
                if str(s.iloc[0]).strip().lower() == str(highlight_total_label).lower():
                    return ["font-weight:700; background-color:#fff7cc;"] * len(s)
                return [""] * len(s)
            stl = stl.apply(_row_style, axis=1)
            try:
                return stl.hide(axis="index")
            except Exception:
                return stl.hide_index()

        # ---------- Señales base necesarias del Paso 1–4 ----------
        if "consumo" not in st.session_state:
            st.error("Falta el consumo del Paso 1.")
            st.stop()
        if "precio_qh_eur_mwh" not in st.session_state:
            st.error("Falta el precio por QH del Paso 4.")
            st.stop()
        if "market" not in st.session_state:
            st.error("Falta la base de mercado en memoria (Paso 4).")
            st.stop()

        # Consumo alineado
        df_c = st.session_state["consumo"].copy()
        df_c["datetime"] = to_naive_utc_index(df_c["datetime"])
        df_c = df_c.dropna(subset=["datetime"]).copy()
        cons = pd.to_numeric(df_c["consumo"], errors="coerce").fillna(0.0).values
        base_dt_index = pd.DatetimeIndex(df_c["datetime"])
        base_dt = base_dt_index.values          # si más abajo necesitas el array
        n_slots = len(base_dt_index)


        # Precio €/MWh por QH alineado
        s_pre = st.session_state["precio_qh_eur_mwh"]
        if isinstance(s_pre, pd.Series):
            s_pre = s_pre.rename("precio_eur_mwh").copy()
            s_pre.index = to_naive_utc_index(s_pre.index)
        else:
            s_pre = s_pre.squeeze().rename("precio_eur_mwh")
            s_pre.index = to_naive_utc_index(s_pre.index)
        precios_uni = s_pre.to_frame().loc[~s_pre.index.duplicated(keep="last")]
        precio_vec = (
            df_c.merge(precios_uni, left_on="datetime", right_index=True, how="left")["precio_eur_mwh"]
            .fillna(method="ffill").fillna(method="bfill").to_numpy()
        )
        # ATR energía €/MWh por QH (si no existe, vector de ceros)
        s_atr = st.session_state.get("atr_qh_eur_mwh", None)
        if s_atr is not None:
            if isinstance(s_atr, pd.Series):
                s_atr = s_atr.rename("atr_qh_eur_mwh").copy()
                s_atr.index = to_naive_utc_index(s_atr.index)
            else:
                s_atr = s_atr.squeeze().rename("atr_qh_eur_mwh")
                s_atr.index = to_naive_utc_index(s_atr.index)

            atr_uni = s_atr.to_frame().loc[~s_atr.index.duplicated(keep="last")]
            atr_vec = (
                df_c.merge(atr_uni, left_on="datetime", right_index=True, how="left")["atr_qh_eur_mwh"]
                .fillna(0.0).to_numpy()
            )
        else:
            atr_vec = np.zeros(n_slots)

        # Excedentes/Generación (si hay FV)
        if st.session_state.get("fv") == "Sí" and "excedentes" in st.session_state and "fecha_exc" in st.session_state:
            df_exc = pd.DataFrame({"datetime": to_naive_utc_index(st.session_state["fecha_exc"]),
                                "exc": pd.to_numeric(st.session_state["excedentes"], errors="coerce")}).dropna()
            excedentes_vec = df_c[["datetime"]].merge(df_exc, on="datetime", how="left")["exc"].fillna(0.0).to_numpy()
        else:
            excedentes_vec = np.zeros(n_slots)
        if st.session_state.get("fv") == "Sí" and "generacion" in st.session_state and "fecha_gen" in st.session_state:
            df_gen = pd.DataFrame({"datetime": to_naive_utc_index(st.session_state["fecha_gen"]),
                                "gen": pd.to_numeric(st.session_state["generacion"], errors="coerce")}).dropna()
            generacion_vec = df_c[["datetime"]].merge(df_gen, on="datetime", how="left")["gen"].fillna(0.0).to_numpy()
        else:
            generacion_vec = np.zeros(n_slots)
        sun_mask = (generacion_vec > 0.0)
        autoconsumo_sin_bess = np.maximum(0.0, generacion_vec - excedentes_vec)
        load_vec = cons + autoconsumo_sin_bess

        # Potencias contratadas por QH (para peak-shaving)
        pot_dict = st.session_state.get("potencias") or {}
        tarifa = (st.session_state.get("tarifa") or "3.0TD").replace(" ", "").upper()
        df_aux = st.session_state.get("consumo_con_mercado")
        if df_aux is None:
            pot_contratada = np.full(n_slots, float(pot_dict.get("P1", 0.0)))
        else:
            dfp_left = df_c.copy(); dfp_left["datetime"] = to_naive_utc_index(dfp_left["datetime"])
            df_aux = df_aux.copy()
            if "datetime" in df_aux.columns:
                df_aux["datetime"] = to_naive_utc_index(df_aux["datetime"])
                dfp = dfp_left.set_index("datetime").join(df_aux.set_index("datetime"), how="left")
            else:
                df_aux.index = to_naive_utc_index(df_aux.index)
                dfp = dfp_left.set_index("datetime").join(df_aux, how="left")
            col_p = "periodos_20td" if tarifa == "2.0TD" else "periodos_no20td"
            per_num = pd.to_numeric(dfp[col_p], errors="coerce")
            miss = per_num.isna()
            if miss.any():
                per_num.loc[miss] = pd.to_numeric(dfp.loc[miss, col_p].astype(str).str.upper().str.extract(r"(\d+)")[0], errors="coerce")
            n_per = 3 if tarifa == "2.0TD" else 6
            per_num = per_num.fillna(1).astype(int).clip(1, n_per)
            per_lbl = "P" + per_num.astype(str)
            pot_contratada = per_lbl.map(lambda p: float(pot_dict.get(p, 0.0))).values

        # Precio de venta FV (si hay FV)
        if st.session_state.get("fv") == "Sí":
            modo_fv = str(st.session_state.get("modalidad_fv", ""))
            if modo_fv == "Precio fijo":
                precio_fv = float(st.session_state.get("precio_fv", {}).get("Precio FV", 0.0))
                precio_venta_fv_vec = np.full(n_slots, precio_fv, dtype=float)
            elif modo_fv == "Indexado":
                cg_fv = float(st.session_state.get("cg_fv", {}).get("Costes Gestion FV", 0.0))
                pv = (precio_vec - cg_fv); pv[pv < 0] = 0.0
                precio_venta_fv_vec = pv
            else:
                precio_venta_fv_vec = np.zeros(n_slots)
        else:
            precio_venta_fv_vec = np.zeros(n_slots)

        def _rank_candidates(cands, P_batt):
            """Ordena por: (1) nº de inversores asc, (2) |P_tot - P_batt| asc, (3) P_tot desc. Devuelve máx. 3."""
            def keyfn(c):
                rows, counts = c
                n_tot = int(sum(counts))
                P_tot = sum(float(r.P_inv) * c_ for r, c_ in zip(rows, counts))
                return (n_tot, abs(P_tot - P_batt), -P_tot)
            cands_sorted = sorted(cands, key=keyfn)
            seen, out = set(), []
            for rows, counts in cands_sorted:
                sig = tuple((r.marca, r.modelo, int(c_)) for r, c_ in zip(rows, counts) if c_ > 0)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append((rows, counts))
                if len(out) >= 3:
                    break
            return out

        def _best_greedy_combos_for_battery(inv_ok_df, P_batt, band=5.0, max_A_try=3, max_units=15):
            """
            Devuelve hasta 3 combinaciones tipo: k*A + 1*B
            - A: inversor más potente (probamos los top max_A_try)
            - k: mínimo para entrar en [P_batt - band, P_batt + band] con un ajuste B
            - B: un único inversor que complete el hueco (puede ser A si cabe)
            - No usa precio para decidir (solo potencia/eficiencia en el sintético).
            """
            if inv_ok_df.empty:
                return []

            inv_sorted = [r[1] for r in inv_ok_df.sort_values("P_inv", ascending=False).iterrows()]
            lo = max(0.0, float(P_batt) - band)
            hi = float(P_batt) + band

            candidates = []

            for A in inv_sorted[:max_A_try]:
                P_A = float(A.P_inv)
                if not np.isfinite(P_A) or P_A <= 0:
                    continue

                # k mínimo (al menos 1) dejando hueco para 1 B si hace falta
                k_base = int(np.floor(lo / max(1e-9, P_A)))
                if k_base < 1:
                    k_base = 1
                if k_base >= max_units:
                    k_base = max_units - 1

                def try_with_k(kb):
                    if kb < 1:
                        return []
                    if kb >= max_units:
                        kb = max_units - 1
                    P_base = kb * P_A
                    out_local = []
                    # Caso en el que kb*A ya entra en rango
                    if lo <= P_base <= hi:
                        out_local.append(([A], [kb]))
                    # Ajuste con un único B dentro del residual
                    res_lo = lo - P_base
                    res_hi = hi - P_base
                    if res_hi < -1e-9:
                        return out_local  # pasado de potencia
                    for B in inv_sorted:
                        P_B = float(B.P_inv)
                        if not np.isfinite(P_B) or P_B <= 0:
                            continue
                        if (kb + 1) > max_units:
                            break
                        if res_lo - 1e-9 <= P_B <= res_hi + 1e-9:
                            out_local.append(([A, B], [kb, 1]))
                    return out_local

                cand_a = []
                cand_a += try_with_k(k_base)
                cand_a += try_with_k(k_base + 1)
                cand_a += try_with_k(k_base - 1)

                cand_a = _rank_candidates(cand_a, P_batt)
                candidates.extend(cand_a)

                if len(candidates) >= 3:
                    break

            return _rank_candidates(candidates, P_batt)

        # ---------- Cargar y filtrar excels ----------
        @st.cache_data(show_spinner=False)
        def _read_excel_cached(path: str):
            return pd.read_excel(path)

        BAT_PATH = "data/bbdd_baterias.xlsx"
        INV_PATH = "data/bbdd_inversores.xlsx"

        df_bat = _read_excel_cached(BAT_PATH)
        df_inv = _read_excel_cached(INV_PATH)
        cb, ci = df_bat.columns, df_inv.columns

        # Map según posiciones que pediste
        BAT = pd.DataFrame({
        "marca":   df_bat[cb[1]].astype(str).str.strip(),
        "modelo":  df_bat[cb[2]].astype(str).str.strip(),
        "mods_S":  pd.to_numeric(df_bat[cb[3]], errors="coerce").fillna(1).astype(int),
        "mods_P":  pd.to_numeric(df_bat[cb[4]], errors="coerce").fillna(1).astype(int),
        "cap_kWh": pd.to_numeric(df_bat[cb[7]], errors="coerce"),
        "DoD":     pd.to_numeric(df_bat[cb[8]], errors="coerce"),
        "P_batt":  pd.to_numeric(df_bat[cb[11]], errors="coerce"),
        "eta_b":   pd.to_numeric(df_bat[cb[13]], errors="coerce"),
        "V_nom":   pd.to_numeric(df_bat[cb[14]], errors="coerce"),
        "precio":  pd.to_numeric(df_bat[cb[17]], errors="coerce"),
        }).dropna(subset=["cap_kWh","DoD","P_batt","eta_b","V_nom","precio"]).reset_index(drop=True)

        INV = pd.DataFrame({
            "marca":  df_inv[ci[1]],
            "modelo": df_inv[ci[2]],
            "fase":   df_inv[ci[4]].astype(str),
            "Vmin":   pd.to_numeric(df_inv[ci[5]], errors="coerce"),
            "Vmax":   pd.to_numeric(df_inv[ci[6]], errors="coerce"),
            "P_inv":  pd.to_numeric(df_inv[ci[7]], errors="coerce"),
            "P_fv":   pd.to_numeric(df_inv[ci[9]], errors="coerce").fillna(0.0),
            "eta_i":  pd.to_numeric(df_inv[ci[10]], errors="coerce"),
            "precio": pd.to_numeric(df_inv[ci[11]], errors="coerce"),
        }).dropna(subset=["fase","Vmin","Vmax","P_inv","eta_i","precio"]).reset_index(drop=True)

        def normaliza_fase(x):
            s = str(x).strip().lower()
            return "tri" if any(k in s for k in ["tri","3f","3-f","trif"]) else "mono"
        INV["fase"] = INV["fase"].apply(normaliza_fase)

        instalacion = (st.session_state.get("instalacion","Trifásica") or "").lower()
        fase_req = "mono" if "mono" in instalacion else "tri"
        INV = INV[INV["fase"] == fase_req].reset_index(drop=True)

        # Arregla posibles filas con Vmin>Vmax
        swap_mask = INV["Vmin"] > INV["Vmax"]
        if swap_mask.any():
            vmin = INV.loc[swap_mask,"Vmax"].copy(); vmax = INV.loc[swap_mask,"Vmin"].copy()
            INV.loc[swap_mask,"Vmin"] = vmin; INV.loc[swap_mask,"Vmax"] = vmax

        if INV.empty:
            st.error("No hay inversores compatibles con la fase seleccionada (Mono/Tri).")
            st.stop()

            # ---------- Crear contexto (foto del estado para los hilos) ----------
        ctx = build_ctx_from_session()
        _prepare_base_signals_from_session_state()

        # ---------- Bucle de búsqueda (PARALELO) ----------
        best_global, mejor_TIR_global, mejor_VAN_global = None, -1e18, -1e18

        tareas = []
        diagnostico = []

        # 1) Filtro de capacidad de batería (como en v2)
        cons_anual = float(st.session_state["consumo"]["consumo"].sum())
        cap_min = cons_anual / 35040.0
        cap_max = cons_anual / 365.0

        MAX_INV_UNITS = 15  # tope global de inversores en una combinación

        for _, b in BAT.iterrows():
            cap_batt = float(b.cap_kWh)
            P_batt   = float(b.P_batt)

            potencias = st.session_state.get("potencias", {}) or {}
            P6 = float(potencias.get("P6", float("nan")))

            if np.isfinite(P6) and P6 > 0.0 and (P_batt > P6 + 1e-9):
                diagnostico.append(
                    f"❌ {b.marca} {b.modelo}: P_batt={P_batt:.1f} kW > P6={P6:.1f} kW (no se simula)"
                )
                continue
            
            # Capacidad fuera de rango → descarta batería
            if cap_batt < cap_min or cap_batt > cap_max:
                diagnostico.append(
                    f"❌ {b.marca} {b.modelo}: {cap_batt:.1f} kWh fuera de [{cap_min:.1f}, {cap_max:.1f}]"
                )
                continue

            # Compatibilidad por tensión (tu filtro existente)
            inv_ok = INV[(INV["Vmin"] <= b.V_nom) & (b.V_nom <= INV["Vmax"])].reset_index(drop=True)
            if inv_ok.empty:
                diagnostico.append(
                    f"❌ {b.marca} {b.modelo}: sin inversores compatibles (V_nom={b.V_nom})"
                )
                continue

            # Greedy: repite A y ajusta con 1 B → máximo 3 combos por batería
            combos = _best_greedy_combos_for_battery(
                inv_ok, P_batt, band=5.0, max_A_try=3, max_units=MAX_INV_UNITS
            )

            if not combos:
                diagnostico.append(
                    f"❌ {b.marca} {b.modelo}: sin combos en ±5 kW de {P_batt:.1f} kW."
                )
                continue

            # Convierte cada combo en "inversor sintético" y crea tarea
            for rows, counts in combos:
                inv_mix = _build_combo_namespace(rows, counts)
                n_total = int(sum(counts))
                tareas.append((b.to_dict(), vars(inv_mix), n_total, ctx))

        st.write(f"✅ Combinaciones simuladas: {len(tareas)}")

        from types import SimpleNamespace

        def _runner(args):
            b_d, inv_d, n, ctx_loc = args
            b_row   = SimpleNamespace(**b_d)
            inv_row = SimpleNamespace(**inv_d)
            return evaluar_combinacion(b_row, inv_row, n, dict(ctx_loc), return_sim=False)

        max_workers = max(1, os.cpu_count() - 1)
        results_local = []

        if not tareas:
            st.error("No hay combinaciones que evaluar (revisa compatibilidad).")
            st.stop()

        # Ejecutar en paralelo por procesos (como en v2)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_runner, t) for t in tareas]
            for fu in as_completed(futs):
                try:
                    results_local.append(fu.result())
                except Exception as e:
                    diagnostico.append(
                        "🔻 Fallo en combinación:\n"
                        + f"{type(e).__name__}: {e}\n"
                        + traceback.format_exc()
                    )

        # ---------- Seleccionar la mejor combinación según la TIR (y VAN como respaldo) ----------
        if not results_local:
            st.error("No se obtuvo ninguna simulación válida de las combinaciones evaluadas.")
            st.stop()

        def _score_res(res):
            """Prioriza TIR; si no hay TIR válida, usa VAN para desempatar."""
            import numpy as np
            tir = res.get("TIR", np.nan)
            van = res.get("VAN", -1e18)
            # Los que NO tienen TIR válida se van al final (flag -1)
            if tir is None or not np.isfinite(tir):
                return (-1, van)
            # Los que sí tienen TIR válida: flag 0, luego ordenamos por TIR y VAN
            return (0, tir, van)

        best_global = max(results_local, key=_score_res)
        mejor_TIR_global = best_global.get("TIR", np.nan)
        mejor_VAN_global = best_global.get("VAN", np.nan)

        import io

        # ---------- Resumen económico de TODAS las simulaciones ----------

        # Costes de la situación inicial (los mismos para todas las combinaciones)
        base_TE          = float(st.session_state.get("coste_mercado_te",        np.nan))
        base_ATR_energia = float(st.session_state.get("atr_total_energia",      np.nan))
        base_ATR_pot     = float(st.session_state.get("atr_total_potencia",     np.nan))
        base_excesos     = float(st.session_state.get("excesos_total",          np.nan))
        base_FNEE        = float(st.session_state.get("fnee_total",             np.nan))
        base_IEE         = float(st.session_state.get("iee_total",              np.nan))
        base_ing_FV      = -float(st.session_state.get("ingresos_fv",           0.0))  # mismo signo que en det_bess
        base_total       = float(st.session_state.get("costes_iniciales_total", np.nan))

        filas_resumen = []
        for res in results_local:
            try:
                ECO  = res.get("ECO")
                HW   = res.get("HW", {})
                diag = res.get("diag", {})
                det  = res.get("det_bess", {}) or {}

                filas_resumen.append({
                    # Identificación de la solución
                    "Bat_marca":          res.get("bat_marca", ""),
                    "Bat_modelo":         res.get("bat_modelo", ""),
                    "Capacidad_bat_kWh":  float(HW.get("E_CAP",  np.nan)),
                    "P_bat_kW":           float(HW.get("P_BATT", np.nan)),
                    "Inv_marca":          res.get("inv_marca", ""),
                    "Inv_modelo":         res.get("inv_modelo", ""),
                    "N_inversores":       int(res.get("n_inv", 0)),

                    # Económicos principales
                    "Capex_€":               float(diag.get("capex", np.nan)),
                    "Ahorro_anual_€":        float(res.get("ahorro_anual", np.nan)),
                    "VAN_€":                 float(res.get("VAN", np.nan)),
                    "TIR_%":                 float(res.get("TIR", np.nan)),
                    "Payback_simple_años":   float(diag.get("payback", np.nan)),

                    # Indicadores de uso de la batería
                    "kWh_cargados_red":      float(diag.get("kWh_carga_red", np.nan)),
                    "kWh_descargados":       float(diag.get("kWh_descarga",   np.nan)),
                    "Ciclos_equivalentes":   float(diag.get("ciclos_eq",      np.nan)),
                    "Spread_compra-venta_€/MWh": float(diag.get("spread", np.nan)),

                    # --- Costes ANTES (situación inicial, iguales para todas las filas) ---
                    "Antes_TE_€":           base_TE,
                    "Antes_ATR_energia_€":  base_ATR_energia,
                    "Antes_ATR_potencia_€": base_ATR_pot,
                    "Antes_Excesos_€":      base_excesos,
                    "Antes_FNEE_€":         base_FNEE,
                    "Antes_IEE_€":          base_IEE,
                    "Antes_Ingresos_FV_€":  base_ing_FV,
                    "Antes_Total_€":        base_total,

                    # --- Costes DESPUÉS (con batería, específicos de cada simulación) ---
                    "Desp_TE_€":           float(det.get("TE",           np.nan)),
                    "Desp_ATR_energia_€":  float(det.get("ATR_energia",  np.nan)),
                    "Desp_ATR_potencia_€": float(det.get("ATR_potencia", np.nan)),
                    "Desp_Excesos_€":      float(det.get("Excesos",      np.nan)),
                    "Desp_FNEE_€":         float(det.get("FNEE",         np.nan)),
                    "Desp_IEE_€":          float(det.get("IEE",          np.nan)),
                    "Desp_Ingresos_FV_€":  float(det.get("Ingresos_FV",  np.nan)),
                    "Desp_Total_€":        float(det.get("Total",        np.nan)),
                })
            except Exception:
                # Si alguna combinación viene rara, la saltamos sin romper todo
                continue

        if filas_resumen:
            df_resumen = pd.DataFrame(filas_resumen)

            # Ordenar como ranking (mejor a peor combinación)
            df_resumen = df_resumen.sort_values(
                by=["TIR_%", "VAN_€", "Ahorro_anual_€"],
                ascending=[False, False, False],
                na_position="last"
            ).reset_index(drop=True)

            # Guardamos en sesión para usarlo en otros sitios si quieres
            st.session_state["resumen_simulaciones"] = df_resumen

            st.markdown("### Resumen económico de todas las combinaciones evaluadas")

            # Vista rápida en pantalla (primeras 30 filas)
            st.dataframe(df_resumen.head(30), use_container_width=True)

            # ---------- Botón descarga en Excel ----------
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_resumen.to_excel(writer, index=False, sheet_name="Simulaciones")
            buffer.seek(0)

            st.download_button(
                label="⬇️ Descargar resumen de simulaciones (Excel)",
                data=buffer,
                file_name="resumen_simulaciones_bess.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info("No se ha podido construir el resumen de simulaciones (lista vacía).")

        # ---------- Selección del mejor global con la MISMA lógica que en v2 ----------
        best_global, mejor_TIR_global, mejor_VAN_global = None, -1e18, -1e18

        for res in results_local:
            tir = res.get("TIR", np.nan)
            van = res.get("VAN", np.nan)
            tir_ok, van_ok = np.isfinite(tir), np.isfinite(van)
            if tir_ok and (best_global is None or tir > mejor_TIR_global):
                mejor_TIR_global, mejor_VAN_global, best_global = tir, (van if van_ok else -1e18), res
            elif (not tir_ok) and van_ok and (best_global is None or (not np.isfinite(mejor_TIR_global) and van > mejor_VAN_global)):
                mejor_VAN_global, best_global = van, res

        if best_global is None:
            st.error("No se encontró ninguna combinación válida (revisa fase o compatibilidad de tensiones).")
            st.stop()

        # ---------- Inyecta la mejor y muestra resumen ----------
        ECO = best_global["ECO"]
        HW  = best_global["HW"]

        ahorro = float(best_global["ahorro_anual"])
        capex  = float(ECO.BASE_IMPONIBLE)
        payback_simple = capex / ahorro if ahorro > 0 else float("inf")

        # 🔁 Simula SOLO la ganadora y adjunta el DF para la UI (ahora con MILP v3)
        df_sim_best = simular_bess_milp(HW)
        best_global["SIM"] = df_sim_best

        # Guarda en session_state como ya hacías
        st.session_state["simul_bess_df"] = df_sim_best
        st.session_state["ECO"] = ECO
        st.session_state["HW"]  = HW

        tir_txt = (f"TIR: {best_global['TIR']:.2f}%  · "
        if np.isfinite(best_global["TIR"])
        else "TIR: —  · ")

        # --- Banner del óptimo con capacidades/potencias ---
        cap_kwh = float(HW["E_CAP"])
        p_batt  = float(HW["P_BATT"])
        p_inv   = float(HW["P_INV"])

        # ====== ESTILO AZUL UNIFICADO PARA TABLAS (PASO 5) ======
        from typing import Optional, Sequence

        _COLOR_HEADER_BG = "#eef4ff"   
        _COLOR_HEADER_TX = "#1a2b4b"   
        _COLOR_BORDER    = "#e9edf5"   
        _COLOR_ZEBRA     = "#fafbff"   

        def _guess_num_format(colname: str) -> Optional[str]:
            """Devuelve formato por nombre de columna: € o kWh si procede."""
            name = colname.lower()
            if any(k in name for k in ["eur", "€", "importe", "coste", "precio", "pago", "ahorro"]):
                return "{:,.2f} €"
            if any(k in name for k in ["kwh", "energ", "consumo", "gener", "descarga", "carga", "excedente"]):
                return "{:,.3f} kWh"
            if any(k in name for k in ["%", "porcentaje", "tasa"]):
                return "{:.2%}"
            return None

        def _formatters_from_df(df: pd.DataFrame):
            from pandas.api.types import is_numeric_dtype
            fmts = {}
            for c in df.columns:
                f = _guess_num_format(c)
                if not f:
                    continue
                # Solo aplicar formato si la columna es numérica
                if is_numeric_dtype(df[c]):
                    fmts[c] = f
                else:
                    # Si no es numérica pero la mayoría serían números al convertir, la dejamos pasar;
                    # si no, NO formateamos (evita el ValueError en columnas ya textuales como "1.234 €")
                    s_num = pd.to_numeric(df[c], errors="coerce")
                    ratio_num = s_num.notna().mean() if len(s_num) else 0.0
                    if ratio_num >= 0.8:
                        fmts[c] = f
            return fmts

        def style_blue(
            df: pd.DataFrame,
            *,
            bold_first_data_row: bool = False,
            total_row_labels: Sequence[str] = ("TOTAL",),
            bold_total_row: bool = False,
            caption: Optional[str] = None,
        ) -> pd.io.formats.style.Styler:
            """Crea un Styler con tema azul. Negrita opcional en primera fila de datos y/o fila TOTAL."""
            sty = df.style

            # Formatos numéricos sugeridos por nombre de columna
            fmts = _formatters_from_df(df)
            if fmts:
                sty = sty.format(fmts, na_rep="—")

            # Estilos base (cabecera y bordes)
            sty = sty.set_table_styles([
            # Cabecera de columnas (azul + negrita)
            {"selector": "th",
            "props": [
                ("background-color", _COLOR_HEADER_BG),
                ("color", f"{_COLOR_HEADER_TX} !important"),
                ("font-weight", "700 !important"),
                ("border-bottom", f"1px solid {_COLOR_BORDER}")
            ]},
            # Bordes y padding homogéneos
            {"selector": "td, th",
            "props": [
                ("border", f"1px solid {_COLOR_BORDER}"),
                ("padding", "6px 10px")]},])
            # Zebra suave
            def _zebra(rows):
                return [f"background-color: {_COLOR_ZEBRA}" if i % 2 else "" for i in range(len(rows))]
            sty = sty.apply(_zebra, axis=0)

            # Negrita en primera fila de datos (si se pide)
            if bold_first_data_row and len(df) > 0:
                def _bold_first_row(_row):
                    return ["font-weight:600" if _row.name == df.index[0] else "" ] * len(df.columns)
                sty = sty.apply(_bold_first_row, axis=1)

            # Negrita en filas TOTAL (por índice o por primera columna)
            if bold_total_row and len(df) > 0:
                total_set = set(str(x).strip().upper() for x in total_row_labels)
                first_col = df.columns[0] if len(df.columns) > 0 else None

                def _bold_total(_row):
                    by_index = str(_row.name).strip().upper() in total_set
                    by_first = False
                    if first_col is not None:
                        try:
                            by_first = str(_row[first_col]).strip().upper() in total_set
                        except Exception:
                            by_first = False
                    is_total = by_index or by_first
                    return ["font-weight:700" if is_total else ""] * len(df.columns)

                sty = sty.apply(_bold_total, axis=1)

            if caption:
                sty = sty.set_caption(caption)
            return sty
        
        # ===== FILA DE ICONOS DE DESCARGA (Batería e Inversores) =====
        from pathlib import Path
        import glob, base64

        # CSS: fila compacta sin huecos + “píldoras” con icono
        st.markdown("""
        <style>
        .icon-row{ display:flex; align-items:center; gap:8px; margin-top:-6px; margin-bottom:10px; }
        .icon-title{ font-size:.90rem; color:#6b778c; line-height:1; white-space:nowrap; }
        .icon-pill{
        display:inline-flex; width:28px; height:28px; border-radius:999px;
        align-items:center; justify-content:center; text-decoration:none;
        background:#F2F4F7; border:1px solid #DDE3EA; color:#1a2b4b; box-shadow:none;
        }
        .icon-pill:hover{ background:#EBEEF3; }
        .icon-pill:active{ transform:translateY(1px); }
        .icon-pill svg{ width:16px; height:16px; }
        </style>
        """, unsafe_allow_html=True)

        @st.cache_data(show_spinner=False)
        def _load_pdf_bytes(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        def _base_dir():
            return Path(__file__).resolve().parent

        def _find_folder(*relative_candidates):
            for rel in relative_candidates:
                p = (_base_dir() / rel).resolve()
                if p.exists():
                    return p
            return _base_dir()

        def _find_pdf_like(folder: Path, *tokens: str) -> Path | None:
            tokens = [t for t in (t.strip() for t in tokens) if t]
            for t in tokens:
                for pat in (f"*{t}*.pdf", f"*{t.replace(' ','*')}*.pdf"):
                    hits = sorted(glob.glob(str(folder / pat)))
                    if hits:
                        return Path(hits[0])
            return None

        def _pdf_link_html(path: Path, title: str) -> str:
            # data:URL para descarga directa, sin usar st.download_button
            b64 = base64.b64encode(_load_pdf_bytes(str(path))).decode()
            svg = """<svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3v10.17l3.59-3.58L17 11l-5 5-5-5 1.41-1.41L11 13.17V3h1zM5 19h14v2H5z"/>
            </svg>"""
            return f'<a class="icon-pill" href="data:application/pdf;base64,{b64}" download="{path.name}" title="{title}">{svg}</a>'

        def render_bateria_icons(marca: str, modelo: str):
            """Una fila con etiqueta + icono para la batería, pegada a la tabla."""
            folder = _find_folder("../Datasheets Bat-Inv/Baterias", "Datasheets Bat-Inv/Baterias")
            MAP = {
                "ebick lv":  "EBick LV - Cegasa - LV - Bat.pdf",
                "escal hv":  "Escal HV - Cegasa - HV - Bat.pdf",
                "expand hv": "Expand HV - Cegasa - HV - Bat.pdf",
                "us5000":    "US5000 - Pylontech - LV - Bat.pdf",
            }
            key = (modelo or "").lower().strip()
            pdf = None
            if key in MAP and (folder / MAP[key]).exists():
                pdf = (folder / MAP[key])
            if pdf is None:
                pdf = _find_pdf_like(folder, modelo, f"{marca} {modelo}")
            if pdf and pdf.exists():
                label = '<span class="icon-title">Descargar datasheet batería</span>'
                html = f'<div class="icon-row no-print">{label}{_pdf_link_html(pdf, f"Datasheet batería ({modelo})")}</div>'
                st.markdown(html, unsafe_allow_html=True)

        def render_inversores_icons(filas_inversores: list[dict]):
            """Etiqueta + 1 icono por modelo (sin duplicados)."""
            folder = _find_folder("../Datasheets Bat-Inv/Inversores", "Datasheets Bat-Inv/Inversores")
            vistos, links = set(), []
            for row in filas_inversores:
                marca  = str(row.get("Marca","")).strip()
                modelo = str(row.get("Modelo","")).strip()
                clave  = (marca.lower(), modelo.lower())
                if clave in vistos:
                    continue
                vistos.add(clave)
                pdf = None
                if marca.lower() == "solis":
                    token = "S6-EH3P" if "S6-EH3P" in modelo else modelo
                    pdf = _find_pdf_like(folder, token)
                else:
                    pdf = _find_pdf_like(folder, modelo, f"{marca} {modelo}")
                if pdf and pdf.exists():
                    links.append(_pdf_link_html(pdf, f"Datasheet inversor ({modelo})"))
            if links:
                label = '<span class="icon-title">Descargar datasheets inversores</span>'
                st.markdown(f'<div class="icon-row no-print">{label}{"".join(links)}</div>', unsafe_allow_html=True)

        # ====== FIN HELPERS ======
        
        # === Catálogo de la solución ganadora + mini presupuesto ===
        # Helpers que faltaban
        def _fmt_eur2(x):
            try:
                return f"{fmt_eur(float(x), 2)} €"
            except Exception:
                return x

        # Recupero objetos de la mejor combinación
        ECO = best_global["ECO"]
        HW  = best_global["HW"]

        # Strings de identificación
        inv_mix_str = str(best_global.get("inv_marca", ""))
        bat_marca   = str(best_global.get("bat_marca", "")).strip()
        bat_modelo  = str(best_global.get("bat_modelo", "")).strip()

        def _norm(s): return str(s).strip().lower()

        if not BAT.empty:
            mask = (BAT["marca"].str.strip().str.lower()==_norm(bat_marca)) & \
                (BAT["modelo"].str.strip().str.lower()==_norm(bat_modelo))
            if mask.any():
                bat_row = BAT.loc[mask].iloc[0]
            else:
                bat_row = pd.Series({"marca": bat_marca, "modelo": bat_modelo,
                                    "mods_S": 1, "mods_P": 1, "precio": ECO.COSTE_BATERIAS})
        else:
            bat_row = pd.Series({"marca": bat_marca, "modelo": bat_modelo,
                                "mods_S": 1, "mods_P": 1, "precio": ECO.COSTE_BATERIAS})

        # Costes para el mini presupuesto
        coste_bat  = float(ECO.COSTE_BATERIAS)
        coste_inv  = float(ECO.COSTE_INVERSORES)
        coste_ems  = float(ECO.COSTE_EMS)
        coste_inst = float(ECO.COSTE_INSTAL)
        base_imp   = float(ECO.BASE_IMPONIBLE)


        # ---------- TABLA BATERÍA ----------
        mods_S = int(pd.to_numeric(best_global.get("bat_mods_S", getattr(bat_row, "mods_S", 1)), errors="coerce") or 1)
        mods_P = int(pd.to_numeric(best_global.get("bat_mods_P", getattr(bat_row, "mods_P", 1)), errors="coerce") or 1)

        bat_tbl = pd.DataFrame([{
            "Tipo": "Batería",
            "Marca": bat_row.marca,
            "Modelo": bat_row.modelo,
            "Configuración": f"S{mods_S} · P{mods_P}",
            "Capacidad": f"{float(HW['E_CAP']):,.1f} kWh".replace(",", "."),
            "Potencia batería": f"{float(HW['P_BATT']):,.1f} kW".replace(",", "."),
            "Precio": _fmt_eur2(float(ECO.COSTE_BATERIAS)),
        }])

        # ---- BATERÍA ----
        sty_bat = style_blue(
            bat_tbl,
            bold_first_data_row=False,       # primera fila en negrita
            bold_total_row=False,           # SIN total
            caption="Batería seleccionada"
        )
        st.table(sty_bat)
        render_bateria_icons(marca=str(bat_row.marca), modelo=str(bat_row.modelo))

        # ---------- TABLA INVERSORES (sin subtotal) ----------
        inv_rows = []
        if inv_mix_str:
            import re
            pat = r"^\s*(\d+)x\s+([^\s]+)\s+(.+?)\s*$"  # "4x Solis S6-..." => unidades, marca, modelo
            for part in [p.strip() for p in inv_mix_str.split("+")]:
                m = re.match(pat, part)
                if not m:
                    continue
                unidades = int(m.group(1))
                marca_i  = m.group(2)
                modelo_i = m.group(3)
                inv_match = INV[(INV["marca"] == marca_i) & (INV["modelo"] == modelo_i)]
                if inv_match.empty:
                    inv_match = INV[INV["marca"] == marca_i].sort_values("P_inv", ascending=False).head(1)
                inv_i = inv_match.iloc[0]
                p_unit = float(inv_i.precio)
                inv_rows.append({
                    "Tipo": "Inversor",
                    "Marca": marca_i,
                    "Modelo": modelo_i,
                    "Unidades": unidades,
                    "Pot. unitaria": f"{float(inv_i.P_inv):,.1f} kW".replace(",", "."),
                    "Precio unitario": _fmt_eur2(p_unit),
                })

        if inv_rows:
            inv_tbl = pd.DataFrame(inv_rows)
            sty_inv = style_blue(
                inv_tbl,
                bold_first_data_row=False,   # encabezado en negrita (no la primera fila de datos)
                bold_total_row=False,        # sin total en catálogo
                caption="Inversores seleccionados"
            )
            st.table(sty_inv)
            render_inversores_icons(inv_rows)

        # ---------- MINI PRESUPUESTO ----------
        st.markdown("##### Presupuesto")
        bud = pd.DataFrame([
            ("Coste batería",        coste_bat),
            ("Coste inversores",     coste_inv),
            ("Coste EMS",            coste_ems),
            ("Coste instalación",    coste_inst),
            ("TOTAL base imponible", base_imp),
        ], columns=["Concepto","Importe (€)"])

        sty_bud = style_blue(
            bud,
            bold_first_data_row=False,
            total_row_labels=("TOTAL BASE IMPONIBLE","TOTAL base imponible","TOTAL"),
            bold_total_row=True,
            caption="Presupuesto de la propuesta"
        )
        st.table(sty_bud)

        # === Auditoría económica (desglose y sanity check) ===
        # --- Situación inicial (Paso 4) → construir det_base con los mismos nombres ---
        det_base = dict(
            TE              = float(st.session_state.get("coste_mercado_te", 0.0)),      # TE neto (como en Paso 4)
            ATR_energia     = float(st.session_state.get("atr_total_energia", 0.0)),
            ATR_potencia    = float(st.session_state.get("atr_total_potencia", 0.0)),
            Excesos         = float(st.session_state.get("excesos_total", 0.0)),
            FNEE            = float(st.session_state.get("fnee_total", 0.0)),
            IEE             = float(st.session_state.get("iee_total", 0.0)),
            Ingresos_FV     = -float(st.session_state.get("ingresos_fv", 0.0)),          # negativo en la base
            Total           = float(st.session_state.get("costes_iniciales_total", 0.0)),
        )
        best_global["det_base"] = det_base  # ← para que la auditoría lo encuentre

        # ================== ESTUDIO ECONÓMICO (solo propuesta ganadora, UI con 3 pestañas) ==================
        st.markdown("## Estudio económico")

        # --- Helpers financieros (robustos y coherentes con la simulación) ---
        def _npv(rate, cashflows):
            return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

        def _irr(cashflows, lower=-0.95, upper=3.0, tol=1e-7, max_iter=200):
            """IRR por bisección (robusto). lower > -1 para evitar singularidades."""
            def f(r): return sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cashflows))
            f_low, f_up = f(lower), f(upper)
            expand = 0
            while f_low * f_up > 0 and upper < 100 and expand < 20:
                upper *= 1.5
                f_up = f(upper)
                expand += 1
            if f_low * f_up > 0:
                return np.nan
            for _ in range(max_iter):
                mid = (lower + upper) / 2.0
                f_mid = f(mid)
                if abs(f_mid) < tol or (upper - lower) / 2.0 < tol:
                    return mid
                if f_low * f_mid <= 0:
                    upper, f_up = mid, f_mid
                else:
                    lower, f_low = mid, f_mid
            return mid

        def _payback(cash):
            """Devuelve el payback en años con decimales (interpolado linealmente)."""
            acc = cash[0]  # incluir inversión inicial (negativa)
            for t in range(1, len(cash)):
                prev_acc = acc
                acc += cash[t]
                if acc >= 0:
                    if cash[t] == 0:
                        return float(t)
                    frac = (0 - prev_acc) / (acc - prev_acc)
                    return (t - 1) + frac
            return float("inf")

        def _payback_descontado(cash, tasa):
            acc = 0.0
            for t, cf in enumerate(cash):
                acc += cf / ((1 + tasa) ** t)
                if acc >= 0:
                    return t
            return float("inf")

        def _build_cashflows(ECO, ahorro_base, escenario: str):
            """Genera flujos 0..N con crecimiento (IPC+electricidad) y degradación, menos OPEX, + valor residual al final."""
            n = int(ECO.VIDA_UTIL_ANIOS)
            cash = [0.0] * (n + 1)
            cash[0] = -float(ECO.BASE_IMPONIBLE)

            if escenario == "Moderado":
                ipc, elec = float(ECO.IPC_MOD), float(ECO.ELEC_MOD)
            elif escenario == "Optimista":
                ipc, elec = float(ECO.IPC_MOD) + float(ECO.IPC_OPT_DELTA), float(ECO.ELEC_OPT)
            else:  # Pesimista
                ipc, elec = float(ECO.IPC_MOD) + float(ECO.IPC_PES_DELTA), float(ECO.ELEC_PES)

            growth = (1 + ipc) * (1 + elec)
            degr   = float(ECO.DEGRAD_ANUAL)
            opex   = float(ECO.OPEX_ANUAL)
            tasa   = float(ECO.TASA_DESCUENTO)

            for t in range(1, n + 1):
                ahorro_t = (float(ahorro_base) * ((1 - degr) ** (t - 1)) * (growth ** (t - 1))) - opex
                cash[t] = ahorro_t
            cash[-1] += float(ECO.VALOR_RESIDUAL)

            van = _npv(tasa, cash)
            tir = _irr(cash)
            pb  = _payback(cash)
            pbd = _payback_descontado(cash, tasa)

            return {
                "cash": cash,
                "VAN": float(van),
                "TIR": (float(tir) * 100.0) if np.isfinite(tir) else np.nan,
                "Payback": float(pb),
                "Payback_desc": float(pbd),
                "params": {"ipc": ipc, "elec": elec, "growth": growth, "degr": degr, "opex": opex, "tasa": tasa}
            }

        ECO = best_global["ECO"]
        ahorro_base = float(best_global["ahorro_anual"])

        escenarios = ["Pesimista", "Moderado", "Optimista"]
        res_esc = {e: _build_cashflows(ECO, ahorro_base, e) for e in escenarios}

        # === RESUMEN INTRODUCTORIO POR ESCENARIO (TIR, VAN, Payback) + CONDICIONES ===

        def _fmt_eur_local(x):
            try:
                return f"{fmt_eur(float(x), 2)} €"
            except Exception:
                return x

        # Tabla KPI (una fila por escenario)
        kpi_rows = []
        for esc in escenarios:
            r = res_esc[esc]
            kpi_rows.append({
                "Escenario": esc,
                "TIR": (f"{r['TIR']:.2f}%" if np.isfinite(r["TIR"]) else "—"),
                "VAN": _fmt_eur_local(r["VAN"]),
                "Payback (años)": (f"{r['Payback']:.2f}" if np.isfinite(r["Payback"]) else "—"),
            })
        df_kpi = pd.DataFrame(kpi_rows, columns=["Escenario","TIR","VAN","Payback (años)"])
        st.table(style_blue(
            df_kpi,
            bold_first_data_row=False,
            bold_total_row=False,
            caption="Indicadores financieros por escenario"
        ))

        # Comentario compacto de condiciones por escenario (sin OPEX ni tasa descuento)
        st.markdown("###### Condiciones de los escenarios")
        txt_conds = []
        for esc in escenarios:
            p = res_esc[esc]["params"]
            txt_conds.append(
                f"**{esc}** — Aumento IPC anual: {p['ipc']*100:.2f} %, "
                f"Aumento coste electricidad anual: {p['elec']*100:.2f} %, "
                f"Degradación anual de la batería: {p['degr']*100:.3f} %"
            )

        # Estilo “comentario” discreto
        st.info("  \n".join(txt_conds))

        # --- Preconstruimos las tablas por escenario ---
        df_proj_by_esc = {}
        for esc in escenarios:
            r = res_esc[esc]
            cash = r["cash"]
            n = len(cash) - 1

            años = list(range(0, n + 1))
            flujo = cash
            tes_acum = np.cumsum(flujo).tolist()

            # SOLO las columnas solicitadas (nada de “descontado”, “factores”, ni OPEX)
            df = pd.DataFrame({
                "Año": años,
                "Flujo (€)": flujo,
                "Tesorería acumulada (€)": tes_acum,
            })
            df_proj_by_esc[esc] = df

        # --- UI: UNA sola tabla con 3 pestañas ---
        st.markdown("##### Proyección inversión durante su vida útil")
        # Orden deseado manualmente
        orden_esc = ["Moderado", "Optimista", "Pesimista"]
        tab_mod, tab_opt, tab_pes = st.tabs(orden_esc)

        # Recorre las pestañas en ese orden fijo
        for esc, tab in zip(orden_esc, [tab_mod, tab_opt, tab_pes]):
            with tab:
                df = df_proj_by_esc[esc]
                sty = style_blue(
                    df,
                    bold_first_data_row=False,
                    bold_total_row=False,
                    caption=f"Proyección {len(df)-1} años – {esc}"
                )
                st.table(sty)
        # ================== /ESTUDIO ECONÓMICO ==================

        # ================== Desglose económico y comprobaciones (sin expander) ==================
        import numpy as np, pandas as pd

        # === Tabla de comparación de costes con mismo estilo que catálogo ===
        st.markdown("##### Comparación costes antes VS después", unsafe_allow_html=True)

        det_base = best_global.get("det_base", {}) or {}
        det_bess = best_global.get("det_bess", {}) or {}

        conceptos = [
            "Mercado – término energía",
            "ATR energía",
            "ATR potencia",
            "Excesos de potencia",
            "FNEE",
            "Impuesto electricidad",
            "Ingresos FV (−)",
        ]
        antes = [
            float(det_base.get("TE", 0.0)),
            float(det_base.get("ATR_energia", 0.0)),
            float(det_base.get("ATR_potencia", 0.0)),
            float(det_base.get("Excesos", 0.0)),
            float(det_base.get("FNEE", 0.0)),
            float(det_base.get("IEE", 0.0)),
            float(det_base.get("Ingresos_FV", 0.0)),
        ]
        despues = [
            float(det_bess.get("TE", 0.0)),
            float(det_bess.get("ATR_energia", 0.0)),
            float(det_bess.get("ATR_potencia", 0.0)),
            float(det_bess.get("Excesos", 0.0)),
            float(det_bess.get("FNEE", 0.0)),
            float(det_bess.get("IEE", 0.0)),
            float(det_bess.get("Ingresos_FV", 0.0)),
        ]

        df_cmp = pd.DataFrame({
            "Concepto": conceptos,
            "Antes (€)": antes,
            "Después (€)": despues,
        })
        df_cmp["Ahorro (€)"] = df_cmp["Antes (€)"] - df_cmp["Después (€)"]

        # Totales
        total_antes  = float(det_base.get("Total", df_cmp["Antes (€)"].sum()))
        total_desp   = float(det_bess.get("Total", df_cmp["Después (€)"].sum()))
        total_ahorro = total_antes - total_desp
        df_cmp = pd.concat([
            df_cmp,
            pd.DataFrame([{
                "Concepto": "TOTAL",
                "Antes (€)": total_antes,
                "Después (€)": total_desp,
                "Ahorro (€)": total_ahorro,
            }])
        ], ignore_index=True)

        sty_desg = style_blue(
            df_cmp,                       
            bold_first_data_row=False,
            total_row_labels=("TOTAL", "Total"),
            bold_total_row=True,
            caption="Comparativa de costes para el calculo del ahorro"
        )
        st.table(sty_desg)

        st.markdown("## Análisis del funcionamiento técnico del sistema")
        import numpy as np
        # --- Cálculo / recuperación de ciclos equivalentes ---
        diag = best_global.get("diag", {}) or {}
        ciclos_eq_anuales = float(diag.get("ciclos_eq", np.nan))

        # Si no hubiera venido de diag (por seguridad), lo recalculamos desde la simulación
        if not np.isfinite(ciclos_eq_anuales) or ciclos_eq_anuales <= 0:
            df_sim_tmp = best_global.get("SIM")
            if df_sim_tmp is not None:
                try:
                    kWh_descarga = float(
                        pd.to_numeric(df_sim_tmp["descarga_kWh"], errors="coerce").fillna(0.0).sum()
                    )
                    e_cap = float(HW["E_CAP"])
                    ciclos_eq_anuales = kWh_descarga / max(e_cap, 1e-9)
                except Exception:
                    ciclos_eq_anuales = np.nan

        # Promedio diario (suponiendo 1 año de simulación)
        if np.isfinite(ciclos_eq_anuales):
            dias_sim = 365.0      # si quieres algo más fino, puedes calcularlo a partir del DF
            ciclos_eq_diarios = ciclos_eq_anuales / dias_sim

            st.markdown(
                f"**Ciclos equivalentes de la batería:** "
                f"{ciclos_eq_anuales:.0f} ciclos/año "
                f"({ciclos_eq_diarios:.2f} ciclos/día de media)."
            )

        # === Donuts lado a lado: Origen de la energía consumida (Antes vs Después) ===
        if st.session_state.get("fv") == "Sí":
            import plotly.express as px
        
            st.markdown("##### Origen de la energía consumida")

            # --- ANTES (situación inicial) ---
            # Consumo anual desde Paso 1
            df_cons_ini = st.session_state["consumo"]  # cols: ['datetime','consumo']
            cons_ini_kWh = float(pd.to_numeric(df_cons_ini["consumo"], errors="coerce").fillna(0.0).sum())

            # Generación y excedentes (Paso 2)
            gen_ini_kWh = float(pd.to_numeric(st.session_state.get("generacion"), errors="coerce").fillna(0.0).sum()) if "generacion" in st.session_state else 0.0
            exc_ini_kWh = float(pd.to_numeric(st.session_state.get("excedentes"), errors="coerce").fillna(0.0).sum()) if "excedentes" in st.session_state else 0.0

            auto_ini_kWh = max(0.0, gen_ini_kWh - exc_ini_kWh)
            # El reparto es sobre el consumo (no sobre la generación): limitamos el autoconsumo a lo realmente consumido
            auto_ini_kWh = min(auto_ini_kWh, cons_ini_kWh)
            red_ini_kWh  = max(0.0, cons_ini_kWh)

            # --- DESPUÉS (propuesta) ---
            # De tu simulación ganadora en Paso 5
            df_sim_best = st.session_state.get("simul_bess_df", best_global.get("SIM"))
            if df_sim_best is None or df_sim_best.empty:
                st.info("No hay simulación disponible para construir el gráfico 'Después'.")
            else:
                # Consumo de red después
                red_pro_kWh = float(pd.to_numeric(df_sim_best["cons_red_pro_kWh"], errors="coerce").fillna(0.0).sum())

                # Generación (usa la que ya venía del Paso 2; es la referencia pedida)
                gen_total_kWh = gen_ini_kWh  # (tal y como indicas)
                # Vertido nuevo tras batería (de la simulación)
                vertido_new_kWh = float(pd.to_numeric(df_sim_best["vertido_kWh"], errors="coerce").fillna(0.0).sum())

                auto_desp_kWh = max(0.0, gen_total_kWh - vertido_new_kWh)
                # El “consumo total después” que mostramos es el que se cubre por red_pro + autoconsumo.
                # Si por red+FV hubiera ligeras inconsistencias numéricas, lo acotamos a positivo.
                cons_desp_kWh = max(0.0, red_pro_kWh + auto_desp_kWh)

                # Normaliza por si el autoconsumo supera al consumo total (caso extremo de datos)
                auto_desp_kWh = min(auto_desp_kWh, cons_desp_kWh)
                red_pro_kWh   = max(0.0, red_pro_kWh)

                # --- UI: dos columnas con donuts ---
                cA, cB = st.columns(2)

                # Donut ANTES
                with cA:
                    df_antes = pd.DataFrame({
                        "Origen": ["RED", "FV"],
                        "kWh":    [red_ini_kWh, auto_ini_kWh]
                    })
                    if df_antes["kWh"].sum() > 1e-9:
                        figA = px.pie(
                            df_antes, names="Origen", values="kWh", hole=0.6,
                            title="Sin BESS · distribución del consumo anual"
                        )
                        figA.update_traces(
                            textposition="inside",
                            texttemplate="%{percent:.1%}",
                            hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent:.1%})<extra></extra>",marker=dict(colors=["#5CA9E6", "#F5A37A"])
                        )
                        figA.update_layout(margin=dict(l=10, r=10, t=50, b=10),
                                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
                        st.plotly_chart(figA, use_container_width=True)
                    else:
                        st.info("Sin consumo anual válido para el gráfico 'Antes'.")

                # Donut DESPUÉS
                with cB:
                    df_desp = pd.DataFrame({
                        "Origen": ["RED", "FV"],
                        "kWh":    [red_pro_kWh, auto_desp_kWh]
                    })
                    if df_desp["kWh"].sum() > 1e-9:
                        figB = px.pie(
                            df_desp, names="Origen", values="kWh", hole=0.6,
                            title="Con BESS · distribución del consumo anual"
                        )
                        figB.update_traces(
                            textposition="inside",
                            texttemplate="%{percent:.1%}",
                            hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent:.1%})<extra></extra>",marker=dict(colors=["#5CA9E6", "#F5A37A"])
                        )
                        figB.update_layout(margin=dict(l=10, r=10, t=50, b=10),
                                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
                        st.plotly_chart(figB, use_container_width=True)
                    else:
                        st.info("Sin consumo anual válido para el gráfico 'Después'.")

            # --- Origen de la energía almacenada en la batería ---
            st.markdown("##### Origen de la energía almacenada en la batería")

            # Usamos la simulación ganadora (ya cargada unos párrafos antes)
            df_sim_best = st.session_state.get("simul_bess_df", best_global.get("SIM"))

            if df_sim_best is None or df_sim_best.empty:
                st.info("No hay simulación disponible para analizar la energía almacenada en la batería.")
            else:
                # Energía anual cargada en batería desde RED y desde FV [kWh]
                carga_red_total = float(
                    pd.to_numeric(df_sim_best["carga_red_kWh"], errors="coerce").fillna(0.0).sum()
                )
                carga_fv_total = float(
                    pd.to_numeric(df_sim_best["carga_exc_kWh"], errors="coerce").fillna(0.0).sum()
                )

                energia_total = carga_red_total + carga_fv_total

                if energia_total > 1e-9:
                    df_origen_bess = pd.DataFrame({
                        "Origen": ["RED", "FV"],
                        "kWh":    [carga_red_total, carga_fv_total],
                    })

                    fig_bess = px.pie(
                        df_origen_bess,
                        names="Origen",
                        values="kWh",
                        hole=0.6,
                    )
                    fig_bess.update_traces(
                        textposition="inside",
                        texttemplate="%{percent:.1%}",
                        hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent:.1%})<extra></extra>",marker=dict(colors=["#5CA9E6", "#F5A37A"])
                        )
                    fig_bess.update_layout(
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.15,
                            xanchor="center",
                            x=0.5,
                        ),
                    )

                    st.plotly_chart(fig_bess, use_container_width=True)
                else:
                    st.info("La batería no ha llegado a cargar energía en la simulación.")

        # === Formateo ===
        def _fmt_eur(x):
            try:
                return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return x

        # ================== Perfil medio diario (96 QH) de la opción ganadora ==================
        import plotly.express as px

        sim = best_global["SIM"].copy()

        # Asegurar columnas clave (si no existen, se crean)
        need = {
            "cons_red_pro_kWh": 0.0,
            "carga_red_kWh": 0.0,
            "carga_exc_kWh": 0.0,
            "descarga_kWh": 0.0,
            "vertido_kWh": 0.0,
            "fv_gen_kWh": 0.0,          # producción FV total
            "autoconsumo_kWh": 0.0      # autoconsumo directo
        }
        for c, v in need.items():
            if c not in sim.columns:
                sim[c] = v

        # Clave cuarto-hora
        sim["QH"] = pd.to_datetime(sim["datetime"]).dt.strftime("%H:%M")

        # Perfil medio anual (líneas)
        df_plot = (
            sim.groupby("QH", sort=True)
            .agg({
                "cons_red_pro_kWh": "mean",
                "carga_red_kWh": "mean",
                "carga_exc_kWh": "mean",
                "descarga_kWh": "mean",
                "vertido_kWh": "mean",
                "autoconsumo_kWh": "mean",
                "fv_gen_kWh": "mean",
            })
            .reset_index()
            .sort_values("QH")
        )

        df_plot["Carga batería (kWh)"]   = df_plot["carga_red_kWh"] + df_plot["carga_exc_kWh"]
        df_plot["Descarga batería (kWh)"] = df_plot["descarga_kWh"]
        df_plot["Consumo de red (kWh)"]   = df_plot["cons_red_pro_kWh"]
        df_plot["Excedentes FV (kWh)"]    = df_plot["vertido_kWh"]
        df_plot["Autoconsumo (kWh)"]      = df_plot["autoconsumo_kWh"]
        df_plot["Producción FV (kWh)"]    = df_plot["fv_gen_kWh"]

        cols_show = [
            "QH",
            "Consumo de red (kWh)",
            "Autoconsumo (kWh)",
            "Producción FV (kWh)",
            "Excedentes FV (kWh)",
            "Carga batería (kWh)",
            "Descarga batería (kWh)",
        ]
        df_show = df_plot[cols_show].copy()

        # ================== Día medio VERANO / INVIERNO (entre semana vs fin de semana) ==================
        sim["datetime"] = pd.to_datetime(sim["datetime"])
        sim["mes"] = sim["datetime"].dt.month
        # 0 = lunes ... 6 = domingo
        sim["dow"] = sim["datetime"].dt.dayofweek

        meses_verano   = {6, 7, 8, 9}   # Jun–Sep
        meses_invierno = {12, 1, 2, 3}     # Dic–Marzo

        def _perfil_medio(df):
            """
            Calcula el perfil medio diario (96 QH) de un subconjunto de simulación.

            Vectores que se devuelven (promedio horario):
            - Demanda del edificio (load_kWh)
            - Carga batería desde red
            - Carga batería desde FV
            - Carga total de batería (red + FV)
            - Descarga de batería
            """
            df = df.copy()
            df["QH"] = df["datetime"].dt.strftime("%H:%M")

            g = (
                df.groupby("QH", sort=True)
                .agg({
                    "load_kWh":      "mean",  # demanda del edificio
                    "carga_red_kWh": "mean",  # carga batería desde red
                    "carga_exc_kWh": "mean",  # carga batería desde FV (excedentes)
                    "descarga_kWh":  "mean",  # descarga batería
                })
                .reset_index()
                .sort_values("QH")
            )

            # Construimos las series que queremos pintar
            g["Demanda (kWh)"]             = g["load_kWh"]
            g["Carga desde red (kWh)"]     = g["carga_red_kWh"]
            g["Carga desde FV (kWh)"]      = g["carga_exc_kWh"]
            g["Carga batería (kWh)"]       = g["carga_red_kWh"] + g["carga_exc_kWh"]
            g["Descarga batería (kWh)"]    = g["descarga_kWh"]

            return g[[
                "QH",
                "Demanda (kWh)",
                "Carga desde red (kWh)",
                "Carga desde FV (kWh)",
                "Carga batería (kWh)",
                "Descarga batería (kWh)",
            ]]

        def _graf_area(df_show, titulo):
            import plotly.graph_objects as go

            x = df_show["QH"]
            series = [
                "Demanda (kWh)",
                "Carga desde red (kWh)",
                "Carga desde FV (kWh)",
                "Carga batería (kWh)",
                "Descarga batería (kWh)",
            ]

            fig = go.Figure()
            for s in series:
                y = df_show[s].astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=s,
                        mode="lines",
                        fill="tozeroy",   # mismo estilo área rellena
                        line=dict(width=2),
                    )
                )

            # Ticks cada 2 horas (2h = 8 tramos de 15 min)
            tick_vals = list(df_show["QH"].iloc[::8])
            fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_vals)

            fig.update_layout(
                title=titulo,
                xaxis_title="Hora del día",
                yaxis_title="kWh por QH (media)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.3,
                    xanchor="center", x=0.5
                ),
                margin=dict(l=10, r=10, t=48, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # -------- Verano: laborable vs fin de semana --------
        df_ver = sim[sim["mes"].isin(meses_verano)]

        if len(df_ver):

            # Entre semana (lunes–viernes, dayofweek = 0..4)
            df_ver_lab = df_ver[df_ver["dow"] < 5]
            if len(df_ver_lab):
                df_show_ver_lab = _perfil_medio(df_ver_lab)
                _graf_area(df_show_ver_lab, "Día medio VERANO – Entre semana")
            else:
                st.info("No hay datos de verano entre semana en la simulación.")

            # Fin de semana (sábado–domingo, dayofweek = 5..6)
            df_ver_we = df_ver[df_ver["dow"] >= 5]
            if len(df_ver_we):
                df_show_ver_we = _perfil_medio(df_ver_we)
                _graf_area(df_show_ver_we, "Día medio VERANO – Fin de semana")
            else:
                st.info("No hay datos de verano en fin de semana en la simulación.")

        # -------- Invierno: laborable vs fin de semana --------
        df_inv = sim[sim["mes"].isin(meses_invierno)]

        if len(df_inv):

            df_inv_lab = df_inv[df_inv["dow"] < 5]
            if len(df_inv_lab):
                df_show_inv_lab = _perfil_medio(df_inv_lab)
                _graf_area(df_show_inv_lab, "Día medio INVIERNO – Entre semana")
            else:
                st.info("No hay datos de invierno entre semana en la simulación.")

            df_inv_we = df_inv[df_inv["dow"] >= 5]
            if len(df_inv_we):
                df_show_inv_we = _perfil_medio(df_inv_we)
                _graf_area(df_show_inv_we, "Día medio INVIERNO – Fin de semana")
            else:
                st.info("No hay datos de invierno en fin de semana en la simulación.")

        # Vista rápida de los primeros 2 días (como auditoría)
        fechas = pd.to_datetime(df_sim_best["datetime"]).dt.date
        dias_unicos = sorted(pd.unique(fechas))
        dias_mostrar = dias_unicos[:2] if len(dias_unicos) >= 2 else dias_unicos
        mask_d2 = fechas.isin(dias_mostrar)
        cols_view = [
        "datetime",
        # Consumo y red
        # (si quieres, puedes añadir aquí también el consumo original cons[t])
        "cons_red_pro_kWh",      # consumo desde red con batería
        "pot_contratada_kW",
        "maximetro_kW",
        # FV
        "fv_gen_kWh",
        "autoconsumo_kWh",
        "vertido_kWh",           # excedentes CON batería
        # Batería
        "carga_red_kWh",
        "carga_exc_kWh",
        "carga_kWh",
        "descarga_kWh",
        "energia_almacenada_kWh",
        "soc_pu",
        # Precio
        "precio_eur_mwh",]
        st.subheader(f"Primeros {len(dias_mostrar)} días · simulación óptima")
        st.dataframe(df_sim_best.loc[mask_d2, cols_view], use_container_width=True)
            # ---- Descargar curvas horarias en Excel (load, red, FV, vertidos) ----
        from io import BytesIO

        df_export = st.session_state.get("simul_bess_df")

        if df_export is not None and len(df_export):
            # Columnas que queremos exportar
            cols_export = [
                "datetime",
                "load_kWh",          # consumo del edificio
                "cons_red_pro_kWh",  # consumo desde red con batería
                "fv_gen_kWh",        # generación FV
                "vertido_kWh",       # excedentes vertidos a red con BESS
                "descarga_kWh"
            ]
            # Por si faltara alguna, filtramos a las que existan
            cols_export = [c for c in cols_export if c in df_export.columns]

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_export[cols_export].to_excel(
                    writer,
                    index=False,
                    sheet_name="Curvas_BESS",
                )
            buffer.seek(0)

            st.download_button(
                label="📥 Descargar curvas de la simulación en Excel",
                data=buffer,
                file_name="curvas_bess_optima.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        # ======================================================================
        # ========= Exportar Paso 5 a PDF (con icono base64) =========
        import base64

        # 1) Icono en base64 (si no existe, seguimos sin icono)
        try:
            with open("logo_pdf.png", "rb") as f:
                pdf_icon_b64 = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            pdf_icon_b64 = ""

        # 2) CSS global de impresión (afecta a toda la página)
        st.markdown("""
        <style>
        .export-wrap { margin-top: 14px; }

        @media print {
            header, footer, [data-testid="stSidebar"], .stToolbar { display:none !important; }
            .block-container { padding-top:0 !important; padding-bottom:0 !important; }
            @page { margin: 12mm; }
            .stPlotlyChart, .stTable { break-inside: avoid; page-break-inside: avoid; }
            .print-break { break-before: page; page-break-before: always; }
            .no-print, .export-wrap { display:none !important; }
        }
        </style>
        """, unsafe_allow_html=True)

        # 3) Botón + JS dentro de un componente (aquí sí se ejecuta)
        html_button = f"""
        <html>
        <head>
        <style>
            body {{
            margin: 0;
            background: transparent;
            }}
            .pdf-btn {{
            display: flex;
            align-items: center;
            gap: 8px;
            height: 38px;
            padding: 6px 14px;
            background: #F2F4F7;
            color: #1a2b4b;
            border: 1px solid #DDE3EA;
            border-radius: 8px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background .15s;
            }}
            .pdf-btn:hover {{
            background: #EBEEF3;
            }}
            .pdf-icon {{
            width: 22px;
            height: 22px;
            }}
        </style>
        </head>
        <body>
        <div class="export-wrap no-print">
            <button class="pdf-btn" onclick="parent.window.print()">
            {f'<img src="data:image/png;base64,{pdf_icon_b64}" class="pdf-icon" alt="PDF" />' if pdf_icon_b64 else ''}
            Descargar propuesta en PDF
            </button>
        </div>
        </body>
        </html>
        """

        components.html(html_button, height=60, scrolling=False)
        # ========= FIN Exportar PDF =========

def render_evaluador():
    from pathlib import Path
    import pandas as pd

    # --- Header bonito (hero) centrado en caja azul ---
    TITLE = "Evaluador de soluciones de sistemas de almacenamiento"

    st.markdown(f"""<div style="background:#0f1c3f;border:1px solid #0a1933;border-radius:16px;padding:26px 28px;
            text-align:center;margin: 8px 0 18px 0;box-shadow: 0 6px 18px rgba(15,28,63,0.15);"><h1 style="
            color:#ffffff;margin:0;font-weight:800;letter-spacing:0.3px;font-size:calc(22px + 0.9vw);line-height:1.15;
            ">{TITLE}</h1></div>""",unsafe_allow_html=True,)

    # --- Stepper (línea de progreso de pasos) ---
    def render_stepper(total_steps=6, current_step=None):
        cur = int(st.session_state.get("step", 1) if current_step is None else current_step)
        cur = max(1, min(total_steps, cur))  # clamp

        # CSS del stepper
        st.markdown("""<style>.stepper-wrap{margin:12px 0 22px 0; display:flex; justify-content:center;}.stepper{display:flex; align-items:center; gap:14px;}
        .stepper .dot{width:26px;height:26px;border-radius:50%;background:#e6ecff;border:2px solid #b8c7ff;color:#4a5d9d;
            display:flex;align-items:center;justify-content:center;font-weight:700;}.stepper .dot.active{ background:#2b64ff; border-color:#1d47c7; color:#fff; }
        .stepper .bar{width:64px;height:6px;border-radius:6px;background:#e6ecff;border:1px solid #b8c7ff;}
        .stepper .bar.active{ background:#2b64ff; border-color:#1d47c7; }@media (max-width: 540px){
            .stepper .bar{width:44px;}}</style>""", unsafe_allow_html=True)

        # HTML dinámico según el paso actual
        parts = []
        for i in range(1, total_steps + 1):
            parts.append(f'<div class="dot {"active" if i <= cur else ""}">{i}</div>')
            if i < total_steps:
                parts.append(f'<div class="bar {"active" if i < cur else ""}"></div>')
        html = '<div class="stepper-wrap"><div class="stepper">' + "".join(parts) + "</div></div>"
        st.markdown(html, unsafe_allow_html=True)

    render_stepper(total_steps=4)


    if "step" not in st.session_state:
        st.session_state.step = 1

    # -------- Paso 1: Consumo --------
    if st.session_state.step == 1:
        st.header("Paso 1 · Datos iniciales - consumo de red")
        tarifa = st.radio("Selecciona tu tarifa:", ["2.0TD", "3.0TD", "6.1TD", "6.2TD", "6.3TD", "6.4TD"], horizontal=True)
        st.session_state["tarifa"] = tarifa

        instalacion = st.radio("Selecciona tu tipo de intalación eléctrica:", ["Trifásica", "Monofásica"], horizontal=True)
        st.session_state["instalacion"] = instalacion
        
        st.subheader("Potencias contratadas (kW)")
        
        # Carga valores previos si ya existen (para que no se pierdan al recargar)
        p_prev = st.session_state.get("potencias", {"P1":0.0,"P2":0.0,"P3":0.0,"P4":0.0,"P5":0.0,"P6":0.0})

        if tarifa == "2.0TD":
            c1, c2, = st.columns(2)
            P1 = c1.number_input("P1", min_value=0.0, step=0.1, value=float(p_prev.get("P1", 0.0)), key="pot_P1")
            P2 = c2.number_input("P2", min_value=0.0, step=0.1, value=float(p_prev.get("P2", 0.0)), key="pot_P2")


            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["potencias"] = {"P1": P1, "P2": P2}

        else:
            c1, c2, c3 = st.columns(3)
            P1 = c1.number_input("P1", min_value=0.0, step=0.1, value=float(p_prev.get("P1", 0.0)), key="pot_P1")
            P2 = c2.number_input("P2", min_value=0.0, step=0.1, value=float(p_prev.get("P2", 0.0)), key="pot_P2")
            P3 = c3.number_input("P3", min_value=0.0, step=0.1, value=float(p_prev.get("P3", 0.0)), key="pot_P3")

            c4, c5, c6 = st.columns(3)
            P4 = c4.number_input("P4", min_value=0.0, step=0.1, value=float(p_prev.get("P4", 0.0)), key="pot_P4")
            P5 = c5.number_input("P5", min_value=0.0, step=0.1, value=float(p_prev.get("P5", 0.0)), key="pot_P5")
            P6 = c6.number_input("P6", min_value=0.0, step=0.1, value=float(p_prev.get("P6", 0.0)), key="pot_P6")

            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["potencias"] = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6}
            
        st.subheader("Carga el consumo cuarto-horario anual de red")
        st.info("Los archivos deben estar en formato .xlsx o .xls y **contener dos columnas**:\n "
            "1) Fecha [formato: yyyy-mm-dd hh:mm:ss]\n"
            "2) Consumo de red [kWh]\n")
        archivo = st.file_uploader("Elige un archivo .xlsx o .xls")

        # Variables df1 = fechas, df2 = consumos
        if archivo is not None:
            try:
                df_raw = pd.read_excel(archivo)
                # Tomamos las dos primeras columnas: fecha, consumo
                fecha_col = df_raw.columns[0]
                consumo_col = df_raw.columns[1]
            
                fechas = pd.to_datetime(df_raw[fecha_col], errors='coerce')
                consumos = pd.to_numeric(df_raw[consumo_col], errors='coerce')
                base = pd.DataFrame({"datetime": fechas, "consumo": consumos}).dropna()
        
                st.session_state["consumo"] = base
                st.session_state["df1"] = base["datetime"]
                st.session_state["df2"] = base["consumo"]
            
                st.caption("Datos leidos correctamente.")

            except Exception as e:
                st.error(f"No pude leer o interpretar el Excel. Detalle: {e}")

            fechas = st.session_state["df1"]
            consumos = st.session_state["df2"]

        from bbdd_mercado import attach_market_to_consumo

        if "consumo" in st.session_state and "market" in st.session_state:
            df_costes = attach_market_to_consumo(st.session_state["consumo"], st.session_state["market"])
            st.session_state["consumo_con_mercado"] = df_costes  # ya alineado por datetime


        c1, c2 = st.columns(2)
        c1.button("Continuar »", use_container_width=True,
                disabled=("consumo" not in st.session_state),
                on_click=lambda: st.session_state.update(step=2))

    # -------- Paso 2: FV sí/no + ficheros --------
    elif st.session_state.step == 2:
        st.header("Paso 2 · ¿La instalación dispone de FV?")
        fv = st.radio("Selecciona una opción:", ["No", "Sí"], horizontal=True)
        st.session_state["fv"] = fv

        if fv == "Sí":
            st.subheader("Sube las curvas cuarto-horarias anuales de la FV")
            st.info("Los archivo debe ser en formato .xlsx o .xls , **contener dos columnas**:\n "
            "1) Fecha [formato: yyyy-mm-dd hh:mm:ss]\n"
            "2) Excedentes/Generación [kWh]\n")
            up_exc = st.file_uploader("Excedentes FV (Excel/CSV)", type=["csv","xlsx","xls"], key="exc")
            up_gen = st.file_uploader("Generación FV (Excel/CSV)", type=["csv","xlsx","xls"], key="auto")

            if up_exc is not None:
                try:
                    df_exc = pd.read_excel(up_exc)
                    # Tomamos la segunda columna
                    excedente_col = df_exc.columns[1]
                    fecha_exc_col = df_exc.columns[0]

                    excedentes = pd.to_numeric(df_exc[excedente_col], errors='coerce')
                    fecha_exc = pd.to_datetime(df_exc[fecha_exc_col], errors='coerce')      
        
                    st.session_state["excedentes"] = excedentes
                    st.session_state["fecha_exc"] = fecha_exc
                    st.caption("Datos leidos correctamente.")
                except Exception as e:
                    st.error(f"No pude leer o interpretar el Excel. Detalle: {e}")

            if up_gen is not None:
                try:
                    df_gen = pd.read_excel(up_gen)
                    # Tomamos la segunda columna
                    generacion_col = df_gen.columns[1]
                    fecha_gen_col = df_gen.columns[0]
                    generacion = pd.to_numeric(df_gen[generacion_col], errors='coerce')
                    fecha_gen = pd.to_datetime(df_gen[fecha_gen_col], errors='coerce')      
        
                    st.session_state["generacion"] = generacion
                    st.session_state["fecha_gen"] = fecha_gen
                    st.caption("Datos leidos correctamente.")
                except Exception as e:
                    st.error(f"No pude leer o interpretar el Excel. Detalle: {e}")

        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=1))

        listo = (fv == "No") or ("excedentes" in st.session_state and "generacion" in st.session_state)
        c2.button("Continuar»", use_container_width=True, disabled=not listo,
                on_click=lambda: st.session_state.update(step=3))
        
    #---- Paso 3 : Datos de contrato electrico-----
    elif st.session_state.step == 3:
        st.header("Paso 3 · Datos del contrato eléctrico")
        
        tarifa = st.session_state.get("tarifa", "(sin seleccionar)")

        #----seleccion modalidad contrato----
        if tarifa == "2.0TD":
            modalidad = st.radio("Selecciona la modalidad de tu contrato eléctrico:", ["Precio fijo", "PVPC", "Indexado pass through"], horizontal=True)
            st.session_state["modalidad"] = modalidad
            # --- PVPC: carga automática desde ruta local ---
            PVPC_PATH = "data/PVPC_QH.xlsx"

            @st.cache_data(show_spinner=False)
            def load_pvpc_excel(path: str) -> pd.DataFrame:
                df_raw = pd.read_excel(path)
                fecha_col  = df_raw.columns[0]   # Columna fecha/hora
                precio_col = df_raw.columns[1]   # Columna precio [€/MWh] por QH
                df = pd.DataFrame({
                "datetime": pd.to_datetime(df_raw[fecha_col], errors="coerce"),
                "precio_eur_mwh": pd.to_numeric(df_raw[precio_col], errors="coerce"),
                }).dropna()
                # Redondear/ajustar a cuartos exactos y quitar tz
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                # Si vienen segundos/milisegundos, llevamos al inicio del cuarto
                df["datetime"] = df["datetime"].dt.floor("15min")
                # Si viniera con zona horaria, la quitamos (naive) para matchear con consumo
                if hasattr(df["datetime"].dt, "tz"):
                    df["datetime"] = df["datetime"].dt.tz_convert(None) if df["datetime"].dt.tz is not None else df["datetime"]

                return df

            if modalidad == "PVPC":
                try:
                    st.session_state["pvpc_df"] = load_pvpc_excel(PVPC_PATH)
                except Exception as e:
                    st.error(f"No se pudo leer el PVPC en {PVPC_PATH}. Detalle: {e}")

        else:
            modalidad = st.radio("Selecciona la modalidad de tu contrato eléctrico:", ["Precio fijo", "Indexado pass pool", "Indexado pass through"], horizontal=True)
            st.session_state["modalidad"] = modalidad
        
        st.info("Todos los precios deben estar en unidades de **€/MWh**")

        if modalidad == "Precio fijo" and st.session_state["tarifa"] == "2.0TD":
            c1, c2, c3 = st.columns(3)
            PrecioP1 = c1.number_input("Precio P1", min_value=0.0, step=0.01, key="Precio P1")
            PrecioP2 = c2.number_input("Precio P2", min_value=0.0, step=0.01, key="Precio P2")
            PrecioP3 = c3.number_input("Precio P3", min_value=0.0, step=0.01, key="Precio P3")

            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["precios_te"] = {"Precio P1": PrecioP1, "Precio P2": PrecioP2,"Precio P3": PrecioP3}
        
        elif modalidad == "Precio fijo" and st.session_state["tarifa"] != "2.0TD":
            c1, c2, c3 = st.columns(3)
            PrecioP1 = c1.number_input("Precio P1", min_value=0.0, step=0.01, key="Precio P1")
            PrecioP2 = c2.number_input("Precio P2", min_value=0.0, step=0.01, key="Precio P2")
            PrecioP3 = c3.number_input("Precio P3", min_value=0.0, step=0.01, key="Precio P3")

            c4, c5, c6 = st.columns(3)
            PrecioP4 = c4.number_input("Precio P4", min_value=0.0, step=0.01, key="Precio P4")
            PrecioP5 = c5.number_input("Precio P5", min_value=0.0, step=0.01, key="Precio P5")
            PrecioP6 = c6.number_input("Precio P6", min_value=0.0, step=0.01, key="Precio P6")

            st.session_state["precios_te"] = {"Precio P1": PrecioP1, "Precio P2": PrecioP2,"Precio P3": PrecioP3,"Precio P4": PrecioP4,"Precio P5": PrecioP5,"Precio P6": PrecioP6}

        elif modalidad == "Indexado pass pool":
            # 1) Tipo de OMIE para el contrato
            st.info("Selecciona cómo se indexa tu contrato al OMIE.")
            tipo_omie = st.radio(
                "Base de OMIE para el término de energía:",
                ["Horario", "Mensual"],
                index=0,
                horizontal=True,
                key="pp_omie_tipo_radio",
            )

            # Guardamos un valor limpio para usar luego en el Paso 4
            st.session_state["pp_omie_tipo"] = (
                "horario" if tipo_omie == "Horario" else "mensual"
            )

            # 2) Coeficientes fijos Ai
            st.info("Introduce el coeficiente fijo (Ai)")
            c1, c2, c3 = st.columns(3)
            PrecioAiP1 = c1.number_input("Precio Ai P1", min_value=0.0, step=0.01, key="Precio Ai P1")
            PrecioAiP2 = c2.number_input("Precio Ai P2", min_value=0.0, step=0.01, key="Precio Ai P2")
            PrecioAiP3 = c3.number_input("Precio Ai P3", min_value=0.0, step=0.01, key="Precio Ai P3")

            c4, c5, c6 = st.columns(3)
            PrecioAiP4 = c4.number_input("Precio Ai P4", min_value=0.0, step=0.01, key="Precio Ai P4")
            PrecioAiP5 = c5.number_input("Precio Ai P5", min_value=0.0, step=0.01, key="Precio Ai P5")
            PrecioAiP6 = c6.number_input("Precio Ai P6", min_value=0.0, step=0.01, key="Precio Ai P6")

            st.session_state["precios_Ai"] = {
                "Precio Ai P1": PrecioAiP1,
                "Precio Ai P2": PrecioAiP2,
                "Precio Ai P3": PrecioAiP3,
                "Precio Ai P4": PrecioAiP4,
                "Precio Ai P5": PrecioAiP5,
                "Precio Ai P6": PrecioAiP6,
            }

            # 3) Coeficientes variables Ci
            st.info("Introduce el coeficiente del término variable (Ci)")
            c1, c2, c3 = st.columns(3)
            PrecioCiP1 = c1.number_input("Precio Ci P1", min_value=0.0, step=0.01, key="Precio Ci P1")
            PrecioCiP2 = c2.number_input("Precio Ci P2", min_value=0.0, step=0.01, key="Precio Ci P2")
            PrecioCiP3 = c3.number_input("Precio Ci P3", min_value=0.0, step=0.01, key="Precio Ci P3")

            c4, c5, c6 = st.columns(3)
            PrecioCiP4 = c4.number_input("Precio Ci P4", min_value=0.0, step=0.01, key="Precio Ci P4")
            PrecioCiP5 = c5.number_input("Precio Ci P5", min_value=0.0, step=0.01, key="Precio Ci P5")
            PrecioCiP6 = c6.number_input("Precio Ci P6", min_value=0.0, step=0.01, key="Precio Ci P6")

            st.session_state["precios_Ci"] = {
                "Precio Ci P1": PrecioCiP1,
                "Precio Ci P2": PrecioCiP2,
                "Precio Ci P3": PrecioCiP3,
                "Precio Ci P4": PrecioCiP4,
                "Precio Ci P5": PrecioCiP5,
                "Precio Ci P6": PrecioCiP6,
            }

        elif modalidad == "Indexado pass through":
            st.info("Ingresa los desvíos de tu comercializadora juntamente con los costes de gestión. Ten en cuenta todos los extras que pueden incluirse, como primas de riesgo o constantes. En caso de que tu comercializadora no fije el valor de los desvíos debes poner 0,328 €/MWh")

            c1, c2 = st.columns(2)
            Desvíos = c1.number_input("Desvíos [€/MWh]", min_value=0.0, step=0.1, key="desvios")
            CG = c2.number_input("CG [€/MWh]", min_value=0.0, step=0.1, key="CG")

            # Guarda todo junto en sesión (para usar en pasos siguientes)
            st.session_state["comer"] = {"desvios": Desvíos, "CG": CG}

        #----- Si hay FV modalidad compra excedentes----
        if st.session_state["fv"] == "Sí":
            modalidad_fv = st.radio("Selecciona la modalidad de tu contrato fotovoltaico:", ["Precio fijo", "Indexado"], horizontal=True)
            st.session_state["modalidad_fv"] = modalidad_fv
            st.info("En caso de **no disponer de compensación de excedentes** se debe **seleccionar modalidad precio fijo 0 €/MWh**")
            
            if modalidad_fv == "Precio fijo":
                PrecioFV = st.number_input("Precio FV", min_value=0.0, step=0.01, key="Precio FV")
                st.session_state["precio_fv"] = {"Precio FV": PrecioFV}

            if modalidad_fv == "Indexado":
                CG_fv = st.number_input("Costes Gestion FV", min_value=0.0, step=0.01, key="Costes Gestion FV")
                st.session_state["cg_fv"] = {"Costes Gestion FV": CG_fv}

        # precio Potencia
        modalidad_pot = st.radio("Selecciona tu termino de potencia:", ["BOE", "No BOE"], horizontal=True)
        st.session_state["modalidad_pot"] = modalidad_pot

        if modalidad_pot == "No BOE" and tarifa == "2.0TD":
            c1, c2 = st.columns(2)
            precio_pot_p1 = c1.number_input("Precio potencia P1 [€/kW año]", min_value=0.0, step=0.1, key="precio_pot_p1")
            precio_pot_p2 = c2.number_input("Precio potencia P2 [€/kW año]", min_value=0.0, step=0.1, key="precio_pot_p2")
            st.session_state["precio_pot"] = {"precio_pot_p1": precio_pot_p1, "recio_pot_p2": precio_pot_p2}

        elif modalidad_pot == "No BOE" and tarifa != "2.0TD":
            c1, c2, c3 = st.columns(3)
            PreciopotP1 = c1.number_input("Precio potencia P1", min_value=0.0, step=0.01, key="Precio potencia P1")
            PreciopotP2 = c2.number_input("Precio potencia P2", min_value=0.0, step=0.01, key="Precio potencia P2")
            PreciopotP3 = c3.number_input("Precio potencia P3", min_value=0.0, step=0.01, key="Precio potencia P3")

            c4, c5, c6 = st.columns(3)
            PreciopotP4 = c4.number_input("Precio potencia P4", min_value=0.0, step=0.01, key="Precio potencia P4")
            PreciopotP5 = c5.number_input("Precio potencia P5", min_value=0.0, step=0.01, key="Precio potencia P5")
            PreciopotP6 = c6.number_input("Precio potencia P6", min_value=0.0, step=0.01, key="Precio potencia P6")

            st.session_state["precio_pot"] = {"Precio potencia P1": PreciopotP1, "Precio potencia P2": PreciopotP2,"Precio potencia P3": PreciopotP3,"Precio potencia P4": PreciopotP4,"Precio potencia P5": PreciopotP5,"Precio potencia P6": PreciopotP6}
    
        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=2))
        c2.button("Generar resumen situación inicial»", use_container_width=True, on_click=lambda: st.session_state.update(step=4))

        
    # ------Paso 4 : Resumen ------
    elif st.session_state.step == 4:
        st.header("Paso 4 · Resumen situación inicial")

        # -------- helpers de estilo --------
        def _styler_base():
            return [
            {"selector":"th", "props":"background:#eef4ff; color:#1a2b4b; font-weight:700;"},
            {"selector":"tbody tr:nth-child(even)", "props":"background:#fafbff;"},
            {"selector":"td, th", "props":"border:1px solid #e9edf5; padding:6px 10px;"},]

        def style_tabla(df, bold_first_col=True, fmt_map=None, highlight_total_label="Total"):
            stl = df.style.set_table_styles(_styler_base())
            if bold_first_col and len(df.columns)>0:
                stl = stl.set_properties(subset=pd.IndexSlice[:, [df.columns[0]]],
                                    **{"font-weight":"700"})
            if fmt_map:
                stl = stl.format(fmt_map)

        # 💡 Resaltar la fila "Total": fondo y negrita
            def _row_style(s):
                if str(s.iloc[0]).strip().lower() == str(highlight_total_label).lower():
                    return ["font-weight:700; background-color:#fff7cc;"] * len(s)
                return [""] * len(s)
            stl = stl.apply(_row_style, axis=1)

            return stl.hide(axis="index")

        # Cargar base de mercado desde disco, sin UI
        from bbdd_mercado import ensure_market_loaded
        try:
            ensure_market_loaded()
        except Exception as e:
            st.session_state["market_error"] = f"{type(e).__name__}: {e}"

        # --- Tarifa ---
        tarifa = st.session_state.get("tarifa", "(sin seleccionar)")

        # --- Potencias contratadas (kW) - compacto ---
        st.subheader("Potencias contratadas (kW)")
        pot = st.session_state.get("potencias", {"P1":0.0,"P2":0.0,"P3":0.0,"P4":0.0,"P5":0.0,"P6":0.0})

        if tarifa == "2.0TD":
            df_pot = pd.DataFrame(
            {"Periodo": ["P1","P2"],
            "Potencia [kW]": [pot.get("P1",0.0), pot.get("P2",0.0)]})
        else:
            df_pot = pd.DataFrame(
            {"Periodo": ["P1","P2","P3","P4","P5","P6"],
            "Potencia [kW]": [pot.get("P1",0.0), pot.get("P2",0.0), pot.get("P3",0.0),
                            pot.get("P4",0.0), pot.get("P5",0.0), pot.get("P6",0.0)]})

        def _fmt_kw(v):
            try:
                return f"{float(v):,.1f}".replace(",", ".")
            except Exception:
                return v

        st.table(style_tabla(df_pot, fmt_map={"Potencia [kW]": _fmt_kw}))

        # ---- Energia ---
        # consumo anual
        total_anual = float(st.session_state["consumo"]["consumo"].sum())
        st.subheader("Resumen consumo anual")
        st.metric("Consumo total del año", f"{total_anual:,.0f} kWh".replace(",", "."))

        # consumo mensual
        c = st.session_state["consumo"].copy()  # DataFrame con columnas: datetime, consumo
        c["mes"] = c["datetime"].dt.month
        cons_m = c.groupby("mes")["consumo"].sum()

        meses_es = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

        fv = st.session_state.get("fv", "No")

        if fv != "Sí":
            meses_presentes = sorted(cons_m.index.tolist())
            df_tabla = pd.DataFrame({"Mes": [meses_es[m] for m in meses_presentes],"Consumo [kWh]": cons_m.reindex(meses_presentes).values})
            df_tabla["Consumo [kWh]"] = df_tabla["Consumo [kWh]"].round(0).astype(int)

            tot_cons = df_tabla["Consumo [kWh]"].sum()
            df_total = pd.DataFrame([{"Mes": "Total","Consumo [kWh]":tot_cons}])

        else:
            df_e = pd.DataFrame({"datetime": pd.to_datetime(st.session_state["fecha_exc"], errors="coerce"),"value":    pd.to_numeric(st.session_state["excedentes"], errors="coerce")}).dropna()
            df_g = pd.DataFrame({"datetime": pd.to_datetime(st.session_state["fecha_gen"], errors="coerce"),"value":    pd.to_numeric(st.session_state["generacion"], errors="coerce")}).dropna()

            df_e["mes"] = df_e["datetime"].dt.month
            df_g["mes"] = df_g["datetime"].dt.month

            exc_m = df_e.groupby("mes")["value"].sum()
            gen_m = df_g.groupby("mes")["value"].sum()

            # Índice común (meses presentes en cualquiera)
            meses_presentes = sorted(set(cons_m.index) | set(gen_m.index) | set(exc_m.index))

            df_tabla = pd.DataFrame({"Mes": [meses_es[m] for m in meses_presentes],"Consumo [kWh]": cons_m.reindex(meses_presentes).fillna(0).values,"Generación FV [kWh]": gen_m.reindex(meses_presentes).fillna(0).values,"Excedentes [kWh]": exc_m.reindex(meses_presentes).fillna(0).values,})

            # Redondeo
            for col in ["Consumo [kWh]", "Generación FV [kWh]", "Excedentes [kWh]"]:
                df_tabla[col] = df_tabla[col].round(0).astype(int)

            # % Autoconsumo = (Generación-Excedentes) / (Consumo red + Generación - excedentes)
            autoconsumo_m = df_tabla["Generación FV [kWh]"] - df_tabla["Excedentes [kWh]"]
            denom = df_tabla["Consumo [kWh]"] + autoconsumo_m
            df_tabla["% Autoconsumo"] = (100 * autoconsumo_m / denom.where(denom != 0, 1)).fillna(0)

            # Fila Total (sumas y % con totales)
            tot_cons = df_tabla["Consumo [kWh]"].sum()
            tot_gen  = df_tabla["Generación FV [kWh]"].sum()
            tot_exc  = df_tabla["Excedentes [kWh]"].sum()
            tot_pct  = 100 * ((tot_gen-tot_exc) / (tot_cons + tot_gen - tot_exc)) if (tot_cons + tot_gen - tot_exc) > 0 else 0.0

            df_total = pd.DataFrame([{"Mes": "Total","Consumo [kWh]": tot_cons,"Generación FV [kWh]": tot_gen,"Excedentes [kWh]": tot_exc,"% Autoconsumo": tot_pct,}])

            # Redondeos bonitos
            for col in ["Consumo [kWh]", "Generación FV [kWh]", "Excedentes [kWh]"]:
                df_tabla[col] = df_tabla[col].round(0).astype(int)
                df_total[col] = int(round(df_total[col].iloc[0], 0))
            df_tabla["% Autoconsumo"] = df_tabla["% Autoconsumo"].round(1)
            df_total["% Autoconsumo"] = round(float(df_total["% Autoconsumo"]), 1)

        # Mostrar
        tabla_final = pd.concat([df_tabla, df_total], ignore_index=True)
        # Mostrar (formateo suave)
        tabla_final_fmt = tabla_final.copy()

        # formateo miles sin decimales en columnas kWh
        cols_kwh = [c for c in tabla_final_fmt.columns if "kWh" in c]
        for col in cols_kwh:
            tabla_final_fmt[col] = tabla_final_fmt[col].map(lambda v: f"{int(v):,}".replace(",", "."))

        if "% Autoconsumo" in tabla_final_fmt.columns:
            tabla_final_fmt["% Autoconsumo"] = tabla_final_fmt["% Autoconsumo"].map(lambda x: f"{x:.1f} %")

        fmt_cols = {c: (lambda v: f"{int(v):,}".replace(",", ".")) for c in tabla_final.columns if "kWh" in c}
        if "% Autoconsumo" in tabla_final.columns:
            fmt_cols["% Autoconsumo"] = lambda v: f"{v:.1f} %"

        st.table(style_tabla(tabla_final, fmt_map=fmt_cols, highlight_total_label="Total"))

        import altair as alt

        st.caption("Grafica de consumo mensual (kWh)")

        orden_meses = list(range(1,13))
        meses_es   = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

        # Consumo en orden fijo
        serie_cons = cons_m.reindex(orden_meses).fillna(0)

        # ¿Hay autoconsumo mensual? (Autoconsumo = Generación - Excedentes)
        serie_auto = None
        if st.session_state.get("fv") == "Sí":
            try:
                serie_gen = gen_m.reindex(orden_meses).fillna(0)
                serie_exc = exc_m.reindex(orden_meses).fillna(0)
                serie_auto = (serie_gen - serie_exc).clip(lower=0)  # autoconsumo ≥ 0
            except NameError:
                pass

        df_m = pd.DataFrame({"MesNum": orden_meses,
                        "Mes": [meses_es[m] for m in orden_meses],
                        "Consumo": serie_cons.values})

        if serie_auto is not None:
            df_m["Autoconsumo"] = serie_auto.values

        # Pasar a formato largo para Altair
        df_long = df_m.melt(id_vars=["MesNum","Mes"], var_name="Concepto", value_name="kWh")
        df_long = df_long[df_long["Concepto"].notna()]  # por si no hay generación

        colores = alt.Scale(
        domain=["Consumo","Autoconsumo"],
        range=["#9ec9ff", "#ffc48a"])

        chart = (
            alt.Chart(df_long)
            .mark_bar()
            .encode(
                x=alt.X("Mes:N", sort=[meses_es[m] for m in orden_meses], title="Mes"),
                y=alt.Y("kWh:Q", title="kWh"),
                color=alt.Color("Concepto:N", scale=colores, title=""),
                tooltip=[alt.Tooltip("Concepto:N"), alt.Tooltip("Mes:N"), alt.Tooltip("kWh:Q", format=",.0f")])
            .properties(height=320))

        st.altair_chart(chart, use_container_width=True)

        #Costes
        from index_pt import _leer_desvios_y_cg, coste_passthrough

        if "market" not in st.session_state:
            st.error("Falta la base de mercado. " +
                f"Detalle: {st.session_state.get('market_error','')}")
            st.stop()

        #ATR Energia
        from index_pt import coste_atr_energia
        atr_total, atr_detalle, atr_qh_eur_mwh = coste_atr_energia(
        df_consumo=st.session_state["consumo"],
        df_mercado=st.session_state["market"],
        tarifa=tarifa,)

        st.session_state["atr_total_energia"] = float(atr_total)
        st.session_state["atr_qh_eur_mwh"] = atr_qh_eur_mwh

        #ATR Potencia
        from ATR_Potencia import coste_atr_potencia

        potencias = st.session_state.get("potencias", {})

        modalidad_pot = st.session_state.get("modalidad_pot")

        if modalidad_pot == "BOE":
            try:
                atr_pot_total, atr_pot_detalle = coste_atr_potencia(tarifa, potencias)
            except Exception as e:
                st.error(f"No se pudo calcular el ATR de potencia: {e}")

            st.session_state["atr_total_potencia"] = float(atr_pot_total)

        else:
            precios_user = st.session_state.get("precio_pot", {}) or {}

            # Construir diccionario coef €/kW·año por periodo, según la tarifa y los nombres guardados en Paso 3
            t = (tarifa or "").replace(" ", "").upper()
            if t == "2.0TD":
                # En Paso 3 guardaste claves en minúsculas y hay un typo en P2 ("recio_pot_p2"), los atendemos todos
                cP1 = precios_user.get("precio_pot_p1") or precios_user.get("Precio potencia P1") or 0.0
                cP2 = precios_user.get("precio_pot_p2") or precios_user.get("Precio potencia P2") or precios_user.get("recio_pot_p2") or 0.0
                coef = {"P1": float(cP1), "P2": float(cP2)}
                periodos = ["P1", "P2"]
            else:
                # Para 3.0TD y 6.XTD guardaste: "Precio potencia P1"... "Precio potencia P6"
                coef = {
                f"P{i}": float(precios_user.get(f"Precio potencia P{i}", 0.0))
                for i in range(1, 7)}
                periodos = [f"P{i}" for i in range(1, 7)]

            # Comprobar que hay al menos un precio informado
            if all(v == 0.0 for v in coef.values()):
                st.warning("No hay precios de potencia informados en el Paso 3 (No BOE).")
                atr_pot_total, atr_pot_detalle = 0.0, pd.DataFrame(columns=["Periodo","Potencia (kW)","Coef €/kW·año","Coste (€)"])
            else:
                # Misma estructura de detalle que el cálculo BOE
                detalle = []
                for p in periodos:
                    pot_kW = float(potencias.get(p, 0.0))
                    c = float(coef.get(p, 0.0))
                    coste = pot_kW * c
                    detalle.append({
                    "Periodo": p,
                    "Potencia (kW)": pot_kW,
                    "Coef €/kW·año": c,
                    "Coste (€)": coste})
                atr_pot_detalle = pd.DataFrame(detalle)
                atr_pot_total = float(atr_pot_detalle["Coste (€)"].sum())

        # Guardar y mostrar
        st.session_state["atr_total_potencia"] = float(atr_pot_total)

        #Index PT
        if st.session_state.get("modalidad") == "Indexado pass through":
            if "market" not in st.session_state:
                st.error("Falta la base de mercado. Revisa la ruta del Excel en bbdd_mercado.py.")
                st.stop()

            tarifa = st.session_state.get("tarifa", "3.0TD")
            des, cg = _leer_desvios_y_cg()

            coste_total, detalle = coste_passthrough(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=tarifa,
                desvios=des,
                cg=cg,)

            st.session_state["coste_mercado_te"] = float(coste_total)

            # --- Dentro de "Indexado pass through" (PT) ---
            coste_total, detalle = coste_passthrough(
            df_consumo=st.session_state["consumo"],
            df_mercado=st.session_state["market"],
            tarifa=tarifa,
            desvios=des,
            cg=cg,)
            # Guarda precio €/MWh por QH (columna "Precio_unitario(€/MWh)")
            st.session_state["precio_qh_eur_mwh"] = detalle["Precio_unitario(€/MWh)"].rename("precio_eur_mwh")

        #PVPC
        if st.session_state.get("modalidad") == "PVPC":
            if "consumo" not in st.session_state:
                st.error("Falta el consumo para calcular PVPC.")
            elif "pvpc_df" not in st.session_state:
                st.error("Falta la tabla PVPC (Paso 3).")
            else:
                df_c = st.session_state["consumo"].copy()   # columnas: datetime, consumo [kWh]
                df_p = st.session_state["pvpc_df"].copy()   # columnas: datetime, precio_eur_mwh

                df = df_c.merge(df_p, on="datetime", how="left")
                if df["precio_eur_mwh"].isna().any():
                    n = int(df["precio_eur_mwh"].isna().sum())
                    st.warning(f"{n} registros de consumo sin precio PVPC. Se omiten del coste.")
                    df = df.dropna(subset=["precio_eur_mwh"])

                df["coste_qh"] = (df["consumo"] / 1000.0) * df["precio_eur_mwh"]  # kWh→MWh
                coste_pvpc_total = float(df["coste_qh"].sum())
                atr_eur = float(st.session_state.get("atr_total_energia", 0.0))
                total_pvpc_neto = coste_pvpc_total - atr_eur

            st.session_state["coste_mercado_te"] = float(total_pvpc_neto)

            # --- Dentro de "PVPC" (después de fusionar consumo con df_p) ---
            # df tiene columnas: ['datetime','consumo','precio_eur_mwh','coste_qh'] si mantienes ese orden
            # Guarda serie precio por QH alineada al consumo
            st.session_state["precio_qh_eur_mwh"] = df.set_index("datetime")["precio_eur_mwh"].rename("precio_eur_mwh")

        #Fijo
        from fijo import coste_fijo_energia

        if st.session_state.get("modalidad") == "Precio fijo":
            if "market" not in st.session_state or "consumo" not in st.session_state:
                st.error("Faltan datos: consumo o base de mercado.")
                st.stop()

            precios_te = st.session_state.get("precios_te", {}) 
            if not precios_te:
                st.warning("Introduce los precios por periodo en el Paso 3.")
            else:
                total_fijo, det_fijo, res_fijo = coste_fijo_energia(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=st.session_state["tarifa"],
                precios_te=precios_te,)

                atr_eur = float(st.session_state.get("atr_total_energia", 0.0))
                total_fijo_neto = total_fijo - atr_eur
                total_txt = fmt_eur(total_fijo_neto, 2) if 'fmt_eur' in globals() else f"{total_fijo_neto:,.2f}"

            st.session_state["coste_mercado_te"] = float(total_fijo_neto)

            total_fijo, det_fijo, res_fijo = coste_fijo_energia(
            df_consumo=st.session_state["consumo"],
            df_mercado=st.session_state["market"],
            tarifa=st.session_state["tarifa"],
            precios_te=precios_te,)
            # Guarda precio €/MWh por QH
            st.session_state["precio_qh_eur_mwh"] = det_fijo["precio_eur_MWh"].rename("precio_eur_mwh")


        #Index PP
        from index_pp import coste_indexado_pp

        if st.session_state.get("modalidad") == "Indexado pass pool":
            if "market" not in st.session_state or "consumo" not in st.session_state:
                st.error("Faltan datos de mercado o consumo.")
                st.stop()

            A = st.session_state.get("precios_Ai", {})
            C = st.session_state.get("precios_Ci", {})

            # Modo OMIE elegido en el Paso 3 (por defecto, mensual para compatibilidad)
            modo_omie = st.session_state.get("pp_omie_tipo", "mensual")

            if not A or not C:
                st.warning("Introduce los coeficientes Ai y Ci en el Paso 3.")
            else:
                total_pp, det_pp, res_pp = coste_indexado_pp(
                    df_consumo=st.session_state["consumo"],
                    df_mercado=st.session_state["market"],
                    tarifa=st.session_state["tarifa"],
                    coef_A=A,
                    coef_C=C,
                    modo_omie=modo_omie,
                )

                atr_eur = float(st.session_state.get("atr_total_energia", 0.0))
                total_pp_neto = total_pp - atr_eur
                total_txt = fmt_eur(total_pp_neto, 2)

            st.session_state["coste_mercado_te"] = float(total_pp_neto)

            # Segunda llamada para guardar el precio €/MWh por QH
            total_pp, det_pp, res_pp = coste_indexado_pp(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=st.session_state["tarifa"],
                coef_A=A,
                coef_C=C,
                modo_omie=modo_omie,
            )
            st.session_state["precio_qh_eur_mwh"] = det_pp["precio_eur_MWh"].rename("precio_eur_mwh")

        #FNEE
        from fnee import coste_fnee

        if "consumo" not in st.session_state:
            st.error("Falta cargar el consumo.")
        else:
            fnee_total, fnee_mwh = coste_fnee(st.session_state["consumo"])

            st.session_state["fnee_total"] = float(fnee_total)


        #FV Fijo
        # --- Excedentes FV en modalidad FIJA ---
        from fv_fijo import coste_excedentes_fijo

        tiene_fv = (st.session_state.get("fv") == "Sí")                 # <- tu flag real del Paso 2
        modo_fv  = st.session_state.get("modalidad_fv")                 # <- guardado en el Paso 3

        if tiene_fv and (str(modo_fv) == "Precio fijo"):
        # Construye el DataFrame esperado por la función (datetime + columna numérica)
            if "excedentes" in st.session_state and "fecha_exc" in st.session_state:
                df_exc = pd.DataFrame({
                "datetime": pd.to_datetime(st.session_state["fecha_exc"], errors="coerce"),
                "excedentes_kWh": pd.to_numeric(st.session_state["excedentes"], errors="coerce"),
                }).dropna()

                # Precio de excedentes tal y como lo guardas en el Paso 3
                precio_exc = float(st.session_state.get("precio_fv", {}).get("Precio FV", 0.0))

                ingresos_exc, energia_exc_mwh, precio_mwh_usado = coste_excedentes_fijo(
                df_excedentes=df_exc,
                precio_excedentes=precio_exc,)
                st.session_state["ingresos_fv"] = float(ingresos_exc)  

            else:
                st.warning("Faltan las curvas de excedentes FV del Paso 2.")

        # --- Excedentes FV en modalidad INDEXADA ---
        from fv_indexado import coste_excedentes_indexado

        if tiene_fv and (str(modo_fv) == "Indexado"):
            if "excedentes" in st.session_state and "fecha_exc" in st.session_state:
                # Construir DataFrame de excedentes (mismo formato que paso 2)
                df_exc = pd.DataFrame({
                "datetime": pd.to_datetime(st.session_state["fecha_exc"], errors="coerce"),
                "excedentes_kWh": pd.to_numeric(st.session_state["excedentes"], errors="coerce")
                }).dropna()

                cg_fv = float(st.session_state.get("cg_fv", {}).get("Costes Gestion FV", 0.0))

                ingresos_indexado, det_fv = coste_excedentes_indexado(
                df_excedentes=df_exc,
                df_mercado=st.session_state["market"],
                cg_fv=cg_fv)

                st.session_state["ingresos_fv"] = float(ingresos_indexado) 

            else:
                st.warning("Faltan las curvas de excedentes FV del Paso 2.")

        # Excesos de Potencia
        from excesos_pot import prepara_base_excesos

        if "consumo" not in st.session_state or "market" not in st.session_state:
            st.error("Faltan datos de consumo o base de mercado.")
        else:
            tarifa = st.session_state.get("tarifa", "3.0TD")
            potencias = st.session_state.get("potencias", {})

            try:
                base_excesos = prepara_base_excesos(
                df_consumo=st.session_state["consumo"],
                df_mercado=st.session_state["market"],
                tarifa=tarifa,
                potencias_dict=potencias)
                st.session_state["base_excesos"] = base_excesos

            except Exception as e:
                st.error(f"Error al preparar base de excesos: {e}")

        # Ex_pot contadores 1,2,3
        # ---- Cálculo excesos (contadores 1,2,3) ----
        from excesos_pot import calcula_excesos_cont_123 , calcula_excesos_cont_45

        if st.session_state.get("tarifa", "") != "2.0TD":
            tipo_cont = base_excesos.attrs.get("contador_tipo")
            if tipo_cont == "contador1,2,3":
                try:
                    total_exceso, res_exceso = calcula_excesos_cont_123(
                        base_excesos=base_excesos,
                        tarifa=st.session_state.get("tarifa", "3.0TD"))

                    # Formateo de columnas para mostrar
                    # Formateo de columnas para mostrar (usa nombres reales devueltos por la función)
                    res_exceso_fmt = (
                    res_exceso
                    .assign(
                    S_sum_raiz=lambda d: d["S_sum_raiz"].round(3),
                    coef=lambda d: d["coef"].round(6),
                    **{"coste_€": lambda d: d["coste_€"].round(2)}))

                except Exception as e:
                    st.error(f"No se pudo calcular el exceso (1,2,3): {e}")

                st.session_state["excesos_total"] = float(total_exceso)

            elif tipo_cont == "contador4,5":
                try:
                    total_exceso_45, res_exceso_45 = calcula_excesos_cont_45(
                    base_excesos=base_excesos,
                    tarifa=st.session_state.get("tarifa", "3.0TD"),)

                    res_exceso_45_fmt = res_exceso_45.assign(
                    coef=lambda d: d["coef"].round(6),
                    **{"coste_€": lambda d: d["coste_€"].round(2)})

                except Exception as e:
                    st.error(f"No se pudo calcular el exceso (4,5): {e}")

                st.session_state["excesos_total"] = float(total_exceso_45)

        # --- Impuesto Especial sobre la Electricidad (IEE) ---
        iee_pct = 5.11269 / 100.0

        base_iee = sum([
        float(st.session_state.get("coste_mercado_te", 0.0)),   # Mercado – término energía
        float(st.session_state.get("atr_total_energia", 0.0)),  # ATR energía
        float(st.session_state.get("atr_total_potencia", 0.0)), # ATR potencia
        float(st.session_state.get("excesos_total", 0.0)),      # Excesos de potencia
        float(st.session_state.get("fnee_total", 0.0)),         # FNEE
        ]) - float(st.session_state.get("ingresos_fv", 0.0))         # <— restar FV (ingreso)

        iee_total = base_iee * iee_pct

        st.session_state["iee_total"] = float(iee_total)

        # --- Resumen de costes anuales (tipo factura) ---
        st.subheader("Resumen de costes anuales")

        base_iee = sum([
        float(st.session_state.get("coste_mercado_te", 0.0)),
        float(st.session_state.get("atr_total_energia", 0.0)),
        float(st.session_state.get("atr_total_potencia", 0.0)),
        float(st.session_state.get("excesos_total", 0.0)),
        float(st.session_state.get("fnee_total", 0.0)),]) - float(st.session_state.get("ingresos_fv", 0.0))

        iee_total = float(st.session_state.get("iee_total", 0.0))
        total_costes = base_iee + iee_total
        st.session_state["costes_iniciales_total"] = float(total_costes)


        lineas = [
        ("Mercado – término energía",  float(st.session_state.get("coste_mercado_te", 0.0))),
        ("ATR energía",                float(st.session_state.get("atr_total_energia", 0.0))),
        ("ATR potencia",               float(st.session_state.get("atr_total_potencia", 0.0))),
        ("Excesos de potencia",        float(st.session_state.get("excesos_total", 0.0))),
        ("FNEE",                       float(st.session_state.get("fnee_total", 0.0))),
        ("Venta excedentes",      -float(st.session_state.get("ingresos_fv", 0.0))),
        ("Base IEE",                    base_iee),
        ("Impuesto electricidad (5,1127%)", iee_total),]

        df_factura = pd.DataFrame(lineas, columns=["Concepto", "Importe (€)"])

        # formateo euros con tu helper fmt_eur()
        fmt_eur_map = {"Importe (€)": lambda v: f"{fmt_eur(v, 2)} €"}
        st.table(style_tabla(df_factura, fmt_map=fmt_eur_map))

        # “caja” de total
        st.markdown(
        f"""
        <div style="
        background:#f6f9ff;border:1px solid #dfe7fb;padding:12px 16px;
        border-radius:10px; display:flex; justify-content:space-between; align-items:center;">
        <div style="font-weight:700;color:#1a2b4b;">TOTAL COSTE ELECTRICIDAD ANUAL</div>
        <div style="font-weight:800;font-size:1.15rem;color:#0f1c3f;">{fmt_eur(total_costes, 2)} €</div>
        </div>
        """,
        unsafe_allow_html=True,)

        st.caption("Distribución del coste anual por conceptos")

        # Construir dataframe de costes (mismos conceptos que en tu tabla)
        df_costes = pd.DataFrame([
        ("Mercado – término energía",  float(st.session_state.get("coste_mercado_te", 0.0))),
        ("ATR energía",                float(st.session_state.get("atr_total_energia", 0.0))),
        ("ATR potencia",               float(st.session_state.get("atr_total_potencia", 0.0))),
        ("Excesos de potencia",        float(st.session_state.get("excesos_total", 0.0))),
        ("FNEE",                       float(st.session_state.get("fnee_total", 0.0))),
        ("Impuesto electricidad",      float(st.session_state.get("iee_total", 0.0))),
        # Nota: “Ingresos FV (restan)” suele ser negativo; lo dejamos fuera del quesito para no distorsionar el %.
        ], columns=["Concepto","Importe"])

        df_costes_pos = df_costes[df_costes["Importe"] > 0].copy()
        suma_pos = df_costes_pos["Importe"].sum()
        if suma_pos > 0:
            df_costes_pos["%"] = df_costes_pos["Importe"] / suma_pos

            pie = (
                alt.Chart(df_costes_pos)
                .mark_arc(innerRadius=60)   # donut
                .encode(
                    theta=alt.Theta("Importe:Q", stack=True),
                    color=alt.Color("Concepto:N", legend=alt.Legend(title="Concepto")),
                    tooltip=[
                    alt.Tooltip("Concepto:N"),
                    alt.Tooltip("Importe:Q", format=",.2f"),
                    alt.Tooltip("%:Q", title="% del total", format=".1%")]).properties(height=320))
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("No hay costes positivos para graficar.")

        st.caption("Nota: los *Ingresos FV* (si existen) reducen el total y por ser negativos no se incluyen en el gráfico.")

        st.session_state["ctx"] = dict(
            market=st.session_state["market"],
            costes_iniciales_total=float(st.session_state["costes_iniciales_total"]),
            tarifa=st.session_state.get("tarifa", "3.0TD"),
            modalidad=st.session_state.get("modalidad"),
            precio_qh_eur_mwh=st.session_state.get("precio_qh_eur_mwh"),
            atr_total_energia=float(st.session_state.get("atr_total_energia", 0.0)),
            atr_total_potencia=float(st.session_state.get("atr_total_potencia", 0.0)),
            potencias=st.session_state.get("potencias", {}),
            fv_flag=(st.session_state.get("fv") == "Sí"),
            modalidad_fv=st.session_state.get("modalidad_fv"),
            precio_fv=st.session_state.get("precio_fv"),
            cg_fv=st.session_state.get("cg_fv"),
            pvpc_df=st.session_state.get("pvpc_df"),
            precios_te=st.session_state.get("precios_te", {}),
            precios_Ai=st.session_state.get("precios_Ai", {}),
            precios_Ci=st.session_state.get("precios_Ci", {}),)

        st.markdown("""<style>/* Botón primario (azul) mejorado */.stButton > button[kind="primary"]{background: linear-gradient(180deg,#1b2e57,#0f1c3f);
        color:#fff;font-weight:800;font-size:1rem;border:0;border-radius:12px;padding:10px 16px;box-shadow:0 6px 16px rgba(43,100,255,.35);
        letter-spacing:.3px;transition:all .15s ease-in-out;}.stButton > button[kind="primary"]:hover{filter:brightness(1.05);box-shadow:0 8px 20px rgba(43,100,255,.45);
        }.stButton > button[kind="primary"]:active{transform:translateY(1px);
        }.stButton > button[kind="primary"]:focus{outline:3px solid rgba(43,100,255,.35);
        outline-offset:2px;}</style>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=3))
        c2.button("Continuar»", use_container_width=True, type="primary", on_click=lambda: st.session_state.update(step=5))

        # ------ Paso 5 (Evaluador): Configurar solución a evaluar ------
    elif st.session_state.step == 5:
        st.header("Paso 5 · Configura la solución a evaluar")

        # =========================
        # 1) Cargar BBDD baterías/inversores (desde carpeta o subida manual)
        # =========================
        BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        bat_default = BASE_DIR / "bbdd_baterias.xlsx"
        inv_default = BASE_DIR / "bbdd_inversores.xlsx"

        with st.sidebar:
            bat_upload = None
            inv_upload = None
            if not bat_default.exists():
                bat_upload = st.file_uploader(
                    "BBDD baterías (bbdd_baterias.xlsx)",
                    type=["xlsx", "xls"],
                    key="ev_bbdd_bat_upload")

            if not inv_default.exists():
                inv_upload = st.file_uploader(
                    "BBDD inversores (bbdd_inversores.xlsx)",
                    type=["xlsx", "xls"],
                    key="ev_bbdd_inv_upload")

        BAT_SRC = bat_upload if bat_upload is not None else (bat_default if bat_default.exists() else None)
        INV_SRC = inv_upload if inv_upload is not None else (inv_default if inv_default.exists() else None)

        if BAT_SRC is None or INV_SRC is None:
            st.error(
                "No se han encontrado las BBDD (bbdd_baterias.xlsx / bbdd_inversores.xlsx). "
                "Colócalas en la misma carpeta que este .py o súbelas desde el menú lateral."
            )
            st.stop()

        @st.cache_data(show_spinner=False)
        def _read_excel_any(src):
            # src puede ser Path o UploadedFile
            return pd.read_excel(src)

        try:
            df_bat = _read_excel_any(BAT_SRC)
            df_inv = _read_excel_any(INV_SRC)
        except Exception as e:
            st.error(f"No se han podido leer las bases de datos Excel. Detalle: {type(e).__name__}: {e}")
            st.stop()

        # =========================
        # 2) Mapear columnas (mismo esquema por posiciones que estás usando en tu optimizador)
        #    OJO: si tu BBDD tiene otro orden de columnas, hay que ajustar aquí.
        # =========================
        cb = df_bat.columns
        ci = df_inv.columns

        # --- Baterías ---
        # Según tu propio código (Web_v4.py) en el paso 5 del optimizador:
        # marca=cb[1], modelo=cb[2], mods_S=cb[3], mods_P=cb[4], cap_kWh=cb[7], DoD=cb[8],
        # P_batt=cb[11], eta_b=cb[13], V_nom=cb[14], precio=cb[17]
        BAT = pd.DataFrame({
            "marca":   df_bat[cb[1]].astype(str).str.strip(),
            "modelo":  df_bat[cb[2]].astype(str).str.strip(),
            "mods_S":  pd.to_numeric(df_bat[cb[3]], errors="coerce").fillna(1).astype(int),
            "mods_P":  pd.to_numeric(df_bat[cb[4]], errors="coerce").fillna(1).astype(int),
            "cap_kWh": pd.to_numeric(df_bat[cb[7]], errors="coerce"),
            "DoD":     pd.to_numeric(df_bat[cb[8]], errors="coerce"),
            "P_batt":  pd.to_numeric(df_bat[cb[11]], errors="coerce"),
            "eta_b":   pd.to_numeric(df_bat[cb[13]], errors="coerce"),
            "V_nom":   pd.to_numeric(df_bat[cb[14]], errors="coerce"),
            "precio":  pd.to_numeric(df_bat[cb[17]], errors="coerce"),
        }).dropna(subset=["marca","modelo","cap_kWh","P_batt"]).reset_index(drop=True)

        # --- Inversores ---
        INV = pd.DataFrame({
            "marca":  df_inv[ci[1]].astype(str).str.strip(),
            "modelo": df_inv[ci[2]].astype(str).str.strip(),
            "fase":   df_inv[ci[4]].astype(str),
            "Vmin":   pd.to_numeric(df_inv[ci[5]], errors="coerce"),
            "Vmax":   pd.to_numeric(df_inv[ci[6]], errors="coerce"),
            "P_inv":  pd.to_numeric(df_inv[ci[7]], errors="coerce"),
            "P_fv":   pd.to_numeric(df_inv[ci[9]], errors="coerce").fillna(0.0),
            "eta_i":  pd.to_numeric(df_inv[ci[10]], errors="coerce"),
            "precio": pd.to_numeric(df_inv[ci[11]], errors="coerce"),
        }).dropna(subset=["marca","modelo","P_inv"]).reset_index(drop=True)

        st.session_state["BAT"] = BAT.copy()
        st.session_state["INV"] = INV.copy()

        # Normalizar fase y filtrar según instalación (igual que optimizador)
        def _norm_fase(x):
            s = str(x).strip().lower()
            return "tri" if any(k in s for k in ["tri", "3f", "3-f", "trif"]) else "mono"

        INV["fase"] = INV["fase"].apply(_norm_fase)
        instalacion = (st.session_state.get("instalacion", "Trifásica") or "").lower()
        fase_req = "mono" if "mono" in instalacion else "tri"
        INV = INV[INV["fase"] == fase_req].reset_index(drop=True)

        if BAT.empty:
            st.error("No se han encontrado baterías en la base de datos.")
            st.stop()
        if INV.empty:
            st.error("No se han encontrado inversores compatibles con la fase seleccionada (Mono/Tri).")
            st.stop()

        # =========================
        # 3) UI: selector de batería con buscador + mostrar potencia (P_batt)
        # =========================
        st.subheader("Batería")

        # Construir etiqueta “buscable” en un único selectbox
        bat_view = BAT.copy()
        bat_view["cap_str"] = bat_view["cap_kWh"].apply(lambda x: f"{float(x):.1f}".rstrip("0").rstrip("."))
        bat_view["label"] = bat_view.apply(
            lambda r: f"{r['marca']} {r['modelo']} · {r['cap_str']} kWh",
            axis=1)

        # Este campo extra mejora muchísimo la búsqueda:
        # permite encontrar por 105, "105 kWh", modelo parcial, etc.
        bat_view["search"] = bat_view.apply(
            lambda r: f"{r['marca']} {r['modelo']} {r['cap_str']}kwh {r['cap_str']} kwh {float(r['cap_kWh']):.0f}kwh",
            axis=1)

        # Añadimos "search" al label pero sin ensuciar visualmente (entre corchetes al final)
        # Así, si el usuario escribe 105 o 105kwh también filtra.
        bat_view["label_searchable"] = bat_view["label"] + "  [" + bat_view["search"] + "]"

        bat_opts = bat_view["label_searchable"].tolist()

        bat_sel = st.selectbox(
            "Buscar/seleccionar batería (escribe marca, modelo o kWh):",
            bat_opts,
            index=0,
            key="eval_bat_label",
            format_func=lambda x: x.split("  [")[0]  # muestra solo "marca modelo · XX kWh"
            )

        # Recuperar fila (ahora sí coincide)
        bat_row = bat_view.loc[bat_view["label_searchable"] == bat_sel].iloc[0]

        # Mostrar potencia carga/descarga
        st.info(f"Potencia de carga/descarga de la batería seleccionada: {float(bat_row['P_batt']):.1f} kW")

        # =========================
        # 4) UI: número de inversores (1..20) + N selectores con buscador
        # =========================
        st.subheader("Inversores")

        n_inv = st.selectbox(
            "Número de inversores:",
            list(range(1, 21)),
            index=int(st.session_state.get("eval_n_inv", 1)) - 1,
            key="eval_n_inv")

        inv_view = INV.copy()
        inv_view["p_str"] = inv_view["P_inv"].apply(lambda x: f"{float(x):.1f}".rstrip("0").rstrip("."))
        inv_view["label"] = inv_view.apply(
            lambda r: (
                f"{r['marca']} {r['modelo']} · "
                f"{r['p_str']} kW · "
                f"V[{float(r['Vmin']):.0f}-{float(r['Vmax']):.0f}]"),axis=1)
        inv_view["search"] = inv_view.apply(
            lambda r: (
                f"{r['marca']} {r['modelo']} "
                f"{r['p_str']}kw {r['p_str']} kw {float(r['P_inv']):.0f}kw"),axis=1)
        inv_view["label_searchable"] = inv_view["label"] + "  [" + inv_view["search"] + "]"

        inv_labels = []
        for i in range(int(n_inv)):  # <-- aquí ya tienes el valor
            st.markdown(f"**Inversor {i+1}**")

            sel = st.selectbox(
                f"Buscar / seleccionar inversor {i+1} (marca, modelo o kW):",
                inv_view["label_searchable"].tolist(),
                index=0,
                key=f"eval_inv_label_{i}")
            inv_labels.append(sel)

        st.session_state["eval_inv_labels"] = inv_labels

        from types import SimpleNamespace
        def _row_from_label_searchable(df_view, label):
            # df_view es el inv_view (o bat_view) que ya tiene label_searchable
            row = df_view.loc[df_view["label_searchable"] == label]
            if row.empty:
                return None
            return row.iloc[0]

        def _group_inv_rows(inv_view, selected_labels):
            """
            Convierte la lista de labels seleccionados (uno por inversor) en:
            - inv_rows: lista de filas (como objetos) únicos
            - counts: cuántas unidades de cada fila
            """
            counts_map = {}
            rows_map = {}

            for lab in selected_labels:
                r = _row_from_label_searchable(inv_view, lab)
                if r is None:
                    continue
                key = str(r["marca"]).strip() + "||" + str(r["modelo"]).strip()
                counts_map[key] = counts_map.get(key, 0) + 1
                rows_map[key] = r

            inv_rows = []
            counts = []
            for key, c in counts_map.items():
                inv_rows.append(SimpleNamespace(**rows_map[key].to_dict()))
                counts.append(int(c))

            return inv_rows, counts
        
        from types import SimpleNamespace

        def _guardar_config_eval_y_continuar(bat_view, inv_view):
            # 1) Recuperar batería seleccionada
            bat_sel = st.session_state.get("eval_bat_label")
            bat_row = bat_view.loc[bat_view["label_searchable"] == bat_sel]

            if bat_row.empty:
                st.error("No se ha podido recuperar la batería seleccionada.")
                return

            bat_row = bat_row.iloc[0]
            st.session_state["eval_bat_row_dict"] = bat_row.to_dict()

            # 2) Recuperar inversores seleccionados
            inv_labels = st.session_state.get("eval_inv_labels", [])
            if not inv_labels:
                st.error("No hay inversores seleccionados.")
                return

            # Agrupar inversores iguales
            counts = {}
            rows = {}

            for lab in inv_labels:
                r = inv_view.loc[inv_view["label_searchable"] == lab].iloc[0]
                key = f"{r['marca']}||{r['modelo']}"
                counts[key] = counts.get(key, 0) + 1
                rows[key] = r

            inv_rows = [SimpleNamespace(**rows[k].to_dict()) for k in counts]
            inv_counts = [counts[k] for k in counts]

            # 3) Crear inversor ficticio (IGUAL que optimizador)
            inv_mix = _build_combo_namespace(inv_rows, inv_counts)

            st.session_state["eval_inv_mix_dict"] = vars(inv_mix)
            st.session_state["eval_n_inv_total"] = 1

            # 2.bis) Guardar filas “humanas” para pintar la tabla de inversores en el Paso 6
            inv_display_rows = []
            for key in counts:
                r = rows[key]
                inv_display_rows.append({
                    "Marca":  str(r["marca"]),
                    "Modelo": str(r["modelo"]),
                    "Unidades": int(counts[key]),
                    "P_inv": float(r["P_inv"]),
                    "precio_unit": float(r["precio"]),
                })
            st.session_state["eval_inv_display_rows"] = inv_display_rows

            # 4) Pasar al paso 6
            st.session_state.update(step=6)

        # =========================
        # VALIDACIÓN: Potencia inversores vs potencia batería (±10 kW)
        # =========================
        P_batt = float(bat_row["P_batt"])  # kW

        potencias = st.session_state.get("potencias", {}) or {}
        P6 = potencias.get("P6", None)

        # Solo aplica si existe P6 (tarifas tipo 3.0TD/6.1TD...). En 2.0TD no hay P6.
        if P6 is not None:
            try:
                P6 = float(P6)
                if P6 > 0.0 and (P_batt > P6 + 1e-9):
                    st.warning(
                        f"La potencia de carga/descarga de la batería ({P_batt:.1f} kW) "
                        f"es mayor que la potencia contratada en P6 ({P6:.1f} kW)."
                    )
            except Exception:
                pass

        # Obtener potencia (P_inv) de cada inversor seleccionado
        inv_powers = []
        for sel in st.session_state.get("eval_inv_labels", []):
            row = inv_view.loc[inv_view["label_searchable"] == sel]
            if row.empty:
                continue
            inv_powers.append(float(row.iloc[0]["P_inv"]))

        P_inv_total = float(sum(inv_powers))  # kW

        tolerancia_kw = 10.0
        ok_potencia = abs(P_inv_total - P_batt) <= tolerancia_kw

        if not ok_potencia:
            st.warning(
                f"La suma de potencias de inversores ({P_inv_total:.1f} kW) "
                f"debe estar a ±{tolerancia_kw:.1f} kW de la potencia de carga/descarga de la batería "
                f"({P_batt:.1f} kW).")

        st.divider()
        c1, c2 = st.columns(2)
        c1.button("« Volver", use_container_width=True, on_click=lambda: st.session_state.update(step=4))
        # Este botón solo avanza; la simulación real la harás en el paso 6
        c2.button("Generar evaluación »",use_container_width=True,type="primary",disabled=not ok_potencia,on_click=_guardar_config_eval_y_continuar,kwargs={"bat_view": bat_view, "inv_view": inv_view},)

    # ------ Paso 6 (Evaluador): Evaluación de la solución configurada ------
    elif st.session_state.step == 6:
        st.header("Evalución de la solución configurada")

        from types import SimpleNamespace
        import pandas as pd
        import numpy as np
        import traceback

        # ---------- Helpers UI tabla ----------
        def _styler_base():
            return [
                {"selector":"th", "props":"background:#eef4ff; color:#1a2b4b; font-weight:700;"},
                {"selector":"tbody tr:nth-child(even)", "props":"background:#fafbff;"},
                {"selector":"td, th", "props":"border:1px solid #e9edf5; padding:6px 10px;"},
            ]
        def style_tabla(df, bold_first_col=True, fmt_map=None, highlight_total_label="Total"):
            stl = df.style.set_table_styles(_styler_base())
            if bold_first_col and len(df.columns) > 0:
                stl = stl.set_properties(subset=pd.IndexSlice[:, [df.columns[0]]], **{"font-weight":"700"})
            if fmt_map: stl = stl.format(fmt_map)
            def _row_style(s):
                if str(s.iloc[0]).strip().lower() == str(highlight_total_label).lower():
                    return ["font-weight:700; background-color:#fff7cc;"] * len(s)
                return [""] * len(s)
            stl = stl.apply(_row_style, axis=1)
            try:
                return stl.hide(axis="index")
            except Exception:
                return stl.hide_index()

        # ---------- Señales base necesarias del Paso 1–4 ----------
        if "consumo" not in st.session_state:
            st.error("Falta el consumo del Paso 1.")
            st.stop()
        if "precio_qh_eur_mwh" not in st.session_state:
            st.error("Falta el precio por QH del Paso 4.")
            st.stop()
        if "market" not in st.session_state:
            st.error("Falta la base de mercado en memoria (Paso 4).")
            st.stop()

        # Consumo alineado
        df_c = st.session_state["consumo"].copy()
        df_c["datetime"] = to_naive_utc_index(df_c["datetime"])
        df_c = df_c.dropna(subset=["datetime"]).copy()
        cons = pd.to_numeric(df_c["consumo"], errors="coerce").fillna(0.0).values
        base_dt_index = pd.DatetimeIndex(df_c["datetime"])
        base_dt = base_dt_index.values          # si más abajo necesitas el array
        n_slots = len(base_dt_index)


        # Precio €/MWh por QH alineado
        s_pre = st.session_state["precio_qh_eur_mwh"]
        if isinstance(s_pre, pd.Series):
            s_pre = s_pre.rename("precio_eur_mwh").copy()
            s_pre.index = to_naive_utc_index(s_pre.index)
        else:
            s_pre = s_pre.squeeze().rename("precio_eur_mwh")
            s_pre.index = to_naive_utc_index(s_pre.index)
        precios_uni = s_pre.to_frame().loc[~s_pre.index.duplicated(keep="last")]
        precio_vec = (
            df_c.merge(precios_uni, left_on="datetime", right_index=True, how="left")["precio_eur_mwh"]
            .fillna(method="ffill").fillna(method="bfill").to_numpy()
        )
        # ATR energía €/MWh por QH (si no existe, vector de ceros)
        s_atr = st.session_state.get("atr_qh_eur_mwh", None)
        if s_atr is not None:
            if isinstance(s_atr, pd.Series):
                s_atr = s_atr.rename("atr_qh_eur_mwh").copy()
                s_atr.index = to_naive_utc_index(s_atr.index)
            else:
                s_atr = s_atr.squeeze().rename("atr_qh_eur_mwh")
                s_atr.index = to_naive_utc_index(s_atr.index)

            atr_uni = s_atr.to_frame().loc[~s_atr.index.duplicated(keep="last")]
            atr_vec = (
                df_c.merge(atr_uni, left_on="datetime", right_index=True, how="left")["atr_qh_eur_mwh"]
                .fillna(0.0).to_numpy()
            )
        else:
            atr_vec = np.zeros(n_slots)

        # Excedentes/Generación (si hay FV)
        if st.session_state.get("fv") == "Sí" and "excedentes" in st.session_state and "fecha_exc" in st.session_state:
            df_exc = pd.DataFrame({"datetime": to_naive_utc_index(st.session_state["fecha_exc"]),
                                "exc": pd.to_numeric(st.session_state["excedentes"], errors="coerce")}).dropna()
            excedentes_vec = df_c[["datetime"]].merge(df_exc, on="datetime", how="left")["exc"].fillna(0.0).to_numpy()
        else:
            excedentes_vec = np.zeros(n_slots)
        if st.session_state.get("fv") == "Sí" and "generacion" in st.session_state and "fecha_gen" in st.session_state:
            df_gen = pd.DataFrame({"datetime": to_naive_utc_index(st.session_state["fecha_gen"]),
                                "gen": pd.to_numeric(st.session_state["generacion"], errors="coerce")}).dropna()
            generacion_vec = df_c[["datetime"]].merge(df_gen, on="datetime", how="left")["gen"].fillna(0.0).to_numpy()
        else:
            generacion_vec = np.zeros(n_slots)
        sun_mask = (generacion_vec > 0.0)
        autoconsumo_sin_bess = np.maximum(0.0, generacion_vec - excedentes_vec)
        load_vec = cons + autoconsumo_sin_bess

        # Potencias contratadas por QH (para peak-shaving)
        pot_dict = st.session_state.get("potencias") or {}
        tarifa = (st.session_state.get("tarifa") or "3.0TD").replace(" ", "").upper()
        df_aux = st.session_state.get("consumo_con_mercado")
        if df_aux is None:
            pot_contratada = np.full(n_slots, float(pot_dict.get("P1", 0.0)))
        else:
            dfp_left = df_c.copy(); dfp_left["datetime"] = to_naive_utc_index(dfp_left["datetime"])
            df_aux = df_aux.copy()
            if "datetime" in df_aux.columns:
                df_aux["datetime"] = to_naive_utc_index(df_aux["datetime"])
                dfp = dfp_left.set_index("datetime").join(df_aux.set_index("datetime"), how="left")
            else:
                df_aux.index = to_naive_utc_index(df_aux.index)
                dfp = dfp_left.set_index("datetime").join(df_aux, how="left")
            col_p = "periodos_20td" if tarifa == "2.0TD" else "periodos_no20td"
            per_num = pd.to_numeric(dfp[col_p], errors="coerce")
            miss = per_num.isna()
            if miss.any():
                per_num.loc[miss] = pd.to_numeric(dfp.loc[miss, col_p].astype(str).str.upper().str.extract(r"(\d+)")[0], errors="coerce")
            n_per = 3 if tarifa == "2.0TD" else 6
            per_num = per_num.fillna(1).astype(int).clip(1, n_per)
            per_lbl = "P" + per_num.astype(str)
            pot_contratada = per_lbl.map(lambda p: float(pot_dict.get(p, 0.0))).values

        # Precio de venta FV (si hay FV)
        if st.session_state.get("fv") == "Sí":
            modo_fv = str(st.session_state.get("modalidad_fv", ""))
            if modo_fv == "Precio fijo":
                precio_fv = float(st.session_state.get("precio_fv", {}).get("Precio FV", 0.0))
                precio_venta_fv_vec = np.full(n_slots, precio_fv, dtype=float)
            elif modo_fv == "Indexado":
                cg_fv = float(st.session_state.get("cg_fv", {}).get("Costes Gestion FV", 0.0))
                pv = (precio_vec - cg_fv); pv[pv < 0] = 0.0
                precio_venta_fv_vec = pv
            else:
                precio_venta_fv_vec = np.zeros(n_slots)
        else:
            precio_venta_fv_vec = np.zeros(n_slots)

        # ---------- Cargar y filtrar excels ----------
        @st.cache_data(show_spinner=False)
        def _read_excel_cached(path: str):
            return pd.read_excel(path)

        BAT_PATH = "data/bbdd_baterias.xlsx"
        INV_PATH = "data/bbdd_inversores.xlsx"

        df_bat = _read_excel_cached(BAT_PATH)
        df_inv = _read_excel_cached(INV_PATH)
        cb, ci = df_bat.columns, df_inv.columns

        # Map según posiciones que pediste
        BAT = pd.DataFrame({
        "marca":   df_bat[cb[1]].astype(str).str.strip(),
        "modelo":  df_bat[cb[2]].astype(str).str.strip(),
        "mods_S":  pd.to_numeric(df_bat[cb[3]], errors="coerce").fillna(1).astype(int),
        "mods_P":  pd.to_numeric(df_bat[cb[4]], errors="coerce").fillna(1).astype(int),
        "cap_kWh": pd.to_numeric(df_bat[cb[7]], errors="coerce"),
        "DoD":     pd.to_numeric(df_bat[cb[8]], errors="coerce"),
        "P_batt":  pd.to_numeric(df_bat[cb[11]], errors="coerce"),
        "eta_b":   pd.to_numeric(df_bat[cb[13]], errors="coerce"),
        "V_nom":   pd.to_numeric(df_bat[cb[14]], errors="coerce"),
        "precio":  pd.to_numeric(df_bat[cb[17]], errors="coerce"),
        }).dropna(subset=["cap_kWh","DoD","P_batt","eta_b","V_nom","precio"]).reset_index(drop=True)

        INV = pd.DataFrame({
            "marca":  df_inv[ci[1]],
            "modelo": df_inv[ci[2]],
            "fase":   df_inv[ci[4]].astype(str),
            "Vmin":   pd.to_numeric(df_inv[ci[5]], errors="coerce"),
            "Vmax":   pd.to_numeric(df_inv[ci[6]], errors="coerce"),
            "P_inv":  pd.to_numeric(df_inv[ci[7]], errors="coerce"),
            "P_fv":   pd.to_numeric(df_inv[ci[9]], errors="coerce").fillna(0.0),
            "eta_i":  pd.to_numeric(df_inv[ci[10]], errors="coerce"),
            "precio": pd.to_numeric(df_inv[ci[11]], errors="coerce"),
        }).dropna(subset=["fase","Vmin","Vmax","P_inv","eta_i","precio"]).reset_index(drop=True)

        # ---------- Crear contexto (foto del estado para los hilos) ----------
        ctx = build_ctx_from_session()
        _prepare_base_signals_from_session_state()

        # ---------- Bucle de búsqueda (PARALELO) ----------
        best_global, mejor_TIR_global, mejor_VAN_global = None, -1e18, -1e18

        tareas = []
        diagnostico = []

        b_dict   = st.session_state["eval_bat_row_dict"]     # batería seleccionada
        inv_dict = st.session_state["eval_inv_mix_dict"]     # inversor sintético (mezcla)
        n_total  = st.session_state["eval_n_inv_total"]      # total inversores físicos
        tareas.append((b_dict, inv_dict, n_total, ctx))

        from types import SimpleNamespace

        def _runner(args):
            b_d, inv_d, n, ctx_loc = args
            b_row   = SimpleNamespace(**b_d)
            inv_row = SimpleNamespace(**inv_d)
            return evaluar_combinacion(b_row, inv_row, n, dict(ctx_loc), return_sim=False)

        max_workers = max(1, os.cpu_count() - 1)
        results_local = []

        if not tareas:
            st.error("No hay combinaciones que evaluar (revisa compatibilidad).")
            st.stop()

        # Ejecutar en paralelo por procesos (como en v2)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_runner, t) for t in tareas]
            for fu in as_completed(futs):
                try:
                    results_local.append(fu.result())
                except Exception as e:
                    diagnostico.append(
                        "🔻 Fallo en combinación:\n"
                        + f"{type(e).__name__}: {e}\n"
                        + traceback.format_exc()
                    )

        # ---------- Seleccionar la mejor combinación según la TIR (y VAN como respaldo) ----------
        if not results_local:
            st.error("No se obtuvo ninguna simulación válida de las combinaciones evaluadas.")
            st.stop()

        def _score_res(res):
            """Prioriza TIR; si no hay TIR válida, usa VAN para desempatar."""
            import numpy as np
            tir = res.get("TIR", np.nan)
            van = res.get("VAN", -1e18)
            # Los que NO tienen TIR válida se van al final (flag -1)
            if tir is None or not np.isfinite(tir):
                return (-1, van)
            # Los que sí tienen TIR válida: flag 0, luego ordenamos por TIR y VAN
            return (0, tir, van)

        best_global = max(results_local, key=_score_res)
        mejor_TIR_global = best_global.get("TIR", np.nan)
        mejor_VAN_global = best_global.get("VAN", np.nan)

        import io

        # ---------- Resumen económico de TODAS las simulaciones ----------

        # Costes de la situación inicial (los mismos para todas las combinaciones)
        base_TE          = float(st.session_state.get("coste_mercado_te",        np.nan))
        base_ATR_energia = float(st.session_state.get("atr_total_energia",      np.nan))
        base_ATR_pot     = float(st.session_state.get("atr_total_potencia",     np.nan))
        base_excesos     = float(st.session_state.get("excesos_total",          np.nan))
        base_FNEE        = float(st.session_state.get("fnee_total",             np.nan))
        base_IEE         = float(st.session_state.get("iee_total",              np.nan))
        base_ing_FV      = -float(st.session_state.get("ingresos_fv",           0.0))  # mismo signo que en det_bess
        base_total       = float(st.session_state.get("costes_iniciales_total", np.nan))

        filas_resumen = []
        for res in results_local:
            try:
                ECO  = res.get("ECO")
                HW   = res.get("HW", {})
                diag = res.get("diag", {})
                det  = res.get("det_bess", {}) or {}

                filas_resumen.append({
                    # Identificación de la solución
                    "Bat_marca":          res.get("bat_marca", ""),
                    "Bat_modelo":         res.get("bat_modelo", ""),
                    "Capacidad_bat_kWh":  float(HW.get("E_CAP",  np.nan)),
                    "P_bat_kW":           float(HW.get("P_BATT", np.nan)),
                    "Inv_marca":          res.get("inv_marca", ""),
                    "Inv_modelo":         res.get("inv_modelo", ""),
                    "N_inversores":       int(res.get("n_inv", 0)),

                    # Económicos principales
                    "Capex_€":               float(diag.get("capex", np.nan)),
                    "Ahorro_anual_€":        float(res.get("ahorro_anual", np.nan)),
                    "VAN_€":                 float(res.get("VAN", np.nan)),
                    "TIR_%":                 float(res.get("TIR", np.nan)),
                    "Payback_simple_años":   float(diag.get("payback", np.nan)),

                    # Indicadores de uso de la batería
                    "kWh_cargados_red":      float(diag.get("kWh_carga_red", np.nan)),
                    "kWh_descargados":       float(diag.get("kWh_descarga",   np.nan)),
                    "Ciclos_equivalentes":   float(diag.get("ciclos_eq",      np.nan)),
                    "Spread_compra-venta_€/MWh": float(diag.get("spread", np.nan)),

                    # --- Costes ANTES (situación inicial, iguales para todas las filas) ---
                    "Antes_TE_€":           base_TE,
                    "Antes_ATR_energia_€":  base_ATR_energia,
                    "Antes_ATR_potencia_€": base_ATR_pot,
                    "Antes_Excesos_€":      base_excesos,
                    "Antes_FNEE_€":         base_FNEE,
                    "Antes_IEE_€":          base_IEE,
                    "Antes_Ingresos_FV_€":  base_ing_FV,
                    "Antes_Total_€":        base_total,

                    # --- Costes DESPUÉS (con batería, específicos de cada simulación) ---
                    "Desp_TE_€":           float(det.get("TE",           np.nan)),
                    "Desp_ATR_energia_€":  float(det.get("ATR_energia",  np.nan)),
                    "Desp_ATR_potencia_€": float(det.get("ATR_potencia", np.nan)),
                    "Desp_Excesos_€":      float(det.get("Excesos",      np.nan)),
                    "Desp_FNEE_€":         float(det.get("FNEE",         np.nan)),
                    "Desp_IEE_€":          float(det.get("IEE",          np.nan)),
                    "Desp_Ingresos_FV_€":  float(det.get("Ingresos_FV",  np.nan)),
                    "Desp_Total_€":        float(det.get("Total",        np.nan)),
                })
            except Exception:
                # Si alguna combinación viene rara, la saltamos sin romper todo
                continue

        if filas_resumen:
            df_resumen = pd.DataFrame(filas_resumen)

            # Ordenar como ranking (mejor a peor combinación)
            df_resumen = df_resumen.sort_values(
                by=["TIR_%", "VAN_€", "Ahorro_anual_€"],
                ascending=[False, False, False],
                na_position="last"
            ).reset_index(drop=True)

            # Guardamos en sesión para usarlo en otros sitios si quieres
            st.session_state["resumen_simulaciones"] = df_resumen

        # ---------- Selección del mejor global con la MISMA lógica que en v2 ----------
        best_global, mejor_TIR_global, mejor_VAN_global = None, -1e18, -1e18

        for res in results_local:
            tir = res.get("TIR", np.nan)
            van = res.get("VAN", np.nan)
            tir_ok, van_ok = np.isfinite(tir), np.isfinite(van)
            if tir_ok and (best_global is None or tir > mejor_TIR_global):
                mejor_TIR_global, mejor_VAN_global, best_global = tir, (van if van_ok else -1e18), res
            elif (not tir_ok) and van_ok and (best_global is None or (not np.isfinite(mejor_TIR_global) and van > mejor_VAN_global)):
                mejor_VAN_global, best_global = van, res

        if best_global is None:
            st.error("No se encontró ninguna combinación válida (revisa fase o compatibilidad de tensiones).")
            st.stop()

        # ---------- Inyecta la mejor y muestra resumen ----------
        ECO = best_global["ECO"]
        HW  = best_global["HW"]

        ahorro = float(best_global["ahorro_anual"])
        capex  = float(ECO.BASE_IMPONIBLE)
        payback_simple = capex / ahorro if ahorro > 0 else float("inf")

        # 🔁 Simula SOLO la ganadora y adjunta el DF para la UI (ahora con MILP v3)
        df_sim_best = simular_bess_milp(HW)
        best_global["SIM"] = df_sim_best

        # Guarda en session_state como ya hacías
        st.session_state["simul_bess_df"] = df_sim_best
        st.session_state["ECO"] = ECO
        st.session_state["HW"]  = HW

        tir_txt = (f"TIR: {best_global['TIR']:.2f}%  · "
        if np.isfinite(best_global["TIR"])
        else "TIR: —  · ")

        # --- Banner del óptimo con capacidades/potencias ---
        cap_kwh = float(HW["E_CAP"])
        p_batt  = float(HW["P_BATT"])
        p_inv   = float(HW["P_INV"])

        # ====== ESTILO AZUL UNIFICADO PARA TABLAS (PASO 5) ======
        from typing import Optional, Sequence

        _COLOR_HEADER_BG = "#eef4ff"   
        _COLOR_HEADER_TX = "#1a2b4b"   
        _COLOR_BORDER    = "#e9edf5"   
        _COLOR_ZEBRA     = "#fafbff"   

        def _guess_num_format(colname: str) -> Optional[str]:
            """Devuelve formato por nombre de columna: € o kWh si procede."""
            name = colname.lower()
            if any(k in name for k in ["eur", "€", "importe", "coste", "precio", "pago", "ahorro"]):
                return "{:,.2f} €"
            if any(k in name for k in ["kwh", "energ", "consumo", "gener", "descarga", "carga", "excedente"]):
                return "{:,.3f} kWh"
            if any(k in name for k in ["%", "porcentaje", "tasa"]):
                return "{:.2%}"
            return None

        def _formatters_from_df(df: pd.DataFrame):
            from pandas.api.types import is_numeric_dtype
            fmts = {}
            for c in df.columns:
                f = _guess_num_format(c)
                if not f:
                    continue
                # Solo aplicar formato si la columna es numérica
                if is_numeric_dtype(df[c]):
                    fmts[c] = f
                else:
                    # Si no es numérica pero la mayoría serían números al convertir, la dejamos pasar;
                    # si no, NO formateamos (evita el ValueError en columnas ya textuales como "1.234 €")
                    s_num = pd.to_numeric(df[c], errors="coerce")
                    ratio_num = s_num.notna().mean() if len(s_num) else 0.0
                    if ratio_num >= 0.8:
                        fmts[c] = f
            return fmts

        def style_blue(
            df: pd.DataFrame,
            *,
            bold_first_data_row: bool = False,
            total_row_labels: Sequence[str] = ("TOTAL",),
            bold_total_row: bool = False,
            caption: Optional[str] = None,
        ) -> pd.io.formats.style.Styler:
            """Crea un Styler con tema azul. Negrita opcional en primera fila de datos y/o fila TOTAL."""
            sty = df.style

            # Formatos numéricos sugeridos por nombre de columna
            fmts = _formatters_from_df(df)
            if fmts:
                sty = sty.format(fmts, na_rep="—")

            # Estilos base (cabecera y bordes)
            sty = sty.set_table_styles([
            # Cabecera de columnas (azul + negrita)
            {"selector": "th",
            "props": [
                ("background-color", _COLOR_HEADER_BG),
                ("color", f"{_COLOR_HEADER_TX} !important"),
                ("font-weight", "700 !important"),
                ("border-bottom", f"1px solid {_COLOR_BORDER}")
            ]},
            # Bordes y padding homogéneos
            {"selector": "td, th",
            "props": [
                ("border", f"1px solid {_COLOR_BORDER}"),
                ("padding", "6px 10px")]},])
            # Zebra suave
            def _zebra(rows):
                return [f"background-color: {_COLOR_ZEBRA}" if i % 2 else "" for i in range(len(rows))]
            sty = sty.apply(_zebra, axis=0)

            # Negrita en primera fila de datos (si se pide)
            if bold_first_data_row and len(df) > 0:
                def _bold_first_row(_row):
                    return ["font-weight:600" if _row.name == df.index[0] else "" ] * len(df.columns)
                sty = sty.apply(_bold_first_row, axis=1)

            # Negrita en filas TOTAL (por índice o por primera columna)
            if bold_total_row and len(df) > 0:
                total_set = set(str(x).strip().upper() for x in total_row_labels)
                first_col = df.columns[0] if len(df.columns) > 0 else None

                def _bold_total(_row):
                    by_index = str(_row.name).strip().upper() in total_set
                    by_first = False
                    if first_col is not None:
                        try:
                            by_first = str(_row[first_col]).strip().upper() in total_set
                        except Exception:
                            by_first = False
                    is_total = by_index or by_first
                    return ["font-weight:700" if is_total else ""] * len(df.columns)

                sty = sty.apply(_bold_total, axis=1)

            if caption:
                sty = sty.set_caption(caption)
            return sty
        
        # ===== FILA DE ICONOS DE DESCARGA (Batería e Inversores) =====
        from pathlib import Path
        import glob, base64

        # CSS: fila compacta sin huecos + “píldoras” con icono
        st.markdown("""
        <style>
        .icon-row{ display:flex; align-items:center; gap:8px; margin-top:-6px; margin-bottom:10px; }
        .icon-title{ font-size:.90rem; color:#6b778c; line-height:1; white-space:nowrap; }
        .icon-pill{
        display:inline-flex; width:28px; height:28px; border-radius:999px;
        align-items:center; justify-content:center; text-decoration:none;
        background:#F2F4F7; border:1px solid #DDE3EA; color:#1a2b4b; box-shadow:none;
        }
        .icon-pill:hover{ background:#EBEEF3; }
        .icon-pill:active{ transform:translateY(1px); }
        .icon-pill svg{ width:16px; height:16px; }
        </style>
        """, unsafe_allow_html=True)

        @st.cache_data(show_spinner=False)
        def _load_pdf_bytes(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        def _base_dir():
            return Path(__file__).resolve().parent

        def _find_folder(*relative_candidates):
            for rel in relative_candidates:
                p = (_base_dir() / rel).resolve()
                if p.exists():
                    return p
            return _base_dir()

        def _find_pdf_like(folder: Path, *tokens: str) -> Path | None:
            tokens = [t for t in (t.strip() for t in tokens) if t]
            for t in tokens:
                for pat in (f"*{t}*.pdf", f"*{t.replace(' ','*')}*.pdf"):
                    hits = sorted(glob.glob(str(folder / pat)))
                    if hits:
                        return Path(hits[0])
            return None

        def _pdf_link_html(path: Path, title: str) -> str:
            # data:URL para descarga directa, sin usar st.download_button
            b64 = base64.b64encode(_load_pdf_bytes(str(path))).decode()
            svg = """<svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3v10.17l3.59-3.58L17 11l-5 5-5-5 1.41-1.41L11 13.17V3h1zM5 19h14v2H5z"/>
            </svg>"""
            return f'<a class="icon-pill" href="data:application/pdf;base64,{b64}" download="{path.name}" title="{title}">{svg}</a>'

        def render_bateria_icons(marca: str, modelo: str):
            """Una fila con etiqueta + icono para la batería, pegada a la tabla."""
            folder = _find_folder("../Datasheets Bat-Inv/Baterias", "Datasheets Bat-Inv/Baterias")
            MAP = {
                "ebick lv":  "EBick LV - Cegasa - LV - Bat.pdf",
                "escal hv":  "Escal HV - Cegasa - HV - Bat.pdf",
                "expand hv": "Expand HV - Cegasa - HV - Bat.pdf",
                "us5000":    "US5000 - Pylontech - LV - Bat.pdf",
            }
            key = (modelo or "").lower().strip()
            pdf = None
            if key in MAP and (folder / MAP[key]).exists():
                pdf = (folder / MAP[key])
            if pdf is None:
                pdf = _find_pdf_like(folder, modelo, f"{marca} {modelo}")
            if pdf and pdf.exists():
                label = '<span class="icon-title">Descargar datasheet batería</span>'
                html = f'<div class="icon-row no-print">{label}{_pdf_link_html(pdf, f"Datasheet batería ({modelo})")}</div>'
                st.markdown(html, unsafe_allow_html=True)

        def render_inversores_icons(filas_inversores: list[dict]):
            """Etiqueta + 1 icono por modelo (sin duplicados)."""
            folder = _find_folder("../Datasheets Bat-Inv/Inversores", "Datasheets Bat-Inv/Inversores")
            vistos, links = set(), []
            for row in filas_inversores:
                marca  = str(row.get("Marca","")).strip()
                modelo = str(row.get("Modelo","")).strip()
                clave  = (marca.lower(), modelo.lower())
                if clave in vistos:
                    continue
                vistos.add(clave)
                pdf = None
                if marca.lower() == "solis":
                    token = "S6-EH3P" if "S6-EH3P" in modelo else modelo
                    pdf = _find_pdf_like(folder, token)
                else:
                    pdf = _find_pdf_like(folder, modelo, f"{marca} {modelo}")
                if pdf and pdf.exists():
                    links.append(_pdf_link_html(pdf, f"Datasheet inversor ({modelo})"))
            if links:
                label = '<span class="icon-title">Descargar datasheets inversores</span>'
                st.markdown(f'<div class="icon-row no-print">{label}{"".join(links)}</div>', unsafe_allow_html=True)

        # ====== FIN HELPERS ======
        
        # === Catálogo de la solución ganadora + mini presupuesto ===
        # Helpers que faltaban
        def _fmt_eur2(x):
            try:
                return f"{fmt_eur(float(x), 2)} €"
            except Exception:
                return x

        # Recupero objetos de la mejor combinación
        ECO = best_global["ECO"]
        HW  = best_global["HW"]

        # Strings de identificación
        inv_mix_str = str(best_global.get("inv_marca", ""))
        bat_marca   = str(best_global.get("bat_marca", "")).strip()
        bat_modelo  = str(best_global.get("bat_modelo", "")).strip()

        def _norm(s): return str(s).strip().lower()

        if not BAT.empty:
            mask = (BAT["marca"].str.strip().str.lower()==_norm(bat_marca)) & \
                (BAT["modelo"].str.strip().str.lower()==_norm(bat_modelo))
            if mask.any():
                bat_row = BAT.loc[mask].iloc[0]
            else:
                bat_row = pd.Series({"marca": bat_marca, "modelo": bat_modelo,
                                    "mods_S": 1, "mods_P": 1, "precio": ECO.COSTE_BATERIAS})
        else:
            bat_row = pd.Series({"marca": bat_marca, "modelo": bat_modelo,
                                "mods_S": 1, "mods_P": 1, "precio": ECO.COSTE_BATERIAS})

        # Costes para el mini presupuesto
        coste_bat  = float(ECO.COSTE_BATERIAS)
        coste_inv  = float(ECO.COSTE_INVERSORES)
        coste_ems  = float(ECO.COSTE_EMS)
        coste_inst = float(ECO.COSTE_INSTAL)
        base_imp   = float(ECO.BASE_IMPONIBLE)


        # ---------- TABLA BATERÍA ----------
        mods_S = int(pd.to_numeric(best_global.get("bat_mods_S", getattr(bat_row, "mods_S", 1)), errors="coerce") or 1)
        mods_P = int(pd.to_numeric(best_global.get("bat_mods_P", getattr(bat_row, "mods_P", 1)), errors="coerce") or 1)

        bat_tbl = pd.DataFrame([{
            "Tipo": "Batería",
            "Marca": bat_row.marca,
            "Modelo": bat_row.modelo,
            "Configuración": f"S{mods_S} · P{mods_P}",
            "Capacidad": f"{float(HW['E_CAP']):,.1f} kWh".replace(",", "."),
            "Potencia batería": f"{float(HW['P_BATT']):,.1f} kW".replace(",", "."),
            "Precio": _fmt_eur2(float(ECO.COSTE_BATERIAS)),
        }])

        # ---- BATERÍA ----
        sty_bat = style_blue(
            bat_tbl,
            bold_first_data_row=False,       # primera fila en negrita
            bold_total_row=False,           # SIN total
            caption="Batería seleccionada"
        )
        st.table(sty_bat)
        render_bateria_icons(marca=str(bat_row.marca), modelo=str(bat_row.modelo))

        # ---------- TABLA INVERSORES (sin subtotal) ----------
        inv_rows = []
        if inv_mix_str:
            import re
            pat = r"^\s*(\d+)x\s+([^\s]+)\s+(.+?)\s*$"  # "4x Solis S6-..." => unidades, marca, modelo
            for part in [p.strip() for p in inv_mix_str.split("+")]:
                m = re.match(pat, part)
                if not m:
                    continue
                unidades = int(m.group(1))
                marca_i  = m.group(2)
                modelo_i = m.group(3)
                inv_match = INV[(INV["marca"] == marca_i) & (INV["modelo"] == modelo_i)]
                if inv_match.empty:
                    inv_match = INV[INV["marca"] == marca_i].sort_values("P_inv", ascending=False).head(1)
                inv_i = inv_match.iloc[0]
                p_unit = float(inv_i.precio)
                inv_rows.append({
                    "Tipo": "Inversor",
                    "Marca": marca_i,
                    "Modelo": modelo_i,
                    "Unidades": unidades,
                    "Pot. unitaria": f"{float(inv_i.P_inv):,.1f} kW".replace(",", "."),
                    "Precio unitario": _fmt_eur2(p_unit),
                })

        if inv_rows:
            inv_tbl = pd.DataFrame(inv_rows)
            sty_inv = style_blue(
                inv_tbl,
                bold_first_data_row=False,   # encabezado en negrita (no la primera fila de datos)
                bold_total_row=False,        # sin total en catálogo
                caption="Inversores seleccionados"
            )
            st.table(sty_inv)
            render_inversores_icons(inv_rows)

        # ---------- MINI PRESUPUESTO ----------
        st.markdown("##### Presupuesto")
        bud = pd.DataFrame([
            ("Coste batería",        coste_bat),
            ("Coste inversores",     coste_inv),
            ("Coste EMS",            coste_ems),
            ("Coste instalación",    coste_inst),
            ("TOTAL base imponible", base_imp),
        ], columns=["Concepto","Importe (€)"])

        sty_bud = style_blue(
            bud,
            bold_first_data_row=False,
            total_row_labels=("TOTAL BASE IMPONIBLE","TOTAL base imponible","TOTAL"),
            bold_total_row=True,
            caption="Presupuesto de la propuesta"
        )
        st.table(sty_bud)

        # === Auditoría económica (desglose y sanity check) ===
        # --- Situación inicial (Paso 4) → construir det_base con los mismos nombres ---
        det_base = dict(
            TE              = float(st.session_state.get("coste_mercado_te", 0.0)),      # TE neto (como en Paso 4)
            ATR_energia     = float(st.session_state.get("atr_total_energia", 0.0)),
            ATR_potencia    = float(st.session_state.get("atr_total_potencia", 0.0)),
            Excesos         = float(st.session_state.get("excesos_total", 0.0)),
            FNEE            = float(st.session_state.get("fnee_total", 0.0)),
            IEE             = float(st.session_state.get("iee_total", 0.0)),
            Ingresos_FV     = -float(st.session_state.get("ingresos_fv", 0.0)),          # negativo en la base
            Total           = float(st.session_state.get("costes_iniciales_total", 0.0)),
        )
        best_global["det_base"] = det_base  # ← para que la auditoría lo encuentre

        # ================== ESTUDIO ECONÓMICO (solo propuesta ganadora, UI con 3 pestañas) ==================
        st.markdown("## Estudio económico")

        # --- Helpers financieros (robustos y coherentes con la simulación) ---
        def _npv(rate, cashflows):
            return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

        def _irr(cashflows, lower=-0.95, upper=3.0, tol=1e-7, max_iter=200):
            """IRR por bisección (robusto). lower > -1 para evitar singularidades."""
            def f(r): return sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cashflows))
            f_low, f_up = f(lower), f(upper)
            expand = 0
            while f_low * f_up > 0 and upper < 100 and expand < 20:
                upper *= 1.5
                f_up = f(upper)
                expand += 1
            if f_low * f_up > 0:
                return np.nan
            for _ in range(max_iter):
                mid = (lower + upper) / 2.0
                f_mid = f(mid)
                if abs(f_mid) < tol or (upper - lower) / 2.0 < tol:
                    return mid
                if f_low * f_mid <= 0:
                    upper, f_up = mid, f_mid
                else:
                    lower, f_low = mid, f_mid
            return mid

        def _payback(cash):
            """Devuelve el payback en años con decimales (interpolado linealmente)."""
            acc = cash[0]  # incluir inversión inicial (negativa)
            for t in range(1, len(cash)):
                prev_acc = acc
                acc += cash[t]
                if acc >= 0:
                    if cash[t] == 0:
                        return float(t)
                    frac = (0 - prev_acc) / (acc - prev_acc)
                    return (t - 1) + frac
            return float("inf")

        def _payback_descontado(cash, tasa):
            acc = 0.0
            for t, cf in enumerate(cash):
                acc += cf / ((1 + tasa) ** t)
                if acc >= 0:
                    return t
            return float("inf")

        def _build_cashflows(ECO, ahorro_base, escenario: str):
            """Genera flujos 0..N con crecimiento (IPC+electricidad) y degradación, menos OPEX, + valor residual al final."""
            n = int(ECO.VIDA_UTIL_ANIOS)
            cash = [0.0] * (n + 1)
            cash[0] = -float(ECO.BASE_IMPONIBLE)

            if escenario == "Moderado":
                ipc, elec = float(ECO.IPC_MOD), float(ECO.ELEC_MOD)
            elif escenario == "Optimista":
                ipc, elec = float(ECO.IPC_MOD) + float(ECO.IPC_OPT_DELTA), float(ECO.ELEC_OPT)
            else:  # Pesimista
                ipc, elec = float(ECO.IPC_MOD) + float(ECO.IPC_PES_DELTA), float(ECO.ELEC_PES)

            growth = (1 + ipc) * (1 + elec)
            degr   = float(ECO.DEGRAD_ANUAL)
            opex   = float(ECO.OPEX_ANUAL)
            tasa   = float(ECO.TASA_DESCUENTO)

            for t in range(1, n + 1):
                ahorro_t = (float(ahorro_base) * ((1 - degr) ** (t - 1)) * (growth ** (t - 1))) - opex
                cash[t] = ahorro_t
            cash[-1] += float(ECO.VALOR_RESIDUAL)

            van = _npv(tasa, cash)
            tir = _irr(cash)
            pb  = _payback(cash)
            pbd = _payback_descontado(cash, tasa)

            return {
                "cash": cash,
                "VAN": float(van),
                "TIR": (float(tir) * 100.0) if np.isfinite(tir) else np.nan,
                "Payback": float(pb),
                "Payback_desc": float(pbd),
                "params": {"ipc": ipc, "elec": elec, "growth": growth, "degr": degr, "opex": opex, "tasa": tasa}
            }

        ECO = best_global["ECO"]
        ahorro_base = float(best_global["ahorro_anual"])

        escenarios = ["Pesimista", "Moderado", "Optimista"]
        res_esc = {e: _build_cashflows(ECO, ahorro_base, e) for e in escenarios}

        # === RESUMEN INTRODUCTORIO POR ESCENARIO (TIR, VAN, Payback) + CONDICIONES ===

        def _fmt_eur_local(x):
            try:
                return f"{fmt_eur(float(x), 2)} €"
            except Exception:
                return x

        # Tabla KPI (una fila por escenario)
        kpi_rows = []
        for esc in escenarios:
            r = res_esc[esc]
            kpi_rows.append({
                "Escenario": esc,
                "TIR": (f"{r['TIR']:.2f}%" if np.isfinite(r["TIR"]) else "—"),
                "VAN": _fmt_eur_local(r["VAN"]),
                "Payback (años)": (f"{r['Payback']:.2f}" if np.isfinite(r["Payback"]) else "—"),
            })
        df_kpi = pd.DataFrame(kpi_rows, columns=["Escenario","TIR","VAN","Payback (años)"])
        st.table(style_blue(
            df_kpi,
            bold_first_data_row=False,
            bold_total_row=False,
            caption="Indicadores financieros por escenario"
        ))

        # Comentario compacto de condiciones por escenario (sin OPEX ni tasa descuento)
        st.markdown("###### Condiciones de los escenarios")
        txt_conds = []
        for esc in escenarios:
            p = res_esc[esc]["params"]
            txt_conds.append(
                f"**{esc}** — Aumento IPC anual: {p['ipc']*100:.2f} %, "
                f"Aumento coste electricidad anual: {p['elec']*100:.2f} %, "
                f"Degradación anual de la batería: {p['degr']*100:.3f} %"
            )

        # Estilo “comentario” discreto
        st.info("  \n".join(txt_conds))

        # --- Preconstruimos las tablas por escenario ---
        df_proj_by_esc = {}
        for esc in escenarios:
            r = res_esc[esc]
            cash = r["cash"]
            n = len(cash) - 1

            años = list(range(0, n + 1))
            flujo = cash
            tes_acum = np.cumsum(flujo).tolist()

            # SOLO las columnas solicitadas (nada de “descontado”, “factores”, ni OPEX)
            df = pd.DataFrame({
                "Año": años,
                "Flujo (€)": flujo,
                "Tesorería acumulada (€)": tes_acum,
            })
            df_proj_by_esc[esc] = df

        # --- UI: UNA sola tabla con 3 pestañas ---
        st.markdown("##### Proyección inversión durante su vida útil")
        # Orden deseado manualmente
        orden_esc = ["Moderado", "Optimista", "Pesimista"]
        tab_mod, tab_opt, tab_pes = st.tabs(orden_esc)

        # Recorre las pestañas en ese orden fijo
        for esc, tab in zip(orden_esc, [tab_mod, tab_opt, tab_pes]):
            with tab:
                df = df_proj_by_esc[esc]
                sty = style_blue(
                    df,
                    bold_first_data_row=False,
                    bold_total_row=False,
                    caption=f"Proyección {len(df)-1} años – {esc}"
                )
                st.table(sty)
        # ================== /ESTUDIO ECONÓMICO ==================

        # ================== Desglose económico y comprobaciones (sin expander) ==================
        import numpy as np, pandas as pd

        # === Tabla de comparación de costes con mismo estilo que catálogo ===
        st.markdown("##### Comparación costes antes VS después", unsafe_allow_html=True)

        det_base = best_global.get("det_base", {}) or {}
        det_bess = best_global.get("det_bess", {}) or {}

        conceptos = [
            "Mercado – término energía",
            "ATR energía",
            "ATR potencia",
            "Excesos de potencia",
            "FNEE",
            "Impuesto electricidad",
            "Ingresos FV (−)",
        ]
        antes = [
            float(det_base.get("TE", 0.0)),
            float(det_base.get("ATR_energia", 0.0)),
            float(det_base.get("ATR_potencia", 0.0)),
            float(det_base.get("Excesos", 0.0)),
            float(det_base.get("FNEE", 0.0)),
            float(det_base.get("IEE", 0.0)),
            float(det_base.get("Ingresos_FV", 0.0)),
        ]
        despues = [
            float(det_bess.get("TE", 0.0)),
            float(det_bess.get("ATR_energia", 0.0)),
            float(det_bess.get("ATR_potencia", 0.0)),
            float(det_bess.get("Excesos", 0.0)),
            float(det_bess.get("FNEE", 0.0)),
            float(det_bess.get("IEE", 0.0)),
            float(det_bess.get("Ingresos_FV", 0.0)),
        ]

        df_cmp = pd.DataFrame({
            "Concepto": conceptos,
            "Antes (€)": antes,
            "Después (€)": despues,
        })
        df_cmp["Ahorro (€)"] = df_cmp["Antes (€)"] - df_cmp["Después (€)"]

        # Totales
        total_antes  = float(det_base.get("Total", df_cmp["Antes (€)"].sum()))
        total_desp   = float(det_bess.get("Total", df_cmp["Después (€)"].sum()))
        total_ahorro = total_antes - total_desp
        df_cmp = pd.concat([
            df_cmp,
            pd.DataFrame([{
                "Concepto": "TOTAL",
                "Antes (€)": total_antes,
                "Después (€)": total_desp,
                "Ahorro (€)": total_ahorro,
            }])
        ], ignore_index=True)

        sty_desg = style_blue(
            df_cmp,                       
            bold_first_data_row=False,
            total_row_labels=("TOTAL", "Total"),
            bold_total_row=True,
            caption="Comparativa de costes para el calculo del ahorro"
        )
        st.table(sty_desg)

        st.markdown("## Análisis del funcionamiento técnico del sistema")
        import numpy as np
        # --- Cálculo / recuperación de ciclos equivalentes ---
        diag = best_global.get("diag", {}) or {}
        ciclos_eq_anuales = float(diag.get("ciclos_eq", np.nan))

        # Si no hubiera venido de diag (por seguridad), lo recalculamos desde la simulación
        if not np.isfinite(ciclos_eq_anuales) or ciclos_eq_anuales <= 0:
            df_sim_tmp = best_global.get("SIM")
            if df_sim_tmp is not None:
                try:
                    kWh_descarga = float(
                        pd.to_numeric(df_sim_tmp["descarga_kWh"], errors="coerce").fillna(0.0).sum()
                    )
                    e_cap = float(HW["E_CAP"])
                    ciclos_eq_anuales = kWh_descarga / max(e_cap, 1e-9)
                except Exception:
                    ciclos_eq_anuales = np.nan

        # Promedio diario (suponiendo 1 año de simulación)
        if np.isfinite(ciclos_eq_anuales):
            dias_sim = 365.0      # si quieres algo más fino, puedes calcularlo a partir del DF
            ciclos_eq_diarios = ciclos_eq_anuales / dias_sim

            st.markdown(
                f"**Ciclos equivalentes de la batería:** "
                f"{ciclos_eq_anuales:.0f} ciclos/año "
                f"({ciclos_eq_diarios:.2f} ciclos/día de media)."
            )

        # === Donuts lado a lado: Origen de la energía consumida (Antes vs Después) ===
        if st.session_state.get("fv") == "Sí":
            import plotly.express as px
        
            st.markdown("##### Origen de la energía consumida")

            # --- ANTES (situación inicial) ---
            # Consumo anual desde Paso 1
            df_cons_ini = st.session_state["consumo"]  # cols: ['datetime','consumo']
            cons_ini_kWh = float(pd.to_numeric(df_cons_ini["consumo"], errors="coerce").fillna(0.0).sum())

            # Generación y excedentes (Paso 2)
            gen_ini_kWh = float(pd.to_numeric(st.session_state.get("generacion"), errors="coerce").fillna(0.0).sum()) if "generacion" in st.session_state else 0.0
            exc_ini_kWh = float(pd.to_numeric(st.session_state.get("excedentes"), errors="coerce").fillna(0.0).sum()) if "excedentes" in st.session_state else 0.0

            auto_ini_kWh = max(0.0, gen_ini_kWh - exc_ini_kWh)
            # El reparto es sobre el consumo (no sobre la generación): limitamos el autoconsumo a lo realmente consumido
            auto_ini_kWh = min(auto_ini_kWh, cons_ini_kWh)
            red_ini_kWh  = max(0.0, cons_ini_kWh)

            # --- DESPUÉS (propuesta) ---
            # De tu simulación ganadora en Paso 5
            df_sim_best = st.session_state.get("simul_bess_df", best_global.get("SIM"))
            if df_sim_best is None or df_sim_best.empty:
                st.info("No hay simulación disponible para construir el gráfico 'Después'.")
            else:
                # Consumo de red después
                red_pro_kWh = float(pd.to_numeric(df_sim_best["cons_red_pro_kWh"], errors="coerce").fillna(0.0).sum())

                # Generación (usa la que ya venía del Paso 2; es la referencia pedida)
                gen_total_kWh = gen_ini_kWh  # (tal y como indicas)
                # Vertido nuevo tras batería (de la simulación)
                vertido_new_kWh = float(pd.to_numeric(df_sim_best["vertido_kWh"], errors="coerce").fillna(0.0).sum())

                auto_desp_kWh = max(0.0, gen_total_kWh - vertido_new_kWh)
                # El “consumo total después” que mostramos es el que se cubre por red_pro + autoconsumo.
                # Si por red+FV hubiera ligeras inconsistencias numéricas, lo acotamos a positivo.
                cons_desp_kWh = max(0.0, red_pro_kWh + auto_desp_kWh)

                # Normaliza por si el autoconsumo supera al consumo total (caso extremo de datos)
                auto_desp_kWh = min(auto_desp_kWh, cons_desp_kWh)
                red_pro_kWh   = max(0.0, red_pro_kWh)

                # --- UI: dos columnas con donuts ---
                cA, cB = st.columns(2)

                # Donut ANTES
                with cA:
                    df_antes = pd.DataFrame({
                        "Origen": ["RED", "FV"],
                        "kWh":    [red_ini_kWh, auto_ini_kWh]
                    })
                    if df_antes["kWh"].sum() > 1e-9:
                        figA = px.pie(
                            df_antes, names="Origen", values="kWh", hole=0.6,
                            title="Sin BESS · distribución del consumo anual"
                        )
                        figA.update_traces(
                            textposition="inside",
                            texttemplate="%{percent:.1%}",
                            hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent:.1%})<extra></extra>",marker=dict(colors=["#5CA9E6", "#F5A37A"])
                        )
                        figA.update_layout(margin=dict(l=10, r=10, t=50, b=10),
                                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
                        st.plotly_chart(figA, use_container_width=True)
                    else:
                        st.info("Sin consumo anual válido para el gráfico 'Antes'.")

                # Donut DESPUÉS
                with cB:
                    df_desp = pd.DataFrame({
                        "Origen": ["RED", "FV"],
                        "kWh":    [red_pro_kWh, auto_desp_kWh]
                    })
                    if df_desp["kWh"].sum() > 1e-9:
                        figB = px.pie(
                            df_desp, names="Origen", values="kWh", hole=0.6,
                            title="Con BESS · distribución del consumo anual"
                        )
                        figB.update_traces(
                            textposition="inside",
                            texttemplate="%{percent:.1%}",
                            hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent:.1%})<extra></extra>",marker=dict(colors=["#5CA9E6", "#F5A37A"])
                        )
                        figB.update_layout(margin=dict(l=10, r=10, t=50, b=10),
                                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
                        st.plotly_chart(figB, use_container_width=True)
                    else:
                        st.info("Sin consumo anual válido para el gráfico 'Después'.")

            # --- Origen de la energía almacenada en la batería ---
            st.markdown("##### Origen de la energía almacenada en la batería")

            # Usamos la simulación ganadora (ya cargada unos párrafos antes)
            df_sim_best = st.session_state.get("simul_bess_df", best_global.get("SIM"))

            if df_sim_best is None or df_sim_best.empty:
                st.info("No hay simulación disponible para analizar la energía almacenada en la batería.")
            else:
                # Energía anual cargada en batería desde RED y desde FV [kWh]
                carga_red_total = float(
                    pd.to_numeric(df_sim_best["carga_red_kWh"], errors="coerce").fillna(0.0).sum()
                )
                carga_fv_total = float(
                    pd.to_numeric(df_sim_best["carga_exc_kWh"], errors="coerce").fillna(0.0).sum()
                )

                energia_total = carga_red_total + carga_fv_total

                if energia_total > 1e-9:
                    df_origen_bess = pd.DataFrame({
                        "Origen": ["RED", "FV"],
                        "kWh":    [carga_red_total, carga_fv_total],
                    })

                    fig_bess = px.pie(
                        df_origen_bess,
                        names="Origen",
                        values="kWh",
                        hole=0.6,
                    )
                    fig_bess.update_traces(
                        textposition="inside",
                        texttemplate="%{percent:.1%}",
                        hovertemplate="%{label}<br>%{value:.0f} kWh (%{percent:.1%})<extra></extra>",marker=dict(colors=["#5CA9E6", "#F5A37A"])
                        )
                    fig_bess.update_layout(
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.15,
                            xanchor="center",
                            x=0.5,
                        ),
                    )

                    st.plotly_chart(fig_bess, use_container_width=True)
                else:
                    st.info("La batería no ha llegado a cargar energía en la simulación.")

        # === Formateo ===
        def _fmt_eur(x):
            try:
                return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return x

        # ================== Perfil medio diario (96 QH) de la opción ganadora ==================
        import plotly.express as px

        sim = best_global["SIM"].copy()

        # Asegurar columnas clave (si no existen, se crean)
        need = {
            "cons_red_pro_kWh": 0.0,
            "carga_red_kWh": 0.0,
            "carga_exc_kWh": 0.0,
            "descarga_kWh": 0.0,
            "vertido_kWh": 0.0,
            "fv_gen_kWh": 0.0,          # producción FV total
            "autoconsumo_kWh": 0.0      # autoconsumo directo
        }
        for c, v in need.items():
            if c not in sim.columns:
                sim[c] = v

        # Clave cuarto-hora
        sim["QH"] = pd.to_datetime(sim["datetime"]).dt.strftime("%H:%M")

        # Perfil medio anual (líneas)
        df_plot = (
            sim.groupby("QH", sort=True)
            .agg({
                "cons_red_pro_kWh": "mean",
                "carga_red_kWh": "mean",
                "carga_exc_kWh": "mean",
                "descarga_kWh": "mean",
                "vertido_kWh": "mean",
                "autoconsumo_kWh": "mean",
                "fv_gen_kWh": "mean",
            })
            .reset_index()
            .sort_values("QH")
        )

        df_plot["Carga batería (kWh)"]   = df_plot["carga_red_kWh"] + df_plot["carga_exc_kWh"]
        df_plot["Descarga batería (kWh)"] = df_plot["descarga_kWh"]
        df_plot["Consumo de red (kWh)"]   = df_plot["cons_red_pro_kWh"]
        df_plot["Excedentes FV (kWh)"]    = df_plot["vertido_kWh"]
        df_plot["Autoconsumo (kWh)"]      = df_plot["autoconsumo_kWh"]
        df_plot["Producción FV (kWh)"]    = df_plot["fv_gen_kWh"]

        cols_show = [
            "QH",
            "Consumo de red (kWh)",
            "Autoconsumo (kWh)",
            "Producción FV (kWh)",
            "Excedentes FV (kWh)",
            "Carga batería (kWh)",
            "Descarga batería (kWh)",
        ]
        df_show = df_plot[cols_show].copy()

        # ================== Día medio VERANO / INVIERNO (entre semana vs fin de semana) ==================
        sim["datetime"] = pd.to_datetime(sim["datetime"])
        sim["mes"] = sim["datetime"].dt.month
        # 0 = lunes ... 6 = domingo
        sim["dow"] = sim["datetime"].dt.dayofweek

        meses_verano   = {6, 7, 8, 9}   # Jun–Sep
        meses_invierno = {12, 1, 2, 3}     # Dic–Marzo

        def _perfil_medio(df):
            """
            Calcula el perfil medio diario (96 QH) de un subconjunto de simulación.

            Vectores que se devuelven (promedio horario):
            - Demanda del edificio (load_kWh)
            - Carga batería desde red
            - Carga batería desde FV
            - Carga total de batería (red + FV)
            - Descarga de batería
            """
            df = df.copy()
            df["QH"] = df["datetime"].dt.strftime("%H:%M")

            g = (
                df.groupby("QH", sort=True)
                .agg({
                    "load_kWh":      "mean",  # demanda del edificio
                    "carga_red_kWh": "mean",  # carga batería desde red
                    "carga_exc_kWh": "mean",  # carga batería desde FV (excedentes)
                    "descarga_kWh":  "mean",  # descarga batería
                })
                .reset_index()
                .sort_values("QH")
            )

            # Construimos las series que queremos pintar
            g["Demanda (kWh)"]             = g["load_kWh"]
            g["Carga desde red (kWh)"]     = g["carga_red_kWh"]
            g["Carga desde FV (kWh)"]      = g["carga_exc_kWh"]
            g["Carga batería (kWh)"]       = g["carga_red_kWh"] + g["carga_exc_kWh"]
            g["Descarga batería (kWh)"]    = g["descarga_kWh"]

            return g[[
                "QH",
                "Demanda (kWh)",
                "Carga desde red (kWh)",
                "Carga desde FV (kWh)",
                "Carga batería (kWh)",
                "Descarga batería (kWh)",
            ]]

        def _graf_area(df_show, titulo):
            import plotly.graph_objects as go

            x = df_show["QH"]
            series = [
                "Demanda (kWh)",
                "Carga desde red (kWh)",
                "Carga desde FV (kWh)",
                "Carga batería (kWh)",
                "Descarga batería (kWh)",
            ]

            fig = go.Figure()
            for s in series:
                y = df_show[s].astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=s,
                        mode="lines",
                        fill="tozeroy",   # mismo estilo área rellena
                        line=dict(width=2),
                    )
                )

            # Ticks cada 2 horas (2h = 8 tramos de 15 min)
            tick_vals = list(df_show["QH"].iloc[::8])
            fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_vals)

            fig.update_layout(
                title=titulo,
                xaxis_title="Hora del día",
                yaxis_title="kWh por QH (media)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=-0.3,
                    xanchor="center", x=0.5
                ),
                margin=dict(l=10, r=10, t=48, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # -------- Verano: laborable vs fin de semana --------
        df_ver = sim[sim["mes"].isin(meses_verano)]

        if len(df_ver):

            # Entre semana (lunes–viernes, dayofweek = 0..4)
            df_ver_lab = df_ver[df_ver["dow"] < 5]
            if len(df_ver_lab):
                df_show_ver_lab = _perfil_medio(df_ver_lab)
                _graf_area(df_show_ver_lab, "Día medio VERANO – Entre semana")
            else:
                st.info("No hay datos de verano entre semana en la simulación.")

            # Fin de semana (sábado–domingo, dayofweek = 5..6)
            df_ver_we = df_ver[df_ver["dow"] >= 5]
            if len(df_ver_we):
                df_show_ver_we = _perfil_medio(df_ver_we)
                _graf_area(df_show_ver_we, "Día medio VERANO – Fin de semana")
            else:
                st.info("No hay datos de verano en fin de semana en la simulación.")

        # -------- Invierno: laborable vs fin de semana --------
        df_inv = sim[sim["mes"].isin(meses_invierno)]

        if len(df_inv):

            df_inv_lab = df_inv[df_inv["dow"] < 5]
            if len(df_inv_lab):
                df_show_inv_lab = _perfil_medio(df_inv_lab)
                _graf_area(df_show_inv_lab, "Día medio INVIERNO – Entre semana")
            else:
                st.info("No hay datos de invierno entre semana en la simulación.")

            df_inv_we = df_inv[df_inv["dow"] >= 5]
            if len(df_inv_we):
                df_show_inv_we = _perfil_medio(df_inv_we)
                _graf_area(df_show_inv_we, "Día medio INVIERNO – Fin de semana")
            else:
                st.info("No hay datos de invierno en fin de semana en la simulación.")

        from io import BytesIO

        # ========= Exportar Paso 5 a PDF (con icono base64) =========
        import base64

        # 1) Icono en base64 (si no existe, seguimos sin icono)
        try:
            with open("logo_pdf.png", "rb") as f:
                pdf_icon_b64 = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            pdf_icon_b64 = ""

        # 2) CSS global de impresión (afecta a toda la página)
        st.markdown("""
        <style>
        .export-wrap { margin-top: 14px; }

        @media print {
            header, footer, [data-testid="stSidebar"], .stToolbar { display:none !important; }
            .block-container { padding-top:0 !important; padding-bottom:0 !important; }
            @page { margin: 12mm; }
            .stPlotlyChart, .stTable { break-inside: avoid; page-break-inside: avoid; }
            .print-break { break-before: page; page-break-before: always; }
            .no-print, .export-wrap { display:none !important; }
        }
        </style>
        """, unsafe_allow_html=True)

        # 3) Botón + JS dentro de un componente (aquí sí se ejecuta)
        html_button = f"""
        <html>
        <head>
        <style>
            body {{
            margin: 0;
            background: transparent;
            }}
            .pdf-btn {{
            display: flex;
            align-items: center;
            gap: 8px;
            height: 38px;
            padding: 6px 14px;
            background: #F2F4F7;
            color: #1a2b4b;
            border: 1px solid #DDE3EA;
            border-radius: 8px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background .15s;
            }}
            .pdf-btn:hover {{
            background: #EBEEF3;
            }}
            .pdf-icon {{
            width: 22px;
            height: 22px;
            }}
        </style>
        </head>
        <body>
        <div class="export-wrap no-print">
            <button class="pdf-btn" onclick="parent.window.print()">
            {f'<img src="data:image/png;base64,{pdf_icon_b64}" class="pdf-icon" alt="PDF" />' if pdf_icon_b64 else ''}
            Descargar propuesta en PDF
            </button>
        </div>
        </body>
        </html>
        """

        components.html(html_button, height=60, scrolling=False)
        # ========= FIN Exportar PDF =========

        # ================== DESCARGA EXCEL VECTORES (auditoría EMS) ==================
        from io import BytesIO
        import numpy as np
        import pandas as pd

        df_sim = st.session_state.get("simul_bess_df", None)

        if df_sim is None or df_sim.empty:
            st.info("No hay simulación BESS en memoria para descargar el Excel de auditoría.")
        else:
            df_sim = df_sim.copy()

            # --- g_load (energía comprada para consumo directo) ---
            # En tu MILP: cons_red_pro_kWh = g_load + carga_red_kWh
            if "cons_red_pro_kWh" in df_sim.columns and "carga_red_kWh" in df_sim.columns:
                df_sim["g_load_kWh"] = (
                    pd.to_numeric(df_sim["cons_red_pro_kWh"], errors="coerce").fillna(0.0)
                    - pd.to_numeric(df_sim["carga_red_kWh"], errors="coerce").fillna(0.0)
                )
            else:
                df_sim["g_load_kWh"] = np.nan

            # --- Periodos por QH (P1..P6 / P1..P3) ---
            tarifa = (st.session_state.get("tarifa") or "3.0TD").replace(" ", "").upper()
            n_per = 3 if tarifa == "2.0TD" else 6

            # Intentamos reconstruir el periodo a partir de consumo_con_mercado (que ya tiene periodos_20td o periodos_no20td)
            per_lbl = None
            try:
                df_aux = st.session_state.get("consumo_con_mercado", None)

                if df_aux is not None and len(df_aux):
                    df_aux = df_aux.copy()
                    if "datetime" in df_aux.columns:
                        df_aux["datetime"] = to_naive_utc_index(df_aux["datetime"])
                        df_join = df_sim[["datetime"]].copy()
                        df_join["datetime"] = to_naive_utc_index(df_join["datetime"])
                        dfp = df_join.set_index("datetime").join(df_aux.set_index("datetime"), how="left")
                    else:
                        df_aux.index = to_naive_utc_index(df_aux.index)
                        dfp = df_sim.set_index(to_naive_utc_index(df_sim["datetime"])).join(df_aux, how="left")

                    col_p = "periodos_20td" if tarifa == "2.0TD" else "periodos_no20td"
                    if col_p in dfp.columns:
                        per_num = pd.to_numeric(dfp[col_p], errors="coerce")
                        miss = per_num.isna()
                        if miss.any():
                            per_num.loc[miss] = pd.to_numeric(
                                dfp.loc[miss, col_p].astype(str).str.upper().str.extract(r"(\d+)")[0],
                                errors="coerce"
                            )
                        per_num = per_num.fillna(1).astype(int).clip(1, n_per)
                        per_lbl = ("P" + per_num.astype(str)).values
            except Exception:
                per_lbl = None

            if per_lbl is None:
                # Fallback: si no podemos reconstruir, dejamos "P?"
                per_lbl = np.array(["P?"] * len(df_sim), dtype=object)

            df_sim["periodo"] = per_lbl

            # --- Normalizar nombres solicitados ---
            # "carga de fv" = carga_exc_kWh en tu DF
            if "carga_exc_kWh" in df_sim.columns and "carga_fv_kWh" not in df_sim.columns:
                df_sim["carga_fv_kWh"] = pd.to_numeric(df_sim["carga_exc_kWh"], errors="coerce").fillna(0.0)
            elif "carga_fv_kWh" not in df_sim.columns:
                df_sim["carga_fv_kWh"] = 0.0

            # Asegurar numéricos (por si vienen como object)
            for c in ["load_kWh", "carga_red_kWh", "carga_fv_kWh", "descarga_kWh", "g_load_kWh"]:
                if c in df_sim.columns:
                    df_sim[c] = pd.to_numeric(df_sim[c], errors="coerce").fillna(0.0)
                else:
                    df_sim[c] = 0.0

            # --- Dataset final a exportar ---
            cols_export = [
                "datetime",
                "periodo",
                "load_kWh",
                "g_load_kWh",
                "carga_red_kWh",
                "carga_fv_kWh",
                "descarga_kWh",
            ]
            df_out = df_sim[cols_export].copy()
            df_out["datetime"] = pd.to_datetime(df_out["datetime"], errors="coerce")

            # --- Excel ---
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Vectores_QH")
            buffer.seek(0)

            st.download_button(
                label="📥 Descargar Excel (load, g_load, cargas, descarga, periodos)",
                data=buffer,
                file_name="vectores_simulacion_bess_qh.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        # ================== FIN DESCARGA EXCEL VECTORES ==================

# =========================
# ROUTER PRINCIPAL
# =========================
if st.session_state["page"] == "Optimizador":
    render_optimizador()
else:
    render_evaluador()













