import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import datetime as _dt



st.set_page_config(layout="wide", page_title="Smart Simulator")

# Logo (si existe)
try:
    st.image("abb_logo.png", width=100)
except:
    pass

st.title("Smart Simulator")
st.markdown("Empareja tu necesidad con la mejor solución de ABB.")

#Botones 
if "step" not in st.session_state:
    st.session_state.step = 1

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1 if st.session_state.step > 1 else 0
    
# Estilos personalizados para los botones
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #e50914;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #ff3b3b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


#AGREGADO 27.08.2025 BAS
# =====================================
# Constantes / Modelo de negocio
# =====================================
NIVELES = ["Esencial", "Básico", "Operación Efectiva", "Smart"]
TRANSICION_A_CASO = {
    ("Esencial", "Básico"): 1,
    ("Básico", "Operación Efectiva"): 2,
    ("Operación Efectiva", "Smart"): 3,
    ("Esencial", "Smart"): 4
}
# Esquemas programados para C2
ESQUEMAS = {
    1: {"alta": 0.50, "baja": 0.50},
    2: {"alta": 0.83, "baja": 0.17},
    3: {"alta": 0.79, "baja": 0.21},
}

# CAPEX por ventilador (valores de ejemplo; reemplazar por Excel si corresponde)
CAPEX_POR_VEN_DEFAULT = {
    ("Esencial", "Básico"):  4000.0,
    ("Básico", "Operación Efectiva"): 7000.0,
    ("Operación Efectiva", "Smart"): 12000.0,
    ("Esencial", "Smart"):  20000.0,
}











# =====================================
# Utilidades de cálculo (consistentes con Excel)
# =====================================
def energia_anual_mwh(p_kw: float, horas: float) -> float:
    return (p_kw * horas) / 1000.0

def calc_caso1(E0_mwh: float, reduccion_pct: float, tarifa: float):
    E1 = E0_mwh * (1 - reduccion_pct) ** 3
    A1_mwh = max(E0_mwh - E1, 0.0)
    A1_usd = A1_mwh * 1000.0 * tarifa
    return E1, A1_mwh, A1_usd

def calc_caso2(p_kw: float, horas: float, esquema_id: int, reduccion_baja_pct: float, tarifa: float, baseline_mwh: float):
    esquema = ESQUEMAS[esquema_id]
    h_alta = horas * esquema["alta"]
    h_baja = horas * esquema["baja"]
    E_alta = energia_anual_mwh(p_kw, h_alta)  # alta sin reducción adicional
    E_baja = energia_anual_mwh(p_kw * (1 - reduccion_baja_pct) ** 3, h_baja)
    E2 = E_alta + E_baja
    A2_mwh = max(baseline_mwh - E2, 0.0)
    A2_usd = A2_mwh * 1000.0 * tarifa
    return E2, A2_mwh, A2_usd

def calc_caso3(baseline_mwh: float, q_rel: float, tarifa: float):
    E3 = baseline_mwh * (q_rel ** 3)
    A3_mwh = max(baseline_mwh - E3, 0.0)
    A3_usd = A3_mwh * 1000.0 * tarifa
    return E3, A3_mwh, A3_usd

# Serie de payback "Excel-like": M1/M3/M5 gastos 40/40/20; ahorro desde M7 (impl=6m)
def serie_payback_excel_like(ahorro_anual_usd: float,
                             capex_usd: float,
                             meses_impl: int = 6,
                             prr_meses = (1,3,5),
                             prr_pct   = (0.40,0.40,0.20),
                             meses: int = 24):
    ahorro_mensual = ahorro_anual_usd / 12.0
    flujo = []
    for m in range(1, meses+1):
        gasto = 0.0
        for mm, pp in zip(prr_meses, prr_pct):
            if m == mm:
                gasto += capex_usd * pp
        ahorro = ahorro_mensual if m > meses_impl else 0.0
        flujo.append(ahorro - gasto)
    acumulado = list(np.cumsum(flujo))
    pb_mes = next((i+1 for i, v in enumerate(acumulado) if v >= 0), None)
    return list(range(1, meses+1)), flujo, acumulado, pb_mes


#AGREGAR MODULO DE ETROLLEY: 21.10.2025

# ===================== E-TROLLEY (2 pasos) — SIN CARGAR EXCEL =====================
# > Bloque autónomo. No modifica VoD.

# ---------- 0) DATA FIJA (del Excel, aquí “congelada”) ----------
ETROLLEY_DATA = {
    "cost_per_km_usd": 2_200_000.0,
    "maintenance_line_usd_y": 120_000.0,
    "default_energy_usd_kwh": 0.11,

    # ← OJO: dejamos 7.0 L/km como valor base en subida
    "truck_catalog": {
        "Komatsu 830E":   {"potencia_kW": 2500, "peso_t": 380, "capacidad_t": 230, "diesel_l_km_subida": 7.0},
        "Komatsu 860E":   {"potencia_kW": 2700, "peso_t": 400, "capacidad_t": 240, "diesel_l_km_subida": 7.0},
        "Komatsu 930E-4": {"potencia_kW": 2850, "peso_t": 410, "capacidad_t": 290, "diesel_l_km_subida": 7.0},
        "Komatsu 980E-4": {"potencia_kW": 3500, "peso_t": 420, "capacidad_t": 360, "diesel_l_km_subida": 7.0},
    },
    "retrofit_cost_usd": {
        "Komatsu 830E":   1_100_000,
        "Komatsu 860E":   1_200_000,
        "Komatsu 930E-4": 1_300_000,
        "Komatsu 980E-4": 1_500_000,
    },
    "new_truck_cost_usd": {
        "Komatsu 830E":   5_800_000,
        "Komatsu 860E":   6_100_000,
        "Komatsu 930E-4": 6_400_000,
        "Komatsu 980E-4": 7_500_000,
    },

    "vel_subida_diesel_kmh": 11.8,
    "factor_vel_trolley": 1.7,    # ↑ velocidad en subida con trolley
    "tasa_desc_npvt": 0.08,
    "horiz_anios": 10,
}

# (si algún modelo no trae 'diesel_l_km_subida', le ponemos 7.0 por defecto)
for _m, _d in ETROLLEY_DATA["truck_catalog"].items():
    _d.setdefault("diesel_l_km_subida", 7.0)



ETROLLEY_DATA.update({
    "diesel_price_usd_l": 1.00,
    "kwh_per_km_trolley": 12.0,
    "cycles_per_truck_per_year": 4200,
    "subidas_por_ciclo": 2,   # por defecto
})

def get_diesel_l_km_subida(model: str) -> float:
    """
    Devuelve L/km en subida para 'model'.
    Prioriza override en st.session_state[f"diesel_subida__{model}"].
    Falls back: truck_catalog[model]['diesel_l_km_subida'] o 7.0.
    """
    override_key = f"diesel_subida__{model}"
    if override_key in st.session_state:
        try:
            return float(st.session_state[override_key])
        except:
            pass
    cat = ETROLLEY_DATA.get("truck_catalog", {}).get(model, {})
    return float(cat.get("diesel_l_km_subida", cat.get("diesel_l_km", 7.0)))





for _m, _d in ETROLLEY_DATA["truck_catalog"].items():
    if "diesel_l_km_subida" not in _d and "diesel_l_km" in _d:
        _d["diesel_l_km_subida"] = _d["diesel_l_km"]




def et_speed_kpi():
    return ETROLLEY_DATA["vel_subida_diesel_kmh"] * ETROLLEY_DATA["factor_vel_trolley"]

def et_costs(dist_km, n_trucks, conting_pct, energy_usd_kwh, maint_line_y, inv_type, model):
    cost_per_km = ETROLLEY_DATA["cost_per_km_usd"]
    capex_line = cost_per_km * float(dist_km)
    unit = (ETROLLEY_DATA["retrofit_cost_usd" if inv_type=="Retrofit" else "new_truck_cost_usd"].get(model, 0.0))
    capex_trucks = unit * int(n_trucks)
    capex_total = (capex_line + capex_trucks) * (1 + conting_pct/100.0)
    opex_energy = 0.0  # pendiente de fórmula exacta
    opex_total = float(maint_line_y) + opex_energy
    return capex_total, opex_total, {"CAPEX línea": capex_line, "CAPEX camiones": capex_trucks}

def et_finance_monthly(capex_total_usd: float,
                       opex_anual_usd: float,
                       ahorro_anual_usd: float,
                       tasa_desc_anual: float = 0.08,
                       meses: int = 120):   # 10 años

    # PRR 40/40/20 en meses 1,3,5
    prr = {1: 0.40, 3: 0.40, 5: 0.20}
    cf = [0.0]*(meses+1)  # index 1..120

    for m, frac in prr.items():
        cf[m] -= capex_total_usd * frac

    # beneficios netos desde M7
    ben_m = (ahorro_anual_usd - opex_anual_usd) / 12.0
    for m in range(7, meses+1):
        cf[m] += ben_m

    # descuento mensual equivalente
    r_m = (1.0 + tasa_desc_anual)**(1/12) - 1
    cum_simple = []
    cum_npv    = []
    acc_s = acc_d = 0.0
    for m in range(1, meses+1):
        acc_s += cf[m]
        acc_d += cf[m]/((1+r_m)**m)
        cum_simple.append(acc_s)
        cum_npv.append(acc_d)

    # payback simple en meses
    pb_m = next((i+1 for i,v in enumerate(cum_simple) if v>=0), None)
    return cf[1:], cum_npv, pb_m, cum_simple



# ====== Camioncitos dinámicos por modelo (SVG) y helpers previos ======
TRUCK_STYLE = {
    "Komatsu 830E":   {"body":"#F2C94C", "cab":"#9CA3AF", "wheel":"#111827", "box_h":12},
    "Komatsu 860E":   {"body":"#F59E0B", "cab":"#9CA3AF", "wheel":"#111827", "box_h":13},
    "Komatsu 930E-4": {"body":"#FCD34D", "cab":"#9CA3AF", "wheel":"#111827", "box_h":14},
    "Komatsu 980E-4": {"body":"#FDBA74", "cab":"#9CA3AF", "wheel":"#111827", "box_h":15},
}

def truck_svg_one(model: str) -> str:
    s = TRUCK_STYLE.get(model, TRUCK_STYLE["Komatsu 830E"])
    W, H = 54, 30
    return f"""
    <svg width="{W}" height="{H}" viewBox="0 0 80 40" xmlns="http://www.w3.org/2000/svg" style="margin:0 6px">
      <rect x="2" y="{16 - s['box_h']/2:.1f}" width="46" height="{s['box_h']}" rx="3" fill="{s['body']}"/>
      <rect x="50" y="18" width="18" height="10" rx="2" fill="{s['cab']}"/>
      <circle cx="16" cy="32" r="6" fill="{s['wheel']}"/>
      <circle cx="40" cy="32" r="6" fill="{s['wheel']}"/>
      <circle cx="58" cy="32" r="6" fill="{s['wheel']}"/>
    </svg>
    """

def trucks_html(n: int, model: str) -> str:
    n = max(0, min(12, int(n)))
    svg = "".join(truck_svg_one(model) for _ in range(n))
    return f'<div style="display:flex;align-items:center;flex-wrap:wrap;padding:6px 4px">{svg}</div>'

def get_diesel_l_km_subida(model: str) -> float:
    cat = ETROLLEY_DATA["truck_catalog"].get(model, {})
    return float(cat.get("diesel_l_km_subida", cat.get("diesel_l_km", 7.0)))

#EMS:
# ===================== EMS (PASO 3 y PASO 4) =====================

# ===================== EMS (PASO 3 y PASO 4) =====================

# ===================== EMS (PASO 3 y PASO 4) =====================

def ems_ui_step3():
    st.header("Paso 3 de 4: Energy Management System — Parámetros del caso")
    st.caption("Caso de referencia basado en el ejemplo de Excel (puedes ajustar todos los valores).")

    # ---------- Parámetros generales ----------
    col1, col2 = st.columns(2)
    with col1:
        # Excel: 95% disponibilidad, 0.05 USD/kWh
        disponibilidad = st.number_input(
            "Disponibilidad promedio (%)",
            0.0, 100.0,
            95.0,
            0.5
        ) / 100.0

        precio_kwh = st.number_input(
            "Precio energía (USD/kWh)",
            0.01, 5.00,
            0.05,
            0.01
        )

        # Excel: 10% optimista, 5% moderado
        ahorro_opt_pct = st.number_input(
            "Ahorro estimado optimista (%)",
            1.0, 30.0,
            10.0,
            0.5
        ) / 100.0

        ahorro_mod_pct = st.number_input(
            "Ahorro estimado moderado (%)",
            1.0, 30.0,
            5.0,
            0.5
        ) / 100.0

    with col2:
        # Excel: costo EF de referencia 700 kUSD
        inversion_kusd = st.number_input(
            "Inversión estimada del EMS (kUSD)",
            0.0, 5_000.0,
            700.0,
            10.0
        )
        inversion_usd = inversion_kusd * 1000.0

        _ = st.checkbox("Incluir subprocesos secundarios", value=True)
        _ = st.number_input("Peso relativo de subprocesos (%)", 0.0, 50.0, 5.0, 0.5)

    st.markdown("---")
    st.subheader("Áreas / procesos a monitorear")

    # ---------- Catálogo base (caso Excel) ----------
    # Potencia MW (EA) y QTY exactamente como en tu hoja:
    # Primary 0.8x2; Overland 0.4x3; SAG 24x2; Ball 16.4x2; ISA 0.2x3 (1+2)
    base_catalog = {
        "Primary Crusher":   {"mw": 0.8,  "qty": 2},
        "Overland Conveyor": {"mw": 0.4,  "qty": 3},
        "SAG Mill":          {"mw": 24.0, "qty": 2},
        "Ball Mill":         {"mw": 16.4, "qty": 2},
        "ISA Mill M3000":    {"mw": 0.2,  "qty": 3},  # 0.2 MW x 3 = 0.6 MW total (dos filas del Excel)
    }

    # CSS para que el checkbox parezca círculo tipo “radio”
    st.markdown("""
        <style>
        .ems-area-row {
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
        }
        .ems-area-row:hover {
            border-color: #d1d5db;
        }
        /* hace circulares los checkboxes de esta sección */
        div[data-testid="stCheckbox"] input[type="checkbox"] {
            border-radius: 50%;
            width: 18px;
            height: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Estado en sesión ----------
    if "ems_procs" not in st.session_state:
        # Inicializa con el catálogo base + flag expanded=True
        st.session_state.ems_procs = {
            k: {"mw": v["mw"], "qty": v["qty"], "expanded": True}
            for k, v in base_catalog.items()
        }
    else:
        # Asegura que todas las claves del catálogo existan
        for k, v in base_catalog.items():
            st.session_state.ems_procs.setdefault(
                k, {"mw": v["mw"], "qty": v["qty"], "expanded": True}
            )
        # Compatibilidad con versión anterior que usaba "enabled"
        for k, data in st.session_state.ems_procs.items():
            if "expanded" not in data:
                data["expanded"] = data.get("enabled", True)

    # ---------- UI por área: círculo SOLO para expandir/colapsar ----------
    for nombre in base_catalog.keys():
        data = st.session_state.ems_procs[nombre]
        expanded_default = data.get("expanded", True)

        with st.container():
            st.markdown('<div class="ems-area-row">', unsafe_allow_html=True)

            # fila principal: círculo + nombre
            col_a = st.columns([1])[0]
            with col_a:
                expanded = st.checkbox(
                    nombre,
                    key=f"ems_sel_{nombre}",
                    value=expanded_default
                )
            st.session_state.ems_procs[nombre]["expanded"] = expanded

            # si está "marcado", mostramos potencia y cantidad (pero SIEMPRE entra al cálculo)
            if expanded:
                col_mw, col_qty = st.columns(2)
                with col_mw:
                    mw = st.number_input(
                        "Potencia MW (EA)",
                        0.0, 200.0,
                        float(data["mw"]),
                        0.1,
                        key=f"mw_{nombre}"
                    )
                with col_qty:
                    qty = st.number_input(
                        "Cantidad (QTY)",
                        0, 20,
                        int(data["qty"]),
                        1,
                        key=f"qty_{nombre}"
                    )
                st.session_state.ems_procs[nombre]["mw"] = mw
                st.session_state.ems_procs[nombre]["qty"] = qty

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Resumen y cálculos (TODAS las áreas cuentan) ----------
    if st.session_state.ems_procs:
        df = pd.DataFrame(
            [{
                "Área / Proceso": k,
                "Potencia MW (EA)": v["mw"],
                "Cantidad (QTY)":   v["qty"],
            } for k, v in st.session_state.ems_procs.items()]
        )
    else:
        df = pd.DataFrame(columns=["Área / Proceso", "Potencia MW (EA)", "Cantidad (QTY)"])

    df["SubTotal MW"] = df["Potencia MW (EA)"] * df["Cantidad (QTY)"]
    df["Disponibilidad"] = disponibilidad
    df["Costo Energía (USD/año)"] = (
    df["SubTotal MW"] * 1000 * 8760 * disponibilidad * precio_kwh
    )

    
    total_cost = float(df["Costo Energía (USD/año)"].sum()) if not df.empty else 0.0
    ahorro_opt = total_cost * ahorro_opt_pct
    ahorro_mod = total_cost * ahorro_mod_pct

    st.markdown("### Resumen calculado")
    st.dataframe(
        df[["Área / Proceso", "Potencia MW (EA)", "Cantidad (QTY)", "SubTotal MW", "Costo Energía (USD/año)"]],
        use_container_width=True,
        height=min(300, 60 + 28 * max(1, len(df)))
    )

  #  colA, colB = st.columns(2)
  #  colA.metric("Costo total de energía (año)", f"KUSD {total_cost/1000:,.0f}")
  #  colB.metric("Costo con EMS (optimista)", f"KUSD {(total_cost - ahorro_opt)/1000:,.0f}")

  #  st.markdown("---")
  #  colC, colD = st.columns(2)
  #  colC.metric("Ahorro anual (optimista)", f"KUSD {ahorro_opt/1000:,.0f}")
  #  colD.metric("Ahorro anual (moderado)", f"KUSD {ahorro_mod/1000:,.0f}")

    # ---------- Navegación ----------
    st.markdown("---")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("◀ Volver a priorización", use_container_width=True, key="ems_back_prior"):
            st.session_state.ems_active = False
            st.session_state.pop("ems_params", None)
            st.session_state.step = 2
            st.rerun()
    with b2:
        if st.button("Calcular rentabilidad ▶", type="primary", use_container_width=True, key="ems_go_calc"):
            st.session_state.ems_active = True
            st.session_state.ems_params = dict(
                total_cost=total_cost,
                ahorro_opt=float(ahorro_opt),
                ahorro_mod=float(ahorro_mod),
                inversion=float(inversion_usd),
            )
            st.session_state.step = 4  # Va al Paso 4 (EMS) donde usas ems_ui_step4()
            st.rerun()



def ems_ui_step4():
    import numpy as np
    import plotly.graph_objects as go

    st.header("Paso 4 de 4: Energy Management System — Cálculo de Rentabilidad")

    # Parámetros calculados en el Paso 3
    P = st.session_state.get("ems_params", {})
    if not P:
        st.warning("Primero completa el Paso 3 (EMS — Parámetros del caso).")
        if st.button("Volver a EMS — Parámetros", use_container_width=True, key="ems_go_back_p3"):
            st.session_state.step = 3
            st.rerun()
        return

    ahorro_opt = float(P["ahorro_opt"])
    ahorro_mod = float(P["ahorro_mod"])
    inversion  = float(P["inversion"])

    # ========= BLOQUE: RESUMEN DE INFORMACIÓN (igual estilo que VoD / E-Trolley) =========
    st.markdown("### Resultado del análisis :")

    card_line_style = """
        background-color: #f7f8f9;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 80%;
    """

    def _get_priority_color(idx: int) -> str:
        colores = ["#fbeaea", "#fff3e0", "#fffde7"]
        return colores[idx] if idx < len(colores) else "#f7f8f9"

    etiquetas_prioridad = [
        "🔴 Prioridad Alta",
        "🟠 Prioridad Media",
        "🟡 Prioridad Baja"
    ]

    col1, col2 = st.columns([1, 1.2])

    # ---- Columna izquierda: resumen de datos base ----
    with col1:
        st.subheader("Resumen de información")

        tipo_mina      = st.session_state.get("tipo_mina", "—")
        tipo_material  = st.session_state.get("tipo_material", "—")
        produccion     = st.session_state.get("produccion", "—")

        st.markdown(f"<div style='{card_line_style}'>• Tipo de mina: <b>{tipo_mina}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Material extraído: <b>{tipo_material}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Producción: <b>{produccion}</b></div>", unsafe_allow_html=True)
       # st.markdown(f"• **Disponibilidad promedio:** {st.session_state.ems_disponibilidad:.2f} %")
       # st.markdown(f"• **Precio energía:** {st.session_state.ems_precio_energia:.3f} USD/kWh")
       # st.markdown(f"• **Ahorro estimado optimista:** {st.session_state.ems_ahorro_opt:.2f} %")
       # st.markdown(f"• **Ahorro estimado moderado:** {st.session_state.ems_ahorro_mod:.2f} %")
       # st.markdown(f"• **Inversión estimada EMS:** {st.session_state.ems_inversion:.1f} kUSD")
    # ---- Columna derecha: desafíos seleccionados ----
    with col2:
        st.subheader("Desafíos seleccionados")

        prioridades = st.session_state.get("prioridades", [])
        for idx, d in enumerate(prioridades[:3]):
            bg_color = _get_priority_color(idx)
            etiqueta = etiquetas_prioridad[idx] if idx < len(etiquetas_prioridad) else f"Prioridad {idx+1}"

            st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    padding: 10px 15px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    font-size: 80%;
                ">
                    <b>{etiqueta}:</b> {d}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ========= BLOQUE: INDICADORES ESTILO E-TROLLEY =========

    def _fmt_short(x: float, decimals: int = 1) -> str:
        x = float(x); a = abs(x)
        if a >= 1e9: return f"{x/1e9:.{decimals}f} B"
        if a >= 1e6: return f"{x/1e6:.{decimals}f} M"
        if a >= 1e3: return f"{x/1e3:.0f} K"
        return f"{x:,.0f}"

    # Flujo mensual para payback (misma lógica que el Excel):
    # CAPEX 40/40/20 en meses 1,3,5 y ahorros desde el mes 7
    meses = list(range(1, 25))
    flujo_opt = [0.0] * 24
    flujo_mod = [0.0] * 24

    prr = {1: 0.40, 3: 0.40, 5: 0.20}  # CAPEX 40/40/20 en meses 1,3,5

    for m, frac in prr.items():
        flujo_opt[m-1] -= inversion * frac
        flujo_mod[m-1] -= inversion * frac

    for i in range(6, 24):  # desde M7 ahorros
        flujo_opt[i] += ahorro_opt / 12.0
        flujo_mod[i] += ahorro_mod / 12.0

    acum_opt = list(np.cumsum(flujo_opt))
    acum_mod = list(np.cumsum(flujo_mod))

    def _payback(acum):
        for i, v in enumerate(acum, start=1):
            if v >= 0:
                return i
        return None

    pb_opt = _payback(acum_opt)
    pb_mod = _payback(acum_mod)

    # ---- Tarjetas tipo KPI (3 columnas, estilo E-Trolley) ----
    st.subheader("Resultados económicos del EMS")

    valor_opt = f"USD {_fmt_short(ahorro_opt,1)}"
    valor_mod = f"USD {_fmt_short(ahorro_mod,1)}"
    valor_inv = f"USD {_fmt_short(inversion,1)}"

    delta_opt = f"Payback optimista: M{pb_opt}" if pb_opt else ""
    delta_mod = f"Payback moderado: M{pb_mod}" if pb_mod else ""
    delta_inv = "Inversión inicial referencial"

    colA, colB, colC = st.columns(3)

    kpi_style = """
        <div style="
            background-color:#f5f5f5;
            border-radius:12px;
            padding:18px 16px;
            text-align:center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        ">
            <h2 style="margin:0; font-weight:700; color:#000;">{valor}</h2>
            <p style="margin:0; color:#555; font-size:15px;">{subtitulo}</p>
            <p style="margin-top:5px; color:#008000; font-size:13px;">{delta}</p>
        </div>
    """

    colA.markdown(
        kpi_style.format(
            valor=valor_opt,
            subtitulo="Ahorro anual (optimista)",
            delta=delta_opt
        ),
        unsafe_allow_html=True
    )
    colB.markdown(
        kpi_style.format(
            valor=valor_mod,
            subtitulo="Ahorro anual (moderado)",
            delta=delta_mod
        ),
        unsafe_allow_html=True
    )
    colC.markdown(
        kpi_style.format(
            valor=valor_inv,
            subtitulo="Inversión estimada EMS",
            delta=delta_inv
        ),
        unsafe_allow_html=True
    )

    st.caption("• Ahorros calculados a partir de la reducción de consumo energético gestionada por el EMS.")

    # ========= GRÁFICA DE FLUJO ACUMULADO =========
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=meses, y=acum_opt, mode="lines+markers", name="Escenario optimista"))
    fig.add_trace(go.Scatter(x=meses, y=acum_mod, mode="lines+markers", name="Escenario moderado"))
    fig.add_hline(y=0, line_dash="dot", annotation_text="Punto de equilibrio")
    fig.update_layout(
        title="Flujo de caja acumulado EMS (24 meses)",
        xaxis_title="Mes",
        yaxis_title="USD",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========= BOTONES DE NAVEGACIÓN =========
    st.markdown("---")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("◀ Volver a EMS — Parámetros", use_container_width=True, key="ems_back_to_p3"):
            st.session_state.step = 3
            st.rerun()
    with b2:
        st.button("📤 Exportar reporte (próx.)", use_container_width=True, key="ems_export")


#FIN EMS #


# ======================= STEP 1 =======================
# ===================== E-TROLLEY: Paso 1 (parámetros) =====================

def et_ui_step1():
    """
    Paso 3 (vista de parámetros) — E-Trolley
    Guarda parámetros en st.session_state['et_params'] y pasa a step 2.
    """
    # Catálogo/valores por defecto seguros
    catalog = list(ETROLLEY_DATA["truck_catalog"].keys()) or ["Komatsu 830E"]
    if "Komatsu 830E" not in ETROLLEY_DATA["truck_catalog"]:
        ETROLLEY_DATA["truck_catalog"]["Komatsu 830E"] = {"diesel_l_km_subida": 7.0}

    c1, c2 = st.columns(2)

    # -------- Columna izquierda (c1)
    with c1:
        dist_km  = st.number_input("Distancia del tramo (km)", 0.5, 20.0, 1.7, 0.1, key="et1_dist_km")
        n_trucks = st.slider("Cantidad de camiones", 2, 12, 4, step=1, key="et1_n_trucks")
        inv_type = st.radio("Tipo de inversión", ["Compra nueva","Retrofit"], horizontal=True, key="et1_inv_type")

        # costo energía
        energy   = st.number_input(
            "Costo de energía (USD/kWh)", 0.0, 2.0,
            ETROLLEY_DATA.get("default_energy_usd_kwh", 0.11), 0.01, key="et1_energy"
        )

    # -------- Columna derecha (c2)
    with c2:
        pendiente = st.number_input("Pendiente promedio (%)", 0.0, 25.0, 8.0, 0.5, key="et1_pend")
        model     = st.selectbox("Modelo Komatsu", catalog, index=0, key="et1_model")
        conting   = st.number_input("Contingencia (%)", 0.0, 30.0, 10.0, 0.5, key="et1_conting")
        maint_y_k = st.number_input(
            "Mantenimiento trolley (kUSD/año)", 0.0, 5000.0,
            ETROLLEY_DATA.get("maintenance_line_usd_y", 120_000.0)/1000.0, 10.0, key="et1_maint_k"
        )
        maint_y   = maint_y_k * 1000.0

    # --- Ajustes avanzados (persisten en sesión) ---
    with st.expander("Ajustes avanzados del modelo (usa valores del Excel)", expanded=False):
        st.session_state["diesel_price_usd_l"] = st.number_input(
            "Precio diésel (USD/L)", 0.10, 5.00,
            float(st.session_state.get("diesel_price_usd_l", ETROLLEY_DATA.get("diesel_price_usd_l", 1.00))), 0.05
        )
        st.session_state["kwh_per_km_trolley"] = st.number_input(
            "Consumo eléctrico trolley (kWh/km)", 1.0, 80.0,
            float(st.session_state.get("kwh_per_km_trolley", ETROLLEY_DATA.get("kwh_per_km_trolley", 12.0))), 0.5
        )
        st.session_state["cycles_per_truck_per_year"] = st.number_input(
            "Ciclos por camión por año (C1)", 2000, 9000,
            int(st.session_state.get("cycles_per_truck_per_year", ETROLLEY_DATA.get("cycles_per_truck_per_year", 4200))), 100
        )
        st.session_state["subidas_por_ciclo"] = st.number_input(
            "Subidas por ciclo (ida+vuelta = 2)", 1, 4,
            int(st.session_state.get("subidas_por_ciclo", ETROLLEY_DATA.get("subidas_por_ciclo", 2))), 1
        )

        # Editor puntual de consumo en subida por modelo
        edit_model = st.selectbox("Modelo para editar consumo diésel (subida)", catalog, index=0, key="et1_edit_model")
        cur_val = ETROLLEY_DATA["truck_catalog"].get(edit_model, {}).get("diesel_l_km_subida",
                   ETROLLEY_DATA["truck_catalog"].get(edit_model, {}).get("diesel_l_km", 7.0))
        new_diesel = st.number_input("Diésel (L/km) en subida — modelo", 0.1, 100.0, float(cur_val), 0.1, key="et1_diesel_up")
        # graba override
        ETROLLEY_DATA["truck_catalog"].setdefault(edit_model, {})
        ETROLLEY_DATA["truck_catalog"][edit_model]["diesel_l_km_subida"] = float(new_diesel)

    # -------- Botones navegación --------
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("◀ Volver a priorización", key="btn_et_back", use_container_width=True):
            st.session_state.et_step = 1
            st.session_state.step = 2
            st.rerun()
    with col_btn2:
        if st.button("Calcular rentabilidad ▶", key="btn_et_calc", type="primary", use_container_width=True):
            st.session_state.et_step = 2
            st.session_state.et_params = dict(
                dist_km=dist_km, pendiente=pendiente, n_trucks=n_trucks, model=model,
                inv_type=inv_type, conting=conting, energy=energy, maint_y=maint_y
            )
            st.session_state.et_selected_model = model
            st.rerun()


def et_savings_anual_usd(dist_km, n_trucks, model, energy_usd_kwh):
    """
    Ahorro neto anual = (diesel ahorrado * precio) - (kWh consumidos * tarifa)
    * Usa km en **subida**: dist_km * ciclos * subidas_por_ciclo * n_trucks
    * diésel L/km es el valor 'en subida' del modelo seleccionado
    """
    cat = ETROLLEY_DATA["truck_catalog"][model]

    cycles  = int(ETROLLEY_DATA.get("cycles_per_truck_per_year", 5200))
    subidas = int(ETROLLEY_DATA.get("subidas_por_ciclo", 2))
    kwh_km  = float(ETROLLEY_DATA.get("kwh_per_km_trolley", 10.0))
    diesel_price = float(ETROLLEY_DATA.get("diesel_price_usd_l", 1.35))

    trips_uphill = cycles * int(n_trucks) * subidas
    km_uphill    = float(dist_km) * trips_uphill

    # Diesel y electricidad referidos a **subida**
    diesel_l_km_up = get_diesel_l_km_subida(model)
    diesel_saved_l = diesel_l_km_up * km_uphill
    elec_kwh       = kwh_km * km_uphill

    ahorro_diesel_usd = diesel_saved_l * diesel_price
    costo_elec_usd    = elec_kwh * float(energy_usd_kwh)

    return ahorro_diesel_usd - costo_elec_usd


# =======================
# Lookups como en Excel (auto-inicializados)
# =======================

# _trolleyB — emula BUSCARH($D64; _trolleyB; <fila>; FALSO)
# Números en **M$** (millones de USD), tomados de tu Excel:
#  - 13 → costo línea por km: 10.3
#   4 → subestación (unidad): 3.3
#   5 → bring power (unidad): 2.0
#   6 → conversión camión (unidad): 5.1
#   7 → mantenimiento línea por km por año: 0.8
_TROLLEYB = {
    "Komatsu 830E":   {
        "line_cost_M_per_km":       10.3,
        "ss_unit_cost_M":            3.3,
        "bring_power_unit_cost_M":   2.0,
        "truck_conv_unit_cost_M":    5.1,
        "line_maint_M_per_km_y":     0.8,
    },
    "Komatsu 860E":   {
        "line_cost_M_per_km":       10.3,
        "ss_unit_cost_M":            3.3,
        "bring_power_unit_cost_M":   2.0,
        "truck_conv_unit_cost_M":    5.1,
        "line_maint_M_per_km_y":     0.8,
    },
    "Komatsu 930E-4": {
        "line_cost_M_per_km":       10.3,
        "ss_unit_cost_M":            3.3,
        "bring_power_unit_cost_M":   2.0,
        "truck_conv_unit_cost_M":    5.1,
        "line_maint_M_per_km_y":     0.8,
    },
    "Komatsu 980E-4": {
        "line_cost_M_per_km":       10.3,
        "ss_unit_cost_M":            3.3,
        "bring_power_unit_cost_M":   2.0,
        "truck_conv_unit_cost_M":    5.1,
        "line_maint_M_per_km_y":     0.8,
    },
}

def _hlookup_trolleyB(model: str, key: str) -> float:
    """Emula BUSCARH sobre _TROLLEYB (valores en M$)."""
    try:
        return float(_TROLLEYB[model][key])
    except KeyError:
        return 0.0

# _scenX — emula BUSCARV(E64; _scenX; COINCIDIR("ss-"&$D64); …)
# Se autogenera con 2025= CAPEX; resto de años=0 (misma lógica del Excel).
def _build_default_scenx(years, model, n_trucks):
    scen = {}
    for y in years:
        scen[y] = {
            f"ss-{model}":   1 if y == years[0] else 0,      # 1 subestación en 2025
            f"bp-{model}":   0,                               # bring power en 0 (tu hoja)
            f"conv-{model}": n_trucks if y == years[0] else 0 # conversiones = #camiones en 2025
        }
    return scen

# Permite sobreescritura futura desde sesión si quisieras
_SCENX = None  # se autollenará en et_ui_step2()

def _vlookup_scenx(year: int, scen_key: str) -> float:
    """Emula BUSCARV sobre _SCENX."""
    global _SCENX
    if _SCENX is None:
        return 0.0
    try:
        return float(_SCENX[year][scen_key])
    except KeyError:
        return 0.0


def et_ui_step2():
    import math
    import pandas as pd
    import plotly.graph_objects as go

   
    # ---------- 1) Parámetros guardados ----------
    P = st.session_state.get("et_params", {})
    if not P:
        st.warning("No hay parámetros. Regresa al Paso 1.")
        if st.button("Volver", use_container_width=True, key="et2_volver_btn"):
            st.session_state.et_step = 1
        return

    # ============ BLOQUE DE RESUMEN (igual estilo que VoD) ============
    
    card_line_style = """
        background-color: #f7f8f9;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 80%;
    """

    def get_priority_color(index):
        colores = ["#fbeaea", "#fff3e0", "#fffde7"]
        return colores[index] if index < len(colores) else "#f7f8f9"

    etiquetas_prioridad = [
        "🔴 Prioridad Alta",
        "🟠 Prioridad Media",
        "🟡 Prioridad Baja"
    ]

    col_res1, col_res2 = st.columns([1, 1])

    # ---- Columna izquierda: resumen general + parámetros E-Trolley ----
    with col_res1:
        st.subheader("Resumen de información")

        tipo_mina     = st.session_state.get("tipo_mina", "-")
        tipo_material = st.session_state.get("tipo_material", "-")
        produccion    = st.session_state.get("produccion", "-")

        st.markdown(
            f"<div style='{card_line_style}'>• Tipo de mina: <b>{tipo_mina}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Material extraído: <b>{tipo_material}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Producción: <b>{produccion}</b></div>",
            unsafe_allow_html=True
        )

        # ---- Parámetros específicos de E-Trolley (de et_params) ----
        st.markdown(
            f"<div style='{card_line_style}'>• Distancia del tramo: "
            f"<b>{P.get('dist_km', 0):.2f} km</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Pendiente promedio: "
            f"<b>{P.get('pendiente', 0):.2f} %</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Cantidad de camiones: "
            f"<b>{int(P.get('n_trucks', 0))}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Modelo: "
            f"<b>{P.get('model', '-')}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Tipo de inversión: "
            f"<b>{P.get('inv_type', '-')}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Contingencia: "
            f"<b>{P.get('conting', 0)} %</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Costo de energía: "
            f"<b>{P.get('energy', 0):.3f} USD/kWh</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Mantenimiento trolley: "
            f"<b>{P.get('maint_y', 0)/1000:.1f} kUSD/año</b></div>",
            unsafe_allow_html=True
        )

    # ---- Columna derecha: desafíos seleccionados ----
    with col_res2:
        st.subheader("Desafíos seleccionados")

        prioridades = st.session_state.get("prioridades", [])
        for idx, d in enumerate(prioridades[:3]):
            bg_color = get_priority_color(idx)
            etiqueta = (
                etiquetas_prioridad[idx]
                if idx < len(etiquetas_prioridad)
                else f"Prioridad {idx+1}"
            )
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    padding: 10px 15px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    font-size: 80%;
                ">
                    <b>{etiqueta}:</b> {d}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")  # separador visual antes de los KPIs y gráficos
    st.markdown("### Resultados proyectados a largo plazo")

    # ============ FIN BLOQUE RESUMEN ============


    # ================== A PARTIR DE AQUÍ, LO QUE YA TENÍAS ==================

    # ---------- 1) Parámetros guardados ----------
    P = st.session_state.get("et_params", {})
    if not P:
        st.warning("No hay parámetros. Regresa al Paso 1.")
        if st.button("Volver", use_container_width=True, key="et2_volver_btn"):
            st.session_state.et_step = 1
        return

    dist_km        = float(P.get("dist_km", 0.0))
    n_trucks       = int(P.get("n_trucks", 0))
    model          = str(P.get("model") or list(ETROLLEY_DATA["truck_catalog"].keys())[0])
    inv_type       = str(P.get("inv_type", "Compra nueva"))
    conting_pct    = float(P.get("conting", 0.0))
    energy_usd_kwh = float(P.get("energy", ETROLLEY_DATA.get("default_energy_usd_kwh", 0.11)))
    maint_line_y   = float(P.get("maint_y", ETROLLEY_DATA.get("maintenance_line_usd_y", 120_000.0)))

    # ---------- 2) Avanzados ----------
    diesel_price_usd_l  = float(st.session_state.get("diesel_price_usd_l", ETROLLEY_DATA.get("diesel_price_usd_l", 1.00)))
    kwh_per_km_trolley  = float(st.session_state.get("kwh_per_km_trolley", ETROLLEY_DATA.get("kwh_per_km_trolley", 12.0)))
    cycles_c1_per_truck = int(st.session_state.get("cycles_per_truck_per_year", ETROLLEY_DATA.get("cycles_per_truck_per_year", 4200)))
    cycles_c2_per_truck = int(st.session_state.get("cycles_per_truck_per_year_c2", int(ETROLLEY_DATA.get("cycles_per_truck_per_year", 4200)*1.7)))
    subidas_por_ciclo   = int(st.session_state.get("subidas_por_ciclo", ETROLLEY_DATA.get("subidas_por_ciclo", 2)))

    # Override opcional de consumo en subida
    ss_key = f"diesel_subida__{model}"
    ETROLLEY_DATA["truck_catalog"].setdefault(model, {})
    if ss_key in st.session_state:
        ETROLLEY_DATA["truck_catalog"][model]["diesel_l_km_subida"] = float(st.session_state[ss_key])
    diesel_l_km_subida = get_diesel_l_km_subida(model)

    # ---------- 3) Horizonte y años ----------
    tasa_desc_anual = ETROLLEY_DATA.get("tasa_desc_npvt", 0.08)
    start_year      = 2025
    horizon_years   = int(ETROLLEY_DATA.get("horiz_anios", 10))
    years           = list(range(start_year, start_year + horizon_years))

    # ---------- Helpers de formato ----------
    def fmt_short(x: float, decimals: int = 1) -> str:
        x = float(x); a = abs(x)
        if a >= 1e9: return f"{x/1e9:.{decimals}f} B"
        if a >= 1e6: return f"{x/1e6:.{decimals}f} M"
        if a >= 1e3: return f"{x/1e3:.0f} K"
        return f"{x:,.0f}"

    def nice_max(v):
        if v <= 0: return 1
        exp = 10 ** int(math.floor(math.log10(v)))
        for m in [1, 2, 5, 10]:
            if v <= m * exp:
                return m * exp
        return 10 * exp

    # ---------- 4) (Cálculo operativo básico) ----------
    trips_uphill_c1_y = cycles_c1_per_truck * n_trucks * subidas_por_ciclo
    trips_uphill_c2_y = cycles_c2_per_truck * n_trucks * subidas_por_ciclo
    km_uphill_c1_y    = dist_km * trips_uphill_c1_y
    km_uphill_c2_y    = dist_km * trips_uphill_c2_y

    diesel_l_y_c1  = diesel_l_km_subida * km_uphill_c1_y
    costo_diesel_y = diesel_l_y_c1 * diesel_price_usd_l

    elec_kwh_y_c2   = kwh_per_km_trolley * km_uphill_c2_y
    costo_elec_y    = elec_kwh_y_c2 * energy_usd_kwh
    opex_trolley_y  = maint_line_y
    opex_c2_total_y = costo_elec_y + opex_trolley_y
    ahorro_operativo_y = max(0.0, costo_diesel_y - opex_c2_total_y)

    # ---------- 5) CAPEX estilo Excel (año 1) ----------
    def _hlookup_trolleyB(model_name, key):
        cost_per_km_M     = ETROLLEY_DATA.get("cost_per_km_usd", 2_200_000.0) / 1_000_000.0
        line_maint_M_y    = ETROLLEY_DATA.get("maintenance_line_usd_y", 120_000.0) / 1_000_000.0
        retrofit_cost_M   = ETROLLEY_DATA.get("retrofit_cost_usd", {}).get(model_name, 1_100_000) / 1_000_000.0
        mapping = {
            "line_cost_M_per_km":       cost_per_km_M,
            "ss_unit_cost_M":          0.8,
            "bring_power_unit_cost_M": 0.6,
            "truck_conv_unit_cost_M":  retrofit_cost_M,
            "line_maint_M_per_km_y":   line_maint_M_y / max(dist_km, 1.0),
        }
        return float(mapping.get(key, 0.0))

    conting_f = 1.0 + conting_pct/100.0
    cap_line_M  = _hlookup_trolleyB(model, "line_cost_M_per_km") * dist_km * conting_f
    cap_ss_M    = _hlookup_trolleyB(model, "ss_unit_cost_M") * 1 * conting_f
    cap_bp_M    = _hlookup_trolleyB(model, "bring_power_unit_cost_M") * 1 * conting_f
    cap_conv_M  = _hlookup_trolleyB(model, "truck_conv_unit_cost_M") * n_trucks * conting_f
    capex_y1_M  = cap_line_M + cap_ss_M + cap_bp_M + cap_conv_M

    # ---------- 6) KPIs con estilo ABB ----------
    def _fmt_short(x: float, decimals: int = 1) -> str:
        x = float(x); a = abs(x)
        if a >= 1e9: return f"{x/1e9:.{decimals}f} B"
        if a >= 1e6: return f"{x/1e6:.{decimals}f} M"
        if a >= 1e3: return f"{x/1e3:.0f} K"
        return f"{x:,.0f}"

    # toneladas / CO2 (por si aún no están)
    cap_t = ETROLLEY_DATA["truck_catalog"].get(model, {}).get("capacidad_t", 230)
    ton_c1_y = trips_uphill_c1_y * cap_t / 1_000_000.0
    ton_c2_y = trips_uphill_c2_y * cap_t / 1_000_000.0
    cum_ton_c1, cum_ton_c2 = [], []
    acc1 = acc2 = 0.0
    for _ in years:
        acc1 += ton_c1_y
        acc2 += ton_c2_y
        cum_ton_c1.append(acc1)
        cum_ton_c2.append(acc2)

    ton_c1_total_M = float(cum_ton_c1[-1]) if cum_ton_c1 else 0.0
    ton_c2_total_M = float(cum_ton_c2[-1]) if cum_ton_c2 else 0.0
    delta_ton_pct  = ((ton_c2_total_M - ton_c1_total_M) / max(ton_c1_total_M, 1e-9)) * 100.0

    diesel_kg_per_l = float(st.session_state.get("et_diesel_kg_l", 2.68))
    grid_kg_per_kwh = float(st.session_state.get("et_grid_kg_kwh", 0.20))
    co2_c1_y_ton = (diesel_l_y_c1 * diesel_kg_per_l) / 1000.0
    co2_c2_y_ton = (elec_kwh_y_c2 * grid_kg_per_kwh) / 1000.0
    co2_saved_y_ton = max(0.0, co2_c1_y_ton - co2_c2_y_ton)
    cum_co2_saved_kt, _acc = [], 0.0
    for _ in years:
        _acc += co2_saved_y_ton
        cum_co2_saved_kt.append(_acc / 1000.0)
    co2_total_kt = float(cum_co2_saved_kt[-1]) if cum_co2_saved_kt else 0.0

    ahorro_operativo_y = max(0.0, costo_diesel_y - opex_c2_total_y)

    toneladas = f"{ton_c2_total_M:,.0f} MTM"
    emisiones = f"{co2_total_kt:.1f} KtCO₂"
    ahorro    = f"USD {_fmt_short(ahorro_operativo_y,1)}"
    capex     = f"USD {_fmt_short(capex_y1_M*1_000_000,1)}"
    delta_ton = f"{delta_ton_pct:+.0f}% vs C1"

    col1, col2, col3, col4 = st.columns(4)
    kpi_style = """
        <div style="
            background-color:#f5f5f5;
            border-radius:12px;
            padding:18px 16px;
            text-align:center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        ">
            <h2 style="margin:0; font-weight:700; color:#000;">{valor}</h2>
            <p style="margin:0; color:#555; font-size:15px;">{subtitulo}</p>
            <p style="margin-top:5px; color:#008000; font-size:13px;">{delta}</p>
        </div>
    """
    col1.markdown(kpi_style.format(valor=toneladas, subtitulo="Toneladas movidas (10 años)", delta=delta_ton), unsafe_allow_html=True)
    col2.markdown(kpi_style.format(valor=emisiones, subtitulo="Emisiones evitadas (10 años)", delta=""), unsafe_allow_html=True)
    col3.markdown(kpi_style.format(valor=ahorro, subtitulo="Ahorro operativo anual", delta=""), unsafe_allow_html=True)
    
    st.caption("• MTM = millones de toneladas-movidas · KtCO₂ = miles de toneladas de CO₂ · Ahorro: diésel vs (eléctrico + mantenimiento).")

    # ⚠️ AQUÍ YA NO HAY ANIMACIÓN DE CAMIONES ⚠️

    # ---------- 7) Toneladas acumuladas (gráfico) ----------
    ymax_ton = nice_max(max(cum_ton_c1 + cum_ton_c2))

    fig_tm = go.Figure()
    fig_tm.add_trace(go.Scatter(
        x=years, y=cum_ton_c1, mode="lines+markers", name="C1 (diésel)",
        line=dict(width=3), marker=dict(size=6), fill="tozeroy", opacity=0.25,
        hovertemplate="Año %{x}<br>%{y:.0f} MTM<extra></extra>",
    ))
    fig_tm.add_trace(go.Scatter(
        x=years, y=cum_ton_c2, mode="lines+markers", name="C2 (e-trolley)",
        line=dict(width=3), marker=dict(size=6), fill="tozeroy", opacity=0.25,
        hovertemplate="Año %{x}<br>%{y:.0f} MTM<extra></extra>",
    ))
    fig_tm.update_yaxes(range=[0, ymax_ton], dtick=ymax_ton/7.0, title="MTM")
    fig_tm.update_xaxes(title="Año")
    fig_tm.update_layout(title="Toneladas acumuladas movidas(M TM)", margin=dict(l=10,r=10,t=60,b=40))
    fig_tm.add_annotation(x=years[-1], y=cum_ton_c1[-1], text=f"{cum_ton_c1[-1]:.0f} MTM",
                          showarrow=True, arrowhead=2, ax=30, ay=-20)
    fig_tm.add_annotation(x=years[-1], y=cum_ton_c2[-1], text=f"{cum_ton_c2[-1]:.0f} MTM",
                          showarrow=True, arrowhead=2, ax=30, ay=-20)

    # ---------- 8) CO₂ acumulado evitado ----------
    with st.expander("Ajustes de CO₂ (para la gráfica)"):
        diesel_kg_per_l = st.number_input("Factor diésel (kg CO₂ / L)", 0.0, 5.0,
                                          float(st.session_state.get("et_diesel_kg_l", 2.68)), 0.01,
                                          key="et_diesel_kg_l")
        grid_kg_per_kwh = st.number_input("Factor red (kg CO₂ / kWh)", 0.0, 2.0,
                                          float(st.session_state.get("et_grid_kg_kwh", 0.20)), 0.01,
                                          key="et_grid_kg_kwh")

    co2_c1_y_ton = (diesel_l_y_c1 * diesel_kg_per_l) / 1000.0
    co2_c2_y_ton = (elec_kwh_y_c2 * grid_kg_per_kwh) / 1000.0
    co2_saved_y_ton = max(0.0, co2_c1_y_ton - co2_c2_y_ton)

    cum_co2_saved_kt, acc = [], 0.0
    for _ in years:
        acc += co2_saved_y_ton
        cum_co2_saved_kt.append(acc / 1000.0)

    ymax_co2 = nice_max(max(cum_co2_saved_kt))

    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(
        x=years, y=cum_co2_saved_kt, mode="lines+markers", name="C2 vs C1",
        line=dict(width=3), marker=dict(size=6), fill="tozeroy", opacity=0.25,
        hovertemplate="Año %{x}<br>%{y:.1f} KtCO₂<extra></extra>",
    ))
    fig_co2.update_yaxes(range=[0, ymax_co2], dtick=max(ymax_co2/7.0, 0.5), title="K tCO₂")
    fig_co2.update_xaxes(title="Año")
    fig_co2.update_layout(title="Ahorro acumulado de CO₂(K tCO₂)", margin=dict(l=10,r=10,t=60,b=40))
    fig_co2.add_annotation(x=years[-1], y=cum_co2_saved_kt[-1],
                           text=f"{cum_co2_saved_kt[-1]:.1f} KtCO₂",
                           showarrow=True, arrowhead=2, ax=30, ay=-20)

    g1, g2 = st.columns(2)
    g1.plotly_chart(fig_tm,  use_container_width=True, key="et_tons_y_only")
    g2.plotly_chart(fig_co2, use_container_width=True, key="et_co2_saved_only")

    with st.expander("🔎 Diagnóstico (valores clave)"):
        df_dbg = pd.DataFrame({
            "Año": years,
            "Ton C1 (acum MTM)": cum_ton_c1,
            "Ton C2 (acum MTM)": cum_ton_c2,
            "CO2 saved (acum Kt)": cum_co2_saved_kt,
        })
        st.dataframe(df_dbg, use_container_width=True)

    st.markdown("---")
    if st.button("⬅ Volver a parámetros de E-Trolley", key="et_back_step1_y", use_container_width=True):
        st.session_state.et_step = 1
        st.rerun()


# ===================== APC (Advanced Process Control) =====================

def apc_step1():
    st.subheader("APC — Paso 1: Datos de proceso y económicos")

    # Valores por defecto tomados del Excel
    throughput = st.number_input(
        "Throughput (tph):",
        min_value=1.0,
        value=st.session_state.get("apc_throughput", 4000.0),
        step=10.0,
        key="apc_throughput_in"
    )

    availability_pct = st.slider(
        "Disponibilidad operativa (%):",
        min_value=50, max_value=100,
        value=st.session_state.get("apc_availability_pct", 90),
        format="%d%%",
        key="apc_availability_in"
    )

    power_mw = st.number_input(
        "Potencia del sistema (MW):",
        min_value=0.1,
        value=st.session_state.get("apc_power_mw", 28.0),
        step=0.1,
        key="apc_power_in"
    )

    energy_cost = st.number_input(
        "Costo de energía (USD/kWh):",
        min_value=0.001,
        value=st.session_state.get("apc_energy_cost", 0.06),
        step=0.001,
        format="%.3f",
        key="apc_energy_cost_in"
    )

    net_worth = st.number_input(
        "Net worth per ton (USD/t):",
        min_value=1.0,
        value=st.session_state.get("apc_net_worth", 22.0),
        step=1.0,
        key="apc_net_worth_in"
    )

    st.markdown("### 📗 Datos económicos adicionales")

    # ============================
    # Costo mensual de reactivos
    # ============================
    reagent_cost_m_musd = st.number_input(
        "Costo mensual de reactivos (MUSD/mes):",
        min_value=0.0,
        value=st.session_state.get("apc_reagent_cost_m_musd", 0.12),  # 0.12 MUSD = 120,000 USD
        step=0.01,
        format="%.3f",
        key="apc_reagent_in"
    )

    # Convertir a USD/mes internamente
    reagent_cost_m = reagent_cost_m_musd * 1_000_000
    st.session_state.apc_reagent_cost_m = reagent_cost_m
    st.session_state.apc_reagent_cost_m_musd = reagent_cost_m_musd


    # ============================
    # Valor económico del 1% de recuperación
    # ============================
    recovery_value_musd = st.number_input(
        "Valor económico del 1% de recuperación (MUSD/año):",
        min_value=0.0,
        value=st.session_state.get("apc_recovery_value_musd", 10.0),  # 10 MUSD = 10M USD
        step=0.5,
        format="%.2f",
        key="apc_recovery_in"
    )

    # Convertir a USD/año internamente
    recovery_value = recovery_value_musd * 1_000_000
    st.session_state.apc_recovery_value = recovery_value
    st.session_state.apc_recovery_value_musd = recovery_value_musd


    # Guardar en sesión (normalizado)
    st.session_state.apc_throughput = float(throughput)
    st.session_state.apc_availability_pct = int(availability_pct)
    st.session_state.apc_availability = float(availability_pct) / 100.0
    st.session_state.apc_power_mw = float(power_mw)
    st.session_state.apc_energy_cost = float(energy_cost)
    st.session_state.apc_net_worth = float(net_worth)
    st.session_state.apc_reagent_cost_m = float(reagent_cost_m)
    st.session_state.apc_recovery_value = float(recovery_value)

    col_a, col_b = st.columns(2)

    # Volver a Paso 2 (desafíos) / cerrar APC
    with col_a:
        if st.button("❌ Cerrar APC", key="apc_close_1", use_container_width=True):
            st.session_state.apc_mode = False
            st.session_state.step = 2
            st.rerun()

    # Ir a Paso 4 (Beneficios APC)
    with col_b:
        if st.button("Calcular beneficios APC ▶", key="apc_go_step2", type="primary", use_container_width=True):
            st.session_state.apc_mode = True   # por seguridad
            st.session_state.step = 4
            st.rerun()



def apc_step2():
    # ⚠️ Ya NO ponemos "APC — Paso 2" aquí, porque el título principal
    # lo define el Paso 4:
    # st.header("Paso 4 de 4: Advanced Process Control (APC) — Beneficios estimados")

    # ========= 1) Recuperar parámetros necesarios =========
    if "apc_throughput" not in st.session_state:
        st.warning("Primero completa los datos de proceso y económicos (Paso 1 de APC).")
        st.session_state.apc_step = 1
        return

    # Datos generales del Paso 1 global
    tipo_mina     = st.session_state.get("tipo_mina", "—")
    tipo_material = st.session_state.get("tipo_material", "—")
    produccion    = st.session_state.get("produccion", "—")

    # Parámetros APC del Paso 1 APC
    tp          = float(st.session_state.apc_throughput)
    avail_pct   = float(st.session_state.apc_availability_pct)
    avail       = float(st.session_state.apc_availability)
    p_mw        = float(st.session_state.apc_power_mw)
    c_kwh       = float(st.session_state.apc_energy_cost)
    net_worth   = float(st.session_state.apc_net_worth)
    reagent_m   = float(st.session_state.apc_reagent_cost_m)
    rec_1pct_val = float(st.session_state.apc_recovery_value)

    # Valores en MUSD para mostrarlos bonitos
    reagent_m_musd = float(st.session_state.get("apc_reagent_cost_m_musd", reagent_m/1_000_000))
    rec_1pct_musd  = float(st.session_state.get("apc_recovery_value_musd", rec_1pct_val/1_000_000))

    # ========= 2) Bloque de RESUMEN (similar a E-Trolley) =========
    card_line_style = """
        background-color: #f7f8f9;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 80%;
    """

    def get_priority_color(index):
        colores = ["#fbeaea", "#fff3e0", "#fffde7"]
        return colores[index] if index < len(colores) else "#f7f8f9"

    etiquetas_prioridad = [
        "🔴 Prioridad Alta",
        "🟠 Prioridad Media",
        "🟡 Prioridad Baja"
    ]

    col_res1, col_res2 = st.columns([1, 1])

    # ----- Columna izquierda: Resumen de información -----
    with col_res1:
        st.subheader("Resumen de información")

        st.markdown(
            f"<div style='{card_line_style}'>• Tipo de mina: "
            f"<b>{tipo_mina}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Material extraído: "
            f"<b>{tipo_material}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Producción: "
            f"<b>{produccion}</b></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='{card_line_style}'>• Throughput de diseño: "
            f"<b>{tp:,.0f} tph</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Disponibilidad operativa: "
            f"<b>{avail_pct:.0f} %</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Potencia del sistema: "
            f"<b>{p_mw:.1f} MW</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Costo de energía: "
            f"<b>{c_kwh:.3f} USD/kWh</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Net worth del mineral: "
            f"<b>{net_worth:,.0f} USD/t</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Costo mensual de reactivos: "
            f"<b>{reagent_m_musd:.3f} MUSD/mes</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Valor del 1% de recuperación: "
            f"<b>{rec_1pct_musd:.1f} MUSD/año</b></div>",
            unsafe_allow_html=True
        )

    # ----- Columna derecha: Desafíos seleccionados -----
    with col_res2:
        st.subheader("Desafíos seleccionados")

        prioridades = st.session_state.get("prioridades", [])
        for idx, d in enumerate(prioridades[:3]):
            bg_color = get_priority_color(idx)
            etiqueta = (
                etiquetas_prioridad[idx]
                if idx < len(etiquetas_prioridad)
                else f"Prioridad {idx+1}"
            )
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    padding: 10px 15px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    font-size: 80%;
                ">
                    <b>{etiqueta}:</b> {d}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")  # separador antes de los KPIs

    # ========= 3) Cálculos APC (igual que antes) =========
    horas_anuales = 8760.0 * avail
    tons_anuales = tp * horas_anuales
    energia_mwh = p_mw * horas_anuales
    costo_energia_anual = energia_mwh * 1000.0 * c_kwh  # (lo puedes usar luego si quieres)

    # Suposiciones básicas:
    costo_reactivos_anual = reagent_m * 12.0
    ahorro_reactivos = costo_reactivos_anual * 0.05          # 5% ahorro
    beneficio_recuperacion = rec_1pct_val                    # 1% recuperación
    beneficio_total = beneficio_recuperacion + ahorro_reactivos

    # ========= 4) KPIs (lo que ya tenías) =========
    card_style = """
        background-color: #f7f8f9;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    """

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Indicadores de operación")
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:24px; font-weight:bold;">{tons_anuales/1e6:,.2f} M ton/año</div>
                <div style="font-size:14px; color:#555;">Throughput anual</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:24px; font-weight:bold;">{energia_mwh/1000:,.2f} GWh/año</div>
                <div style="font-size:14px; color:#555;">Consumo energético estimado</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Indicadores económicos APC")
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:24px; font-weight:bold;">MUSD {beneficio_recuperacion/1_000_000:,.0f}</div>
                <div style="font-size:14px; color:#555;">Beneficio por +1% recuperación</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:24px; font-weight:bold;">KUSD {ahorro_reactivos/1000:,.0f}</div>
                <div style="font-size:14px; color:#555;">Ahorro anual estimado en reactivos (5%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="font-size:24px; font-weight:bold;">MUSD {beneficio_total/1_000_000:,.0f}</div>
                <div style="font-size:14px; color:#555;">Beneficio económico total estimado (sin CAPEX)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========= 5) Botones de navegación =========
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("◀ Volver a editar datos APC", key="apc_back_step1", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    with col_b:
        if st.button("❌ Cerrar APC", key="apc_close_2", use_container_width=True):
            st.session_state.apc_mode = False
            st.session_state.step = 2
            st.rerun()







# Inicializar estados
if "step" not in st.session_state:
    st.session_state.step = 1
if "prioridades" not in st.session_state:
    st.session_state.prioridades = []





# Paso 1 de 4
if st.session_state.step == 1:
    st.header("Paso 1 de 4: Información General")

    st.session_state.tipo_mina = st.selectbox("Seleccione el tipo de mina:", ["Subterránea", "Tajo abierto"])
    st.session_state.tipo_material = st.selectbox("Seleccione el tipo de material a extraer:", ["Polimetálico", "Fosfatos", "Cobre", "Hierro"])
    st.session_state.produccion = st.selectbox("Tonelada diaria procesada:", [
        "Menor a 5,000 ton", 
        "5,000 a 60,000 ton", 
        "Mayor a 60,000 ton"
    ])

    #col1, col2 = st.columns([1, 1])
    #with col1:
     #   st.button("Siguiente", on_click=next_step)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Siguiente ▶", on_click=next_step, key="next1", type="primary")
    with col2:
        st.empty()








# Paso 2 de 4
if st.session_state.step == 2:
    st.header("Paso 2 de 4: Desafíos Prioritarios")
    st.caption("(Selecciona hasta tres desafíos)")

    if "mostrar_warning" not in st.session_state:
        st.session_state.mostrar_warning = False

    # Mostrar advertencia en una fila superior con columnas
    if st.session_state.mostrar_warning:
        colw1, colw2 = st.columns([0.01, 0.9])  # 10% - 90% ancho pantalla
        with colw2:
            st.warning("Debes seleccionar al menos un desafío.")
        st.session_state.mostrar_warning = False  # Ocultar luego del render

    desafios = [
        "Aumentar la recuperación",
        "Aumentar los índices de seguridad",
        "Aumentar rendimiento de equipos",
        "Centralizar la visibilidad de la operación",
        "Reducir consumo energético",
        "Reducir consumo de combustible fósil",
        "Reducir la huella hídrica"       
    ]

    for i, desafio in enumerate(desafios):
        if st.checkbox(desafio, key=f"desafio_{i}"):
            if desafio not in st.session_state.prioridades:
                st.session_state.prioridades.append(desafio)
        else:
            if desafio in st.session_state.prioridades:
                st.session_state.prioridades.remove(desafio)

    if st.session_state.prioridades:
        st.markdown("<hr>", unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state.prioridades):
        color = ["🔴 Prioridad Alta", "🟠 Prioridad Media", "🟡 Prioridad Baja"]
        if idx < 3:
            st.markdown(
                f"<span style='margin-left:10px;'>{item}: <b style='color:red'>{color[idx]}</b></span>",
                unsafe_allow_html=True
            )


  
    # --- Navegación ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.button("◀ Anterior", on_click=prev_step, key="prev2", type="secondary")

    with col2:
        if st.button("Siguiente ▶", key="next2", type="primary"):
            if len(st.session_state.prioridades) == 0:
                st.session_state.mostrar_warning = True
                st.rerun()
            else:
                next_step()
                st.rerun()


   


# ---------------- Config (reemplaza TRANSICION_A_CASO) ----------------
NIVELES_ORDEN = ["Esencial", "Básico", "Operación Efectiva", "Smart"]
CASO_CONSECUTIVO = {
    ("Esencial", "Básico"): 1,
    ("Básico", "Operación Efectiva"): 2,
    ("Operación Efectiva", "Smart"): 3,
}

CASO_A_TRANSICION = {
    1: ("Esencial", "Básico"),
    2: ("Básico", "Operación Efectiva"),
    3: ("Operación Efectiva", "Smart"),
    4: ("Esencial", "Smart"),  # por si luego lo usas
}

def ruta_secuencial(n_act: str, n_obj: str):
    i_act = NIVELES_ORDEN.index(n_act)
    i_obj = NIVELES_ORDEN.index(n_obj)
    if i_act == i_obj:
        return []              # mismo nivel
    if i_act > i_obj:
        return None            # downgrade no permitido
    return NIVELES_ORDEN[i_act:i_obj+1]

def casos_desde_ruta(ruta: list[str]):
    return [CASO_CONSECUTIVO[(a, b)] for a, b in zip(ruta, ruta[1:])]


    
# Paso 3 de 4
# Paso 3 de 4
if st.session_state.step == 3:
    import unicodedata

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", (s or "")).encode("ascii","ignore").decode().strip().lower()

    prioridades = st.session_state.get("prioridades", [])

    # Por defecto, NO estamos en modo APC
    st.session_state.apc_mode = False

    # ---------- 1) COMBO APC (P1,P2,P3) ----------
    apc_auto = (
        len(prioridades) >= 3 and
        _norm(prioridades[0]) == _norm("Aumentar la recuperación") and
        _norm(prioridades[1]) == _norm("Aumentar rendimiento de equipos") and
        _norm(prioridades[2]) == _norm("Reducir consumo energético")
    )

    if apc_auto:
        st.session_state.apc_mode = True
        st.header("Paso 3 de 4: Advanced Process Control (APC) — Datos de proceso y económicos")
        apc_step1()          # 👈 mostramos directamente el Paso 1 de APC
        st.stop()            # 👈 no seguimos a EMS / VoD / Trolley

    # ---------- 2) SI NO ES APC, seguimos con la lógica existente (EMS / Trolley / VoD) ----------
    priority1_exact = prioridades[0] if len(prioridades) >= 1 else None
    priority2_exact = prioridades[1] if len(prioridades) >= 2 else None

    # 🔌 Hook E-Trolley: mina a tajo abierto + prioridad en combustible fósil
    if _norm(priority2_exact) == _norm("Reducir consumo de combustible fósil") and \
       st.session_state.get("tipo_mina", "") == "Tajo abierto":

        # Inicializa sub-pasos de E-Trolley si es la primera vez
        if "et_step" not in st.session_state:
            st.session_state.et_step = 1

        # 🔌 Hook E-Trolley: mina a tajo abierto + prioridad en combustible fósil
    if _norm(priority2_exact) == _norm("Reducir consumo de combustible fósil") and \
       st.session_state.get("tipo_mina", "") == "Tajo abierto":

        # Inicializa sub-pasos de E-Trolley si es la primera vez
        if "et_step" not in st.session_state:
            st.session_state.et_step = 1

        # 👉 Título distinto según sub-paso
        if st.session_state.et_step == 1:
            st.header("Paso 3 de 4: E-Trolley — Parámetros del caso")
            et_ui_step1()
        else:
            st.header("Paso 4 de 4: E-Trolley — Resultados del análisis")
            et_ui_step2()

        st.stop()




    # Hook EMS si P1=centralizar visibilidad y P2=reducción energética
    if (priority1_exact and priority2_exact and
        _norm(priority1_exact) == _norm("Centralizar la visibilidad de la operación") and
        _norm(priority2_exact) == _norm("Reducir consumo energético")):
        st.session_state.ems_active = True
        ems_ui_step3()
        st.stop()

    # Si no estamos en EMS ni E-Trolley, forzamos VoD (como ya tenías)
    st.session_state.ems_active = False
    st.session_state.pop("ems_params", None)
    st.header("Paso 3 de 4: Nivel actual y objetivo")
    ...



    col1, col2 = st.columns(2)
    with col1:
        n_act = st.selectbox("Nivel actual", NIVELES_ORDEN, index=0, key="nivel_actual")
    with col2:
        n_obj = st.selectbox("Nivel objetivo", NIVELES_ORDEN, index=1, key="nivel_objetivo")

    ruta = ruta_secuencial(n_act, n_obj)
    siguiente_habilitado = True

    # Apaga Caso 4 si ya no aplica
    if st.session_state.get("usar_caso_4") and not (n_act == "Esencial" and n_obj == "Smart"):
        st.session_state.usar_caso_4 = False

    if ruta is None:
        st.warning("No se permite una transición descendente. Selecciona un nivel objetivo igual o superior.")
        siguiente_habilitado = False
    elif len(ruta) == 0:
        st.info("El nivel actual y el objetivo son iguales. Selecciona una transición distinta.")
        siguiente_habilitado = False
    else:
        casos = casos_desde_ruta(ruta)
        st.session_state.ruta = ruta
        st.session_state.casos = casos
        st.session_state.caso = casos[0]  # compat

        usar_c4 = (n_act == "Esencial" and n_obj == "Smart" and
                   st.toggle("Usar implementación conjunta (Caso 4)", key="usar_caso_4",
                             value=st.session_state.get("usar_caso_4", False)))
        if usar_c4:
            st.info("Has seleccionado **Caso 4** (CAPEX específico; no suma lineal de 1+2+3).")
            st.session_state.casos = [4]
            st.session_state.caso = 4
        else:
            st.success(f"Transición detectada: {' → '.join(ruta)}")
            st.caption(f"Casos a ejecutar en orden: {', '.join(map(str, casos))}. (1→2→3).")

        st.info("""
        **Esencial** → Operación básica y segura, mínima automatización.  
        **Básico** → Controles simples (PLC), mayor estabilidad.  
        **Operación Efectiva** → Integración en DCS, eficiencia energética y operativa.  
        **Smart** → Nivel avanzado con IA y analítica para optimización total.
        """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◀ Anterior", on_click=prev_step, key="prev3", type="secondary")
    with col2:
        clicked = st.button("Siguiente ▶", key="next3", type="primary", disabled=not siguiente_habilitado)
        if clicked:
            next_step()
            st.rerun()




# Paso 4 de 4
if st.session_state.step == 4:

    # ---------- 1) Si estamos en modo APC, mostrar Beneficios APC ----------
    if st.session_state.get("apc_mode"):
        st.header("Paso 4 de 4: Advanced Process Control (APC) — Beneficios estimados")
        apc_step2()
        st.stop()

    # ---------- 2) Si el flujo activo es EMS, mostrar cálculo EMS ----------
    if st.session_state.get("ems_active"):
        ems_ui_step4()
        st.stop()


    st.header("Paso 4 de 4: Parámetros técnicos")

    if "mostrar_warning_tec" not in st.session_state:
        st.session_state.mostrar_warning_tec = False

    # Mostrar advertencia en parte superior
    if st.session_state.mostrar_warning_tec:
        colw1, colw2 = st.columns([0.01, 0.9])
        with colw2:
            st.warning("Completa los campos técnicos iniciales.")
        st.session_state.mostrar_warning_tec = False  # Se limpia tras mostrarla

    # --- Parámetros técnicos siempre visibles ---
    ventiladores_prim = st.number_input("Cantidad de ventiladores primarios:", min_value=0, max_value=20, step=1,
                                        value=st.session_state.get("ventiladores_prim", 0))
    potencia_prim = st.number_input("Potencia promedio por ventilador primario(kW):",
                                    value=st.session_state.get("potencia_prim", 75.0))
    ventiladores_comp = st.number_input("Cantidad de ventiladores complementarios:", min_value=0, max_value=20, step=1,
                                        value=st.session_state.get("ventiladores_comp", 0))
    potencia_comp = st.number_input("Potencia promedio por ventilador complementario(kW):",
                                    value=st.session_state.get("potencia_comp", 1.5))

    tarifa = st.selectbox("Costo de energía (USD/kWh):", [
        "Mayor a 0.1",
        "Entre 0.076 a 0.1",
        "Entre 0.05 a 0.075",
        "Menor a 0.05"
    ], index=["Mayor a 0.1","Entre 0.076 a 0.1","Entre 0.05 a 0.075","Menor a 0.05"].index(
        st.session_state.get("tarifa","Mayor a 0.1"))
    )

    # --- Parámetros según caso ---
    st.markdown("---")
    st.subheader("Parámetros según caso")

    _caso = st.session_state.get("caso", None)

    # Defaults
    _red_c1 = st.session_state.get("reduccion_c1_pct", 0.08)      # 8 %
    _esquema = st.session_state.get("esquema_id", 2)              # 1..3
    _red_baja = st.session_state.get("reduccion_baja_pct", 0.25)  # 25 %
    _q_rel = st.session_state.get("q_rel_smart", 0.85)            # 85 %

    if _caso == 1:
        st.markdown("**Caso 1 • Reducción directa de velocidad**")
        _red_c1 = st.slider(
            "Reducción de velocidad inicial (%)",
            min_value=1.0, max_value=10.0, value=_red_c1*100, step=0.5
        ) / 100.0

    elif _caso == 2:
        # ---- Título y explicación dinámica ----
        st.markdown("**Caso 2 • Configurar: Horas en operación del ventilador**")

        CASO_A_TRANSICION = {
            1: ("Esencial", "Básico"),
            2: ("Básico", "Operación Efectiva"),
            3: ("Operación Efectiva", "Smart"),
            4: ("Esencial", "Smart"),
        }
        nivel_desde, nivel_hasta = CASO_A_TRANSICION.get(2, ("Esencial", "Básico"))

        st.caption(
            f"A partir del nivel de automatización **{nivel_desde} → {nivel_hasta}** "
            f"se usa una reducción de **{_red_c1*100:.0f}%** en la operación.  \n"
            "Selecciona cuál opción representa las **horas en operación del ventilador**."
        )

        # ====== estilos de las tarjetas / barras ======
        st.markdown("""
            <style>
            .scheme-card {
                border: 2px solid #e5e7eb;
                border-radius: 14px;
                padding: 12px 14px;
                background: #ffffff;
                transition: box-shadow .2s ease, border-color .2s ease;
            }
            .scheme-card.selected {
                border-color: #2563eb;
                box-shadow: 0 4px 18px rgba(37,99,235,.15);
            }
            .scheme-title {
                font-weight: 700;
                text-align: center;
                margin-bottom: 8px;
            }
            .bar-wrap {
                width: 100%;
                height: 28px;
                background: #eef2f7;
                border-radius: 999px;
                overflow: hidden;
                display: flex;
            }
            .bar-alta {
                background: #335b89;
                height: 100%;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#fff;
                font-size:12px;
            }
            .bar-baja {
                background: #dfe7f2;
                height: 100%;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#1f2937;
                font-size:12px;
            }
            .labels {
                display:flex;
                justify-content: space-between;
                margin-top: 6px;
                font-size: 12px;
                color:#4b5563;
            }
            </style>
        """, unsafe_allow_html=True)

        def scheme_card_html(num:int, selected:bool, alta_frac:float, baja_frac:float) -> str:
            alta_pct = round(alta_frac*100)
            baja_pct = 100 - alta_pct
            return f"""
                <div class="scheme-card {'selected' if selected else ''}">
                    <div class="scheme-title">Esquema {num}</div>
                    <div class="bar-wrap">
                        <div class="bar-alta" style="width:{alta_pct}%"
                             title="Alta {alta_pct}%">{alta_pct}%</div>
                        <div class="bar-baja" style="width:{baja_pct}%"
                             title="Baja {baja_pct}%">{baja_pct}%</div>
                    </div>
                    <div class="labels"><span>Alta</span><span>Baja</span></div>
                </div>
            """

        # ====== render de las 3 tarjetas con selección ======
        colS1, colS2, colS3 = st.columns(3)
        for idx, col in enumerate([colS1, colS2, colS3], start=1):
            with col:
                col.markdown(
                    scheme_card_html(
                        num=idx,
                        selected=(_esquema == idx),
                        alta_frac=ESQUEMAS[idx]["alta"],
                        baja_frac=ESQUEMAS[idx]["baja"],
                    ),
                    unsafe_allow_html=True
                )
                if st.button(f"Seleccionar esquema {idx}", key=f"pick_scheme_{idx}", use_container_width=True):
                    _esquema = idx

        st.write(
            f"Esquema seleccionado: **{_esquema}**  •  "
            f"Alta: **{int(ESQUEMAS[_esquema]['alta']*100)}%**  |  "
            f"Baja: **{int(ESQUEMAS[_esquema]['baja']*100)}%**"
        )

        # ====== persistencia en sesión ======
        st.session_state.esquema_id = _esquema
        st.session_state.reduccion_baja_pct = _red_baja
   

    elif _caso == 3:
        st.markdown("**Caso 3 • Sensorización / Smart**")

        # --- estado persistente ---
        base = float(st.session_state.get("c3_base", 0.85))   # valor "Base" por defecto
        delta = float(st.session_state.get("c3_delta", 0.05)) # separación para Conservador/Optimista
        use_custom = bool(st.session_state.get("c3_use_custom", False))

        with st.expander("Opciones de preset (avanzado)", expanded=False):
            use_custom = st.checkbox("Definir mi valor base", value=use_custom)
            if use_custom:
                base = st.number_input("Valor base (Qᵣₑₗ)", min_value=0.60, max_value=1.00, value=base, step=0.01, format="%.2f")
                delta = st.number_input("Separación δ (±)", min_value=0.00, max_value=0.20, value=delta, step=0.01, format="%.2f")
            st.session_state.c3_use_custom = use_custom

        # clamp helper
        def clamp(x, lo=0.60, hi=1.00): 
            return max(lo, min(hi, x))

        q_con = clamp(base + delta)
        q_base = clamp(base)
        q_opt  = clamp(base - delta)

        # --- mostrar como etiquetas (no seleccionables) ---
        st.caption("Valores de referencia (no seleccionables aquí):")
        colA, colB, colC = st.columns(3)
        pill_css = """
            <div style="
                display:flex; flex-direction:column; gap:6px; align-items:center;
                background:#f3f4f6; border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; width:100%;
            ">
                <div style="font-weight:700;">{title}</div>
                <div style="font-size:14px; color:#111827;">{val:.2f}</div>
            </div>
        """
        with colA:
            st.markdown(pill_css.format(title="Conservador", val=q_con), unsafe_allow_html=True)
        with colB:
            st.markdown(pill_css.format(title="Base",        val=q_base), unsafe_allow_html=True)
        with colC:
            st.markdown(pill_css.format(title="Optimista",   val=q_opt),  unsafe_allow_html=True)

        st.caption("El caudal promedio se ajusta automáticamente según la demanda medida por sensores. Una menor fracción de caudal implica mayor ahorro (potencia ~ Q³).")

        # --- persistir para usar en Paso 5 (gráficas por escenario) ---
        st.session_state.c3_base  = base
        st.session_state.c3_delta = delta
        st.session_state.q_rel_conservador = q_con
        st.session_state.q_rel_base        = q_base
        st.session_state.q_rel_optimista   = q_opt

        # Compatibilidad: si tu cálculo actual usa un único Qᵣₑₗ, toma el "Base"
        st.session_state.q_rel_smart = q_base



    elif _caso == 4:
        with st.expander("Caso 4 • Implementación conjunta (C1+C2+C3)", expanded=False):
            _red_c1 = st.slider("C1: Reducción velocidad (%)", 0.0, 20.0, _red_c1*100, 1.0)/100.0
            _esquema = st.radio("C2: Esquema horario", [1,2,3], index=[1,2,3].index(_esquema), horizontal=True)
            _red_baja = st.slider("C2: Reducción en baja (%)", 0.0, 40.0, _red_baja*100, 1.0)/100.0
            _q_rel = st.slider("C3: Caudal relativo Smart (Qᵣₑₗ)", 0.60, 1.00, _q_rel, 0.01)

    # Guardar en session_state
    st.session_state.reduccion_c1_pct = _red_c1
    st.session_state.esquema_id = _esquema
    st.session_state.reduccion_baja_pct = _red_baja
    st.session_state.q_rel_smart = _q_rel

    # ---------------- Botones ----------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◀ Anterior", on_click=prev_step, key="prev4", type="secondary")
    with col2:
        if st.button("Siguiente ▶", key="next4", type="primary"):
            if ventiladores_prim < 1 or potencia_prim <= 0:
                st.session_state.mostrar_warning_tec = True
                st.rerun()
            else:
                st.session_state.ventiladores_prim = ventiladores_prim
                st.session_state.potencia_prim = potencia_prim
                st.session_state.ventiladores_comp = ventiladores_comp
                st.session_state.potencia_comp = potencia_comp
                st.session_state.tarifa = tarifa
                st.session_state.step += 1
                st.rerun()





# Resultado de análisis **************************************************************

# ================== Paso 5: Resultado de análisis ==================
if st.session_state.step == 5:
    st.header("Resultado del análisis :")
    import plotly.graph_objects as go

    def get_priority_color(index):
        colores = ["#fbeaea", "#fff3e0", "#fffde7"]
        return colores[index] if index < len(colores) else "#f7f8f9"

    card_line_style = """
        background-color: #f7f8f9;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 80%;
    """

    # --- columnas principales ---
    col1, col2 = st.columns([1, 1.2])

    # ================== COLUMNA IZQUIERDA ==================
    with col1:
        st.subheader("Resumen de información")

        st.markdown(
            f"<div style='{card_line_style}'>• Tipo de mina: "
            f"<b>{st.session_state.tipo_mina}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Material extraído: "
            f"<b>{st.session_state.tipo_material}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Producción: "
            f"<b>{st.session_state.produccion}</b></div>",
            unsafe_allow_html=True
        )

        # ---------- SOLO SI HAY CASO E-TROLLEY ----------
        et_params = st.session_state.get("et_params")
        if et_params:
            st.markdown("---")
            st.subheader("Parámetros E-Trolley")

            st.markdown(
                f"<div style='{card_line_style}'>• Distancia del tramo: "
                f"<b>{et_params.get('dist_km', 0):.2f} km</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Pendiente promedio: "
                f"<b>{et_params.get('pendiente', 0):.1f} %</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Cantidad de camiones: "
                f"<b>{int(et_params.get('n_trucks', 0))}</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Modelo seleccionado: "
                f"<b>{et_params.get('model', '-')}</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Tipo de inversión: "
                f"<b>{et_params.get('inv_type', '-')}</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Contingencia: "
                f"<b>{et_params.get('conting', 0):.1f} %</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Costo de energía: "
                f"<b>{et_params.get('energy', 0):.3f} USD/kWh</b></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='{card_line_style}'>• Mantenimiento trolley: "
                f"<b>{et_params.get('maint_y', 0)/1000:.1f} kUSD/año</b></div>",
                unsafe_allow_html=True
            )

        # ---------- PARÁMETROS VENTILACIÓN / VoD ----------
        st.markdown("---")
        st.subheader("Parámetros Ventilación / VoD")

        n_act = st.session_state.get("nivel_actual")
        n_obj = st.session_state.get("nivel_objetivo")
        caso  = st.session_state.get("caso")

        if caso is not None:
            st.markdown(
                f"<div style='{card_line_style}'>• Caso detectado: "
                f"<b>{caso}</b></div>",
                unsafe_allow_html=True
            )
        if n_act:
            st.markdown(
                f"<div style='{card_line_style}'>• Nivel actual: "
                f"<b>{n_act}</b></div>",
                unsafe_allow_html=True
            )
        if n_obj:
            st.markdown(
                f"<div style='{card_line_style}'>• Nivel objetivo: "
                f"<b>{n_obj}</b></div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<div style='{card_line_style}'>• Ventiladores primarios: "
            f"<b>{st.session_state.ventiladores_prim}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Potencia promedio (primarios): "
            f"<b>{st.session_state.potencia_prim} kW</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Ventiladores complementarios: "
            f"<b>{st.session_state.ventiladores_comp}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Potencia promedio (complementarios): "
            f"<b>{st.session_state.potencia_comp} kW</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='{card_line_style}'>• Tarifa eléctrica: "
            f"<b>{st.session_state.tarifa}</b></div>",
            unsafe_allow_html=True
        )

    # ================== COLUMNA DERECHA ==================
    with col2:
        st.subheader("Desafíos seleccionados")

        etiquetas_prioridad = [
            "🔴 Prioridad Alta",
            "🟠 Prioridad Media",
            "🟡 Prioridad Baja"
        ]

        for idx, d in enumerate(st.session_state.prioridades[:3]):
            bg_color = get_priority_color(idx)
            etiqueta = (
                etiquetas_prioridad[idx]
                if idx < len(etiquetas_prioridad)
                else f"Prioridad {idx+1}"
            )

            st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    padding: 10px 15px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    font-size: 80%;
                ">
                    <b>{etiqueta}:</b> {d}
                </div>
            """, unsafe_allow_html=True)

    # ------------ (de aquí hacia abajo dejas TODO igual: cálculos, gráficos, KPIs, botones) ------------
    # ... tu bloque de cálculos energéticos, curvas, KPIs y botones permanece sin cambios ...



    # ================== Cálculos energéticos ==================
    # Potencias y horas
    n_prim = st.session_state.ventiladores_prim
    p_prim = st.session_state.potencia_prim
    n_comp = st.session_state.ventiladores_comp
    p_comp = st.session_state.potencia_comp
    p_tot  = (n_prim * p_prim) + (n_comp * p_comp)
    horas_anuales = 8760.0

    # Tarifa numérica
    tarifa_dict = {
        "Mayor a 0.1": 0.11,
        "Entre 0.076 a 0.1": 0.09,
        "Entre 0.05 a 0.075": 0.065,
        "Menor a 0.05": 0.04
    }
    tarifa_real = tarifa_dict[st.session_state.tarifa]

    # Parámetros de casos desde Paso 4
    # C1: usamos 8% si no hay valor válido
    r_c1_cfg = st.session_state.get("reduccion_c1_pct", 0.08)
    try:
        r_c1_val = float(r_c1_cfg)
    except:
        r_c1_val = 0.08
    r_c1 = min(max(r_c1_val, 0.02), 0.30)  # forzar a rango sano (2%..30%)
    esquema_id = int(st.session_state.get("esquema_id", 2))

    # Caso 3 presets
    def clamp(x, lo=0.60, hi=1.00): return max(lo, min(hi, x))
    q_rel_con = st.session_state.get("q_rel_conservador")
    q_rel_bas = st.session_state.get("q_rel_base")
    q_rel_opt = st.session_state.get("q_rel_optimista")
    if q_rel_con is None or q_rel_bas is None or q_rel_opt is None:
        base = float(st.session_state.get("c3_base", 0.85))
        delta = float(st.session_state.get("c3_delta", 0.05))
        q_rel_con, q_rel_bas, q_rel_opt = clamp(base+delta), clamp(base), clamp(base-delta)

    # -------- Funciones de cálculo (consistentes con Excel) --------
    def energia_anual_mwh(p_kw: float, horas: float) -> float:
        return (p_kw * horas) / 1000.0

    def calc_caso1(E0_mwh: float, reduccion_pct: float, tarifa: float):
        E1 = E0_mwh * (1 - reduccion_pct) ** 3
        A1_mwh = max(E0_mwh - E1, 0.0)
        A1_usd = A1_mwh * 1000.0 * tarifa
        return E1, A1_mwh, A1_usd

    # 👇 Caso 2: parte de E1 y pondera Alta/Baja; en baja vuelve a aplicar (1-r)^3
    ESQUEMAS = {
        1: {"alta": 0.50, "baja": 0.50},
        2: {"alta": 0.83, "baja": 0.17},
        3: {"alta": 0.79, "baja": 0.21},
    }
    def calc_caso2_desde_E1(E1_mwh: float, esquema_id: int, r: float, tarifa: float):
        esq = ESQUEMAS.get(esquema_id, ESQUEMAS[2])
        f_alta, f_baja = esq["alta"], esq["baja"]
        E2 = E1_mwh * (f_alta + f_baja * (1.0 - r) ** 3)
        A2_mwh = max(E1_mwh - E2, 0.0)
        A2_usd = A2_mwh * 1000.0 * tarifa
        return E2, A2_mwh, A2_usd

    def calc_caso3(baseline_mwh: float, q_rel: float, tarifa: float):
        E3 = baseline_mwh * (q_rel ** 3)
        A3_mwh = max(baseline_mwh - E3, 0.0)
        A3_usd = A3_mwh * 1000.0 * tarifa
        return E3, A3_mwh, A3_usd

    # Serie payback Excel-like
    def serie_payback_excel_like(ahorro_anual_usd: float, capex_usd: float,
                                 meses_impl: int = 6,
                                 prr_meses=(1,3,5), prr_pct=(0.40,0.40,0.20),
                                 meses: int = 24):
        ahorro_mensual = ahorro_anual_usd / 12.0
        flujo, acumulado, c = [], [], 0.0
        for m in range(1, meses+1):
            gasto = 0.0
            for mm, pp in zip(prr_meses, prr_pct):
                if capex_usd > 0 and m == mm:
                    gasto += capex_usd * pp
            ahorro = ahorro_mensual if m > meses_impl else 0.0
            f = ahorro - gasto
            flujo.append(f)
            c += f
            acumulado.append(c)
        pb_mes = None
        if capex_usd > 0:
            for i, v in enumerate(acumulado):
                if v >= 0:
                    pb_mes = i+1
                    break
        return list(range(1, meses+1)), flujo, acumulado, pb_mes

    # -------- Cálculo encadenado --------
    E0 = energia_anual_mwh(p_tot, horas_anuales)
    E1, A1_mwh, A1_usd = calc_caso1(E0, r_c1, tarifa_real)
    E2, A2_mwh, A2_usd = calc_caso2_desde_E1(E1, esquema_id, r_c1, tarifa_real)
    E3_base, A3_mwh_base, A3_usd_base = calc_caso3(E2, q_rel_bas, tarifa_real)

    # Selección por caso
    if caso == 1:
        E_final = E1; ahorro_mwh_total = A1_mwh; ahorro_usd_total = A1_usd; nivel_vod = 1
    elif caso == 2:
        E_final = E2; ahorro_mwh_total = A1_mwh + A2_mwh; ahorro_usd_total = A1_usd + A2_usd; nivel_vod = 2
    else:  # 3 ó 4
        E_final = E3_base; ahorro_mwh_total = A1_mwh + A2_mwh + A3_mwh_base; ahorro_usd_total = A1_usd + A2_usd + A3_usd_base; nivel_vod = 3

    # -------------------- CAPEX (con fallback por Caso) --------------------
    total_vent = float(n_prim + n_comp)

    # 1) Intentar por transición (nivel actual -> nivel objetivo)
    capex_unit = None
    if n_act is not None and n_obj is not None:
        capex_unit = CAPEX_POR_VEN_DEFAULT.get((n_act, n_obj))

    # 2) Fallback por Caso si no hay niveles o no hace match (ajusta a valores reales de tu Excel)
    if capex_unit is None:
        CAPEX_POR_VEN_POR_CASO = {1: 4000.0, 2: 7000.0, 3: 12000.0, 4: 20000.0}
        capex_unit = CAPEX_POR_VEN_POR_CASO.get(caso, 0.0)

    capex_total = float(capex_unit) * total_vent

    # -------- Diagnóstico rápido --------
    with st.expander("🔎 Diagnóstico de cálculo (ver valores)", expanded=False):
        st.write({
            "n_act": n_act, "n_obj": n_obj, "caso": caso,
            "r_c1(%)": round(r_c1*100,2), "esquema_id": esquema_id,
            "E0 (MWh)": round(E0,2), "E1 (MWh)": round(E1,2),
            "E2 (MWh)": round(E2,2), "E3_base (MWh)": round(E3_base,2),
            "ahorro_mwh_total": round(ahorro_mwh_total,2),
            "ahorro_usd_total": round(ahorro_usd_total,2),
            "capex_unit_usd/vent": float(capex_unit),
            "total_vent": total_vent,
            "CAPEX total": round(capex_total,2),
        })

    # ================== Recomendación ==================
    st.markdown(f"""
    <div style='background-color:#DFF0D8; padding:15px; font-size:150%; font-weight:bold; text-align:center; border-radius:8px;'>
    ✅ Recomendación: Implementar Ventilation On Demand (Nivel {nivel_vod})
    </div>
    """, unsafe_allow_html=True)

     # ================== Gráfico 1: Consumo anual ==================
    # Convertimos todo a GWh
    E0_gwh      = E0 / 1000.0
    E_final_gwh = E_final / 1000.0
    delta_gwh   = E0_gwh - E_final_gwh
    pct_ahorro  = (delta_gwh / E0_gwh * 100.0) if E0_gwh > 0 else 0.0

    fig1 = go.Figure()

    # Barra 1: control actual
    fig1.add_trace(go.Bar(
        name="Con control actual",
        x=["Total"],
        y=[E0_gwh],
        marker_color="#d3d3d3",
        text=[f"{E0_gwh:.2f} GWh"],
        textposition="inside"
    ))

    # Barra 2: VoD
    fig1.add_trace(go.Bar(
        name=f"Con ABB VoD (Caso {caso})",
        x=["Total"],
        y=[E_final_gwh],
        marker_color="#439889",
        text=[f"{E_final_gwh:.2f} GWh"],
        textposition="inside"
    ))

    # Línea entre los centros superiores de ambas barras (en GWh)
    fig1.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0.25, y0=E0_gwh,
        x1=0.75, y1=E_final_gwh,
        line=dict(color="black", width=2)
    )

    # Etiqueta en el medio de la línea: % y GWh
    fig1.add_annotation(
        xref="paper", yref="y",
        x=0.5,
        y=(E0_gwh + E_final_gwh) / 2,
        text=f"-{pct_ahorro:.1f}% ({delta_gwh:.2f} GWh)",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )

    # 👉 centra texto dentro de cada barra
    fig1.update_traces(
        textposition="inside",
        insidetextanchor="middle"
    )

    fig1.update_layout(
        title="🔌 Consumo Energético Anual",
         yaxis_title="Energía (GWh)",
        xaxis_title="Sistema",
        barmode='group'
    )

    # ================== Gráfico 2: Ahorro Económico Acumulado (24m) ==================
    # Parámetros estilo Excel (ajústalos si tu hoja usa otros)
    MESES_IMPL_POR_CASO = {1: 3, 2: 4, 3: 6, 4: 6}   # tiempo de implementación por caso
    PRR_MESES_DEFAULT   = (1, 3, 5)                 # meses de desembolso CAPEX
    PRR_PCT_DEFAULT     = (0.40, 0.40, 0.20)        # % de desembolso CAPEX

    # Permitir override desde session_state si más adelante agregas controles
    meses_impl = st.session_state.get("meses_impl", MESES_IMPL_POR_CASO.get(caso, 6))
    prr_meses  = st.session_state.get("prr_meses", PRR_MESES_DEFAULT)
    prr_pct    = st.session_state.get("prr_pct",   PRR_PCT_DEFAULT)

    # Curva del caso DETECTADO (solo el caso actual)
    fig2 = go.Figure()

    def curva_payback(ahorro_anual_usd: float, capex: float, meses_impl: int):
        meses_lbl, _, acumulado, pb = serie_payback_excel_like(
            ahorro_anual_usd, capex,
            meses_impl=meses_impl, prr_meses=prr_meses, prr_pct=prr_pct, meses=24
        )
        return meses_lbl, acumulado, pb

    meses_lbl, acumulado_base, pb_base = curva_payback(ahorro_usd_total, capex_total, meses_impl)

    fig2.add_trace(go.Scatter(
        name=f"Ahorro acumulado (Caso {caso})",
        x=meses_lbl, y=acumulado_base,
        mode="lines+markers",
        line=dict(width=3)
    ))

    # Marca de inicio de ahorros y payback (si corresponde)
    fig2.add_vline(x=meses_impl + 1, line=dict(color="gray", dash="dot"))
    fig2.add_annotation(x=meses_impl + 1, y=0, text="Inicio de ahorros", showarrow=True, yshift=30)

    if pb_base:
        fig2.add_vline(x=pb_base, line=dict(color="#439889", dash="dot"))
        fig2.add_annotation(x=pb_base, y=max(acumulado_base), text=f"Payback M{pb_base}", showarrow=True)

    # === CAPEX inicial (primer desembolso) como rectángulo pequeño ===
    if capex_total > 0 and acumulado_base:
        # Valor acumulado en el mes 1 = primer desembolso (negativo)
        y_capex = acumulado_base[0]          # ej. -22,400 si es 40% de 56,000
        primer_capex = abs(y_capex)          # magnitud del primer desembolso

        # Altura del rectángulo = 25% de ese primer desembolso
        mag = primer_capex
        banda = mag * 0.25                   # rectángulo más delgado

        # Rectángulo pegado al valor real (parte baja) y más delgado hacia arriba
        y1 = y_capex                         # parte inferior = valor real (negativo)
        y0 = y_capex + banda                 # parte superior, más cerca de 0

        # Rectángulo ligeramente a la izquierda del mes 1
        fig2.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=0.3,       # más a la izquierda de M1
            x1=0.9,
            y0=y0,
            y1=y1,
            fillcolor="rgba(255,136,0,0.30)",
            line=dict(color="rgba(255,80,0,1)", width=2),
            layer="above"
        )

        # Etiqueta centrada en el rectángulo, mostrando el primer desembolso en kUSD
        fig2.add_annotation(
            x=0.6,
            y=(y0 + y1) / 2.0,
            text=f"Inicial 40% \nUSD {primer_capex/1000:,.0f}k",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="left"
        )


    fig2.update_layout(
        title="💰 Ahorro Económico Acumulado (24 meses)",
        xaxis_title="Meses",
        yaxis_title="USD",
        legend=dict(orientation="h")
    )

    # Mostrar gráficos
    colg1, colg2 = st.columns(2)
    colg1.plotly_chart(fig1, use_container_width=True)
    colg2.plotly_chart(fig2, use_container_width=True)

    # Mensaje de payback si aplica
    if pb_base:
        st.success(f"💡 **Payback estimado**: mes {pb_base}. Desde aquí, los ahorros superan la inversión.")

    # ================== KPIs (Base) ==================
    card_style = """
        background-color: #f7f8f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    """
    indicadores_vod = {
        "Ahorro de energía anual": f"{ahorro_mwh_total/1000:.1f} GWh",
        "Consumo energético actual anual": f"{E0/1000:.1f} GWh",
        "Porcentaje de ahorro energético anual": f"{(ahorro_mwh_total/E0*100):.1f} %"
    }
    indicadores_economicos = {
        "Consumo con VoD anual": f"{E_final/1000:.1f} GWh",
        "Ahorro económico anual": f"{ahorro_usd_total/1000:,.0f} K USD",
        "Reducción de emisiones CO₂": f"{ahorro_mwh_total * 0.3:.1f} t/año"
    }

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Indicadores VoD (Base)")
        for k, v in indicadores_vod.items():
            st.markdown(f"""
                <div style="{card_style}">
                    <div style="font-size:26px; font-weight:bold; color:#1a1a1a;">{v}</div>
                    <div style="font-size:14px; color:#666;">{k}</div>
                </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("### Ahorro económico (Base)")
        for k, v in indicadores_economicos.items():
            st.markdown(f"""
                <div style="{card_style}">
                    <div style="font-size:26px; font-weight:bold; color:#1a1a1a;">{v}</div>
                    <div style="font-size:14px; color:#666;">{k}</div>
                </div>
            """, unsafe_allow_html=True)

    st.info("""
    ✔️ **Simulación completa**
    Propuesta técnica y económica.
    Evaluación VoD con indicadores Base y curva de payback por caso.
    """)

    colr1, colr2 = st.columns([1, 1])
    with colr1:
        if st.button("🔄 Reiniciar simulación", key="reiniciar", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with colr2:
        st.button("📧 Contactar a especialista ABB", key="contactar", type="primary")


