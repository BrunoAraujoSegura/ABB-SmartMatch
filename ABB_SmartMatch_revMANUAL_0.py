
import streamlit as st
import plotly.graph_objects as go

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
        color = ["🔴 Prioridad 1", "🟠 Prioridad 2", "🟡 Prioridad 3"]
        if idx < 3:
            st.markdown(
                f"<span style='margin-left:10px;'>{item}: <b style='color:red'>{color[idx]}</b></span>",
                unsafe_allow_html=True
            )

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
if st.session_state.step == 3:
#    st.header("Paso 3 de 4: Nivel de Automatización")
    st.header("Paso 3 de 4: Nivel actual y objetivo")
#    opciones_auto = [
#        "Sin automatización",
#        "Automatización básica PLC",
#        "Sistema de Control Distribuido (DCS)",
#        "Automatización avanzada con IA"
#    ]
#    seleccion_auto = st.radio("Seleccione el nivel de automatización:", opciones_auto)
#    st.session_state.nivel_auto = seleccion_auto
#
#    col1, col2 = st.columns([1, 1])
#    with col1:
#        st.button("◀ Anterior", on_click=prev_step, key="prev3", type="secondary")
#    with col2:
#        if st.button("Siguiente ▶", key="next3", type="primary"):
#            if not st.session_state.nivel_auto:
#                st.warning("Por favor selecciona un nivel de automatización.")
#            else:
#                next_step()
#                st.rerun()   
 

    col1, col2 = st.columns(2)
    with col1:
        n_act = st.selectbox("Nivel actual", NIVELES_ORDEN, index=0, key="nivel_actual")
    with col2:
        n_obj = st.selectbox("Nivel objetivo", NIVELES_ORDEN, index=1, key="nivel_objetivo")

    ruta = ruta_secuencial(n_act, n_obj)
    siguiente_habilitado = True

    # Apaga el toggle si cambiaste a algo distinto de Esencial→Smart
    if st.session_state.get("usar_caso_4") and not (n_act == "Esencial" and n_obj == "Smart"):
        st.session_state.usar_caso_4 = False

    if ruta is None:
        st.warning("No se permite una transición descendente. Selecciona un nivel objetivo igual o superior.")
        siguiente_habilitado = False

    elif len(ruta) == 0:
        st.info("El nivel actual y el objetivo son iguales. Selecciona una transición distinta.")
        siguiente_habilitado = False

    else:
        # Secuencia E→B→OE→S
        casos = casos_desde_ruta(ruta)
        st.session_state.ruta = ruta
        st.session_state.casos = casos
        st.session_state.caso = casos[0]  # compatibilidad

        usar_c4 = False
        if n_act == "Esencial" and n_obj == "Smart":
            usar_c4 = st.toggle("Usar implementación conjunta (Caso 4)", key="usar_caso_4",
                                value=st.session_state.get("usar_caso_4", False))

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



    # ---------------- Botones ----------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◀ Anterior", on_click=prev_step, key="prev3", type="secondary")
    with col2:
        clicked = st.button("Siguiente ▶", key="next3", type="primary", disabled=not siguiente_habilitado)
        if clicked:
            next_step()
            st.rerun()






 

# Paso 4 de 4

# Paso 4 de 4
if st.session_state.step == 4:
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
        st.markdown("**Caso 2 • Configurar: Horarios programados**")
        st.caption(f"Este caso parte del resultado de Caso 1 usando reducción = {_red_c1*100:.0f} %")

        # ====== estilos de las tarjetas / barras ======
        st.markdown("""
            <style>
            .scheme-card {
                border: 2px solid #e5e7eb; border-radius: 14px; padding: 12px 14px; background: #ffffff;
                transition: box-shadow .2s ease, border-color .2s ease;
            }
            .scheme-card.selected { border-color: #2563eb; box-shadow: 0 4px 18px rgba(37,99,235,.15); }
            .scheme-title { font-weight: 700; text-align: center; margin-bottom: 8px; }
            .bar-wrap { width: 100%; height: 28px; background: #eef2f7; border-radius: 999px; overflow: hidden; display: flex; }
            .bar-alta { background: #335b89; height: 100%; display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; }
            .bar-baja { background: #dfe7f2; height: 100%; display:flex; align-items:center; justify-content:center; color:#1f2937; font-size:12px; }
            .labels { display:flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color:#4b5563; }
            </style>
        """, unsafe_allow_html=True)

        def scheme_card_html(num:int, selected:bool, alta_frac:float, baja_frac:float) -> str:
            alta_pct = round(alta_frac*100)
            baja_pct = 100 - alta_pct
            return f"""
                <div class="scheme-card {'selected' if selected else ''}">
                    <div class="scheme-title">Esquema {num}</div>
                    <div class="bar-wrap">
                        <div class="bar-alta" style="width:{alta_pct}%" title="Alta {alta_pct}%">{alta_pct}%</div>
                        <div class="bar-baja" style="width:{baja_pct}%" title="Baja {baja_pct}%">{baja_pct}%</div>
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
                # botón de selección (ligero) bajo cada tarjeta
                if st.button(f"Seleccionar esquema {idx}", key=f"pick_scheme_{idx}", use_container_width=True):
                    _esquema = idx

        st.write(f"Esquema seleccionado: **{_esquema}**  •  "
                 f"Alta: **{int(ESQUEMAS[_esquema]['alta']*100)}%**  |  "
                 f"Baja: **{int(ESQUEMAS[_esquema]['baja']*100)}%**")


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
    with col1:
        st.subheader("Resumen de información")
        st.markdown(f"<div style='{card_line_style}'>• Tipo de mina: <b>{st.session_state.tipo_mina}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Material extraído: <b>{st.session_state.tipo_material}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Producción: <b>{st.session_state.produccion}</b></div>", unsafe_allow_html=True)

        # Niveles y caso
        n_act = st.session_state.get("nivel_actual")
        n_obj = st.session_state.get("nivel_objetivo")
        caso  = st.session_state.get("caso")
        if n_act: st.markdown(f"<div style='{card_line_style}'>• Nivel actual: <b>{n_act}</b></div>", unsafe_allow_html=True)
        if n_obj: st.markdown(f"<div style='{card_line_style}'>• Nivel objetivo: <b>{n_obj}</b></div>", unsafe_allow_html=True)
        if caso:  st.markdown(f"<div style='{card_line_style}'>• Caso detectado: <b>{caso}</b></div>", unsafe_allow_html=True)

        # Parámetros técnicos
        st.markdown(f"<div style='{card_line_style}'>• Ventiladores primarios: <b>{st.session_state.ventiladores_prim}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Potencia promedio (primarios): <b>{st.session_state.potencia_prim} kW</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Ventiladores complementarios: <b>{st.session_state.ventiladores_comp}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Potencia promedio (complementarios): <b>{st.session_state.potencia_comp} kW</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Tarifa eléctrica: <b>{st.session_state.tarifa}</b></div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Desafíos seleccionados")
        for idx, d in enumerate(st.session_state.prioridades[:3]):
            bg_color = get_priority_color(idx)
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 10px 15px; border-radius: 8px; margin-bottom: 8px; font-size: 80%;">
                    {idx + 1}. {d}
                </div>
            """, unsafe_allow_html=True)

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
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="Con control actual", x=["Total"], y=[E0], marker_color="#d3d3d3"))
    fig1.add_trace(go.Bar(name=f"Con ABB VoD (Caso {caso})", x=["Total"], y=[E_final], marker_color="#439889"))
    fig1.update_layout(title="🔌 Consumo Energético Anual", yaxis_title="Energía (MWh)", xaxis_title="Sistema", barmode='group')

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

    # Línea de inversión (solo si hay CAPEX)
    if capex_total > 0:
        fig2.add_trace(go.Scatter(
            name="Inversión (CAPEX)",
            x=[0] + meses_lbl,
            y=[capex_total] * (len(meses_lbl) + 1),
            mode="lines",
            line=dict(dash="dash", color="red")
        ))

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


