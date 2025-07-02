
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="ABB SmartMatch™")

# Logo (si existe)
try:
    st.image("abb_logo.png", width=100)
except:
    pass

st.title("ABB SmartMatch™")
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
    st.session_state.produccion = st.selectbox("Tonelada diaria producida:", [
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



            
# Paso 3 de 4
if st.session_state.step == 3:
    st.header("Paso 3 de 4: Nivel de Automatización")
    opciones_auto = [
        "Sin automatización",
        "Automatización básica PLC",
        "Sistema de Control Distribuido (DCS)",
        "Automatización avanzada con IA"
    ]
    seleccion_auto = st.radio("Seleccione el nivel de automatización:", opciones_auto)
    st.session_state.nivel_auto = seleccion_auto

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◀ Anterior", on_click=prev_step, key="prev3", type="secondary")
    with col2:
        if st.button("Siguiente ▶", key="next3", type="primary"):
            if not st.session_state.nivel_auto:
                st.warning("Por favor selecciona un nivel de automatización.")
            else:
                next_step()
                st.rerun()

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

    ventiladores_prim = st.number_input("Cantidad de ventiladores primarios:", min_value=0, max_value=20, step=1)
    potencia_prim = st.number_input("Potencia promedio por ventilador primario(kW):", value=75.0)
    ventiladores_comp = st.number_input("Cantidad de ventiladores complementarios:", min_value=0, max_value=20, step=1)
    potencia_comp = st.number_input("Potencia promedio por ventilador complementario(kW):", value=1.5)

    tarifa = st.selectbox("Costo de energía (USD/kWh):", [
        "Mayor a 0.1", 
        "Entre 0.076 a 0.1", 
        "Entre 0.05 a 0.075",
        "Menor a 0.05"
    ])

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


# Resultado de análisis

# Función para obtener color pastel según prioridad
def get_priority_color(index):
    colores = ["#fbeaea", "#fff3e0", "#fffde7"]  # rojo claro, naranja claro, amarillo claro
    return colores[index] if index < len(colores) else "#f7f8f9"

if st.session_state.step == 5:
    st.header("Resultado del análisis :")

    # ↓ Estilo para resumen de información
    card_info_style = """
        background-color: #f7f8f9;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 80%;
    """
    
    # ↓ Estilo para cada línea tipo tarjeta (fondo gris claro, similar a los indicadores VoD)
    card_line_style = """
        background-color: #f7f8f9;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 80%;
    """

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Resumen de información")
        st.markdown(f"<div style='{card_line_style}'>• Tipo de mina: <b>{st.session_state.tipo_mina}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Material extraído: <b>{st.session_state.tipo_material}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Producción: <b>{st.session_state.produccion}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{card_line_style}'>• Nivel de automatización: <b>{st.session_state.nivel_auto}</b></div>", unsafe_allow_html=True)
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

    # ✅ Recomendación destacada
    st.markdown("<div style='background-color:#DFF0D8; padding:10px; font-size:150%; font-weight:bold;'>✅ Recomendación: Implementar Ventilation On Demand (Nivel 1)</div>", unsafe_allow_html=True)

    # ▶️ Gráfico comparativo energético anual
    energia_actual = 192.3
    energia_vod = 119.3

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="with existing control method", x=["Total"], y=[energia_actual], marker_color="#d3d3d3"))
    fig1.add_trace(go.Bar(name="with ABB drive control", x=["Total"], y=[energia_vod], marker_color="#439889"))

    # Agregar línea diagonal de ahorro
    fig1.add_shape(
        type="line",
        x0=-0.15, y0=energia_actual,
        x1=0.15, y1=energia_vod,
        line=dict(color="red", width=2, dash="dash"),
    )

    # Agregar texto del ahorro
    fig1.add_annotation(
        x=0, y=(energia_actual + energia_vod)/2,
        text="Ahorro 38%",
        showarrow=False,
        font=dict(size=12, color="red"),
        yshift=10
    )

    fig1.update_layout(
        title="🔌 Consumo Energético Anual",
        yaxis_title="Energy consumption (MWh)",
        xaxis_title="Sistema",
        barmode='group'
    )


    # ▶️ Gráfico económico acumulado
    ahorro_anual = 7302
    ahorro_mensual = ahorro_anual / 12
    inversion_inicial = 10000
    meses = list(range(1, 7))
    ahorro_acumulado = [i * ahorro_mensual for i in meses]

    # Detectar mes de payback
    payback_mes = next((i + 1 for i, ahorro in enumerate(ahorro_acumulado) if ahorro >= inversion_inicial), None)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(name="Ahorro acumulado", x=meses, y=ahorro_acumulado,
                              mode="lines+markers", line=dict(color="#0072C6", width=3)))  # azul ABB
    fig2.add_trace(go.Scatter(name="Inversión inicial", x=meses, y=[inversion_inicial]*6,
                              mode="lines", line=dict(dash="dash", color="red")))

    if payback_mes:
        fig2.add_shape(
            type="line",
            x0=payback_mes,
            x1=payback_mes,
            y0=0,
            y1=max(max(ahorro_acumulado), inversion_inicial),
            line=dict(color="#439889", dash="dot")
        )
        fig2.add_trace(go.Scatter(
            name="Ahorro neto",
            x=meses[payback_mes-1:],
            y=ahorro_acumulado[payback_mes-1:],
            fill='tozeroy',
            mode='none',
            fillcolor="rgba(67, 152, 137, 0.2)",
            showlegend=False
        ))

    fig2.update_layout(
        title="💰 Ahorro Económico Acumulado",
        xaxis_title="Meses",
        yaxis_title="USD"
    )

    colg1, colg2 = st.columns(2)
    colg1.plotly_chart(fig1, use_container_width=True)
    colg2.plotly_chart(fig2, use_container_width=True)

    if payback_mes:
        st.success(f"💡 **Payback estimado**: mes {payback_mes}. A partir de aquí, los ahorros netos superan la inversión.")

    # Indicadores VoD y Ahorro económico
    st.markdown("## Indicadores VoD & Ahorro económico")

    card_style = """
        background-color: #f7f8f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    """

    indicadores_vod = {
        "Ahorro de energía anual": "73 MWh",
        "Consumo energético actual": "192.3 MWh",
        "Porcentaje de ahorro energético": "38 %"
    }

    indicadores_economicos = {
        "Consumo con VoD": "119.3 MWh",
        "Ahorro económico anual": "7,302 USD",
        "Reducción de emisiones CO₂": "21.9 t/año"
    }

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Indicadores VoD")
        for k, v in indicadores_vod.items():
            st.markdown(f"""
                <div style="{card_style}">
                    <div style="font-size:26px; font-weight:bold; color:#1a1a1a;">{v}</div>
                    <div style="font-size:14px; color:#666;">{k}</div>
                </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("### Ahorro económico")
        for k, v in indicadores_economicos.items():
            st.markdown(f"""
                <div style="{card_style}">
                    <div style="font-size:26px; font-weight:bold; color:#1a1a1a;">{v}</div>
                    <div style="font-size:14px; color:#666;">{k}</div>
                </div>
            """, unsafe_allow_html=True)

    # Mensaje de cierre
    st.info("""
    ✔️ **Simulación completa**  
    Propuesta técnica y económica.  
    La evaluación para implementar VoD ha sido completada.  
    Incluye indicadores técnicos, operativos y sostenibles.
    """)

    # Botones
    colr1, colr2 = st.columns([1, 1])
    with colr1:
        if st.button("🔄 Reiniciar simulación", key="reiniciar", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with colr2:
        st.button("📧 Contactar a especialista ABB", key="contactar", type="primary")

