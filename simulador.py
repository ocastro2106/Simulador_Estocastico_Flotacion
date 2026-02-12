import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.ticker as mtick
import datetime  # <--- AGREGA ESTA LÍNEA
# --- CONFIGURACIÓN DE PÁGINA Y ESTILO ---
st.set_page_config(
    page_title="Flotation Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- CONFIGURACIÓN DE SEGURIDAD ---
# 1. Define tu clave maestra
CLAVE_ACCESO = "mineria2026" 

# 2. Define la fecha de vencimiento (Año, Mes, Día)
# Ejemplo: Vence el 30 de Octubre de 2024
FECHA_VENCIMIENTO = datetime.date(2026, 3, 30) 

def check_login():
    """Función que gestiona el bloqueo de pantalla."""
    
    # Inicializar estado de sesión
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    # Si ya está autenticado, no hacer nada y dejar pasar
    if st.session_state['authenticated']:
        return True

    # --- INTERFAZ DE LOGIN ---
    st.markdown("### 🔒 Acceso Restringido")
    st.markdown("Este simulador es confidencial. Ingrese su clave de acceso.")
    
    input_pass = st.text_input("Contraseña:", type="password")
    btn_login = st.button("Ingresar")

    if btn_login:
        hoy = datetime.date.today()
        
        # Validación 1: Fecha
        if hoy > FECHA_VENCIMIENTO:
            st.error(f"⛔ El periodo de acceso expiró el {FECHA_VENCIMIENTO}.")
            return False
            
        # Validación 2: Contraseña
        if input_pass == CLAVE_ACCESO:
            st.session_state['authenticated'] = True
            st.rerun() # Recarga la página para mostrar el contenido
        else:
            st.error("❌ Contraseña incorrecta.")
            return False

    return False

# --- EJECUCIÓN DEL BLOQUEO ---
# Si la función check_login devuelve False, detenemos toda la app aquí.
if not check_login():
    st.stop() 

# ==========================================
# A PARTIR DE AQUÍ CONTINÚA TU CÓDIGO NORMAL
# (Estilos CSS, Funciones, Tabs, etc.)
# ==========================================
# Estilos CSS personalizados para un look más profesional
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #eef2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #5d6d7e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #2980b9;
        border-top: 3px solid #2980b9;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DE ESTADO GLOBAL ---
if 'rec_delta_global' not in st.session_state:
    st.session_state['rec_delta_global'] = 0.0

# --- FUNCIONES CORE (Reutilizables) ---


# --- CLASE DE EXTRAPOLACIÓN ---
class ExtrapolatedSpline:
    def __init__(self, x, y):
        # Creamos la Spline con condición 'natural'
        self.cs = CubicSpline(x, y, bc_type='natural')
        self.x_min = x.min()
        self.x_max = x.max()
        
        # Pre-calculamos valores para las rectas de los extremos
        self.y_min = self.cs(self.x_min)
        self.y_max = self.cs(self.x_max)
        self.d_min = self.cs(self.x_min, 1) # Pendiente inicio
        self.d_max = self.cs(self.x_max, 1) # Pendiente fin
    
    def __call__(self, x_in):
        # Esta función permite que la clase se comporte como f(x)
        x = np.atleast_1d(x_in)
        y = np.zeros_like(x)
        
        mask_low = x < self.x_min
        mask_high = x > self.x_max
        mask_in = ~(mask_low | mask_high)
        
        # Interpolación (centro)
        if np.any(mask_in):
            y[mask_in] = self.cs(x[mask_in])
        
        # Extrapolación (izquierda)
        if np.any(mask_low):
            y[mask_low] = self.y_min + self.d_min * (x[mask_low] - self.x_min)
            
        # Extrapolación (derecha)
        if np.any(mask_high):
            y[mask_high] = self.y_max + self.d_max * (x[mask_high] - self.x_max)
            
        if np.ndim(x_in) == 0:
            return y[0]
        return y

# --- AQUÍ ESTÁ LA SOLUCIÓN: Usar cache_resource ---
def get_spline_data(df):
    """Genera la función con extrapolación lineal y los puntos extendidos."""
    df = df.sort_values(by="P80")
    try:
        # Creamos el objeto matemático
        spline_wrapper = ExtrapolatedSpline(df["P80"].values, df["Recuperacion"].values)
        
        # Definimos el rango de graficación extendido
        x_range = df["P80"].max() - df["P80"].min()
        buffer = x_range * 0.2 
        
        x_min_buff = float(df["P80"].min() - buffer)
        x_max_buff = float(df["P80"].max() + buffer)
        
        x_smooth = np.linspace(x_min_buff, x_max_buff, 500)
        y_smooth = spline_wrapper(x_smooth)
        y_smooth = np.clip(y_smooth, 0, 100)
        
        return spline_wrapper, x_smooth, y_smooth
    except Exception as e:
        st.error(f"Error en la interpolación: {e}")
        return None, None, None
def calculate_mean_recovery(mean_p80, std_p80, spline_func, n_sims=5000):
    """Función ligera para calcular solo la recuperación media con protección de límites."""
    np.random.seed(42 + int(mean_p80))
    p80_samples = np.random.normal(mean_p80, std_p80, n_sims)
    
    # Calcular valores crudos
    rec_samples = spline_func(p80_samples)
    
    # --- CORRECCIÓN DE SEGURIDAD ---
    # np.clip fuerza a que cualquier valor < 0 sea 0 y cualquier valor > 100 sea 100
    rec_samples = np.clip(rec_samples, 0.0, 100.0)
    
    return np.mean(rec_samples)

def plot_scenario(mean_p80, std_p80, cs, x_smooth, y_smooth, df_curve, label_escenario, n_sims=3000):
    """Genera el gráfico combinado y calcula la media con la leyenda corregida."""
    # 1. Generación de datos estocásticos
    np.random.seed(42 + int(mean_p80))
    p80_samples = np.random.normal(mean_p80, std_p80, n_sims)
    
    # 2. Cálculo de recuperación
    rec_samples = cs(p80_samples)
    rec_samples = np.clip(rec_samples, 0, 100)
    mean_rec = np.mean(rec_samples)
    
    # 3. Configuración del Gráfico
    # Aumentamos ligeramente la altura (6) para dar espacio a la leyenda abajo
    fig, ax1 = plt.subplots(figsize=(8, 6)) 
    
    # --- Eje Y1: Curva de Recuperación (Azul) ---
    color_curve = '#2980b9' 
    ax1.set_xlabel('P80 (micrones)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Recuperación (%)', color=color_curve, fontweight='bold', fontsize=10)
    
    # Graficar curva y puntos
    line1, = ax1.plot(x_smooth, y_smooth, color=color_curve, linewidth=2.5, label='Curva Rec. vs P80')
    scatter1 = ax1.scatter(df_curve["P80"], df_curve["Recuperacion"], color='#c0392b', s=60, zorder=5, label='Datos Lab')
    
    ax1.tick_params(axis='y', labelcolor=color_curve)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Eje Y2: Histograma (Naranja) ---
    ax2 = ax1.twinx()
    color_hist = '#e67e22' 
    ax2.set_ylabel('Densidad de Probabilidad (P80)', color=color_hist, fontweight='bold', fontsize=10)
    
    # Graficar histograma
    # Nota: ax2.hist devuelve 3 valores, guardamos los patches para referencia futura si se necesita
    n, bins, patches = ax2.hist(p80_samples, bins=40, density=True, alpha=0.5, color=color_hist, edgecolor=color_hist, label='Distribución P80')
    
    # Creamos un "Patch" manual para asegurar que la leyenda del histograma se muestre correctamente
    from matplotlib.patches import Patch
    patch_hist = Patch(color=color_hist, alpha=0.5, label='Distribución P80')

    ax2.tick_params(axis='y', labelcolor=color_hist, labelsize=9)
    ax2.set_ylim(bottom=0)
    
    # --- LEYENDA EXTERNA (CORRECCIÓN) ---
    # Recolectamos los elementos para la leyenda combinada
    lines = [line1, scatter1, patch_hist]
    labels = [l.get_label() for l in lines]
    
    # bbox_to_anchor=(0.5, -0.15) coloca la leyenda centrada debajo del eje X
    ax1.legend(lines, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), 
               ncol=3, 
               frameon=False,
               fontsize=9)
    
    # Título
    plt.title(f"{label_escenario}\nRecuperación Media: {mean_rec:.2f}%", pad=10, fontweight='bold', color='#34495e')
    
    # tight_layout ajusta los márgenes para que la leyenda externa no se corte
    plt.tight_layout()
    
    return fig, mean_rec

# --- BARRA LATERAL COMÚN ---
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    st.markdown("Define la curva base de recuperación.")
    default_data = {
        "P80": [18, 50, 90, 140, 200, 230],
        "Recuperacion": [30, 70, 90, 90, 78, 39]
    }
    df_curve = st.data_editor(pd.DataFrame(default_data), num_rows="dynamic", height=250, use_container_width=True)
    cs_global, x_smooth_global, y_smooth_global = get_spline_data(df_curve)
    
    if cs_global is None:
        st.stop()
        
    st.divider()
    st.info("Esta configuración aplica a todas las secciones.")

# --- ESTRUCTURA PRINCIPAL ---
st.title("Sistema de Optimización de Flotación")
st.markdown("---")

# Definición de Tabs
tab_home, tab_model, tab_sens, tab_econ = st.tabs([
    "🏠 Home (Presentación)", 
    "📈 Flotation Model (Estocástico)", 
    "📊 Sensitivity Analysis", 
    "💰 Economic Evaluation"
])

# ===================== TAB 1: HOME =====================
with tab_home:
    col_text, col_img = st.columns([2, 3])
    
    with col_text:
        st.subheader("El Impacto Crítico de la Variabilidad del P80")
        st.markdown("""
        En el proceso de flotación, el tamaño de partícula (P80) es una variable clave que impacta directamente la recuperación metalúrgica.

        **¿Por qué es vital controlar la dispersión?**
        
        Las curvas de recuperación suelen tener una forma de "campana" o meseta. Existe un rango óptimo de P80 donde la recuperación es máxima.
        
        1.  **Pérdidas por Gruesos:** Si el P80 es muy alto (desplazamiento a la derecha), las partículas son demasiado pesadas.
        2.  **Pérdidas por Finos:** Si el P80 es muy bajo (desplazamiento a la izquierda), ocurren fenómenos de "sliming".
        
        Una **alta desviación estándar** significa que, aunque el promedio sea correcto, una gran parte del tonelaje se procesa en zonas de baja recuperación.
        
        > **El objetivo no es solo apuntar al P80 medio correcto, sino minimizar la variabilidad (hacer la curva más angosta).**
        """)
    
    with col_img:
        # --- GRÁFICO CONCEPTUAL MEJORADO ---
        fig_concept, ax_c = plt.subplots(figsize=(8, 6)) 
        
        # 1. Definimos un rango X amplio para que las curvas caigan a cero visualmente
        x_concept_range = np.linspace(0, 300, 1000)
        
        # 2. Curva de Recuperación (Azul) - La extendemos o usamos la spline global
        # Nota: Usamos la spline global si existe, evaluada en este rango amplio
        if 'cs_global' in locals() or 'cs_global' in globals():
             # Usamos la función spline interpolada pero limitada a 0-100
             y_rec_concept = cs_global(x_concept_range)
             y_rec_concept = np.clip(y_rec_concept, 0, 100)
        else:
             # Fallback si no hay datos cargados aún (curva dummy)
             y_rec_concept = np.interp(x_concept_range, [18, 50, 90, 140, 200, 230], [30, 70, 90, 90, 78, 39])

        ax_c.plot(x_concept_range, y_rec_concept, color='#2980b9', linewidth=3, label='Curva Recuperación (Modelo)')
        
        # 3. Generación de Campanas Simétricas (Gaussianas)
        # Centro: 140 (coincide aprox con el óptimo)
        mu = 140
        
        # Curva "Mala" (Rosada/Ancha): Sigma grande (45) para que se vea bien ancha
        sigma_bad = 45
        y_bad = np.exp(-0.5 * ((x_concept_range - mu) / sigma_bad)**2) * 95 # Altura 95 para que toque casi el tope
        
        # Curva "Buena" (Verde/Angosta): Sigma pequeña (15) para que sea punzante
        sigma_good = 15
        y_good = np.exp(-0.5 * ((x_concept_range - mu) / sigma_good)**2) * 95
        
        # 4. Graficar Áreas (Fill Between)
        # Rosa (Alta Variabilidad)
        ax_c.fill_between(x_concept_range, y_bad, color='#ff7979', alpha=0.4, label='Alta Variabilidad (Inestable)')
        
        # Verde (Baja Variabilidad)
        ax_c.fill_between(x_concept_range, y_good, color='#2ecc71', alpha=0.6, label='Baja Variabilidad (Estable)')
        
        # 5. Estética Final
        ax_c.set_title("Concepto: Estabilidad del Proceso", fontweight='bold', color='#34495e', pad=15)
        ax_c.set_xlabel("Tamaño de Partícula P80 (µm)")
        ax_c.set_ylabel("Escala Relativa")
        ax_c.set_yticks([]) # Ocultar números del eje Y para mantenerlo conceptual
        ax_c.set_xlim(0, 280) # Cortamos visualmente donde las colas ya son casi cero
        ax_c.set_ylim(0, 110)
        
        # Leyenda centrada abajo
        ax_c.legend(loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15), 
                   ncol=3, 
                   frameon=False)
                   
        ax_c.grid(False)
        ax_c.set_facecolor('#fdfdfd')
        
        plt.tight_layout()
        st.pyplot(fig_concept)

# ===================== TAB 2: FLOTATION MODEL =====================
with tab_model:
    st.subheader("Comparativa de Escenarios Operativos")
    st.markdown("Simule y compare el impacto de diferentes estrategias de control de molienda en la recuperación global.")
    
    c_inputs, c_graphs = st.columns([1, 3])
    
    with c_inputs:
        st.markdown("#### 📋 Parámetros de Entrada")
        with st.expander("Escenario 1 (Base)", expanded=True):
            mean1 = st.number_input("Media P80 (µm)", value=200.0, step=1.0, key="m1")
            std1 = st.number_input("Desv. Std (µm)", value=28.0, step=0.1, key="s1")
            
        with st.expander("Escenario 2 (Propuesto)", expanded=True):
            mean2 = st.number_input("Media P80 (µm)", value=180.0, step=1.0, key="m2")
            std2 = st.number_input("Desv. Std (µm)", value=15.0, step=0.1, key="s2")
            
        n_sims_model = st.slider("Iteraciones Monte Carlo", 1000, 10000, 3000, key="nsim")

    with c_graphs:
        g1, g2 = st.columns(2)
        with g1:
            fig1, rec1 = plot_scenario(mean1, std1, cs_global, x_smooth_global, y_smooth_global, df_curve, "ESCENARIO 1 (BASE)", n_sims_model)
            st.pyplot(fig1)
            st.metric("Recuperación Media E1", f"{rec1:.2f}%")
            
        with g2:
            fig2, rec2 = plot_scenario(mean2, std2, cs_global, x_smooth_global, y_smooth_global, df_curve, "ESCENARIO 2 (PROPUESTO)", n_sims_model)
            st.pyplot(fig2)
            delta = rec2 - rec1
            st.metric("Recuperación Media E2", f"{rec2:.2f}%", delta=f"{delta:.2f}%")
            
        # GUARDAR EL DELTA EN SESSION STATE PARA LA TAB 4
        st.session_state['rec_delta_global'] = delta

        if delta > 0:
            st.success(f"✅ Conclusión: El Escenario 2 mejora la recuperación en **+{delta:.2f}%** puntos porcentuales.")
        elif delta < 0:
            st.error(f"🔻 Conclusión: El Escenario 2 reduce la recuperación en **{delta:.2f}%** puntos porcentuales.")

# ===================== TAB 3: SENSITIVITY ANALYSIS =====================
with tab_sens:
    st.subheader("Análisis de Sensibilidad")
    st.markdown("Evalúe cómo varía la recuperación media al mover independientemente el P80 promedio o su desviación estándar.")
    
    sens_tab1, sens_tab2 = st.tabs(["Variando P80 Promedio", "Variando Desviación Estándar"])
    
    # --- Sub-tab 1: Variando Media ---
    with sens_tab1:
        c_s1_in, c_s1_out = st.columns([1, 3])
        with c_s1_in:
            st.markdown("#### Configuración")
            fixed_std = st.number_input("Desv. Std Fija (µm)", value=20.0, step=1.0)
            st.markdown("---")
            p80_min = st.number_input("P80 Promedio Min", value=50.0)
            p80_max = st.number_input("P80 Promedio Max", value=250.0)
            
        with c_s1_out:
            # Cálculo de sensibilidad
            p80_range = np.linspace(p80_min, p80_max, 50)
            rec_results_mean = []
            for p_val in p80_range:
                rec_results_mean.append(calculate_mean_recovery(p_val, fixed_std, cs_global))
            
            fig_s1, ax_s1 = plt.subplots(figsize=(10, 4))
            ax_s1.plot(p80_range, rec_results_mean, color='#8e44ad', linewidth=3)
            ax_s1.set_title(f"Sensibilidad: Recuperación vs P80 Promedio (Std Fija={fixed_std})", fontweight='bold')
            ax_s1.set_xlabel("P80 Promedio (µm)")
            ax_s1.set_ylabel("Recuperación Media (%)")
            ax_s1.grid(True, linestyle='--')
            ax_s1.yaxis.set_major_formatter(mtick.PercentFormatter())
            st.pyplot(fig_s1)
            
    # --- Sub-tab 2: Variando Desviación Estándar ---
    with sens_tab2:
        c_s2_in, c_s2_out = st.columns([1, 3])
        with c_s2_in:
            st.markdown("#### Configuración")
            fixed_mean = st.number_input("P80 Promedio Fijo (µm)", value=140.0, step=1.0)
            st.markdown("---")
            std_min = st.number_input("Desv. Std Min", value=1.0)
            std_max = st.number_input("Desv. Std Max", value=60.0)
            
        with c_s2_out:
            # Cálculo de sensibilidad
            std_range = np.linspace(std_min, std_max, 50)
            rec_results_std = []
            for s_val in std_range:
                rec_results_std.append(calculate_mean_recovery(fixed_mean, s_val, cs_global))
            
            fig_s2, ax_s2 = plt.subplots(figsize=(10, 4))
            ax_s2.plot(std_range, rec_results_std, color='#d35400', linewidth=3)
            ax_s2.set_title(f"Sensibilidad: Recuperación vs Desviación Std (P80 Fijo={fixed_mean})", fontweight='bold')
            ax_s2.set_xlabel("Desviación Estándar (µm)")
            ax_s2.set_ylabel("Recuperación Media (%)")
            ax_s2.grid(True, linestyle='--')
            ax_s2.yaxis.set_major_formatter(mtick.PercentFormatter())
            st.pyplot(fig_s2)

# ===================== TAB 4: ECONOMIC EVALUATION =====================
with tab_econ:
    st.subheader("Evaluación Económica del Cambio")
    st.markdown("Estime el valor financiero del aumento de recuperación calculado en el modelo.")
    
    # Recuperar el delta del estado global
    delta_rec_in = st.session_state.get('rec_delta_global', 0.0)
    
    ce_in, ce_out = st.columns([2, 3])
    
    with ce_in:
        st.markdown("#### 💲 Parámetros Económicos")
        
        # Usamos el valor calculado como default, pero permitimos editarlo
        rec_diff_input = st.number_input("Diferencia de Recuperación (%)", 
                                         value=float(delta_rec_in), 
                                         format="%.2f",
                                         help="Valor traído automáticamente de la pestaña 'Flotation Model'. Puede ser editado.")
        
        st.markdown("---")
        tpd = st.number_input("Tonelaje Diario (tpd)", value=50000, step=1000, format="%d")
        head_grade = st.number_input("Ley de Cabeza Cu (%)", value=0.80, step=0.05, format="%.2f")
        cu_price_lb = st.number_input("Precio Cobre ($/lb)", value=3.80, step=0.1, format="%.2f")
        
        # Conversiones
        price_per_tonne = cu_price_lb * 2204.62 # lb/tonne
        grade_decimal = head_grade / 100.0
        rec_diff_decimal = rec_diff_input / 100.0
        
    with ce_out:
        st.markdown("#### 📊 Resultados Financieros")
        
        # Cálculos
        # Cu Fino Adicional Diario (Tonnes) = TPD * Ley * DeltaRec
        daily_cu_tonnes = tpd * grade_decimal * rec_diff_decimal
        
        # Valor Diario = Tonnes Cu * Precio/Tonne
        daily_value = daily_cu_tonnes * price_per_tonne
        
        # Valor Anual
        annual_value = daily_value * 365
        
        # Visualización con Tarjetas Métricas Personalizadas (usando HTML/CSS simple)
        st.markdown(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 20px;">
            <div class="metric-card" style="flex: 1;">
                <h3 style="margin-bottom: 0; color: #7f8c8d;">Cu Fino Adicional</h3>
                <h2 style="font-size: 2.5rem; color: #2c3e50; margin: 10px 0;">{daily_cu_tonnes:.2f} <span style="font-size: 1.2rem">t/día</span></h2>
            </div>
            <div class="metric-card" style="flex: 1; border-bottom: 5px solid #27ae60;">
                <h3 style="margin-bottom: 0; color: #7f8c8d;">Valor Diario Estimado</h3>
                <h2 style="font-size: 2.5rem; color: #27ae60; margin: 10px 0;">$ {daily_value:,.0f}</h2>
            </div>
        </div>
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <div class="metric-card" style="flex: 2; background-color: #d5f5e3; border: 2px solid #27ae60;">
                <h3 style="margin-bottom: 0; color: #1e8449;">💰 Proyección de Valor Anual (365 días)</h3>
                <h1 style="font-size: 4rem; color: #1e8449; margin: 15px 0;">$ {annual_value:,.0f}</h1>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if daily_value < 0:
            st.warning("⚠️ Atención: La diferencia de recuperación es negativa, lo que resulta en pérdidas económicas proyectadas.")