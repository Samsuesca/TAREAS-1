import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


st.set_page_config(page_title='S6',
                    page_icon=':book:',
                    layout='wide')


# Título de la semana
st.title('Semana 6 - Modelos Truncados y de Sesgo de Selección')


# Funciones para las tareas específicas
def task_1():
    st.write('---')
    st.subheader('**Tarea 1: Simulación de modelo truncado hacia la derecha:**')
    # Agregar contenido de la tarea 1

def task_2():
    st.write('---')
    st.subheader('**Tarea 2: Demostración de momentos Truncados de una normal Estándar:**')
        # Descripción introductoria
    st.write("En el modelo Tobit, el error de regresión ε sigue una distribución normal, y mostraremos los siguientes resultados derivados:")

    # Proposición 16.1 - Momentos Truncados de la Normal Estándar
    st.write("**Momentos Truncados de la Normal Estándar**")
    st.write("Supongamos que z sigue una distribución N(0, 1). Entonces, los momentos truncados a la izquierda de z son:"
             "Supongamos que z sigue una distribución \(N(0, 1)\). Entonces, los momentos truncados a la izquierda de \(z\) son:")
   
    st.latex(r""" E[z|z > c] = \frac{\phi(c)}{1 - \Phi(c)}""")
    st.latex(r"""E[z^2|z > c] = 1 + \frac{c\phi(c)}{1 - \Phi(c)}""")
    st.latex(r"""V[z|z > c] = 1 + \frac{c\phi(c)}{1 - \Phi(c)} - \left(\frac{\phi(c)}{1 - \Phi(c)}\right)^2
    """)
    st.write('---')
    #demostracion:
    st.write('**Deducción**')
    # Descripción

    
    st.markdown('Aquí se muestran los pasos para la deducción de los momentos truncados.')

    st.latex(r'\text{Dadas las condiciones expuestas y } \Phi(\beta)=1 \text{ and } \phi(\beta)=0.')


    st.write('$E(z|z>c)$')
    st.latex(r'E(z|z>c) = 0 + \frac{0-\phi(c)}{1-\Phi(c)}')
    st.latex(r'E(z|z>c) = \frac{\phi(c)}{1-\Phi(c)}')

    st.write('$Var(z|z>c)$')
    st.latex(r'Var(z|z>c) = 1 \cdot \left[ 1 - \frac{0-c\phi(c)}{1-\Phi(c)} - \left(\frac{0-\phi(c)}{1-\Phi(c)}\right)^2\right]')
    st.latex(r'Var(z|z>c) = 1 + \frac{c\phi(c)}{1-\Phi(c)} - \left(\frac{\phi(c)}{1-\Phi(c)}\right)^2')


    st.write('$E(z²|z>c)$')
    st.latex(r'E(z^2|z>c) = \left(\frac{\phi(c)}{1-\Phi(c)}\right)^2 + 1 + \frac{c\phi(c)}{1-\Phi(c)} - \left(\frac{\phi(c)}{1-\Phi(c)}\right)^2')
    st.latex(r''' E(z^2|z>c) = 1 + \frac{c\phi(c)}{1-\Phi(c)} ''')



    # Explicación y gráficos
    st.write('---')
    st.subheader("Explicación y Gráficos:")
    st.write("Consideremos la truncación de z ~ N(0, 1) desde abajo en c, donde c varía de -2 a 2. La curva más baja es la densidad normal estándar φ(c) evaluada en c. La curva del medio es la función de distribución acumulativa (CDF) normal estándar Φ(c) evaluada en c, que da la probabilidad de truncación cuando la truncación es en c.")
    st.write("La curva superior muestra la media truncada E[z|z > c] = φ(c) / [1 - Φ(c)]. Como era de esperar, esta media es cercana a E[z] = 0 para c = -2, ya que entonces hay poca truncación, y E[z|z > c] > c.")
    st.write("Lo que no se espera a priori es que φ(c) / [1 - Φ(c)] sea aproximadamente lineal, especialmente para c > 0.")

    # Gráfico de las curvas en Matplotlib
    c_values = np.linspace(-2, 2, 400)
    phi_c = stats.norm.pdf(c_values)
    Phi_c = stats.norm.cdf(c_values)
    truncated_mean = phi_c / (1 - Phi_c)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c_values, phi_c, label='Densidad Normal φ(c)')
    ax.plot(c_values, Phi_c, label='CDF Normal Φ(c)')
    ax.plot(c_values, truncated_mean, label='Media Truncada E[z|z > c]')
    ax.set_xlabel('c')
    ax.set_ylabel('Valor')
    ax.set_title('Momentos Truncados de la Normal Estándar')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.legend()
    st.pyplot(fig)

    


 

# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    'Tarea 1': task_1,
    'Tarea 2': task_2
    # Agregar funciones para las demás tareas
}

st.write('---')
selected = option_menu('Selección de Tarea', options=list(task_functions.keys()), 
    icons=['book' for i in task_functions.keys()], default_index=0,orientation="horizontal")

# Llama a la función de tarea seleccionada
if selected in task_functions:
    task_functions[selected]()
else:
    st.write('Selecciona una tarea válida en el menú de opciones.')
