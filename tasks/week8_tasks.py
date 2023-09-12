import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='S8',
                    page_icon=':book:',
                    layout='wide')


# Título de la semana
st.title('Semana 8 - Método de los Momentos (MM) y Método Generalizado de los Momentos (GMM):')


# Funciones para las tareas específicas
def task_1():
    st.write('---')
    st.subheader('Tarea 1: Cálculo de $β^{GMM}$')

    st.write('En esta tarea encontraremos el estimador de Método Generalizado de los Momentos (GMM) de un modelo lineal con variable instrumental partiendo de la siguiente forma cuadrática:')
    
    st.latex(r'Q_N(\beta) = \left[\frac{1}{N}(y - X\beta)^T Z\right]W_N \left[\frac{1}{N} Z^T (y - X\beta)\right]')
    st.write('Considerando la forma general:')
    st.latex(r'Q_N(\theta) = g_N^T(\theta)W_N g_N(\theta)')
    st.write('y')
    st.latex(r'G_N(\theta)W_N g_N(\theta) = 0')
    st.write('donde')
    st.latex(r'G_N(\theta)=\frac{dg_N(\theta)}{d\theta^T}')
    st.write('Para nuestro caso')
    st.latex(r'g_N(\beta) = \frac{1}{N}Z^T(y - X \beta)')
    st.write('Y entonces:')
    st.latex(r'G_N(\beta)^T=-\frac{X^TZ}{N}')
    # st.latex(r'')
    # Derivación
    st.write('Tenemos entonces que la derivada es')
    st.latex(r'\frac{dQ_N(\beta)}{d\beta} = -\frac{1}{N^2}X^TZW_NZ^T(y - X\beta)')
   

    # Igualar a 0
    st.write('Al Igualar a 0:')
    st.latex(r'X^TZW_NZ^T(y - X\beta) = 0')

    # Despejar β
    st.write('Despejando para $\\beta$:')
    st.latex(r'X^TZW_NZ^Ty - X^TZW_NZ^TX\beta = 0')
    st.latex(r'X^TZW_NZ^TX\beta = X^TZW_NZ^Ty')
    st.latex(r'\beta^{\text{GMM}} = (X^TZW_NZ^TX)^{-1}X^TZW_NZ^Ty')


# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    'Tarea 1': task_1
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
