import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title='S5',
                    page_icon=':book:',
                    layout='wide')


# Título de la semana
st.title('Semana 5 - Modelos de Conteo')


# Funciones para las tareas específicas
def task_1():
    st.write('---')
    # Título de la tarea
    st.subheader('Tarea 1: Modelo Binomial Negativo')

    # Descripción del modelo
    st.write('El Modelo Binomial Negativo (NBM) es un modelo de mezcla continua con una función de densidad marginal de $y$ definida como:')
    st.latex(r'''
    h(y|\mu,\alpha) = \int f(y|\mu,v)g(v|\alpha) dv
    ''')

    st.write('Donde se asume que $\\mu$ es una función determinista de $x$, $v$ es una variable gamma positiva aleatoria con media 1 y varianza $\\delta^{-1}$, $y$  sigue una distribución de Poisson, con su función de densidad definida como:')
    st.latex(r'''
    f(y|\lambda) = \frac{\lambda^y e^{-\lambda}}{y!}
    ''')
    st.latex(r'''
    \lambda = \mu v
    ''')
    st.write('Es importante destacar que $\\alpha$ es el parámetro desconocido del modelo de mezcla.')

    
    st.write('Dado que v sigue una distribución gamma con media 1 y varianza $\\delta^{-1}$,, utilizando la configuración de escala-tasa, obtenemos:')
    st.latex(r'''
    g(v) = \frac{v^{\delta-1} e^{-\delta v} \delta^\delta}{\Gamma(\delta)}
    ''')

    st.write('Por lo tanto, podemos calcular $h(y|μ,α)$ como sigue:')
    st.latex(r'''
    h(y|\mu,\alpha) = \int_0^\infty f(y|\mu v)g(v|\alpha) dv
    ''')
    st.latex(r'''
    = \int_0^\infty \frac{(\mu v)^y e^{-\mu v}}{y!} \frac{v^{\delta-1} e^{-\delta v} \delta^\delta}{\Gamma(\delta)} dv
    ''')
    st.latex(r'''
    = \frac{\mu^y \delta^\delta}{\Gamma(\delta) y!} \int_0^\infty v^{y+\delta-1} e^{-(\mu+\delta)v} dv
    ''')

    st.write('Utilizando propiedades de la función gamma, la integral se simplifica, y obtenemos:')
    st.latex(r'''
    h(y|\mu,\alpha) = \frac{\mu^y \delta^\delta \Gamma(y+\delta)}{\Gamma(\delta) y! (\mu+\delta)^{y+\delta}}
    ''')

    st.write('Esto confirma que el modelo binomial negativo tiene la densidad proporcionada.')


def task_2():
    st.write('---')
    st.subheader('Tarea 2: Modelo Truncado hacia la Derecha')

    # Descripción de la tarea
    st.write('En un modelo con presencia de truncamiento desde la derecha, la función de densidad de probabilidad de la variable de respuesta observada seria:')
    st.latex(r'''
    f(y) = f^*(y|y<U) = \frac{f^*(y)}{Pr(y|y<U)} = \frac{f^*(y)}{F^*(U)}
    ''')

    st.write('En consecuencia, la log-verosimilitud es:')
    st.latex(r'''
    \mathcal{L}(\theta) = \sum_{i=1}^{N} \ln[f^*(y_i|x_i,\theta)] - \ln[F^*(U_i|x_i,\theta)]
    ''')
def task_3():
    st.write('---')
    st.subheader('Tarea: Modelo Censurado hacia la Derecha')

    # Introducción a la tarea
    st.write('En esta tarea, exploraremos el modelo censurado hacia la derecha y su log-verosimilitud')

    # Definición de la censura desde arriba
    st.write('Comencemos recordando que tenemos censura desde arriba cuando:')
    st.latex(r'''
    y = \begin{cases} y^* & \text{si } y^* < U, \\ U & \text{si } y^* \geq U. \end{cases}
    ''')

    # Densidad condicional
    st.write('En este escenario, la densidad condicional se define como:')
    st.latex(r'''
    f(y|x) = \begin{cases} f^*(y|x) \cdot (1 - F^*(U|x)) & \text{si } y < U, \\ 0 & \text{si } y = U. \end{cases}
    ''')

    # Variable indicadora
    st.write('Aquí, $d$ es una variable indicadora que toma el valor de 1 si $y < U$ y 0 si $y = U$.')
    st.latex(r'''
    d = \begin{cases} 1 & \text{si } y < U, \\ 0 & \text{si } y = U. \end{cases}
    ''')

    # Densidad condicional con variable indicadora
    st.write('Entonces, la densidad condicional se puede expresar como:')
    st.latex(r'''
    f(y|x) = [f^*(y|x)]^d \cdot [1 - F^*(U|x)]^{1-d}
    ''')

    # Log-verosimilitud
    st.write('La log-verosimilitud correspondiente a este escenario es la siguiente:')
    st.latex(r'''
    \mathcal{L}(\theta) = \sum_{i=1}^N d_i \left[ y_i \ln(\mu_i) - \mu_i - \ln(y_i!) \right] (1 - d_i) \ln\left[1 - \sum_{j=0}^{U-1} e^{-\mu_i} \frac{\mu_i^j}{j!}\right]
    ''')



# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    'Tarea 1': task_1,
    'Tarea 2': task_2,
    'Tarea 3': task_3
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
