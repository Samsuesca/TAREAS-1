import streamlit as st

# Variable global para la longitud de la lista de opciones
num_options = 2  # Puedes cambiar esto según tus necesidades

# Título de la semana
st.title('Semana 3 - ')

# Menú de opciones
option = st.sidebar.selectbox('Selecciona una tarea:', range(1, num_options + 1))

# Funciones para las tareas específicas
def task_1():
    st.subheader('Tarea 1: Varianza del Estimador de MV')

    # Descripción
    st.markdown('En esta tarea, demostraremos la varianza del estimador de Máxima Verosimilitud (MV) para un proceso que sigue una distribución Bernoulli. '
                 'Supongamos que tenemos una muestra de N observaciones, donde cada observación es una variable binaria con probabilidad de éxito $p_i=F(x_i^{T}\\beta)$.') 

    # Fórmula de la función de verosimilitud
    st.markdown("La función de verosimilitud para el modelo es:")
    st.latex(r'V_N(\beta) = \prod_{i=1}^N F(x_i^T\beta)^{y_i}(1 - F(x_i^T\beta))^{1 - y_i}')

    st.write('Y sacando logaritmo en ambos lados tenemos la Log-Likelihood Function:')
    st.latex(r'L_N(\beta)=\ln V_N(\beta) = \sum_{i=1}^N \left( y_i \ln(F(x_i^T\beta)) + (1 - y_i) \ln(1 - F(x_i^T\beta)) \right)')

    st.write('Maximizando la funcion de log-likelihood respecto a el parámetro $\\beta$')
    st.latex(r'\frac{\partial L_N(\beta)}{\partial \beta} = \sum_{i=1}^N \left( y_i \frac{1}{F(x_i^T\beta)} \cdot \frac{\partial}{\partial \beta} F(x_i^T\beta) + (1 - y_i) \frac{1}{1 - F(x_i^T\beta)} \cdot \frac{\partial}{\partial \beta} [1 - F(x_i^T\beta)]\right)')
    st.latex(r'\frac{\partial L_N(\beta)}{\partial \beta} = \sum_{i=1}^N \left( \frac{y_i}{F(x_i^T\beta)} \cdot F\'(x_i^T\beta) x_i - \frac{1 - y_i}{1 - F(x_i^T\beta)} \cdot F\'(x_i^T\beta) x_i \right)')

    st.write('Haciendo la expresion en una sola fracción:')
    st.latex(r'\frac{\partial L_N(\beta)}{\partial \beta} = \sum_{i=1}^N \frac{(y_i - F(x_i^T\beta))\cdot F\'(x_i^T\beta) x_i}{F(x_i^T\beta)(1 - F(x_i^T\beta))} ')

    st.write('Tomando la segunda derivada:')
    st.latex(r'\frac{\partial^2 L_N(\beta)}{\partial \beta \partial \beta^T} = \sum_{i=1}^N \frac{(y_i - F(x_i^T\beta))}{F(x_i^T\beta)(1 - F(x_i^T\beta))} \cdot (F\'(x_i^T\beta))^2 x_ix_i^T - \frac{1}{F(x_i^T\beta)(1 - F(x_i^T\beta))} \cdot (F\'(x_i^T\beta))^2 x_ix_i^T')



    st.write('Tomando el meos valor esperado y por la estimación de varianza asintotica tenemos que:')

    st.latex(r'V[\hat{\beta}_{\text{ML}}] = \left(\sum_{i=1}^N \frac{1}{F(x_i\hat{\beta})(1 - F(x_i\hat{\beta}))} F\'(x_i\hat{\beta})^2x_ix_i^T\right)^{-1}')


def task_2():
    st.subheader('Tarea 2: Descripción de la tarea 2')
    # Agregar contenido de la tarea 2

def task_3():
    st.subheader('Tarea 3: Descripción de la tarea 2')
    # Agregar contenido de la tarea 2

# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    1: task_1,
    2: task_2,
    3: task_3
    # Agregar funciones para las demás tareas
}

# Llama a la función de tarea seleccionada
if option in task_functions:
    task_functions[option]()
else:
    st.write('Selecciona una tarea válida en el menú de opciones.')
