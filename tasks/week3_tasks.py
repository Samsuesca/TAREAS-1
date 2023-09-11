import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Variable global para la longitud de la lista de opciones
num_options = 3  # Puedes cambiar esto según tus necesidades

# Título de la semana
st.title('Semana 3 - MLEs y Modelos Binomiales')

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



    st.write('Tomando menos valor esperado y por la estimación de varianza asintotica tenemos que:')

    st.latex(r'V[\hat{\beta}_{\text{ML}}] = \left(\sum_{i=1}^N \frac{1}{F(x_i\hat{\beta})(1 - F(x_i\hat{\beta}))} F\'(x_i\hat{\beta})^2x_ix_i^T\right)^{-1}')


def task_2(): 
    st.write('---')   
    # Título de la tarea
    st.subheader('Tarea 2: Estimación de Parámetros con Máxima Verosimilitud')

    ## Introducción
    st.write('En esta tarea, vamos a estimar los parámetros de un modelo de regresión utilizando el método de Máxima Verosimilitud (MLE). '
            'El modelo está definido como sigue:')
    st.latex(r'''
    Y_i = \beta_0 + \beta_1 X_{i2} + \beta_2 X_{i3} + E_i
    ''')
    st.write('Donde:')
    st.latex(r'''
    E_i \sim N(0, \sqrt{\sigma^2})
    ''')
    st.latex(r'\sigma = \frac{e^{\alpha_1 + \alpha_2W_i}}{n}')
    st.latex(r'W_i \sim N(1,1)')
    st.write('Nuestro objetivo es encontrar los valores de los parámetros $\\beta_0$, $\\beta_1$, $\\beta_2$, $\\alpha_1$, y $\\alpha_2$ '
            'que maximizan la verosimilitud de los datos generados.')

    ## Simulación de Datos
    st.write('---')
    st.subheader('**Simulación de Datos**')
    st.write('Primero, generamos datos simulados con los siguientes parámetros:')
    st.write('- $\\beta_0 = 0.5$')
    st.write('- $\\beta_1 = 1$')
    st.write('- $\\beta_2 = -0.7$')
    st.write('- $\\alpha_1 = 0.5$')
    st.write('- $\\alpha_2 = 1$')

    # Código de simulación 
    # Parámetros
    n = 500
    beta = np.array([0.5, 1, -0.7])
    alpha = np.array([0.5, 1])

    # Generación de variables
    np.random.seed(7)
    W_i = np.random.normal(loc=1, scale=1, size=n)
    X_i2 = np.random.normal(loc=0, scale=1, size=n)
    u = np.random.uniform(low=0, high=1, size=n)
    X_i3 = np.random.binomial(n=1, p=u, size=n)
    X_i = np.column_stack((np.ones(n), X_i2, X_i3))
    sigma2 = np.exp(alpha[0] + alpha[1] * W_i) / n
    E_i = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=n)
    Y_i = np.dot(X_i, beta) + E_i

    with st.expander('Ver Código Simulación'):
        st.code('''    n = 500
    beta = np.array([0.5, 1, -0.7])
    alpha = np.array([0.5, 1])

    # Generación de variables
    np.random.seed(7)
    W_i = np.random.normal(loc=1, scale=1, size=n)
    X_i2 = np.random.normal(loc=0, scale=1, size=n)
    u = np.random.uniform(low=0, high=1, size=n)
    X_i3 = np.random.binomial(n=1, p=u, size=n)
    X_i = np.column_stack((np.ones(n), X_i2, X_i3))
    sigma2 = np.exp(alpha[0] + alpha[1] * W_i) / n
    E_i = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=n)
    Y_i = np.dot(X_i, beta) + E_i''')

    ## Estimación por Máxima Verosimilitud (MLE)
    st.write('---')
    st.subheader('**Estimación por Máxima Verosimilitud (MLE)**')
    st.write('Ahora, utilizaremos el método de Máxima Verosimilitud para estimar los parámetros del modelo. '
            'Definimos nuestra función de log-verosimilitud negativa y la maximizamos para encontrar los estimadores MLE.')
    
    # Código de estimación MLE

    def neg_log_likelihood(params):
        beta0, beta1, beta2, alpha1, alpha2 = params
        beta = np.array([beta0, beta1, beta2])
        alpha = np.array([alpha1, alpha2])
        sigma2 = np.exp(alpha[0] + alpha[1] * W_i) / n
        e = Y_i - np.dot(X_i, beta)
        return np.sum(np.log(np.sqrt(2 * np.pi * sigma2)) + (e ** 2) / (2 * sigma2))

    result = minimize(neg_log_likelihood, x0=[0, 0, 0, 0, 0], method='BFGS')
    estimated_params = result.x

    

    # Visualización de los resultados
    st.write('---')
    st.subheader('**Resultados MLE y Errores Estándar**')
    st.write('A continuación, se presentan las estimaciones MLE de los parámetros y sus errores estándar correspondientes:')
    param_names = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$', '$\\alpha_1$', '$\\alpha_2$']
    true_values = [0.5, 1, -0.7, 0.5, 1]

    #calculo de los errores estandar
    std_errors = np.sqrt(np.diag(result.hess_inv))


    #diseño
    cols = st.columns(4) 
    with cols[0]:
        st.write('Parámetro:')
    with cols[1]:
        st.write('Estimación:')
    with cols[2]:
        st.write('Desviación')
    with cols[3]:
        st.write('Real:')
    for i, param_name in enumerate(param_names):
        with cols[0]:
            st.write(f'{param_name}')
        with cols[1]: 
            st.write(f'{estimated_params[i]:.4f}')
        with cols[2]:
            st.write(f'{std_errors[i]:.4f}')
        with cols[3]:
            st.write(f'{true_values[i]}')

    with st.expander('Ver Código Estimación'):
        st.code('''
    def neg_log_likelihood(params):
        beta0, beta1, beta2, alpha1, alpha2 = params
        beta = np.array([beta0, beta1, beta2])
        alpha = np.array([alpha1, alpha2])
        sigma2 = np.exp(alpha[0] + alpha[1] * W_i) / n
        e = Y_i - np.dot(X_i, beta)
        return np.sum(np.log(np.sqrt(2 * np.pi * sigma2)) + (e ** 2) / (2 * sigma2))

    result = minimize(neg_log_likelihood, x0=[0, 0, 0, 0, 0], method='BFGS')
    estimated_params = result.x

    

    # Visualización de los resultados
    st.subheader('**Resultados MLE y Errores Estándar**')
    st.write('A continuación, se presentan las estimaciones MLE de los parámetros y sus errores estándar correspondientes:')
    param_names = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$', '$\\alpha_1$', '$\\alpha_2$']
    true_values = [0.5, 1, -0.7, 0.5, 1]

    std_errors = np.sqrt(np.diag(result.hess_inv))
    std_beta0, std_beta1, std_beta2, std_alpha1, std_alpha2 = std_errors

    cols = st.columns(4) 
    with cols[0]:
        st.write('Parámetro:')
    with cols[1]:
        st.write('Estimación:')
    with cols[2]:
        st.write('Desviación')
    with cols[3]:
        st.write('Real:')
    for i, param_name in enumerate(param_names):
        with cols[0]:
            st.write(f'{param_name}')
        with cols[1]: 
            st.write(f'{estimated_params[i]:.4f}')
        with cols[2]:
            st.write(f'{std_errors[i]:.4f}')
        with cols[3]:
            st.write(f'{true_values[i]}') ''')


    # Visualización de los resultados
    st.subheader('Visualización de los Resultados')
    st.write('A continuación, visualizamos cómo se comparan los valores verdaderos de los parámetros con las estimaciones MLE.')

    fig, ax = plt.subplots(figsize=(8, 6))
    x_labels = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$', '$\\alpha_1$', '$\\alpha_2$']
    plt.bar(x_labels, estimated_params, width=0.4, label='Estimación MLE')
    plt.bar(x_labels, [0.5, 1, -0.7, 0.5, 1], width=0.2, alpha=0.5, label='Valor Verdadero')
    plt.xlabel('Parámetro')
    plt.ylabel('Valor')
    plt.title('Comparación de Estimaciones MLE con Valores Verdaderos')
    plt.legend()
    st.pyplot(fig)


def task_3():
    st.write('---')
    st.subheader('Tarea 3: Estimación de Betas en modelo logit con Máxima Verosimilitud')
    st.write(
        'En esta tarea, se realizará la estimación de los parámetros beta en un modelo logístico utilizando el método de Máxima Verosimilitud (MLE). '
        'El modelo es una regresión logística binaria que relaciona dos variables explicativas con la probabilidad de un evento binario.'
    )

    # Configurar semilla para reproducibilidad
    np.random.seed(123)

    # Número de observaciones
    num_obs = 10000

    # Generación de variables explicativas
    Xi_12 = np.random.normal(size=num_obs)
    Xi_22 = np.random.beta(0.6, 0.4, size=num_obs)

    # Definir los valores verdaderos de beta (para comparación)
    beta_vals = np.array([1, 1, 1])

    # Calcular la probabilidad logística
    def logit_prob(x):
        return np.exp(x) / (1 + np.exp(x))

    # Generar las probabilidades para Yi = 1
    prob_Yi = logit_prob(beta_vals[0] + beta_vals[1] * Xi_12 + beta_vals[2] * Xi_22)

    # Generar la variable dependiente binaria (Yi)
    Yi = np.random.binomial(1, prob_Yi, size=num_obs)

    # Crear un data frame para almacenar los datos
    data_frame = np.column_stack((Yi, Xi_12, Xi_22))

    with st.expander('Ver Código de la Simulación'):
        st.code(''' # Configurar semilla para reproducibilidad
    np.random.seed(123)

    # Número de observaciones
    num_obs = 10000

    # Generación de variables explicativas
    Xi_12 = np.random.normal(size=num_obs)
    Xi_22 = np.random.beta(0.6, 0.4, size=num_obs)

    # Definir los valores verdaderos de beta (para comparación)
    beta_vals = np.array([1, 1, 1])

    # Calcular la probabilidad logística
    def logit_prob(x):
        return np.exp(x) / (1 + np.exp(x))

    # Generar las probabilidades para Yi = 1
    prob_Yi = logit_prob(beta_vals[0] + beta_vals[1] * Xi_12 + beta_vals[2] * Xi_22)

    # Generar la variable dependiente binaria (Yi)
    Yi = np.random.binomial(1, prob_Yi, size=num_obs) ''')


    st.write('---')
    # Estimación por Máxima Verosimilitud (MLE)
    st.header('Estimación por Máxima Verosimilitud (MLE)')

    # Descripción del modelo
    st.write(
        'El modelo de regresión logística binaria se define de la siguiente manera:'
    )
    st.latex(
        r'''
        P(Y_i=1) = \frac{e^{\beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2}}}{1 + e^{\beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2}}}
        '''
    )
    st.write(
        'Donde:'
    )
    st.write(
        '- $Y_i$ es una variable binaria que indica la ocurrencia de un evento.'
    )
    st.write(
        '- $X_{i1}$ y $X_{i2}$ son las variables explicativas que influyen en la probabilidad del evento.'
    )
    st.write(
        '- $\\beta_0$, $\\beta_1$, y $\\beta_2$ son los parámetros a estimar que determinan la forma de la curva logística.'
    )

    # Función de verosimilitud negativa
    def neg_log_likelihood(beta):
        eta = beta[0] + beta[1] * Xi_12 + beta[2] * Xi_22
        prob_Yi = logit_prob(eta)
        log_likelihood = np.sum(Yi * np.log(prob_Yi) + (1 - Yi) * np.log(1 - prob_Yi))
        return -log_likelihood

    # Estimación de los parámetros con MLE
    result = minimize(neg_log_likelihood, x0=[0, 0, 0], method='BFGS')
    estimated_params = result.x
    param_names = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$']
    true_values = [1,1,1]
    #calculo de los errores estandar
    std_errors = np.sqrt(np.diag(result.hess_inv))

    

    st.write('---')
    # Resultados y conclusiones
    st.header('Resultados MLE de los Parámetros Beta')
    cols = st.columns(4) 
    with cols[0]:
        st.write('Parámetro:')
    with cols[1]:
        st.write('Estimación:')
    with cols[2]:
        st.write('Desviación')
    with cols[3]:
        st.write('Real:')
    for i, param_name in enumerate(param_names):
        with cols[0]:
            st.write(f'{param_name}')
        with cols[1]: 
            st.write(f'{estimated_params[i]:.4f}')
        with cols[2]:
            st.write(f'{std_errors[i]:.4f}')
        with cols[3]:
            st.write(f'{true_values[i]}')

    with st.expander('Ver código Estimación:'):
        st.code('''    # Función de verosimilitud negativa
    def neg_log_likelihood(beta):
        eta = beta[0] + beta[1] * Xi_12 + beta[2] * Xi_22
        prob_Yi = logit_prob(eta)
        log_likelihood = np.sum(Yi * np.log(prob_Yi) + (1 - Yi) * np.log(1 - prob_Yi))
        return -log_likelihood

    # Estimación de los parámetros con MLE
    result = minimize(neg_log_likelihood, x0=[0, 0, 0], method='BFGS')
    estimated_params = result.x
    param_names = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$']
    true_values = [1,1,1]
    #calculo de los errores estandar
    std_errors = np.sqrt(np.diag(result.hess_inv)) ''')

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
