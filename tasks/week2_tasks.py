import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

st.set_page_config(page_title='S2',
                    page_icon=':book:',
                    layout='wide')


# Título de la semana
st.title('Semana 2 - Introducción a Máxima Verosimilitud')


# Funciones para las tareas específicas
def task_1():
    st.write('---')
    # Título de la tarea
    st.header('Tarea 1: Explorar las Diferencias entre Estimadores de $\sigma^2$ en Regresión Lineal')

    ## Introducción
    st.subheader('**Introducción**')
    st.write('En esta tarea se explora las diferencias entre el estimador de MCO y MLE para la varianza del error $\sigma^2$ en un modelo de regresión lineal. '
            'Ambos estimadores de σ² en el contexto de la regresión lineal múltiple tienen características distintivas '
            'que son esenciales para comprender su desempeño y aplicaciones. Exploraremos en detalle '
            'las diferencias entre el estimador de σ² con MCO y el estimador MLE.')

    ## Estimador de σ² con MCO
    st.write('---')
    st.subheader('**Estimador de σ² con MCO**')
    st.write('El estimador de σ² con MCO se basa en los residuos de la regresión. Su fórmula es:')
    st.latex(r"\hat{\sigma}^2 = \frac{1}{n-k} \sum_{i=1}^{n} \hat{u}_i^2")

    ### Características
    st.subheader('Características')
    st.write('1. **Insesgado**: Este estimador es insesgado, lo que significa que su valor esperado es igual a la verdadera varianza, '
            'es decir, $E(\hat{\sigma}^2) = \sigma^2$.')

    st.write('2. **Consistente**: Conforme el tamaño de la muestra ($n$) aumenta, el estimador de σ² con MCO converge en probabilidad al '
            'valor verdadero de la varianza ($\sigma^2$).')

    st.write('3. **Eficiente**: Dentro de la clase de estimadores lineales insesgados, el estimador de MCO tiene la varianza más pequeña, '
            'lo que lo hace eficiente en el sentido del Criterio de Eficiencia de BLUE.')


    st.write('---')
    ## Estimador de σ² con MLE
    st.subheader('**Estimador de σ² con MLE**')
    st.write('El estimador MLE de σ² se basa en la Máxima Verosimilitud y se define como:')
    st.latex(r"\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^{n} \hat{u}_i^2")

    ### Características
    st.subheader('Características')
    st.write('1. **Sesgado**: El estimador MLE es sesgado para muestras finitas, es decir, $E[\hat{\sigma}^2_{MLE}] < \sigma^2$. '
            'Sin embargo, a medida que $n$ tiende a infinito, se vuelve asintóticamente insesgado, lo que significa que '
            '$E[\hat{\sigma}^2_{MLE}] = \sigma^2$.')

    st.write('2. **Consistente**: El estimador MLE de σ² es consistente, ya que converge en probabilidad al valor verdadero de '
            'la varianza a medida que $n$ aumenta.')

    st.write('3. **Eficiencia**: En condiciones regulares, el MLE es asintóticamente eficiente, lo que significa que tiene la '
            'menor varianza dentro de la clase de estimadores consistentes cuando la muestra tiende a infinito.')

   

    ## Espacio para Simulaciones y Visualizaciones
    st.subheader('**Espacio para Simulaciones y Visualizaciones**')

    def simular_RL(N,K,sigma,trues):
        np.random.seed(110)
        # Generar datos simulados para la regresión lineal
        n = N # Tamaño de la muestra
        k = K    # Número de variables predictoras
        true_sigma_sq = sigma  # Valor verdadero de la varianza

        # Generar variables predictoras aleatorias
        X = np.random.randn(n, k)

        # Generar el vector de errores aleatorios
        epsilon = np.random.randn(n) * np.sqrt(true_sigma_sq)

        # Generar la variable de respuesta Y
        true_beta = trues  # Coeficientes verdaderos
        Y = np.dot(X, true_beta) + epsilon

        # Calcular los residuos para MCO y MLE
        Y_hat_mco = np.dot(X, np.linalg.lstsq(X, Y, rcond=None)[0])
        residuals_mco = Y - Y_hat_mco

        Y_hat_mle = np.dot(X, np.linalg.pinv(X).dot(Y))
        residuals_mle = Y - Y_hat_mle
        
        var_mco = np.sum(residuals_mco**2)*(1/(n-k))
        var_mle = np.sum(residuals_mle**2)*(1/n)
        return var_mco, var_mle
    
    # k = 4
    # sigma=2.3
    # trues = np.array([2.0,-1.0,0.3,0.2])
    # sample_sizes = [10,20,50,100,200,500,800,1000,2500,4000,5000]
    # vars_mco = [simular_RL(n,k,sigma=sigma,trues=trues)[0] for n in sample_sizes]
    # vars_mle = [simular_RL(n,k,sigma=sigma,trues=trues)[1] for n in sample_sizes]

    # # Visualizar histogramas de los residuos para MCO y MLE
    # st.subheader('Tendencia a converger en una simulación')
    # st.write('Se simula el caso de una regresión con 4 regresores y $\\sigma^2 = 2.3$')

    # fig, axes = plt.subplots(figsize=(12, 5))
    # axes.plot(sample_sizes,vars_mco,label='MCO')
    # axes.plot(sample_sizes,vars_mle,label='MLE')
    # axes.axhline(2.3, color='red', linestyle='--', label='Valor Verdadero')
    # axes.set_title('Varianza Residuos')
    # axes.set_xlabel('N')
    # axes.set_ylabel('$\sigma^2$')
    # axes.legend()
    # fig.savefig('images/s2t1f1.png')
    st.image('images/s2t1f1.png')

    st.write('Podemos ver que con menos muestras el estimador de MCO es más cercano al valor poblacional, pero'
             'a medida que comienza aumentar el tamaño de la muestra el MLE tiende a acercarse más rápido al parámetro real.'
             'Para explorar si sucede esto en la mayoria de casos podemos ver muchas simulaciones para muestras dadas')
    
    with st.expander('Ver Código Gráfica y Simulación'):
        st.code('''def simular_RL(N,K,sigma,trues):
        np.random.seed(110)
        # Generar datos simulados para la regresión lineal
        n = N # Tamaño de la muestra
        k = K    # Número de variables predictoras
        true_sigma_sq = sigma  # Valor verdadero de la varianza

        # Generar variables predictoras aleatorias
        X = np.random.randn(n, k)

        # Generar el vector de errores aleatorios
        epsilon = np.random.randn(n) * np.sqrt(true_sigma_sq)

        # Generar la variable de respuesta Y
        true_beta = trues  # Coeficientes verdaderos
        Y = np.dot(X, true_beta) + epsilon

        # Calcular los residuos para MCO y MLE
        Y_hat_mco = np.dot(X, np.linalg.lstsq(X, Y, rcond=None)[0])
        residuals_mco = Y - Y_hat_mco

        Y_hat_mle = np.dot(X, np.linalg.pinv(X).dot(Y))
        residuals_mle = Y - Y_hat_mle
        
        var_mco = np.sum(residuals_mco**2)*(1/(n-k))
        var_mle = np.sum(residuals_mle**2)*(1/n)
        return var_mco, var_mle
    
    k = 4
    sigma=2.3
    trues = np.array([2.0,-1.0,0.3,0.2])
    sample_sizes = [10,20,50,100,200,500,800,1000,2500,4000,5000]
    vars_mco = [simular_RL(n,k,sigma=sigma,trues=trues)[0] for n in sample_sizes]
    vars_mle = [simular_RL(n,k,sigma=sigma,trues=trues)[1] for n in sample_sizes]

    fig, axes = plt.subplots(figsize=(12, 5))
    axes.plot(sample_sizes,vars_mco,label='MCO')
    axes.plot(sample_sizes,vars_mle,label='MLE')
    axes.axhline(2.3, color='red', linestyle='--', label='Valor Verdadero')
    axes.set_title('Varianza Residuos')
    axes.set_xlabel('N')
    axes.set_ylabel('$\sigma^2$')
    axes.legend()''')


    st.write('---')
    # Realizar simulaciones para ver la consistencia de los estimadores de σ²
    st.subheader('Simulación de Consistencia')
    st.write('A continuación, realizaremos una simulación para demostrar la consistencia de los estimadores de σ². '
            'A medida que aumentamos el tamaño de la muestra, veremos cómo los estimadores convergen al valor verdadero '
            'de la varianza $\sigma^2$.')
    
    # sample_sizes = [50, 100, 200, 400, 600,1000]  # Tamaños de muestra
    # num_simulations = 500

    # estimated_sigmas_mco = np.zeros((len(sample_sizes), num_simulations))
    # estimated_sigmas_mle = np.zeros((len(sample_sizes), num_simulations))

    # for i, n in enumerate(sample_sizes):
    #     for j in range(num_simulations):
    #         # Generar datos de muestra con el nuevo tamaño
    #         X_sample = np.random.randn(n, k)
    #         epsilon_sample = np.random.randn(n) * np.sqrt(sigma)
    #         Y_sample = np.dot(X_sample, trues) + epsilon_sample

    #         # Calcular estimadores de σ² con MCO y MLE
    #         Y_hat_mco_sample = np.dot(X_sample, np.linalg.lstsq(X_sample, Y_sample, rcond=None)[0])
    #         residuals_mco_sample = Y_sample - Y_hat_mco_sample
    #         estimated_sigmas_mco[i, j] = np.mean(residuals_mco_sample ** 2)

    #         Y_hat_mle_sample = np.dot(X_sample, np.linalg.pinv(X_sample).dot(Y_sample))
    #         residuals_mle_sample = Y_sample - Y_hat_mle_sample
    #         estimated_sigmas_mle[i, j] = np.mean(residuals_mle_sample ** 2)

    # # Visualizar la convergencia de los estimadores
    

    # fig, ax = plt.subplots(figsize=(10, 6))
    # for i, n in enumerate(sample_sizes):
    #     if i == 0:
    #         ax.plot([n] * num_simulations, estimated_sigmas_mco[i, :], 'bo', alpha=0.4, color='blue',label='MCO')
    #         ax.plot([n] * num_simulations, estimated_sigmas_mle[i, :], 'ro', alpha=0.1, color='green',label='MLE')
    #     else:
    #         ax.plot([n] * num_simulations, estimated_sigmas_mco[i, :], 'bo', alpha=0.4, color='blue')
    #         ax.plot([n] * num_simulations, estimated_sigmas_mle[i, :], 'ro', alpha=0.1, color='green')

    # ax.axhline(sigma, color='red', linestyle='--', label='$\sigma^2$ Verdadera')
    # ax.set_xlabel('Tamaño de la Muestra (n)')
    # ax.set_ylabel('Estimador de $\sigma^2$')
    # ax.legend()
    # fig.savefig('images/s2t1f2.png')
    st.image('images/s2t1f2.png')
    st.write('En la gráfica anterior, observamos cómo los estimadores de σ² con MCO y MLE convergen al valor verdadero '
            'de la varianza $\sigma^2$ a medida que aumenta el tamaño de la muestra. Sin embargo, notamos que con muestras pequeñas, MLE tiende a subestimar'
            'el valor poblacional, pero también, las diferentes simulaciones muestran una desviación menor que en caso de MCO')
    with st.expander('Ver Código Simulación y Gráfica:'):
        st.code('''sample_sizes = [50, 100, 200, 400, 600,1000]  # Tamaños de muestra
    num_simulations = 500

    estimated_sigmas_mco = np.zeros((len(sample_sizes), num_simulations))
    estimated_sigmas_mle = np.zeros((len(sample_sizes), num_simulations))

    for i, n in enumerate(sample_sizes):
        for j in range(num_simulations):
            # Generar datos de muestra con el nuevo tamaño
            X_sample = np.random.randn(n, k)
            epsilon_sample = np.random.randn(n) * np.sqrt(sigma)
            Y_sample = np.dot(X_sample, trues) + epsilon_sample

            # Calcular estimadores de σ² con MCO y MLE
            Y_hat_mco_sample = np.dot(X_sample, np.linalg.lstsq(X_sample, Y_sample, rcond=None)[0])
            residuals_mco_sample = Y_sample - Y_hat_mco_sample
            estimated_sigmas_mco[i, j] = np.mean(residuals_mco_sample ** 2)

            Y_hat_mle_sample = np.dot(X_sample, np.linalg.pinv(X_sample).dot(Y_sample))
            residuals_mle_sample = Y_sample - Y_hat_mle_sample
            estimated_sigmas_mle[i, j] = np.mean(residuals_mle_sample ** 2)

    # Visualizar la convergencia de los estimadores
    

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, n in enumerate(sample_sizes):
        if i == 0:
            ax.plot([n] * num_simulations, estimated_sigmas_mco[i, :], 'bo', alpha=0.4, color='blue',label='MCO')
            ax.plot([n] * num_simulations, estimated_sigmas_mle[i, :], 'ro', alpha=0.1, color='green',label='MLE')
        else:
            ax.plot([n] * num_simulations, estimated_sigmas_mco[i, :], 'bo', alpha=0.4, color='blue')
            ax.plot([n] * num_simulations, estimated_sigmas_mle[i, :], 'ro', alpha=0.1, color='green')

    ax.axhline(sigma, color='red', linestyle='--', label='$\sigma^2$ Verdadera')
    ax.set_xlabel('Tamaño de la Muestra (n)')
    ax.set_ylabel('Estimador de $\sigma^2$')
    ax.legend() ''')

def task_2():
    st.write('---')
    st.subheader('Tarea 2: Segunda Condición de Regularidad en Máxima Verosimilitud')
    
    # Descripción de la Matriz de Información de Fisher
    st.markdown("La matriz de varianzas y covarianzas de $x$, evaluada en el verdadero valor del parámetro $\\theta$, también se conoce como la matriz de información de Fisher.")
    st.markdown("Esta matriz, denotada como $I(\\theta;x)$, se define como:")
    st.latex(r'I(\theta;x) = \text{Var}\left[\frac{\partial \ln f(x;\theta)}{\partial \theta}\right] = \mathbb{E}\left[\frac{\partial \ln f(x;\theta)}{\partial \theta} \frac{\partial \ln f(x;\theta)}{\partial \theta^T}\right]')

    # Explicación de la Expansión de la Derivada
    st.markdown("Aplicando la definición integral de valor esperado, intercambiando derivadas por integrales y expandiendo tenemos que")
    st.latex(r'\int \left[\frac{\partial^2 \ln f(x;\theta)}{\partial \theta \partial \theta^T} f(x;\theta) + \frac{\partial \ln f(x;\theta)}{\partial \theta} \frac{\partial f(x;\theta)}{\partial \theta^T} \right] dx  =  0')

    st.markdown('De la expresión anterior, si volvemos a la notacion de valor esperado, tenemos:')
    st.latex(r'\mathbb{E}\left[\frac{\partial^2 \ln f(x;\theta)}{\partial \theta \partial \theta^T}\right] + \mathbb{E}\left[\frac{\partial \ln f(x;\theta)}{\partial \theta} \frac{\partial \ln f(x;\theta)}{\partial \theta^T}\right] = \mathbb{E}\left[\frac{\partial^2 \ln f(x;\theta)}{\partial \theta \partial \theta^T}\right] + I(\theta;x) =0')

    st.markdown('Al despejar, deducimos la segunda condición de regularidad')

    st.latex(r'I(\theta;x) = \mathbb{E}\left[-\frac{\partial^2 \ln f(x;\theta)}{\partial \theta \partial \theta^T}\right]')

# Define un diccionario para mapear la selección a la función de tarea correspondiente
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