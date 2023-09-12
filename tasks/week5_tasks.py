import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma, gammaln



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

def task_4():
    

    st.subheader("Binomial Negativa: Ejercicio de simulación y estimación")

    true_beta = [0.3, -0.2, 1.4]
    true_delta = 2
    # Función para generar datos simulados
    def generate_simulated_data(n,true_beta,true_delta):
        np.random.seed(0)
        beta = np.array(true_beta)
        x1 = np.random.normal(size=n)
        x2 = np.random.binomial(1, 0.5, size=n)
        X = np.column_stack((np.ones(n), x1, x2))
        mu = np.exp(np.dot(X, beta))
        delta_gm = true_delta
        p = delta_gm / (mu + delta_gm)
        mu_nb = delta_gm * (1 - p) / p
        theta_nb = delta_gm
        y = np.zeros(n)
        for i in range(n):
            y[i] = np.random.negative_binomial(theta_nb, theta_nb / (mu_nb[i] + theta_nb))
        return X, y


    # Función para ajustar el modelo de regresión binomial negativa
    def negative_binomial_regression(X, y):
        n, m = X.shape
        params = np.ones(m + 1)

        def neg_log_likelihood(params):
            delta = params[0]
            beta = params[1:]
            mu = np.exp(np.dot(X, beta))
            p = delta / (mu + delta)
            logL = 0
            for i in range(n):
                bincof = gamma(delta + y[i]) / (gamma(delta) * gamma(y[i] + 1))
                logL += gammaln(delta + y[i]) - (gammaln(delta) + gammaln(y[i] + 1)) + delta * np.log(p[i]) + y[i] * np.log(1 - p[i])
            return -logL

        result = minimize(neg_log_likelihood, params, method='BFGS')
        std_errors = np.sqrt(np.diag(result.hess_inv))
        delta_hat= result.x[0]
        beta_hat = result.x[1:]
        return delta_hat, beta_hat, std_errors

    # Simular datos
    n = 1000
    X, y = generate_simulated_data(n,true_beta,true_delta)
   

    # Ajustar el modelo
    delta_hat, beta_hat, std_errors = negative_binomial_regression(X, y)

    st.write('---')
    st.subheader("Simulación de Datos ")
    # Descripción en formato LaTeX
    st.write("En esta simulación, hemos generado $n$ observaciones utilizando un modelo de regresión binomial negativa con los siguientes parámetros verdaderos:")
    st.latex(r"\beta = [0.3, -0.2, 1.4]")
    st.latex(r"\delta = 2")
    data = pd.DataFrame({'Y':y,'1':X[:, 0],'X1':X[:, 1],'X2':X[:, 2]})
    st.write('Un pequeño ejemplo de los datos simulados')
    st.dataframe(data,width=1000,height=450)



    

    with st.expander('Ver Código Datos Simulados:'):
        st.code('''  
        true_beta = [0.3, -0.2, 1.4]
        true_delta = 2    
        def generate_simulated_data(n,true_beta,true_delta):
        np.random.seed(0)
        beta = np.array(true_beta)
        x1 = np.random.normal(size=n)
        x2 = np.random.binomial(1, 0.5, size=n)
        X = np.column_stack((np.ones(n), x1, x2))
        mu = np.exp(np.dot(X, beta))
        delta_gm = true_delta
        p = delta_gm / (mu + delta_gm)
        mu_nb = delta_gm * (1 - p) / p
        theta_nb = delta_gm
        y = np.zeros(n)
        for i in range(n):
            y[i] = np.random.negative_binomial(theta_nb, theta_nb / (mu_nb[i] + theta_nb))
        return X, y
        # Simular datos
        n = 10000
        X, y = generate_simulated_data(n,true_beta,true_delta)
      ''')
        
    st.write('---')
        
    col = st.columns(4)
    with col[0]:
        st.write("### Parámetros:")
        st.write("Theta / Delta: " )
        st.write("B0:")
        st.write("B1:")
        st.write("B2:")
    with col[1]:
        st.write("### Verdadero:")
        st.write(f'{true_delta}')
        st.write(f'{true_beta[0]}')
        st.write(f'{true_beta[1]}')
        st.write(f'{true_beta[2]}')
    with col[2]:
        st.write("### Estimación:")
        st.write(f'{delta_hat:.4f}')
        st.write(f'{beta_hat[0]:.4f}')
        st.write(f'{beta_hat[1]:.4f}')
        st.write(f'{beta_hat[2]:.4f}')           
    with col[3]:
        st.write("### Error Estandar:")
        st.write(f'{std_errors[0]:.4f}')
        st.write(f'{std_errors[1]:.4f}')
        st.write(f'{std_errors[2]:.4f}')
        st.write(f'{std_errors[3]:.4f}')

    st.write('---')

    with st.expander('Ver Código Estimación:'):
        st.code(''' # Función para ajustar el modelo de regresión binomial negativa
    def negative_binomial_regression(X, y):
        n, m = X.shape
        params = np.ones(m + 1)

        def neg_log_likelihood(params):
            delta = params[0]
            beta = params[1:]
            mu = np.exp(np.dot(X, beta))
            p = delta / (mu + delta)
            logL = 0
            for i in range(n):
                bincof = gamma(delta + y[i]) / (gamma(delta) * gamma(y[i] + 1))
                logL += gammaln(delta + y[i]) - (gammaln(delta) + gammaln(y[i] + 1)) + delta * np.log(p[i]) + y[i] * np.log(1 - p[i])
            return -logL

        result = minimize(neg_log_likelihood, params, method='BFGS')
        std_errors = np.sqrt(np.diag(result.hess_inv))
        delta_hat= result.x[0]
        beta_hat = result.x[1:]
        return delta_hat, beta_hat, std_errors

         # Ajustar el modelo
        delta_hat, beta_hat, std_errors = negative_binomial_regression(X, y)
          ''')
    st.write('---')
   


# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    'Tarea 1': task_1,
    'Tarea 2': task_2,
    'Tarea 3': task_3,
    'Tarea 4': task_4
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
