import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma, gammaln


st.set_page_config(page_title='S6',
                    page_icon=':book:',
                    layout='wide')


results_task_1 = {'delta':[32.7874,1.0869],'beta_0':[0.3045,
0.0130],'beta_1':[-0.2112,0.0139],'beta_2':[0.9930,0.0151]}



# Título de la semana
st.title('Semana 6 - Modelos Truncados y de Sesgo de Selección')


# Funciones para las tareas específicas
def task_1():
    st.write('---')
    st.subheader("**Tarea 1: Modelo Truncado hacia la Derecha: Simulación y Estimación**")
    

    # Función para generar datos simulados truncados hacia la derecha
    def generate_truncated_data(n, true_beta, true_delta, truncation_value):
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
            if y[i] >= truncation_value:
                y[i] = truncation_value - 1
        return X, y

    # Función para ajustar el modelo de regresión truncado hacia la derecha
    def truncated_regression(X, y):
        n, m = X.shape
        params = np.ones(m + 1)

        def neg_log_likelihood(params):
            delta = params[0]
            beta = params[1:]
            mu = np.exp(np.dot(X, beta))
            p = delta / (mu + delta)
            logL = 0
            for i in range(n):
                # bincof = gamma(delta + y[i]) / (gamma(delta) * gamma(y[i] + 1))
                logL += gammaln(delta + y[i]) - (gammaln(delta) + gammaln(y[i] + 1)) + delta * np.log(p[i]) + y[i] * np.log(1 - p[i])
            return -logL

        result = minimize(neg_log_likelihood, params, method='BFGS')
        std_errors = np.sqrt(np.diag(result.hess_inv))
        delta_hat = result.x[0]
        beta_hat = result.x[1:]
        return delta_hat, beta_hat, std_errors

    # Parámetros verdaderos
    true_beta = [0.3,-0.3,1.2]
    true_delta = 4
    truncation_value = 7

    #Generar datos truncados hacia la derecha
    # n = 10000
    # X, y = generate_truncated_data(n, true_beta, true_delta, truncation_value)

    # # Ajustar el modelo truncado hacia la derecha
    data = pd.read_csv('data/s6t1d1.csv')
    X = np.array(data[['1','X1','X2']])
    y= np.array(data['Y'])
    # delta_hat, beta_hat, std_errors = truncated_regression(X, y)

    

    # Descripción en formato LaTeX
    st.write("En esta simulación, hemos generado $n$ observaciones utilizando un modelo de regresión binomial negativa truncado hacia la derecha con los siguientes parámetros verdaderos:")
    st.latex(r"\beta = [2.3,-0.3,2]")
    st.latex(r"\delta = 1")
    st.write("Valor de Truncación = 7")

    # Mostrar los datos generados
    st.write('---')
    st.subheader("Datos Generados")
    # data = pd.DataFrame({'Y':y,'1':X[:, 0],'X1':X[:, 1],'X2':X[:, 2]})
    # data.to_csv('data/s6t1d1.csv')
    st.write('Un ejemplo de los datos simulados')
    st.dataframe(data,width=1000,height=450)
    st.write('---')
        # Código para generar datos truncados hacia la derecha
    with st.expander('Ver Código Datos Truncados hacia la Derecha:'):
        st.code(f'''  
        # Parámetros verdaderos
        true_beta = {true_beta}
        true_delta = {true_delta}
        truncation_value = {truncation_value}

        
        # Función para generar datos simulados truncados hacia la derecha
        def generate_truncated_data(n, true_beta, true_delta, truncation_value):
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
                if y[i] >= truncation_value:
                    y[i] = truncation_value - 1
            return X, y

        n = 1000
        X, y = generate_truncated_data(n, true_beta, true_delta, truncation_value)
        ''')
    st.write('---')

    # Mostrar los resultados de la estimación
    st.subheader("Resultados de la Estimación")
    col = st.columns(4)
    with col[0]:
        st.write("### Parámetros:")
        st.write("Theta / Delta:")
        st.write("B0:")
        st.write("B1:")
        st.write("B2:")
    with col[1]:
        st.write("### Verdadero:")
        st.write(true_delta)
        st.write(true_beta[0])
        st.write(true_beta[1])
        st.write(true_beta[2])
    with col[2]:
        st.write("### Estimación:")
        st.write(f'{results_task_1["delta"][0]}')
        st.write(f'{results_task_1["beta_0"][0]}')
        st.write(f'{results_task_1["beta_1"][0]}')
        st.write(f'{results_task_1["beta_2"][0]}')
        # st.write(f'{delta_hat:.4f}')
        # st.write(f'{beta_hat[0]:.4f}')
        # st.write(f'{beta_hat[1]:.4f}')
        # st.write(f'{beta_hat[2]:.4f}')
    with col[3]:
        st.write("### Error Estandar:")
        st.write(f'{results_task_1["delta"][1]}')
        st.write(f'{results_task_1["beta_0"][1]}')
        st.write(f'{results_task_1["beta_1"][1]}')
        st.write(f'{results_task_1["beta_2"][1]}')
        # st.write(f'{std_errors[0]:.4f}')
        # st.write(f'{std_errors[1]:.4f}')
        # st.write(f'{std_errors[2]:.4f}')
        # st.write(f'{std_errors[3]:.4f}')

    st.write('---')


    # Código para estimar el modelo truncado hacia la derecha
    with st.expander('Ver Código Estimación del Modelo Truncado hacia la Derecha:'):
        st.code(f'''  
        
        # Función para ajustar el modelo de regresión truncado hacia la derecha
        def truncated_regression(X, y, truncation_value):
            n, m = X.shape
            params = np.ones(m + 1)

            def neg_log_likelihood(params):
                delta = params[0]
                beta = params[1:]
                mu = np.exp(np.dot(X, beta))
                p = delta / (mu + delta)
                logL = 0
                for i in range(n):
                    logL += gammaln(delta + y[i]) - (gammaln(delta) + gammaln(y[i] + 1)) + delta * np.log(p[i]) + y[i] * np.log(1 - p[i])
                return -logL

            result = minimize(neg_log_likelihood, params, method='BFGS')
            std_errors = np.sqrt(np.diag(result.hess_inv))
            delta_hat = result.x[0]
            beta_hat = result.x[1:]
            return delta_hat, beta_hat, std_errors

        # Ajustar el modelo truncado hacia la derecha
        delta_hat, beta_hat, std_errors = truncated_regression(X, y, truncation_value)
        ''')
    st.write('---')

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
