import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats  #
from streamlit_option_menu import option_menu


st.set_page_config(page_title='S1',
                    page_icon=':book:',
                    layout='wide')


# Título de la semana
st.title('Semana 1 - Teoría Asintótica')



# Función para la tarea 1
def task_1():
    st.write('---')
    # Título de la página
    st.header("Tarea 1: Consistencia del Estimador MCO")

    # Descripción del problema
    st.write("Se estudiará la consistencia (convergencia) del estimador MCO, para el siguiente modelo:")

    st.latex(r"Y_i = 1 + 0.5X_i + \epsilon_i")

    st.write("donde:")
    st.write(r"$X_i \sim N(0, 2^2)$")
    st.write(r"$\epsilon_i \sim U(-1, 1)$")
    st.write("Se considerarán valores de $n$ = { 50,100,500,1000, 5000,10000, 50000,100000 }.")

    # Función para la simulación y cálculo de estimadores MCO
    def simulate_MCO(n):
        np.random.seed(10)
        Xi = np.random.normal(0, 2, n)
        ei = np.random.uniform(-1, 1, n)
        Yi = 1 + 0.5 * Xi + ei
        X = np.column_stack((np.ones(n), Xi))
        inv_XX = np.linalg.inv(np.dot(X.T, X) / n)
        beta_hat = np.dot(np.dot(inv_XX, X.T), Yi) / n
        return beta_hat[1]  # Retorna el estimador MCO de beta
    
    # Simulación y cálculo de estimadores MCO para diferentes tamaños de muestra
    n_values = [50,100,500,1000, 5000, 10000,50000,100000]
    estimated_betas = {n:simulate_MCO(n) for n in n_values}
    
    # Gráfico de convergencia
    st.write('---')
    st.write("Estimaciones de MCO para diferentes tamaños de muestra:")
    st.write(estimated_betas)    

    with st.expander("Ver Código Simulación"):
        st.write("Código de la simulación de estimadores MCO:")
        st.code("""
    # Función para la simulación y cálculo de estimadores MCO
    def simulate_MCO(n):
        np.random.seed(10)
        Xi = np.random.normal(0, 2, n)
        ei = np.random.uniform(-1, 1, n)
        Yi = 1 + 0.5 * Xi + ei
        X = np.column_stack((np.ones(n), Xi))
        inv_XX = np.linalg.inv(np.dot(X.T, X) / n)
        beta_hat = np.dot(np.dot(inv_XX, X.T), Yi) / n
        return beta_hat[1]  # Retorna el estimador MCO de beta
        # Simulación y cálculo de estimadores MCO para diferentes tamaños de muestra
        n_values = [50,100,500,1000, 5000, 10000,50000,100000]
        estimated_betas = {n:simulate_MCO(n) for n in n_values}

    """)

    st.write('---')
    # fig, ax = plt.subplots(figsize=(10, 8))

    # ax.plot(n_values, estimated_betas, marker='o')
    # ax.axhline(0.5, color='red', linestyle='--', label='Valor Verdadero')
    # ax.set_xlabel('Tamaño de Muestra (n)')
    # ax.set_ylabel('Estimador MCO de Beta')
    # ax.legend()
    # fig.savefig('images/s1t1f1.png')
    st.write('Gráfica de Consistencia del Estimador:')
    st.image("images/s1t1f1.png")

    with st.expander('Ver Código Gráfica:'):
        # Gráfico de convergencia en distribución
        st.write("Código de la generación del Gráfica de Convergencia del Estimador")
        st.code('''
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(n_values, estimated_betas, marker='o')
    ax.axhline(0.5, color='red', linestyle='--', label='Valor Verdadero')
    ax.set_xlabel('Tamaño de Muestra (n)')
    ax.set_ylabel('Estimador MCO de Beta')
    ax.legend()
    fig.savefig('images/s1t1f1.png')
    st.write('Gráfica de Consistencia del Estimador:')
    st.image("images/s1t1f1.png")
            ''')
    
    st.write('---')
    # Descripción de la convergencia en distribución
    st.write("La convergencia en distribución se da por el Teorema Central del Límite (TCL).")
    st.write("En particular, tenemos que:")
    st.latex(r"\frac{(\hat{\beta}_n - \beta)}{\sqrt{Var(\hat{\beta}_n)}} \stackrel{d}{\rightarrow} N(0,1)")

   
    # Simulación para la convergencia en distribución
    def simulate_distribution_convergence(n):
        np.random.seed(7)
        Zs = []
        Xi = np.random.normal(0, 2, n)
        X = np.column_stack((np.ones(n), Xi))
        XX = np.dot(X.T, X) / n
        XXi = np.linalg.inv(XX)
        xxi2 = XXi[1, 1]
        for _ in range(10000):
            ei = np.random.normal(0, 1, n)
            Yi = 1 + 0.5 * Xi + ei
            beta_hat = np.dot(np.dot(XXi, X.T), Yi) / n
            res = Yi - np.dot(X, beta_hat)
            sigma = np.std(res)
            Z = np.sqrt(n) * (beta_hat[1] - 0.5) / (sigma * np.sqrt(xxi2))
            Zs.append(Z)
        return Zs

    # Simulación de convergencia en distribución
    # distribution_convergence_data = simulate_distribution_convergence(10000)
    with st.expander("Ver Código Simulación"):
        # Mostrar el código de la simulación de convergencia en distribución
        st.write("Código de la simulación de convergencia en distribución:")
        st.code("""
        # Simulación para la convergencia en distribución
        def simulate_distribution_convergence(n):
            np.random.seed(7)
            Zs = []
            Xi = np.random.normal(0, 2, n)
            X = np.column_stack((np.ones(n), Xi))
            XX = np.dot(X.T, X) / n
            XXi = np.linalg.inv(XX)
            xxi2 = XXi[1, 1]
            for _ in range(10000):
                ei = np.random.normal(0, 1, n)
                Yi = 1 + 0.5 * Xi + ei
                beta_hat = np.dot(np.dot(XXi, X.T), Yi) / n
                res = Yi - np.dot(X, beta_hat)
                sigma = np.std(res)
                Z = np.sqrt(n) * (beta_hat[1] - 0.5) / (sigma * np.sqrt(xxi2))
                Zs.append(Z)
            return Zs
        """)
    

    # Gráfico de convergencia en distribución
    st.write("### Verificación de Convergencia en Distribución:")
    # fig1, ax1 = plt.subplots(figsize=(10, 8))
    # ax1.hist(distribution_convergence_data, bins=50, density=True, alpha=0.6, color='g')
    # ax1.plot(np.linspace(-3, 3, 100), stats.norm.pdf(np.linspace(-3, 3, 100), 0, 1), 'r-', lw=2, label='N(0,1)')
    # ax1.set_xlabel('Beta')
    # ax1.set_ylabel('Densidad de Probabilidad')
    # ax1.legend()
    # fig1.savefig('images/s1t1f2.png')
    st.image("images/s1t1f2.png")
    with st.expander('Ver Código Gráfica:'):
        # Gráfico de convergencia en distribución
        st.write("Código de la generación del gráfico de convergencia en distribución:")
        st.code('''
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            ax1.hist(distribution_convergence_data, bins=50, density=True, alpha=0.6, color='g')
            ax1.plot(np.linspace(-3, 3, 100), stats.norm.pdf(np.linspace(-3, 3, 100), 0, 1), 'r-', lw=2, label='N(0,,1)')
            ax1.set_xlabel('Beta')
            ax1.set_ylabel('Densidad de Probabilidad')
            ax1.legend()
            fig1.savefig('images/s1t1f2.png')
            st.image("images/s1t1f2.png")''')
        
    st.write('---')
    st.write('**Desarrollo Matemático:**')
    st.write('Desarrollaremos ahora la parte matemática para mostrar las propiedades asintóticas del estimador de MCO')
    st.markdown("Consistencia")
    st.write("Existen dos formas de demostrar que el estimador MCO (que para simplificar, de aquí en adelante llamaremos $\\hat{\\beta}$) es consistente. Directamente, mostrando que converge en probabilidad a $\\beta$, o indirectamente, demostrando que converge en segundo momento a $\\beta$ y, por lo tanto, en probabilidad. Para simplificar y mostrar la falta de sesgo del estimador MCO, emplearemos la demostración indirecta.")

    st.write("Comenzamos calculando el error muestral, ya que es útil para encontrar el sesgo y la varianza analítica de $\\hat{\\beta}$.")
    st.latex(r"\hat{\beta} - \beta = (X^TX)^{-1}X^Ty - \beta ")
    st.latex(r'\hat{\beta} - \beta= (X^TX)^{-1}X^T(X\beta + \epsilon) - \beta ')
    st.latex(r'\hat{\beta} - \beta = \beta + (X^TX)^{-1}X^T\epsilon - \beta ')
    st.latex(r'\hat{\beta} - \beta =  = (X^TX)^{-1}X^T\epsilon')

    st.write("La sencilla álgebra matricial y la suposición de media nula del error nos llevan a concluir que el sesgo es cero.")
    st.latex(r"bias(\hat{\beta}) = E(\hat{\beta} - \beta)  = (X^TX)^{-1}X^TE(\epsilon) = 0")
    st.latex(r'bias(\hat{\beta}) = E((X^TX)^{-1}X^T\epsilon)')
    st.latex(r'bias(\hat{\beta}) = (X^TX)^{-1}X^TE(\epsilon) = 0')
    st.write("Definimos $A = (X^TX)^{-1}X^T$, entonces la varianza de $\hat{\\beta}$ es:")
    st.latex(r"Var(\hat{\beta}) = Var(\hat{\beta} - \beta) = Var((X^TX)^{-1}X^T\epsilon)")
    st.latex(r'Var(\hat{\beta}) = AVar(\epsilon)\ A^T = A\sigma^2I_n A^T ')
    st.latex(r' Var(\hat{\beta})  = A\sigma^2AA^T = \sigma^2AA^T')
    st.latex(r'Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}X^T((X^TX)^{-1}X^T)^T = \sigma^2((X^TX)^T)^{-1} = \sigma^2(X^TX)^{-1}')
    
    st.write('Entonces, puesto que la matriz inversa (X^TX)^{-1} aumenta su información cunando la muestra es asintótica, tenemos que:')
    st.latex(r"\lim_{n\to\infty}MSE(\hat{\beta}) = \lim_{n\to\infty}\sigma^2(X^TX)^{-1} = 0.")

    st.write("Por lo tanto, $\hat{\\beta}$ converge en segundo momento a $\\beta$ y, en consecuencia, converge en probabilidad a $\\beta$, es decir, es consistente.")

    st.write("A partir de lo anterior, sabemos que $\\hat{\\beta}$ converge en distribución a $\\beta$, ya que esta es una convergencia menos exigente que la convergencia en probabilidad.")

    st.write("Esto concluye la demostración.")



# Función para la tarea 2
def task_2():
    st.write('---')
    # Título de la tarea
    st.header('Tarea 2: Convergencia en Probabilidad y la Desigualdad de Chebyshev')

    # Desigualdad de Chebyshev
    st.subheader('Desigualdad de Chebyshev')

    st.write(
        "La desigualdad de Chebyshev es muy utilizada en estadística y probabilidad ya que nos proporciona una cuota superior "
        "para la probabilidad de que una variable aleatoria se desvíe de su valor esperado por más de un cierto valor, "
        "y se puede aplicar a cualquier distribución con varianza finita. Concretamente, par una variable aleatoria X, con $𝐸(𝑋_n)=\mu$ y $𝑉𝑎𝑟(X_n)=\sigma^2$"
        "la desigualdad anuncia:"
    )

    st.latex(r"P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}")

    st.write('Exploraremos como demostrar la convergencia en probablidad usando la desigualdad de Chebyshev, para eso consideramos el caso de la variable aleatoria donde:')
    st.latex(r'\mathbb{E}(X_{\bar{n}}) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}(X_i) = \mu')
    st.latex(r'\text{Var}(X_{\bar{n}}) = \frac{1}{n^2} \sum_{i=1}^{n} \text{Var}(X_i) = \frac{\sigma^2}{n}')

    st.write('Aplicando a la desigualdad, tenemos:')
    st.latex(r'P(|X_{\bar{n}} - \mu| \geq \epsilon) \leq \frac{\text{Var}(X_{\bar{n}})}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}')
    st.write('Sacando el complemento de la probabilidad:')
    st.latex(r'P(|X_{\bar{n}} - \mu| < \epsilon) = 1 - P(|X_{\bar{n}} - \mu| \geq \epsilon)')
    st.write('Operando tenemos entonces:')
    st.latex(r'P(|X_{\bar{n}} - \mu| < \epsilon) \geq 1 - \frac{\sigma^2}{n\epsilon^2}')
    st.write('Finalmente, tomando el limite, demostramos convergencia en probabilidad a partir de condiciones indiciales:')
    st.latex(r'\lim_{n \to \infty} P(|X_{\bar{n}} - \mu| < \epsilon) = 1')

    # Descripción del problema

    st.write('---')
    st.write('**Caso MCO**')
    st.write(''' La desigualdad de Chebyshev puede también utilizarse para demostrar la convergencia en probabilidad 
    (consistencia) de los estimadores de MCO en términos de la varianza de los estimadores.'''
             '''En el contexto de los estimadores de MCO, podemos aplicar la desigualdad de Chebyshev a la diferencia
              entre el estimador de MCO $\hat{\\beta_j}$ y el valor verdadero del parámetro $\\beta_j$.''')

    st.write('''Dado que estamos interesados en demostrar la convergencia en probabilidad de los estimadores, 
             nos gustaría que la probabilidad de que $|\\hat{\\beta}_j - \\beta_j|$ 
             sea mayor que una cantidad positiva $\\epsilon$ tienda a cero a medida que el tamaño de
               la muestra ($n$) aumenta, es decir:''')
    st.latex(r"\lim_{{n \to \infty}} P(|\hat{\beta}_j^n - \beta_j| \geq \epsilon) = 0")
                
    st.write('''Entonces, podemos establecer $k$ de la siguiente manera:''')

    st.latex(r"k = \frac{\sigma(\hat{\beta}_j)}{\epsilon}")

    st.write('''Donde $\sigma(\\hat{\\beta}_j)$ es la desviación estándar del estimador de MCO. Ahora, 
             aplicamos la desigualdad de Chebyshev:''')

    st.latex(r"P(|\hat{\beta_j} - \beta_j| \geq \epsilon) \leq \frac{\sigma(\hat{\beta}_j)^2}{\epsilon^2}")

    st.write('''El objetivo es mostrar que esta probabilidad tiende a cero a medida que $n$ tiende a infinito.
      Para lograrlo, necesitamos demostrar que la varianza del estimador de MCO $\sigma(\\hat{\\beta_j})^2$ 
      tiende a cero a medida que $n$ aumenta.''')

    st.write('''Bajo las condiciones estándar de los estimadores de MCO (como linealidad, 
    independencia de los errores y exogeneidad débil), es posible demostrar que la varianza del estimador 
    de MCO tiende a cero a medida que $n$ aumenta. Para demostrar que la varianza del estimador de MCO 
    $\\hat{\\beta_j}$ tiende a cero a medida que $n$ tiende a infinito, podemos usar propiedades de
    las matrices y el hecho de que $\sigma^2$ es una constante.''')

    st.write('''Dado que estamos hablando de la varianza del estimador de MCO, estamos interesados en
              $Var(\\hat{\\beta_j})$. Esta es la varianza del estimador $\\hat{\\beta_j}$ y se puede
                expresar en términos de la matriz de covarianza de los errores y la matriz de diseño $X$.
                  La fórmula para la varianza de $\\hat{\\beta_j}$ es:''')

    st.latex(r"Var(\hat{\beta}_j) = \sigma^2 \cdot (X^T X)^{-1}_{jj}")

    st.write('''Donde $\sigma^2$ es la varianza de los errores y $(X^T X)^{-1}_{jj}$ es 
             el elemento $jj$ de la matriz inversa de $X^T X$.''')

    st.write('''A medida que $n$ aumenta, la matriz de diseño $X$ se vuelve más 'informativa' y bien condicionada, 
             lo que significa que $(X^T X)^{-1}$ se aproxima a cero. Esto implica que la varianza de $\\hat{\\beta}_j$ 
             se reduce, y a medida que $n$ tiende a infinito, la varianza tiende a cero. Por lo tanto, podemos afirmar que:''')

    st.latex(r"\lim_{n \to \infty} Var(\hat{\beta}_j) = 0")

    st.write('''Esta es la justificación de por qué la varianza del estimador de MCO tiende a cero a medida que 
n tiende a infinito. Esto es un resultado,
que dado los plantemientos anteriores nos permite afirmar la consistencia del estimador por MCO usando la desigualdad
del Chebyshev.''')


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