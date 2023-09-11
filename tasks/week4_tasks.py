import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit



st.set_page_config(page_title='S4',
                    page_icon=':book:',
                    layout='wide')

# Título de la semana
st.title('Semana 4 - Modelos Multinomiales ')


# Funciones para las tareas específicas
def task_1():

    # Título y descripción de la tarea
    st.subheader('**Tarea 1: Demostración de la Distribución Asintótica Logit Condicional**')
    st.write(
    'En esta tarea, demostraremos la distribución asintótica de los estimadores MLE '
    'para el modelo de Regresión Logística Condicional.'
    )

    # Modelo de Logit Condicional
    st.subheader('Modelo de Logit Condicional')
    st.latex(
    r'''
    p_{ij} = \frac{e^{x_{ij}\beta}}{\sum_{l=1}^{m} e^{x_{il}\beta}}
    '''
    )
    st.write(
    'Donde:'
    )
    st.latex(
    r'''
    - p_{ij} \text{ es la probabilidad de que } Y_{ij} = 1
    
    '''
    )
    st.latex(r'- x_{ij} \text{ son los regresores que pueden afectar la probabilidad}')
    st.latex(r''' - \beta \text{ es el vector de parámetros a estimar}
    ''')
    st.latex(r'- m \text{ es el número de categorías de } Y')
    st.write(
    'Derivando la probabilidad $p_{ij}$ con respecto a  $\\beta$ tenemos que:'
    )
    st.latex(
    r'''
    \frac{\partial p_{ij}}{\partial \beta} = p_{ij} (x_{ij} - \bar{x}_i)
    '''
    )
    st.write(
    'Donde $\\bar{x}_i$ es el promedio ponderado de los regresores:'
    )
    st.latex(
    r'''
    \bar{x}_i = \sum_{u=1}^{m} p_{iu} x_{iu}
    '''
    )

    st.write('Podemos expresar la función de verosimilitud de la siguiente manera:')
    st.latex(r'L_N = \prod_{i=1}^N \prod_{j=1}^m p_{ij}^{y_{ij}}')
    st.write(
    'Por tanto, la función de log-verosimilitud se define como:'
    )
    st.latex(
    r'''
   \mathcal{L}(\beta) = \sum_{i=1}^{N} \sum_{j=1}^{m} y_{ij} \log(p_{ij})
    '''
    )
    st.write(
    'Derivando para obtener la condición de primer orden:'
    )
    st.latex(
    r'''
    \frac{\partial\mathcal{L}(\beta)}{\partial \beta} = \sum_{i=1}^{N} \sum_{j=1}^{m} y_{ij} (x_{ij} - \bar{x}_i) 
    '''
    )
    st.write(
    'Para maximizar, debemos sacar la segunda derivada, pero consideramos primero que $x_{ij}$ no depende en si de $\\beta$, entonces:'
    )
    st.latex(r'\frac{\partial^2  \mathcal{L}(\beta)}{\partial \beta \partial \beta^T} = -\sum_{i=1}^N \sum_{j=1}^m y_{ij} \frac{\partial \bar{x}}{\partial \beta^T}')
    st.latex(
    r''' \frac{\partial^2  \mathcal{L}(\beta)}{\partial \beta \partial \beta^T} = -\sum_{i=1}^N \sum_{j=1}^m y_{ij} \sum_{l=1}^m p_{il} (x_{il} - \bar{x}) x^T_{il}
    '''
    )
    st.write(
    '''Dado que $\sum_{j=1}^m y_{ij} = 1$ y considerando la propiedad de que $\\sum_{j=1}^m p_{ij}(x_{ij} - \\bar{x}_i)\\bar{x}_i^T = \sum_{j=1}^m (\\bar{x}_i - p_{ij}\\bar{x}_i)\\bar{x}_i^T = 0
$ puesto que la suma de las probabilidades debe ser igual a uno, esta expresión se reduce a:'''
    )
    st.latex(
    r'''
 \frac{\partial^2  \mathcal{L}(\beta)}{\partial \beta \partial \beta^T}= -\sum_{i=1}^N \sum_{j=1}^m p_{ij} (x_{xij} - \bar{x}\bar{x}_i)(x_{xij} - \bar{x}\bar{x}_i)^T

    '''
    )
    st.write(
    'La matriz de información de Fisher es entonces:'
    )
    st.latex(
    r'''
    I(\beta) = -E\left[ \frac{\partial^2\mathcal{L}(\beta)}{\partial \beta^2} \right]
    '''
    )
    st.write(
    'Y se da:'
    )
    st.latex(
    r'''
    = \sum_{i=1}^{N} \sum_{j=1}^{m} p_{ij} (x_{ij} - \bar{x}_i)(x_{ij} - \bar{x}_i)^T
    '''
    )

    # Resultado final
    st.header('Resultado Final')
    st.write(
    'Con las propiedades asintóticas del MLE, concluimos que los estimadores MLE de los parámetros del logit condicional:'
    )
    st.write(
    'siguen una distribución normal asintótica:'
    )
    st.latex(
    r'''
    \hat{\beta}_{CL} \sim^a N\left( \beta, \left(\sum_{i=1}^{N} \sum_{j=1}^{m} p_{ij} (x_{ij} - \bar{x}_i)(x_{ij} - \bar{x}_i)^T\right)^{-1} \right)
    '''
    )

    # Fin de la demostración
    st.write('Esta es la demostración de la distribución asintótica de los estimadores MLE en el modelo de Regresión Logística Condicional.')


def task_2():
    # Título y descripción de la tarea
    st.subheader('**Tarea 2: Demostración de la Distribución Asintótica de Estimadores MLE en Modelos Multinomiales**')
    st.write(
        'En esta demostración, exploraremos la distribución asintótica de los estimadores MLE en modelos multinomiales. '
        'Particularmente, analizaremos el caso general del Modelo Multinomial y presentaremos la derivación matemática paso a paso.'
    )

    
    st.subheader('Modelo Multinomial')
    st.write(
        'Un modelo multinomial describe el comportamiento de una variable discreta que puede tener más de dos resultados. '
        'En este contexto, se considera que hay m alternativas y una variable dependiente y que toma el valor j si se elige la j-ésima alternativa.'
    )

    st.subheader('Probabilidad y Log-Likelihood')
    st.write(
        'La probabilidad de que un individuo elija la alternativa j se denota como pij y se relaciona con un modelo Fj(xx, β) '
        'que depende de regresores xx y parámetros β.'
    )
    st.latex(
        r'''
        p_{ij} = Pr[y_{ij} = j] = F_{j}(x_i, \beta)
        '''
    )
    st.write(
        'La densidad multinomial para una observación se expresa como el producto de las probabilidades:'
    )
    st.latex(
        r'''
        f(y) = \prod_{i=1}^{m} p_{ij}
        '''
    )
    st.write(
        'Y el log-likelihood se define como la suma de logaritmos de probabilidades:'
    )
    st.latex(
        r'''
        \mathcal{L} = \ln L = \sum_{i=1}^{N} \sum_{j=1}^{m} y_{ij} \ln(p_{ij})
        '''
    )

    # Condiciones de Primer Orden
    st.subheader('Condiciones de Primer Orden')
    st.write(
        'Las condiciones de primer orden se obtienen al derivar el log-likelihood con respecto a los parámetros β:'
    )
    st.latex(
        r'''
        \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{N} \sum_{j=1}^{m} \frac{y_{ij}}{p_{ij}} \frac{\partial p_{ij}}{\partial \beta} = 0
        '''
    )

   
    st.subheader('Segunda Derivada ')
    st.write(
        'La segunda derivada (Hessiana) se obtiene al calcular la esperanza de la segunda derivada del log-likelihood con respecto a β:'
    )
    st.latex(
        r'''
        \frac{\partial^2 \mathcal{L}}{\partial \beta \partial \beta^T}  =  \sum_{i=1}^{N} \sum_{j=1}^{m} \frac{-y_{ij}}{p_{ij}^2} \cdot \frac{\partial p_{ij}}{\partial \beta} \cdot \frac{\partial p_{ij}}{\partial \beta^T} + \frac{y_{ij}}{p_{ij}} \cdot \frac{\partial^2 p_{ij}}{\partial \beta \partial \beta^T} 
        '''
    )
    st.write('Tomando menos el valor esperado y considerando que $E[y_{ij}]=p_{ij}$, obtenemos el resultado asintótico')
    st.subheader('Resultado Asintótico')
    st.write(
        'La teoría asintótica indica que el estimador MLE $\sqrt{N} (\hat{\\beta}-\\beta)$ converge en distribución a una distribución normal con '
        'media cero y varianza la inversa del valor esperado de la matriz hessiana.'
        'En el caso del Modelo Multinomial, el estimador converge a:'
    )
    
    st.latex(
        r'''
        \sqrt{N} (\hat{\beta_{MLE}}-\beta) \sim \mathcal{N}\left( 0, \left[ \sum_{i=1}^{N} \sum_{j=1}^{m} \frac{1}{p_{ij}} \cdot \frac{\partial p_{ij}}{\partial \beta} \cdot \frac{\partial p_{ij}}{\partial \beta^T} - \frac{\partial^2 p_{ij}}{\partial \beta \partial \beta^T} \right]^{-1} \right)
        '''
    )

    # Fin de la demostración
    st.write('Esta es la demostración de la distribución asintótica de los estimadores MLE en modelos multinomiales.')

def task_3():
    st.subheader('**Tarea 2: Segunda derivada en el MNL (logit multinomial)**')
 

    st.write("En esta tarea, exploraremos el modelo MNL, específicamente deducir la segunda derivada del log-likelihood.")

    

    # Explicación del modelo MNL
    st.write("El modelo multinomial logit (MNL) es utilizado para describir el comportamiento de una variable discreta con más de dos resultados posibles. En este modelo, se asume que las probabilidades de cada alternativa están relacionadas a través de una función logística.")

    # Fórmula MNL
    st.latex(r'''
    p_{ij} = \frac{e^{x_{ij}\beta_j}}{\sum_{l=1}^{m} e^{x_{il}\beta_l}}
    ''')

    # Explicación de las derivadas parciales en MNL
    st.write("Para maximizar la log-verosimilitud en el modelo MNL, se calculan las derivadas parciales de las probabilidades con respecto a los parámetros (β). A continuación, se presentan estas derivadas parciales:")

    # Fórmula de la derivación parcial
    st.latex(r'''
    \frac{\partial p_{ij}}{\partial \beta_k} = p_{ij}(δ_{jk} - p_{ik})x_i
    ''')

    # Explicación de las primeras derivadas en MNL
    st.write("Definimos las primeras derivadas o las condiciones de primer orden para la estimación de los parámetros (β) en el modelo MNL, esto basandonos en:")
    
    st.latex(
        r'''
        \frac{\partial \mathcal{L}}{\partial \beta_k} = \sum_{i=1}^{N} \sum_{j=1}^{m} \frac{y_{ij}}{p_{ij}} \frac{\partial p_{ij}}{\partial \beta} = 0
        '''
    )
    st.write('Por tanto:')
    # Fórmula de las primeras derivadas
    st.latex(r'''
    \frac{\partial \mathcal{L}}{\partial \beta_k} = \sum_{i=1}^{N}(y_{ik} - p_{ik})x_{ij}
    ''')

    # Explicación de la segunda derivada en MNL
    st.write("A continuación, se presenta la derivación de la segunda derivada, en donde se considera que ${y_ij}$ no depende de $\\beta$:")

   
    st.latex(r'''\frac{\partial^2\mathcal{L}(\beta)}{\partial\beta_k\partial\beta_k^T} = -\sum_{i=1}^{N} \frac{\partial p_{ik}}{\partial\beta^T} x_{i}
''')
    st.write('Usando el punto (la primera derivada calculada) de partida tenemos:')
    st.latex(r'''
    \frac{\partial^2 \mathcal{L}}{\partial \beta_k \partial \beta_k^T} = -\sum_{i=1}^{N}p_{ij}(δ_{jk} - p_{ik})x_{ij}x_{ij}^T
    ''')

    # Resultado asintótico en MNL
    st.write("El resultado asintótico establece que los estimadores MLE de β convergen en distribución a una distribución normal con media β. La matriz de información de Fisher se calcula de la siguiente manera:")

    # Matriz de información de Fisher en MNL
    st.latex(r'''
    I(\beta) = -E\left[\frac{\partial^2 \mathcal{L}}{\partial \beta_k \partial \beta_k^T} \right]= \sum_{i=1}^{N}\sum_{j=1}^{m}p_{ij}(δ_{jk} - p_{ik})x_{ij}x_{ij}^T
    ''')


def task_4():
    st.write('---')
    # Configuración de la aplicación de Streamlit
    st.subheader("**Tarea 4: Simulación y Comparación de Modelos Conditional Logit**")

     #Introducción a la tarea
    st.write('En esta tarea, simularemos un dataset considerando una problematica y estimaremos un modelo Conditional Logit utilizando la librería Statsmodels, pero también una implementación desde cero usando Máxima Verosimilitud')


        # Problemáticas
    st.write('---')
    st.subheader('Formulación de la Problemática')
    st.write('Ahora, formularemos una problemática basada en los datos simulados.')

    # Descripción de la problemática (puedes personalizar esto)
    st.write('Supongamos que estamos estudiando la probabilidad de que un individuo compre un producto basado en una variable explicativa (X) y agrupado por ciertas categorías (grupos). Queremos entender cómo la variable X afecta la decisión de compra en diferentes grupos.')


    # Generación de datos simulados con variables explicativas de dos dimensiones
    np.random.seed(123)
    n_observaciones = 1000
    real = [0.3,2.3]
    grupos = np.kron(np.arange(100), np.ones(10)).astype(int)

    # Variables explicativas con dos dimensiones
    variables_explicativas = np.random.normal(size=(n_observaciones, 2))

    probabilidades = 1 / (1 + np.exp(-real[0]*variables_explicativas[:, 0] - real[1]*variables_explicativas[:, 1]))

    # probabilidades = 1 / (1 + np.exp(-variables_explicativas))
    variables_dependientes = (np.random.uniform(size=n_observaciones) < probabilidades).astype(int)

     # Visualización de datos simulados
    st.write('---')
    st.write('Ejemplo de observaciones del dataset simulado:')
    data_simulada = pd.DataFrame({'Grupo': grupos, 'X1': variables_explicativas[:, 0], 'X2': variables_explicativas[:, 1], 'Y': variables_dependientes})
    st.dataframe(data_simulada, height=400, width=1000)
    
    

    with st.expander('Ver código simulación datos:'):
        st.code('''# Generación de datos simulados con variables explicativas de dos dimensiones
    np.random.seed(123)
    n_observaciones = 10000
    grupos = np.kron(np.arange(100), np.ones(10)).astype(int)

    # Variables explicativas con dos dimensiones
    variables_explicativas = np.random.normal(size=(n_observaciones, 2))

    probabilidades = 1 / (1 + np.exp(-real[0]*variables_explicativas[:, 0] - real[1]*variables_explicativas[:, 1]))

    # probabilidades = 1 / (1 + np.exp(-variables_explicativas))
    variables_dependientes = (np.random.uniform(size=n_observaciones) < probabilidades).astype(int)  ''')

    def log_likelihood_conditional(beta, X, y, grupos):
        eta = X.dot(beta)  
        pr = expit(eta)
        log_likelihood = 0
        unique_grupos = np.unique(grupos)

        for grupo in unique_grupos:
            mask = grupos == grupo
            pr_grupo = pr[mask]
            y_grupo = y[mask]
            log_likelihood += np.sum(y_grupo * np.log(pr_grupo) + (1 - y_grupo) * np.log(1 - pr_grupo))

        return -log_likelihood

    

    res_optim = minimize(log_likelihood_conditional, np.zeros(2), args=(variables_explicativas, variables_dependientes, grupos), method='BFGS')
    estimated_beta_scratch = res_optim.x


    #calculo de los errores estandar
    std_errors = np.sqrt(np.diag(res_optim.hess_inv))


    #diseño de los resultados
    st.write('---')
    # Implementación del modelo Conditional Logit con maxima verosmilitud
    st.subheader('Estimación del Modelo Conditional Logit con implementación:')
    cols = st.columns(4) 
    with cols[0]:
        st.write('Parámetro:')
        st.write(f'Beta 1')
        st.write(f'Beta 2')
    with cols[1]:
        st.write('Estimación:')
        st.write(f'{estimated_beta_scratch[0]:.4f}')
        st.write(f'{estimated_beta_scratch[1]:.4f}')
    with cols[2]:
        st.write('Desviación:')
        st.write(f'{std_errors[0]:.4f}')
        st.write(f'{std_errors[1]:.4f}')
    with cols[3]:
        st.write('Real:')
        st.write(f'{real[0]:.4f}')
        st.write(f'{real[1]:.4f}')    

    with st.expander('Ver código estimación:'):
        st.code('''
    def log_likelihood_conditional(beta, X, y, grupos):
        eta = X.dot(beta)  
        pr = expit(eta)
        log_likelihood = 0
        unique_grupos = np.unique(grupos)

        for grupo in unique_grupos:
            mask = grupos == grupo
            pr_grupo = pr[mask]
            y_grupo = y[mask]
            log_likelihood += np.sum(y_grupo * np.log(pr_grupo) + (1 - y_grupo) * np.log(1 - pr_grupo))

        return -log_likelihood

    

    res_optim = minimize(log_likelihood_conditional, np.zeros(2), args=(variables_explicativas, variables_dependientes, grupos), method='BFGS')
    estimated_beta_scratch = res_optim.x


    #calculo de los errores estandar
    std_errors = np.sqrt(np.diag(res_optim.hess_inv))
 ''')
    st.write('---')

    # Implementación del modelo Conditional Logit de Statsmodels
    st.subheader('Estimación del Modelo Conditional Logit de Statsmodels')
    st.write('Utilizaremos el modelo Conditional Logit de la librería Statsmodels para analizar esta problemática.')

    # Creación del modelo Conditional Logit de Statsmodels
    modelo_statsmodels = ConditionalLogit(endog=variables_dependientes, exog=variables_explicativas, groups=grupos)
    results_statsmodels = modelo_statsmodels.fit()

    # Mostrar resultados del modelo de Statsmodels
    st.write(results_statsmodels.summary())

    with st.expander('Ver código estimación:'):
        st.code('''
  # Creación del modelo Conditional Logit de Statsmodels
    modelo_statsmodels = ConditionalLogit(endog=variables_dependientes, exog=variables_explicativas, groups=grupos)
    results_statsmodels = modelo_statsmodels.fit()

    # Mostrar resultados del modelo de Statsmodels
    st.write(results_statsmodels.summary())
 ''')
    st.write('---')

    st.write('**Conclusiones**')
    st.write('Dado que, de acuerdo con la documentación de la función implementada, el algoritmo usa conditional likelihood.  Los resultados pueden ser un poco diferentes, '
             'además, de acuerdo al método de optimización usado ambos pueden diverger un poco. Aún asī, encontramos que ambos métodos se acercan muy bien a los valores poblacionales con apenas 1000 datos.')

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
