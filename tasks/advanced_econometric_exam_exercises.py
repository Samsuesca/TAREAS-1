import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
import pandas as pd
from IPython.display import display, Math, Latex
from sympy import symbols, exp, sqrt, pi, log, Eq

# Configuración de la página
st.set_page_config(
    page_title="Parcial Advanced Econometrics II - Angel Samuel Suesca Rios",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Advanced Econometrics II - Ejercicios Teóricos")
st.write("Autor: Angel Samuel Suesca Rios")

# Tabs para cada ejercicio
tabs = st.tabs(["E.1: Distribución Logística", "E.4: Modelo de Variable Latente", 
                "E.5: Modelos de Selección", "E.6: Distribución Poisson"])

# TAB 1: Ejercicio 1 - Distribución Logística
def ej1():
    st.header("E.1: Propiedades de la Distribución Logística")
    
    st.markdown(r"""
    Para la distribución logística verificaremos que:
    
    $$\Lambda(x) = (1 + \exp(-x))^{-1}$$
    
    ### a) $\frac{d}{dx}\Lambda(x) = \Lambda(x)(1 - \Lambda(x))$
    
    Planteamos $\Lambda(x) = f(g(x))^{-1}$ donde $g(x) = 1 + e^{-x}$. 
    
    Por regla de la cadena:
    $$\frac{df}{dx} = \frac{df}{dg}\frac{dg}{dx}$$
    
    Tenemos:
    $$\frac{d}{dx}\Lambda(x) = - \frac{1}{g^2} \cdot \frac{d}{dx}(1 + e^{-x}) = - \frac{1}{g^2} \cdot (-1)e^{-x} = \frac{1}{g^2}e^{-x} = \frac{1}{(1+e^{-x})^2}e^{-x}$$
    
    Separando los factores en el denominador:
    $$\frac{d}{dx}\Lambda(x) = \frac{1}{(1+e^{-x})} \cdot \frac{e^{-x}}{(1+e^{-x})} = \Lambda(x) \left[ \frac{e^{-x}}{1+e^{-x}} \right]$$
    
    Si sumamos y restamos 1 en el numerador del segundo factor:
    $$\frac{d}{dx}\Lambda(x) = \Lambda(x) \left[ \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right] = \Lambda(x) \left[ \frac{1 + e^{-x}}{1 + e^{-x}} - \frac{1}{1 + e^{-x}} \right]$$
    
    Simplificando:
    $$\frac{d}{dx}\Lambda(x) = \Lambda(x) [1 - \Lambda(x)]$$
    
    ### b) $h_{logit}(x) = \frac{d}{dx}\log \Lambda(x) = 1 - \Lambda(x)$
    
    Sea $g(x) = \log(\Lambda(x)) = \log[(1 + e^{-x})^{-1}] = -\log(1 + e^{-x})$
    
    Derivando:
    $$\frac{d}{dx}g(x) = -\frac{d \log(1 + e^{-x})}{dx} = -\frac{1}{1+e^{-x}} \cdot \frac{d(1+e^{-x})}{dx}$$
    
    $$\frac{d}{dx}g(x) = -\frac{1}{1+e^{-x}} \cdot (-e^{-x}) = \frac{e^{-x}}{1+e^{-x}}$$
    
    Sumando y restando en el numerador:
    $$\frac{d}{dx}g(x) = \frac{1+e^{-x}-1}{1+e^{-x}} = \frac{1+e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}} = 1 - \Lambda(x)$$
    
    Por tanto: $h_{logit}(x) = 1 - \Lambda(x)$
    
    ### c) $H_{logit}(x) = -\frac{d^2}{dx^2}\log \Lambda(x) = \Lambda(x)(1 - \Lambda(x))$
    
    Partiendo de b), sabemos que $\frac{d}{dx}\log \Lambda(x) = 1 - \Lambda(x)$
    
    Derivando nuevamente:
    $$-\frac{d^2}{dx^2}\log(\Lambda(x)) = -\frac{d}{dx}(1 - \Lambda(x)) = -\left[\frac{d}{dx}[1] - \frac{d}{dx}\Lambda(x)\right] = \frac{d}{dx}\Lambda(x)$$
    
    De a) sabemos que $\frac{d}{dx}\Lambda(x) = \Lambda(x)(1 - \Lambda(x))$
    
    Por tanto: $H_{logit}(x) = \Lambda(x)(1 - \Lambda(x))$
    
    ### d) $|H_{logit}(x)| \leq 1$
    
    Podemos interpretar esta desigualdad como:
    $$-1 \leq H_{logit}(x) \leq 1$$
    
    Para la cota inferior:
    $$H_{logit}(x) \geq -1$$
    $$\Lambda(x)(1 - \Lambda(x)) \geq -1$$
    
    Como $\Lambda(x) \in (0,\infty) \forall x \in \mathbb{R}$, es siempre positiva. Por tanto:
    $$\Lambda(x)(1 - \Lambda(x)) \geq 0 \geq -1$$
    
    Para la cota superior, demostramos que $H_{logit}(x)$ tiene un único máximo en $x = 0$:
    
    Derivando $H_{logit}(x)$:
    $$H'_{logit}(x) = \frac{d}{dx}[\Lambda(x)(1 - \Lambda(x))]$$
    
    Aplicando la regla del producto:
    $$H'_{logit}(x) = \frac{d\Lambda(x)}{dx}(1 - \Lambda(x)) + \Lambda(x)\frac{d(1-\Lambda(x))}{dx}$$
    
    Sustituyendo $\frac{d\Lambda(x)}{dx} = \Lambda(x)(1 - \Lambda(x))$:
    $$H'_{logit}(x) = \Lambda(x)(1 - \Lambda(x))(1 - \Lambda(x)) - \Lambda(x)^2(1 - \Lambda(x))$$
    
    Factorizando:
    $$H'_{logit}(x) = \Lambda(x)(1 - \Lambda(x))[(1 - \Lambda(x)) - \Lambda(x)]$$
    $$H'_{logit}(x) = \Lambda(x)(1 - \Lambda(x))[1 - 2\Lambda(x)]$$
    
    Este valor es cero cuando $\Lambda(x) = \frac{1}{2}$, lo que ocurre en $x = 0$.
    
    Evaluando $H_{logit}(0) = \Lambda(0)(1 - \Lambda(0)) = \frac{1}{2}(1 - \frac{1}{2}) = \frac{1}{4} < 1$
    
    Como el punto crítico es un máximo (verificable por la segunda derivada) y $H_{logit}(0) = \frac{1}{4} < 1$, concluimos que $|H_{logit}(x)| \leq 1$ para todo $x$.
    """)
    
    # Gráfica de la distribución logística y su derivada
    st.subheader("Visualización de la Distribución Logística")
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    x = np.linspace(-5, 5, 1000)
    
    # Función logística
    lambda_x = 1 / (1 + np.exp(-x))
    
    # Derivada (demostrada en inciso a)
    d_lambda_x = lambda_x * (1 - lambda_x)
    
    # H_logit (demostrada en inciso c)
    h_logit = d_lambda_x
    
    # Graficar
    ax[0].plot(x, lambda_x, 'b-', linewidth=2)
    ax[0].set_title('Función Logística $\Lambda(x)$')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('$\Lambda(x)$')
    ax[0].grid(True)
    
    ax[1].plot(x, d_lambda_x, 'r-', linewidth=2)
    ax[1].set_title('Derivada $\Lambda(x)(1-\Lambda(x))$')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('$\Lambda\'(x)$')
    ax[1].set_ylim(-0.1, 0.3)
    ax[1].grid(True)
    
    ax[2].plot(x, h_logit, 'g-', linewidth=2)
    ax[2].axhline(y=1, color='k', linestyle='--', alpha=0.7)
    ax[2].axhline(y=-1, color='k', linestyle='--', alpha=0.7)
    ax[2].set_title('$H_{logit}(x)$ con cotas $\pm 1$')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('$H_{logit}(x)$')
    ax[2].set_ylim(-1.1, 1.1)
    ax[2].grid(True)
    
    st.pyplot(fig)

# TAB 2: Ejercicio 4 - Modelo de Variable Latente
def ej2():
    st.header("E.4: Modelo de Variable Latente")
    
    st.markdown(r"""
    Consideramos un modelo de variable latente:
    
    $$Y^* = \beta_0 + X\beta_1 + e$$
    $$e | X \sim N(0, \sigma^2(X))$$
    $$\sigma^2(X) = \gamma_0 + X^2\gamma_1$$
    $$Y = \max(Y^*, 0)$$
    
    donde $X$ es escalar, $\gamma_0 > 0$, y $\gamma_1 > 0$.
    
    ### a) Función de log-verosimilitud condicional
    
    La función de densidad de probabilidad (pdf) de $Y^*$ condicionada en $X$ es normal:
    
    $$\phi(Y^*_i) = \frac{1}{\sqrt{2\pi\sigma^2(X_i)}}\exp\left(-\frac{(Y^*_i - \beta_0 - X_i\beta_1)^2}{2\sigma^2(X_i)}\right)$$
    
    La distribución conjunta para la muestra se construye considerando que:
    
    $$f^*(Y|X) = \sigma^{-2}(X_i)\phi\left(\frac{Y_i-\beta_0-X_i\beta_1}{\sigma^2(X_i)}\right)$$
    
    y que $F(0|X) = 1 - \Phi\left(\frac{\beta_0 + X_i\beta_1}{\sigma(X_i)}\right)$, donde $\Phi$ es la función de distribución acumulada normal estándar.
    
    Para una muestra de observaciones, la función de verosimilitud es:
    
    $$L(\theta|X) = \prod_{i\in\{Y_i>0\}} \frac{1}{\sqrt{2\pi\sigma^2(X_i)}}\exp\left(-\frac{(Y_i - \beta_0 - X_i\beta_1)^2}{2\sigma^2(X_i)}\right) \prod_{i\in\{Y_i=0\}} \Phi\left(-\frac{\beta_0 + X_i\beta_1}{\sigma(X_i)}\right)$$
    
    Aplicando logaritmo natural a ambos lados:
    
    $$\ln(\theta) = \log L(\theta|X) = \sum_{i\in\{Y_i>0\}} \left[-\frac{1}{2}\log(2\pi\sigma^2(X_i)) - \frac{(Y_i - \beta_0 - X_i\beta_1)^2}{2\sigma^2(X_i)}\right] + \sum_{i\in\{Y_i=0\}} \log\Phi\left(-\frac{\beta_0 + X_i\beta_1}{\sigma(X_i)}\right)$$
    
    Sustituyendo $\sigma^2(X_i) = \gamma_0 + X_i^2\gamma_1$:
    
    $$\ln(\theta) = \sum_{i\in\{Y_i>0\}} \left[-\frac{1}{2}\log(2\pi[\gamma_0 + X_i^2\gamma_1]) - \frac{(Y_i - \beta_0 - X_i\beta_1)^2}{2[\gamma_0 + X_i^2\gamma_1]}\right] + \sum_{i\in\{Y_i=0\}} \log\Phi\left(-\frac{\beta_0 + X_i\beta_1}{\sqrt{\gamma_0 + X_i^2\gamma_1}}\right)$$
    
    ### b) Identificación de parámetros
    
    Los parámetros $\beta_0$, $\beta_1$, $\gamma_0$, y $\gamma_1$ están identificados por las siguientes razones:
    
    1. La función de verosimilitud contiene suficiente variación en $X$ e $Y$.
    
    2. Los parámetros $\beta_0$ y $\beta_1$ se identifican por la variación en $X$ e $Y$ cuando la variable dependiente no está censurada. Estos parámetros afectan la media de la distribución normal.
    
    3. Los parámetros $\gamma_0$ y $\gamma_1$ afectan la varianza del modelo, que varía con $X$. La variación en $X$ permite identificar estos parámetros.
    
    4. Como $\sigma$ aparece de forma separada de los parámetros $\beta$, hay suficiente información para identificar todos los parámetros.
    
    La identificación se puede verificar también comprobando la concavidad global de la función de log-verosimilitud en el dominio de los parámetros, lo que garantizaría un único máximo global.
    """)
    
    # Visualización del modelo de variable latente censurada
    st.subheader("Visualización del Modelo de Variable Latente Censurada")
    
    beta0 = 1.0
    beta1 = 2.0
    gamma0 = 1.0
    gamma1 = 0.5
    
    # Función para generar Y*
    @st.cache_data
    def generate_latent_data(x, beta0, beta1, gamma0, gamma1, seed=42):
        np.random.seed(seed)
        sigma2 = gamma0 + x**2 * gamma1
        e = np.random.normal(0, np.sqrt(sigma2))
        y_star = beta0 + beta1 * x + e
        y = np.maximum(y_star, 0)
        return y_star, y, sigma2
    
    x = np.linspace(-3, 3, 100)
    y_star, y, sigma2 = generate_latent_data(x, beta0, beta1, gamma0, gamma1)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graficar Y* y Y
    ax[0].plot(x, y_star, 'b-', label='$Y^*$ (Variable Latente)')
    ax[0].plot(x, y, 'r-', label='$Y$ (Variable Observada)')
    ax[0].axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax[0].set_title('Modelo de Variable Latente Censurada')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].legend()
    ax[0].grid(True)
    
    # Graficar la varianza heteroscedástica
    ax[1].plot(x, sigma2, 'g-', linewidth=2)
    ax[1].set_title('Varianza Heteroscedástica $\sigma^2(X) = \gamma_0 + X^2\gamma_1$')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('$\sigma^2(X)$')
    ax[1].grid(True)
    
    st.pyplot(fig)

# TAB 3: Ejercicio 5 - Modelos de Selección
def ej3():
    st.header("E.5: Modelos de Selección")
    
    st.markdown(r"""
    Consideramos el modelo:
    
    $$S = 1\{X'\gamma + u > 0\}$$
    
    $$Y = \begin{cases} 
    X'\beta + e & \text{si } S = 1 \\
    \text{missing} & \text{si } S = 0
    \end{cases}$$
    
    $$\left(\begin{array}{c} e \\ u \end{array}\right) \sim N\left(0, \left(\begin{array}{cc} \sigma^2 & \sigma_{21} \\ \sigma_{21} & 1 \end{array}\right)\right)$$
    
    Demostraremos que $\mathbb{E}[Y|X, S=1] = X'\beta + \sigma_{21}\lambda(X'\gamma)$.
    
    ### Demostración
    
    Podemos expresar la esperanza condicional como:
    
    $$\mathbb{E}(Y|X, S=1) = \mathbb{E}(Y|S=1, X) = \mathbb{E}(X'\beta + e|X'\gamma + u > 0, X)$$
    
    Simplificando:
    
    $$\mathbb{E}(Y|X, S=1) = X'\beta + \mathbb{E}(e|u > -X'\gamma, X)$$
    
    Dado que $e$ y $u$ siguen una distribución normal multivariada, podemos expresar $e$ en términos de $u$:
    
    $$e = \sigma_{21}u + \epsilon$$
    
    donde $\epsilon$ es independiente de $u$.
    
    Entonces:
    
    $$\mathbb{E}(Y|X, S=1) = X'\beta + \mathbb{E}(\sigma_{21}u + \epsilon|u > -X'\gamma, X)$$
    
    $$\mathbb{E}(Y|X, S=1) = X'\beta + \sigma_{21}\mathbb{E}(u|u > -X'\gamma)$$
    
    Para una variable normal estándar $u \sim N(0,1)$, tenemos:
    
    $$\mathbb{E}(u|u > c) = \frac{\phi(c)}{1-\Phi(c)}$$
    
    donde $\phi(\cdot)$ es la función de densidad normal estándar y $\Phi(\cdot)$ es la función de distribución acumulada.
    
    Definiendo la razón inversa de Mills como $\lambda(c) = \frac{\phi(c)}{1-\Phi(c)}$, para $c = -X'\gamma$:
    
    $$\mathbb{E}(u|u > -X'\gamma) = \lambda(X'\gamma)$$
    
    Por tanto:
    
    $$\mathbb{E}[Y|X, S=1] = X'\beta + \sigma_{21}\lambda(X'\gamma)$$
    
    Este resultado muestra cómo la selección no aleatoria afecta la esperanza condicional, introduciendo un término de corrección proporcional a la razón inversa de Mills.
    """)
    
    # Visualización del sesgo de selección de muestra
    st.subheader("Visualización del Sesgo de Selección")
    
    beta = 1.5
    gamma = 1.0
    sigma21 = 0.8
    
    # Generando datos
    @st.cache_data
    def generate_selection_data(n=1000, beta=1.5, gamma=1.0, sigma21=0.8, seed=123):
        np.random.seed(seed)
        
        # Matriz de covarianza
        cov_matrix = np.array([[1, sigma21], [sigma21, 1]])
        
        # Generar variables aleatorias normales multivariadas
        errors = np.random.multivariate_normal([0, 0], cov_matrix, n)
        e = errors[:, 0]
        u = errors[:, 1]
        
        # Generar X
        x = np.random.normal(0, 1, n)
        
        # Modelo de selección
        s_star = gamma * x + u
        s = (s_star > 0).astype(int)
        
        # Modelo principal
        y_star = beta * x + e
        
        # Y observado solo si S=1
        y = np.where(s == 1, y_star, np.nan)
        
        # Calcular lambda (razón inversa de Mills)
        from scipy.stats import norm
        lambda_x = norm.pdf(-gamma * x) / (1 - norm.cdf(-gamma * x))
        
        # Y corregido por selección
        y_corrected = beta * x + sigma21 * lambda_x
        
        return x, y, s, y_corrected, y_star
    
    x, y, s, y_corrected, y_star = generate_selection_data()
    
    # Convertir a DataFrame para fácil manipulación
    data = pd.DataFrame({
        'x': x,
        'y_star': y_star,
        'y': y,
        's': s,
        'y_corrected': y_corrected
    })
    
    # Crear gráfica
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Primer gráfico: Datos completos vs seleccionados
    ax[0].scatter(data[data['s'] == 1]['x'], data[data['s'] == 1]['y'], alpha=0.6, label='Observados (S=1)')
    ax[0].scatter(data[data['s'] == 0]['x'], data[data['s'] == 0]['y_star'], alpha=0.3, color='gray', label='No observados (S=0)')
    
    # Línea de regresión verdadera
    x_line = np.linspace(-3, 3, 100)
    y_line = beta * x_line
    ax[0].plot(x_line, y_line, 'r-', label='E[Y*|X] = β·X')
    
    ax[0].set_title('Datos Observados vs No Observados')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].legend()
    ax[0].grid(True)
    
    # Segundo gráfico: Corrección por selección
    ax[1].scatter(data[data['s'] == 1]['x'], data[data['s'] == 1]['y'], alpha=0.6, label='Observados (S=1)')
    
    # Línea de regresión verdadera
    ax[1].plot(x_line, y_line, 'r-', label='E[Y*|X] = β·X')
    
    # Línea de E[Y|X,S=1]
    y_corrected_line = beta * x_line + sigma21 * norm.pdf(-gamma * x_line) / (1 - norm.cdf(-gamma * x_line))
    ax[1].plot(x_line, y_corrected_line, 'g-', label='E[Y|X,S=1] = β·X + σ₂₁·λ(X\'γ)')
    
    ax[1].set_title('Corrección del Sesgo de Selección')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].legend()
    ax[1].grid(True)
    
    st.pyplot(fig)

# TAB 4: Ejercicio 6 - Distribución Poisson
def ej4():
    """
    Ejercicio 6: Distribución Poisson con Efectos No Observados
    
    Pr{Y = j|X, ε} = exp(-λ(X, ε))λ(X, ε)^j/j!
    λ(X, ε) = exp(β₀ + β₁X), para j = 1, 2, ...
    β₀ = γ₀ + ε, con γ₀ y β₁ parámetros escalares desconocidos
    ε es una variable aleatoria no observada con E[ε] = 0 y var(ε) = σ² > 0
    ε es estadísticamente independiente de X.
    """

    st.header("E.6: Distribución Poisson con Efectos No Observados")
    
    # Parámetros para visualización
    gamma0 = 0.5
    beta1 = 1.0
    sigma2 = 0.5  # varianza de epsilon
    
    # Demostración teórica
    st.markdown(r"""
    Consideramos el modelo:
    
    $$\text{Pr}\{Y = j|X, \varepsilon\} = \exp(-\lambda(X, \varepsilon))\lambda(X, \varepsilon)^j/j!$$
    
    donde $\lambda(X, \varepsilon) = \exp(\beta_0 + \beta_1 X)$, para $j = 1, 2, \ldots$, $\beta_0 = \gamma_0 + \varepsilon$, con $\gamma_0$ y $\beta_1$ parámetros escalares desconocidos, y $\varepsilon$ es una variable aleatoria no observada con $\mathbb{E}[\varepsilon] = 0$ y $\text{var}(\varepsilon) = \sigma^2 > 0$, estadísticamente independiente de la variable aleatoria escalar $X$.
    
    ### a) Demostrar que $\lambda(X, \varepsilon) = \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \times W$
    
    Partimos de $\lambda(X, \varepsilon) = \exp(\beta_0 + \beta_1 X) = \exp(\gamma_0 + \varepsilon + \beta_1 X)$
    
    Podemos reescribir esto como:
    
    $$\lambda(X, \varepsilon) = \exp(\gamma_0 + \beta_1 X) \cdot \exp(\varepsilon)$$
    
    Definamos $W = \frac{\exp(\varepsilon)}{\mathbb{E}[\exp(\varepsilon)]}$, entonces:
    
    $$\exp(\varepsilon) = \mathbb{E}[\exp(\varepsilon)] \cdot W$$
    
    Sustituyendo:
    
    $$\lambda(X, \varepsilon) = \exp(\gamma_0 + \beta_1 X) \cdot \mathbb{E}[\exp(\varepsilon)] \cdot W$$
    
    $$\lambda(X, \varepsilon) = \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \cdot W$$
    
    ### b) Demostrar que $\mathbb{E}(W) = 1$
    
    Por definición:
    
    $$W = \frac{\exp(\varepsilon)}{\mathbb{E}[\exp(\varepsilon)]}$$
    
    Calculando la esperanza:
    
    $$\mathbb{E}(W) = \mathbb{E}\left(\frac{\exp(\varepsilon)}{\mathbb{E}[\exp(\varepsilon)]}\right) = \frac{1}{\mathbb{E}[\exp(\varepsilon)]}\mathbb{E}[\exp(\varepsilon)] = 1$$
    
    Nota: $\mathbb{E}[\exp(\varepsilon)]$ no es una variable aleatoria y puede salir de la esperanza.
    
    ### c) Demostrar que $\mathbb{E}[Y|X] = \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$
    
    Sabemos que para una distribución Poisson con parámetro $\lambda$, $\mathbb{E}[Y|\lambda] = \lambda$.
    
    Por tanto:
    
    $$\mathbb{E}[Y|X, \varepsilon] = \lambda(X, \varepsilon)$$
    
    Para encontrar $\mathbb{E}[Y|X]$, usamos la ley de expectativas iteradas:
    
    $$\mathbb{E}[Y|X] = \mathbb{E}[\mathbb{E}[Y|X, \varepsilon]]$$
    
    $$\mathbb{E}[Y|X] = \mathbb{E}[\lambda(X, \varepsilon)]$$
    
    Sustituyendo la expresión de $\lambda(X, \varepsilon)$ del inciso a):
    
    $$\mathbb{E}[Y|X] = \mathbb{E}[\exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \cdot W]$$
    
    Como $\exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$ no depende de $\varepsilon$, podemos sacarlo de la esperanza:
    
    $$\mathbb{E}[Y|X] = \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \cdot \mathbb{E}[W]$$
    
    Del inciso b), sabemos que $\mathbb{E}[W] = 1$, por lo tanto:
    
    $$\mathbb{E}[Y|X] = \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$$
    
    ### d) Encontrar una expresión para $\mathbb{E}[Y^2|X]$
    
    Para una variable aleatoria Poisson con parámetro $\lambda$, sabemos que:
    
    $$\mathbb{E}[Y^2|\lambda] = \lambda^2 + \lambda$$
    
    Entonces:
    
    $$\mathbb{E}[Y^2|X, \varepsilon] = \lambda(X, \varepsilon)^2 + \lambda(X, \varepsilon)$$
    
    Para encontrar $\mathbb{E}[Y^2|X]$, usamos la ley de expectativas iteradas:
    
    $$\mathbb{E}[Y^2|X] = \mathbb{E}[\mathbb{E}[Y^2|X, \varepsilon]]$$
    
    $$\mathbb{E}[Y^2|X] = \mathbb{E}[\lambda(X, \varepsilon)^2 + \lambda(X, \varepsilon)]$$
    
    Sustituyendo la expresión de $\lambda(X, \varepsilon)$:
    
    $$\mathbb{E}[Y^2|X] = \mathbb{E}[(\exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \cdot W)^2 + \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \cdot W]$$
    
    $$\mathbb{E}[Y^2|X] = \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])) \cdot \mathbb{E}[W^2] + \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) \cdot \mathbb{E}[W]$$
    
    Sabemos que $\mathbb{E}[W] = 1$, por lo tanto:
    
    $$\mathbb{E}[Y^2|X] = \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])) \cdot \mathbb{E}[W^2] + \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$$
    
    ### e) Encontrar una expresión para $\mathbb{E}^2[Y|X]$
    
    Del inciso c), tenemos:
    
    $$\mathbb{E}[Y|X] = \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$$
    
    Por lo tanto:
    
    $$\mathbb{E}^2[Y|X] = (\exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]))^2 = \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]))$$
    
    ### f) Demostrar que $\text{Var}(Y|X) > \mathbb{E}[Y|X]$
    
    La varianza condicional se define como:
    
    $$\text{Var}(Y|X) = \mathbb{E}[Y^2|X] - \mathbb{E}^2[Y|X]$$
    
    Sustituyendo las expresiones obtenidas en los incisos d) y e):
    
    $$\text{Var}(Y|X) = \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])) \cdot \mathbb{E}[W^2] + \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]) - \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]))$$
    
    Factorizando:
    
    $$\text{Var}(Y|X) = \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])) \cdot (\mathbb{E}[W^2] - 1) + \exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$$
    
    Notemos que el término $\exp(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])$ es precisamente $\mathbb{E}[Y|X]$. 
    
    Además, por la desigualdad de Jensen, sabemos que $\mathbb{E}[W^2] > (\mathbb{E}[W])^2 = 1$ para cualquier variable aleatoria no constante.
    
    Por lo tanto:
    
    $$\text{Var}(Y|X) = \mathbb{E}[Y|X] + \exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)])) \cdot (\mathbb{E}[W^2] - 1)$$
    
    Como $\mathbb{E}[W^2] - 1 > 0$ y el término $\exp(2(\gamma_0 + \beta_1 X + \ln \mathbb{E}[\exp(\varepsilon)]))$ es siempre positivo, tenemos:
    
    $$\text{Var}(Y|X) > \mathbb{E}[Y|X]$$
    
    Este resultado demuestra que la varianza es mayor que la media, lo cual es una característica de la sobredispersión. Este fenómeno es común en los datos de conteo cuando hay heterogeneidad no observada, como en este caso con la inclusión del término de error $\varepsilon$.
    """)
    
    # Visualización de la sobredispersión en distribución Poisson con efectos aleatorios
    st.subheader("Visualización de la Sobredispersión")
    
    # Función para generar datos de una distribución Poisson con efectos aleatorios
    def generate_poisson_data(x_values, gamma0, beta1, sigma, n_samples=100, seed=42):
        np.random.seed(seed)
        x_repeated = np.repeat(x_values, n_samples)
        
        # Generar epsilon ~ N(0, sigma²)
        epsilon = np.random.normal(0, sigma, len(x_repeated))
        
        # Calcular lambda
        log_lambda = gamma0 + beta1 * x_repeated + epsilon
        lambda_values = np.exp(log_lambda)
        
        # Generar datos Poisson
        y = np.random.poisson(lambda_values)
        
        return x_repeated, y, lambda_values
    
    x_range = np.linspace(-2, 2, 20)
    x, y, lambda_values = generate_poisson_data(x_range, gamma0, beta1, np.sqrt(sigma2))
    
    # Calcular valor esperado teórico
    def expected_value(x, gamma0, beta1, sigma2):
        # E[exp(epsilon)] para epsilon ~ N(0, sigma²) es exp(sigma²/2)
        E_exp_epsilon = np.exp(sigma2/2)
        return np.exp(gamma0 + beta1 * x + np.log(E_exp_epsilon))
    
    x_smooth = np.linspace(-2, 2, 100)
    expectation = expected_value(x_smooth, gamma0, beta1, sigma2)
    
    # Calculando media y varianza empíricas
    bins = {}
    for xi in x_range:
        mask = np.isclose(x, xi, rtol=1e-10)
        bins[xi] = y[mask]
    
    empirical_means = [np.mean(bins[xi]) for xi in x_range]
    empirical_vars = [np.var(bins[xi]) for xi in x_range]
    
    # Crear visualización
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot de datos con línea de esperanza
    ax[0].scatter(x, y, alpha=0.3, label='Observaciones')
    ax[0].plot(x_smooth, expectation, 'r-', linewidth=2, label='E[Y|X]')
    ax[0].set_title('Distribución Poisson con Efectos Aleatorios')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].legend()
    ax[0].grid(True)
    
    # Comparación de media y varianza empíricas
    ax[1].scatter(x_range, empirical_means, color='blue', label='Media Empírica')
    ax[1].scatter(x_range, empirical_vars, color='red', label='Varianza Empírica')
    ax[1].plot(x_range, empirical_means, 'b-', alpha=0.5)
    ax[1].plot(x_range, empirical_vars, 'r-', alpha=0.5)
    ax[1].plot(x_range, empirical_means, 'k--', alpha=0.7, label='Línea de Referencia')
    ax[1].set_title('Sobredispersión: Varianza > Media')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Estadísticas Empíricas')
    ax[1].legend()
    ax[1].grid(True)
    
    st.pyplot(fig)
    
    # Conclusión
    st.markdown("""
    **Conclusión:** La gráfica muestra claramente que la varianza empírica (puntos rojos) es sistemáticamente mayor 
    que la media empírica (puntos azules) para todos los valores de X, confirmando nuestra demostración teórica 
    de que Var(Y|X) > E[Y|X]. Esta sobredispersión es una consecuencia directa de la heterogeneidad no observada 
    introducida por el término de error aleatorio ε.
    """)




# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    'Ejercicio 1': ej1,
    'Ejercicio 2': ej2,
    'Ejercicio 3': ej3,
    'Ejercicio 4': ej4
    # Agregar funciones para las demás tareas
}

st.write('---')
selected = option_menu('Selección de Ejercicio', options=list(task_functions.keys()), 
    icons=['book' for i in task_functions.keys()], default_index=0,orientation="horizontal")

# Llama a la función de tarea seleccionada
if selected in task_functions:
    task_functions[selected]()
else:
    st.write('Selecciona un Ejercicio válido en el menú de opciones.')

