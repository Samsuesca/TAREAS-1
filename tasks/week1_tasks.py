import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Variable global para la longitud de la lista de opciones
num_options = 2

# Título de la semana
st.title('Semana 1 - Teoría Asintótica')

# Menú de opciones
option = st.sidebar.selectbox('Selecciona una tarea:', range(1, num_options + 1))

# Función para la tarea 1
def task_1():
    # Título de la página
    st.title("Consistencia del Estimador MCO")

    # Descripción del problema
    st.write("Vamos a estudiar la consistencia del estimador MCO utilizando simulación. El proceso generador de datos será:")

    st.latex(r"Y_i = 1 + 0.5X_i + \epsilon_i")

    st.write("donde:")
    st.write(r"$X_i \sim N(0, 2^2)$")
    st.write(r"$\epsilon_i \sim U(-1, 1)$")
    st.write("para valores de $n$ en {50, 500, 5000, 50000}.")

    # Función para la simulación y cálculo de estimadores MCO
    def simulate_MCO(n):
        Xi = np.random.normal(0, 2, n)
        ei = np.random.uniform(-1, 1, n)
        Yi = 1 + 0.5 * Xi + ei
        X = np.column_stack((np.ones(n), Xi))
        inv_XX = np.linalg.inv(np.dot(X.T, X) / n)
        beta_hat = np.dot(np.dot(inv_XX, X.T), Yi) / n
        return beta_hat[1]  # Retorna el estimador MCO de beta

    # Simulación y cálculo de estimadores MCO para diferentes tamaños de muestra
    n_values = [50, 500, 5000, 50000]
    estimated_betas = [simulate_MCO(n) for n in n_values]

    # Gráfico de convergencia
    st.write("Estimaciones de MCO para diferentes tamaños de muestra:")
    st.write(estimated_betas)

    plt.plot(n_values, estimated_betas, marker='o')
    plt.axhline(0.5, color='red', linestyle='--', label='Valor Verdadero')
    plt.xlabel('Tamaño de Muestra (n)')
    plt.ylabel('Estimador MCO de Beta')
    plt.legend()
    st.pyplot(plt)

    # Descripción de la convergencia en distribución
    st.write("La convergencia en distribución se da por el Teorema Central del Límite (TCL).")
    st.write("En particular, tenemos que:")
    st.latex(r"\frac{(\hat{\beta}_n - \beta)}{\sqrt{Var(\hat{\beta}_n)}} \stackrel{d}{\rightarrow} N(0,1)")

    # Simulación para la convergencia en distribución
    def simulate_distribution_convergence(n):
        Zs = []
        Xi = np.random.normal(0, 2, n)
        X = np.column_stack((np.ones(n), Xi))
        XX = np.dot(X.T, X) / n
        XXi = np.linalg.inv(XX)
        xxi2 = XXi[1, 1]
        for _ in range(1000):
            ei = np.random.normal(0, 1, n)
            Yi = 1 + 0.5 * Xi + ei
            beta_hat = np.dot(np.dot(XXi, X.T), Yi) / n
            res = Yi - np.dot(X, beta_hat)
            sigma = np.std(res)
            Z = np.sqrt(n) * (beta_hat[1] - 0.5) / (sigma * np.sqrt(xxi2))
            Zs.append(Z)
        return Zs

    # Simulación de convergencia en distribución
    distribution_convergence_data = simulate_distribution_convergence(1000)

    # Gráfico de convergencia en distribución
    st.write("Verificación de Convergencia en Distribución:")
    st.write("La línea roja representa una distribución normal estándar (N(0,1)).")
    plt.hist(distribution_convergence_data, bins=50, density=True, alpha=0.6, color='g')
    plt.plot(np.linspace(-3, 3, 100), stats.norm.pdf(np.linspace(-3, 3, 100), 0, 1), 'r-', lw=2, label='N(0,1)')
    plt.xlabel('Z')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
st.pyplot(plt)


# Función para la tarea 2
def task_2():
    st.subheader('Tarea 2: Regresión Lineal y Verificación de Convergencia')
    st.markdown("En esta tarea, realizaremos una regresión lineal y verificaremos la convergencia de los estimadores.")

    # Generar datos simulados
    np.random.seed(42)
    N = 1000
    X = np.random.normal(1, 2, N)
    epsilon = np.random.uniform(-1, 1, N)
    Y = 1 + 0.5 * X + epsilon

    # Realizar la regresión lineal
    beta_hat = np.cov(X, Y)[0, 1] / np.var(X)
    var_beta_hat = np.var(Y) / (N * np.var(X))

    st.write("Parámetros verdaderos:")
    st.write("Beta_0 (intercepto) = 1")
    st.write("Beta_1 (pendiente) = 0.5")

    st.write("Estimaciones de MCO:")
    st.write(f"Beta_0 estimado = {beta_hat:.4f}")
    st.write(f"Beta_1 estimado = {var_beta_hat:.4f}")

    # Verificar la convergencia
    st.subheader("Verificación de Convergencia")
    st.markdown("Verificaremos la convergencia de Beta_1 estimado a 0.5 y la convergencia de la raíz de N * (Beta_1 estimado - 0.5) a una distribución normal.")

    # Comprobar si Beta_1 estimado converge a 0.5
    if np.isclose(beta_hat, 0.5, atol=0.1):
        st.success("Beta_1 estimado converge a 0.5.")
    else:
        st.error("Beta_1 estimado no converge a 0.5.")

    # Comprobar la convergencia de la raíz de N * (Beta_1 estimado - 0.5)
    z_score = (beta_hat - 0.5) / np.sqrt(var_beta_hat)
    if np.abs(z_score) < 1.96:
        st.success("Raíz de N * (Beta_1 estimado - 0.5) converge a una distribución normal.")
    else:
        st.error("Raíz de N * (Beta_1 estimado - 0.5) no converge a una distribución normal.")

# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    1: task_1,
    2: task_2
}

# Llama a la función de tarea seleccionada
if option in task_functions:
    task_functions[option]()
else:
    st.write('Selecciona una tarea válida en el menú de opciones.')
