import streamlit as st

# Variable global para la longitud de la lista de opciones
num_options = 2  # Puedes cambiar esto según tus necesidades

# Título de la semana
st.title('Semana 3 - ')

# Menú de opciones
option = st.sidebar.selectbox('Selecciona una tarea:', range(1, num_options + 1))

# Funciones para las tareas específicas
def task_1():
    st.subheader('Tarea 1: Descripción de la tarea 1')
    # Agregar contenido de la tarea 1

def task_2():
    st.subheader('Tarea 2: Descripción de la tarea 2')
    # Agregar contenido de la tarea 2

# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    1: task_1,
    2: task_2
    # Agregar funciones para las demás tareas
}

# Llama a la función de tarea seleccionada
if option in task_functions:
    task_functions[option]()
else:
    st.write('Selecciona una tarea válida en el menú de opciones.')
