import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='S8',
                    page_icon=':book:',
                    layout='wide')


# Título de la semana
st.title('Semana 8 - Método de los Momentos y Método Generalizado de los Momentos:')


# Funciones para las tareas específicas
def task_1():
    st.subheader('Tarea 1: Descripción de la tarea 1')
    # Agregar contenido de la tarea 1

# Define un diccionario para mapear la selección a la función de tarea correspondiente
task_functions = {
    'Tarea 1': task_1
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
