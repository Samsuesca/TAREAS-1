import streamlit as st
from st_pages import Page, show_pages, add_page_title

# Variables globales
nombre = "Tu Nombre"
correo = "tu_correo@example.com"
id = "tu_id"
github_url = "https://github.com/tu_usuario/tu_proyecto"


pages = [
    Page("app.py", "🏠 Inicio"),
    Page("tasks/week1_tasks.py", " Semana", "1️⃣ "),
    Page("tasks/week2_tasks.py", " Semana", "2️⃣ "),
    Page("tasks/week3_tasks.py", " Semana", "3️⃣ "),
    Page("tasks/week4_tasks.py", " Semana", "4️⃣ "),
    Page("tasks/week5_tasks.py", " Semana", "5️⃣ "),
    Page("tasks/week6_tasks.py", " Semana", "6️⃣ "),
    Page("tasks/week7_tasks.py", " Semana", "7️⃣ "),
    Page("tasks/week8_tasks.py", " Semana", "8️⃣ ")
]

# Inicializa la estructura de páginas
add_page_title()

# Muestra las páginas en la barra lateral
show_pages(pages)

# Contenido de la página de inicio
st.title(f'Bienvenido, {nombre}!')
st.write(f'Correo: {correo}')
st.write(f'ID: {id}')
st.write(f'Directorio de GitHub del proyecto: [link]({github_url})')



# import streamlit as st
# import os

# # Variables globales
# nombre = "Tu Nombre"
# correo = "tu_correo@example.com"
# id = "tu_id"
# github_url = "https://github.com/tu_usuario/tu_proyecto"

# # Lista de semanas disponibles en el proyecto
# available_weeks = [f"Semana {i}" for i in range(1, 9)]

# # Título de la aplicación
# st.title('Tareas de Tópicos en Econometría')

# # Barra de navegación
# navigation_option = st.sidebar.radio('Navegación', ('Inicio', 'Selecciona una Semana'))

# if navigation_option == 'Inicio':
#     # Página de bienvenida
#     st.header(f'By {nombre}!')
#     st.write(f'Correo: {correo}')
#     st.write(f'ID: {id}')
#     st.write(f'Directorios de GitHub del proyecto: [link]({github_url})')
# else:
#     # Sidebar para la navegación entre semanas
#     selected_week = st.sidebar.selectbox('Selecciona la Semana:', available_weeks)

#     # Importa dinámicamente el archivo de tareas de la semana seleccionada
#     week_tasks = f'tasks/{selected_week.lower().replace(" Semana","week").replace(" ", "")}_tasks.py'
#     if os.path.exists(week_tasks):
#         st.sidebar.info(f"Has seleccionado la {selected_week}.")
#         with open(week_tasks, "r") as task_file:
#             exec(task_file.read())  # Ejecuta el archivo de tareas seleccionado
#     else:
#         st.sidebar.error("Archivo de tareas no encontrado para la semana seleccionada.")

# # Botón para volver a la página de inicio
# if navigation_option != 'Inicio':
#     st.sidebar.write('---')
#     if st.sidebar.button('Volver a Inicio'):
#         st.sidebar.radio('Navegación', ('Inicio', 'Selecciona una Semana'), index=0)
