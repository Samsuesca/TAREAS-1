import streamlit as st
from st_pages import Page, show_pages, add_page_title

st.set_page_config(page_title='TAREAS TÓPICOS EN ECONÓMETRIA',
                    page_icon=':book:',
                    layout='wide')

# Variables globales
nombre = "Tu Nombre"
correo = "tu_correo@example.com"
id = "tu_id"
github_url = "https://github.com/tu_usuario/tu_proyecto"


pages = [
    Page("app.py", "🏠 Inicio"),
    Page("tasks/week1_tasks.py", " Semana 1", "1️⃣ "),
    Page("tasks/week2_tasks.py", " Semana 2", "2️⃣ "),
    Page("tasks/week3_tasks.py", " Semana 3", "3️⃣ "),
    Page("tasks/week4_tasks.py", " Semana 4", "4️⃣ "),
    Page("tasks/week5_tasks.py", " Semana 5", "5️⃣ "),
    Page("tasks/week6_tasks.py", " Semana 6", "6️⃣ "),
    Page("tasks/week8_tasks.py", " Semana 8", "8️⃣ ")
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

