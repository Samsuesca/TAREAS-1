import os
import shutil

# Directorio de origen del código común
source_code_path = "common_code.py"

# Directorio donde se encuentran tus archivos .py
target_directory = "tasks"

# Obtén una lista de los archivos .py en el directorio de destino
py_files = [f for f in os.listdir(target_directory) if f.endswith(".py")]

# Copia el código común en cada archivo .py
for py_file in py_files:
    target_file = os.path.join(target_directory, py_file)
    # Lee el contenido del archivo .py
    with open(target_file, "r") as f:
        content = f.readlines()
    
    # Abre el archivo .py en modo escritura
    with open(target_file, "w") as f:
        # Escribe el código común al principio del archivo
        with open(source_code_path, "r") as common_code:
            common_content = common_code.readlines()
            f.writelines(common_content)
        # Escribe el contenido original del archivo después del código común
        f.writelines(content)

print("El código común se ha copiado en todos los archivos .py en el directorio de destino.")
