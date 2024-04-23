from cx_Freeze import setup, Executable
from setuptools import find_packages

# Reemplaza 'app.py' con el nombre de tu archivo principal
executables = [Executable("main.py")]

# Busca automáticamente todos los paquetes en tu proyecto
packages = find_packages()
# Lista de paquetes adicionales que deseas incluir
packages += ["tensorflow"]

# Agrega cualquier otro archivo o paquete necesario
additional_files = [("templates", "templates"), ("static", "static"), ("latest.h5", "latest.h5")]

# Configuración de la aplicación
setup(
    name="mi_aplicacion",
    version="1.0",
    description="Descripción de tu aplicación",
    executables=executables,
    options={"build_exe": {"packages": packages, "include_files": additional_files}},
)