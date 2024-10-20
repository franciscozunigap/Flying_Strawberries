import subprocess

# Leer los requerimientos desde un archivo y devolverlos como un conjunto.
def leer_requerimientos(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        return {linea.strip() for linea in archivo if linea.strip()}

# Obtener un conjunto de las dependencias instaladas.
def obtener_dependencias_instaladas():
    resultado = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    return {linea.split('==')[0] for linea in resultado.stdout.splitlines()}

# Verificar e instalar dependencias a partir de un archivo requirements.txt.
def instalar_dependencias(ruta_requerimientos):
    # Leer requerimientos del archivo
    requerimientos_necesarios = leer_requerimientos(ruta_requerimientos)

    # Obtener dependencias instaladas
    dependencias_instaladas = obtener_dependencias_instaladas()

    # Verificar si todas las dependencias ya están instaladas
    dependencias_faltantes = requerimientos_necesarios - dependencias_instaladas

    if dependencias_faltantes:
        print("Las siguientes dependencias no están instaladas:")
        print(dependencias_faltantes)

        # Ejecutar el comando para instalar las dependencias que faltan
        comando_instalacion = "pip install -r " + ruta_requerimientos
        resultado = subprocess.run(comando_instalacion, shell=True, capture_output=True, text=True)

        # Imprimir la salida
        print("Salida del comando:")
        print(resultado.stdout)

        # Imprimir cualquier error
        if resultado.stderr:
            print("Error:")
            print(resultado.stderr)
    else:
        print("Todas las dependencias ya están instaladas.")

ruta_requerimientos = "requirements.txt"
instalar_dependencias(ruta_requerimientos)