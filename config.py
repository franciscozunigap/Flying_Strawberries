import subprocess

# Leer los requerimientos desde un archivo y devolverlos como un conjunto.
def leer_requerimientos(ruta):
    with open(ruta, 'r') as f:
        return {line.strip() for line in f if line.strip()}

# Obtener un conjunto de las dependencias instaladas.
def dependencias_instaladas():
    resultado = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    return {line.split('==')[0] for line in resultado.stdout.splitlines()}

#Verificar e instalar dependencias a partir de un archivo requirements.txt.
def instalar_dependencias(ruta_requerimientos):
    # Leer requerimientos del archivo
    requerimientos = leer_requerimientos(ruta_requerimientos)

    # Obtener dependencias instaladas
    instaladas = dependencias_instaladas()

    # Verificar si todas las dependencias ya están instaladas
    falta_instalar = requerimientos - instaladas

    if falta_instalar:
        print("Las siguientes dependencias no están instaladas:")
        print(falta_instalar)

        # Ejecutar el comando para instalar las dependencias que faltan
        comando = "pip install -r " + ruta_requerimientos
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)

        # Imprimir la salida
        print("Salida del comando:")
        print(resultado.stdout)

        # Imprimir cualquier error
        if resultado.stderr:
            print("Error:")
            print(resultado.stderr)
    else:
        print("Todas las dependencias ya están instaladas.")
