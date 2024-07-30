import numpy as np
import random

# Crear el vector columna A de 20 individuos binarios aleatorios de tipo string
A = [''.join(random.choice('01') for _ in range(10)) for _ in range(20)]

# Crear el vector columna B de 20 números aleatorios comprendidos en el intervalo (0, 1)
B = np.round(np.random.rand(20), 2)

# Función para mutar un alelo aleatorio en un cromosoma
def mutar_cromosoma(cromosoma):
    indice = random.randint(0, len(cromosoma) - 1)
    alelo_nuevo = '1' if cromosoma[indice] == '0' else '0'
    cromosoma_mutado = cromosoma[:indice] + alelo_nuevo + cromosoma[indice + 1:]
    return cromosoma_mutado

# Almacenar los cromosomas mutados en el vector columna C
C = [mutar_cromosoma(A[i]) if B[i] < 0.09 else A[i] for i in range(20)]

# Mostrar los cromosomas originales, los números aleatorios y los cromosomas mutados por consola con índices
print("Comparación de Vectores A (Original), B (Aleatorio) y C (Mutado):")
for i, (cromosoma_a, numero_b, cromosoma_c) in enumerate(zip(A, B, C), start=1):
    modificado = "Sí" if cromosoma_a != cromosoma_c else "No"
    if modificado == "Sí":
        print(f"[{i}] A: {cromosoma_a}, B: {numero_b:.2f}, C: {cromosoma_c}, Modificado")
    else:
        print(f"[{i}] A: {cromosoma_a}, B: {numero_b:.2f}, C: {cromosoma_c}")