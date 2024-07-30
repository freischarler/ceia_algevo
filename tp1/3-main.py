###################################################################
# Algoritmo Genetico que encuentra el maximo de la funcion g = (2 * c) / (4+0.8*c+c ** 2+0.2*c ** 3)
# Seleccion por ruleta
# Pc = 0.85
# Pm = 0.07
###################################################################
import random
import matplotlib.pyplot as plt
import numpy as np


# Parámetros
TAMANIO_POBLACION = 4
# Para 2 decimales entre [0,10] necesitamos 2^10 = 1024 valores posibles
LONGITUD_CROMOSOMA = 10
TASA_MUTACION = 0.07
TASA_CRUCE = 0.85
GENERACIONES = 10

###################################################################
# Aptitud (g = (2 * c) / (4+0.8*c+c ** 2+0.2*c ** 3))
###################################################################
def aptitud(cromosoma):
    c = int(cromosoma, 2)
    # Ajustar el valor para reflejar los 2 decimales de precisión
    c = c / 100.0
    return (2 * c) / (4+0.8*c+c ** 2+0.2*c ** 3)

###################################################################
# Inicializar la población
###################################################################
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for z in range(tamanio_poblacion):
        cromosoma = ""
        for t in range(longitud_cromosoma):
            cromosoma = cromosoma+str(random.randint(0, 1))
        poblacion.append(cromosoma)
    return poblacion

###################################################################
# Seleccion por ruleta
###################################################################
def seleccion_ruleta(poblacion, aptitud_total):
    seleccion = random.uniform(0, aptitud_total)
    aptitud_actual = 0
    for individuo in poblacion:
        aptitud_actual = aptitud_actual+aptitud(individuo)
        if aptitud_actual > seleccion:
            return individuo

###################################################################
# Cruce monopunto con probabilidad de cruza pc = 0.92
###################################################################
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2

###################################################################
# mutacion
###################################################################
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < tasa_mutacion:
            cromosoma_mutado = cromosoma_mutado+str(int(not int(bit)))
        else:
            cromosoma_mutado = cromosoma_mutado+bit
    return cromosoma_mutado

###################################################################
# aplicacion de operadores geneticos
###################################################################
def algoritmo_genetico(tamaño_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamaño_poblacion, longitud_cromosoma)
    mejores_aptitudes = []
    
    for generacion in range(generaciones):
        print("Generación:", generacion + 1)

        # Calcular aptitud total para luego
        aptitud_total = 0
        for cromosoma in poblacion:
            aptitud_total = aptitud_total+aptitud(cromosoma)

        print("Aptitud total:", aptitud_total)

        # ..................................................................
        # seleccion
        # de progenitores con el metodo ruleta
        # se crea una lista vacia de progenitores primero
        progenitores = []
        for _ in range(tamaño_poblacion):
            progenitores.append(seleccion_ruleta(poblacion, aptitud_total))

        # ..................................................................
        # Cruce
        descendientes = []
        for i in range(0, tamaño_poblacion, 2):
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        # ..................................................................
        # mutacion
        descendientes_mutados = []
        for descendiente in descendientes:
            descendientes_mutados.append(mutacion(descendiente, tasa_mutacion))

        # Aqui se aplica elitismo
        # se reemplazar los peores cromosomas con los mejores progenitores
        poblacion.sort(key=aptitud)
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]
                
        # Guardar la mejor aptitud de la generación
        mejores_aptitudes.append(max(aptitud(individuo) for individuo in poblacion))

        # mostrar el mejor individuo de la generacion
        mejor_individuo = max(poblacion, key=aptitud)
        print("Mejor individuo:", int(mejor_individuo, 2) / 100.0, "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

    return max(poblacion, key=aptitud), mejores_aptitudes

###################################################################
# algoritmo genetico ejecucion principal
###################################################################
print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
mejor_solucion, mejores_aptitudes = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", int(mejor_solucion, 2) / 100.0, "Aptitud:", aptitud(mejor_solucion))

# Graficar g(c) en el intervalo [-1, 20]
c_vals = np.linspace(0, 10, 1000)
g_vals = [(2 * c) / (4 + 0.8 * c + c ** 2 + 0.2 * c ** 3) for c in c_vals]

plt.figure(figsize=(14, 6))

# Graficar la función g(c) y el máximo encontrado
plt.subplot(1, 2, 1)
plt.plot(c_vals, g_vals, label='g(c)')
plt.scatter([int(mejor_solucion, 2) / 100.0], [aptitud(mejor_solucion)], color='red', zorder=5, label='Máximo encontrado')
plt.title('Máximo de g(c) usando Algoritmo Genético')
plt.xlabel('c')
plt.ylabel('g(c)')
plt.legend()
plt.grid(True)

# Graficar la evolución de las mejores aptitudes
plt.subplot(1, 2, 2)
plt.plot(range(GENERACIONES), mejores_aptitudes, marker='o', color='b', label='Mejor Aptitud')
plt.title('Evolución de la Mejor Aptitud')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend()

plt.tight_layout()
plt.show()