###################################################################
# Algoritmo Gneñetico que encuentra el maximo de la funcion x^2
# Seleccion por ruleta
# Pc = 0.92
# Pm = 0.1
###################################################################
import random
import matplotlib.pyplot as plt
import numpy as np


# Parámetros
TAMANIO_POBLACION = 4
LONGITUD_CROMOSOMA_X = 15
LONGITUD_CROMOSOMA_Y = 15
TASA_MUTACION = 0.07
TASA_CRUCE = 0.85
GENERACIONES = 10

# Rango de x y y
RANGO_X = (-10, 10)
RANGO_Y = (0, 20)

###################################################################
# Aptitud (c(x,y) = 7.7 + 0.15x + 0.22y − 0.05x**2 − 0.016y**2 − 0.007*x*y)
###################################################################
def aptitud(cromosoma):
    x_bin = cromosoma[:LONGITUD_CROMOSOMA_X]
    y_bin = cromosoma[LONGITUD_CROMOSOMA_X:]
    
    x = int(x_bin, 2) * (RANGO_X[1] - RANGO_X[0]) / (2**LONGITUD_CROMOSOMA_X - 1) + RANGO_X[0]
    y = int(y_bin, 2) * (RANGO_Y[1] - RANGO_Y[0]) / (2**LONGITUD_CROMOSOMA_Y - 1) + RANGO_Y[0]
    
    return 7.7 + 0.15 * x + 0.22 * y - 0.05 * x**2 - 0.016 * y**2 - 0.007 * x * y

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
        print("Mejor individuo:", int(mejor_individuo, 2) / 1000.0, "Aptitud:", aptitud(mejor_individuo))
        print("_________________________________________________________________________________")

    return max(poblacion, key=aptitud), mejores_aptitudes

###################################################################
# algoritmo genetico ejecucion principal
###################################################################
print("_________________________________________________________________________________")
print("_________________________________________________________________________________")
print()
mejor_solucion, mejores_aptitudes = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA_X+LONGITUD_CROMOSOMA_Y, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
print("Mejor solución:", int(mejor_solucion, 2) / 1000.0, "Aptitud:", aptitud(mejor_solucion))

x_bin = mejor_solucion[:LONGITUD_CROMOSOMA_X]
y_bin = mejor_solucion[LONGITUD_CROMOSOMA_X:]

mejor_x = int(x_bin, 2) * (RANGO_X[1] - RANGO_X[0]) / (2**LONGITUD_CROMOSOMA_X - 1) + RANGO_X[0]
mejor_y = int(y_bin, 2) * (RANGO_Y[1] - RANGO_Y[0]) / (2**LONGITUD_CROMOSOMA_Y - 1) + RANGO_Y[0]
mejor_aptitud = aptitud(mejor_solucion)

# Graficar f(x, y) en una rejilla
x_vals = np.linspace(RANGO_X[0], RANGO_X[1], 400)
y_vals = np.linspace(RANGO_Y[0], RANGO_Y[1], 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 7.7 + 0.15 * X + 0.22 * Y - 0.05 * X**2 - 0.016 * Y**2 - 0.007 * X * Y

plt.figure(figsize=(12, 6))

# Graficar la función f(x, y)
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z, 50, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.scatter([mejor_x], [mejor_y], color='red', zorder=5, label='Máximo encontrado')
plt.title('Máximo de f(x, y) usando Algoritmo Genético')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Graficar la evolución de las mejores aptitudes
plt.subplot(1, 2, 2)
plt.plot(range(GENERACIONES), mejores_aptitudes, marker='o', color='b', label='Mejor Aptitud')
plt.title('Evolución de la Mejor Aptitud')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.legend()

plt.tight_layout()
plt.show()