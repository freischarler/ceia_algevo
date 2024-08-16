# .........................................................................
# Algoritmo MOPSO (Optimización por Enjambre de Partículas Multi-Objetivo)
# .........................................................................

import numpy as np
import random
import matplotlib.pyplot as plt


# funciones objetivo
def f1(x):
    return x ** 2  # Función objetivo 1: f1(x) = x^2


def f2(x):
    return (x - 2) ** 2  # Función objetivo 2: f2(x) = (x-2)^2


# Evaluación de la función objetivo
def evaluar(x):
    return [f1(x), f2(x)]  # Evalúa ambas funciones objetivo y devuelve los resultados


# Dominancia: True si y1 domina a y2
def domina(y1, y2):
    es_menor_o_igual = True  # Flag para verificar si todos los valores en y1 son <= en y2
    es_estrictamente_menor = False  # Flag para verificar si al menos un valor en y1 es < en y2

    for solucion_1, solucion_2 in zip(y1, y2):
        if solucion_1 > solucion_2:
            es_menor_o_igual = False  # Si algún valor en y1 es mayor que en y2, y1 no domina
        if solucion_1 < solucion_2:
            es_estrictamente_menor = True  # Si algún valor en y1 es menor que en y2, y1 domina estrictamente

    return es_menor_o_igual and es_estrictamente_menor  # y1 domina a y2 si todos los valores en y1 son <= en y2 y al menos uno es <


# Inicialización del repositorio
def actualizar_repositorio(repositorio, particula):
    no_dominadas = []  # Lista para almacenar soluciones no dominadas
    for sol in repositorio:
        if domina(particula, sol):
            continue  # Si la partícula domina una solución en el repositorio, no se necesita agregar
        elif domina(sol, particula):
            return repositorio  # Si la solución en el repositorio domina la partícula, no se actualiza el repositorio
        else:
            no_dominadas.append(sol)  # Si ninguna de las dos domina a la otra, se agrega a la lista de no dominadas
    no_dominadas.append(particula)  # Agrega la partícula al repositorio si no es dominada por ninguna solución
    return no_dominadas  # Retorna el repositorio actualizado


# Parámetros del algoritmo
num_particulas = 30  # Número de partículas en el algoritmo
num_iteraciones = 100  # Número de iteraciones del algoritmo
w = 0.5  # Factor de inercia, controla la influencia de la velocidad previa
c1 = 1.5  # Coeficiente de aceleración personal
c2 = 1.5  # Coeficiente de aceleración global

# Inicialización de partículas
particulas = np.random.uniform(low=-2, high=4, size=(num_particulas, 1))  # Inicializa posiciones de partículas aleatoriamente en el intervalo [-2, 4]
velocidades = np.random.uniform(low=-1, high=1,
                                size=(num_particulas, 1))  # Inicializa velocidades de partículas aleatoriamente
pbest = particulas.copy()  # Inicializa la mejor solución personal (pbest) igual a las posiciones iniciales

# Inicialización de valor_pbest usando un bucle for
valor_pbest = []  # Lista para almacenar los valores de fitness de las mejores soluciones personales
for x in pbest:
    valor_pbest.append(evaluar(x))  # Evalúa cada posición inicial y agrega el resultado a la lista
valor_pbest = np.array(valor_pbest)  # Convierte la lista en un array de numpy

# Inicialización del repositorio
repositorio = []  # Lista para almacenar el frente de Pareto aproximado

# Algoritmo MOPSO (Optimización por Enjambre de Partículas Multi-Objetivo)
for _ in range(num_iteraciones):
    # Actualizar repositorio
    print("Iteracion: ", _)
    for i in range(num_particulas):
        repositorio = actualizar_repositorio(repositorio, valor_pbest[
            i])  # Actualiza el repositorio con las mejores soluciones personales

    # Seleccionar mejor global (gbest) aleatoriamente del repositorio
    gbest = random.choice(repositorio)  # Selecciona aleatoriamente una solución del repositorio como la mejor global

    for i in range(num_particulas):
        # Actualización de la velocidad y posición
        r1 = random.random()  # Número aleatorio para la componente cognitiva
        r2 = random.random()  # Número aleatorio para la componente social
        velocidades[i] = w * velocidades[i] + c1 * r1 * (pbest[i] - particulas[i]) + c2 * r2 * (
                    gbest[0] - particulas[i])  # Actualiza la velocidad
        particulas[i] = particulas[i] + velocidades[i]  # Actualiza la posición

        # Restringir partículas al dominio permitido
        particulas[i] = np.clip(particulas[i], -2, 4)  # Asegura que las posiciones se mantengan en el intervalo [-2, 4]

        # Evaluar y actualizar pbest
        valor_actual = evaluar(particulas[i])  # Evalúa la posición actual de la partícula
        if domina(valor_actual, valor_pbest[i]):
            pbest[i] = particulas[i]  # Actualiza la mejor solución personal si la nueva es mejor
            valor_pbest[i] = valor_actual  # Actualiza el valor de fitness de la mejor solución personal

# Mostrar frente de Pareto aproximado
frente_pareto = np.array([evaluar(x[0]) for x in
                          repositorio])  # Evalúa todas las soluciones en el repositorio para obtener el frente de Pareto
plt.scatter(frente_pareto[:, 0], frente_pareto[:, 1], color='red')  # Grafica el frente de Pareto en rojo
plt.title('Frente de Pareto Aproximado')  # Título del gráfico
plt.xlabel('$f_1(x)$')  # Etiqueta del eje x
plt.ylabel('$f_2(x)$')  # Etiqueta del eje y
plt.grid(True)  # Muestra la cuadrícula en el gráfico
plt.show()  # Muestra el gráfico
