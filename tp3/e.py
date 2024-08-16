import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
fabrication_capacity = 640
finishing_capacity = 960

# Función objetivo a maximizar
def f(x):
    return 375 * x[0] + 275 * x[1] + 475 * x[2] + 325 * x[3]

# Restricciones
def g1(x):
    return 2.5 * x[0] + 1.5 * x[1] + 2.75 * x[2] + 2 * x[3] <= fabrication_capacity 

def g2(x):
    return 3.5 * x[0] + 3 * x[1] + 3 * x[2] + 2 * x[3] <= finishing_capacity

def g3(x):
    return 2.5 * x[0] + 2 * x[1] + 2 * x[2] + 1 * x[3] <= finishing_capacity

# Parámetros del PSO
n_particles = 20  # número de partículas en el enjambre
n_dimensions = 4  # dimensiones del espacio de búsqueda (x1 y x2)
max_iterations = 50  # número máximo de iteraciones para la optimización
c1 = c2 = 1.4944  # coeficientes de aceleración
w = 0.6  # Factor de inercia

def pso_optimization_with_constraints(constraints):
    x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las partículas
    v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las partículas
    pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
    pbest_fit = -np.inf * np.ones(n_particles)  # vector para las mejores aptitudes personales (inicialmente -infinito)
    gbest = np.zeros(n_dimensions)  # mejor solución global
    gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

    # inicialización de partículas factibles
    for i in range(n_particles):
        while True:  # bucle para asegurar que la partícula sea factible
            x[i] = np.random.uniform(0, 10, n_dimensions)  # inicialización posición aleatoria en el rango [0, 10]
            if all(constraint(x[i]) for constraint in constraints):  # se comprueba si la posición cumple las restricciones
                break  # Salir del bucle si es factible
        v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
        pbest[i] = x[i].copy()  # se establece el mejor valor personal inicial como la posición actual
        fit = f(x[i])  # cálculo la aptitud de la posición inicial
        if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
            pbest_fit[i] = fit  # se actualiza el mejor valor personal
            
    # Optimización
    gbest_history = []
    for _ in range(max_iterations):
        for i in range(n_particles):
            fit = f(x[i])
            if fit > pbest_fit[i] and all(constraint(x[i]) for constraint in constraints):
                pbest_fit[i] = fit
                pbest[i] = x[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest = x[i].copy()

            # Actualización de la velocidad y posición de la partícula
            v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
            x[i] += v[i]

            # Asegurar que la nueva posición esté dentro de las restricciones y no negativa
            if not all(constraint(x[i]) for constraint in constraints) or np.any(x[i] < 0):
                x[i] = pbest[i].copy()
        
        gbest_history.append(gbest_fit)
    
    return gbest, gbest_fit, gbest_history

# Llamada a la función de optimización pasando las restricciones g1 y g2
constraints = [g1, g2]
gbest_original, gbest_fit_original, gbest_history_original = pso_optimization_with_constraints(constraints)

# Llamada a la función de optimización pasando las restricciones g1 y g3
constraints = [g1, g3]
gbest_new, gbest_fit_new, gbest_history_new = pso_optimization_with_constraints(constraints)

# Calcular la reducción en el valor óptimo
reduction = gbest_fit_original - gbest_fit_new
reduction_percentage = (reduction / gbest_fit_original) * 100

# Imprimir las mejores soluciones encontradas y sus valores óptimos
print(f"Mejor solución: {gbest_original}")
print(f"Valor óptimo: {gbest_fit_original}")
print(f"Mejor solución con nueva restricción: {gbest_new}")
print(f"Valor óptimo con nueva restricción: {gbest_fit_new}")
print(f"Reducción en el valor óptimo: {reduction}")
print(f"Porcentaje de reducción: {reduction_percentage:.2f}%")

# Gráfico de gbest en función de las iteraciones para ambas soluciones
plt.plot(gbest_history_original, label='Mejor solución Original')
plt.plot(gbest_history_new, label='Mejor solución Nueva')
plt.xlabel('Iteraciones')
plt.ylabel('gbest')
plt.title('Evolución de la mejor solución (gbest) a lo largo de las iteraciones')
plt.legend()
plt.show()