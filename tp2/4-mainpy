import numpy as np
from pyswarm import pso

# Definición de la función objetivo
def objective_function(x):
    x1, x2 = x
    eq1 = 3 * x1 + 2 * x2 - 9
    eq2 = x1 - 5 * x2 - 4
    return eq1**2 + eq2**2

# Definir los límites de búsqueda para x1 y x2
lb = [-5, -5]  # Límite inferior para x1 y x2
ub = [5, 5]    # Límite superior para x1 y x2

# Parámetros del PSO
c1 = 2.0          # Coeficiente de aceleración personal
c2 = 2.0          # Coeficiente de aceleración global
w = 0.5           # Peso de inercia
max_iter = 50     # Número máximo de iteraciones
num_particles = 30 # Número de partículas

# Ejecución del PSO
xopt, fopt = pso(objective_function, lb, ub, swarmsize=num_particles, maxiter=max_iter, omega=w, phip=c1, phig=c2)

# Mostrar los resultados
print(f'Solución óptima encontrada: x1 = {xopt[0]:.4f}, x2 = {xopt[1]:.4f}')
print(f'Valor objetivo óptimo: {fopt:.4e}')


