from pyswarm import pso
import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo
def objective_function(x):
    x1, x2 = x
    return (x1 - a) ** 2 + (x2 + b) ** 2

# Entrada de valores para a y b
def solicitar_valor(mensaje):
    while True:
        try:
            valor = float(input(mensaje))
            if -50 <= valor <= 50:
                return valor
            else:
                print("El valor debe estar entre -50 y 50. Intente de nuevo.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número.")

a = solicitar_valor("Ingrese el valor de a (entre -50 y 50): ")
b = solicitar_valor("Ingrese el valor de b (entre -50 y 50): ")

# Definición de los límites para las variables
lb = [-100, -100]  # límites inferiores para x1 y x2
ub = [100, 100]    # límites superiores para x1 y x2

# Parámetros del PSO
num_particles = 20
max_iterations = 10
c1 = 2
c2 = 2
w = 0.7

# Ejecutar PSO usando pyswarm
xopt, fopt = pso(objective_function, lb, ub, swarmsize=num_particles, maxiter=max_iterations, debug=False)

# Mostrar la solución óptima encontrada
print(f'Solución óptima encontrada: (x1, x2) = ({xopt[0]:.2f}, {xopt[1]:.2f})')
print(f'Valor objetivo óptimo: f(x1, x2) = {fopt:.2f}')

# Graficar la función objetivo y el punto mínimo encontrado
x_vals = np.linspace(-100, 100, 400)
y_vals = np.linspace(-100, 100, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = objective_function([X, Y])

fig, ax1 = plt.subplots(figsize=(12, 6))

# Función objetivo y el punto mínimo encontrado
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(xopt[0], xopt[1], color='red', label=f'Mínimo Encontrado: (x1, x2) = ({xopt[0]:.2f}, {xopt[1]:.2f}), f(x1, x2) = {fopt:.2f}')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('Minimización de la Función Objetivo usando PSO')
ax1.legend()
fig.colorbar(contour, ax=ax1)
ax1.grid(True)

plt.tight_layout()
plt.show()