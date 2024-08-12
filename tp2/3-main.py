import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo
def objective_function(x, y, a, b):
    return (x - a) ** 2 + (y + b) ** 2

# Parámetros del PSO
num_particles = 20
max_iterations = 10
c1 = 2
c2 = 2
w = 0.7

def pso(a, b, w):
    # Inicialización de las partículas
    x = np.random.uniform(-100, 100, num_particles)
    y = np.random.uniform(-100, 100, num_particles)
    v_x = np.zeros(num_particles)
    v_y = np.zeros(num_particles)
    personal_best_x = np.copy(x)
    personal_best_y = np.copy(y)
    personal_best_fitness = objective_function(x, y, a, b)
    global_best_index = np.argmin(personal_best_fitness)
    global_best_x = personal_best_x[global_best_index]
    global_best_y = personal_best_y[global_best_index]
    global_best_fitness = personal_best_fitness[global_best_index]

    gbest_history = [global_best_fitness]

    # Bucle de optimización
    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Actualización de la velocidad
            r1 = np.random.rand()
            r2 = np.random.rand()
            v_x[i] = (w * v_x[i] + c1 * r1 * (personal_best_x[i] - x[i]) + 
                      c2 * r2 * (global_best_x - x[i]))
            v_y[i] = (w * v_y[i] + c1 * r1 * (personal_best_y[i] - y[i]) + 
                      c2 * r2 * (global_best_y - y[i]))
            
            # Actualización de la posición
            x[i] += v_x[i]
            y[i] += v_y[i]
            
            # Restricción de las posiciones al intervalo [-100, 100]
            x[i] = np.clip(x[i], -100, 100)
            y[i] = np.clip(y[i], -100, 100)
            
            # Evaluación de la función objetivo
            fitness = objective_function(x[i], y[i], a, b)
            
            # Actualización del mejor personal
            if fitness < personal_best_fitness[i]:
                personal_best_x[i] = x[i]
                personal_best_y[i] = y[i]
                personal_best_fitness[i] = fitness
                
        # Actualización del mejor global
        if np.min(personal_best_fitness) < global_best_fitness:
            global_best_index = np.argmin(personal_best_fitness)
            global_best_x = personal_best_x[global_best_index]
            global_best_y = personal_best_y[global_best_index]
            global_best_fitness = personal_best_fitness[global_best_index]

        gbest_history.append(global_best_fitness)
    
    return global_best_x, global_best_y, global_best_fitness, gbest_history

# Entrada de valores para a y b
# Función para solicitar un valor dentro del rango
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

# Entrada de valores para a y b
a = solicitar_valor("Ingrese el valor de a (entre -50 y 50): ")
b = solicitar_valor("Ingrese el valor de b (entre -50 y 50): ")

# Ejecutar PSO con los parámetros dados
global_best_x, global_best_y, global_best_fitness, gbest_history = pso(a, b, w)

# Graficar la función objetivo y el punto mínimo encontrado
x_vals = np.linspace(-100, 100, 400)
y_vals = np.linspace(-100, 100, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = objective_function(X, Y, a, b)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Primer subplot: Función objetivo y el punto mínimo encontrado
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(global_best_x, global_best_y, color='red', label=f'Mínimo Encontrado: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f}), f(x, y) = {global_best_fitness:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Minimización de la Función Objetivo usando PSO')
ax1.legend()
fig.colorbar(contour, ax=ax1)
ax1.grid(True)

# Segundo subplot: Evolución de g_best en función de las iteraciones
ax2.plot(gbest_history, label=r'$g_{\text{best}}$')
ax2.set_xlabel('Iteración')
ax2.set_ylabel(r'$g_{\text{best}}$')
ax2.set_title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Mostrar la solución óptima encontrada
print(f'Solución óptima encontrada: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f})')
print(f'Valor objetivo óptimo: f(x, y) = {global_best_fitness:.2f}')

# Establecer el coeficiente de inercia w en 0 y ejecutar el algoritmo nuevamente
w = 0
global_best_x, global_best_y, global_best_fitness, gbest_history = pso(a, b, w)

# Graficar la función objetivo y el punto mínimo encontrado con w = 0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Primer subplot: Función objetivo y el punto mínimo encontrado con w = 0
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(global_best_x, global_best_y, color='red', label=f'Mínimo Encontrado con w=0: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f}), f(x, y) = {global_best_fitness:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Minimización de la Función Objetivo usando PSO con w=0')
ax1.legend()
fig.colorbar(contour, ax=ax1)
ax1.grid(True)

# Segundo subplot: Evolución de g_best en función de las iteraciones con w = 0
ax2.plot(gbest_history, label=r'$g_{\text{best}}$ con w=0')
ax2.set_xlabel('Iteración')
ax2.set_ylabel(r'$g_{\text{best}}$')
ax2.set_title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones con w=0')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Mostrar la solución óptima encontrada con w = 0
print(f'Solución óptima encontrada con w=0: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f})')
print(f'Valor objetivo óptimo con w=0: f(x, y) = {global_best_fitness:.2f}')

num_particles = 4

# Ejecutar PSO con los parámetros dados
global_best_x, global_best_y, global_best_fitness, gbest_history = pso(a, b, w)

# Mostrar la solución óptima encontrada
print(f'Solución óptima encontrada con 4 partículas: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f})')
print(f'Valor objetivo óptimo con 4 partículas: f(x, y) = {global_best_fitness:.2f}')

# Graficar la función objetivo y el punto mínimo encontrado
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Primer subplot: Función objetivo y el punto mínimo encontrado
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(global_best_x, global_best_y, color='red', label=f'Mínimo Encontrado: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f}), f(x, y) = {global_best_fitness:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Minimización de la Función Objetivo usando PSO con 4 partículas')
ax1.legend()
fig.colorbar(contour, ax=ax1)
ax1.grid(True)

# Segundo subplot: Evolución de g_best en función de las iteraciones
ax2.plot(gbest_history, label=r'$g_{\text{best}}$')
ax2.set_xlabel('Iteración')
ax2.set_ylabel(r'$g_{\text{best}}$')
ax2.set_title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones con 4 partículas')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

num_particles = 6

# Ejecutar PSO con los parámetros dados
global_best_x, global_best_y, global_best_fitness, gbest_history = pso(a, b, w)

# Mostrar la solución óptima encontrada
print(f'Solución óptima encontrada con 6 partículas: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f})')
print(f'Valor objetivo óptimo con 6 partículas: f(x, y) = {global_best_fitness:.2f}')

# Graficar la función objetivo y el punto mínimo encontrado
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Primer subplot: Función objetivo y el punto mínimo encontrado
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(global_best_x, global_best_y, color='red', label=f'Mínimo Encontrado: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f}), f(x, y) = {global_best_fitness:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Minimización de la Función Objetivo usando PSO con 6 partículas')
ax1.legend()
fig.colorbar(contour, ax=ax1)
ax1.grid(True)

# Segundo subplot: Evolución de g_best en función de las iteraciones
ax2.plot(gbest_history, label=r'$g_{\text{best}}$')
ax2.set_xlabel('Iteración')
ax2.set_ylabel(r'$g_{\text{best}}$')
ax2.set_title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones con 6 partículas')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

num_particles = 10

# Ejecutar PSO con los parámetros dados
global_best_x, global_best_y, global_best_fitness, gbest_history = pso(a, b, w)

# Mostrar la solución óptima encontrada
print(f'Solución óptima encontrada con 10 partículas: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f})')
print(f'Valor objetivo óptimo con 10 partículas: f(x, y) = {global_best_fitness:.2f}')

# Graficar la función objetivo y el punto mínimo encontrado
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Primer subplot: Función objetivo y el punto mínimo encontrado
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
ax1.scatter(global_best_x, global_best_y, color='red', label=f'Mínimo Encontrado: (x, y) = ({global_best_x:.2f}, {global_best_y:.2f}), f(x, y) = {global_best_fitness:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Minimización de la Función Objetivo usando PSO con 10 partículas')
ax1.legend()
fig.colorbar(contour, ax=ax1)
ax1.grid(True)

# Segundo subplot: Evolución de g_best en función de las iteraciones
ax2.plot(gbest_history, label=r'$g_{\text{best}}$')
ax2.set_xlabel('Iteración')
ax2.set_ylabel(r'$g_{\text{best}}$')
ax2.set_title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones con 10 partículas')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()