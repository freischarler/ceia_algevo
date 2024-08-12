import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo
def objective_function(x):
    return np.sin(x) + np.sin(x ** 2)

# Parámetros del PSO
max_iterations = 30
c1 = 1.49
c2 = 1.49
w = 0.5

def pso(num_particles):
    # Inicialización de las partículas
    x = np.random.uniform(0, 10, num_particles)
    v = np.zeros(num_particles)
    personal_best_x = np.copy(x)
    personal_best_fitness = objective_function(x)
    global_best_x = x[np.argmax(personal_best_fitness)]
    global_best_fitness = np.max(personal_best_fitness)

    gbest_history = [global_best_fitness]

    # Bucle de optimización
    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Actualización de la velocidad
            r1 = np.random.rand()
            r2 = np.random.rand()
            v[i] = (w * v[i] + c1 * r1 * (personal_best_x[i] - x[i]) + 
                    c2 * r2 * (global_best_x - x[i]))
            
            # Actualización de la posición
            x[i] += v[i]
            
            # Restricción de las posiciones al intervalo [0, 10]
            x[i] = np.clip(x[i], 0, 10)
            
            # Evaluación de la función objetivo
            fitness = objective_function(x[i])
            
            # Actualización del mejor personal
            if fitness > personal_best_fitness[i]:
                personal_best_x[i] = x[i]
                personal_best_fitness[i] = fitness
                
        # Actualización del mejor global
        if np.max(personal_best_fitness) > global_best_fitness:
            global_best_x = personal_best_x[np.argmax(personal_best_fitness)]
            global_best_fitness = np.max(personal_best_fitness)

        gbest_history.append(global_best_fitness)
    
    return global_best_x, global_best_fitness, gbest_history

# Función para graficar los resultados
def plot_results(num_particles, global_best_x, global_best_fitness, gbest_history):
    # Graficar la función objetivo
    x_vals = np.linspace(0, 10, 400)
    y_vals = objective_function(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=r'Función Objetivo: $f(x) = \sin(x) + \sin(x^2)$')
    plt.scatter(global_best_x, global_best_fitness, color='black', label=f'Máximo Encontrado: x = {global_best_x:.2f}, f(x) = {global_best_fitness:.2f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Optimización de la Función Objetivo usando PSO con {num_particles} Partículas')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar gbest en función de las iteraciones
    plt.figure(figsize=(10, 6))
    plt.plot(gbest_history, label=f'{num_particles} Partículas')
    plt.xlabel('Iteración')
    plt.ylabel(r'$g_{\text{best}}$')
    plt.title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejecutar PSO con diferentes números de partículas y mostrar resultados
particle_counts = [2, 4, 6, 10]

for num_particles in particle_counts:
    global_best_x, global_best_fitness, gbest_history = pso(num_particles)
    print(f'Número de partículas: {num_particles}')
    print(f'Solución óptima encontrada: x = {global_best_x:.2f}')
    print(f'Valor objetivo óptimo: f(x) = {global_best_fitness:.2f}')
    plot_results(num_particles, global_best_x, global_best_fitness, gbest_history)
