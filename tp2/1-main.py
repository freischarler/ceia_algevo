import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo
def objective_function(x):
    return (2 * np.sin(x)) - (x**2) / 2

# Parámetros del PSO
max_iterations = 20
c1 = 2
c2 = 2
w = 0.7

def pso(num_particles):
    # Inicialización de las partículas
    x = np.random.uniform(0, 4, num_particles)
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
            
            # Restricción de las posiciones al intervalo [0, 4]
            x[i] = np.clip(x[i], 0, 4)
            
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

# Ejecutar PSO con 2 partículas y graficar la función objetivo
global_best_x, global_best_fitness, _ = pso(2)

# Números de partículas a probar
num_particles_list = [4, 10, 100, 200, 400]

# Ejecutar PSO para diferentes números de partículas y almacenar las historias de gbest
histories = {}
for num_particles in num_particles_list:
    _, _, gbest_history = pso(num_particles)
    histories[num_particles] = gbest_history

# Crear los subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Primer subplot: Función objetivo y el punto máximo encontrado
x_vals = np.linspace(0, 4, 400)
y_vals = objective_function(x_vals)

ax1.plot(x_vals, y_vals, label=r'Función Objetivo: $f(x) = 2\sin(x) - \frac{x^2}{2}$')
ax1.scatter(global_best_x, global_best_fitness, color='green', label=f'Máximo Encontrado: x = {global_best_x:.2f}, f(x) = {global_best_fitness:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Optimización de la Función Objetivo usando PSO')
ax1.legend()
ax1.grid(True)

# Segundo subplot: Evolución de g_best en función de las iteraciones
for num_particles, history in histories.items():
    ax2.plot(history, label=f'{num_particles} Partículas')

ax2.set_xlabel('Iteración')
ax2.set_ylabel(r'$g_{\text{best}}$')
ax2.set_title(r'Evolución de $g_{\text{best}}$ en función de las iteraciones para diferentes números de partículas')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
