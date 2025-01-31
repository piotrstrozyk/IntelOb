import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

# Parametry dla Morse'a
rho_values = [3, 6, 10, 14]

# Funkcja Morse'a
def morse_potential(r, rho):
    return np.exp(rho * (1 - r)) * (np.exp(rho * (1 - r)) - 2)

# Obliczanie całkowitej energii klastra
def total_morse_energy(positions, rho):
    distances = pdist(positions)
    energy = np.sum(morse_potential(distances, rho))
    return energy

# Generowanie losowych początkowych pozycji
def generate_initial_positions(N):
    return np.random.rand(N, 3) * 2  # W przedziale [0,2]

# Algorytm Genetyczny
def genetic_algorithm(N, rho, pop_size=100, generations=500, mutation_rate=0.01):
    # Inicjalizacja populacji
    population = [generate_initial_positions(N) for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitness = np.array([total_morse_energy(individual, rho) for individual in population])
        best_index = np.argmin(fitness)
        
        if fitness[best_index] < best_fitness:
            best_fitness = fitness[best_index]
            best_solution = population[best_index]

        # Selekcja (turniej)
        selected = [population[np.random.choice(np.arange(pop_size))] for _ in range(pop_size)]
        
        # Krzyżowanie
        children = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            crossover_point = np.random.randint(1, N*3-1)
            child1 = np.concatenate([parent1.flatten()[:crossover_point], parent2.flatten()[crossover_point:]]).reshape(N, 3)
            child2 = np.concatenate([parent2.flatten()[:crossover_point], parent1.flatten()[crossover_point:]]).reshape(N, 3)
            children.append(child1)
            children.append(child2)
        
        # Mutacja
        for child in children:
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(0, N*3)
                child.flatten()[idx] = np.random.rand() * 2
        
        # Aktualizacja populacji
        population = children

    return best_solution, best_fitness

# Algorytm PSO
def pso(N, rho, swarm_size=100, iterations=500, w=0.5, c1=2, c2=2):
    positions = [generate_initial_positions(N) for _ in range(swarm_size)]
    velocities = [np.random.rand(N, 3) * 0.1 for _ in range(swarm_size)]
    p_best_positions = positions[:]
    p_best_scores = [total_morse_energy(pos, rho) for pos in positions]
    g_best_position = p_best_positions[np.argmin(p_best_scores)]
    g_best_score = min(p_best_scores)

    for iteration in range(iterations):
        for i in range(swarm_size):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best_positions[i] - positions[i]) + c2 * r2 * (g_best_position - positions[i])
            positions[i] += velocities[i]

            fitness = total_morse_energy(positions[i], rho)
            if fitness < p_best_scores[i]:
                p_best_scores[i] = fitness
                p_best_positions[i] = positions[i]
                if fitness < g_best_score:
                    g_best_score = fitness
                    g_best_position = positions[i]

    return g_best_position, g_best_score

# Eksperymenty dla różnych wartości N i rho
N_values = [5, 6, 7, 8]
results_ga = {}
results_pso = {}

for rho in rho_values:
    results_ga[rho] = []
    results_pso[rho] = []
    for N in N_values:
        best_solution_ga, best_fitness_ga = genetic_algorithm(N, rho)
        best_solution_pso, best_fitness_pso = pso(N, rho)
        results_ga[rho].append((N, best_fitness_ga))
        results_pso[rho].append((N, best_fitness_pso))

# Wypisanie wyników
print("Wyniki Algorytmu Genetycznego:")
for rho, results in results_ga.items():
    print(f"rho = {rho}:")
    for N, fitness in results:
        print(f"  N = {N}, VM = {fitness}")

print("\nWyniki Algorytmu PSO:")
for rho, results in results_pso.items():
    print(f"rho = {rho}:")
    for N, fitness in results:
        print(f"  N = {N}, VM = {fitness}")

# Wizualizacja najlepszego klastra
def visualize_cluster(positions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=50, c='b', marker='o')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Przykładowa wizualizacja najlepszego klastra dla N=8, rho=6
best_cluster_ga, _ = genetic_algorithm(8, 6)
best_cluster_pso, _ = pso(8, 6)

visualize_cluster(best_cluster_ga, "Best Cluster (GA, N=8, rho=6)")
visualize_cluster(best_cluster_pso, "Best Cluster (PSO, N=8, rho=6)")

def export_to_pdb(positions, filename):
    with open(filename, 'w') as f:
        for i, pos in enumerate(positions):
            f.write(f"ATOM  {i+1:5d}  CA  ALA A   1    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C  \n")
        f.write("END\n")

# Użycie funkcji:
positions = generate_initial_positions(10)  # Generuje losowe pozycje dla 10 cząstek
export_to_pdb(positions, "cluster.pdb")  # Eksportuje pozycje do pliku PDB

import os

# Użycie funkcji:
positions = generate_initial_positions(10)  # Generuje losowe pozycje dla 10 cząstek
pdb_filename = "cluster.pdb"
export_to_pdb(positions, pdb_filename)  # Eksportuje pozycje do pliku PDB

# Otwarcie pliku PDB w PyMOL
os.system(f"pymol {pdb_filename}")