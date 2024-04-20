from stochasticmethods import StochasticMethods
from numpy.typing import NDArray
import numpy as np
import time
import matplotlib.pyplot as plt

class GeneticAlgorithm(StochasticMethods):
    def __init__(
        self,
        population_size: int = 20000,
        selection_rate: float = 0.2,
        random_selection_rate: float = 0.1,
        max_generations: int = 1000,
        individual_mutation_rate: float = 0.45,
        cell_mutation_rate: float = 0.005,
        restart_after_n_generations: int = 20
        ):

        self.population_size = population_size
        self.selection_rate = selection_rate
        self.random_selection_rate = random_selection_rate
        self.max_generations = max_generations
        self.individual_mutation_rate = individual_mutation_rate
        self.cell_mutation_rate = cell_mutation_rate
        self.restart_after_n_generations = restart_after_n_generations

        super().__init__()
    
    def __call__(self, puzzle: NDArray[np.int_]) -> NDArray[np.int_]:
        puzzle = np.array(puzzle)

        start_time = time.time()

        #start live plot
        plt.ion()

        #initialize variables
        new_selection_rate_amount = int(self.selection_rate * self.population_size)
        new_random_selection_rate_amount = int(self.random_selection_rate * self.population_size)
        children_amount = self.population_size - new_selection_rate_amount - new_random_selection_rate_amount
        
        fitness_over_time = []
        solution = np.empty(puzzle.shape, dtype=int)
        count = 0
        found_solution = False
        local_minima_loop_count = 0

        #initialize population
        fixed_indices = self.get_fixed_indices(puzzle)
        current_generation = self.create_first_generation(puzzle, self.population_size)

        while count < self.max_generations and not found_solution:
            next_generation = np.empty(current_generation.shape, dtype=int)

            #calculate fitness
            fitness = self.get_fitness(current_generation, fixed_indices)
            fitness_indices = np.argsort(fitness) #TODO: choose different sorting method
            fitness_over_time.append(fitness[fitness_indices[0]])

            if fitness_over_time[-1] == 0:
                found_solution = True
                solution = current_generation[fitness_indices[0]]
            
            if fitness_over_time[-1] == min(fitness_over_time):
                fittest_individual = current_generation[fitness_indices[0]]

            if count > 2 and fitness_over_time[-1] == fitness_over_time[-2]:
                local_minima_loop_count += 1
            else:
                local_minima_loop_count = 0
            
            if local_minima_loop_count > self.restart_after_n_generations:
                print(f"encountered local minima at generation {count}, restarting...")
                current_generation = self.create_first_generation(puzzle, self.population_size)
                local_minima_loop_count = 0
                continue  

            # add most fit to next generation according to selection rate
            next_generation[:new_selection_rate_amount] = current_generation[fitness_indices[:new_selection_rate_amount]]

            # add random individuals to next generation according to random selection rate
            random_indices = np.random.choice(self.population_size, new_random_selection_rate_amount)
            next_generation[new_selection_rate_amount:new_selection_rate_amount + new_random_selection_rate_amount] = current_generation[random_indices]

            # create children
            for i in range(children_amount):
                father, mother = current_generation[np.random.choice(self.population_size, 2, replace=False)]
                next_generation[new_selection_rate_amount + new_random_selection_rate_amount + i] = self.create_child(father, mother)

            # mutate next generation
            for i in range(self.population_size):
                next_generation[i] = self.mutate_individual(next_generation[i], fixed_indices)

            current_generation = next_generation
            count += 1

            print(f"current generation: {count} \
                \t current best fitness: {fitness_over_time[-1]} \
                \t current median fitness: {fitness[fitness_indices[self.population_size//2]]}", end = '\r')
            
            self.contious_loss_plot(
                                    fitness_over_time, 
                                    xlabel='generation', 
                                    ylabel='fitness', 
                                    population_size=self.population_size,
                                    selection_rate=self.selection_rate,
                                    random_selection_rate=self.random_selection_rate,
                                    individual_mutation_rate=self.individual_mutation_rate,
                                    cell_mutation_rate=self.cell_mutation_rate,
                                    best_fitness=fitness_over_time[-1],
                                    elapsed_time=time.time() - start_time
                                    )
        if found_solution:
            print(f"Solution found at generation {count}, printing solution:")
            print(solution)
            return solution

        else:
            print(f"Solution not found after {count} generations, printing best solution found:")
            print(fittest_individual)
            return fittest_individual

    def create_first_generation(self, puzzle: NDArray[np.int_], population_size: int) -> NDArray[np.int_]:
        """create a random population from the given puzzle"""
        population = np.empty((population_size, 9, 9), dtype=int)
        for i in range(population_size):
            population[i] = self.create_initial_solution(puzzle)
        return population
    
    def create_child(self, father: NDArray[np.int_], mother: NDArray[np.int_]) -> NDArray[np.int_]:
        """create a child from a father and a mother"""
        child = father.copy()
        for i, j in np.ndindex(child.shape):
            if np.random.rand() < 0.5:
                child[i, j] = mother[i, j]
        return child

    def get_fitness(self, population: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        """calculate fitness of a population"""
        fitness = np.empty(len(population), dtype=int)
        for i, individual in enumerate(population):
            fitness[i] = self.generic_loss_function(individual, fixed_indices)
        return fitness
    
    def mutate_individual(self, individual: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        """ Mutate an individual based on mutation rates and fixed indices. """
        if np.random.rand() < self.individual_mutation_rate:
            for i, j in np.ndindex(individual.shape):
                # Check if (i, j) is not in fixed indices
                if not np.any(np.all(fixed_indices == (i, j), axis=1)):
                    if np.random.rand() < self.cell_mutation_rate:
                        # Ensure mutation respects Sudoku constraints; this requires additional logic
                        individual[i, j] = np.random.randint(1, 10)  # Correct range for Sudoku values
        return individual
    
if __name__ == "__main__":
    # ga = GeneticAlgorithm()
    # puzzle = np.array([
    # [5, 3, 0, 0, 7, 0, 0, 0, 0],
    # [6, 0, 0, 1, 9, 5, 0, 0, 0],
    # [0, 9, 8, 0, 0, 0, 0, 6, 0],
    # [8, 0, 0, 0, 6, 0, 0, 0, 3],
    # [4, 0, 0, 8, 0, 3, 0, 0, 1],
    # [7, 0, 0, 0, 2, 0, 0, 0, 6],
    # [0, 6, 0, 0, 0, 0, 2, 8, 0],
    # [0, 0, 0, 4, 1, 9, 0, 0, 5],
    # [0, 0, 0, 0, 8, 0, 0, 7, 9]
    # ])
    # ga(puzzle)
    # plt.ioff()
    # plt.show()

    ga = GeneticAlgorithm()
    puzzle = np.array(
        [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9]
        ]
    )
    ga(puzzle)
        