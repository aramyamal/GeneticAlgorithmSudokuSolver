from stochasticmethods import StochasticMethods
from numpy.typing import NDArray
import numpy as np
import time
import matplotlib.pyplot as plt

#TODO: add type hints, docstrings, and comments, and remove print statements, make the new functions into methods of the class

class GeneticAlgorithm(StochasticMethods):
    def __init__(
        self,
        population_size: int = 10000, #20000
        selection_rate: float = 0.2, #0.2
        random_selection_rate: float = 0.0, #0.1
        max_generations: int = 40000, #400
        individual_mutation_rate: float = 0.25, #0.45
        cell_mutation_amount: float = 1, #1
        restart_after_n_generations: int = 40 #40
        ):

        self.population_size = population_size
        self.selection_rate = selection_rate
        self.random_selection_rate = random_selection_rate
        self.max_generations = max_generations
        self.individual_mutation_rate = individual_mutation_rate
        self.cell_mutation_amount = cell_mutation_amount
        self.restart_after_n_generations = restart_after_n_generations

        super().__init__()
    
    def __call__(self, puzzle: NDArray[np.int_], show_live_plot: bool = False) -> NDArray[np.int_]:
        puzzle = np.array(puzzle, dtype=np.int8)

        start_time = time.time()

        # start live plot
        if show_live_plot:  
            plt.ion()  
            plt.figure(figsize=(10, 5)) 

        #initialize variables
        new_selection_rate_amount = int(self.selection_rate * self.population_size)
        new_random_selection_rate_amount = int(self.random_selection_rate * self.population_size)
        children_amount = self.population_size - new_selection_rate_amount - new_random_selection_rate_amount
        
        fitness_over_time = []
        solution = np.empty(puzzle.shape, dtype=np.int8)
        count = 0
        found_solution = False
        local_minima_loop_count = 0

        #initialize population
        fixed_indices = self.get_fixed_indices(puzzle)

        print(f"generating initial population...")
        current_generation = self.create_first_generation(puzzle, self.population_size)

        print(f"generation shape: {current_generation.shape}")

        while count < self.max_generations and not found_solution:
            next_generation = np.empty(current_generation.shape, dtype=np.int8)

            #calculate fitness
            fitness = self.get_fitness(current_generation, fixed_indices)

            fitness_indices = np.argsort(fitness)

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

            # add children to next generation
            children = self.create_children(current_generation, children_amount)
            next_generation[new_selection_rate_amount + new_random_selection_rate_amount:] = children

            # mutate next generation
            for i in range(self.population_size):
                next_generation[i] = self.mutate_individual(next_generation[i], fixed_indices)

            current_generation = next_generation
            count += 1

            if count % 5 == 0:
                print(f"current generation: {count} \
                    \t current best fitness: {fitness_over_time[-1]} \
                    \t current median fitness: {fitness[fitness_indices[self.population_size//2]]}")
                
                if show_live_plot:
                    self.contious_loss_plot(
                                            fitness_over_time, 
                                            xlabel='generation', 
                                            ylabel='fitness', 
                                            population_size=self.population_size,
                                            selection_rate=self.selection_rate,
                                            random_selection_rate=self.random_selection_rate,
                                            children_rate_= 1 - self.selection_rate - self.random_selection_rate,
                                            individual_mutation_rate=self.individual_mutation_rate,
                                            cell_mutation_amount=self.cell_mutation_amount,
                                            best_fitness=fitness_over_time[-1],
                                            elapsed_time=time.time() - start_time
                                            )
            
        if found_solution:
            print(f"\n Solution found at generation {count} and time {time.time() - start_time}, printing solution:")
            print(solution)
            plt.ioff()
            self.plot_loss(fitness_over_time, xlabel='generation', ylabel='fitness', population_size=self.population_size, selection_rate=self.selection_rate, random_selection_rate=self.random_selection_rate, children_rate_= 1 - self.selection_rate - self.random_selection_rate, individual_mutation_rate=self.individual_mutation_rate, cell_mutation_amount=self.cell_mutation_amount, best_fitness=fitness_over_time[-1], elapsed_time=time.time() - start_time)
            return solution

        else:
            print(f"\n Solution not found after {count} generations and time {time.time() - start_time}, printing best solution found:")
            print(fittest_individual)
            plt.ioff()
            self.plot_loss(fitness_over_time, xlabel='generation', ylabel='fitness', population_size=self.population_size, selection_rate=self.selection_rate, random_selection_rate=self.random_selection_rate, children_rate_= 1 - self.selection_rate - self.random_selection_rate, individual_mutation_rate=self.individual_mutation_rate, cell_mutation_amount=self.cell_mutation_amount, best_fitness=fitness_over_time[-1], elapsed_time=time.time() - start_time)
            return fittest_individual

    def create_first_generation(self, puzzle: NDArray[np.int_], population_size: int) -> NDArray[np.int_]:
        """create a random population from the given puzzle"""
        population = np.empty((population_size, puzzle.shape[0], puzzle.shape[1]), dtype=np.int8)
        for i in range(population_size):
            population[i] = self.create_initial_solution(puzzle)
        return population
    
    def create_child(self, father: NDArray[np.int_], mother: NDArray[np.int_]) -> NDArray[np.int_]:
        """create a child from a father and a mother"""
        child = father.copy(dtype=np.int8)
        for i, j in np.ndindex(child.shape):
            if np.random.rand() < 0.5:
                child[i, j] = mother[i, j]
        return child

    def create_children(self, current_generation, children_amount):
        """
        Create children from pairs of parents.
        children_amount: The number of children to produce.
        """
        # Generate indices for parents
        parent_indices = np.random.choice(self.population_size, size=(2, children_amount), replace=True)

        # Select parents based on these indices
        fathers = current_generation[parent_indices[0, :]]
        mothers = current_generation[parent_indices[1, :]]

        # Create a mask for crossover
        crossover_mask = np.random.rand(children_amount, *fathers.shape[1:]) < 0.5
        
        # Create children using where operation
        children = np.where(crossover_mask, fathers, mothers).astype(np.int8)
        return children
    
    def get_fitness(self, population: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Calculate fitness for a Sudoku population. A fitness of 0 means the solution is correct.
        Higher values indicate more violations of Sudoku rules.
        """
        num_individuals = population.shape[0]
        fitness = np.zeros(num_individuals)
        size = population.shape[-1]  # Standard Sudoku size
        block_size = int(np.sqrt(size))  # Block size in standard Sudoku

        # Check for number conflicts in rows and columns
        for axis in [1, 2]:  # 1 for rows, 2 for columns
            data = np.swapaxes(population, 1, axis) if axis == 2 else population
            for i in range(size):
                slice_ = data[:, i, :]
                conflicts = np.sum(slice_[:, :, None] == slice_[:, None, :], axis=(1, 2)) - size
                fitness += conflicts

        # Check for number conflicts in blocks
        for block_row in range(0, size, block_size):
            for block_col in range(0, size, block_size):
                block = population[:, block_row:block_row+block_size, block_col:block_col+block_size].reshape(num_individuals, -1)
                block_conflicts = np.sum(block[:, :, None] == block[:, None, :], axis=(1, 2)) - block_size*block_size
                fitness += block_conflicts

        # Penalize incorrect fixed values heavily
        if fixed_indices.size > 0:
            correct_values = population[0, fixed_indices[:, 0], fixed_indices[:, 1]]
            penalties = 10 * np.sum(population[:, fixed_indices[:, 0], fixed_indices[:, 1]] != correct_values, axis=1)
            fitness += penalties

        return fitness
    
    def mutate_individual(self, individual, fixed_indices):
        if np.random.rand() < self.individual_mutation_rate:
            for cell_mutations in range(self.cell_mutation_amount):
                i, j = np.random.randint(0, 9, 2)
                if not np.any(np.all(fixed_indices == (i, j), axis=1)):
                    individual[i, j] = np.random.randint(1, 10)
        return individual
    
if __name__ == "__main__":
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

    hard = np.array(
    [
            [0, 0, 6, 1, 0, 0, 0, 0, 8], 
            [0, 8, 0, 0, 9, 0, 0, 3, 0], 
            [2, 0, 0, 0, 0, 5, 4, 0, 0], 
            [4, 0, 0, 0, 0, 1, 8, 0, 0], 
            [0, 3, 0, 0, 7, 0, 0, 4, 0], 
            [0, 0, 7, 9, 0, 0, 0, 0, 3], 
            [0, 0, 8, 4, 0, 0, 0, 0, 6], 
            [0, 2, 0, 0, 5, 0, 0, 8, 0], 
            [1, 0, 0, 0, 0, 2, 5, 0, 0]
        ])

    # solution_puzzle = np.array([
    #             [5,3,4,6,7,8,9,1,2],
    #             [6,7,2,1,9,5,3,4,8],
    #             [1,9,8,3,4,2,5,6,7],
    #             [8,5,9,7,6,1,4,2,3],
    #             [4,2,6,8,5,3,7,9,1],
    #             [7,1,3,9,2,4,8,5,6],
    #             [9,6,1,5,3,7,2,8,4],
    #             [2,8,7,4,1,9,6,3,5],
    #             [3,4,5,2,8,6,1,7,9]
    #             ])
    ga(puzzle, show_live_plot=True)