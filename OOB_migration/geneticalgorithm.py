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
        max_generations: int = 10000,
        individual_mutation_rate: float = 0.45,
        cell_mutation_amount: float = 2,
        restart_after_n_generations: int = 20
        ):

        self.population_size = population_size
        self.selection_rate = selection_rate
        self.random_selection_rate = random_selection_rate
        self.max_generations = max_generations
        self.individual_mutation_rate = individual_mutation_rate
        self.cell_mutation_amount = cell_mutation_amount
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
        solution = np.empty(puzzle.shape)
        count = 0
        found_solution = False
        local_minima_loop_count = 0

        #initialize population
        fixed_indices = self.get_fixed_indices(puzzle)

        print(f"generating initial population...", end='\r')
        current_generation = self.create_first_generation(puzzle, self.population_size)

        print(f"intitialization took {time.time() - start_time} seconds") #remove

        print(f"generation shape: {current_generation.shape}")

        while count < self.max_generations and not found_solution:
            next_generation = np.empty(current_generation.shape)

            
            
            start_time= time.time() #remove

            #calculate fitness
            fitness = self.get_fitness(current_generation, fixed_indices)

            print(f"get_fitness took {time.time()-start_time} seconds") #remove

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
                print(f"encountered local minima at generation {count}, restarting...", end='\r')
                current_generation = self.create_first_generation(puzzle, self.population_size)
                local_minima_loop_count = 0
                continue  

            # add most fit to next generation according to selection rate
            next_generation[:new_selection_rate_amount] = current_generation[fitness_indices[:new_selection_rate_amount]]

            # add random individuals to next generation according to random selection rate
            random_indices = np.random.choice(self.population_size, new_random_selection_rate_amount)
            

            next_generation[new_selection_rate_amount:new_selection_rate_amount + new_random_selection_rate_amount] = current_generation[random_indices]

            start_time = time.time()

            # create children
            for i in range(children_amount):
                father, mother = current_generation[np.random.choice(self.population_size, 2, replace=False)]
                next_generation[new_selection_rate_amount + new_random_selection_rate_amount + i] = self.create_child(father, mother)
            print(f"creating children took {time.time()-start_time} seconds") #remove
            start_time = time.time() #remove

            # mutate next generation
            for i in range(self.population_size):
                next_generation[i] = self.mutate_individual(next_generation[i], fixed_indices)
            print(f"mutating next generation took {time.time()-start_time} seconds") #remove
            start_time = time.time() #remove
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
                                    children_rate_= 1 - self.selection_rate - self.random_selection_rate,
                                    individual_mutation_rate=self.individual_mutation_rate,
                                    cell_mutation_amount=self.cell_mutation_amount,
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
        population = np.empty((population_size, puzzle.shape[0], puzzle.shape[1]))
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
        fitness = np.empty(population.shape[0])
        for i, individual in enumerate(population):
            fitness[i] = self.generic_loss_function(individual, fixed_indices)
        return fitness
    
    # def get_fitness(self, population: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> NDArray[np.int_]:
    #     """
    #     Calculate fitness of a population. The fitness function measures how close
    #     each solution in the population is to being a valid Sudoku solution.
    #     Here, fitness is calculated based on the number of repeating numbers
    #     in each row, column, and block (3x3 subgrid), which must be minimized.
    #     """
    #     num_individuals = population.shape[0]
    #     fitness = np.zeros(num_individuals)
        
    #     # Iterate over each row, column, and 3x3 block to calculate fitness
    #     size = population.shape[1]  # Assuming a square grid (9x9 for standard Sudoku)
    #     block_size = int(np.sqrt(size))
        
    #     for i in range(size):
    #         # Rows and columns
    #         row_conflicts = size - np.unique(population[:, i, :], axis=1).shape[1]
    #         col_conflicts = size - np.unique(population[:, :, i], axis=1).shape[1]
    #         fitness += row_conflicts + col_conflicts

    #         # 3x3 Blocks
    #         block_row = (i // block_size) * block_size
    #         block_col = (i % block_size) * block_size
    #         block = population[:, block_row:block_row+block_size, block_col:block_col+block_size]
    #         block_conflicts = 3*block_size - np.unique(block.reshape(num_individuals, block_size*block_size), axis=1).shape[1]
    #         fitness += block_conflicts

    #     # Adjust fitness by penalizing wrong fixed values
    #     for idx in fixed_indices:
    #         correct_value = population[0, idx[0], idx[1]]  # Assuming the correct value is in the first individual
    #         mask = population[:, idx[0], idx[1]] != correct_value
    #         fitness[mask] += 10  # Heavy penalty for incorrect fixed values

        

        
    
    def mutate_individual_old(self, individual: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        """ Mutate an individual based on mutation rates and fixed indices. """
        if np.random.rand() < self.individual_mutation_rate:
            for i, j in np.ndindex(individual.shape):
                # Check if (i, j) is not in fixed indices
                if not np.any(np.all(fixed_indices == (i, j), axis=1)):
                    if np.random.rand() < self.cell_mutation_rate:
                        # Ensure mutation respects Sudoku constraints; this requires additional logic
                        individual[i, j] = np.random.randint(1, 10)  # Correct range for Sudoku values
        return individual
    
    def mutate_individual(self, individual, fixed_indices):
        if np.random.rand() < self.individual_mutation_rate:
            for cell_mutations in range(self.cell_mutation_amount):
                i, j = np.random.randint(0, 9, 2)
                if not np.any(np.all(fixed_indices == (i, j), axis=1)):
                    individual[i, j] = np.random.randint(1, 10)
        return individual

    
class DynamicGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self):
        
        super().__init__(
            population_size=20000,
            max_generations=1000,
            selection_rate=0.2,
            cell_mutation_amount=1,
            restart_after_n_generations=50
        )
    
    def __call__(self, puzzle: NDArray[np.int_]) -> NDArray[np.int_]:
        puzzle = np.array(puzzle)

        start_time = time.time()

        #initialize variables
        new_selection_rate_amount = int(self.selection_rate * self.population_size)
        new_random_selection_rate_amount = int(self.random_selection_rate * self.population_size)
        children_amount = self.population_size - new_selection_rate_amount - new_random_selection_rate_amount
        
        fitness_over_time = []
        solution = np.empty(puzzle.shape)
        count = 0
        found_solution = False
        local_minima_loop_count = 0

        #initialize population
        fixed_indices = self.get_fixed_indices(puzzle)
        # print(f"fixed indices: {fixed_indices}")

        print(f"generating initial population...", end='\r')
        current_generation = self.create_first_generation(puzzle, self.population_size)

        # print(f"intitialization took {time.time() - start_time} seconds") #remove

        

        while count < self.max_generations and not found_solution:
            next_generation = np.empty(current_generation.shape)

            # print(f"generation shape: {current_generation.shape}")
            
            # start_time= time.time() #remove 

            #calculate fitness
            fitness = self.get_fitness(current_generation, fixed_indices)

            # print(f"get_fitness took {time.time()-start_time} seconds") #remove

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
                print(f"\n encountered local minima at generation {count}, restarting...")
                current_generation = self.create_first_generation(puzzle, self.population_size)
                local_minima_loop_count = 0
                continue  

            


            self.individual_mutation_rate = fitness_over_time[-1] / max(fitness_over_time)
            if self.individual_mutation_rate < 0.25:
                self.individual_mutation_rate = 0.25

            self.children_rate = 1 - self.individual_mutation_rate
            children_amount = int(self.population_size * self.children_rate)

            self.random_selection_rate = 1 - self.selection_rate - self.children_rate
            #if random selection rate is less than 0, set it to 0
            if self.random_selection_rate < 0:
                self.random_selection_rate = 0
                self.children_rate = 1 - self.selection_rate
                children_amount = int(self.population_size * self.children_rate)

            new_random_selection_rate_amount = int(self.random_selection_rate * self.population_size)

            if new_selection_rate_amount + new_random_selection_rate_amount + children_amount != self.population_size:
                children_amount = self.population_size - new_selection_rate_amount - new_random_selection_rate_amount

            # print(f"Population coverage: {new_selection_rate_amount} elite + {new_random_selection_rate_amount} random + {children_amount} children = {new_selection_rate_amount + new_random_selection_rate_amount + children_amount}")

            # self.selection_rate = 1 - self.random_selection_rate - self.children_rate
            # new_selection_rate_amount = int(self.selection_rate * self.population_size)

            # add most fit to next generation according to selection rate
            next_generation[:new_selection_rate_amount] = current_generation[fitness_indices[:new_selection_rate_amount]]

            # add random individuals to next generation according to random selection rate
            random_indices = np.random.choice(self.population_size, new_random_selection_rate_amount, replace=False)

            next_generation[new_selection_rate_amount:new_selection_rate_amount + new_random_selection_rate_amount] = current_generation[random_indices]

            # start_time = time.time()
            

            # create children
            for i in range(children_amount):
                father, mother = current_generation[np.random.choice(self.population_size, 2, replace=False)]
                next_generation[new_selection_rate_amount + new_random_selection_rate_amount + i] = self.create_child(father, mother)
            # print(f"creating children took {time.time()-start_time} seconds") #remove
            # start_time = time.time() #remove

            # mutate next generation
            for i in range(self.population_size):
                next_generation[i] = self.mutate_individual(next_generation[i], fixed_indices)
            # print(f"mutating next generation took {time.time()-start_time} seconds") #remove
            # start_time = time.time() #remove
            current_generation = next_generation
            count += 1

            if count % 1 == 0:
                print(f"current generation: {count} \
                    \t current best fitness: {fitness_over_time[-1]} \
                    \t current median fitness: {fitness[fitness_indices[self.population_size//2]]}", end = '\r')


            
            #start live plot
            plt.ion()
            
            self.contious_loss_plot(
                                    fitness_over_time, 
                                    xlabel='generation', 
                                    ylabel='fitness', 
                                    population_size=self.population_size,
                                    selection_rate=self.selection_rate,
                                    random_selection_rate=self.random_selection_rate,
                                    children_rate_= self.children_rate,
                                    individual_mutation_rate=self.individual_mutation_rate,
                                    cell_mutation_amount=self.cell_mutation_amount,
                                    best_fitness=fitness_over_time[-1],
                                    elapsed_time=time.time() - start_time
                                    )
        if found_solution:
            print(f"Solution found at generation {count} after {time.time() - start_time} seconds, printing solution: \n")
            print(solution)
            self.plot_loss(fitness_over_time, xlabel='generation', ylabel='fitness', population_size=self.population_size, selection_rate=self.selection_rate, random_selection_rate=self.random_selection_rate, children_rate_= self.children_rate, individual_mutation_rate=self.individual_mutation_rate, cell_mutation_amount=self.cell_mutation_amount, best_fitness=fitness_over_time[-1], elapsed_time=time.time() - start_time)
            plt.savefig('solution.png')
            return solution

        else:
            print(f"Solution not found after {count} generations after {time.time() - start_time} seconds, printing best solution found: \n")
            print(fittest_individual)
            self.plot_loss(fitness_over_time, xlabel='generation', ylabel='fitness', population_size=self.population_size, selection_rate=self.selection_rate, random_selection_rate=self.random_selection_rate, children_rate_= self.children_rate, individual_mutation_rate=self.individual_mutation_rate, cell_mutation_amount=self.cell_mutation_amount, best_fitness=fitness_over_time[-1], elapsed_time=time.time() - start_time)
            plt.savefig('best_solution.png')
            return fittest_individual
    
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
        