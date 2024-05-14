from abc import ABC, abstractmethod
import numpy as np
from time import time

class SudokuAlgorithm(ABC):
    """Abstract base class for Sudoku solving algorithms"""

    @abstractmethod
    def __call__(self, sudoku: np.ndarray) -> np.ndarray:
        pass

class SudokuSolver:
    """Sudoku solver class that uses a SudokuAlgorithm to solve the puzzle"""

    def __init__(self, algorithm: SudokuAlgorithm):
        self.algorithm = algorithm
    
    def solve(self, sudoku: np.ndarray) -> np.ndarray:
        solution = self.algorithm(sudoku)
        self.print_puzzle(solution)
        return solution
    
    @staticmethod
    def print_puzzle(puzzle: np.ndarray) -> None:
        """Print the sudoku puzzle"""
        print("\n", end="")
        for i in range(9):
            if i % 3 == 0 and i != 0:
                # Print a horizontal separator line
                print("------+-------+------")
            
            # Print each row with vertical separators
            row_format = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_format += "| "
                row_format += f"{puzzle[i, j]} "
            
            # Print the formatted row with row index
            print(f"{row_format.strip()}")
    

class StochasticOperations:

    @staticmethod
    def get_fixed_indices(puzzle: np.ndarray) -> np.ndarray:
        """Return the indices of fixed values in the puzzle"""
        return np.argwhere(puzzle != 0)

    @staticmethod
    def get_fitness(population: np.ndarray, fixed_indices: np.ndarray) -> np.ndarray:
        """
        Calculate fitness for a Sudoku population. A fitness of 0 means the solution is correct.
        Higher values indicate more violations of Sudoku rules.

        Parameters
        ----------
        population : np.ndarray
            Population of Sudoku solutions to evaluate with shape (num_individuals, 9, 9)
        fixed_indices : np.ndarray
            Indices of fixed values in the Sudoku puzzle with shape (num_fixed_values, 2)
        """
        num_individuals = population.shape[0]
        fitness = np.zeros(num_individuals)
        size = population.shape[-1] # Standard Sudoku size 9
        block_size = int(np.sqrt(size)) # Block size in standard Sudoku 3

        # Check for number conflicts in rows and columns
        for axis in [1, 2]: # 1 for rows, 2 for columns
            data = np.swapaxes(population, 1, axis) if axis == 2 else population
            for i in range(size):
                slice_ = data[:, i, :]
                conflicts = np.sum(slice_[:, :, None] == slice_[:, None, :], axis=(1, 2)) - size
                fitness += conflicts
        
        # Check for number conflicts in blocks
        for block_row in range(0, size, block_size):
            for block_col in range(0, size, block_size):
                block = population[:, block_row:block_row+block_size, block_col:block_col+block_size].reshape(num_individuals, -1)
                block_conflicts = np.sum(block[:, :, None] == block[:, None, :], axis=(1, 2)) - block_size * block_size
                fitness += block_conflicts
        
        # Heavily penalize incorrect fixed values heavily
        correct_values = population[0, fixed_indices[:, 0], fixed_indices[:, 1]]
        penalties = 10 * np.sum(population[:, fixed_indices[:, 0], fixed_indices[:, 1]] != correct_values, axis=1)
        fitness += penalties

        return fitness
    
    @staticmethod
    def create_initial_solution(puzzle: np.ndarray) -> np.ndarray:
        """Create a random solution from the given puzzle"""
        solution = puzzle.copy()
        empty_indices = np.argwhere(solution == 0)
        for i, j in empty_indices:
            solution[i, j] = np.random.randint(1, 10)
        return solution
    
    @staticmethod
    def create_initial_solution_bounded(puzzle: np.ndarray) -> np.ndarray:
        """Create a random solution from the given puzzle, but making sure that each
        block contains the numbers 1-9 exactly once."""
        solution = np.copy(puzzle)
        for block_row, block_col in np.ndindex(3, 3):
            block = puzzle[block_row*3:block_row*3+3, block_col*3:block_col*3+3]
            block_values = block.flatten()
            missing_values = np.setdiff1d(np.arange(1, 10), block_values)
            np.random.shuffle(missing_values)
            missing_values = list(missing_values)
            block_empty_indices = np.argwhere(block == 0)
            for i, j in block_empty_indices:
                solution[block_row*3 + i, block_col*3 + j] = missing_values.pop()
        return solution

    @classmethod
    def create_initial_population(cls, puzzle: np.ndarray, population_size: int) -> np.ndarray:
        """Create a random population from the given puzzle of shape (population_size, 9, 9)"""
        population = np.empty((population_size, puzzle.shape[0], puzzle.shape[1]), dtype=np.int8)
        for i in range(population_size):
            population[i] = cls.create_initial_solution(puzzle)
        return population
    
    @classmethod
    def create_initial_population_bounded(cls, puzzle: np.ndarray, population_size: int) -> np.ndarray:
        """Create a random population from the given puzzle of shape (population_size, 9, 9), but each 
        block of each board contains the numbers 1-9 exactly once.
        """
        population = np.empty((population_size, puzzle.shape[0], puzzle.shape[1]), dtype=np.int8)
        for i in range(population_size):
            population[i] = cls.create_initial_solution_bounded(puzzle)
        return population
    
    @staticmethod
    def create_children(current_generation: np.ndarray, children_amount: int):
        """Create children from the current generation using pairs of random parents."""
        # Generate indices for random pairs of parents
        parent_indices = np.random.choice(current_generation.shape[0], size = (2, children_amount), replace=True)

        # Select parents based on the indices
        fathers = current_generation[parent_indices[0, :]]
        mothers = current_generation[parent_indices[1, :]]

        # Create a mask for random selection of genes from father and mother
        crossover_mask = np.random.rand(children_amount, *fathers.shape[1:]) < 0.5

        # Create children using where operation and the mask
        children = np.where(crossover_mask, fathers, mothers).astype(np.int8)

        return children
    
    @staticmethod #TODO: Vectorize this function for better performance
    def mutate_sudoku_population(population: np.ndarray, fixed_indices: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutate a population of Sudoku arrays with a given mutation rate.
            If a board from the population is selected for mutation, a random cell is changed to a random value
        """
        mutated_population = population.copy()
        for individual_index in range(population.shape[0]):
            if np.random.rand() < mutation_rate:
                i, j = np.random.randint(0, 9, 2)
                while np.any(np.all(fixed_indices == (i, j), axis=1)):
                    i, j = np.random.randint(0, 9, 2)
                mutated_population[individual_index, i, j] = np.random.randint(1, 10)
        return mutated_population
    
    @staticmethod
    def get_neighbors(current_population: np.ndarray, fixed_indices: np.ndarray, number_of_swaps: int = 2):
        """Create a new population by swapping random cells in the current population except for fixed indices."""
        new_population = current_population.copy()
        for _ in range(number_of_swaps):
                i, j = np.random.randint(0, 9, 2)
                # Make sure that the indices are not fixed
                while np.any(np.all(fixed_indices == (i, j), axis=1)):
                    i, j = np.random.randint(0, 9, 2)
                i_new, j_new = np.random.randint(0, 9, 2)
                # Make sure that the new indices are not fixed
                while np.any(np.all(fixed_indices == (i_new, j_new), axis=1)):
                    i_new, j_new = np.random.randint(0, 9, 2)
                # Swap the values in the new population
                new_population[:, [i, i_new], [j, j_new]] = new_population[:, [i_new, i], [j_new, j]]
        return new_population
    
    @staticmethod
    def accept_population(current_population: np.ndarray, 
                          new_population: np.ndarray, 
                          current_energies: np.ndarray, 
                          new_energies: np.ndarray, 
                          temperature: float) -> np.ndarray: #TODO: Fix this putput type hint
        """Accept or reject new population based on probability from Boltzmann distribution"""

        # Comparison to get mask for better solutions
        better_solutions = new_energies < current_energies

        # Calculate probability for worse solutions
        worse_probabilities = np.exp((current_energies - new_energies) / temperature)

        # Combine probabilities
        probabilities = np.where(better_solutions, 1, worse_probabilities)

        # Accept or reject new population based on probabilities
        accept_mask = np.random.rand(current_population.shape[0]) < probabilities
        accepted_population = np.where(accept_mask[:, None, None], new_population, current_population)
        accepted_energies = np.where(accept_mask, new_energies, current_energies)

        return accepted_population, accepted_energies
    
    @staticmethod
    def accepatance_probability_old(current_energy, new_energy, temperature):
        if new_energy < current_energy:
            return 1
        else:
            return np.exp((current_energy - new_energy) / temperature)


class GeneticAlgorithm(SudokuAlgorithm):
    def __init__(self,
                    population_size: int = 10000,
                    selection_rate: float = 0.2,
                    max_generations: int = 40000,
                    individual_mutation_rate: float = 0.25,
                    restart_after_n_generations: int = 40,
                    ):
        
        self.so = StochasticOperations() # Dependency injection

        self.population_size = population_size
        self.selection_rate = selection_rate
        self.max_generations = max_generations
        self.individual_mutation_rate = individual_mutation_rate
        self.restart_after_n_generations = restart_after_n_generations
        self.fitness_history = []

    def __call__(self, sudoku: np.ndarray, show_live_plot: bool = False, show_end_plot: bool = False) -> np.ndarray:
        self.fitness_history = []
        sudoku = np.array(sudoku, dtype=np.int8)

        start_time = time()

        if show_live_plot:
            pass #TODO: Implement live plot

        #Initizalize variables
        selection_amount = int(self.population_size * self.selection_rate)
        children_amount = self.population_size - selection_amount
        iteration = 0
        found_solution = False
        local_minima_loop_count = 0

        # Create initial population
        solution = np.empty((9, 9), dtype=np.int8)
        fixed_indices = self.so.get_fixed_indices(sudoku)
        current_generation = self.so.create_initial_population(puzzle = sudoku, population_size = self.population_size)

        # Main loop
        while iteration < self.max_generations and not found_solution:

            # Calculate fitness
            fitness = self.so.get_fitness(current_generation, fixed_indices)
            fitness_indices = np.argsort(fitness)
            
            # Store best fitness
            self.fitness_history.append(fitness[fitness_indices[0]])

            # Check if solution is found
            if fitness[fitness_indices[0]] == 0:
                found_solution = True
                solution = current_generation[fitness_indices[0]]
            
            # Count if we are stuck in a local minima
            if iteration > 2 and self.fitness_history[-1] == self.fitness_history[-2]:
                local_minima_loop_count += 1
            else:
                local_minima_loop_count = 0

            # Check if we are stuck in a local minima
            if local_minima_loop_count >= self.restart_after_n_generations:
                print(f"\nStuck in local minima for {local_minima_loop_count} generations at iteration {iteration}. Restarting population.")
                current_generation = self.so.create_initial_population(sudoku, self.population_size)
                local_minima_loop_count = 0
                continue
            
            # Initialize the next generation
            next_generation = np.empty_like(current_generation, dtype=np.int8)

            # Add most fit individuals to the next generation
            next_generation[:selection_amount] = current_generation[fitness_indices[:selection_amount]]

            # Create children from the current generation and add to the next generation
            children = self.so.create_children(current_generation, children_amount)
            next_generation[selection_amount:] = children

            # Mutate the next generation
            next_generation = self.so.mutate_sudoku_population(next_generation, fixed_indices, self.individual_mutation_rate)

            # Update current generation
            current_generation = next_generation

            # Increment iteration
            iteration += 1

            # Print progress
            if iteration % 5 == 0:
                print("-----------------------------")
                print(f"current generation: {iteration} \
                    \ncurrent best fitness: {self.fitness_history[-1]} \
                    \nelapsed time: {time() - start_time:.2f}")
            
                #Update live plot 
                if show_live_plot: #TODO: Implement live plot
                    pass
        
        # Show final plot
        if show_end_plot: #TODO: Implement final plot
            pass

        # Print final message
        if found_solution:
            print("-----------------------------")
            print(f"\nSolution found after {iteration} generations and {time() - start_time:.2f} seconds.")
            return solution
        else:
            print("-----------------------------")
            print(f"\nNo solution found after {iteration} generations and {time() - start_time:.2f} seconds.")
            print('Returning best solution found.')
            return current_generation[fitness_indices[0]]
        
class SimulatedAnnealing(SudokuAlgorithm):
    def __init__(
            self,
            final_temperature: float = 0.01,
            end_after_n_restarts: int = 10,
            restart_after_n_reheats: int = 5,
            ):
        
        self.so = StochasticOperations() # Dependency injection

        self.final_temperature = final_temperature
        self.end_after_n_restarts = end_after_n_restarts
        self.restart_after_n_reheats = restart_after_n_reheats
        self.energy_history = []

    def __call__(self, sudoku: np.ndarray, show_live_plot: bool = False, show_end_plot: bool = False) -> np.ndarray:
        self.energy_history = []
        sudoku = np.array(sudoku, dtype=np.int8)

        start_time = time()

        if show_live_plot: #TODO: Implement live plot
            pass

        fixed_indices = self.so.get_fixed_indices(sudoku)

        # Calculate initial temperature, which is the standard deviation of the energy
        random_solutions = self.so.create_initial_population_bounded(sudoku, 100)
        random_solutions_energies = self.so.get_fitness(random_solutions, fixed_indices)
        initial_temperature = np.std(random_solutions_energies) / 3

        # Calculate population size, which should be proportional to the number of fixed values
        population_size = fixed_indices.shape[0]

        # Initialize variables
        cooling_rate = 1 - self.final_temperature / 10
        restart_counts = 0
        iteration = 0
        found_solution = False

        # Initialize solution
        solution = np.empty((9, 9), dtype=np.int8)

        # Outer loop for restarts
        while restart_counts < self.end_after_n_restarts and not found_solution:

            # Create initial population
            current_populaion = self.so.create_initial_population_bounded(sudoku, population_size)
            current_energies = self.so.get_fitness(current_populaion, fixed_indices)
            self.energy_history.append(np.min(current_energies))

            # Reset variables
            temperature = initial_temperature
            reheats = 0

            # Inner loop for Simulated Annealing
            while temperature > self.final_temperature and not found_solution:
                
                # Check if solution is found
                if self.energy_history[-1] == 0:
                    found_solution = True
                    solution = current_populaion[np.argmin(current_energies)]

                # Create new population
                number_of_swaps = np.random.randint(1, 4)
                new_population = self.so.get_neighbors(current_populaion, fixed_indices, number_of_swaps)

                # Calculate new population energies
                new_energies = self.so.get_fitness(new_population, fixed_indices)

                # Accept or reject mebers of the new population
                current_populaion, current_energies = self.so.accept_population(current_populaion, new_population, 
                                                              current_energies, new_energies, temperature)

                # Store lowest energy
                self.energy_history.append(np.min(current_energies))

                # Update cooling rate and temperature
                if temperature < self.final_temperature * 2:
                    cooling_rate = 1 - self.final_temperature / 100
                else:
                    cooling_rate = 1 - self.final_temperature / 10
                
                temperature *= cooling_rate

                # Check if reheating is needed
                if temperature < self.final_temperature and reheats < self.restart_after_n_reheats:
                    temperature *= (1 / self.final_temperature) * 1.1**reheats
                    reheats += 1
                
                # Increment iteration
                iteration += 1
                
                # Print progress
                if iteration % 2000 == 0:
                    print("\n-----------------------------")
                    print(f"current iteration: {iteration} \
                        \ncurrent lowest energy: {self.energy_history[-1]} \
                        \ncurrent temperature: {temperature:.3f} \
                        \nelapsed time: {time() - start_time:.2f}")
                    
                    # Update live plot
                    if show_live_plot: #TODO: Implement live plot
                        pass
                
            # Increment restart counts
            restart_counts += 1

            # Check if restart is needed
            if not found_solution and restart_counts < self.end_after_n_restarts:
                print(f"Restarting population {restart_counts} after {iteration} iterations.")
        
        # Show final plot
        if show_end_plot: #TODO: Implement final plot
            pass
        
        if found_solution:
            print("-----------------------------")
            print(f"\nSolution found after {iteration} iterations and {time() - start_time:.2f} seconds.")
            return solution
        else:
            print("-----------------------------")
            print(f"\nNo solution found after {iteration} iterations and {time() - start_time:.2f} seconds.")
            print('Returning best solution found.')
            return current_populaion[np.argmin(current_energies)]



if __name__ == '__main__':
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

    # solver = SudokuSolver(algorithm = GeneticAlgorithm())
    solver = SudokuSolver(algorithm = SimulatedAnnealing())
    solution = solver.solve(puzzle)



        
                



            





           