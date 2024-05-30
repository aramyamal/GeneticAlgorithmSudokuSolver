import numpy as np

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
        
        # Heavily penalize incorrect fixed values
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
        # print(f'fathers: {fathers.shape}')
        mothers = current_generation[parent_indices[1, :]]

        # Create a mask for random selection of genes from father and mother, where each gene is a 3x3 block
        crossover_mask = np.random.rand(children_amount, 3, 3) < 0.5

        crossover_mask = np.repeat(np.repeat(crossover_mask, repeats = 3, axis = 1), repeats = 3, axis = 2)

        # Create children using where operation and the mask
        children = np.where(crossover_mask, fathers, mothers).astype(np.int8)

        return children
    
    # @staticmethod
    # def create_children(current_generation: np.ndarray, children_amount: int):
    #     """Creates children from the current generation using vectorized block-based crossover."""

    
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
    
    @staticmethod #TODO: Vectorize this function for better performance
    def mutate_sudoku_population_bounded(population: np.ndarray, fixed_indices: np.ndarray, mutation_rate: float, number_of_swaps=3) -> np.ndarray:
        """Mutate a population of Sudoku arrays with a given mutation rate.
            If a board from the population is selected for mutation, two random cells are swapped within a random square
        """
        # mutated_population = population.copy()
        # for individual_index in range(population.shape[0]):
        #     if np.random.rand() < mutation_rate:
        #         for _ in range(0, number_of_swaps):
        #             square_index = np.random.randint(0, 3, 2) * 3
        #             i, j = np.random.randint(0, 3, 2) + square_index
        #             # Make sure that the indices are not fixed
        #             while np.any(np.all(fixed_indices == (i, j), axis=1)):
        #                 square_index = np.random.randint(0, 3, 2) * 3
        #                 i, j = np.random.randint(0, 3, 2) + square_index
        #             i_new, j_new = np.random.randint(0, 3, 2) + square_index
        #             # Make sure that the new indices are not fixed
        #             while np.any(np.all(fixed_indices == (i_new, j_new), axis=1)):
        #                 i_new, j_new = np.random.randint(0, 3, 2) + square_index
        #             mutated_population[individual_index, i, j], mutated_population[individual_index, i_new, j_new] = mutated_population[individual_index, i_new, j_new], mutated_population[individual_index, i, j]
        # return mutated_population
    
        mutate_mask = np.random.rand(population.shape[0]) < mutation_rate
        new_population = population.copy()
        for _ in range(np.random.randint(0, number_of_swaps)):
                square_index = np.random.randint(0, 3, 2) * 3
                i, j = np.random.randint(0, 3, 2) + square_index
                # Make sure that the indices are not fixed
                while np.any(np.all(fixed_indices == (i, j), axis=1)):
                    square_index = np.random.randint(0, 3, 2) * 3
                    i, j = np.random.randint(0, 3, 2) + square_index
                i_new, j_new = np.random.randint(0, 3, 2) + square_index
                # Make sure that the new indices are not fixed
                while np.any(np.all(fixed_indices == (i_new, j_new), axis=1)):
                    i_new, j_new = np.random.randint(0, 3, 2) + square_index
                # Swap the values in the new population
                new_population[:, [i, i_new], [j, j_new]] = new_population[:, [i_new, i], [j_new, j]]
        return np.where(mutate_mask[:, None, None], new_population, population)

    @staticmethod
    def get_neighbors(current_population: np.ndarray, fixed_indices: np.ndarray, number_of_swaps: int = 2):
        """Create a new population by swapping random cells in the current population except for fixed indices."""
        new_population = current_population.copy()
        for _ in range(number_of_swaps):
                square_index = np.random.randint(0, 3, 2) * 3
                i, j = np.random.randint(0, 3, 2) + square_index
                # Make sure that the indices are not fixed
                while np.any(np.all(fixed_indices == (i, j), axis=1)):
                    square_index = np.random.randint(0, 3, 2) * 3
                    i, j = np.random.randint(0, 3, 2) + square_index
                i_new, j_new = np.random.randint(0, 3, 2) + square_index
                # Make sure that the new indices are not fixed
                while np.any(np.all(fixed_indices == (i_new, j_new), axis=1)):
                    i_new, j_new = np.random.randint(0, 3, 2) + square_index
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
    