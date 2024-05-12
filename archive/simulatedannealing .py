from stochasticmethods import StochasticMethods
from numpy.typing import NDArray
import numpy as np
import time
import typing
import matplotlib.pyplot as plt

class SimulatedAnnealing(StochasticMethods):
    def __init__(
        self,
        initial_temperature: float = 1,
        final_temperature: float = 0.01,
        cooling_rate: float = .999,
                ):
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.losses = []

        super().__init__()
    
    def __call__(self, puzzle: NDArray[np.int_], show_live_plot: bool = False, restarts: int = 5, reheats = 5) -> NDArray[np.int_]:
        self.losses = []
        puzzle = np.array(puzzle)

        fixed_indices = self.get_fixed_indices(puzzle)

        # calculate initial temperature by taking standard deviation of 100 random solutions
        print('Calculating initial temperature...')
        random_solutions = np.array([self.create_initial_solution_bounded(puzzle) for _ in range(100)])
        random_solution_energies = self.get_fitness(random_solutions, fixed_indices=fixed_indices)
        self.initial_temperature = np.std(random_solution_energies) 
        print(f'Initial temperature: {self.initial_temperature}')

        self.initial_temperature /= 3

        #calculate population size per iteration which should be number of fixed indices
        population_size = fixed_indices.shape[0] 
        # population_size = 1

        start_time = time.time()

        fixed_indices = self.get_fixed_indices(puzzle)

        restart_counts = 0
        iteration = 0
        solution_found = False


        while restart_counts < restarts and not solution_found:
            solution = self.create_first_generation_bounded(puzzle, population_size)
            current_loss = self.get_fitness(solution, fixed_indices=fixed_indices)
            self.losses.append(np.min(current_loss))

            temperature = self.initial_temperature

            reheat_counts = 0

            while temperature > self.final_temperature and not solution_found:
                if self.losses[-1] == 0:
                    solution_found = True
                    best_solution = new_solution[np.argmin(new_loss)]

                number_of_swaps = np.random.randint(1, 4)
                new_solution = self.vectorized_get_neighbor_bounded_semi_optimized(solution, fixed_indices= fixed_indices, number_of_swaps=number_of_swaps)
                new_loss = self.get_fitness(new_solution, fixed_indices=fixed_indices)

                for i in range(population_size):
                    if self.acceptance_probability(current_loss[i], new_loss[i], temperature) > np.random.rand():
                        solution[i] = new_solution[i]

                current_loss = self.get_fitness(solution, fixed_indices=fixed_indices)

                self.losses.append(np.min(current_loss))

                if temperature < self.final_temperature * 5:
                    self.cooling_rate = 0.9999
                else:
                    self.cooling_rate = 0.999

                temperature *= self.cooling_rate

                if temperature < self.final_temperature and reheat_counts < reheats:
                    temperature *= (1/self.final_temperature) * 1.1**reheat_counts
                    reheat_counts += 1

                if iteration % 1000 == 0:
                    print(f'Iteration: {iteration}, Temperature: {temperature}, Loss: {self.losses[-1]}')
                    print(f'Elapsed time: {time.time() - start_time} seconds')

                    if show_live_plot:
                        self.contious_loss_plot(self.losses, xlabel='iteration', 
                                                ylabel='loss', 
                                                scatter=False,
                                                temperature=temperature, 
                                                lowest_current_loss=np.min(current_loss), 
                                                lowest_loss = np.min(self.losses), 
                                                restart_counts=restart_counts,
                                                population_size=population_size,
                                                reheat_counts=reheat_counts)

                iteration += 1
            
            restart_counts += 1
            if not solution_found and restart_counts < restarts:
                print(f'Restarting {restart_counts}...')

        if not solution_found:
            print(f'Solution not found after {iteration} iterations and {restart_counts} restarts. Printing best solution:')
            print(solution[0])
            best_solution = solution[0]
        
        else:
            print(f'Solution found after {iteration} iterations and temperature {temperature}. Printing solution:')
            print(best_solution)

        self.plot_loss(self.losses, xlabel='iteration', ylabel='loss', scatter=False, temperature=temperature, lowerst_current_loss=np.min(current_loss), lowest_loss = np.min(self.losses), restart_counts=restart_counts)
        return best_solution
    
    # def get_neighbor(self, solution: NDArray[np.int_], fixed_indices: NDArray[np.int_], number_of_swaps = 2) -> NDArray[np.int_]:
    #     new_solution = solution.copy()
    #     for _ in range(number_of_swaps):
    #         i, j = np.random.randint(0, 9, 2)
    #         while np.any(np.all(fixed_indices == (i, j), axis=1)):
    #             i, j = np.random.randint(0, 9, 2)
    #         new_solution[i, j] = np.random.randint(1, 10)
    #     return new_solution
    
    def vectorized_get_neighbor(self, current_solution_population: NDArray[np.int_], fixed_indices, number_of_swaps = 2,) -> NDArray[np.int_]:
        new_solution_population = current_solution_population.copy() 
        for _ in range(number_of_swaps):
            i, j = np.random.randint(0, 9, 2)
            while np.any(np.all(fixed_indices == (i, j), axis=1)):
                i, j = np.random.randint(0, 9, 2)
            new_solution_population[:, i, j] = np.random.randint(1, 10, current_solution_population.shape[0])
        return new_solution_population

    def get_neighbor_bounded_not_vectorized(self, current_solution_population, fixed_indices, number_of_swaps = 2):
        new_solution_population = current_solution_population.copy()
        for _ in range(number_of_swaps):
            for individual_index in range(new_solution_population.shape[0]):
                i, j = np.random.randint(0, 9, 2)
                while np.any(np.all(fixed_indices == (i, j), axis=1)):
                    i, j = np.random.randint(0, 9, 2)
                i_new, j_new = np.random.randint(0, 9, 2)
                while np.any(np.all(fixed_indices == (i_new, j_new), axis=1)):
                    i_new, j_new = np.random.randint(0, 9, 2)
                new_solution_population[individual_index, i, j], new_solution_population[individual_index, i_new, j_new] = new_solution_population[individual_index, i_new, j_new], new_solution_population[individual_index, i, j]
        return new_solution_population
    
    def get_neighbor_bounded_vectorized(self, current_solution_population, fixed_indices, number_of_swaps=2):

        new_solution_population = current_solution_population.copy()
        population_size = new_solution_population.shape[0]

        all_indices = np.array(np.meshgrid(np.arange(9), np.arange(9))).T.reshape(-1, 2)
        valid_indices = all_indices[~np.any(np.all(all_indices == fixed_indices[:, None], axis=-1), axis=0)]
        # print('________________________')
        # print(valid_indices.shape)
        # print('________________________')
        for _ in range(number_of_swaps):

            # Generate valid swap indices for all individuals in the population
            random_index = np.random.randint(valid_indices.shape[0])
            i, j = valid_indices[random_index]
            random_index = np.random.randint(valid_indices.shape[0])
            i_new, j_new = valid_indices[random_index]

            # Perform the swap using advanced indexing
            new_solution_population[:, i, j], new_solution_population[:, i_new, j_new] = new_solution_population[:, i_new, j_new], new_solution_population[:, i, j]

        return new_solution_population
    
    def vectorized_get_neighbor_bounded_semi_optimized(self, current_solution_population, fixed_indices, number_of_swaps = 2):
        new_solution_population = current_solution_population.copy()
        for _ in range(number_of_swaps):
                i, j = np.random.randint(0, 9, 2)
                while np.any(np.all(fixed_indices == (i, j), axis=1)):
                    i, j = np.random.randint(0, 9, 2)
                i_new, j_new = np.random.randint(0, 9, 2)
                while np.any(np.all(fixed_indices == (i_new, j_new), axis=1)):
                    i_new, j_new = np.random.randint(0, 9, 2)
                new_solution_population[:, [i, i_new], [j, j_new]] = new_solution_population[:, [i_new, i], [j_new, j]]
                # new_solution_population[:, i, j], new_solution_population[:, i_new, j_new] = new_solution_population[:, i_new, j_new], new_solution_population[:, i, j]
        return new_solution_population

    # def vectorized_get_neighbor_bounded(self, current_solution_population, fixed_indices, number_of_swaps = 2):
    #     def generate_valid_swap_indices(population_size, fixed_indices):
    #         """
    #         Generate valid swap indices for a population while avoiding fixed indices.
    #         """
    #         all_indices = np.array(list(np.ndindex(9, 9)))
    #         valid_indices = np.array([idx for idx in all_indices if not np.any(np.all(fixed_indices == idx, axis=1))])

    #         random_index = np.random.choice(valid_indices.shape[0], size=population_size, replace=True)
    #         return valid_indices[random_index][:, 0], valid_indices[random_index][:, 1] 
        
    #     population_size = current_solution_population.shape[0]
    #     new_solution_population = current_solution_population.copy()

    #     for _ in range(number_of_swaps):

    #         # Generate valid swap indices for all individuals in the population
    #         i, j = generate_valid_swap_indices(population_size, fixed_indices)
    #         i_new, j_new = generate_valid_swap_indices(population_size, fixed_indices)

    #         # Perform the swap using advanced indexing
    #         new_solution_population[:, i, j], new_solution_population[:, i_new, j_new] = new_solution_population[:, i_new, j_new], new_solution_population[:, i, j]

    #     return new_solution_population

    def acceptance_probability(self, current_loss: int, new_loss: int, temperature: float) -> float:
        if new_loss < current_loss:
            return 1.0
        else: 
            return np.exp((current_loss - new_loss) / temperature)
        
    def create_initial_solution_bounded(self, puzzle: np.ndarray) -> np.ndarray:
        """Create a random solution from the given puzzle, but making sure that each
        block contains the numbers 1-9 exactly once."""
        solution = np.copy(puzzle)
        empty_indices = np.argwhere(puzzle == 0)
        # for i, j in empty_indices:
        #     block_row, block_col = i // 3, j // 3
        #     block = puzzle[block_row*3:block_row*3+3, block_col*3:block_col*3+3]
        #     block_values = block.flatten()
        #     missing_values = np.setdiff1d(np.arange(1, 10), block_values)
        #     solution[i, j] = np.random.choice(missing_values)
        # return solution

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

    

    
    def create_first_generation_bounded(self, puzzle, population_size: int) -> np.ndarray:
        population = np.empty((population_size, puzzle.shape[0], puzzle.shape[1]), dtype=np.int8)
        for i in range(population_size):
            population[i] = self.create_initial_solution_bounded(puzzle)
        return population
        
    def get_energy(self, puzzle, fixed_indices):
        puzzle = puzzle[None, :, :]
        return self.get_fitness(puzzle, fixed_indices)[0]
    
    def get_energies(self, population: NDArray[np.int_]) -> NDArray[np.int_]:
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

        return fitness
        
if __name__ == '__main__':
    sm = SimulatedAnnealing()
    puzzle = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])
    
    hard = [
            [0, 0, 6, 1, 0, 0, 0, 0, 8], 
            [0, 8, 0, 0, 9, 0, 0, 3, 0], 
            [2, 0, 0, 0, 0, 5, 4, 0, 0], 
            [4, 0, 0, 0, 0, 1, 8, 0, 0], 
            [0, 3, 0, 0, 7, 0, 0, 4, 0], 
            [0, 0, 7, 9, 0, 0, 0, 0, 3], 
            [0, 0, 8, 4, 0, 0, 0, 0, 6], 
            [0, 2, 0, 0, 5, 0, 0, 8, 0], 
            [1, 0, 0, 0, 0, 2, 5, 0, 0]
        ]
    sm(puzzle, show_live_plot=True, restarts=50)
    # fixed_indices = sm.get_fixed_indices(puzzle)
    # population = sm.create_first_generation(puzzle, 10)
    # print(sm.vectorized_get_neighbor(population, fixed_indices))
    # block_values = [0, 1, 0 ,0, 3, 2, 4, 5, 0]
    # print(np.setdiff1d(np.arange(1, 10), block_values))

    # print(sm.create_initial_solution_bounded(puzzle))

