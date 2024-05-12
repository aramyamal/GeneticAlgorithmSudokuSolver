from sudokumethods import SudokuMethods
from numpy.typing import NDArray
import typing
import numpy as np
import matplotlib.pyplot as plt

class StochasticMethods(SudokuMethods):
    def __init__(self):
        super().__init__()
    
    def create_initial_solution(self, puzzle: NDArray[np.int_]) -> NDArray[np.int_]:
        """create a random solution from the given puzzle"""
        solution = puzzle.copy()
        empty_indices = self.get_empty_indices(solution)
        for i, j in empty_indices:
            solution[i, j] = np.random.randint(1, 10)
        return solution
    
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
    
    def create_first_generation(self, puzzle: NDArray[np.int_], population_size: int) -> NDArray[np.int_]:
        """create a random population from the given puzzle"""
        population = np.empty((population_size, puzzle.shape[0], puzzle.shape[1]), dtype=np.int8)
        for i in range(population_size):
            population[i] = self.create_initial_solution(puzzle)
        return population
    
    def plot_loss(self, loss: NDArray[np.float_], xlabel='iteration', ylabel= 'loss', scatter = False, **kwargs):
        x_list = range(1, len(loss) + 1)
        if scatter:
            plt.scatter(x_list, loss, s=1)
        else:
            plt.plot(x_list, loss)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs {xlabel}')

        if kwargs:
            text = [f'{key}: {value:.2f}' for key, value in kwargs.items()]
            text = '\n'.join(text)

            plt.figtext(0.15, 0.15,
                        text,
                        horizontalalignment='left',
                        wrap=False, fontsize=8,
                        bbox ={'facecolor':'white', 'alpha':0.3, 'pad':5})
        plt.show()
    
    def contious_loss_plot(self, loss: NDArray[np.float_], xlabel='iteration', ylabel= 'loss', scatter = False, **kwargs):
        x_list = range(1, len(loss) + 1)
        if scatter:
            plt.scatter(x_list, loss, s=1)
        else:
            plt.plot(x_list, loss)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs {xlabel}')

        if kwargs:
            text = [f'{key}: {value:.2f}' for key, value in kwargs.items()]
            text = '\n'.join(text)

            plt.figtext(0.15, 0.15,
                        text,
                        horizontalalignment='left',
                        wrap=False, fontsize=8,
                        bbox ={'facecolor':'white', 'alpha':0.3, 'pad':5})
        plt.pause(0.1)
        plt.clf()
    
if __name__ == "__main__":
    sm = StochasticMethods()
    print(sm.test(np.array([1, 2, 2, 4, 5, 6, 7, 8, 8])))
