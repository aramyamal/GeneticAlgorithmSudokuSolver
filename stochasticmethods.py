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
    
    def plot_loss(self, loss: NDArray[np.float_], xlabel='iteration', ylabel= 'loss', **kwargs):
        x_list = range(1, len(loss) + 1)
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
    
    def contious_loss_plot(self, loss: NDArray[np.float_], xlabel='iteration', ylabel= 'loss', **kwargs):
        x_list = range(1, len(loss) + 1)
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
