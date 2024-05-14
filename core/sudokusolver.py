from algorithms.sudokualgorithm import SudokuAlgorithm
import numpy as np

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