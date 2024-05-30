from .sudokualgorithm import SudokuAlgorithm
import numpy as np
from time import time

class BacktrackAlgorithm(SudokuAlgorithm):
    def __call__(self, sudoku: np.ndarray) -> np.ndarray:
        start = time()
        sudoku = np.array(sudoku, dtype=np.int8)
        self.backtrack(sudoku)
        print("-----------------------------")
        print(f"Solution found after {(time() - start)*1000:.2f} milliseconds using backtracking.")
        return sudoku

    @staticmethod
    def is_valid(sudoku, row, col, num):
        if num in sudoku[row]:
            return False
        
        if num in sudoku[:,col]:
            return False
        
        row_start = row - row % 3
        col_start = col - col % 3
        if num in sudoku[row_start:row_start+3, col_start:col_start+3]:
            return False

        return True

    @classmethod
    def backtrack(cls, sudoku):
        for row, col in np.ndindex(sudoku.shape):
            if sudoku[row, col] == 0:
                for num in range(1, 10):
                    if cls.is_valid(sudoku, row, col, num):
                        sudoku[row, col] = num
                        if cls.backtrack(sudoku):
                            return True
                        sudoku[row, col] = 0
                return False
        return True
