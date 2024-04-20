import numpy as np
from numpy.typing import NDArray
import typing
from random import randint, choices, choice, random

class SudokuMethods:
    def __init__(self):
        pass

    def get_duplicate_indices_1d(self, array: NDArray[np.int_]
                               = np.empty((9,))) -> NDArray[np.int_]:
        """returns the indices of the integer duplicates in a 1d array except for those of zeros"""

        # Filter out zeros and get indices of non-zero elements
        non_zero_indices = np.nonzero(array)[0]
        non_zero_values = array[non_zero_indices]
        
        # Find unique values and their first occurrence indices
        unique_values, index, counts = np.unique(non_zero_values, return_index=True, return_counts=True)

        # Mask to find values occurring more than once
        duplicates_mask = counts > 1
        
        # Get indices of all occurrences of duplicate values
        duplicate_values = unique_values[duplicates_mask]
        return non_zero_indices[np.isin(non_zero_values, duplicate_values)]

    def get_incorrect_indices(self, puzzle: NDArray[np.int_]) -> NDArray[np.int_]:
        """return all incorrect indices in a sudoku array"""
        incorrect_indices = []

         # Check rows for duplicates
        for i in range(9):
            duplicate_indices_row = self.get_duplicate_indices_1d(puzzle[i])
            incorrect_indices.extend([(i, idx) for idx in duplicate_indices_row])

       # Check columns for duplicates
        for j in range(9):
            duplicate_indices_column = self.get_duplicate_indices_1d(puzzle[:, j])
            incorrect_indices.extend([(idx, j) for idx in duplicate_indices_column])
        
        # Check 3x3 squares for duplicates
        mesh = (0, 3, 6)
        for k, l in np.nditer((mesh, mesh)):
            flattened_square = puzzle[k:k+3, l:l+3].flatten()
            for duplicate_indices_square in self.get_duplicate_indices_1d(flattened_square):
                incorrect_indices.extend([(k + duplicate_indices_square // 3, l + duplicate_indices_square % 3)])
        
        return np.array(incorrect_indices) 
    
    def get_empty_indices(self, puzzle: NDArray[np.int_]) -> NDArray[np.int_]:
        """return all empty indices in a sudoku array"""
        return np.argwhere(puzzle == 0)
    
    def get_fixed_indices(self, unmodified_puzzle: NDArray[np.int_]) -> NDArray[np.int_]:
        """return all fixed indices in an umodified sudoku array"""
        return np.argwhere(unmodified_puzzle != 0)
    
    def remove_fixed_indices_from_incorrect_indices(self, incorrect_indices: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        """remove fixed indices from incorrect indices"""
        # Create a boolean array of shape (len(incorrect_indices), len(fixed_indices))
        # Each entry is True if the row in incorrect_indices matches a row in fixed_indices
        mask = (incorrect_indices[:, None] == fixed_indices).all(-1)
        
        # Check if any True exists in each row; if True, then it matches and should be excluded
        filter_mask = ~mask.any(1)
        
        # Return the rows in incorrect_indices that do not match any row in fixed_indices
        return incorrect_indices[filter_mask]
    
    def generic_loss_function(self, puzzle: NDArray[np.int_], fixed_indices: NDArray[np.int_]) -> int:
        """generic loss function for sudoku puzzle that checks for number of incorrect indices in a puzzle"""
        incorrect_indices = self.get_incorrect_indices(puzzle)
        incorrect_indices = self.remove_fixed_indices_from_incorrect_indices(incorrect_indices, fixed_indices)
        return len(incorrect_indices)

# if __name__ == '__main__':

#     puzzle = np.array([
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [6, 9, 8, 0, 0, 0, 0, 6, 0],
#     [8, 0, 0, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ])

# helper = SudokuMethods()
# incorrect_indices = np.array([[0, 1], [1, 2], [3, 4], [9,3]])
# fixed_indices = np.array([[1, 2], [3, 4]])

# result = helper.remove_fixed_indices_from_incorrect_indices(incorrect_indices, fixed_indices)
# print("Filtered incorrect indices:", result)

