from abc import ABC, abstractmethod
import numpy as np

class SudokuAlgorithm(ABC):
    """Abstract base class for Sudoku solving algorithms"""

    @abstractmethod
    def __call__(self, sudoku: np.ndarray) -> np.ndarray:
        pass