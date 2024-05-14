# File to solve sudoku in command line
import numpy as np
from algorithms.geneticalgorithm import GeneticAlgorithm
from algorithms.simulatedannealing import SimulatedAnnealing
from algorithms.sudokualgorithm import SudokuAlgorithm
from core.sudokusolver import SudokuSolver
import argparse

if __name__ == "__main__":
    print('Sudoku Solver')
    print('-------------')
        
    puzzle = np.array([
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9]
        ])
    
    print('Puzzle is:')
    SudokuSolver.print_puzzle(puzzle)
    print('Choose algorithm:')
    print('1. Genetic Algorithm')
    print('2. Simulated Annealing')
    print('3. Exit')
    choice = int(input('Enter choice: '))
    if choice == 1:
        algorithm = GeneticAlgorithm()
    elif choice == 2:
        algorithm = SimulatedAnnealing()
    else:
        exit()
    print('Solving...')
    solver = SudokuSolver(algorithm)
    solution = solver.solve(puzzle)

    