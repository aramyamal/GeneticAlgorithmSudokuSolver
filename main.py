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
    
    choice = None
    print('Write your own puzzle? (y/n)')
    while choice not in ['y', 'n']:
        choice = input('Enter choice: ')
        if choice == 'y':
            puzzle = np.zeros((9, 9), dtype=np.int8)
            print('Enter the puzzle row by row, with 0s as empty cells')
            for i in range(9):
                row = input(f'Enter row {i+1}: ')
                if len(row) != 9:
                    print('Please enter a valid row. Exiting...')
                    exit()
                else:
                    puzzle[i] = [int(x) for x in row]

        elif choice == 'n':
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
        else:
            print('Please enter a valid choice. Write your own puzzle? (y/n)')


    print('-------------')
    print('Puzzle is:')
    SudokuSolver.print_puzzle(puzzle)
    print('-------------')
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

    