# GeneticAlgorithmSudokuSolver
This project allows you to solve a Sudoku board using a genetic algorithm. The pseduo code for the algorithm is as follows:

```
generate initial population
repeat
    rank the solutions, and retain only the percentage specified by selection rate
    repeat
        randomly select two solutions from the population
        randomly choose a crossover point
        recombine the solutions to produce n new solutions
        apply the mutation operator to the solutions
    until a new population has been produced
until a solution is found or the maximum number of generations is reached
```

Call an instance of GeneticAlgorithm class with a sudoku board array as input.
Note that two example boards are already included in the sudoku_examples.py file, one easy and one hard, along with their respective solutions. 
You can also choose to see a continous plot of the best fitness.

Example:

```
ga = GeneticAlgorithm()
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
ga(puzzle, show_live_plot=True)
```

The parameters for the algorithm can be changed in the init.
The following parameters are the ones I have found to work best.

```
population_size: int = 10000,
selection_rate: float = 0.2,
random_selection_rate: float = 0.0
max_generations: int = 40000
individual_mutation_rate: float = 0.25,
cell_mutation_amount: float = 1,
restart_after_n_generations: int = 40
```

The following video shows a timelapse of the continous plot when solving the preloaded easy sudoku board with
the parameters above,

https://user-images.githubusercontent.com/116388893/235634335-00561fae-b327-4eb1-9f51-0c79d8440d2e.mp4

where the following solution was printed:
```
[[5 3 4 6 7 8 9 1 2]
 [6 7 2 1 9 5 3 4 8]
 [1 9 8 3 4 2 5 6 6]
 [8 5 9 7 6 1 4 2 3]
 [4 2 6 8 5 3 7 9 1]
 [7 1 3 9 2 4 8 5 6]
 [9 6 1 5 3 7 2 8 4]
 [2 8 7 4 1 9 6 3 5]
 [3 4 5 2 8 6 1 7 9]]
```
