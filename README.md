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

In the main.py file, input the sudoku board you wish to be solved as a 9x9 array in the ga.sudokuGA() fuction.
Note that two example boards are already included in the sudoku_examples.py file, one easy and one hard, and their respective solutions are
also included. You can also choose to see a continous plot of the best fitness
or/and see a final plot when the algorithm has stopped running. In both plots the parameters for the algorithm is included.

```
ga.sudokuGA(sudoku_examples.easy, show_continous_plot=True, show_final_plot=True, print_final_board=True)
```

If you wish to change the parameters you can do so in the definition of the sudokuGa funtion in GeneticAlgorithmSudokuSolver.py.
The following are the standard parameters I have found to work best for me, but of course there is ample room for improvement so feel free
to change them.

```
population_size = 20000
selection_rate = 0.2
random_selection_rate = 0.1
max_generations = 1000
individual_mutation_rate = 0.45
cell_mutation_rate = 0.005
restart_after_n_generations = 20
```

The following video shows a timelapse of the continous plot when solving the preloaded easy sudoku board with
the parameters above.

https://user-images.githubusercontent.com/116388893/235634335-00561fae-b327-4eb1-9f51-0c79d8440d2e.mp4

