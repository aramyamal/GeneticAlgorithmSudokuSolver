from core.utils.stochasticoperations import StochasticOperations
from algorithms.sudokualgorithm import SudokuAlgorithm
import numpy as np
from time import time

class GeneticAlgorithm(SudokuAlgorithm):
    def __init__(self,
                    so: StochasticOperations = StochasticOperations(),
                    population_size: int = 500,
                    selection_rate: float = 0.5,
                    max_generations: int = 20000,
                    individual_mutation_rate: float = 0.65,
                    restart_after_n_generations: int = 100,
                    ):
        
        self.so = so # Dependency injection

        self.population_size = population_size
        self.selection_rate = selection_rate
        self.max_generations = max_generations
        self.individual_mutation_rate = individual_mutation_rate
        self.restart_after_n_generations = restart_after_n_generations
        self.fitness_history = []

    def __call__(self, sudoku: np.ndarray, show_live_plot: bool = False, show_end_plot: bool = False) -> np.ndarray:
        self.fitness_history = []
        sudoku = np.array(sudoku, dtype=np.int8)

        start_time = time()

        if show_live_plot:
            pass #TODO: Implement live plot

        #Initizalize variables
        selection_amount = int(self.population_size * self.selection_rate)
        children_amount = self.population_size - selection_amount
        iteration = 0
        found_solution = False
        local_minima_loop_count = 0

        # Create initial population
        solution = np.empty((9, 9), dtype=np.int8)
        fixed_indices = self.so.get_fixed_indices(sudoku)
        current_generation = self.so.create_initial_population_bounded(puzzle = sudoku, population_size = self.population_size)

        # Main loop
        while iteration < self.max_generations and not found_solution:

            # Calculate fitness
            fitness = self.so.get_fitness(current_generation, fixed_indices)
            fitness_indices = np.argsort(fitness)
            
            # Store best fitness
            self.fitness_history.append(fitness[fitness_indices[0]])

            # Check if solution is found
            if fitness[fitness_indices[0]] == 0:
                found_solution = True
                solution = current_generation[fitness_indices[0]]
            
            # Count if we are stuck in a local minima
            if iteration > 2 and self.fitness_history[-1] == self.fitness_history[-2]:
                local_minima_loop_count += 1
            else:
                local_minima_loop_count = 0

            # Check if we are stuck in a local minima
            if local_minima_loop_count >= self.restart_after_n_generations:
                print(f"\nStuck in local minima for {local_minima_loop_count} generations at iteration {iteration}. Restarting population.")
                current_generation = self.so.create_initial_population_bounded(sudoku, self.population_size)
                local_minima_loop_count = 0
                continue
            
            # Initialize the next generation
            next_generation = np.empty_like(current_generation, dtype=np.int8)

            # Add most fit individuals to the next generation
            next_generation[:selection_amount] = current_generation[fitness_indices[:selection_amount]]

            # Create children from the current generation and add to the next generation
            children = self.so.create_children(current_generation, children_amount)
            # # Mutate children
            # children = self.so.mutate_sudoku_population_bounded(children, fixed_indices, self.individual_mutation_rate)
            next_generation[selection_amount:] = children

            # Mutate the next generation
            next_generation = self.so.mutate_sudoku_population_bounded(next_generation, fixed_indices, self.individual_mutation_rate)

            # Update current generation
            current_generation = next_generation

            # Increment iteration
            iteration += 1

            # Print progress
            if iteration % 200 == 0:
                print("-----------------------------")
                print(f"current generation: {iteration} \
                    \ncurrent best fitness: {self.fitness_history[-1]} \
                    \nelapsed time: {time() - start_time:.2f}")
            
                #Update live plot 
                if show_live_plot: #TODO: Implement live plot
                    pass
        
        # Show final plot
        if show_end_plot: #TODO: Implement final plot
            pass

        # Print final message
        if found_solution:
            print("-----------------------------")
            print(f"\nSolution found after {iteration} generations and {time() - start_time:.2f} seconds.")
            return solution
        else:
            print("-----------------------------")
            print(f"\nNo solution found after {iteration} generations and {time() - start_time:.2f} seconds.")
            print('Returning best solution found.')
            return current_generation[fitness_indices[0]]
