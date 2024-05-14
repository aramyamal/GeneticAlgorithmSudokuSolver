from core.utils.stochasticoperations import StochasticOperations
from algorithms.sudokualgorithm import SudokuAlgorithm
import numpy as np
from time import time

class SimulatedAnnealing(SudokuAlgorithm):
    def __init__(
            self,
            final_temperature: float = 0.01,
            end_after_n_restarts: int = 10,
            restart_after_n_reheats: int = 5,
            ):
        
        self.so = StochasticOperations() # Dependency injection

        self.final_temperature = final_temperature
        self.end_after_n_restarts = end_after_n_restarts
        self.restart_after_n_reheats = restart_after_n_reheats
        self.energy_history = []

    def __call__(self, sudoku: np.ndarray, show_live_plot: bool = False, show_end_plot: bool = False) -> np.ndarray:
        self.energy_history = []
        sudoku = np.array(sudoku, dtype=np.int8)

        start_time = time()

        if show_live_plot: #TODO: Implement live plot
            pass

        fixed_indices = self.so.get_fixed_indices(sudoku)

        # Calculate initial temperature, which is the standard deviation of the energy
        random_solutions = self.so.create_initial_population_bounded(sudoku, 100)
        random_solutions_energies = self.so.get_fitness(random_solutions, fixed_indices)
        initial_temperature = np.std(random_solutions_energies) / 3

        # Calculate population size, which should be proportional to the number of fixed values
        population_size = fixed_indices.shape[0]

        # Initialize variables
        cooling_rate = 1 - self.final_temperature / 10
        restart_counts = 0
        iteration = 0
        found_solution = False

        # Initialize solution
        solution = np.empty((9, 9), dtype=np.int8)

        # Outer loop for restarts
        while restart_counts < self.end_after_n_restarts and not found_solution:

            # Create initial population
            current_populaion = self.so.create_initial_population_bounded(sudoku, population_size)
            current_energies = self.so.get_fitness(current_populaion, fixed_indices)
            self.energy_history.append(np.min(current_energies))

            # Reset variables
            temperature = initial_temperature
            reheats = 0

            # Inner loop for Simulated Annealing
            while temperature > self.final_temperature and not found_solution:
                
                # Check if solution is found
                if self.energy_history[-1] == 0:
                    found_solution = True
                    solution = current_populaion[np.argmin(current_energies)]

                # Create new population
                number_of_swaps = np.random.randint(1, 4)
                new_population = self.so.get_neighbors(current_populaion, fixed_indices, number_of_swaps)

                # Calculate new population energies
                new_energies = self.so.get_fitness(new_population, fixed_indices)

                # Accept or reject mebers of the new population
                current_populaion, current_energies = self.so.accept_population(current_populaion, new_population, 
                                                              current_energies, new_energies, temperature)

                # Store lowest energy
                self.energy_history.append(np.min(current_energies))

                # Update cooling rate and temperature
                if temperature < self.final_temperature * 2:
                    cooling_rate = 1 - self.final_temperature / 100
                else:
                    cooling_rate = 1 - self.final_temperature / 10
                
                temperature *= cooling_rate

                # Check if reheating is needed
                if temperature < self.final_temperature and reheats < self.restart_after_n_reheats:
                    temperature *= (1 / self.final_temperature) * 1.1**reheats
                    reheats += 1
                
                # Increment iteration
                iteration += 1
                
                # Print progress
                if iteration % 2000 == 0:
                    print("\n-----------------------------")
                    print(f"current iteration: {iteration} \
                        \ncurrent lowest energy: {self.energy_history[-1]} \
                        \ncurrent temperature: {temperature:.3f} \
                        \nelapsed time: {time() - start_time:.2f}")
                    
                    # Update live plot
                    if show_live_plot: #TODO: Implement live plot
                        pass
                
            # Increment restart counts
            restart_counts += 1

            # Check if restart is needed
            if not found_solution and restart_counts < self.end_after_n_restarts:
                print(f"Restarting population {restart_counts} after {iteration} iterations.")
        
        # Show final plot
        if show_end_plot: #TODO: Implement final plot
            pass
        
        if found_solution:
            print("-----------------------------")
            print(f"\nSolution found after {iteration} iterations and {time() - start_time:.2f} seconds.")
            return solution
        else:
            print("-----------------------------")
            print(f"\nNo solution found after {iteration} iterations and {time() - start_time:.2f} seconds.")
            print('Returning best solution found.')
            return current_populaion[np.argmin(current_energies)]
