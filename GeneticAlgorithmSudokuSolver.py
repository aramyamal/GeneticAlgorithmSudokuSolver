# generate initial population
# repeat
#     rank the solutions, and retain only the percentage specified by selection rate
#     repeat
#         randomly select two solutions from the population
#         randomly choose a crossover point
#         recombine the solutions to produce n new solutions
#         apply the mutation operator to the solutions
#     until a new population has been produced
# until a solution is found or the maximum number of generations is reached

import numpy as np
from random import randint, choices, choice, random
import matplotlib.pyplot as plt
import time

def return_duplicate_index(list):
    """#returns index of all duplicates except for 0-duplicates, input: list"""
    dup = {}
    for i,x in enumerate(list):
        dup.setdefault(x,[]).append(i)
    duplicate_index = [x for i,x in dup.items() if len(x) > 1 and i > 0]
    return duplicate_index

def sudoku_validifier(puzzle):      
    """returns all duplicates of sudoku board and the corresponding
        2d-index of each duplicate. Return example of 1 duplicate:
        '[(3, 5), (6, 5)]'. 2 duplicates: '[(3, 5), (3, 8), (2, 8), (3, 8)]'.
        Multiple of the same index occurs because duplicates in both row/column and/or square
    """
    puzzle = np.array(puzzle)       
    duplicate_indices = []          
                                     
    for i in range(9):
        for dup in return_duplicate_index(puzzle[i]):
            for dup_index in dup:
                duplicate_indices.append((i, dup_index))

    for j in range(9):
        for dup in return_duplicate_index(puzzle[:,j]):
            for dup_index in dup:
                duplicate_indices.append((dup_index, j))

    mesh = (0,3,6)
    for k, l in np.nditer([mesh, mesh]):
        list = puzzle[k:k+3,l:l+3].flatten()
        for dup in return_duplicate_index(list):
            for dup_index in dup:
                duplicate_indices.append((dup_index//3 + k, dup_index%3 + l))
    return duplicate_indices

def remove_fixed_indices(badboard, fixedboard):   
    """Removes fixed indices from list. fix = list of fixed indices as '[(2,5), (3,7), ...].
        bad = list of new boardnumbers as [(3,7,9), (2,5,9), ...]
        where first two integers are indices and third is their number
    """                                  
    return [badboard[x] for x in range(len(badboard)) if badboard[x][:-1] not in fixedboard]  

def fitness_function(puzzle, fixed_indices):
    """number of duplicates without the fixed numbers"""
    return len(remove_fixed_indices(sudoku_validifier(puzzle), fixed_indices))

def create_child(father, mother):
    """Create matrix from father and mother"""
    father, mother = np.array(father), np.array(mother)
    child = father.copy()
    for i,j in np.ndindex((child.shape)):
        if randint(0,1) == 1:
            child[i,j] = mother[i,j]
    return child

def create_ancestors(puzzle, population_size):
    """Create first generation"""
    current_generation = []
    print("generating new ancestors")
    for _ in range(population_size):
        common_ancestor = np.copy(puzzle)
        for i,j in np.ndindex(puzzle.shape):
            if puzzle[i,j] == 0:
                common_ancestor[i,j] = randint(1,9)
        current_generation.append(common_ancestor)
    return current_generation

def plot(ylist, population_size, selection_rate, random_selection_rate,
         individual_mutation_rate, cell_mutation_rate, best_fitness, time=0):
    """Plot fitness over time with labeled parameters"""
    xlist = range(len(ylist))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xlist, ylist)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.ylim(0, 120)
    
    text = f"Current model: \n \
            Population size: {population_size} \n \
            Selection rate: {selection_rate} \n \
            Random selection rate: {random_selection_rate} \n \
            Individual mutation rate: {individual_mutation_rate} \n \
            Cell mutation rate: {cell_mutation_rate} \n \
            Number of children: {1/((selection_rate + random_selection_rate)/ 2):0.2f} \n \
            Best fitness = {best_fitness} \n \
            Elapsed time: {time:0.1f} s"
    
    plt.figtext(0.49, 0.59,
                text,
                horizontalalignment ="left",
                wrap = False, fontsize = 8,
                bbox ={'facecolor':'white', 'alpha':0.3, 'pad':5})
    plt.show()

def sudokuGA(puzzle):
    start_time = time.time()

    puzzle = np.array(puzzle)
    fitness_over_time = []
    population_size = 20000
    selection_rate = 0.20
    random_selection_rate = 0.10
    number_of_children = float("NaN") #redundat variable, determinded by selection_rate and random_selection_rate
    #((selection_rate + random_selection_rate)/ 2) * number_of_children = 1
    max_generations = 300
    individual_mutation_rate = 0.45
    cell_mutation_rate = 0.005
    restart_after_n_generations = 25
   
    current_generation = create_ancestors(puzzle, population_size)   
    
    fixed_indices = []
    for i,j in np.ndindex(puzzle.shape): #create list with fixed indices
            if puzzle[i,j] != 0:
                fixed_indices.append((i,j))
    
    new_selection_rate = int(population_size*selection_rate)
    new_random_selection_rate = int(population_size*random_selection_rate)
    children = population_size - new_selection_rate - new_random_selection_rate

    count = 0
    found_solution = False
    local_minima_loop = 0
    solution = []

    while count < max_generations and found_solution == False:
        next_generation = []

        #2.80 seconds
        fitness_list = [fitness_function(individual, fixed_indices) for individual in current_generation]
        fitness_list_indices = np.argsort(fitness_list, kind='heapsort')
        fitness_over_time.append(fitness_list[fitness_list_indices[0]]) #append for plot
        
        if fitness_over_time[-1] == 0:
            print("solution found")
            found_solution = True
            solution.append(current_generation[fitness_list_indices[0]])

        if fitness_over_time[-1] == min(fitness_over_time):
            best_solution = current_generation[fitness_list_indices[0]]

        for x in range(new_selection_rate): #add most fit to next generation
            next_generation.append(current_generation[fitness_list_indices[x]])

        #add randomly to next generation      
        next_generation += choices(current_generation, k = new_random_selection_rate)

        #0.95 seconds
        next_generation_children = []
        for child in range(children):   #add children to next generation
            next_generation_children.append(create_child(choice(next_generation), choice(next_generation)))
        next_generation += next_generation_children

        # #0.02 seconds #alternate faster mutation algorithm
        # for individual in next_generation:  #mutate next generation
        #     if random() < individual_mutation_rate:
        #         mutated_cell_index = (randint(0,8), randint(0,8))
        #         if mutated_cell_index not in fixed_indices:
        #             individual[mutated_cell_index] = randint(1,9)
        #             mutated_cells_count +=1

        #0.49 seconds
        for individual in next_generation:  #mutate next generation
            if random() < individual_mutation_rate:
                for i,j in np.ndindex((9,9)):
                    if (i,j) not in fixed_indices and random() < cell_mutation_rate:
                        individual[i,j] = randint(1,9)

        if count > 2 and fitness_over_time[-1] == fitness_over_time[-2]:
            local_minima_loop += 1
        else:
            local_minima_loop = 0
        
        if local_minima_loop > restart_after_n_generations:
            print("encountered local minima")
            next_generation = create_ancestors(puzzle, population_size)
            local_minima_loop = 0
        
        current_generation = next_generation
        
        count += 1

        print(f"current generation: {count} \
                \t current best fitness: {fitness_over_time[-1]} \
                \t current median fitness: {fitness_list[fitness_list_indices[population_size//2]]}")

    end_time = time.time()       
    plot(fitness_over_time, population_size, selection_rate, random_selection_rate,
          individual_mutation_rate, cell_mutation_rate, min(fitness_over_time), (end_time-start_time))
    
    return best_solution if found_solution == False else solution

hard = [
            [0, 0, 6, 1, 0, 0, 0, 0, 8], 
            [0, 8, 0, 0, 9, 0, 0, 3, 0], 
            [2, 0, 0, 0, 0, 5, 4, 0, 0], 
            [4, 0, 0, 0, 0, 1, 8, 0, 0], 
            [0, 3, 0, 0, 7, 0, 0, 4, 0], 
            [0, 0, 7, 9, 0, 0, 0, 0, 3], 
            [0, 0, 8, 4, 0, 0, 0, 0, 6], 
            [0, 2, 0, 0, 5, 0, 0, 8, 0], 
            [1, 0, 0, 0, 0, 2, 5, 0, 0]
        ]

hard_solution = [
            [3, 4, 6, 1, 2, 7, 9, 5, 8], 
            [7, 8, 5, 6, 9, 4, 1, 3, 2], 
            [2, 1, 9, 3, 8, 5, 4, 6, 7], 
            [4, 6, 2, 5, 3, 1, 8, 7, 9], 
            [9, 3, 1, 2, 7, 8, 6, 4, 5], 
            [8, 5, 7, 9, 4, 6, 2, 1, 3], 
            [5, 9, 8, 4, 1, 3, 7, 2, 6],
            [6, 2, 4, 7, 5, 9, 3, 8, 1],
            [1, 7, 3, 8, 6, 2, 5, 9, 4]
        ]

easy = [
                [5,3,0,0,7,0,0,0,0],
                [6,0,0,1,9,5,0,0,0],
                [0,9,8,0,0,0,0,6,0],
                [8,0,0,0,6,0,0,0,3],
                [4,0,0,8,0,3,0,0,1],
                [7,0,0,0,2,0,0,0,6],
                [0,6,0,0,0,0,2,8,0],
                [0,0,0,4,1,9,0,0,5],
                [0,0,0,0,8,0,0,7,9]
            ]

solution_easy = [
                [5,3,4,6,7,8,9,1,2],
                [6,7,2,1,9,5,3,4,8],
                [1,9,8,3,4,2,5,6,7],
                [8,5,9,7,6,1,4,2,3],
                [4,2,6,8,5,3,7,9,1],
                [7,1,3,9,2,4,8,5,6],
                [9,6,1,5,3,7,2,8,4],
                [2,8,7,4,1,9,6,3,5],
                [3,4,5,2,8,6,1,7,9]
                ]

found_solution_easy = [
                [5,3,4,6,7,8,9,1,2],
                [6,7,2,1,9,5,3,4,8],
                [1,9,8,0,4,2,5,6,7],
                [8,5,9,7,6,1,4,2,3],
                [4,2,6,8,5,3,7,9,1],
                [7,1,3,9,2,4,8,5,6],
                [9,6,1,5,3,7,2,8,4],
                [2,8,7,4,1,9,6,3,5],
                [3,4,5,2,8,6,1,7,9]
                ]

testmatrix = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]

print(sudokuGA(easy))
