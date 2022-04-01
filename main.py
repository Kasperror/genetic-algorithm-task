import json
from functools import partial
from default import DEFAULT_PARAMS
from genetics import GeneticAlgorithm
from connector import chrom_fitness_func, chrom_to_x, calculate_chrom_size
from cli import collect_params

TEST = False

params = DEFAULT_PARAMS
if not TEST:
    params = collect_params()
    print("Parameter 'c' is collected due to requirements but is omitted further on "
          "as it has no impact on result and could slow down population learning")

A = params['A']
b = params['b']
c = params['c']
dec_width = params['dec_width']  # width of 'decimal' part in binary vector
dim = params['dim']  # dimenstionality of the problem
crossover_prob = params['cross_p']
mutation_prob = params['mutation_p']

population_size = params['pop']
iters_limit = params['iters_no']

fp_width = 7  # width of 'floating' part in binary vector, default 7 to fit .99

fitness_func = partial(chrom_fitness_func, A=A, b=b, dec_width=dec_width)


ga = GeneticAlgorithm()
ga.auto_setup(
    fitness_func=fitness_func,
    crossover_prob=crossover_prob,
    mutation_prob=mutation_prob,
    population_size=population_size,
    chrom_size=calculate_chrom_size(dec_width, fp_width, dim)
)

ga.run(iter_limit=iters_limit)

print()
fittest = ga.get_fittest()

print(f"{fittest=}")

x_val = chrom_to_x(fittest, dim, dec_width)
print(f"Optimal value for X is {x_val}")

max_fitness = ga.get_max_fitness()
print(f'Maximal value of function is {max_fitness + c}')

mean_fitness = ga.get_mean_fitness()
print(f'Mean fitness of latest generation is {mean_fitness + c}')

latest_gen = ga.population
latest_gen.sort_members_by_fitness()

print()
show_gen = input("Do you want to see latest generation? (y/n) ")
if show_gen == 'y':
    for m in latest_gen.members:
        print(m, 'Target value:', m.fitness(ga.population.fitness_func) + c)


# print(f"{fittest=}")
#
# x_val = chrom_to_x(fittest, dim, dec_width)
# print(f"{x_val=}")
#
# max_fitness = ga.get_max_fitness()
# print(f'{max_fitness=}')
#
# min_fitness = ga.get_min_fitness()
# print(f'{min_fitness=}')
#
# mean_fitness = ga.get_mean_fitness()
# print(f'{mean_fitness=}')
#



