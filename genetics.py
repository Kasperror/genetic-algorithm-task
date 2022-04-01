import itertools
from typing import Callable, TypeVar
import numpy as np
from dataclasses import dataclass, field

TChromosome = TypeVar('TChromosome', bound='Chromosome')
TPopulation = TypeVar('TPopulation', bound='Population')
TFitnessFunc = Callable[[TChromosome], float]

DEFAULT_CHROM_SIZE = 10


def flip_bit(bit: int):
    assert bit in [0, 1]
    return int(not bit)


@dataclass
class Chromosome:

    genes: list[int]
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.genes)
        assert self.size > 1

    @classmethod
    def from_crossover(cls, p1: TChromosome, p2: TChromosome) -> TChromosome:
        """Creates a new Chromosome from crossover of p1 and p2 chromosomes"""
        assert p1.size == p2.size
        crossover_point = np.random.choice(range(1, p1.size))
        offspring = p1.crossover(p2, crossover_point)
        return offspring

    def crossover(self, other: TChromosome, at_point: int) -> TChromosome:
        """Implementation of crossover with other Chromosome at provided point (index)"""
        assert self.size > 1
        assert self.size == other.size
        assert at_point < self.size
        new_genes = self.genes[:at_point] + other.genes[at_point:]
        return Chromosome(genes=new_genes)


    def mutate(self, prob: float) -> None:
        """Applies mutation on each bit according to probability"""
        mutate_flags = np.random.choice([True, False], p=[prob, 1-prob], size=self.size)
        new_genes = [flip_bit(gene) if should_flip else gene for gene, should_flip in zip(self.genes, mutate_flags)]
        self.genes = new_genes

    def fitness(self, fitness_func: TFitnessFunc) -> float:
        """Returns the value Chromosome's fitness according to provided function"""
        return fitness_func(self)



def get_chroms_sorted_by_fitness(members: list[Chromosome], fitness_func: TFitnessFunc):
    return sorted(members, key=lambda member: member.fitness(fitness_func), reverse=True)


@dataclass
class Population:

    members: list[Chromosome]
    # fitness function that should take a binary array and return float value
    fitness_func: TFitnessFunc
    crossover_prob: float
    mutation_prob: float
    size: int = field(init=False)

    def __post_init__(self):
        assert 0 <= self.mutation_prob <= 1
        assert 0 <= self.crossover_prob <= 1

        self.size = len(self.members)
        assert self.size > 1

        self._sort_members_by_fitness()

    def add_members(self, new_members: [TChromosome]) -> [TChromosome]:
        self.size += len(new_members)
        self.members.extend(new_members)
        return self.members

    def _get_crossover_prob(self) -> list[float]:
        """Returns a list of probabilities of crossover for each member in order"""
        bias = 0.01
        fitness_vals = self._get_members_fitness()
        min_fitness = min(fitness_vals)

        # scaling the fitness values to normalize probability
        positive_only_fitness_vals = [val + abs(min_fitness) + bias for val in fitness_vals]
        fitness_vals_sum = sum(positive_only_fitness_vals)

        prob = [val/fitness_vals_sum for val in positive_only_fitness_vals]
        return prob


    # def _get_crossover_prob2(self) -> list[float]:
    #     fit_vals = self._get_members_fitness()
    #     min_fit = min(fit_vals)
    #     max_fit = min(fit_vals)
    #
    #     # shifting all values so that they minimally be 0
    #     positive_only_fitness_vals = [val + min_fit for val in fit_vals]
    #     fitness_vals_sum = sum(positive_only_fitness_vals)
    #
    #     probs = []


    def _get_parents_by_roulette_wheel(self) -> list | np.ndarray:
        """Returns a pair of population members selected by roulette wheel selection"""
        return np.random.choice(self.members, size=2, p=self._get_crossover_prob())

    def _create_single_offspring(self):
        """Returns a single offspring of the current population"""
        p1, p2 = self._get_parents_by_roulette_wheel()
        offspring = Chromosome.from_crossover(p1, p2)
        return offspring

    def _mutate_members(self) -> None:
        """Applies mutation on all members of the population"""
        for member in self.members:
            member.mutate(prob=self.mutation_prob)

    def _sort_members_by_fitness(self):
        """Sorts population members by their fitness"""
        self.members = get_chroms_sorted_by_fitness(self.members, self.fitness_func)

    def _get_members_fitness(self) -> list[float]:
        """Returns a list of fitness values for each of members in order"""
        return [member.fitness(self.fitness_func) for member in self.members]

    def _get_fitness_sorted_members(self):
        return get_chroms_sorted_by_fitness(self.members, self.fitness_func)

    def _get_youngest_fittest_members(self, no):
        """Returns no - number of fittest and youngest members of the population."""
        assert no <= self.size
        # crossover offspring is fitness-sorted before passing to next generation
        # non-crossover offspring are always placed at the end of the list
        # also in the fitness order
        # thus this function becomes really simple
        return self.members[:no]

    def mean_fitness(self):
        return np.mean(self._get_members_fitness())

    def max_fitness(self):
        return max(self._get_members_fitness())

    def min_fitness(self):
        return min(self._get_members_fitness())

    def get_fittest(self, no=1) -> TChromosome | list[TChromosome]:
        fitness_sorted_members = self._get_fitness_sorted_members()
        if no == 1:
            return fitness_sorted_members[0]
        return fitness_sorted_members[:no]

    def sort_members_by_fitness(self):
        return self._sort_members_by_fitness()

    def get_nex_gen(self) -> TPopulation:
        """Returns next generation derived from current population"""
        def do_crossover():
            return np.random.choice(
                [True, False], p=[self.crossover_prob, 1 - self.crossover_prob]
            )

        new_members = [self._create_single_offspring() for i in range(self.size) if do_crossover()]

        new_gen = Population(
            members=new_members,
            fitness_func=self.fitness_func,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob
        )

        # if there are still slots left for members
        # in new generation due to no crossover
        # place youngest and fittest members at the bottom
        # of the members list (important to keep FIFO strategy)
        slots_remaining = self.size - new_gen.size
        non_crossover_new_members = self._get_youngest_fittest_members(no=slots_remaining)
        new_gen.add_members(non_crossover_new_members)

        assert new_gen.size == self.size

        # applying mutation according to probability
        new_gen._mutate_members()

        return new_gen


class GeneticAlgorithm:

    population: Population
    generation_no: int = 0

    def __init__(self, initial_population=None):
        self.population = initial_population


    @classmethod
    def generate_random_population(cls, size: int, chrom_size: int, fitness_func, crossover_prob: float, mutation_prob: float) -> Population:
        """Generates population with random members according to provided parameters"""
        bin_vectors = [list(np.random.choice([0, 1], size=chrom_size)) for i in range(size)]
        members_ = [Chromosome(vec) for vec in bin_vectors]
        return Population(
            members=members_,
            fitness_func=fitness_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob
        )

    def auto_setup(self,
                   fitness_func,
                   population_size=50,
                   chrom_size=DEFAULT_CHROM_SIZE,
                   crossover_prob=0.75,
                   mutation_prob=0.05) -> None:
        """Generates random population"""
        self.population = GeneticAlgorithm.generate_random_population(
            size=population_size,
            chrom_size=chrom_size,
            fitness_func=fitness_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob
        )

    def evolve(self):
        """Brings population to next generation"""
        self.population = self.population.get_nex_gen()
        self.generation_no += 1

    def get_fittest(self):
        """Returns the fittest member of current population"""
        return self.population.get_fittest()

    def get_mean_fitness(self):
        """Returns the mean fitness value of current population"""
        return self.population.mean_fitness()

    def get_max_fitness(self):
        """Returns the max fitness value of current population"""
        return self.population.max_fitness()

    def get_min_fitness(self):
        """Returns the min fitness value of current population"""
        return self.population.min_fitness()


    def run(self, iter_limit=10000):
        """Runs the algorithm on the population for iter_limit times"""
        assert self.population is not None
        for i in itertools.count():
            # stopping condition
            if i >= iter_limit:
                break
            self.evolve()
            print(f'CURRENT GEN: {self.generation_no}, ITERATION: {i}/{iter_limit} MAX FITNESS: {self.get_max_fitness()}', end='\r')





if __name__ == "__main__":

    GENES_NO = 5

    def gene_sum(genes):
        return sum(genes)

    # ch = Chromosome([1]*GENES_NO)
    # assert ch.fitness(fitness_func=gene_sum) == GENES_NO
    #
    # ch.mutate(1)
    # assert ch.fitness(fitness_func=gene_sum) == 0
    # ch.mutate(1)  # revert to start state
    #
    # ch2 = Chromosome([0]*GENES_NO)
    #
    # assert ch.crossover(ch2, GENES_NO-1).genes == [1]*(GENES_NO - 1) + [0]
    #
    # members = [Chromosome([1, 1, 1]) for i in range(10)]
    #
    # pop = Population(members=members, crossover_prob=1, mutation_prob=0, fitness_func=gene_sum)
    #
    # for i in range(100):
    #     new_pop = pop.get_nex_gen()
    #     assert new_pop == pop
    #     assert new_pop.get_fittest() == Chromosome([1, 1, 1])
    #     assert new_pop.min_fitness() == new_pop.max_fitness() == new_pop.mean_fitness() == 3
    #
    #
    # pop = Population(members=members, crossover_prob=1, mutation_prob=1, fitness_func=gene_sum)
    #
    # for i in range(100):
    #     new_pop = pop.get_nex_gen()
    #     assert new_pop != pop
    #     assert new_pop.get_fittest() == Chromosome([0, 0, 0])
    #     assert new_pop.min_fitness() == new_pop.max_fitness() == new_pop.mean_fitness() == 0


    rand_pop = GeneticAlgorithm.generate_random_population(size=2, chrom_size=3, mutation_prob=1, crossover_prob=0, fitness_func=gene_sum)
    for i in rand_pop.members:
        print(i, i.fitness(fitness_func=gene_sum))

    print(rand_pop._get_crossover_prob())

    # print()
    # print()
    # nex_gen = rand_pop.get_nex_gen()
    # for i in rand_pop.members:
    #     print(i, i.fitness(fitness_func=gene_sum))
    #
    # print(nex_gen.get_fittest())






























