from typing import Callable, TypeVar
import numpy as np
from dataclasses import dataclass, field

TChromosome = TypeVar('TChromosome', bound='Chromosome')
TPopulation = TypeVar('TPopulation', bound='Population')


def flip_bit(bit: int):
    assert bit in [0, 1]
    return int(not bit)


@dataclass
class Chromosome:

    genes: list[int]
    size: int

    def __post_init__(self):
        self.size = len(self.genes)

    @classmethod
    def from_crossover(cls, p1: TChromosome, p2: TChromosome) -> TChromosome:
        """Creates a new Chromosome from crossover of p1 and p2 chromosomes"""
        assert len(p1) == len(p2)

        crossover_point = np.random.choice(range(1, len(p1)))
        child_genes = p1.genes[:crossover_point] + p2.genes[crossover_point:]
        child = Chromosome(genes=child_genes)
        return child

    def mutate(self, prob: float):
        """Applies mutation on each bit according to probability"""
        mutate_flags = np.random.choice([True, False], p=[prob, 1-prob])
        new_genes = [flip_bit(gene) if should_flip else gene for gene, should_flip in zip(self.genes, mutate_flags)]
        self.genes = new_genes

    def fitness(self, fitness_func: Callable) -> float:
        return fitness_func(self.genes)


@dataclass
class Population:

    members: list[Chromosome]
    # fitness function that should take
    fitness_func: Callable
    crossover_prob: float
    mutation_prob: float
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.members)
        assert self.size > 1
    #
    # def _get_member_fitness(self, member):
    #     return self.fitness_func(member)

    def _get_members_fitness(self) -> list[float]:
        """Returns a list of fitness values for each of members in order"""
        return [member.fitness(self.fitness_func) for member in self.members]
    
    def _get_crossover_prob(self) -> list[float]:
        """Returns a list of probabilities of crossover for each member in order"""
        fitness_vals = self._get_members_fitness()
        # probability is the member's normalized fitness value in population
        normal_form = np.linalg.norm(fitness_vals)
        return fitness_vals / normal_form

    def _get_parents_by_roulette_wheel(self) -> list | np.ndarray:
        """Returns a pair of population members selected by roulette wheel selection"""
        return np.random.choice(self.members, size=2, p=self._get_crossover_prob())

    def _create_single_offspring(self):
        """Returns a single offspring of the current population"""
        p1, p2 = self._get_parents_by_roulette_wheel()
        offspring = Chromosome.from_crossover(p1, p2)
        return offspring

    def mean_fitness(self):
        return np.mean(self._get_members_fitness())

    def max_fitness(self):
        return max(self._get_members_fitness())

    def min_fitness(self):
        return min(self._get_members_fitness())

    def _get_fitness_sorted_members(self):
        return sorted(self.members, key=lambda member: member.fitness)

    def get_fittest(self, no = 1) -> TChromosome | list[TChromosome]:
        fitness_sorted_members = self._get_fitness_sorted_members()
        if no == 1:
            return fitness_sorted_members[0]
        return fitness_sorted_members[:no]

    def _get_youngest_fittest_members(self, no):
        """Returns no - number of fittest and youngest members of the population."""
        assert no <= self.size
        # crossover offspring is fitness-sorted before passing to next generation
        # non-crossover offspring are always placed at the end of the list
        # also in the fitness order
        # thus this function becomes really simple
        return self.members[:no]

    def get_nex_gen(self) -> TPopulation:
        """Returns next generation derived from current population"""
        def do_crossover():
            return np.random.choice(
                [True, False], p=[self.crossover_prob, 1 - self.crossover_prob]
            )

        new_members = []
        for i in range(self.size):
            if do_crossover():
                offspring = self._create_single_offspring()
                new_members.append(offspring)

        # sorting with fitness value
        new_members.sort(key=lambda member: member.fitness)

        # if there are still slots left for members
        # in new generation due to no crossover
        # place youngest and fittest members at the bottom
        # of the members list (important to keep FIFO strategy)
        slots_remaining = self.size - len(new_members)
        non_crossover_new_members = self._get_youngest_fittest_members(no=slots_remaining)
        new_members.append(non_crossover_new_members)

        # applying mutation according to probability
        for member in new_members:
            member.mutate(self.mutation_prob)

        next_gen = Population(
            members=new_members,
            fitness_func=self.fitness_func,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob
        )
        return next_gen
