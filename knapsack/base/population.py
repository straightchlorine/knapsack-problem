import numpy as np

from knapsack.base.chromosome import Chromosome
from knapsack.selectors.selector import Selector


class Population:
    def __init__(self, selector: Selector, population_size=100, gene_length=5):
        """Population object, representing a group of chromosomes.

        Args:
            selector (Selector): Selector object, used to select the parents.
            population_size (int): Size of the population.
            gene_length (int): Length of the
        """
        self.population_size = population_size
        self.gene_length = gene_length

        self.selector = selector

        # initialize population
        self._init_population()

    def _gen_genes(self):
        """Generate a chromosome object with random genes."""
        genes = np.random.choice([0, 1], size=self.gene_length)
        return Chromosome(genes)

    def _init_population(self):
        """Initialize the population with chromosomes."""
        self.chromosomes = [
            self._gen_genes() for _ in range(self.population_size)
        ]

    def select_parents(self):
        """Select parents for the next generation."""
        return self.selector.select()

    def add_chromosome(self, chromosome):
        """Add a chromosome to the population."""
        self.chromosomes.append(chromosome)
