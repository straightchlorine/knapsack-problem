import numpy as np

from knapsack.selectors.selector import Selector


class Population:
    def __init__(self, selector: Selector, population_size=100, gene_length=5):
        """Population object, representing a group of chromosomes.

        Args:
            selector (Selector): Selector object, used to select the parents.
            population_size (int): Size of the population.
            gene_length (int): Length of the gene.
        """
        self.population_size = population_size
        self.gene_length = gene_length

        # set the selector
        self.selector = selector

        # empty list of chromosomes
        self.chromosomes: np.ndarray = np.array([])

    def update_selector(self):
        """Update the selector with a new population of chromosomes."""
        self.selector.population = self.chromosomes

    def __gen_genes(self):
        """Generate a chromosome with random genes.

        Returns:
            np.ndarray: Chromosome with random genes.
        """
        genes = np.random.choice([0, 1], size=self.gene_length)
        return genes

    def initialize(self):
        """Initialize the population with random genes."""
        initial_population = [
            self.__gen_genes() for _ in range(self.population_size)
        ]

        self.chromosomes = np.array(initial_population)
        self.update_selector()

    def select_parents(self) -> np.ndarray:
        """Select parents for the next generation.

        Returns:
            np.ndarray: Array of selected parents.
        """
        return self.selector.select()

    def add_chromosome(self, chromosome: np.ndarray):
        """Add a chromosome or a list of them to the population.

        Args:
            chromosome (np.ndarray): Chromosome to add.

        Returns:
            int: Index of the added chromosome.
        """
        # initialize `self.chromosomes` as an empty array if it hasn't been yet
        if not hasattr(self, "chromosomes") or self.chromosomes.size == 0:
            self.chromosomes = np.empty(
                (0, chromosome.shape[1]), dtype=chromosome.dtype
            )

        self.chromosomes = np.vstack([self.chromosomes, chromosome])
        return len(self.chromosomes) - 1
