import numpy as np


class Chromosome:
    def __init__(self, genes: np.ndarray):
        """Class representing a single chromosome i.e. solution within the
        population.

        Args:
            genes (np.ndarray): A binary vector where each gene represents
            which item is included within a solution.
        """
        self.genes = genes

    def __repr__(self):
        return f"Chromosome(genes={self.genes})"

    def __getitem__(self, index):
        """Allow access to genes using array-like indexing."""
        return self.genes[index]

    def __array__(self):
        """Return the chromosome as a NumPy array."""
        return self.genes

    @property
    def length(self):
        """The length property."""
        return self.genes.size
