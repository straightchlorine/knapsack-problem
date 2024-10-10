class Chromosome:
    def __init__(self, genes):
        """Class representing a single chromosome i.e. solution within the
        population.

        Args:
            genes (np.ndarray): A binary vector where each gene represents
            which item is included within a solution.
        """
        self.genes = genes

    def __repr__(self):
        return f"Chromosome(genes={self.genes})"
