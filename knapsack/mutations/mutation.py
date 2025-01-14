import numpy as np


class Mutation:
    """Base class for mutation operators."""

    _probability: float  # Probability of mutation for each gene

    def __init__(self, probability: float):
        """Initialize the mutation operator.

        Args:
            probability (float): Probability of mutation for each gene.
        """
        if not (0 <= probability <= 1):
            raise ValueError("Mutation probability must be between 0 and 1.")
        self._probability = probability

    def __str__(self):
        return f"Mutation: {self.__class__.__name__}"

    def mutate(self, population):
        """Apply mutation to the population.

        This method must be implemented by subclasses to define
        specific mutation logic.

        Returns:
            np.ndarray: The mutated population.
        """
        raise NotImplementedError('Method "mutate" must be implemented in a subclass.')

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability: float):
        if not (0 <= probability <= 1):
            raise ValueError("Mutation probability must be between 0 and 1.")
        self._probability = probability
