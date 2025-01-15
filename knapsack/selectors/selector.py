import numpy as np
from knapsack.evaluators.evaluator import Evaluator


class Selector:
    """Base class for the selector objects."""

    _population: np.ndarray
    evaluator: Evaluator

    def __init__(self):
        self._population = None

    def __str__(self):
        return f"Selector: {self.__class__.__name__}"

    def select(self):
        """Select parents for the next generation.

        Returns:
            list: List of selected chromosomes
        """
        raise NotImplementedError('Method "select" must be implemented in a subclass.')

    @property
    def population(self):
        if self._population is None:
            raise ValueError("Population is not set.")
        return self._population

    @population.setter
    def population(self, population: np.ndarray):
        if not isinstance(population, np.ndarray) or len(population) == 0:
            raise ValueError("Population must be a non-empty numpy array.")
        self._population = population

    def validate_population_size(self, required_size: int):
        """Validate that the population size meets the required minimum."""
        if len(self.population) < required_size:
            raise ValueError(
                f"Population size ({len(self.population)}) must be at least {required_size}."
            )

    def random_sample(self, size: int, replace: bool = False):
        """Select a random sample of individuals from the population."""
        return self.population[
            np.random.choice(len(self.population), size, replace=replace)
        ]
