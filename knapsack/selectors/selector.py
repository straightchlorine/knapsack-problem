import numpy as np


class Selector:
    """Base class for the selector objects."""

    _population: np.ndarray

    def __init__(self):
        print(f"Selecting parents parents via {self.__class__.__name__}")

    def select(self):
        """Select parents for the next generation.

        Returns:
            list: List of selected chromosomes
        """
        raise NotImplementedError(
            'Method "select" must be implemented in a subclass.'
        )

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population: np.ndarray):
        self._population = population
