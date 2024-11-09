import numpy as np
from knapsack.base.population import Population
from knapsack.selectors.selector import Selector


class RandomSelector(Selector):
    def __init__(self, population: Population):
        """Basic random selector."""
        super().__init__(population)

    def select(self):
        return np.random.choice(self.population.chromosomes, 2)
