import numpy as np

from knapsack.selectors.selector import Selector


class RandomSelector(Selector):
    def __init__(self):
        """Basic random selector."""
        super().__init__()

    def select(self):
        """Selects two random chromosomes from the population."""
        return self.population[
            np.random.choice(self.population.shape[0], 2, replace=False)
        ]
