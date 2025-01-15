import numpy as np
from knapsack.operators.crossover import Crossover


class BlendCrossover(Crossover):
    def __init__(self, alpha=0.5, dev=False):
        """
        Args:
            alpha (float): Blend factor. Determines the range of exploration.
            dev (bool, optional): Debug mode. Defaults to False.
        """
        super().__init__(dev)
        self.alpha = alpha

    def crossover(self, parent_a, parent_b):
        diff = np.abs(parent_a - parent_b)
        lower_bound = np.minimum(parent_a, parent_b) - self.alpha * diff
        upper_bound = np.maximum(parent_a, parent_b) + self.alpha * diff

        child_1 = np.random.uniform(lower_bound, upper_bound)
        child_2 = np.random.uniform(lower_bound, upper_bound)

        if self.dev:
            print(f"Blend Crossover on {parent_a} and {parent_b}")
            print(f"Alpha: {self.alpha}")
            print(f"Generated children: {child_1} and {child_2}")
            print("=" * 20)

        return np.array([child_1, child_2])
