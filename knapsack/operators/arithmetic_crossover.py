import numpy as np
from knapsack.operators.crossover import Crossover


class ArithmeticCrossover(Crossover):
    def __init__(self, alpha=0.5, dev=False):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        super().__init__(dev)
        self.alpha = alpha

    def crossover(self, parent_a, parent_b):
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        child_1 = self.alpha * parent_a + (1 - self.alpha) * parent_b
        child_2 = (1 - self.alpha) * parent_a + self.alpha * parent_b

        if self.dev:
            print(f"Arithmetic Crossover on {parent_a} and {parent_b}")
            print(f"Alpha: {self.alpha}")
            print(f"Generated children: {child_1} and {child_2}")
            print("=" * 20)

        return np.array([child_1, child_2])
