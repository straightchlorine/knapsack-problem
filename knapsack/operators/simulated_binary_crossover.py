from knapsack.operators.crossover import Crossover
import numpy as np


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, eta=2, dev=False):
        super().__init__(dev)
        self.eta = eta

    def crossover(self, parent_a, parent_b):
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        rand = np.random.rand(parent_a.size)
        beta = np.empty_like(rand)
        beta[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1 / (self.eta + 1))
        beta[rand > 0.5] = (1 / (2 * (1 - rand[rand > 0.5]))) ** (1 / (self.eta + 1))

        child_1 = 0.5 * ((1 + beta) * parent_a + (1 - beta) * parent_b)
        child_2 = 0.5 * ((1 - beta) * parent_a + (1 + beta) * parent_b)

        if self.dev:
            print(f"SBX Crossover on {parent_a} and {parent_b}")
            print(f"Generated beta: {beta}")
            print(f"Generated children: {child_1} and {child_2}")
            print("=" * 20)

        return np.array([child_1, child_2])
