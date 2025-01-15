import numpy as np

from knapsack.operators.crossover import Crossover

"""Simples class providing crossover operation for chromosomes.

This class picks a random point in the chromosome and swaps the genes
between two parents to create two children.

Test cases:
    ====================
    Crossover on [1 0 0 1 1] and [1 0 0 1 1]
    Picked random point for crossover: 3
    Generated children: [1 0 0 1 1] and [1 1 1 0 0]
    Child 0 parts: [1 0 0] and [1 1]
    Child 1 parts: [1 1] and [1 0 0]
    ====================
    Crossover on [1 0 0 1 1] and [1 0 0 1 1]
    Picked random point for crossover: 3
    Generated children: [1 0 0 1 1] and [1 1 1 0 0]
    Child 0 parts: [1 0 0] and [1 1]
    Child 1 parts: [1 1] and [1 0 0]
    ====================
    Crossover on [1 0 0 1 1] and [1 0 0 1 1]
    Picked random point for crossover: 1
    Generated children: [1 0 0 1 1] and [0 0 1 1 1]
    Child 0 parts: [1] and [0 0 1 1]
    Child 1 parts: [0 0 1 1] and [1]
    ====================
"""


class FixedPointCrossover(Crossover):
    def __init__(self, dev=False, fixed_point=None):
        super().__init__(dev)
        self.fixed_point = fixed_point

    def crossover(self, parent_a, parent_b):
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        if self.fixed_point is not None:
            if not (1 <= self.fixed_point < parent_a.size):
                raise ValueError("Fixed_point must be between 1 and parent_a.size - 1.")
            point = self.fixed_point
        else:
            point = np.random.randint(1, parent_a.size)

        child_1 = np.concatenate([parent_a[:point], parent_b[point:]])
        child_2 = np.concatenate([parent_b[:point], parent_a[point:]])

        if self.dev:
            print(f"FixedPointCrossover:\nUsing crossover point {point}")
            print(f"Crossover on {parent_a} and {parent_b}")
            print(f"Generated children: {child_1} and {child_2}")
            print("=" * 20)

        return np.array([child_1, child_2])
