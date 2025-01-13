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
        """Initialize the FixedPointCrossover operator.

        Args:
            dev (bool, optional): Whether to display debug information.
                Defaults to False.
            fixed_point (int, optional): Fixed crossover point. If None, a
            random point between 1 and parent_a.size - 1 will be chosen.
        """
        super().__init__(dev)
        self.fixed_point = fixed_point

    def crossover(self, parent_a, parent_b):
        # validate the fixed point parameter
        if self.fixed_point is not None:
            if not (1 <= self.fixed_point < parent_a.size):
                raise ValueError(
                    "fixed_point must be between 1 and parent_a.size - 1"
                )
            point = self.fixed_point
        else:
            point = np.random.randint(1, parent_a.size - 1)

        # create two children
        genes = (
            np.concatenate([parent_a[:point], parent_b[point:]]),
            np.concatenate([parent_a[point:], parent_b[:point]]),
        )

        # debug for presentation
        if self.dev:
            print(f"FixedPointCrossover:\nUsing crossover point {point}")
            print(f"Crossover on {parent_a} and {parent_b}")
            print(f"Generated children: {genes[0]} and {genes[1]}")
            print(f"Child 0 parts: {parent_a[:point]} and {parent_b[point:]}")
            print(f"Child 1 parts: {parent_a[point:]} and {parent_b[:point]}")
            print(20 * "=")

        return np.array([genes[0], genes[1]])
