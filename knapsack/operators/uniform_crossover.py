import numpy as np

from knapsack.operators.crossover import Crossover


"""Class providing uniform crossover operation for chromosomes.

This operator randomly selects genes from two parents to create two children.

Test cases:
    ====================
    Crossover on [1 1 0 1 0] and [0 0 1 1 0]
    Generated mask: [ True  True False  True  True]
    Generated children: [1 1 1 1 0] and [0 0 0 1 0]
    ====================
    Crossover on [1 1 0 1 0] and [0 0 1 1 0]
    Generated mask: [False False False False  True]
    Generated children: [0 0 1 1 0] and [1 1 0 1 0]
    ====================
    Crossover on [1 1 0 1 0] and [0 0 1 1 0]
    Generated mask: [ True False False False  True]
    Generated children: [1 0 1 1 0] and [0 1 0 1 0]
    ====================
"""


class UniformCrossover(Crossover):
    def __init__(self, dev=False):
        super().__init__(dev)

    def crossover(self, parent_a, parent_b):
        # random binary mask
        # its task is to determine which parent's gene to take
        mask = np.random.randint(0, 2, size=parent_a.size).astype(bool)

        # if the mask is 1, the gene is taken from the first parent_a for the
        # first child
        genes = [
            np.where(mask, parent_a, parent_b),
            np.where(mask, parent_b, parent_a),
        ]

        # debug for presentation
        if self.dev:
            print(f"Crossover on {parent_a} and {parent_b}")
            print(f"Generated mask: {mask}")
            print(f"Generated children: {genes[0]} and {genes[1]}")
            print("=" * 20)

        return np.array(genes)
