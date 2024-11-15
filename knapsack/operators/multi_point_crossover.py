import numpy as np
from knapsack.operators.crossover import Crossover

"""Class providing crossover operation for chromosomes, with multiple points.

This class randomly picks given number of points in the chromosome and swaps
the genes between two parents to create two children.

Test cases:
    ====================
    Crossover on [0 0 0 1 0] and [0 0 1 0 1]
    Picked random points for crossover: [2 3]
    Generated children: [0 0 1 1 0] and [0 0 0 0 1]
    ====================
    Crossover on [0 0 0 1 0] and [0 0 1 0 1]
    Picked random points for crossover: [1 4]
    Generated children: [0 0 1 0 0] and [0 0 0 1 1]
    ====================
    Crossover on [0 0 0 1 0] and [0 0 1 0 1]
    Picked random points for crossover: [1 2]
    Generated children: [0 0 0 1 0] and [0 0 1 0 1]
    ====================

Because of the length of the dataset (only 5 elements), the effects of the
crossover often are identical to each other or to the parents. This is due
to the amount of points (2 in those cases) and length of the segments.
"""


class MultiPointCrossover(Crossover):
    def __init__(self, points: int, dev=False):
        super().__init__(dev)
        self.points = points

    def crossover(self, parent_a, parent_b):
        # Select random crossover points and sort them
        points = np.sort(
            np.random.choice(
                range(1, parent_a.size), size=self.points, replace=False
            )
        )

        # start with both children with parents genes
        genes = [parent_a.copy(), parent_b.copy()]

        # alternate between parents and swap segments between the points
        for i in range(len(points)):
            if i % 2 == 0:
                genes[0][
                    points[i] : points[i + 1] if i + 1 < len(points) else None
                ] = parent_b[
                    points[i] : points[i + 1] if i + 1 < len(points) else None
                ]
                genes[1][
                    points[i] : points[i + 1] if i + 1 < len(points) else None
                ] = parent_a[
                    points[i] : points[i + 1] if i + 1 < len(points) else None
                ]

        if self.dev:
            print(f"Crossover on {parent_a} and {parent_b}")
            print(f"Picked random points for crossover: {points}")
            print(f"Generated children: {genes[0]} and {genes[1]}")
            print("=" * 20)

        return np.array(genes)
