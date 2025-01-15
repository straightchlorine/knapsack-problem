import logging

import numpy as np

from knapsack.operators.crossover import Crossover
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
    def __init__(self, points=None, dev=False):
        self.points = points
        self.dev = dev
        self._last_crossover_points = None

    def _validate_points(self, parent):
        if parent.size < 2:
            raise ValueError("Parent size must be at least 2.")
        if self._last_crossover_points is not None:
            logging.info("Using cached crossover points")
            return self._last_crossover_points

        if self.points is not None:
            if not all(1 <= p < parent.size for p in self.points):
                raise ValueError("All points must be between 1 and parent.size - 1.")
            points = np.sort(self.points)
        else:
            num_points = np.random.randint(1, min(5, parent.size - 1))
            points = np.sort(
                np.random.choice(range(1, parent.size), size=num_points, replace=False)
            )
        return points

    def crossover(self, parent_a, parent_b):
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        points = self._validate_points(parent_a)
        logging.info(f"MultiPointCrossover: Using crossover points {points}")

        children = [parent_a.copy(), parent_b.copy()]
        for i, start in enumerate(points):
            end = points[i + 1] if i + 1 < len(points) else None
            children[i % 2][start:end], children[(i + 1) % 2][start:end] = (
                children[(i + 1) % 2][start:end],
                children[i % 2][start:end],
            )

        self._last_crossover_points = points
        if self.dev:
            logging.info(f"Crossover on {parent_a} and {parent_b}")
            logging.info(f"Generated children: {children[0]} and {children[1]}")

        return np.array(children)
