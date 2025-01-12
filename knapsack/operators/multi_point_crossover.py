import numpy as np
import logging

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
        """
        Initialize the MultiPointCrossover operator.

        Args:
            points (list[int], optional): List of crossover points.
                If None, a random number of points between 1 and 5 will be
                chosen. Defaults to None.
            dev (bool, optional): Whether to display debug information.
                Defaults to False.
        """
        self.points = points
        self.dev = dev
        self._last_crossover_points = None

    def _validate_points(self, parent):
        """
        Validate and return the crossover points.

        Args:
            parent (np.ndarray): The parent array to validate against.

        Returns:
            np.ndarray: Sorted array of valid crossover points.
        """
        if self._last_crossover_points is not None:
            return self._last_crossover_points

        if self.points is not None:
            if not all(1 <= p < parent.size for p in self.points):
                raise ValueError(
                    "All points must be between 1 and parent_a.size - 1"
                )
            points = np.sort(self.points)
        else:
            num_points = np.random.randint(1, min(5, parent.size - 1))
            points = np.sort(
                np.random.choice(
                    range(1, parent.size), size=num_points, replace=False
                )
            )
        return points

    def crossover(self, parent_a, parent_b):
        # validate points
        points = self._validate_points(parent_a)
        if self.dev:
            logging.debug(
                f"MultiPointCrossover: Using crossover points {points}"
            )

        # start with both children having parents' genes
        genes = [parent_a.copy(), parent_b.copy()]

        # alternate between parents' genes at the crossover
        for i, start in enumerate(points):
            end = points[i + 1] if i + 1 < len(points) else None
            if i % 2 == 0:
                genes[0][start:end], genes[1][start:end] = (
                    genes[1][start:end],
                    genes[0][start:end],
                )

        # cache the resulting points
        self._last_crossover_points = points

        if self.dev:
            logging.debug(f"Crossover on {parent_a} and {parent_b}")
            logging.debug(f"Generated children: {genes[0]} and {genes[1]}")

        return np.array(genes)
