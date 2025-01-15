import numpy as np
from knapsack.mutations.mutation import Mutation


class BitFlipMutation(Mutation):
    """Applies bit-flip mutation to a binary population."""

    def mutate(self, population) -> np.ndarray:
        """Flip bits in chromosomes with the given mutation probability.

        Returns:
            np.ndarray: The mutated population.
        """
        mutated_population = population.copy()
        for i in range(mutated_population.shape[0]):
            for j in range(mutated_population.shape[1]):
                if np.random.rand() < self.probability:
                    mutated_population[i, j] = 1 - mutated_population[i, j]
        return mutated_population
