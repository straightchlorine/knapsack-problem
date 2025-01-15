import numpy as np
from knapsack.mutations.mutation import Mutation


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
        mutation_mask = np.random.rand(*mutated_population.shape) < self.probability
        mutated_population[mutation_mask] ^= 1
        return mutated_population
