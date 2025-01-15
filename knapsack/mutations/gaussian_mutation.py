import numpy as np
from knapsack.mutations.mutation import Mutation


class GaussianMutation(Mutation):
    """Applies Gaussian-like mutation to a binary population."""

    def mutate(self, population):
        """
        Flip bits in chromosomes with a probability influenced by Gaussian noise.

        Returns:
            np.ndarray: The mutated population.
        """
        mutated_population = population.copy()
        noise = np.random.normal(loc=0, scale=0.1, size=mutated_population.shape)
        probabilities = np.clip(self.probability + noise, 0, 1)
        mutation_mask = np.random.rand(*mutated_population.shape) < probabilities
        mutated_population[mutation_mask] ^= 1
        return mutated_population
