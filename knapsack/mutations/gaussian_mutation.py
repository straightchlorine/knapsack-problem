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
        noise = np.random.normal(loc=0, scale=1, size=mutated_population.shape)
        probabilities = self.probability * (1 + noise)
        probabilities = np.clip(probabilities, 0, 1)  # Ensure probabilities are valid
        for i in range(mutated_population.shape[0]):
            for j in range(mutated_population.shape[1]):
                if np.random.rand() < probabilities[i, j]:
                    mutated_population[i, j] = 1 - mutated_population[i, j]
        return mutated_population
