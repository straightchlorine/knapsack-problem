import numpy as np
from knapsack.mutations.mutation import Mutation


class DynamicMutation(Mutation):
    """Applies dynamic mutation based on the generation number."""

    def __init__(self, probability, max_generations):
        super().__init__(probability)
        self.max_generations = max_generations

    def mutate(self, population, generation=0):
        """
        Flip bits in chromosomes with a probability that changes dynamically.

        Args:
            population (np.ndarray): The population to mutate.
            generation (int): The current generation number.

        Returns:
            np.ndarray: The mutated population.
        """
        mutated_population = population.copy()
        dynamic_probability = self.probability * (1 - generation / self.max_generations)
        for i in range(mutated_population.shape[0]):
            for j in range(mutated_population.shape[1]):
                if np.random.rand() < dynamic_probability:
                    mutated_population[i, j] = 1 - mutated_population[i, j]
        return mutated_population
