import numpy as np

from knapsack.mutations.mutation import Mutation


class DynamicMutation(Mutation):
    """Applies dynamic mutation based on the generation number."""

    def __init__(self, probability, max_generations, min_probability=0.01):
        self.probability = probability
        self.max_generations = max_generations
        self.min_probability = min_probability

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
        dynamic_probability = max(
            self.probability * (1 - generation / self.max_generations),
            self.min_probability,
        )
        mutation_mask = np.random.rand(*mutated_population.shape) < dynamic_probability
        mutated_population[mutation_mask] ^= 1
        return mutated_population
