import numpy as np
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class AdaptiveRouletteSelector(Selector):
    def __init__(self, evaluator: Evaluator, adapt_factor: float = 0.1):
        """Adaptive roulette selection mechanism.

        Args:
            evaluator (Evaluator): Evaluator object.
            adapt_factor (float): Adjustment factor for fitness probabilities.
        """
        super().__init__()
        self.evaluator = evaluator
        self.adapt_factor = adapt_factor

    def select(self, number_of_parents=2):
        """Select parents using adaptive roulette selection.

        Args:
            number_of_parents (int): Number of parents to select.

        Returns:
            list: List of selected chromosomes.
        """
        fitness_scores = np.array([self.evaluator.evaluate(c) for c in self.population])
        mean_fitness = fitness_scores.mean()
        std_fitness = fitness_scores.std()

        # adjust scores based on population diversity (std_fitness)
        adjustment = self.adapt_factor * (std_fitness / (mean_fitness + 1e-10))
        adjusted_scores = fitness_scores + adjustment

        # prevent negative or zero probabilities
        adjusted_scores[adjusted_scores < 1e-10] = 1e-10

        probabilities = adjusted_scores / adjusted_scores.sum()
        selected_indices = np.random.choice(
            len(self.population),
            size=number_of_parents,
            p=probabilities,
        )
        return [self.population[i] for i in selected_indices]
