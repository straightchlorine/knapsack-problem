import numpy as np
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class RankingSelector(Selector):
    def __init__(self, evaluator: Evaluator, scale_factor: float = 1.5):
        """Ranking selection mechanism.

        Args:
            evaluator (Evaluator): Evaluator object.
            scale_factor (float): Scaling factor for ranking probabilities.
        """
        super().__init__()
        self.evaluator = evaluator
        self.scale_factor = scale_factor

    def select(self, number_of_parents=2):
        """Select parents using ranking selection.

        Args:
            number_of_parents (int): Number of parents to select.

        Returns:
            list: List of selected chromosomes.
        """
        # rank chromosomes based on fitness
        fitness_scores = np.array([self.evaluator.evaluate(c) for c in self.population])
        ranks = np.argsort(np.argsort(-fitness_scores))  # Higher fitness = lower rank

        # calculate probabilities based on ranks
        probabilities = np.exp(-ranks / self.scale_factor)
        probabilities /= probabilities.sum()

        # select parents based on probabilities
        selected_indices = np.random.choice(
            len(self.population),
            size=number_of_parents,
            p=probabilities,
        )
        return [self.population[i] for i in selected_indices]
