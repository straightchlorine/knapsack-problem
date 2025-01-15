import numpy as np
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class RouletteSelector(Selector):
    def __init__(self, evaluator: Evaluator):
        super().__init__()
        self.evaluator = evaluator

    def select(self, number_of_parents=2):
        fitness_scores = np.array([self.evaluator.evaluate(c) for c in self.population])
        if (fitness_scores <= 0).all():
            fitness_scores = fitness_scores - fitness_scores.min() + 1e-10

        probabilities = fitness_scores / fitness_scores.sum()
        selected_indices = np.random.choice(
            len(self.population), size=number_of_parents, p=probabilities
        )
        return [self.population[i] for i in selected_indices]
