import numpy as np
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class RouletteSelector(Selector):
    def __init__(self, evaluator: Evaluator):
        """Proportional (roulette-wheel) selection mechanism.

        Args:
            evaluator (Evaluator): Evaluator object.
        """
        super().__init__()
        self.evaluator = evaluator

    def select(self, number_of_parents=2):
        """Select parents using roulette-wheel selection.

        Args:
            number_of_parents (int): Number of parents to select.

        Returns:
            list: List of selected chromosomes.
        """
        # evaluate the fitness of the entire population
        fitness_scores = np.array(
            [self.evaluator.evaluate(c) for c in self.population]
        ).flatten()
        total_fitness = fitness_scores.sum()

        # in case of negative selection, shift it
        if np.min(fitness_scores) < 0:
            fitness_scores = fitness_scores - np.min(fitness_scores) + 1e-10

        if total_fitness == 0:
            raise ValueError(
                "Total fitness is zero. Cannot perform selection."
            )

        # calculate the probabilities of the selection and normalise, so they
        # add up to 1
        probabilities = fitness_scores / total_fitness
        probabilities = probabilities / probabilities.sum()

        # based on calculated probabilities, select indexes of selected parents
        selected_indices = np.random.choice(
            len(self.population),
            size=number_of_parents,
            p=probabilities,
        )

        return [self.population[i] for i in selected_indices]
