from knapsack.population import Population
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class ElitismSelector(Selector):
    def __init__(self, evaluator: Evaluator):
        """Elitism selector, selects the best chromosomes as parents.

        Args:
            population (Population): Population object.
            evaluator (Evaluator): Evaluator object.
        """
        super().__init__()
        self.evaluator = evaluator

    def select(self):
        sorted_population = sorted(
            self.population,
            key=lambda c: self.evaluator.evaluate(c),
            reverse=True,
        )
        return sorted_population[:2]
