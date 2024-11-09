from knapsack.base.population import Population
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class ElitismSelector(Selector):
    def __init__(self, population: Population, evaluator: Evaluator):
        """Elitism selector, selects the best chromosomes as parents.

        Args:
            population (Population): Population object.
            evaluator (Evaluator): Evaluator object.
        """
        super().__init__(population)
        self.chromosomes = population.chromosomes
        self.population_size = population.population_size
        self.evaluator = evaluator

    def select(self):
        sorted_population = sorted(
            self.chromosomes,
            key=lambda c: self.evaluator.evaluate(c),
            reverse=True,
        )
        # fix it
        return sorted_population[: self.population_size // 2]
