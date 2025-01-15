import numpy as np
from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class VariableTournamentSelector(Selector):
    def __init__(
        self,
        evaluator: Evaluator,
        min_tournament_size: int = 2,
        max_tournament_size: int = 5,
        number_of_parents: int = 2,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.min_tournament_size = min_tournament_size
        self.max_tournament_size = max_tournament_size
        self.number_of_parents = number_of_parents

    def select(self):
        parents = []
        for _ in range(self.number_of_parents):
            tournament_size = np.random.randint(
                self.min_tournament_size, self.max_tournament_size + 1
            )
            self.validate_population_size(tournament_size)

            tournament = self.random_sample(tournament_size)
            winner = max(tournament, key=lambda c: self.evaluator.evaluate(c))
            parents.append(winner)
        return parents
