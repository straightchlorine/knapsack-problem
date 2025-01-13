import numpy as np

from knapsack.evaluators.evaluator import Evaluator
from knapsack.selectors.selector import Selector


class TournamentSelector(Selector):
    def __init__(
        self,
        evaluator: Evaluator,
        tournament_size: int = 3,
        number_of_parents: int = 2,
    ):
        """Tournament selection mechanism.

        Args:
            evaluator (Evaluator): Evaluator object.
            tournament_size (int): Number of individuals in each tournament.
                Defaults to 3.
        """
        super().__init__()
        self.evaluator = evaluator
        self.tournament_size = tournament_size
        self.number_of_parents = number_of_parents

    def select(self):
        """Select parents using tournament selection.

        Returns:
            list: List of the two selected parents
        """
        parents = []
        for _ in range(self.number_of_parents):
            if self.tournament_size > len(self.population):
                raise ValueError(
                    "Tournament size cannot be larger than the population size."
                )

            # select individuals for the tournament
            tournament_indices = np.random.choice(
                range(len(self.population)),
                size=self.tournament_size,
                replace=False,
            )
            tournament = self.population[tournament_indices]

            # select the best individual in the tournament
            winner = max(tournament, key=lambda c: self.evaluator.evaluate(c))
            parents.append(winner)
        return parents
