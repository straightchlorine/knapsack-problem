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
        """Tournament selection mechanism with variable tournament sizes.

        Args:
            evaluator (Evaluator): Evaluator object.
            min_tournament_size (int): Minimum number of participants in a tournament.
            max_tournament_size (int): Maximum number of participants in a tournament.
            number_of_parents (int): Number of parents to select.
        """
        super().__init__()
        self.evaluator = evaluator
        self.min_tournament_size = min_tournament_size
        self.max_tournament_size = max_tournament_size
        self.number_of_parents = number_of_parents

    def select(self):
        """Select parents using tournament selection with variable sizes.

        Returns:
            list: List of selected parents.
        """
        parents = []
        for _ in range(self.number_of_parents):
            # randomly determine the tournament size based on the constraints
            tournament_size = np.random.randint(
                self.min_tournament_size, self.max_tournament_size + 1
            )

            if tournament_size > len(self.population):
                raise ValueError(
                    f"Tournament size {tournament_size} cannot be larger than the population size ({len(self.population)})."
                )

            # select individuals for the tournament
            tournament_indices = np.random.choice(
                range(len(self.population)),
                size=tournament_size,
                replace=False,
            )
            tournament = self.population[tournament_indices]

            # select the best individual in the tournament
            winner = max(tournament, key=lambda c: self.evaluator.evaluate(c))
            parents.append(winner)

        return parents
