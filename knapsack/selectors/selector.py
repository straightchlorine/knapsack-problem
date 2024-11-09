# from knapsack.base.population import Population


class Selector:
    """Base class for the selector objects."""

    # def __init__(self, population: Population):
    def __init__(self, population):
        print(f"Selecting parents parents via {self.__class__.__name__}")
        self.population = population

    def select(self):
        """Select parents for the next generation.

        Returns:
            list: List of selected chromosomes
        """
        raise NotImplementedError(
            'Method "select" must be implemented in a subclass.'
        )
