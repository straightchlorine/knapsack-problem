class Evaluator:
    """Base class for chromosome evaluators."""

    def __init__(self, problem):
        print(f"Evaluating chromosomes via {self.__class__.__name__}")
        self.problem = problem

    def evaluate(self, chromosome):
        """Evaluate the chromosome."""
        raise NotImplementedError(
            "Method 'evaluate' must be implemented in a subclass."
        )
