class Evaluator:
    """Base class for chromosome evaluators."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __str__(self):
        print(35 * "=")
        return f"{self.dataset}\nEvaluator: {self.__class__.__name__}"

    def evaluate(self, chromosome):
        """Evaluate the chromosome."""
        raise NotImplementedError(
            "Method 'evaluate' must be implemented in a subclass."
        )
