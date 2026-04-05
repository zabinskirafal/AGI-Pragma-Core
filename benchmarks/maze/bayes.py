class BetaTracker:
    """
    Beta(a, b) distribution tracker for a binary event probability.
    mean = a / (a + b)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = float(a)
        self.b = float(b)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def update(self, event: bool):
        if event:
            self.a += 1.0
        else:
            self.b += 1.0
