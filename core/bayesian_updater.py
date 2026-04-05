class BayesianUpdater:
    """
    Incremental belief tracker using Beta distribution.
    Maintains internal state across decisions.
    Replaces stateless BayesianUpdate.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def update_beliefs(self, observation: bool) -> float:
        self.update(observation)
        return self.mean

    def reset(self) -> None:
        self.alpha = 1.0
        self.beta = 1.0