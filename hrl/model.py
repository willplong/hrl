from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.stats import norm


class DecisionModel(ABC):
    _NUM_ACTIONS = 2

    def __init__(
        self,
        params: np.ndarray,
        param_names: list[str],
        param_bounds: Bounds | None = None,
        param_constraints: LinearConstraint | None = None,
    ) -> None:
        if type(params) is not np.ndarray:
            raise ValueError("'params' must be a numpy array.")
        if len(params) != len(param_names):
            raise ValueError("Length of 'params' must match length of 'param_names'.")
        self._params = params
        self._param_names = param_names
        self._param_bounds = param_bounds
        self._param_constraints = param_constraints

    @property
    def params(self) -> np.ndarray:
        return self._params

    @params.setter
    def params(self, new_params: np.ndarray) -> None:
        self._params = new_params

    @property
    def param_bounds(self) -> Bounds | None:
        return self._param_bounds

    @property
    def param_constraints(self) -> LinearConstraint | None:
        return self._param_constraints

    @abstractmethod
    def action_probabilities(self, stimuli: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, stimuli: float | np.ndarray) -> np.ndarray:
        probabilities = self.action_probabilities(stimuli)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        return np.array(
            [np.random.choice(self._NUM_ACTIONS, p=prob) for prob in probabilities]
        )

    def likelihood(
        self, stimuli: float | np.ndarray, actions: float | np.ndarray
    ) -> np.ndarray:
        probabilities = self.action_probabilities(stimuli)
        if type(actions) is not np.ndarray:
            return probabilities[0, actions]
        likelihoods = np.zeros_like(actions, dtype=float)
        for i in range(probabilities.shape[-1]):
            likelihoods[actions == i] = probabilities[actions == i, i]
        return likelihoods

    def __repr__(self) -> str:
        param_str = ", ".join(
            f"{name}={value}" for name, value in zip(self._param_names, self.params)
        )
        return f"{self.__class__.__name__}({param_str})"


class CategoricalDecisionModel(DecisionModel):
    def __init__(self, probabilities: np.ndarray | list | None = None) -> None:
        if probabilities is None:
            probabilities = np.random.random(self._NUM_ACTIONS)
            probabilities /= probabilities.sum()
        elif type(probabilities) is list:
            probabilities = np.array(probabilities)
        param_names = [f"p{i}" for i in range(len(probabilities))]
        param_bounds = Bounds(0, 1, keep_feasible=True)
        param_constraints = LinearConstraint(
            np.ones(self._NUM_ACTIONS), lb=1, ub=1, keep_feasible=True
        )
        super().__init__(probabilities, param_names, param_bounds, param_constraints)

    def action_probabilities(self, stimuli: float | np.ndarray) -> np.ndarray:
        if type(stimuli) is not np.ndarray:
            return self._params
        return np.tile(self._params, reps=stimuli.shape + (1,))


class LogisticDecisionModel(DecisionModel):
    def __init__(
        self, bias: float | None = None, stim_weight: float | None = None
    ) -> None:
        if bias is None:
            bias = np.random.random()
        if stim_weight is None:
            stim_weight = np.random.random()
        params = np.array([bias, stim_weight])
        param_names = ["bias", "w_stim"]
        super().__init__(params, param_names)

    def action_probabilities(
        self, stimuli: float | np.ndarray | None = None
    ) -> np.ndarray:
        if type(stimuli) is not np.ndarray:
            stimuli = np.array([stimuli])
        bias, weight = self._params
        p = 1 / (1 + np.exp(-bias - weight * stimuli))
        return np.stack([1 - p, p], axis=-1)


class RLDecisionModel(DecisionModel):
    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        gamma: float | None = None,
        lamda: float | None = None,
    ) -> None:
        params = np.array(
            [np.random.random() if p is None else p for p in [mu, sigma, gamma, lamda]]
        )
        param_names = ["mu", "sigma", "gamma", "lamda"]
        param_bounds = Bounds(
            lb=np.array([-np.inf, 0, 0, 0]),
            ub=np.array([np.inf, np.inf, 1, 1]),
            keep_feasible=True,
        )
        param_constraints = LinearConstraint(
            np.array([[0, 0, 1, 1], [0, 0, 1, -1]]),
            lb=0,
            ub=np.array([1, 0]),
            keep_feasible=True,
        )
        super().__init__(params, param_names, param_bounds, param_constraints)

    def action_probabilities(
        self, stimuli: float | np.ndarray | None = None
    ) -> np.ndarray:
        if type(stimuli) is not np.ndarray:
            stimuli = np.array([stimuli])
        mu, sigma, gamma, lamda = self._params
        p = gamma + (1 - gamma - lamda) * norm.cdf(stimuli, mu, sigma)
        return np.stack([1 - p, p], axis=-1)
