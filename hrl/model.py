from abc import ABC, abstractmethod

import numpy as np
import scipy
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

    def action_probabilities(self, stimuli: float | np.ndarray) -> np.ndarray:
        stimuli = self._convert_array(stimuli)
        return self._action_probabilities_impl(stimuli)

    @abstractmethod
    def _action_probabilities_impl(self, stimuli: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, stimuli: float | np.ndarray) -> np.ndarray:
        probabilities = self.action_probabilities(stimuli)
        return np.array(
            [np.random.choice(self._NUM_ACTIONS, p=prob) for prob in probabilities]
        )

    def log_likelihood(
        self,
        stimuli: float | np.ndarray,
        actions: int | np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """
        Computes the log likelihood of the model given the stimuli and actions.

        Args:
            stimuli: The stimuli presented to the subject.
            actions: The actions taken by the subject.
            weights: The weights to apply to each trial (e.g., for EM algorithms). If
                None, all trials are weighted equally.
        """
        probabilities = self.action_probabilities(stimuli)
        if type(actions) is not np.ndarray:
            if len(probabilities) > 1:
                raise ValueError("Must provide array of actions for multiple trials.")
            if weights is not None:
                raise ValueError("Cannot provide weights for single trial.")
            return np.log(probabilities[0, actions])
        if weights is None:
            weights = 1
        likelihoods = np.zeros_like(actions, dtype=float)
        for i in range(probabilities.shape[-1]):
            likelihoods[actions == i] = probabilities[actions == i, i]
        return np.sum(weights * np.log(likelihoods))

    def fit(
        self,
        stimuli: float | np.ndarray,
        actions: int | np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Fits the model to the given data.

        Args:
            stimuli: The stimuli presented to the subject.
            actions: The actions taken by the subject.
            weights: The weights to apply to each trial (e.g., for EM algorithms). If
                None, all trials are weighted equally.
        """

        def nll(params):
            self.params = params
            return -self.log_likelihood(stimuli, actions, weights)

        self.params = scipy.optimize.minimize(
            nll,
            self.params,
            method="trust-constr",
            bounds=self.param_bounds,
            constraints=self.param_constraints,
        ).x

    def _convert_array(self, stimuli: float | np.ndarray) -> np.ndarray:
        """Ensures stimuli is always a 2D array of shape (trials, features)."""
        if type(stimuli) is not np.ndarray:
            stimuli = np.array([stimuli])
        if stimuli.ndim == 1:
            stimuli = stimuli.reshape(1, -1)
        elif stimuli.ndim > 2:
            raise ValueError("Stimuli must be a 1D or 2D array.")
        return stimuli


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

    def _action_probabilities_impl(self, stimuli: float | np.ndarray) -> np.ndarray:
        if type(stimuli) is not np.ndarray:
            return self._params
        return np.tile(self._params, reps=stimuli.shape + (1,))


class LogisticDecisionModel(DecisionModel):
    def __init__(
        self, bias: float | None = None, stim_weight: float | None = None
    ) -> None:
        if bias is None:
            bias = np.random.random()
        if weights is None:
            weights = np.random.random()
        try:
            params = np.array([bias, *weights])
            param_names = ["bias", *[f"w{i}" for i in range(len(weights))]]
        except TypeError:
            params = np.array([bias, weights])
            param_names = ["bias", "weight"]
        super().__init__(params, param_names)

    def _action_probabilities_impl(self, stimuli: np.ndarray) -> np.ndarray:
        bias, weights = self.params[0], self.params[1:]
        p = 1 / (1 + np.exp(-bias - np.sum(weights * stimuli, axis=1)))
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

    def _action_probabilities_impl(self, stimuli: np.ndarray) -> np.ndarray:
        mu, sigma, gamma, lamda = self._params
        p = gamma + (1 - gamma - lamda) * norm.cdf(stimuli, mu, sigma)
        return np.stack([1 - p, p], axis=-1)
