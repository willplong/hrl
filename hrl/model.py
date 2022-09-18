from abc import ABC, abstractmethod

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint
from scipy.stats import norm


class DecisionModel(ABC):
    NUM_ACTIONS = 2

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

    def __repr__(self) -> str:
        param_str = ", ".join(
            f"{name}={value}" for name, value in zip(self._param_names, self.params)
        )
        return f"{self.__class__.__name__}({param_str})"

    @property
    def params(self) -> np.ndarray:
        return self._params

    @params.setter
    def params(self, new_params: np.ndarray) -> None:
        self._params = new_params

    @property
    def param_names(self) -> list[str]:
        return self._param_names

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
        """Simulates the model given the stimuli."""
        probabilities = self.action_probabilities(stimuli)
        if len(probabilities) == 1:
            return np.random.choice(self.NUM_ACTIONS, p=probabilities[0])
        return np.array(
            [np.random.choice(self.NUM_ACTIONS, p=prob) for prob in probabilities]
        )

    def log_likelihood(
        self,
        stimuli: float | np.ndarray,
        actions: int | np.ndarray,
        sum_over_trials: bool = True,
    ) -> float:
        """
        Computes the log likelihood of the model across all trials.

        Args:
            stimuli: The stimuli presented to the subject.
            actions: The actions taken by the subject.
            sum_over_trials: Whether to sum the log likelihood across trials. If False,
                the log likelihood is returned for each trial.
        """
        probabilities = self.action_probabilities(stimuli)
        if type(actions) is not np.ndarray:
            if len(probabilities) > 1:
                raise ValueError("Must provide array of actions for multiple trials.")
            return np.log(probabilities[0, actions])
        likelihoods = np.zeros_like(actions, dtype=float)
        for i in range(probabilities.shape[-1]):
            likelihoods[actions == i] = probabilities[actions == i, i]
        result = np.log(likelihoods)
        if sum_over_trials:
            return result.sum()
        return result

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
            weights: The weights to use for each trial. If None, all of the trials are
                weighted equally.
        """
        if weights is None:
            weights = np.ones_like(actions)

        def nll(params):
            self.params = params
            return -np.sum(
                weights * self.log_likelihood(stimuli, actions, sum_over_trials=False)
            )

        self.params = scipy.optimize.minimize(
            nll,
            self.params,
            method="trust-constr",
            bounds=self.param_bounds,
            constraints=self.param_constraints,
        ).x

    def _convert_array(self, stimuli: float | np.ndarray) -> np.ndarray:
        if type(stimuli) is not np.ndarray:
            stimuli = np.array([stimuli])
        if stimuli.ndim == 1:
            stimuli = stimuli.reshape(1, -1)
        return stimuli


class CategoricalDecisionModel(DecisionModel):
    def __init__(self, probabilities: np.ndarray | list | None = None) -> None:
        if probabilities is None:
            probabilities = np.random.random(self.NUM_ACTIONS)
            probabilities /= probabilities.sum()
        elif type(probabilities) is list:
            probabilities = np.array(probabilities)
        param_names = [f"p{i}" for i in range(len(probabilities))]
        param_bounds = Bounds(0, 1, keep_feasible=True)
        param_constraints = LinearConstraint(
            np.ones(self.NUM_ACTIONS), lb=1, ub=1, keep_feasible=True
        )
        super().__init__(probabilities, param_names, param_bounds, param_constraints)

    def _action_probabilities_impl(self, stimuli: float | np.ndarray) -> np.ndarray:
        if type(stimuli) is not np.ndarray:
            return self._params
        return np.tile(self._params, reps=stimuli.shape + (1,))


class LogisticDecisionModel(DecisionModel):
    def __init__(
        self,
        bias: float | None = None,
        weights: float | list | np.ndarray | None = None,
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
        p = 1 / (1 + np.exp(-bias - np.sum(weights * stimuli, axis=-1)))
        return np.stack([1 - p, p], axis=-1)


class RLDecisionModel(DecisionModel):
    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        gamma_l: float | None = None,
        gamma_h: float | None = None,
    ) -> None:
        # TODO: Avoid hard-coding mu based on the IBL dataset.
        mu = np.random.uniform(10, 15) if mu is None else mu
        sigma = np.random.uniform(0.1, 1) if sigma is None else sigma
        gamma_l = np.random.uniform(0, 0.2) if gamma_l is None else gamma_l
        gamma_h = np.random.uniform(0, 0.2) if gamma_h is None else gamma_h
        params = np.array([mu, sigma, gamma_l, gamma_h])
        param_names = ["mu", "sigma", "gamma_l", "gamma_h"]
        # TODO: Check whether bounds and constraints are correct.
        param_bounds = Bounds(
            lb=np.array([-np.inf, 0, 0, 0]),
            ub=np.array([np.inf, np.inf, 0.5, 0.5]),
            keep_feasible=True,
        )
        param_constraints = LinearConstraint(
            np.array([[0, 0, 1, 1]]), lb=0, ub=1, keep_feasible=True
        )
        super().__init__(params, param_names, param_bounds, param_constraints)

    def _action_probabilities_impl(self, stimuli: np.ndarray) -> np.ndarray:
        mu, sigma, gamma_l, gamma_h = self.params
        p = gamma_l + (1 - gamma_l - gamma_h) * norm.cdf(stimuli, mu, sigma)
        p = p.squeeze()
        return np.stack([1 - p, p], axis=-1)
