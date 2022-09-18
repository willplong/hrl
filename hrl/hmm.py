import numpy as np
from tqdm import tqdm

from .model import DecisionModel


class HMM:
    def __init__(self, num_states: int, decision_type: type[DecisionModel]):
        self._num_states = num_states
        self._num_actions = decision_type.NUM_ACTIONS
        self._pi = np.random.random(num_states)
        self._pi /= np.sum(self._pi)
        self._A = np.random.random((num_states, num_states))
        self._A /= np.sum(self._A, axis=1)[:, np.newaxis]
        self._decision_models = [decision_type() for _ in range(num_states)]

    def __repr__(self) -> str:
        return f"HMM(num_states={self._num_states}, num_actions={self._num_actions})"

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def initial_probs(self) -> np.ndarray:
        return self._pi

    @initial_probs.setter
    def initial_probs(self, initial_probs: np.ndarray) -> None:
        if initial_probs.ndim != 1:
            raise ValueError("'initial_probs' must be a 1D array.")
        if initial_probs.shape[0] != self._num_states:
            raise ValueError("'initial_probs' must be size num_states.")
        if not np.allclose(np.sum(initial_probs), 1):
            raise ValueError("'initial_probs' must sum to 1.")
        self._pi = initial_probs

    @property
    def transition_probs(self) -> np.ndarray:
        return self._A

    @transition_probs.setter
    def transition_probs(self, transition_probs: np.ndarray) -> None:
        if transition_probs.ndim != 2:
            raise ValueError("'transition_probs' must be a 2D array.")
        if (
            transition_probs.shape[0] != self._num_states
            or transition_probs.shape[1] != self._num_states
        ):
            raise ValueError(
                "'transition_probs' must be size (num_states x num_states)."
            )
        if not np.allclose(np.sum(transition_probs, axis=1), 1):
            raise ValueError("'transition_probs' must sum to 1 along each row.")
        self._A = transition_probs

    @property
    def decision_params(self) -> list[np.ndarray]:
        return [dm.params for dm in self._decision_models]

    @decision_params.setter
    def decision_params(self, decision_params: list[np.ndarray]) -> None:
        if len(decision_params) != self._num_states:
            raise ValueError("'decision_params' must be a list of length 'num_states'.")
        for i, params in enumerate(decision_params):
            self._decision_models[i].params = params

    def simulate(self, stimuli: np.ndarray) -> np.ndarray:
        if stimuli.ndim != 3:
            raise ValueError("stimuli must be a 3D array (trials x time x features)")
        num_trials, time_steps, _ = stimuli.shape
        actions = np.zeros((num_trials, time_steps), dtype=int)
        for i in range(num_trials):
            stim_i = stimuli[i]
            states = [np.random.choice(self._num_states, p=self._pi)]
            for _ in range(time_steps - 1):
                states.append(np.random.choice(self._num_states, p=self._A[states[-1]]))
            for t in range(time_steps):
                actions[i, t] = self._decision_models[states[t]].simulate(stim_i[t])
        return actions

    # TODO: Implement
    def log_likelihood(
        self,
        stimuli: float | np.ndarray,
        actions: int | np.ndarray,
    ) -> float:
        pass

    def fit(
        self,
        stimuli: np.ndarray,
        actions: np.ndarray,
        num_iters: int = 100,
        progress: bool = True,
    ) -> None:
        for _ in tqdm(range(num_iters), disable=not progress):
            gamma, xi = self._e_step(stimuli, actions)
            self._pi, self._A = self._m_step(gamma, xi)
            for i, dm in enumerate(self._decision_models):
                dm.fit(stimuli, actions, gamma[:, :, i])
            print(self.initial_probs)
            print(self.transition_probs)
            print(self.decision_params)

    def _e_step(
        self, stimuli: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        num_trials, time_steps, _ = stimuli.shape
        log_likelihoods = np.zeros((num_trials, time_steps, self._num_states))
        for i in range(self._num_states):
            log_likelihoods[:, :, i] = self._decision_models[i].log_likelihood(
                stimuli,
                actions,
                sum_over_trials=False,
            )
        gamma = np.zeros((num_trials, time_steps, self._num_states))
        xi = np.zeros((num_trials, time_steps - 1, self._num_states, self._num_states))
        for i, log_likes_i in enumerate(log_likelihoods):
            gamma[i], xi[i] = self._e_step_helper(log_likes_i)
        return gamma, xi

    def _m_step(self, gamma, xi):
        initial_probs = np.mean(gamma[:, 0, :], axis=0)
        transition_probs = (
            np.sum(xi, axis=(0, 1))
            / np.sum(gamma[:, :-1, :], axis=(0, 1))[:, np.newaxis]
        )
        return initial_probs, transition_probs

    def _e_step_helper(
        self, seq_log_likes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        time_steps = len(seq_log_likes)
        alpha = self._forward_seq(seq_log_likes)
        beta = self._backward_seq(seq_log_likes)
        gamma = alpha + beta
        gamma = np.exp(gamma)
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        xi = np.zeros((time_steps - 1, self._num_states, self._num_states))
        for t in range(time_steps - 1):
            for i in range(self._num_states):
                for j in range(self._num_states):
                    xi[t, i, j] = alpha[t, i] + seq_log_likes[t + 1, j] + beta[t + 1, j]
                    xi[t, i, j] = np.exp(xi[t, i, j])
                    xi[t, i, j] *= self._A[i, j]
        xi /= np.sum(xi, axis=(1, 2), keepdims=True)
        return gamma, xi

    def _forward_seq(self, seq_log_likes: np.ndarray) -> np.ndarray:
        time_steps = len(seq_log_likes)
        alpha = np.zeros((time_steps, self._num_states))
        alpha[0] = np.log(self._pi) + seq_log_likes[0]
        for t in range(1, time_steps):
            alpha_max = np.max(alpha[t - 1])
            for s in range(self._num_states):
                alpha[t, s] = (
                    alpha_max
                    + np.log(np.sum(np.exp(alpha[t - 1] - alpha_max) * self._A[:, s]))
                    + seq_log_likes[t, s]
                )
        return alpha

    def _backward_seq(self, seq_log_likes: np.ndarray) -> np.ndarray:
        time_steps = len(seq_log_likes)
        beta = np.zeros((time_steps, self._num_states))
        beta[-1] = np.zeros(self._num_states)
        for t in range(time_steps - 2, -1, -1):
            val = beta[t + 1] + seq_log_likes[t + 1]
            val_max = np.max(val)
            for s in range(self._num_states):
                beta[t, s] = val_max + np.log(
                    np.sum(np.exp(val - val_max) * self._A[s])
                )
        return beta
