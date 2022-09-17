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
                actions[i, t] = self._decision_models[states[t]].sample(stim_i[t])
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
                dm.fit(stimuli, actions, gamma[:, i, :])
            print(self.initial_probs)
            print(self.transition_probs)
            print(self.decision_params)

    def _e_step(
        self, stimuli: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        num_trials, time_steps, _ = stimuli.shape
        gamma = np.zeros((num_trials, self._num_states, time_steps))
        xi = np.zeros((num_trials, self._num_states, self._num_states, time_steps - 1))
        for i in range(num_trials):
            stim_i = stimuli[i]
            action_i = actions[i]
            gamma[i], xi[i] = self._e_step_helper(stim_i, action_i)
        return gamma, xi

    def _m_step(self, gamma, xi):
        initial_probs = np.mean(gamma[:, :, 0], axis=0)
        transition_probs = (
            np.sum(xi, axis=(0, 3))
            / np.sum(gamma[:, :, :-1], axis=(0, 2))[:, np.newaxis]
        )
        return initial_probs, transition_probs

    def _e_step_helper(
        self, seq_stim: np.ndarray, seq_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = self._forward_seq(seq_stim, seq_actions)
        beta = self._backward_seq(seq_stim, seq_actions)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0)
        xi = np.zeros((self._num_states, self._num_states, len(seq_actions) - 1))
        for t in range(len(seq_actions) - 1):
            for i in range(self._num_states):
                for j in range(self._num_states):
                    likelihood = self._decision_models[j].likelihood(
                        seq_stim[t + 1], seq_actions[t + 1]
                    )
                    xi[i, j, t] = (
                        alpha[i, t] * self._A[i, j] * likelihood * beta[j, t + 1]
                    )
        xi /= np.sum(xi, axis=(0, 1))
        return gamma, xi

    def _forward_seq(self, seq_stim: np.ndarray, seq_actions: np.ndarray) -> np.ndarray:
        alpha = np.zeros((self._num_states, len(seq_actions)))
        alpha[:, 0] = self._pi * [
            dm.likelihood(seq_stim[0], seq_actions[0]) for dm in self._decision_models
        ]
        for t in range(1, len(seq_actions)):
            for s in range(self._num_states):
                alpha[s, t] = np.sum(
                    alpha[:, t - 1] * self._A[:, s]
                ) * self._decision_models[s].likelihood(seq_stim[t], seq_actions[t])
            alpha[:, t] /= np.sum(alpha[:, t])
        return alpha

    def _backward_seq(
        self, seq_stim: np.ndarray, seq_actions: np.ndarray
    ) -> np.ndarray:
        beta = np.zeros((self._num_states, len(seq_actions)))
        beta[:, -1] = 1
        for t in range(len(seq_actions) - 2, -1, -1):
            for s in range(self._num_states):
                likelihoods = [
                    dm.likelihood(seq_stim[t + 1], seq_actions[t + 1])
                    for dm in self._decision_models
                ]
                beta[s, t] = np.sum(beta[:, t + 1] * self._A[s, :] * likelihoods)
        return beta
