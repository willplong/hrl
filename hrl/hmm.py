import numpy as np
import scipy
from tqdm import tqdm

from .model import DecisionModel


def generate_data(
    initial_probs, transition_probs, decision_models, time_steps, num_sequences
):
    stimuli = []
    data = []
    for _ in tqdm(range(num_sequences)):
        stim, observations = generate_sequence(
            initial_probs, transition_probs, decision_models, time_steps
        )
        stimuli.append(stim)
        data.append(observations)
    return np.array(stimuli), np.array(data)


def generate_sequence(initial_probs, transition_probs, decision_models, time_steps):
    stimuli = np.random.random(time_steps) * 2 - 1
    num_states = len(initial_probs)
    states = [np.random.choice(num_states, p=initial_probs)]
    for _ in range(time_steps - 1):
        states.append(np.random.choice(num_states, p=transition_probs[states[-1]]))
    observations = []
    for stimulus, state in zip(stimuli, states):
        observations.append(decision_models[state].sample(stimulus))
    observations = np.concatenate(observations)
    return stimuli, observations


def forward(
    stimuli: np.ndarray,
    sequence: np.ndarray,
    initial_probs: np.ndarray,
    transition_probs: np.ndarray,
    decision_models: list[DecisionModel],
) -> np.ndarray:
    num_states = len(initial_probs)
    alpha = np.zeros((num_states, len(sequence)))
    alpha[:, 0] = initial_probs * [
        dm.likelihood(stimuli[0], sequence[0]) for dm in decision_models
    ]
    for t in range(1, len(sequence)):
        for s in range(num_states):
            alpha[s, t] = np.sum(
                alpha[:, t - 1] * transition_probs[:, s]
            ) * decision_models[s].likelihood(stimuli[t], sequence[t])
        alpha[:, t] /= np.sum(alpha[:, t])
    return alpha


def backward(stimuli, sequence, initial_probs, transition_probs, decision_models):
    num_states = len(initial_probs)
    beta = np.zeros((num_states, len(sequence)))
    beta[:, -1] = 1
    for t in range(len(sequence) - 2, -1, -1):
        for s in range(num_states):
            beta[s, t] = np.sum(
                beta[:, t + 1]
                * transition_probs[s, :]
                * [
                    dm.likelihood(stimuli[t + 1], sequence[t + 1])
                    for dm in decision_models
                ]
            )
    return beta


def e_step(stimuli, data, initial_probs, transition_probs, decision_models):
    gamma = []
    xi = []
    for stim, seq in zip(stimuli, data):
        gamma_s, xi_s = e_step_helper(
            stim, seq, initial_probs, transition_probs, decision_models
        )
        gamma.append(gamma_s)
        xi.append(xi_s)
    gamma = np.array(gamma)
    xi = np.array(xi)
    return gamma, xi


def e_step_helper(stimuli, sequence, initial_probs, transition_probs, decision_models):
    num_states = len(initial_probs)
    alpha = forward(stimuli, sequence, initial_probs, transition_probs, decision_models)
    beta = backward(stimuli, sequence, initial_probs, transition_probs, decision_models)
    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=0)
    xi = np.zeros((num_states, num_states, len(sequence) - 1))
    for t in range(len(sequence) - 1):
        for i in range(num_states):
            for j in range(num_states):
                xi[i, j, t] = (
                    alpha[i, t]
                    * transition_probs[i, j]
                    * decision_models[j].likelihood(stimuli[t + 1], sequence[t + 1])
                    * beta[j, t + 1]
                )
    xi /= np.sum(xi, axis=(0, 1))
    return gamma, xi


def m_step_latents(data, gamma, xi):
    initial_probs = np.mean(gamma[:, :, 0], axis=0)
    transition_probs = (
        np.sum(xi, axis=(0, 3)) / np.sum(gamma[:, :, :-1], axis=(0, 2))[:, np.newaxis]
    )
    return initial_probs, transition_probs


def m_step_observations(stimuli, data, gamma, decision_models):
    def decision_model_nll(params, gamma, decision_model):
        decision_model.params = params
        return -np.sum(gamma * np.log(decision_model.likelihood(stimuli, data)))

    for i, dm in enumerate(decision_models):
        dm.params = scipy.optimize.minimize(
            decision_model_nll,
            dm.params,
            args=(gamma[:, i, :], dm),
            method="trust-constr",
            bounds=dm.param_bounds,
            constraints=dm.param_constraints,
        ).x


def baum_welch(stimuli, data, num_states, model_type, num_iters=100):
    initial_probs = np.random.random(num_states)
    initial_probs /= initial_probs.sum()
    transition_probs = np.random.random((num_states, num_states))
    transition_probs /= transition_probs.sum(axis=1)[:, np.newaxis]
    decision_models = [model_type() for _ in range(num_states)]
    for _ in tqdm(range(num_iters)):
        gamma, xi = e_step(
            stimuli, data, initial_probs, transition_probs, decision_models
        )
        initial_probs, transition_probs = m_step_latents(data, gamma, xi)
        m_step_observations(stimuli, data, gamma, decision_models)
        print(initial_probs)
        print(transition_probs)
        print([dm.params for dm in decision_models])
    return initial_probs, transition_probs, decision_models
