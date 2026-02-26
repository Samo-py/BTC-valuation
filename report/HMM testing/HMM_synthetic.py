import numpy as np
import matplotlib.pyplot as plt

def synthetic_data(A, mus, sigmas, start_state, days):

    rng = np.random.default_rng()

    states = np.zeros(days + 1, dtype=int)

    states[0] = start_state

    for t in range(days):
        states[t + 1] = rng.choice(3, p=A[states[t]])

    observations = rng.normal(mus[states], sigmas[states])

    return observations, states



if __name__ == "__main__":

    days = 730

    A = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])

    mus = np.array([-10, 0, 10])
    sigmas = np.array([1, 1, 1])

    start_state = 0

    observations, states = synthetic_data(A, mus, sigmas, start_state, days)

    print(observations)
    print(states)