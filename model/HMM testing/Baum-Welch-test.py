import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from HMM_synthetic import synthetic_data

days = 730

A_true = np.array([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9]
])

mus_true = np.array([-1, 0, 1])
sigmas_true = np.array([3, 1, 3])

start_state_true = 0

observations, states = synthetic_data(A_true, mus_true, sigmas_true, start_state_true, days)

K = 3
N = days + 1

def pdf(x, mu, sigma):

    coeficient = 1/(np.sqrt(np.pi * 2)*sigma)
    exponent = -((x - mu)**2)/(2*sigma**2)
    return coeficient * np.exp(exponent)

def log_pdf(x, mu, sigma):
    return -np.log(sigma) - 0.5*np.log(2*np.pi) - ((x - mu)**2) / (2*sigma**2)

log_coefs = np.log(np.ones(K) / K)

A = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])

mus = np.quantile(observations, [0.2, 0.5, 0.8])
sigmas = np.array([1.0, 1.0, 1.0])

iterations = 50
tol = 1e-4
prev_log_likelihood = -np.inf

### Baum-Welch

mus_history = []
sigmas_history = []

for iteration in range(iterations):

    ### Expectation
    ### Forward-Backward Alpha/Beta

    log_alpha = np.zeros((N, K))

    for k in range(K):
        log_alpha[0, k] = log_coefs[k] + log_pdf(observations[0], mus[k], sigmas[k])

    for t in range(1, N):
        for j in range(K):
            log_terms = log_alpha[t-1, :] + np.log(A[:, j])
            log_sum = np.logaddexp.reduce(log_terms)

            log_alpha[t, j] = log_pdf(observations[t], mus[j], sigmas[j]) + log_sum

    log_likelihood = np.logaddexp.reduce(log_alpha[N-1, :])

    log_beta = np.zeros((N, K))
    log_beta[N-1, :] = 0

    for t in range(N-2, -1, -1):
        for i in range(K):
            log_terms = np.log(A[i, :]) + log_pdf(observations[t+1], mus[i], sigmas[i]) + log_beta[t+1, :]
            log_beta[t, i] = np.logaddexp.reduce(log_terms)
  
    ### Posterior probabilities Gamma/Xi

    log_gamma = log_alpha + log_beta
    for t in range(N):
        log_sum = np.logaddexp.reduce(log_gamma[t, :])
        log_gamma[t, :] -= log_sum
    gamma = np.exp(log_gamma)

    log_xi = np.zeros((N-1, K, K))
    for t in range(N-1):
        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = log_alpha[t, i] + np.log(A[i, j]) + log_beta[t+1, j] + log_pdf(observations[t+1], mus[j], sigmas[j])
        log_sum = np.logaddexp.reduce(log_xi[t, :, :].flatten())
        log_xi[t, :, :] -= log_sum

    xi = np.exp(log_xi)

    ### Maximization

    log_coefs = log_gamma[0, :]

    for i in range(K):
        for j in range(K):
            A[i,j] = xi[:, i, j].sum()/ gamma[:-1,i].sum()
            A[i, j] = np.clip(A[i,j], 0.01, 0.99)

    for k in range(K):
        mus[k] = (observations * gamma[:, k]).sum() / gamma[:, k].sum()

    for k in range(K):
        sigma = np.sqrt((gamma[:, k]*(observations - mus[k])**2).sum()/gamma[:,k].sum())
        sigmas[k] = max(sigma, 0.1)
    
    mus_history.append(mus.copy())
    sigmas_history.append(sigmas.copy())

    if iteration > 0:
        improvement = log_likelihood - prev_log_likelihood
        print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.2f}, improvement = {improvement:.4f}")
        
        if improvement < tol:
            print(f"Converged at iteration {iteration}")
            break
    else:
        print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.2f}")

    prev_log_likelihood = log_likelihood

### Virterbi


log_delta = np.zeros((N, K)) - np.inf
psi = np.zeros((N, K), dtype=int)

for k in range(K):
    log_delta[0, k] = log_coefs[k] + log_pdf(observations[0], mus[k], sigmas[k])

for t in range(1, N):
    for j in range(K):
        best_log = -np.inf
        best_i = 0

        for i in range(K):
            log_val = log_delta[t-1, i] + np.log(max(A[i, j], 1e-100))
            if log_val >= best_log:

                best_log = log_val
                best_i = i

        emission = max(log_pdf(observations[t], mus[j], sigmas[j]), -100)
        log_delta[t, j] = best_log + emission
        psi[t, j] = best_i

### Virterbi path

best_path = np.zeros(N, dtype=int)
best_path[N-1] = np.argmax(log_delta[N-1])

for t in range(N-2, -1, -1):
    best_path[t] = psi[t+1, best_path[t+1]]



print("Succes rate : ", 100 * np.mean(best_path == states), "%")




### visualization
linspace = 1000

x_plot = np.linspace(-8, 8, linspace)

true_pdf_plot = np.zeros((linspace, K))
for k in range(K):
    true_pdf_plot[:,k] = 1/K * pdf(x_plot, mus_true[k], sigmas_true[k])

guess_pdf_plot = np.zeros((linspace, K))
for k in range(K):
    guess_pdf_plot[:,k] = 1/K * pdf(x_plot, mus_history[0][k], sigmas_history[0][k])

fig = make_subplots()
fig.add_trace(go.Histogram(x=observations, nbinsx=200, histnorm='probability density', opacity=0.6, name='Data'),)
for k in range(K):
    fig.add_trace(go.Scatter(x=x_plot, y=true_pdf_plot[:,k], mode='lines', name=f'True PDF {k+1}', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=x_plot, y=guess_pdf_plot[:,k], mode='lines', name=f'Guess PDF {k+1}'))

slider_steps = []
for i in range(0, len(mus_history)):
    guess_pdf_plot = np.zeros((linspace, K))
    for k in range(K):
        guess_pdf_plot[:,k] = 1/K * pdf(x_plot, mus_history[i][k], sigmas_history[i][k])
    step = dict(
    method="update",
    args=[
        {"y": [
            None,
            true_pdf_plot[:,0],
            guess_pdf_plot[:,0],
            true_pdf_plot[:,1],
            guess_pdf_plot[:,1],
            true_pdf_plot[:,2],
            guess_pdf_plot[:,2],
        ],
         "x": [
             observations,
             x_plot,
             x_plot,
             x_plot,
             x_plot,
             x_plot,
             x_plot,
         ]}
    ],
    label=str(i)
)
    slider_steps.append(step)

fig.update_layout(
    height=700,
    sliders=[dict(active=0, steps=slider_steps)],
    showlegend=True
)

fig.show()
