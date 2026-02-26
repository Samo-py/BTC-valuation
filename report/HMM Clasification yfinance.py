import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

df = yf.download("BTC-USD", auto_adjust=True)[["Close"]]
df.columns = ["price_change"]
df.index.name = None

dataframe = df.pct_change().dropna()
dataframe.columns = ["price_change"]
dataframe.index.name = None

print(df)
print(dataframe)

### Bear market

dates = [
    ["2017-12-11", "2018-12-18"],
    ["2021-11-01", "2022-11-21"]
    ]

### Bull market

# dates = [
#     ["2018-12-11", "2021-11-08"],
#     ["2022-11-14", datetime.strftime(datetime.now(), format="%Y-%m-%d")]
# ]

learning_data = df[dates[0][0]:dates[0][1]].pct_change().dropna()["price_change"].values + 1
observations = df[dates[1][0]:dates[1][1]].pct_change().dropna()["price_change"].values + 1


print(learning_data)
K = 3
N_learning = len(learning_data)
N_observed = len(observations)

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

mus = np.quantile(learning_data, [0.2, 0.5, 0.8])
sigmas = np.array([0.02, 0.01, 0.02])
iterations = 500
tol = 1e-4
prev_log_likelihood = -np.inf

### Baum-Welch

mus_history = []
sigmas_history = []

for iteration in range(iterations):

    ### Expectation
    ### Forward-Backward Alpha/Beta

    log_alpha = np.zeros((N_learning, K))

    for k in range(K):
        log_alpha[0, k] = log_coefs[k] + log_pdf(learning_data[0], mus[k], sigmas[k])

    for t in range(1, N_learning):
        for j in range(K):
            log_terms = log_alpha[t-1, :] + np.log(A[:, j])
            log_sum = np.logaddexp.reduce(log_terms)

            log_alpha[t, j] = log_pdf(learning_data[t], mus[j], sigmas[j]) + log_sum

    log_likelihood = np.logaddexp.reduce(log_alpha[N_learning-1, :])

    log_beta = np.zeros((N_learning, K))
    log_beta[N_learning-1, :] = 0

    for t in range(N_learning-2, -1, -1):
        for i in range(K):
            log_terms = np.log(A[i, :]) + log_pdf(learning_data[t+1], mus, sigmas) + log_beta[t+1, :]
            log_beta[t, i] = np.logaddexp.reduce(log_terms)
  
    ### Posterior probabilities Gamma/Xi

    log_gamma = log_alpha + log_beta
    for t in range(N_learning):
        log_sum = np.logaddexp.reduce(log_gamma[t, :])
        log_gamma[t, :] -= log_sum
    gamma = np.exp(log_gamma)

    log_xi = np.zeros((N_learning-1, K, K))
    for t in range(N_learning-1):
        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = log_alpha[t, i] + np.log(A[i, j]) + log_beta[t+1, j] + log_pdf(learning_data[t+1], mus[j], sigmas[j])
        log_sum = np.logaddexp.reduce(log_xi[t, :, :].flatten())
        log_xi[t, :, :] -= log_sum

    xi = np.exp(log_xi)

    ### Maximization

    log_coefs = log_gamma[0, :]

    for i in range(K):
        for j in range(K):
            A[i,j] = xi[:, i, j].sum()/ gamma[:-1,i].sum()
            A[i, j] = np.clip(A[i,j], 0.01, 0.99)
        
        A[i, :] /= A[i, :].sum()

    for k in range(K):
        mus[k] = (learning_data * gamma[:, k]).sum() / gamma[:, k].sum()

    for k in range(K):
        sigma = np.sqrt((gamma[:, k]*(learning_data - mus[k])**2).sum()/gamma[:,k].sum())
        sigmas[k] = max(sigma, 0.001)
    
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

print("means : ", mus)
print("dispersion : ", sigmas)



log_alpha = np.zeros((N_observed, K))

for k in range(K):
    log_alpha[0, k] = log_coefs[k] + log_pdf(observations[0], mus[k], sigmas[k])

for t in range(1, N_observed):
    for j in range(K):
        log_terms = log_alpha[t-1, :] + np.log(A[:, j])
        log_sum = np.logaddexp.reduce(log_terms)

        log_alpha[t, j] = log_pdf(observations[t], mus[j], sigmas[j]) + log_sum
regimes = np.argmax(log_alpha, axis=1)





time = pd.date_range(start=dates[1][0], end=dates[1][1]).astype("str")[:len(regimes)]

colors = ["#FFDDDD", "#DDFFDD", "#DDDDFF"]
regime_labels = ["Regime 0", "Regime 1", "Regime 2"]

fig, ax = plt.subplots(figsize=(12, 4))

start = 0
for t in range(1, N_observed):
    if regimes[t] != regimes[t-1] or t == N_observed-1:
        ax.axvspan(start, t, color=colors[regimes[t-1]], alpha=0.3)
        start = t

ax.plot(time, df.loc[dates[1][0]]["price_change"] * observations.cumprod(), color='tab:blue', linewidth=2)
for t in range(1, N_observed):
    if regimes[t] != regimes[t-1]:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.5)


ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.set_xticks(time[::len(time) // 5])
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=colors[i], alpha=0.3, label=regime_labels[i]) for i in range(3)]
ax.legend(handles=legend_patches, loc='upper right')


plt.tight_layout()
plt.savefig("images/bear market.png")
plt.show()