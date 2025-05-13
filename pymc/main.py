import arviz
import numpy as np
import pymc as pm
import time

y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = 8

def priors():
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    return mu, tau

def main():
    with pm.Model() as eight_schools_centered:
        mu, tau = priors()
        theta = pm.Normal("theta_trans", mu, tau, shape=J)
        # theta = pm.Deterministic("theta", mu + tau * theta_trans)
        obs = pm.Normal("obs", theta, sigma, observed=y)

        start = time.time()
        trace = pm.sample(draws=10000, tune=10000, chains=1)
        # nuts_sampler="blackjax"
        end = time.time()

    print(f"took {end - start} seconds")
    # print(arviz.summary(trace))  # needs 2 chains

if __name__ == "__main__":
    main()
