# import arviz
import numpy as np
import pymc as pm
import time

y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = 8

def main():
    with pm.Model() as eight_schools_centered:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, shape=J)
        obs = pm.Normal("obs", theta, sigma, observed=y)

        start = time.time()
        trace = pm.sample(draws=10000, tune=10000, chains=1)
        end = time.time()

    print(f"took {end - start} seconds")
    # print(arviz.summary(trace))  # needs 2 chains

if __name__ == "__main__":
    main()
