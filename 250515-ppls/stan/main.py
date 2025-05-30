from cmdstanpy import CmdStanModel
from pathlib import Path
import time

DATA = {
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    "J": 8,
}

def main():
    stan_file = Path(__file__).parent / "eight_schools_centered.stan"
    model = CmdStanModel(stan_file=stan_file)
    x = time.time()
    fit = model.sample(data=DATA, chains=1,
                       iter_warmup=10000, save_warmup=False,
                       iter_sampling=10000, thin=10)
    y = time.time()
    print(fit.summary())
    print(f"Time taken: {y - x} seconds")


if __name__ == "__main__":
    main()
