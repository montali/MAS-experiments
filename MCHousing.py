import numpy as np
import random
import math
import matplotlib.pyplot as plt

N = 500


def run_simulation(u, over_average, probability):
    alpha = np.random.rand()
    beta = np.random.rand()
    print(f"Simulation started, parameters are {alpha}, {beta}")
    sample_mean = None
    samples = []
    picked = None
    for i in range(N):
        this_house = np.random.beta(alpha, beta)
        sample_mean = (
            this_house
            if not sample_mean
            else ((sample_mean * (i - 1)) + this_house) / i
        )
        hoef_bound = math.exp(-2 * (i ** 2) * u)
        if (
            this_house > sample_mean + over_average
            and hoef_bound < probability
            and picked == None
        ):
            picked = i
            picked_score = this_house
            print(f"Our agent decided to pick house {i} with score {this_house}")
        samples.append(sample_mean)
    plt.plot(range(N), samples, label="Sample mean")
    plt.plot(
        range(N),
        [picked_score] * N,
        label=f"Picked house (n={picked}, score={round(picked_score,3)})",
    )
    plt.xlabel("Visited houses")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def run_n_simulations(u, over_average, probability, n):
    results = []
    for simulation in range(n):
        alpha = np.random.rand()
        beta = np.random.rand()
        sample_mean = None
        picked = None
        for i in range(100):
            this_house = np.random.beta(alpha, beta)
            sample_mean = (
                this_house
                if not sample_mean
                else ((sample_mean * (i - 1)) + this_house) / i
            )
            hoef_bound = math.exp(-2 * (i ** 2) * u)
            if (
                this_house > sample_mean + over_average
                and hoef_bound < probability
                and picked == None
            ):
                picked = i
                picked_score = this_house
                print(f"Our agent decided to pick house {i} with score {this_house}")
            if picked:
                results.append([picked, this_house - sample_mean])
    avg_pick = np.average(np.array(results), axis=0)
    print(
        f"The agent got the house in an average of {avg_pick[0]} visits with an average delta of {avg_pick[1]}"
    )


if __name__ == "__main__":
    run_simulation(0.05, 0.3, 0.005)
