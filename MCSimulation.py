import numpy as np
from env import GridWorld
import seaborn as sns

TRIALS_PER_STATE = 1000
walls = [
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 6),
    (3, 6),
    (4, 6),
    (5, 6),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 4),
]
pitfalls = [(6, 5)]
env = GridWorld(
    walls=walls,
    pitfalls=pitfalls,
)
values = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        state_value_sum = 0
        print(f"Simulating state {i},{j}")
        if (i, j) not in walls:
            for _ in range(TRIALS_PER_STATE):
                env.reset()
                env.player_position = [i, j]
                value_sum = 0
                done = False
                while not done:
                    if not ((i, j) == (8, 8) or (i, j) in pitfalls):
                        position, reward, done = env.step(
                            np.random.choice(["up", "down", "left", "right"])
                        )
                        value_sum += reward
                    elif (i, j) == (8, 8):
                        value_sum += 50
                        done = True
                    else:
                        value_sum -= 50
                        done = True
                state_value_sum += value_sum
            values[i, j] = state_value_sum / TRIALS_PER_STATE
print(values)
sns.heatmap(values)
import matplotlib.pyplot as plt

plt.show()
