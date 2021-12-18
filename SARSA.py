# SARSA updates the Q-function moving it by an alpha parameter towards [r_t + gamma * Q(s_t+1, a_t+1) - Q(s_t, a_t)]
import numpy as np
from env import GridWorld
import random


def print_policy(policy):
    nice_repr = np.full((policy.shape), "⬆️")
    nice_repr[np.where(policy == 1)] = "⬇️"
    nice_repr[np.where(policy == 2)] = "⬅️"
    nice_repr[np.where(policy == 3)] = "▶️"
    print(nice_repr)


env = GridWorld(
    walls=[
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
    ],
    pitfalls=[(6, 5)],
)
policy_changed = True
q_values = np.zeros((*env.size, 4))
policy = np.full((*env.size,), "down")
action_space = ["up", "down", "left", "right"]
alpha = 0.1
gamma = 0.9
epsilon = 0.6
print(env)
for _ in range(1000000):
    policy_changed = False
    if random.random() < epsilon:
        action = np.random.randint(0, 4)
    else:
        action = np.argmax(q_values[tuple(env.player_position)])
    old_position = tuple(env.player_position[:])

    next_state, reward, done = env.step(action_space[action])
    if random.random() < epsilon:
        next_action = np.random.randint(0, 4)
    else:
        next_action = np.argmax(q_values[tuple(next_state)])

    update = alpha * (
        reward
        + gamma * q_values[next_state][next_action]
        - q_values[old_position][action]
    )
    q_values[old_position][action] += update
print_policy(np.argmax(q_values, axis=2))
