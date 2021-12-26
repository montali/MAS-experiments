# SARSA updates the Q-function moving it by an alpha parameter towards [r_t + gamma * Q(s_t+1, a_t+1) - Q(s_t, a_t)]
import numpy as np
from env import GridWorld
import random
import matplotlib.pyplot as plt

arg_to_move = {0: "up", 1: "down", 2: "left", 3: "right"}


def print_policy(policy):
    nice_repr = np.full((policy.shape), "⬆️")
    nice_repr[np.where(policy == 1)] = "⬇️"
    nice_repr[np.where(policy == 2)] = "⬅️"
    nice_repr[np.where(policy == 3)] = "▶️"
    print(nice_repr)


def values_to_latex(values):
    result = ""
    for i in values:
        for j in i:
            result += f"{round(j,3)} &"
        result = result[:-1] + "\\\\"
    return result


def mc_simulation(policy, env, walls, pitfalls):
    env.reset()
    TRIALS_PER_STATE = 1000
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
                    position = [i, j]
                    steps = 0
                    while not done and steps < 50:  # Avoid loops
                        steps += 1
                        if not ((i, j) == (8, 8) or (i, j) in pitfalls):
                            old_position = position
                            position, reward, done = env.step(
                                arg_to_move[policy[tuple(position)]]
                            )
                            if old_position == position:
                                # STUCK
                                break
                            value_sum += reward
                        elif (i, j) == (8, 8):
                            value_sum += 50
                            done = True
                        else:
                            value_sum -= 50
                            done = True
                    state_value_sum += value_sum
                values[i, j] = state_value_sum / TRIALS_PER_STATE
    print(values_to_latex(values))


STEPS = 1000000
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
policy_changed = True
q_values = np.zeros((*env.size, 4))
policy = np.full((*env.size,), "down")
action_space = ["up", "down", "left", "right"]
alpha = 0.1
gamma = 0.9
epsilon = 0.6
print(env)
history = np.zeros((STEPS, *q_values.shape))
for _ in range(STEPS):
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
    history[_] = q_values
diffn = np.zeros((STEPS))
for _ in range(STEPS):
    diffn[_] = np.sum(np.abs(history[_] - q_values))
# plt.plot(range(STEPS), diffn)
# plt.xlabel("Iterations")
# plt.ylabel("Absolute difference from last Q")
# plt.show()
policy = np.argmax(q_values, axis=2)
print_policy(policy)
print("===== MC simulation =====")
mc_simulation(policy, env, walls, pitfalls)
