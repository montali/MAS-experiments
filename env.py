import numpy as np


class InvalidMoveException(Exception):
    pass


class GridWorld:
    """
    Responsible for the game.
    Walls represent cells of the grid we cannot enter. A transition to these can't happen.
    Pitfalls represent cells of the grid that will end the game if reached, with a negative reward.
    The goal is the only cell returning a reward and ending the game.
    """

    def __init__(
        self,
        walls: np.array,
        pitfalls: np.array,
        size: tuple = (9, 9),
        start: tuple = (0, 0),
        goal: tuple = (8, 8),
    ):
        """
        Inits a new game.
        :param walls: the walls of the game
        :param pitfalls: the pitfalls of the game
        :param size: the size of the grid
        :param start: the starting position
        :param goal: the goal position
        """
        assert goal not in pitfalls, "Goal cannot be a pitfall"
        assert start not in pitfalls, "Start cannot be a pitfall"
        assert goal not in walls, "Goal cannot be a wall"
        assert start not in walls, "Start cannot be a wall"
        self.walls = walls
        self.pitfalls = pitfalls
        self.size = size
        self.start = start
        self.goal = goal
        self.reset()
        self.walls = walls

    def reset(self):
        """
        Resets the game.
        """
        self.player_position = list(self.start)
        self.done = False

    def step(self, move: str):
        """
        Steps the agent in a direction.
        :param move: the direction to move: ["up", "down", "left", "right"]
        """
        assert move in ["up", "down", "left", "right"], "Invalid move"
        target = self.player_position[:]
        if move == "up" and self.player_position[0] > 0:
            target[0] -= 1
        elif move == "down" and self.player_position[0] < self.size[0] - 1:
            target[0] += 1
        elif move == "left" and self.player_position[1] > 0:
            target[1] -= 1
        elif move == "right" and self.player_position[1] < self.size[1] - 1:
            target[1] += 1
        else:
            return tuple(self.player_position), 0, False
            # raise InvalidMoveException(
            #     f"Invalid move: can't go {move} in {self.player_position}"
            # )
        if tuple(target) == self.goal:
            self.done = True
            return tuple(self.player_position), 50, True
        elif tuple(target) in self.pitfalls:
            self.done = True
            return tuple(self.player_position), -50, True
        elif tuple(target) in self.walls:
            return tuple(self.player_position), 0, False
        else:
            self.player_position = target
            return tuple(self.player_position), 0, False

    def __repr__(self):
        """
        Represents the game status graphically.
        """
        if self.done:
            return f"Game over"
        nice_repr = np.full(self.size, "🏝")
        for wall in self.walls:
            nice_repr[wall] = "🚧"
        for pit in self.pitfalls:
            nice_repr[pit] = "🕳"
        nice_repr[self.goal] = "🏆"
        nice_repr[tuple(self.player_position)] = "🐒"
        return f"GridWorld:\n{nice_repr}"
