import numpy as np
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import math

B = 1
T = 2


class Node:
    """
    Responsible for the creation of a game tree, starting from the root node.
    """

    def __init__(self, info: str, distance: tuple = np.inf):
        """
        Inits a new node with a game state and a move.
        :param game: the game object
        :param move: the move that led to this node
        """
        self.info = info
        self.children = []
        self.parent = None
        self.visits = 0
        self.set_distance(distance)
        self.depth = len(info)

    def set_distance(self, distance):
        """
        Sets the distance of the node.
        :param distance: the distance of the node
        """
        self.distance = distance
        self.value = B * math.exp(-distance / T)

    def add_child(self, child):
        """
        Adds a child to the node.
        :param child: the child node
        """
        self.children.append(child)
        child.parent = self
        return child

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return f"Node {self.info}, having score {self.value}"


class MCTS:
    """
    Responsible for the Monte Carlo Tree Search algorithm.
    """

    def __init__(self, root, target, c=3.0, verbose=False, target_depth=12):
        """
        Inits a new MCTS object.
        :param root: the root node of the game tree
        :param c: the exploration parameter
        :param verbose: whether to print the root's children
        """
        self.root = root
        self.c = c
        self.verbose = verbose
        self.target_depth = target_depth
        self.target = target

    def search(self, time_limit=2):
        """
        Performs a search round, given a time limit.
        :param time_limit: the time limits in seconds

        :return: the best move
        """
        start_time = time.time()
        if self.verbose:
            print("Root node before search: " + str(self.root))
        while time.time() - start_time < time_limit:
            node = self.selection(self.root)
            self.expansion(node)
            node = self.simulation(node)
            self.backpropagation(node)
        if self.verbose:
            print(
                f"Root's children after search:{len(self.root.children)} "
                + str(self.root.children)
            )
        for (
            child
        ) in self.root.children:  # In the last run, the target is a child of the root
            if child.info == self.target:
                return child
        return max(self.root.children, key=lambda x: x.value / x.visits)

    def selection(self, node):
        """
        Travels down the tree until a leaf node is reached.
        :param node: the current node

        :return: the leaf node
        """
        while node.children:
            node = self.best_child(node, self.c)
        return node

    def compute_distance(self, info):
        """
        Computes the distance of a given leaf.
        :param info: the state

        :return: the distance of the state
        """
        distance = 0
        for i, letter in enumerate(info):
            if self.target[i] != letter:
                distance += 1
        return distance

    def expansion(self, node):
        """
        Expands a leaf node with children containing the available moves
        :param node: the leaf node
        """
        if not node.children and node.depth < self.target_depth:
            for addition in ["L", "R"]:
                child = Node(node.info + addition)
                node.add_child(child)

    def simulation(self, node):
        """
        Simulates a random game from the current node.
        :param node: the current node

        :return: the leaf node (containing a reward!)
        """
        while node.depth < self.target_depth:
            addition = np.random.choice(["L", "R"])
            child = Node(
                node.info + addition,
            )
            if node != self.root:
                node.add_child(child)
            node = child
        node.set_distance(self.compute_distance(node.info))
        return node

    def backpropagation(self, leaf):
        """
        Propagates the reward up the tree.
        :param leaf: the leaf node
        """
        node = leaf.parent
        while node is not None:
            node.visits += 1
            node.value += leaf.value
            node = node.parent

    def best_child(self, node, c):
        """
        Returns the best child of a node, given the exploration parameter.

        score = exploit + c*explore
        exploit = ratio between wins and visits for the child
        explore = square root of the ratio between the log of the root's visits and the child's visits

        :param node: the current node
        :param c: the exploration parameter

        :return:
        the best child
        """
        best_score = -1
        best_children = []
        for child in node.children:
            if child.visits == 0:
                score = np.inf
            else:
                exploit = child.value / child.visits
                explore = np.sqrt(np.log(node.visits) / child.visits)
                score = exploit + c * explore
            if score == best_score:
                best_children.append(child)
            if score > best_score:
                best_children = [child]
                best_score = score
        return np.random.choice(best_children)


if __name__ == "__main__":
    target_depth = 12
    target = "".join(np.random.choice(["L", "R"], size=target_depth))
    result = ""
    while len(result) < target_depth:
        root = Node(result)
        mcts = MCTS(root, target, verbose=True)
        picked_letter = mcts.search(time_limit=2).info
        result = picked_letter
        print(f"Searched string now {result}, target={target}")
