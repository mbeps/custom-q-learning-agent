# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
from collections import defaultdict
from typing import Dict, Tuple, List, DefaultDict

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    def __init__(self, state: GameState):
        # Extract position
        self.position = state.getPacmanPosition()

        # Get food positions
        food = state.getFood()
        food_positions = [
            (x, y) for x in range(food.width) for y in range(food.height) if food[x][y]
        ]

        # Get ghost positions
        ghost_positions = state.getGhostPositions()

        # Calculate distances
        self.nearest_food_distance = (
            min(
                [
                    util.manhattanDistance(self.position, food_pos)
                    for food_pos in food_positions
                ]
            )
            if food_positions
            else 0
        )

        self.nearest_ghost_distance = (
            min(
                [
                    util.manhattanDistance(self.position, ghost_pos)
                    for ghost_pos in ghost_positions
                ]
            )
            if ghost_positions
            else 999
        )

        # Discretize distances for better generalization
        self.nearest_food_distance = min(5, self.nearest_food_distance)
        self.nearest_ghost_distance = min(5, self.nearest_ghost_distance)

        # Check for food in each direction
        walls = state.getWalls()
        x, y = self.position
        self.has_food_north = (
            food[x][y + 1] if y + 1 < food.height and not walls[x][y + 1] else False
        )
        self.has_food_east = (
            food[x + 1][y] if x + 1 < food.width and not walls[x + 1][y] else False
        )
        self.has_food_south = (
            food[x][y - 1] if y - 1 >= 0 and not walls[x][y - 1] else False
        )
        self.has_food_west = (
            food[x - 1][y] if x - 1 >= 0 and not walls[x - 1][y] else False
        )

        # Count available exits (non-wall directions)
        self.exit_count = sum(
            [
                1
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if 0 <= x + dx < walls.width
                and 0 <= y + dy < walls.height
                and not walls[x + dx][y + dy]
            ]
        )

        # Ghost threat level
        self.is_ghost_threatening = self.nearest_ghost_distance <= 2

    def __eq__(self, other) -> bool:
        if not isinstance(other, GameStateFeatures):
            return False
        return (
            self.position == other.position
            and self.nearest_food_distance == other.nearest_food_distance
            and self.nearest_ghost_distance == other.nearest_ghost_distance
            and self.has_food_north == other.has_food_north
            and self.has_food_east == other.has_food_east
            and self.has_food_south == other.has_food_south
            and self.has_food_west == other.has_food_west
            and self.exit_count == other.exit_count
            and self.is_ghost_threatening == other.is_ghost_threatening
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.position,
                self.nearest_food_distance,
                self.nearest_ghost_distance,
                self.has_food_north,
                self.has_food_east,
                self.has_food_south,
                self.has_food_west,
                self.exit_count,
                self.is_ghost_threatening,
            )
        )


class QLearnAgent(Agent):
    def __init__(
        self,
        alpha: float = 0.2,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        maxAttempts: int = 30,
        numTraining: int = 10,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0

        # Initialize Q-values and visit counts
        self.q_values: DefaultDict[Tuple[GameStateFeatures, Directions], float] = (
            defaultdict(float)
        )
        self.visit_counts: DefaultDict[Tuple[GameStateFeatures, Directions], int] = (
            defaultdict(int)
        )

        # Store the previous state and action for learning
        self.lastState = None
        self.lastAction = None

        # Exploration constant for UCB
        self.exploration_constant = 2.0

        # Track total visits for UCB
        self.total_visits = 0

    def incrementEpisodesSoFar(self) -> None:
        self.episodesSoFar += 1

    def getEpisodesSoFar(self) -> int:
        return self.episodesSoFar

    def getNumTraining(self) -> int:
        return self.numTraining

    def setEpsilon(self, value: float) -> None:
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float) -> None:
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        # Base reward is the score difference
        reward = endState.getScore() - startState.getScore()

        # Add extra rewards for winning/losing
        if endState.isWin():
            reward += 500
        elif endState.isLose():
            reward -= 500

        # Add small distance-based rewards
        start_pos = startState.getPacmanPosition()
        end_pos = endState.getPacmanPosition()

        # Get food positions in start state
        start_food = startState.getFood()
        start_food_positions = [
            (x, y)
            for x in range(start_food.width)
            for y in range(start_food.height)
            if start_food[x][y]
        ]

        # Get food positions in end state
        end_food = endState.getFood()
        end_food_positions = [
            (x, y)
            for x in range(end_food.width)
            for y in range(end_food.height)
            if end_food[x][y]
        ]

        # Calculate distance to nearest food in both states
        start_food_dist = (
            min(
                [
                    util.manhattanDistance(start_pos, food_pos)
                    for food_pos in start_food_positions
                ]
            )
            if start_food_positions
            else 0
        )

        end_food_dist = (
            min(
                [
                    util.manhattanDistance(end_pos, food_pos)
                    for food_pos in end_food_positions
                ]
            )
            if end_food_positions
            else 0
        )

        # Reward for getting closer to food
        if start_food_dist > end_food_dist and not endState.isLose():
            reward += 1

        return reward

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        # Find all actions we've seen for this state
        actions = [action for (s, action) in self.q_values.keys() if s == state]

        if not actions:
            return 0.0

        return max([self.getQValue(state, action) for action in actions])

    def learn(
        self,
        state: GameStateFeatures,
        action: Directions,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        # Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        old_value = self.getQValue(state, action)
        next_max = self.maxQValue(nextState)

        # Update Q-value
        new_value = old_value + self.alpha * (
            reward + self.gamma * next_max - old_value
        )
        self.q_values[(state, action)] = new_value

    def updateCount(self, state: GameStateFeatures, action: Directions) -> None:
        self.visit_counts[(state, action)] += 1
        self.total_visits += 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        return self.visit_counts.get((state, action), 0)

    def explorationFn(self, utility: float, counts: int) -> float:
        if counts == 0:
            return float("inf")  # Encourage exploration of unseen states

        # UCB-inspired exploration
        return (
            utility
            + self.exploration_constant
            * ((self.total_visits + 1) / (counts + 1)) ** 0.5
        )

    def getAction(self, state: GameState) -> Directions:
        # Get legal actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Convert to feature state
        state_features = GameStateFeatures(state)

        # If this is not the first action of an episode, we need to learn from the previous step
        if self.lastState is not None and self.lastAction is not None:
            reward = self.computeReward(self.lastState, state)
            last_features = GameStateFeatures(self.lastState)
            self.learn(last_features, self.lastAction, reward, state_features)

        # Choose action according to exploration policy
        if not legal:
            return Directions.STOP

        # With probability epsilon, choose randomly
        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            # Choose action with highest Q-value, using exploration function
            action_values = [
                (
                    action,
                    self.explorationFn(
                        self.getQValue(state_features, action),
                        self.getCount(state_features, action),
                    ),
                )
                for action in legal
            ]
            # Select action with highest value
            action = max(action_values, key=lambda x: x[1])[0]

        # Update count for the selected action
        self.updateCount(state_features, action)

        # Remember state and action for next time
        self.lastState = state
        self.lastAction = action

        return action


def final(self, state: GameState) -> None:
    # Learn from the last action to the terminal state
    if self.lastState is not None and self.lastAction is not None:
        reward = self.computeReward(self.lastState, state)
        last_features = GameStateFeatures(self.lastState)
        curr_features = GameStateFeatures(state)
        self.learn(last_features, self.lastAction, reward, curr_features)

    # Reset for the next episode
    self.lastState = None
    self.lastAction = None

    # Decay epsilon slightly to reduce exploration over time
    if self.getEpisodesSoFar() < self.getNumTraining():
        self.epsilon = max(0.01, self.epsilon * 0.995)

    print(f"Game {self.getEpisodesSoFar()} just ended!")

    # Keep track of the number of games played, and set learning
    # parameters to zero when we are done with the pre-set number
    # of training episodes
    self.incrementEpisodesSoFar()
    if self.getEpisodesSoFar() == self.getNumTraining():
        msg = "Training Done (turning off epsilon and alpha)"
        print("%s\n%s" % (msg, "-" * len(msg)))
        self.setAlpha(0)
        self.setEpsilon(0)
