from __future__ import absolute_import, print_function

import random
import sys
from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Tuple

from pacman import Directions, GameState
from pacman_utils import util
from pacman_utils.game import Agent


class GameStateFeatures:
    """Wrapper class around a game state for Q-learning algorithm"""

    def __init__(self, state: GameState) -> None:
        self.pos: Tuple[int, int] = state.getPacmanPosition()

        self.ghost_pos: Tuple[Tuple[float, float], ...] = tuple(
            state.getGhostPositions()
        )

        # Enhanced features for better state representation
        food_grid = state.getFood()
        food_positions: List[Tuple[int]] = [
            (x, y)
            for x in range(food_grid.width)
            for y in range(food_grid.height)
            if food_grid[x][y]
        ]

        # Distance to closest food
        if food_positions:
            self.closest_food_dist: int = min(
                util.manhattanDistance(self.pos, food_pos)
                for food_pos in food_positions
            )
        else:
            self.closest_food_dist = 0

        # Distance to closest ghost
        if self.ghost_pos:
            self.closest_ghost_dist = min(
                util.manhattanDistance(self.pos, ghost_pos)
                for ghost_pos in self.ghost_pos
            )
        else:
            self.closest_ghost_dist = 99  # Large value if no ghosts

        # Improved directional ghost danger features
        x, y = self.pos
        if self.ghost_pos:
            # Calculate danger in each direction
            self.ghost_danger_north: float = min(
                [util.manhattanDistance((x, y + 1), ghost) for ghost in self.ghost_pos]
                + [999]
            )
            self.ghost_danger_south: float = min(
                [util.manhattanDistance((x, y - 1), ghost) for ghost in self.ghost_pos]
                + [999]
            )
            self.ghost_danger_east: float = min(
                [util.manhattanDistance((x + 1, y), ghost) for ghost in self.ghost_pos]
                + [999]
            )
            self.ghost_danger_west: float = min(
                [util.manhattanDistance((x - 1, y), ghost) for ghost in self.ghost_pos]
                + [999]
            )

        # Food count feature
        self.food_count: int = food_grid.count()

        self.food_map = state.getFood()

        self.legalActions: List[str] = state.getLegalPacmanActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

    def __hash__(self) -> int:
        return hash((self.pos, self.ghost_pos, self.food_map))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GameStateFeatures):
            return False

        return (
            self.pos == other.pos
            and self.ghost_pos == other.ghost_pos
            and self.food_map == other.food_map
        )


class QLearner:
    """Handles the Q-learning algorithm logic"""

    def __init__(
        self, alpha: float, epsilon: float, gamma: float, maxAttempts: int
    ) -> None:
        # Learning hyperparameters
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.maxAttempts: int = maxAttempts

        # Optimistic initialization to small positive values
        self.QValue: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(
            lambda: 1.0
        )

        self.visitedTimes: DefaultDict[Tuple[GameStateFeatures, str], int] = (
            defaultdict(int)
        )

    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        """Returns the Q-value for a state-action pair"""
        return self.QValue[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        """Returns the maximum Q-value attainable from the state"""
        if not state.legalActions:
            return 0.0
        else:
            return max([self.getQValue(state, a) for a in state.legalActions])

    def learn(
        self,
        state: GameStateFeatures,
        action: str,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        """Performs a Q-learning update"""
        QValue_last: float = self.getQValue(state, action)

        maxQValue_current: float = self.maxQValue(nextState)

        new_QValue: float = QValue_last + self.alpha * (
            reward + self.gamma * maxQValue_current - QValue_last
        )

        # Update the value of state-action pair
        self.QValue[(state, action)] = new_QValue

    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        """Updates the visitation count for a state-action pair"""
        self.visitedTimes[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: str) -> int:
        """Returns the number of times an action has been taken in a state"""
        return self.visitedTimes[(state, action)]

    def explorationFn(self, utility: float, counts: int) -> float:
        """Computes the exploration function value with more aggressive exploration"""
        if counts == 0:
            return float("inf")  # Strong incentive to explore never-visited actions

        # Improved balance of exploitation with exploration
        # Using square root to make the exploration bonus decay more slowly
        return utility + (self.maxAttempts / (counts**0.5))

    def getBestAction(self, state: GameStateFeatures) -> str:
        """Improved best action selection with tie-breaking"""
        if not state.legalActions:
            return Directions.STOP

        # Get all actions with maximum Q-value (might be multiple)
        qValues: List[Tuple[str | float]] = [
            (a, self.getQValue(state, a)) for a in state.legalActions
        ]
        maxValue: float = max(qValues, key=lambda x: x[1])[1]
        bestActions: List[str | float] = [a for a, v in qValues if v == maxValue]

        # If multiple actions have the same value, break ties with exploration counts
        if len(bestActions) > 1:
            return min(bestActions, key=lambda a: self.getCount(state, a))

        return bestActions[0]

    def getExplorationAction(self, state: GameStateFeatures) -> str:
        """Returns an action based on exploration strategy"""
        # With probability epsilon, choose a random action (exploration)
        if util.flipCoin(self.epsilon):
            return random.choice(state.legalActions)

        # Otherwise, choose the best action using count-based exploration
        return max(
            state.legalActions,
            key=lambda a: self.explorationFn(
                self.getQValue(state, a), self.getCount(state, a)
            ),
        )


class QLearnAgent(Agent):
    def __init__(
        self,
        alpha: float = 0.2,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        maxAttempts: int = 30,
        numTraining: int = 10,
    ) -> None:
        super().__init__()
        self.alpha: float = float(alpha)
        self.epsilon: float = float(epsilon)
        self.gamma: float = float(gamma)
        self.maxAttempts: int = int(maxAttempts)
        self.numTraining: int = int(numTraining)

        # Store initial values for adaptive parameters
        self.initial_epsilon: float = float(epsilon)
        self.initial_alpha: float = float(alpha)

        # Count the number of games we have played
        self.episodesSoFar: int = 0

        # Create Q-learner instance
        self.qLearner: QLearner = QLearner(
            alpha=self.alpha,
            epsilon=self.epsilon,
            gamma=self.gamma,
            maxAttempts=self.maxAttempts,
        )

        # Store the last action pacman took, initialized to "West"
        self.lastAction: str = "West"

        # Store the last state where pacman was, initialized to None
        self.lastState: Optional[GameState] = None

    def incrementEpisodesSoFar(self) -> None:
        self.episodesSoFar += 1

    def getEpisodesSoFar(self) -> int:
        return self.episodesSoFar

    def getNumTraining(self) -> int:
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float) -> None:
        self.epsilon = value
        self.qLearner.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float) -> None:
        self.alpha = value
        self.qLearner.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        return self.qLearner.getQValue(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        return self.qLearner.maxQValue(state)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(
        self,
        state: GameStateFeatures,
        action: str,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        self.qLearner.learn(state, action, reward, nextState)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        self.qLearner.updateCount(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self, state: GameStateFeatures, action: str) -> int:
        return self.qLearner.getCount(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self, utility: float, counts: int) -> float:
        return self.qLearner.explorationFn(utility, counts)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> str:
        if self.lastState is None:
            self.lastState = state

        lastStateFeatures = GameStateFeatures(self.lastState)
        currentStateFeatures = GameStateFeatures(state)

        reward: float = self.computeReward(self.lastState, state)

        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        action: str = self.qLearner.getExplorationAction(currentStateFeatures)

        self.updateCount(currentStateFeatures, action)

        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState) -> None:
        # Calculate final reward
        reward: float = self.computeReward(self.lastState, state)

        self.learn(
            GameStateFeatures(self.lastState),
            self.lastAction,
            reward,
            GameStateFeatures(state),
        )

        # Adjust epsilon and alpha based on training progress
        if self.getEpisodesSoFar() < self.getNumTraining():
            # Linearly decrease epsilon from initial value to 0.01
            progress: float = self.getEpisodesSoFar() / self.getNumTraining()
            new_epsilon: float = max(0.01, self.initial_epsilon * (1.0 - progress))
            self.setEpsilon(new_epsilon)

            # Also adapt alpha based on training progress
            new_alpha: float = max(0.1, self.initial_alpha * (1.0 - progress * 0.5))
            self.setAlpha(new_alpha)

            sys.stdout.write(
                f"\rEpisode {self.getEpisodesSoFar() + 1}/{self.getNumTraining()}"
            )
            sys.stdout.flush()

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = "\nTraining Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
