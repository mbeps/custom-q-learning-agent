from __future__ import absolute_import
from __future__ import print_function

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

import random
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, DefaultDict, Any
import sys


class GameStateFeatures:
    """Wrapper class around a game state for Q-learning algorithm"""

    def __init__(self, state: GameState):
        self.pos: Tuple[int, int] = state.getPacmanPosition()

        self.ghost_pos: Tuple[Tuple[float, float], ...] = tuple(
            state.getGhostPositions()
        )

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

    def __init__(self, alpha: float, epsilon: float, gamma: float, maxAttempts: int):
        # Learning hyperparameters
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.maxAttempts: int = maxAttempts

        self.QValue: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(
            float
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

        maxQValue_current = self.maxQValue(nextState)

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
        """Computes the exploration function value"""
        if counts == 0:
            return float("inf")  # Strong incentive to explore never-visited actions

        # Balance exploitation with exploration
        return utility + (self.maxAttempts / (counts + 1))

    def getBestAction(self, state: GameStateFeatures) -> str:
        """Returns the best action from legal actions based on Q-values"""
        if not state.legalActions:
            return Directions.STOP

        bestAction = state.legalActions[0]
        for a in state.legalActions:
            if self.getQValue(state, a) > self.getQValue(state, bestAction):
                bestAction = a
        return bestAction

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

        if self.getEpisodesSoFar() < self.getNumTraining():
            sys.stdout.write(
                f"\rEpisode {self.getEpisodesSoFar()}/{self.getNumTraining()} ended!"
            )
            sys.stdout.flush()

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
