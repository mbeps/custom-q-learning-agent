from __future__ import absolute_import, print_function

import random
import sys
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

from pacman import Directions, GameState
from pacman_utils import util
from pacman_utils.game import Agent, Grid


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm.

    This implementation focuses on relational features (directions to objects)
    rather than just absolute positions to improve generalisation.

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState) -> None:
        """
        Initialise the game state features with the given game state.

        Args:
            state: A given game state object
        """
        # Basic position information
        self.pacman: Tuple[int, int] = state.getPacmanPosition()
        self.ghost_pos: Tuple[Tuple[float, float], ...] = tuple(
            state.getGhostPositions()
        )
        self.walls: list = state.getWalls().asList()

        # Get food data
        food_grid: Grid[bool] = state.getFood()
        self.food: list = food_grid.asList()
        self.food_map: Grid[bool] = food_grid
        self.food_count: int = food_grid.count()

        # Legal actions excluding STOP
        self.legalActions: List[str] = state.getLegalPacmanActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

        # Find nearest food direction and distance using BFS
        (self.food_distance, self.food_direction) = self.findNearest(
            self.pacman, self.food
        )

        # Find nearest ghost direction and distance using BFS
        ghost_pos_list = list(self.ghost_pos)
        (self.ghost_distance, self.ghost_direction) = self.findNearest(
            self.pacman, ghost_pos_list
        )

        # Measure relative positioning of ghost and food
        if self.ghost_direction != self.food_direction:
            self.ghost_to_food_distance = None
        else:
            self.ghost_to_food_distance = self.ghost_distance - self.food_distance

        # Detailed ghost danger in each direction (useful for immediate decision making)
        x, y = self.pacman
        if self.ghost_pos:
            # Calculate danger in each direction
            self.ghost_danger_north: float = min(
                [util.manhattanDistance((x, y + 1), ghost) for ghost in self.ghost_pos]
                + [999]  # Default high value
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

    def findNearest(
        self,
        location: Tuple[int, int],
        objects_to_compare: List[Tuple[int, int]],
        distance: int = 0,
        visited: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[int, Optional[str]]:
        """
        Find the nearest object to the given location using BFS pathfinding

        Args:
            location: Starting location (x, y)
            objects_to_compare: List of object locations to find
            distance: Current distance (used internally)
            visited: Set of already visited locations (used internally)

        Returns:
            Tuple of (distance to nearest object, direction to move)
        """
        max_search_depth: int = 8  # Prevent searching too far
        min_dist: int = 1000  # Default minimum distance

        # Map surrounding directions with coordinates
        (x, y) = location
        direction_mapping: List[Tuple[Tuple[int, int], str]] = [
            ((x, y + 1), Directions.NORTH),
            ((x, y - 1), Directions.SOUTH),
            ((x - 1, y), Directions.WEST),
            ((x + 1, y), Directions.EAST),
        ]

        # If we've found an object or reached max depth
        if distance == max_search_depth or location in objects_to_compare:
            return distance, None

        if visited is None:
            visited = [location]

        min_direction: Optional[str] = None

        for coord, direction in direction_mapping:
            if coord not in self.walls and coord not in visited:
                (current_dist, _) = self.findNearest(
                    coord, objects_to_compare, distance + 1, visited + [coord]
                )
                if current_dist < min_dist and current_dist != max_search_depth:
                    min_dist = current_dist
                    min_direction = direction

        return min_dist, min_direction

    def __hash__(self) -> int:
        """
        Hash function based on relational features rather than just absolute positions
        for better state generalisation
        """
        return hash(
            (
                self.pacman,  # Keep pacman position for uniqueness
                self.ghost_direction,
                self.food_direction,
                self.ghost_to_food_distance,
                self.food_count,  # Add food count for better state distinction
            )
        )

    def __eq__(self, other) -> bool:
        """
        Equality based on relational features rather than exact coordinates

        Args:
            other: Another GameStateFeatures object

        Returns:
            True if states are equivalent, False otherwise
        """
        if not isinstance(other, GameStateFeatures):
            return False

        return (
            self.pacman == other.pacman
            and self.ghost_direction == other.ghost_direction
            and self.food_direction == other.food_direction
            and self.ghost_to_food_distance == other.ghost_to_food_distance
            and self.food_count == other.food_count
        )


class QLearner:
    """Handles the Q-learning algorithm logic"""

    def __init__(
        self, alpha: float, epsilon: float, gamma: float, maxAttempts: int
    ) -> None:
        """
        Initialise the Q-learning algorithm parameters and data structures

        Args:
            alpha: Learning rate
            epsilon: Exploration rate
            gamma: Discount factor
            maxAttempts: Maximum number of attempts for exploration bonus
        """
        # Learning hyperparameters
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.maxAttempts: int = maxAttempts

        # Optimistic initialisation to small positive values for better exploration
        self.QValue: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(
            lambda: 1.0
        )
        self.visitedTimes: DefaultDict[Tuple[GameStateFeatures, str], int] = (
            defaultdict(int)
        )
        self.exploration_k_value: float = 10.0

    def getMaxAttempts(self) -> int:
        """
        Get the maximum number of attempts for exploration.

        Returns:
            Maximum number of attempts
        """
        return self.maxAttempts

    def setMaxAttempts(self, value: int) -> None:
        """
        Set the maximum number of attempts.

        Args:
            value: New maximum attempts value
        """
        self.maxAttempts = value

    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        """
        Get Q-value for a state-action pair

        Args:
            state: The state
            action: The action

        Returns:
            Q-value
        """
        return self.QValue[(state, action)]

    def setQValue(self, state: GameStateFeatures, action: str, value: float) -> None:
        """
        Set Q-value for a state-action pair

        Args:
            state: The state
            action: The action
            value: New Q-value
        """
        self.QValue[(state, action)] = value

    def getVisitCount(self, state: GameStateFeatures, action: str) -> int:
        """
        Get visit count for a state-action pair

        Args:
            state: The state
            action: The action

        Returns:
            Visit count
        """
        return self.visitedTimes[(state, action)]

    def incrementVisitCount(self, state: GameStateFeatures, action: str) -> None:
        """
        Increment visit count for a state-action pair

        Args:
            state: The state
            action: The action
        """
        self.visitedTimes[(state, action)] += 1

    def getBestAction(
        self, state: GameStateFeatures, qValues: List[Tuple[str, float]]
    ) -> str:
        """
        Get the best action based on Q-values with tie-breaking

        Args:
            state: Current state
            qValues: List of (action, q-value) pairs

        Returns:
            Best action to take
        """
        if not state.legalActions:
            return Directions.STOP

        # Find maximum Q-value
        maxValue: float = max(qValues, key=lambda x: x[1])[1] if qValues else 0.0

        # Get all actions with the maximum value
        bestActions: List[str] = [a for a, v in qValues if v == maxValue]

        # If multiple actions have the same value, break ties with exploration counts
        if len(bestActions) > 1:
            return min(bestActions, key=lambda a: self.getVisitCount(state, a))

        return bestActions[0] if bestActions else Directions.STOP


class QLearnAgent(Agent):
    """
    Q-learning agent for playing Pacman. Implements the Q-learning algorithm
    to learn optimal policies through experience.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        maxAttempts: int = 30,
        numTraining: int = 10,
    ) -> None:
        """
        Initialise the Q-learning agent with learning parameters.

        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        Args:
            alpha: Learning rate
            epsilon: Exploration rate
            gamma: Discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: Number of training episodes
        """
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
        self.episodesCount: int = 0

        # Create Q-learner instance to handle the learning algorithm
        self.qLearner: QLearner = QLearner(
            alpha=self.alpha,
            epsilon=self.epsilon,
            gamma=self.gamma,
            maxAttempts=self.maxAttempts,
        )

        # Store the last action pacman took, initialised to a default value
        self.lastAction: str = Directions.WEST

        # Store the last state where pacman was
        self.lastState: Optional[GameState] = None

    def incrementEpisodesSoFar(self) -> None:
        """
        Increment the counter for episodes completed.
        """
        self.episodesCount += 1

    def getEpisodesSoFar(self) -> int:
        """
        Get the number of episodes completed.

        Returns:
            Number of episodes completed
        """
        return self.episodesCount

    def getNumTraining(self) -> int:
        """
        Get the number of training episodes.

        Returns:
            Number of training episodes
        """
        return self.numTraining

    def getEpsilon(self) -> float:
        """
        Get the exploration rate.

        Returns:
            Exploration rate
        """
        return self.epsilon

    def setEpsilon(self, value: float) -> None:
        """
        Set the exploration rate and update the Q-learner's epsilon too.

        Args:
            value: New exploration rate
        """
        self.epsilon = value
        self.qLearner.epsilon = value

    def getAlpha(self) -> float:
        """
        Get the learning rate.

        Returns:
            Learning rate
        """
        return self.alpha

    def setAlpha(self, value: float) -> None:
        """
        Set the learning rate and update the Q-learner's alpha too.

        Args:
            value: New learning rate
        """
        self.alpha = value
        self.qLearner.alpha = value

    def getGamma(self) -> float:
        """
        Get the discount factor.

        Returns:
            Discount factor
        """
        return self.gamma

    def getMaxAttempts(self) -> int:
        """
        Get the maximum number of attempts for exploration.

        Returns:
            Maximum number of attempts
        """
        return self.maxAttempts

    def setMaxAttempts(self, value: int) -> None:
        """
        Set the maximum number of attempts and update the Q-learner's maxAttempts too.

        Args:
            value: New maximum attempts value
        """
        self.maxAttempts = value
        self.qLearner.maxAttempts = value

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Compute the reward for a transition as the score difference.

        Args:
            startState: Starting game state
            endState: Ending game state

        Returns:
            Reward value
        """
        # Simple reward function that compares the score before and after the action
        return endState.getScore() - startState.getScore()

    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        """
        Get the Q-value for a state-action pair.

        Args:
            state: State features
            action: Action to take

        Returns:
            Q-value
        """
        return self.qLearner.getQValue(state, action)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Get the maximum Q-value for a state.

        Args:
            state: State features

        Returns:
            Maximum Q-value
        """
        if not state.legalActions:
            return 0.0

        return max([self.getQValue(state, a) for a in state.legalActions])

    def learn(
        self,
        state: GameStateFeatures,
        action: str,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        """
        Update Q-values based on transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            nextState: Resulting state
        """
        current_q_value: float = self.getQValue(state, action)
        next_max_q_value: float = self.maxQValue(nextState)

        # Q-learning update formula
        new_q_value: float = current_q_value + self.alpha * (
            reward + self.gamma * next_max_q_value - current_q_value
        )

        # Update the value of state-action pair
        self.qLearner.setQValue(state, action, new_q_value)

    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        """
        Update visit counts for a state-action pair.

        Args:
            state: Current state
            action: Action taken
        """
        self.qLearner.incrementVisitCount(state, action)

    def getCount(self, state: GameStateFeatures, action: str) -> int:
        """
        Get visit count for a state-action pair.

        Args:
            state: Current state
            action: Action to check

        Returns:
            Visit count
        """
        return self.qLearner.getVisitCount(state, action)

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Compute exploration function value.

        Args:
            utility: Q-value
            counts: Visit count

        Returns:
            Exploration value
        """
        # If action never tried, strongly encourage exploration
        if counts == 0:
            return float("inf")

        # Otherwise balance exploration and exploitation
        return utility + (self.qLearner.exploration_k_value / (counts**0.5))

    def getExplorationAction(self, state: GameStateFeatures) -> str:
        """
        Returns an action based on exploration strategy, combining
        epsilon-greedy and count-based exploration

        Args:
            state: Current state

        Returns:
            Action to take
        """
        # With probability epsilon, choose a random action (pure exploration)
        if util.flipCoin(self.epsilon):
            return random.choice(state.legalActions)

        # Otherwise, use count-based exploration to balance exploration and exploitation
        return max(
            state.legalActions,
            key=lambda a: self.explorationFn(
                self.getQValue(state, a), self.getCount(state, a)
            ),
        )

    def getBestAction(self, state: GameStateFeatures) -> str:
        """
        Get the best action based on current Q-values, with tie-breaking

        Args:
            state: Current state

        Returns:
            Best action to take
        """
        if not state.legalActions:
            return Directions.STOP

        # Get all actions with maximum Q-value (might be multiple)
        qValues: List[Tuple[str, float]] = [
            (a, self.getQValue(state, a)) for a in state.legalActions
        ]

        # Use the QLearner's getBestAction method
        return self.qLearner.getBestAction(state, qValues)

    def getAction(self, state: GameState) -> str:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning.

        Args:
            state: The current game state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal: list = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if self.lastState is None:
            self.lastState = state
            # For the first action, create features and choose based on exploration
            currentStateFeatures = GameStateFeatures(state)

            # Use getExplorationAction directly which uses our explorationFn
            action = self.getExplorationAction(currentStateFeatures)

            self.lastAction = action
            return action

        lastStateFeatures = GameStateFeatures(self.lastState)
        currentStateFeatures = GameStateFeatures(state)

        # Use computeReward and learn directly
        reward: float = self.computeReward(self.lastState, state)
        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        # Choose action based on whether we're still training or not
        if self.getEpisodesSoFar() < self.getNumTraining():
            action: str = self.getExplorationAction(currentStateFeatures)
        else:
            action: str = self.getBestAction(currentStateFeatures)

        # Use updateCount directly
        self.updateCount(currentStateFeatures, action)

        # Remember current state and action for next step
        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState) -> None:
        """
        Handle the end of episodes. This is called by the game after a win or a loss.
        Updates Q-values and adjusts learning parameters.

        Args:
            state: The final game state
        """
        finalReward: float = self.computeReward(self.lastState, state)

        # Final update to Q-values using learn directly
        self.learn(
            GameStateFeatures(self.lastState),
            self.lastAction,
            finalReward,
            GameStateFeatures(state),
        )

        # Adjust epsilon and alpha based on training progress for better learning
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

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
