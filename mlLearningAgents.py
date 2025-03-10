from __future__ import absolute_import
from __future__ import print_function

import random
import sys
from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Set, Tuple

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm.

    This implementation focuses on relational features (directions to objects)
    rather than just absolute positions to improve generalization.

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState) -> None:
        """
        Args:
            state: A given game state object
        """
        # Basic position information
        self.pacman: Tuple[int, int] = state.getPacmanPosition()
        self.ghost_pos: Tuple[Tuple[float, float], ...] = tuple(
            state.getGhostPositions()
        )
        self.walls = state.getWalls().asList()

        # Get food data
        food_grid = state.getFood()
        self.food = food_grid.asList()
        self.food_map = food_grid  # Keep the actual grid for comparison
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
        max_search_depth = 8  # Prevent searching too far
        min_dist = 1000  # Default minimum distance

        # Map surrounding directions with coordinates
        (x, y) = location
        direction_mapping = [
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

        min_direction = None

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
        for better state generalization
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

    def __eq__(self, other: Any) -> bool:
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
        Initialize the Q-learning algorithm parameters and data structures

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

        # Optimistic initialization to small positive values for better exploration
        self.QValue: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(
            lambda: 1.0
        )

        # Visit counter for state-action pairs
        self.visitedTimes: DefaultDict[Tuple[GameStateFeatures, str], int] = (
            defaultdict(int)
        )

        # Exploration constant
        self.exploration_k_value: float = 10.0

    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        """
        Returns the Q-value for a state-action pair

        Args:
            state: State features
            action: Action to take

        Returns:
            Q-value for the state-action pair
        """
        return self.QValue[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Returns the maximum Q-value attainable from the state

        Args:
            state: State features

        Returns:
            Maximum Q-value across all legal actions
        """
        if not state.legalActions:
            return 0.0

        # Return the highest Q-value among all legal actions
        return max([self.getQValue(state, a) for a in state.legalActions])

    def learn(
        self,
        state: GameStateFeatures,
        action: str,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        """
        Performs a Q-learning update using the standard Q-learning formula:
        Q(s,a) = Q(s,a) + α * [R + γ * max_a' Q(s',a') - Q(s,a)]

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
        self.QValue[(state, action)] = new_q_value

    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        """
        Updates the visitation count for a state-action pair

        Args:
            state: Current state
            action: Action taken
        """
        self.visitedTimes[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: str) -> int:
        """
        Returns the number of times an action has been taken in a state

        Args:
            state: Current state
            action: Action to check

        Returns:
            Number of visits to the state-action pair
        """
        return self.visitedTimes[(state, action)]

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes the exploration function value, combining both the current utility
        and an exploration bonus based on visit counts.

        Args:
            utility: Q-value for the state-action pair
            counts: Number of visits to the state-action pair

        Returns:
            Exploration value
        """
        # If action never tried, strongly encourage exploration
        if counts == 0:
            return float("inf")

        # Otherwise balance exploration and exploitation
        # Using square root to make the exploration bonus decay more slowly
        # as the state-action pair is visited more times
        return utility + (self.exploration_k_value / (counts**0.5))

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
        maxValue: float = max(qValues, key=lambda x: x[1])[1]
        bestActions: List[str] = [a for a, v in qValues if v == maxValue]

        # If multiple actions have the same value, break ties with exploration counts
        # Prefer less-explored actions for tie-breaking
        if len(bestActions) > 1:
            return min(bestActions, key=lambda a: self.getCount(state, a))

        return bestActions[0]

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

        # Otherwise, use count-based exploration to balance
        # exploration and exploitation
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
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
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
        self.episodesSoFar: int = 0

        # Create Q-learner instance to handle the learning algorithm
        self.qLearner: QLearner = QLearner(
            alpha=self.alpha,
            epsilon=self.epsilon,
            gamma=self.gamma,
            maxAttempts=self.maxAttempts,
        )

        # Store the last action pacman took, initialized to a default value
        self.lastAction: str = Directions.WEST

        # Store the last state where pacman was
        self.lastState: Optional[GameState] = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self) -> None:
        """Increment episode counter"""
        self.episodesSoFar += 1

    def getEpisodesSoFar(self) -> int:
        """Get the number of episodes completed"""
        return self.episodesSoFar

    def getNumTraining(self) -> int:
        """Get the number of training episodes"""
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float) -> None:
        """
        Set the exploration rate and update the Q-learner's epsilon too
        """
        self.epsilon = value
        self.qLearner.epsilon = value

    def getAlpha(self) -> float:
        """Get the learning rate"""
        return self.alpha

    def setAlpha(self, value: float) -> None:
        """
        Set the learning rate and update the Q-learner's alpha too
        """
        self.alpha = value
        self.qLearner.alpha = value

    def getGamma(self) -> float:
        """Get the discount factor"""
        return self.gamma

    def getMaxAttempts(self) -> int:
        """Get the maximum number of attempts for exploration"""
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Compute the reward for a transition as the score difference

        Args:
            startState: Starting game state
            endState: Ending game state

        Returns:
            Reward value
        """
        # Simple reward function that compares the score before and after the action
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Get the Q-value for a state-action pair

        Args:
            state: State features
            action: Action to take

        Returns:
            Q-value
        """
        return self.qLearner.getQValue(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Get the maximum Q-value for a state

        Args:
            state: State features

        Returns:
            Maximum Q-value
        """
        return self.qLearner.maxQValue(state)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(
        self,
        state: GameStateFeatures,
        action: Directions,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        """
        Update Q-values based on transition

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            nextState: Resulting state
        """
        self.qLearner.learn(state, action, reward, nextState)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self, state: GameStateFeatures, action: Directions) -> None:
        """
        Update visit counts for a state-action pair

        Args:
            state: Current state
            action: Action taken
        """
        self.qLearner.updateCount(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Get visit count for a state-action pair

        Args:
            state: Current state
            action: Action to check

        Returns:
            Visit count
        """
        return self.qLearner.getCount(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Compute exploration function value

        Args:
            utility: Q-value
            counts: Visit count

        Returns:
            Exploration value
        """
        return self.qLearner.explorationFn(utility, counts)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Initialize last state if this is the first action
        if self.lastState is None:
            self.lastState = state
            # For the first action, create features and choose based on exploration
            currentStateFeatures = GameStateFeatures(state)
            action = self.qLearner.getExplorationAction(currentStateFeatures)
            self.lastAction = action
            return action

        # Create feature representations
        lastStateFeatures = GameStateFeatures(self.lastState)
        currentStateFeatures = GameStateFeatures(state)

        # Calculate reward from last action
        reward: float = self.computeReward(self.lastState, state)

        # Update Q-values based on last action and reward
        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        # Choose action based on exploration strategy
        action: str = self.qLearner.getExplorationAction(currentStateFeatures)

        # Update visit counts
        self.updateCount(currentStateFeatures, action)

        # Remember current state and action for next step
        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState) -> None:
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Calculate final reward
        reward: float = self.computeReward(self.lastState, state)

        # Final update to Q-values
        self.learn(
            GameStateFeatures(self.lastState),
            self.lastAction,
            reward,
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

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
