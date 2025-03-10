from __future__ import absolute_import, print_function

import random
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

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
        self.walls: List[Tuple[int, int]] = state.getWalls().asList()

        # Get food data
        food_grid: Grid[bool] = state.getFood()
        self.food: List[Tuple[int, int]] = food_grid.asList()
        self.food_map: Grid[bool] = food_grid
        self.food_count: int = food_grid.count()

        # Legal actions excluding STOP
        self.legalActions: List[str] = state.getLegalPacmanActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

        # Find nearest food direction and distance using BFS
        (self.food_distance, self.food_direction) = self._findNearest(
            self.pacman, self.food
        )

        # Find nearest ghost direction and distance using BFS
        ghost_pos_list = list(self.ghost_pos)
        (self.ghost_distance, self.ghost_direction) = self._findNearest(
            self.pacman, ghost_pos_list
        )

        # Measure relative positioning of ghost and food
        if self.ghost_direction != self.food_direction:
            self.ghost_to_food_distance = None
        else:
            self.ghost_to_food_distance = self.ghost_distance - self.food_distance

        # Calculate danger in each direction if there are ghosts
        self._calculateGhostDangers()

    def _calculateGhostDangers(self) -> None:
        """
        Calculate the danger levels in each cardinal direction based on ghost positions.
        """
        x, y = self.pacman
        default_danger = 999  # Default high value

        if self.ghost_pos:
            # Calculate danger in each direction
            self.ghost_danger_north = min(
                [util.manhattanDistance((x, y + 1), ghost) for ghost in self.ghost_pos]
                + [default_danger]
            )
            self.ghost_danger_south = min(
                [util.manhattanDistance((x, y - 1), ghost) for ghost in self.ghost_pos]
                + [default_danger]
            )
            self.ghost_danger_east = min(
                [util.manhattanDistance((x + 1, y), ghost) for ghost in self.ghost_pos]
                + [default_danger]
            )
            self.ghost_danger_west = min(
                [util.manhattanDistance((x - 1, y), ghost) for ghost in self.ghost_pos]
                + [default_danger]
            )

    def _findNearest(
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

        if distance == max_search_depth or location in objects_to_compare:
            return distance, None

        if visited is None:
            visited = [location]

        min_direction: Optional[str] = None

        for coord, direction in direction_mapping:
            if coord not in self.walls and coord not in visited:
                (current_dist, _) = self._findNearest(
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

        # Q-values and visit counts
        self.QValues: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(
            lambda: 1.0  # Optimistic initialisation
        )
        self.visitCounts: DefaultDict[Tuple[GameStateFeatures, str], int] = defaultdict(
            int
        )
        self.exploration_k_value: float = 10.0

        # Store the last state and action
        self.lastState: Optional[GameState] = None
        self.lastAction: str = Directions.WEST

    def computeReward(self, startState: GameState, endState: GameState) -> float:
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
        return self.QValues[(state, action)]

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
        self.QValues[(state, action)] = new_q_value

    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        """
        Update visit counts for a state-action pair.

        Args:
            state: Current state
            action: Action taken
        """
        self.visitCounts[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: str) -> int:
        """
        Get visit count for a state-action pair.

        Args:
            state: Current state
            action: Action to check

        Returns:
            Visit count
        """
        return self.visitCounts[(state, action)]

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
        return utility + (self.exploration_k_value / (counts**0.5))

    def _selectActionWithExploration(self, state: GameStateFeatures) -> str:
        """
        Select an action using an exploration strategy that combines
        epsilon-greedy and count-based exploration.

        Args:
            state: Current state features

        Returns:
            Selected action
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

    def _selectBestAction(self, state: GameStateFeatures) -> str:
        """
        Select the best action based on current Q-values with tie-breaking.

        Args:
            state: Current state features

        Returns:
            Best action
        """
        if not state.legalActions:
            return Directions.STOP

        # Find actions with the maximum Q-value
        qValues: List[Tuple[str, float]] = [
            (a, self.getQValue(state, a)) for a in state.legalActions
        ]

        if not qValues:
            return Directions.STOP

        # Find maximum Q-value
        maxValue: float = max(qValues, key=lambda x: x[1])[1]

        # Get all actions with the maximum value
        bestActions: List[str] = [a for a, v in qValues if v == maxValue]

        # If multiple actions have the same value, break ties with exploration counts
        if len(bestActions) > 1:
            return min(bestActions, key=lambda a: self.getCount(state, a))

        return bestActions[0]

    def _updateLearningParameters(self) -> None:
        """
        Update learning parameters (epsilon, alpha) based on training progress.
        """
        if self.episodesCount < self.numTraining:
            # Linearly decrease epsilon from initial value to 0.01
            progress: float = self.episodesCount / self.numTraining
            new_epsilon: float = max(0.01, self.initial_epsilon * (1.0 - progress))
            self.epsilon = new_epsilon

            # Also adapt alpha based on training progress
            new_alpha: float = max(0.1, self.initial_alpha * (1.0 - progress * 0.5))
            self.alpha = new_alpha

            sys.stdout.write(f"\rEpisode {self.episodesCount + 1}/{self.numTraining}")
            sys.stdout.flush()
        elif self.episodesCount == self.numTraining:
            # Turn off learning when training is complete
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.alpha = 0
            self.epsilon = 0

    def getAction(self, state: GameState) -> str:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning.

        Args:
            state: The current game state

        Returns:
            The action to take
        """
        # Create feature representation of current state
        currentStateFeatures = GameStateFeatures(state)

        # Handle first action case
        if self.lastState is None:
            self.lastState = state
            action = self._selectActionWithExploration(currentStateFeatures)
            self.lastAction = action
            return action

        # Process previous action's outcome and learn from it
        lastStateFeatures = GameStateFeatures(self.lastState)
        reward: float = self.computeReward(self.lastState, state)
        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        # Choose action based on training phase
        if self.episodesCount < self.numTraining:
            action: str = self._selectActionWithExploration(currentStateFeatures)
        else:
            action: str = self._selectBestAction(currentStateFeatures)

        # Update visit count for the selected action
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
        # Learn from the final transition
        if self.lastState is not None:
            finalReward: float = self.computeReward(self.lastState, state)
            self.learn(
                GameStateFeatures(self.lastState),
                self.lastAction,
                finalReward,
                GameStateFeatures(state),
            )

        # Increment episode counter
        self.episodesCount += 1

        # Update learning parameters for next episode
        self._updateLearningParameters()

        # Reset for next episode
        self.lastState = None
