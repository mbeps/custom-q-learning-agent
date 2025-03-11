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
        Initialise the game state features using the provided game state.

        Args:
            state: A game state object.
        """
        # Store Pacman position
        self.pacman: Tuple[int, int] = state.getPacmanPosition()

        # Store positions of ghosts as a tuple
        self.ghost_pos: Tuple[Tuple[float, float], ...] = tuple(state.getGhostPositions())

        # Get a list of wall coordinates
        self.walls: List[Tuple[int, int]] = state.getWalls().asList()

        # Extract food grid and derive useful food information
        food_grid: Grid[bool] = state.getFood()
        self.food: List[Tuple[int, int]] = food_grid.asList()
        self.food_map: Grid[bool] = food_grid
        self.food_count: int = food_grid.count()

        # Get legal Pacman actions and remove the STOP action if present
        self.legalActions: List[str] = state.getLegalPacmanActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

        # Find nearest food direction and distance
        (self.food_distance, self.food_direction) = self._findNearest(self.pacman, self.food)

        # Find nearest ghost direction and distance
        ghost_pos_list = list(self.ghost_pos)
        (self.ghost_distance, self.ghost_direction) = self._findNearest(self.pacman, ghost_pos_list)

        # Calculate relative distance difference if ghost and food are in same direction
        if self.ghost_direction != self.food_direction:
            self.ghost_to_food_distance = None
        else:
            self.ghost_to_food_distance = self.ghost_distance - self.food_distance

        # Calculate danger levels in each cardinal direction based on ghost positions
        self._calculateGhostDangers()

    def _calculateGhostDangers(self) -> None:
        """
        Calculate the danger level (measured as Manhattan distance) in each cardinal direction relative to Pacman's position.
        """
        x, y = self.pacman
        default_danger = 999  # A high default value for danger

        if self.ghost_pos:
            # Calculate danger in the north direction
            self.ghost_danger_north = min(
                [util.manhattanDistance((x, y + 1), ghost) for ghost in self.ghost_pos] + [default_danger]
            )
            # Calculate danger in the south direction
            self.ghost_danger_south = min(
                [util.manhattanDistance((x, y - 1), ghost) for ghost in self.ghost_pos] + [default_danger]
            )
            # Calculate danger in the east direction
            self.ghost_danger_east = min(
                [util.manhattanDistance((x + 1, y), ghost) for ghost in self.ghost_pos] + [default_danger]
            )
            # Calculate danger in the west direction
            self.ghost_danger_west = min(
                [util.manhattanDistance((x - 1, y), ghost) for ghost in self.ghost_pos] + [default_danger]
            )

    def _findNearest(
        self,
        location: Tuple[int, int],
        objects_to_compare: List[Tuple[int, int]],
        distance: int = 0,
        visited: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[int, Optional[str]]:
        """
        Find the nearest object to a given location using Breadth-First Search (BFS).

        Args:
            location: The starting coordinate (x, y).
            objects_to_compare: List of target object coordinates.
            distance: Current distance from the starting location (used recursively).
            visited: List of coordinates already visited (used to avoid cycles).

        Returns:
            A tuple containing the distance to the nearest object and the first direction to move in.
        """
        max_search_depth: int = 8  # Maximum allowed search depth to prevent excessive search
        min_dist: int = 1000       # A high default distance

        (x, y) = location

        # Map each cardinal direction to the corresponding coordinate change
        direction_mapping: List[Tuple[Tuple[int, int], str]] = [
            ((x, y + 1), Directions.NORTH),
            ((x, y - 1), Directions.SOUTH),
            ((x + 1, y), Directions.EAST),
            ((x - 1, y), Directions.WEST),
        ]

        # If maximum search depth is reached or current location is one of the target objects, return current distance
        if distance == max_search_depth or location in objects_to_compare:
            return distance, None

        if visited is None:
            visited = [location]

        min_direction: Optional[str] = None

        # Recursively explore each direction
        for coord, direction in direction_mapping:
            # Only explore if the coordinate is not a wall and hasn't been visited
            if coord not in self.walls and coord not in visited:
                (current_dist, _) = self._findNearest(coord, objects_to_compare, distance + 1, visited + [coord])
                if current_dist < min_dist and current_dist != max_search_depth:
                    min_dist = current_dist
                    min_direction = direction

        return min_dist, min_direction

    def __hash__(self) -> int:
        """
        Custom hash function based on relational features rather than absolute positions.
        This helps with state generalisation in the Q-learning algorithm.
        """
        return hash(
            (
                self.pacman,                    # Pacman position ensures uniqueness
                self.ghost_direction,           # Relative ghost direction
                self.food_direction,            # Relative food direction
                self.ghost_to_food_distance,    # Relative distance between ghost and food
                self.food_count,                # Total food count for further state distinction
            )
        )

    def __eq__(self, other) -> bool:
        """
        Define equality based on relational features rather than exact coordinates.

        Args:
            other: Another GameStateFeatures instance.

        Returns:
            True if both states have equivalent features, otherwise False.
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
    Q-learning agent for playing Pacman. Implements the Q-learning algorithm to learn optimal policies based on experience.
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
            alpha: Learning rate.
            epsilon: Exploration rate.
            gamma: Discount factor.
            maxAttempts: Number of attempts for each action in a state.
            numTraining: Number of training episodes.
        """
        super().__init__()
        self.alpha: float = float(alpha)
        self.epsilon: float = float(epsilon)
        self.gamma: float = float(gamma)
        self.maxAttempts: int = int(maxAttempts)
        self.numTraining: int = int(numTraining)

        # Store initial exploration and learning rates for adaptive adjustments later
        self.initial_epsilon: float = float(epsilon)
        self.initial_alpha: float = float(alpha)

        # Count the number of episodes played
        self.episodesCount: int = 0

        # Initialise Q-values with an optimistic value of 1.0 and visit counts to 0
        self.QValues: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(lambda: 1.0)
        self.visitCounts: DefaultDict[Tuple[GameStateFeatures, str], int] = defaultdict(int)
        self.exploration_k_value: float = 10.0

        # Variables to store the last game state and action taken
        self.lastState: Optional[GameState] = None
        self.lastAction: str = Directions.WEST

    def computeReward(self, startState: GameState, endState: GameState) -> float:
        """
        Compute the reward for a transition based on the difference in score.

        Args:
            startState: The game state before the action.
            endState: The game state after the action.

        Returns:
            The reward calculated as the score difference.
        """
        return endState.getScore() - startState.getScore()

    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        """
        Retrieve the Q-value for the given state-action pair.

        Args:
            state: The feature representation of the state.
            action: The action being considered.

        Returns:
            The Q-value associated with the state-action pair.
        """
        return self.QValues[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Find the maximum Q-value across all legal actions in the given state.

        Args:
            state: The feature representation of the state.

        Returns:
            The maximum Q-value, or 0.0 if there are no legal actions.
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
        Update the Q-value for a given state-action pair based on the reward received and next state.

        Args:
            state: The previous state features.
            action: The action taken.
            reward: The reward received from the transition.
            nextState: The feature representation of the new state.
        """
        current_q_value: float = self.getQValue(state, action)
        next_max_q_value: float = self.maxQValue(nextState)

        # Apply the Q-learning update formula
        new_q_value: float = current_q_value + self.alpha * (
            reward + self.gamma * next_max_q_value - current_q_value
        )

        # Store the updated Q-value
        self.QValues[(state, action)] = new_q_value

    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        """
        Increment the visit count for the specified state-action pair.

        Args:
            state: The feature representation of the state.
            action: The action taken.
        """
        self.visitCounts[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: str) -> int:
        """
        Retrieve the visit count for a specific state-action pair.

        Args:
            state: The feature representation of the state.
            action: The action in question.

        Returns:
            The number of times the state-action pair has been visited.
        """
        return self.visitCounts[(state, action)]

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Calculate the exploration function value, balancing exploration and exploitation.

        Args:
            utility: The current Q-value of the action.
            counts: The number of times this action has been taken in the state.

        Returns:
            A value that promotes actions with less visitation when counts are low.
        """
        # If the action has never been tried, encourage exploration strongly
        if counts == 0:
            return float("inf")
        
        # Otherwise balance exploration and exploitation
        return utility + (self.exploration_k_value / (counts**0.5))

    def _selectActionWithExploration(self, state: GameStateFeatures) -> str:
        """
        Choose an action using a combination of epsilon-greedy and count-based exploration.

        Args:
            state: The current state's feature representation.

        Returns:
            The action selected either randomly (exploration) or based on the exploration function.
        """
        # With probability epsilon, choose a random legal action for pure exploration
        if util.flipCoin(self.epsilon):
            return random.choice(state.legalActions)

        # Otherwise, choose the action with the highest exploration function value
        return max(
            state.legalActions,
            key=lambda a: self.explorationFn(self.getQValue(state, a), self.getCount(state, a)),
        )

    def _selectBestAction(self, state: GameStateFeatures) -> str:
        """
        Select the best action based solely on the current Q-values with tie-breaking based on visit counts.

        Args:
            state: The current state's feature representation.

        Returns:
            The action with the highest Q-value or the one with the lower visit count in case of ties.
        """
        if not state.legalActions:
            return Directions.STOP

        # Generate list of tuples (action, Q-value) for all legal actions
        qValues: List[Tuple[str, float]] = [(a, self.getQValue(state, a)) for a in state.legalActions]

        if not qValues:
            return Directions.STOP

        # Determine the maximum Q-value
        maxValue: float = max(qValues, key=lambda x: x[1])[1]

        # Identify all actions that have the maximum Q-value
        bestActions: List[str] = [a for a, v in qValues if v == maxValue]

        # If there is a tie, select the action that has been tried the fewest times
        if len(bestActions) > 1:
            return min(bestActions, key=lambda a: self.getCount(state, a))

        return bestActions[0]

    def _updateLearningParameters(self) -> None:
        """
        Update the learning parameters (epsilon and alpha) based on the number of episodes completed.
        """
        if self.episodesCount < self.numTraining:
            # Linearly decrease epsilon from its initial value to a minimum of 0.01 during training
            progress: float = self.episodesCount / self.numTraining
            new_epsilon: float = max(0.01, self.initial_epsilon * (1.0 - progress))
            self.epsilon = new_epsilon

            # Similarly, adjust alpha based on the training progress
            new_alpha: float = max(0.1, self.initial_alpha * (1.0 - progress * 0.5))
            self.alpha = new_alpha

            sys.stdout.write(f"\rEpisode {self.episodesCount + 1}/{self.numTraining}")
            sys.stdout.flush()
        elif self.episodesCount == self.numTraining:
            # Once training is complete, disable further learning
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.alpha = 0
            self.epsilon = 0

    def getAction(self, state: GameState) -> str:
        """
        Determine the action to take for the current game state.

        Args:
            state: The current game state.

        Returns:
            The chosen action to maximise reward while ensuring adequate exploration.
        """
        # Convert the current state into its feature representation
        currentStateFeatures = GameStateFeatures(state)

        # For the very first action, simply select an action with exploration
        if self.lastState is None:
            self.lastState = state
            action = self._selectActionWithExploration(currentStateFeatures)
            self.lastAction = action
            return action

        # Compute the reward from the previous transition and learn from it
        lastStateFeatures = GameStateFeatures(self.lastState)
        reward: float = self.computeReward(self.lastState, state)
        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        # Choose the next action: if still training, use exploration; otherwise, choose the best action
        if self.episodesCount < self.numTraining:
            action: str = self._selectActionWithExploration(currentStateFeatures)
        else:
            action: str = self._selectBestAction(currentStateFeatures)

        # Update the visit count for the chosen action
        self.updateCount(currentStateFeatures, action)

        # Save the current state and action for the next iteration
        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState) -> None:
        """
        Finalise the episode by learning from the last transition and updating learning parameters.

        Args:
            state: The final game state at the end of the episode.
        """
        # If a previous state exists, learn from the final transition
        if self.lastState is not None:
            finalReward: float = self.computeReward(self.lastState, state)
            self.learn(
                GameStateFeatures(self.lastState),
                self.lastAction,
                finalReward,
                GameStateFeatures(state),
            )

        # Increment the episode counter
        self.episodesCount += 1

        # Update learning parameters for subsequent episodes
        self._updateLearningParameters()

        # Reset lastState for the next episode
        self.lastState = None