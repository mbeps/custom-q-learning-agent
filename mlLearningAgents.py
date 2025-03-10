from __future__ import absolute_import, print_function

import random
import sys
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

from pacman import Directions, GameState
from pacman_utils import util
from pacman_utils.game import Agent, Grid

# --------------------------------------------------------------------------------
# Class: GameStateFeatures
# This class encapsulates the features of a given game state for the Q-learning agent.
# It extracts and processes relevant data from the GameState (such as positions of Pacman,
# ghosts, walls, and food) to provide a more generalised representation. This supports
# better state abstraction in reinforcement learning.
# --------------------------------------------------------------------------------
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
        # Get the current position of Pacman and positions of ghosts and walls.
        self.pacman_position: Tuple[int, int] = state.getPacmanPosition()
        self.ghost_positions: Tuple[Tuple[float, float], ...] = tuple(state.getGhostPositions())
        self.wall_positions: list = state.getWalls().asList()

        # Get food data
        # Retrieve the grid of food and convert it to a list for processing.
        food_grid: Grid[bool] = state.getFood()
        self.food_positions: list = food_grid.asList()
        self.food_grid: Grid[bool] = food_grid
        self.food_count: int = food_grid.count()

        # Legal actions excluding STOP
        # Obtain the legal actions Pacman can take and remove the STOP action to encourage movement.
        self.legal_actions: List[str] = state.getLegalPacmanActions()
        if Directions.STOP in self.legal_actions:
            self.legal_actions.remove(Directions.STOP)

        # Find nearest food direction and distance using BFS
        # This uses a Breadth-First Search (BFS) to determine the distance and the direction
        # Pacman should move to reach the closest food. BFS guarantees finding the shortest path.
        (self.food_distance, self.food_direction) = self.findNearest(self.pacman_position, self.food_positions)

        # Find nearest ghost direction and distance using BFS
        # Similarly, compute the direction and distance to the nearest ghost to inform avoidance strategies.
        ghost_positions_list = list(self.ghost_positions)
        (self.ghost_distance, self.ghost_direction) = self.findNearest(self.pacman_position, ghost_positions_list)

        # Measure relative positioning of ghost and food
        # If the ghost and food are in the same direction, compute their distance difference.
        # Otherwise, set to None to indicate no direct alignment.
        if self.ghost_direction != self.food_direction:
            self.ghost_to_food_distance = None
        else:
            self.ghost_to_food_distance = self.ghost_distance - self.food_distance

        # Detailed ghost danger in each direction (useful for immediate decision making)
        # Calculate the Manhattan distance to ghosts from neighbouring positions, which serves as a measure
        # of "danger" in each direction. A lower distance indicates higher immediate threat.
        x, y = self.pacman_position
        if self.ghost_positions:
            self.ghost_danger_north: float = min(
                [util.manhattanDistance((x, y + 1), ghost) for ghost in self.ghost_positions] + [999]  # Default high value
            )
            self.ghost_danger_south: float = min(
                [util.manhattanDistance((x, y - 1), ghost) for ghost in self.ghost_positions] + [999]
            )
            self.ghost_danger_east: float = min(
                [util.manhattanDistance((x + 1, y), ghost) for ghost in self.ghost_positions] + [999]
            )
            self.ghost_danger_west: float = min(
                [util.manhattanDistance((x - 1, y), ghost) for ghost in self.ghost_positions] + [999]
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
            objects_to_compare: List of object locations to find (e.g. food or ghost positions)
            distance: Current distance (used internally for recursive BFS)
            visited: List of already visited locations to avoid cycles

        Returns:
            Tuple of (distance to nearest object, direction to move)
        
        Explanation:
            - Implements a recursive Breadth-First Search (BFS) approach.
            - 'max_search_depth' limits the search radius.
            - If the current location is one of the target objects or the search depth is exceeded,
              the search terminates.
            - The function returns both the distance and the first direction taken from the starting position.
        """
        max_search_depth: int = 8  # Prevent searching too far
        min_dist: int = 1000  # Default minimum distance

        # Map surrounding directions with coordinates (North, South, West, East)
        (x, y) = location
        direction_mapping: List[Tuple[Tuple[int, int], str]] = [
            ((x, y + 1), Directions.NORTH),
            ((x, y - 1), Directions.SOUTH),
            ((x - 1, y), Directions.WEST),
            ((x + 1, y), Directions.EAST),
        ]

        # If we've found an object or reached max depth, return current distance and no direction.
        if distance == max_search_depth or location in objects_to_compare:
            return distance, None

        if visited is None:
            visited = [location]

        min_direction: Optional[str] = None

        # Explore neighbouring coordinates recursively
        for coord, direction in direction_mapping:
            if coord not in self.wall_positions and coord not in visited:
                (current_dist, _) = self.findNearest(coord, objects_to_compare, distance + 1, visited + [coord])
                if current_dist < min_dist and current_dist != max_search_depth:
                    min_dist = current_dist
                    min_direction = direction

        return min_dist, min_direction

    def __hash__(self) -> int:
        """
        Hash function based on relational features rather than just absolute positions
        for better state generalisation

        Explanation:
            - Hashing is important for using states as keys in dictionaries (e.g. Q-value lookups).
            - This function combines features like Pacman's position, ghost and food directions,
              and food count to uniquely identify a game state.
        """
        return hash(
            (
                self.pacman_position,  # Keep pacman position for uniqueness
                self.ghost_direction,
                self.food_direction,
                self.ghost_to_food_distance,
                self.food_count,  # Add food count for better state distinction
            )
        )

    def __eq__(self, other: GameState) -> bool:
        """
        Equality based on relational features rather than exact coordinates

        Args:
            other: Another GameStateFeatures object

        Returns:
            True if states are equivalent, False otherwise
        
        Explanation:
            - This allows the Q-learning algorithm to recognise states as identical even if the
              absolute positions differ but the relational features (e.g. direction to food) are the same.
        """
        if not isinstance(other, GameStateFeatures):
            return False

        return (
            self.pacman_position == other.pacman_position
            and self.ghost_direction == other.ghost_direction
            and self.food_direction == other.food_direction
            and self.ghost_to_food_distance == other.ghost_to_food_distance
            and self.food_count == other.food_count
        )


# --------------------------------------------------------------------------------
# Class: QLearner
# This class implements the core Q-learning algorithm.
# It stores Q-values for state-action pairs and updates them based on rewards.
# It also implements an exploration strategy to balance between exploring new actions
# and exploiting known good actions.
# --------------------------------------------------------------------------------
class QLearner:
    """Handles the Q-learning algorithm logic"""

    def __init__(self, alpha: float, epsilon: float, gamma: float, maxAttempts: int) -> None:
        """
        Initialise the Q-learning algorithm parameters and data structures

        Args:
            alpha: Learning rate (how quickly the Q-value is updated)
            epsilon: Exploration rate (probability of taking a random action)
            gamma: Discount factor (importance of future rewards)
            maxAttempts: Maximum number of attempts for exploration bonus (limits count-based exploration)
        
        Explanation:
            - Optimistic initialisation (default value of 1.0) is used to encourage exploration.
            - Q-values and visit counts are stored in dictionaries for efficient lookup.
        """
        # Learning hyperparameters
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.maxAttempts: int = maxAttempts

        # Optimistic initialisation to small positive values for better exploration
        self.q_values: DefaultDict[Tuple[GameStateFeatures, str], float] = defaultdict(lambda: 1.0)
        self.visited_times: DefaultDict[Tuple[GameStateFeatures, str], int] = defaultdict(int)
        self.exploration_k_value: float = 10.0

    def getQValue(self, state: GameStateFeatures, action: str) -> float:
        """
        Returns the Q-value for a state-action pair

        Args:
            state: State features
            action: Action to take

        Returns:
            Q-value for the state-action pair
        
        Explanation:
            - This function retrieves the Q-value, which estimates the expected future reward
              when taking a specific action in a given state.
        """
        return self.q_values[(state, action)]

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Returns the maximum Q-value attainable from the state

        Args:
            state: State features

        Returns:
            Maximum Q-value across all legal actions
        
        Explanation:
            - This is used to estimate the value of the best possible action from the current state,
              which is a key component in the Q-learning update formula.
        """
        if not state.legal_actions:
            return 0.0

        return max([self.getQValue(state, a) for a in state.legal_actions])

    def learn(
        self,
        state: GameStateFeatures,
        action: str,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        """
        Performs a Q-learning update using the standard Q-learning formula:
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state features
            action: Action taken
            reward: Reward received after taking the action
            nextState: State features after the action
        
        Explanation:
            - The update adjusts the Q-value based on the observed reward and the estimated
              maximum future reward (using the discount factor gamma).
        """
        current_q_value: float = self.getQValue(state, action)
        next_max_q_value: float = self.maxQValue(nextState)

        # Q-learning update formula
        new_q_value: float = current_q_value + self.alpha * (reward + self.gamma * next_max_q_value - current_q_value)

        # Update the value of state-action pair
        self.q_values[(state, action)] = new_q_value

    def updateCount(self, state: GameStateFeatures, action: str) -> None:
        """
        Updates the visitation count for a state-action pair

        Args:
            state: Current state features
            action: Action taken
        
        Explanation:
            - Visit counts are used in the exploration function to encourage the agent
              to try actions that have not been explored extensively.
        """
        self.visited_times[(state, action)] += 1

    def getCount(self, state: GameStateFeatures, action: str) -> int:
        """
        Returns the number of times an action has been taken in a state

        Args:
            state: Current state features
            action: Action to check

        Returns:
            Number of visits to the state-action pair
        """
        return self.visited_times[(state, action)]

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes the exploration function value, combining both the current utility
        and an exploration bonus based on visit counts.
        If an action is never tried, it is strongly encouraged to explore it.
        Otherwise, the exploration bonus decays as the state-action pair is visited more.

        Args:
            utility: Q-value for the state-action pair
            counts: Number of times the action has been tried in the state

        Returns:
            Combined value of the Q-value and exploration bonus
        
        Explanation:
            - This function implements an epsilon-greedy strategy with a bonus that decays
              with the square root of the number of visits. This ensures that rarely taken actions
              are given an extra incentive to be tried.
        """
        # If action never tried, strongly encourage exploration
        if counts == 0:
            return float('inf')

        # Otherwise balance exploration and exploitation
        return utility + (self.exploration_k_value / (counts ** 0.5))

    def getBestAction(self, state: GameStateFeatures) -> str:
        """
        Get the best action based on current Q-values, with tie-breaking

        Args:
            state: Current state features

        Returns:
            Best action to take
        
        Explanation:
            - If multiple actions yield the same maximum Q-value, the one with fewer prior visits
              is chosen to promote exploration.
        """
        if not state.legal_actions:
            return Directions.STOP

        # Get all actions with maximum Q-value (might be multiple)
        qValues: List[Tuple[str, float]] = [(a, self.getQValue(state, a)) for a in state.legal_actions]
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
            state: Current state features

        Returns:
            Action to take
        
        Explanation:
            - With probability epsilon, a random action is chosen to explore.
            - Otherwise, the action with the highest exploration function value is selected.
        """
        # With probability epsilon, choose a random action (pure exploration)
        if util.flipCoin(self.epsilon):
            return random.choice(state.legal_actions)

        # Otherwise, use count-based exploration to balance exploration and exploitation
        return max(
            state.legal_actions,
            key=lambda a: self.explorationFn(self.getQValue(state, a), self.getCount(state, a)),
        )


# --------------------------------------------------------------------------------
# Class: QLearnAgent
# This class integrates the QLearner into an agent that interacts with the Pacman game.
# It manages episodes, training, and updates the Q-values using transitions observed during gameplay.
# --------------------------------------------------------------------------------
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
        
        Explanation:
            - The agent stores both the current and initial values of alpha and epsilon
              for adaptive learning. This allows a gradual reduction in exploration as training progresses.
        """
        super().__init__()
        self.alpha: float = float(alpha)
        self.epsilon: float = float(epsilon)
        self.gamma: float = float(gamma)
        self.maxAttempts: int = int(maxAttempts)
        self.num_training: int = int(numTraining)

        # Store initial values for adaptive parameters
        self.initial_epsilon: float = float(epsilon)
        self.initial_alpha: float = float(alpha)

        # Count the number of games we have played
        self.episodes_count: int = 0

        # Create Q-learner instance to handle the learning algorithm
        self.qLearner: QLearner = QLearner(
            alpha=self.alpha,
            epsilon=self.epsilon,
            gamma=self.gamma,
            maxAttempts=self.maxAttempts,
        )

        # Store the last action pacman took, initialised to a default value
        self.last_action: str = Directions.WEST

        # Store the last state where pacman was
        self.last_state: Optional[GameState] = None

    def incrementEpisodesSoFar(self) -> None:
        """
        Increment the counter for episodes completed.
        
        Explanation:
            - Each episode represents a complete game run (either a win or a loss).
            - This counter is used to determine training progress.
        """
        self.episodes_count += 1

    def getEpisodesSoFar(self) -> int:
        """
        Get the number of episodes completed.

        Returns:
            Number of episodes completed
        """
        return self.episodes_count

    def getNumTraining(self) -> int:
        """
        Get the number of training episodes.

        Returns:
            Number of training episodes
        """
        return self.num_training

    def setEpsilon(self, value: float) -> None:
        """
        Set the exploration rate and update the Q-learner's epsilon too.

        Args:
            value: New exploration rate
        
        Explanation:
            - Adjusting epsilon allows the agent to reduce randomness as it learns.
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
        
        Explanation:
            - Lowering alpha over time can help stabilise learning once the agent has explored enough.
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

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Compute the reward for a transition as the score difference.

        Args:
            startState: Starting game state
            endState: Ending game state

        Returns:
            Reward value
        
        Explanation:
            - This simple reward function assumes that an increase in score reflects a positive outcome.
        """
        return endState.getScore() - startState.getScore()

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Get the Q-value for a state-action pair.

        Args:
            state: State features
            action: Action to take

        Returns:
            Q-value
        
        Explanation:
            - This is a wrapper to the underlying QLearner's method.
        """
        return self.qLearner.getQValue(state, action)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Get the maximum Q-value for a state.

        Args:
            state: State features

        Returns:
            Maximum Q-value
        
        Explanation:
            - Retrieves the best possible value from the current state by considering all legal actions.
        """
        return self.qLearner.maxQValue(state)

    def learn(
        self,
        state: GameStateFeatures,
        action: Directions,
        reward: float,
        nextState: GameStateFeatures,
    ) -> None:
        """
        Update Q-values based on transition.

        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            nextState: State features after the action
        
        Explanation:
            - This function encapsulates the learning update process as defined by the Q-learning algorithm.
        """
        self.qLearner.learn(state, action, reward, nextState)

    def updateCount(self, state: GameStateFeatures, action: Directions) -> None:
        """
        Update visit counts for a state-action pair.

        Args:
            state: Current state features
            action: Action taken
        
        Explanation:
            - Keeping track of how often an action is taken in a particular state supports the exploration bonus.
        """
        self.qLearner.updateCount(state, action)

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Get visit count for a state-action pair.

        Args:
            state: Current state features
            action: Action to check

        Returns:
            Visit count
        """
        return self.qLearner.getCount(state, action)

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Compute exploration function value.

        Args:
            utility: Q-value for the state-action pair
            counts: Visit count

        Returns:
            Exploration value
        
        Explanation:
            - This wrapper combines the current Q-value with an exploration bonus,
              aiding in the decision-making process.
        """
        return self.qLearner.explorationFn(utility, counts)

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning.

        Args:
            state: The current game state

        Returns:
            The action to take
        
        Explanation:
            - For the first move, the agent relies solely on exploration.
            - For subsequent moves, the agent updates Q-values based on the reward received
              and then chooses the next action using the exploration strategy.
        """
        # The data we have about the state of the game
        legal: list = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if self.last_state is None:
            self.last_state = state
            # For the first action, create features and choose based on exploration
            currentStateFeatures = GameStateFeatures(state)
            action = self.qLearner.getExplorationAction(currentStateFeatures)
            self.last_action = action
            return action

        # Create feature representations for the last and current states
        lastStateFeatures = GameStateFeatures(self.last_state)
        currentStateFeatures = GameStateFeatures(state)
        # Compute reward based on the change in state (score difference)
        reward: float = self.computeReward(self.last_state, state)
        # Update Q-values with the observed transition
        self.learn(lastStateFeatures, self.last_action, reward, currentStateFeatures)
        # Choose the next action using the exploration strategy
        action: str = self.qLearner.getExplorationAction(currentStateFeatures)
        self.updateCount(currentStateFeatures, action)

        # Remember current state and action for next step
        self.last_state = state
        self.last_action = action

        return action

    def final(self, state: GameState) -> None:
        """
        Handle the end of episodes. This is called by the game after a win or a loss.
        Updates Q-values and adjusts learning parameters.

        Args:
            state: The final game state
        
        Explanation:
            - At the end of an episode, a final reward is computed and a last Q-value update is performed.
            - The agent then adapts the exploration rate (epsilon) and learning rate (alpha) based on training progress.
            - When training is complete, both epsilon and alpha are set to 0 to stop further learning.
        """
        finalReward: float = self.computeReward(self.last_state, state)

        # Final update to Q-values
        self.learn(
            GameStateFeatures(self.last_state),
            self.last_action,
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

            sys.stdout.write(f"\rEpisode {self.getEpisodesSoFar() + 1}/{self.getNumTraining()}")
            sys.stdout.flush()

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)