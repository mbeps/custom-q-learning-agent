# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

import random
from collections import defaultdict


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # in my GameStateFeatures class, I extract

        # the position of pacman
        self.pos = state.getPacmanPosition()

        # the position of ghost
        self.ghost_pos = tuple(state.getGhostPositions())

        # the coordinate of food
        self.food_map = state.getFood()

        # the legal actions that pacman can execute in this state (except for Stop)
        self.legalActions = state.getLegalPacmanActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

    # GameStateFeatures object will act as the key in dict,
    # so must override the __hash__ methond inherited from the object class
    def __hash__(self):
        return hash((self.pos, self.ghost_pos, self.food_map))

    def __eq__(self, other):
        return (
            self.pos == other.pos
            and self.ghost_pos == other.ghost_pos
            and self.food_map == other.food_map
        )


class QLearnAgent(Agent):
    def __init__(
        self,
        alpha: float = 0.2,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        maxAttempts: int = 30,
        numTraining: int = 10,
    ):
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
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)

        # Count the number of games we have played.
        self.episodesSoFar = 0

        # QValue is a dict which used to store the value of each state-action pair.
        self.QValue = defaultdict(float)

        # Store the last action pacman took, initialized by "West"
        self.lastAction = "West"

        # Store the last state where pacman stay in, initialized by None
        self.lastState = None

        # the dict visitedTimes is used to store the action times that pacman has been
        # taken in a specific state.
        self.visitedTimes = defaultdict(int)

        # the dict maxActionValue is used to store the max action value in a specific state.
        self.maxActionValue = defaultdict(int)

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.QValue[(state, action)]

    # Find the action which can get the most value from legalActions
    def getBestAction(self, state: GameStateFeatures):
        bestAction = state.legalActions[0]
        for a in state.legalActions:
            if self.getQValue(state, a) > self.getQValue(state, bestAction):
                bestAction = a
        return bestAction

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        # return the maximum estimated Q-value, note that if pacman
        # has reached the terminal state, the legal action is an empty list
        if not state.legalActions:
            return 0.0
        else:
            return max([self.getQValue(state, a) for a in state.legalActions])

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(
        self,
        state: GameStateFeatures,
        action: Directions,
        reward: float,
        nextState: GameStateFeatures,
    ):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # the value of last state-action pair
        QValue_last = self.getQValue(state, action)

        # the most value pacman can get in this state, here,
        # pacman emploies the greedy policy, so take the maximum
        maxQValue_current = self.maxQValue(nextState)

        # the iteration equation of Q learning algorithm
        new_QValue = QValue_last + self.getAlpha() * (
            reward + self.getGamma() * maxQValue_current - QValue_last
        )

        # update the new value of state-action pair
        self.QValue[(state, action)] = new_QValue

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self, state: GameStateFeatures, action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        self.visitedTimes[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.visitedTimes[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        if counts < self.getMaxAttempts():
            return self.maxActionValue[
                (GameStateFeatures(self.lastState), self.lastAction)
            ]
        else:
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Due to the lastState initialized by None, so in the first
        # episode we assign the current state to lastState (avoid null pointer exception)
        if self.lastState is None:
            self.lastState = state

        # Encapsulate lastState and state into GameStateFeatures class
        # which has already extracted features.
        lastStateFeatures = GameStateFeatures(self.lastState)
        currentStateFeatures = GameStateFeatures(state)

        # Compute the reward between lastState and current state
        reward = self.computeReward(self.lastState, state)

        # Update the Qvalue of last state and last action
        self.learn(lastStateFeatures, self.lastAction, reward, currentStateFeatures)

        # if the probabiliy is less than epsilon(0.05), executes exploration
        if util.flipCoin(self.epsilon):
            action = random.choice(currentStateFeatures.legalActions)
        # otherwise, pacman choose the best action from current state, that is, exploitation
        else:
            action = self.getBestAction(currentStateFeatures)

        # Store the current state for the next iteration
        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        reward = self.computeReward(self.lastState, state)

        # When pacman reaches terminal state,
        # update the value of last State-action pair in this episode
        self.learn(
            GameStateFeatures(self.lastState),
            self.lastAction,
            reward,
            GameStateFeatures(state),
        )

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
