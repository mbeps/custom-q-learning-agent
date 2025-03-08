# mlLearningCraigAgents.py
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

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Represents relevant features of a game state that can be used to identify common state situations
    and generalize for all game states, not relying on exact locations of all objects.

    We use pacman location relative to food and ghosts to help generalize so can be used for games of
    all sizes and configurations.
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        # get pacman position
        self.pacman = state.getPacmanPosition()
        # get the ghost positions
        self.ghost = tuple(state.getGhostPositions())
        # get all the wall positions
        self.walls = state.getWalls().asList()
        # get all the food positions
        self.food = state.getFood().asList()
        # save legal actions, removing Stop
        self.legalActions = state.getLegalActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

        # find nearest food direction and distance from pacman
        (distanceToFood, self.foodDirection) = self.findNearest(self.pacman, self.food)
        # find nearest ghost direction and distance from pacman
        (distanceToGhost, self.ghostDirection) = self.findNearest(self.pacman, self.ghost)

        # measure distance between nearest ghost and nearest food, when direction is the same
        if self.ghostDirection != self.foodDirection:
            self.ghostToFoodDistance = None
        else:
            self.ghostToFoodDistance = distanceToGhost - distanceToFood

    def findNearest(self, location, objectsToCompare, distance=0, visited=None):
        """
        Find the nearest object to the given location
        e.g. food or ghost
        :param location: location to compare e.g (2,1)
        :param objectsToCompare: object to compare with location, e.g. food or ghost
        :param distance: distance so far, used internally
        :param visited: list of locations already visited to avoid revisiting, used internally
        :return: a tuple of distance to object, and closest direction to reach that object (distance, direction)
        """
        maxSearchDepth = 8  # prevent searching further than max search
        minDist = 1000  # default minimum distance

        # map surrounding directions with coordinates
        (x, y) = location
        directionMapping = [((x, y + 1), Directions.NORTH), ((x, y - 1), Directions.SOUTH), ((x - 1, y), Directions.WEST),
                            ((x + 1, y), Directions.EAST)]

        if distance == maxSearchDepth or location in objectsToCompare:
            return distance, None

        if visited is None:
            visited = [location]

        minDirection = None

        for map in directionMapping:
            (coord, direction) = map
            if coord not in self.walls and coord not in visited:
                (currentDist, _) = self.findNearest(coord, objectsToCompare, distance + 1, visited + [coord])
                if currentDist < minDist and currentDist != maxSearchDepth:
                    minDist = currentDist
                    minDirection = direction

        return minDist, minDirection

    def __hash__(self):
        return hash((self.pacman, self.ghostDirection, self.foodDirection, self.ghostToFoodDistance))

    def __eq__(self, other):
        return (self.pacman == other.pacman and self.ghostDirection == other.ghostDirection
                and self.foodDirection == other.foodDirection and self.ghostToFoodDistance == other.ghostToFoodDistance)


class QLearnCraigAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
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
        # Count the number of games we have played
        self.episodesSoFar = 0

        # map states to action rewards
        self.statesToActionQValues = dict()
        # map states to action counts
        self.statesToActionCounter = dict()
        # define an exploration k value
        self.explorationKValue = 10

        self.prevAction = None
        self.prevStateFeatures = None
        self.prevState = None

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
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """

        # simple reward function that compares the score before and after the action taken
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """

        if state not in self.statesToActionQValues:
            return 0
        if action not in self.statesToActionQValues[state]:
            return 0

        return self.statesToActionQValues[state][action]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        if state not in self.statesToActionQValues:
            return 0

        maxQValue = 0
        for action in self.statesToActionQValues[state]:
            utility = self.statesToActionQValues[state][action]
            count = self.getCount(state, action)
            qValue = max(self.explorationFn(utility, count), self.statesToActionQValues[state][action])
            if qValue > maxQValue:
                maxQValue = qValue

        return maxQValue

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """

        # get current q-value
        currentQValue = self.getQValue(state, action)

        # the best q-value available in the next state
        maxQValue = self.maxQValue(nextState)

        # derive new q-value
        a = self.getAlpha()
        g = self.getGamma()
        newQValue = currentQValue + a * (reward + g * maxQValue - currentQValue)

        # update the q-value
        if state in self.statesToActionQValues:
            self.statesToActionQValues[state][action] = newQValue
        else:
            self.statesToActionQValues[state] = {action: newQValue}

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        if state in self.statesToActionCounter:
            if action in self.statesToActionCounter[state]:
                self.statesToActionCounter[state][action] += 1
            else:
                self.statesToActionCounter[state][action] = 1
        else:
            self.statesToActionCounter[state] = {action: 1}

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        if state in self.statesToActionCounter:
            if action in self.statesToActionCounter[state]:
                return self.statesToActionCounter[state][action]

        return 0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
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

        # if unexplored return maximum k-value
        if counts == 0:
            return utility + self.explorationKValue

        # otherwise return utility + weighted exploration value so unexplored actions
        # are more likely to be explored
        return utility + self.explorationKValue / counts

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

        currentStateFeatures = GameStateFeatures(state)

        # Compute the reward between previous state and current state
        if self.prevState is not None:
            reward = self.computeReward(self.prevState, state)
            # Learn from previous action
            self.learn(self.prevStateFeatures, self.prevAction, reward, currentStateFeatures)

        # determine whether to exploit or explore
        if util.flipCoin(self.epsilon):
            # explore
            action = random.choice(currentStateFeatures.legalActions)
        else:
            # otherwise, exploit
            action = self.findBestAction(currentStateFeatures)

        # update action counts
        self.updateCount(self.prevStateFeatures, action)

        # record current state for next iteration
        self.prevStateFeatures = currentStateFeatures
        self.prevState = state
        self.prevAction = action

        return action

    def findBestAction(self, state: GameStateFeatures) -> Directions:
        """
        Finds the best action to take.
        :param state: current state
        :return: best direction to take
        """
        vals = self.statesToActionQValues
        bestAction = None
        bestQValue = None
        if state in vals:
            for (action, value) in vals[state].items():
                if bestQValue is None or value > bestQValue:
                    bestQValue = value
                    bestAction = action

            return bestAction

        # no action available, make it random
        return random.choice(state.legalActions)

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """

        # calculate reward for last move
        reward = self.computeReward(self.prevState, state)

        # update learning
        self.learn(self.prevStateFeatures, self.prevAction, reward, GameStateFeatures(state))

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
