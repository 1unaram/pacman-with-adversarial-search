# multiAgents.py
# --------------
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


import random

import util
from game import Agent, Directions
from pacman import GameState

# Constants
INFINITY = float('inf')
NEG_INFINITY = float('-inf')

GHOST_DIST_BOUND = 3
CAPSULE_DIST_BOUND = 3


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        # "Add more of your code here if you want to"
        # print(f"Chosen action: {legalMoves[chosenIndex]} with score {bestScore}")
        # print("=" * 80)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Evaluation function for your reflex agent (question 1).

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        # Do not take STOP action
        if action == Directions.STOP:
            return NEG_INFINITY

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # [1] Distance to the nearest food
        foodDistances = [util.manhattanDist(newPos, food) for food in newFood.asList()]
        nearestFoodDistance = min(foodDistances) if foodDistances else 0

        # [2] Distance to the nearest ghost
        ghostDistances = [util.manhattanDist(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhostDistance = min(ghostDistances) if ghostDistances else 0

        # [3] Pellet
        pelletDistances = [util.manhattanDist(newPos, pellet) for pellet in successorGameState.getPellets()]
        nearestPelletDistance = min(pelletDistances) if pelletDistances else 0

        GHOST_DIST_BOUND = 3
        CAPSULE_DIST_BOUND = 3

        weights = {
            'distanceToFood': -0.01,
            'distanceToGhost': -100 if nearestGhostDistance <= GHOST_DIST_BOUND else 0,
            'distanceToScaredGhost': 70 if nearestGhostDistance <= GHOST_DIST_BOUND and any(newScaredTimes) else 0,
            'distanceToPellet': 30 if nearestPelletDistance <= CAPSULE_DIST_BOUND else 0
        }

        finalScore = (
            successorGameState.getScore()
            + weights['distanceToFood'] * nearestFoodDistance
            + weights['distanceToGhost'] * (GHOST_DIST_BOUND - nearestGhostDistance)
            + weights['distanceToScaredGhost'] * nearestGhostDistance
            + weights['distanceToPellet'] * (CAPSULE_DIST_BOUND - nearestPelletDistance)
        )

        # print(f"<{action[0]}> ", end="")
        # print(f"[GetScore] {successorGameState.getScore()} / ", end="")
        # print(f"[Food] {weights['distanceToFood']} * {nearestFoodDistance} / ", end="")
        # print(f"[Ghost] {weights['distanceToGhost']} * {nearestGhostDistance} / ", end="")
        # print(f"[Pellet] {weights['distanceToPellet']} / {nearestPelletDistance} / ", end="")

        # print(f"[Score] {finalScore}")

        return finalScore


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, state, depth, agentIndex):

        # [1] Terminal state
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # [2] Pacman's turn (max)
        if agentIndex == 0:
            v = NEG_INFINITY

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)

                # Start from the first ghost
                v = max(v, self.minimax(successor, depth, 1))

            return v

        # [3] Ghost's turn (min)
        else:
            v = INFINITY
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                # Last ghost: Go to next depth / Else: Go to next ghost
                nextDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
                nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
                v = min(v, self.minimax(successor, nextDepth, nextAgentIndex))

            return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        """
        Pacman이 현재 state에서 움직이기 시작 (depth = self.depth)
        1. Pacman이 움직임 (depth = self.depth)
        2. 첫 번째 유령이 움직임 (depth = self.depth)
        3. 두 번째 유령이 움직임 (depth = self.depth)
        4. Pacman이 움직임 (depth = self.depth - 1)
        5. 첫 번째 유령이 움직임 (depth = self.depth - 1)
        6. 두 번째 유령이 움직임 (depth = self.depth - 1)
        7. Pacman이 움직임 (depth = self.depth - 2)
        ...
        마지막 유령이 움직임 (depth = 0)
        -> 종료
        """

        # Randomize legal actions
        legalActions = gameState.getLegalActions(0)
        random.shuffle(legalActions)

        # Find the optimal action for Pacman
        optimalAction = None
        optimalScore = NEG_INFINITY
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            # Start minimax from the first ghost at depth == self.depth
            score = self.minimax(successor, self.depth, 1)

            if score > optimalScore:
                optimalScore = score
                optimalAction = action

        return optimalAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBetaPruning(self, state, depth, agentIndex, alpha, beta):

        # [1] Terminal state
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # [2] Pacman's turn (max - alpha)
        if agentIndex == 0:
            v = NEG_INFINITY
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)

                # Start from the first ghost
                v = max(v, self.alphaBetaPruning(successor, depth, 1, alpha, beta))

                if v > beta:
                    return v

                alpha = max(alpha, v)
            return v

        # [3] Ghost's turn (min - beta)
        else:
            v = INFINITY
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                # Last ghost: Go to next depth / Else: Go to next ghost
                nextDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
                nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
                v = min(v, self.alphaBetaPruning(successor, nextDepth, nextAgentIndex, alpha, beta))

                if v < alpha:
                    return v

                beta = min(beta, v)

            return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Find the optimal action for Pacman
        optimalAction = None
        optimalScore = NEG_INFINITY
        alpha = NEG_INFINITY
        beta = INFINITY

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            # Start alpha-beta pruning from the first ghost at depth == self.depth
            v = self.alphaBetaPruning(successor, self.depth, 1, alpha, beta)

            if v > optimalScore:
                optimalScore = v
                optimalAction = action

            alpha = max(alpha, v)

        return optimalAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, state, depth, agentIndex):

        # [1] Terminal state
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # [2] Pacman's turn (max)
        if agentIndex == 0:
            v = NEG_INFINITY

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)

                # Start from the first ghost
                v = max(v, self.expectimax(successor, depth, 1))

            return v

        # [3] Ghost's turn (exp)
        else:
            v = 0

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                # Last ghost: Go to next depth / Else: Go to next ghost
                nextDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
                nextAgentIndex = (agentIndex + 1) % state.getNumAgents()

                p = 1 / len(state.getLegalActions(agentIndex)) # Uniformly at random
                v += p * self.expectimax(successor, nextDepth, nextAgentIndex)

            return v

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        # Randomize legal actions to avoid bias back and forth
        legalActions = gameState.getLegalActions(0)
        random.shuffle(legalActions)

        # Find the optimal action for Pacman
        optimalAction = None
        optimalScore = NEG_INFINITY
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            # Start expectimax from the first ghost at depth == self.depth
            score = self.expectimax(successor, self.depth, 1)

            if score > optimalScore:
                optimalScore = score
                optimalAction = action

        return optimalAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    pelletList = currentGameState.getPellets()

    weights = {
        'distanceToFood': -0.1,
        'distanceToGhost': -100,
        'distanceToScaredGhost': 70,
        'distanceToPellet': 30
    }

    # [1] Distance to the nearest food
    if foodList:
        foodDistances = [util.manhattanDist(pacmanPos, food) for food in foodList]
        nearestFoodDistance = min(foodDistances)
        score += weights['distanceToFood'] * nearestFoodDistance

    # [2] Distance to the nearest ghost
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = util.manhattanDist(pacmanPos, ghostPos)

        # Not afraid of scared ghosts
        if ghost.scaredTimer > 0:
            score += weights['distanceToScaredGhost'] * ghostDistance
        elif ghostDistance <= GHOST_DIST_BOUND:
            score += weights['distanceToGhost'] * ghostDistance

    # [3] Distance to the nearest pellet
    if pelletList:
        pelletDistances = [util.manhattanDist(pacmanPos, pellet) for pellet in pelletList]
        nearestPelletDistance = min(pelletDistances)

        if nearestPelletDistance <= CAPSULE_DIST_BOUND:
            score += weights['distanceToPellet'] * nearestPelletDistance

    return score


# Abbreviation
better = betterEvaluationFunction
