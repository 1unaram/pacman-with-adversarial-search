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

        # [3] Capsule
        capsuleDistances = [util.manhattanDist(newPos, capsule) for capsule in successorGameState.getCapsules()]
        nearestCapsuleDistance = min(capsuleDistances) if capsuleDistances else 0

        GHOST_DIST_BOUND = 3
        CAPSULE_DIST_BOUND = 3

        weights = {
            'distanceToFood': 10,
            'distanceToGhost': -500 if nearestGhostDistance <= GHOST_DIST_BOUND and not any(newScaredTimes) else 0,
            'distanceToScaredGhost': 50 if any(newScaredTimes) else 0,
            'distanceToCapsule': 30 if nearestCapsuleDistance <= CAPSULE_DIST_BOUND else 0
        }

        finalScore = (
            successorGameState.getScore()
            + (weights['distanceToFood'] / (nearestFoodDistance + 1))
            + (weights['distanceToGhost'] / (nearestGhostDistance + 1))
            + (weights['distanceToScaredGhost'] / (nearestGhostDistance + 1))
            + (weights['distanceToCapsule'] / (nearestCapsuleDistance + 1))
        )

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
        optimalAction = Directions.STOP
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
        optimalAction = Directions.STOP
        optimalScore = NEG_INFINITY
        alpha = NEG_INFINITY
        beta = INFINITY


        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            # Start alpha-beta pruning from the first ghost at depth == self.depth
            v = self.alphaBetaPruning(successor, self.depth, 1, alpha, beta)

            if v > optimalScore:
                optimalScore = v
                optimalAction = action

            alpha = max(alpha, v)

        return optimalAction


debug = False

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
        optimalAction = Directions.STOP
        optimalScore = NEG_INFINITY
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            print(f"<{action[:2]}>: ", end="") if debug else None


            # Start expectimax from the first ghost at depth == self.depth
            score = self.expectimax(successor, self.depth, 1)

            if score > optimalScore:
                optimalScore = score
                optimalAction = action

        print(f">>> Chosen action: {optimalAction} with score {optimalScore}") if debug else None


        return optimalAction



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    # [1] Terminal state
    if currentGameState.isWin():
        return INFINITY
    if currentGameState.isLose():
        return NEG_INFINITY

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()

    GHOST_DIST_BOUND = 3

    weights = {
        'foodLeft': -10,
        'distanceToFood': 10,
        'distanceToGhost': -500,
        'distanceToScaredGhost': 100,
        'distanceToCapsule': 30
    }
    print('\t', end='') if debug else None
    # [2] Food
    if foodList:
        foodDistances = [util.manhattanDist(pacmanPos, food) for food in foodList]
        nearestFoodDistance = min(foodDistances)

        # [2-1] Nearest food
        score += (weights['distanceToFood'] / (nearestFoodDistance + 1))
        print(f"[Food] {weights['distanceToFood'] / (nearestFoodDistance + 1)} / ", end="") if debug else None

        # [2-2] Leftover food
        score += (weights['foodLeft'] * len(foodList))
        print(f"[Food Left] {weights['foodLeft'] * len(foodList)} / ", end="") if debug else None

    # [3] Ghosts
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = util.manhattanDist(pacmanPos, ghostPos)

        # [3-1] Scared ghosts
        if ghost.scaredTimer > 0:
            if ghost.scaredTimer > ghostDistance + 3:
                score += (weights['distanceToScaredGhost'] / (ghostDistance + 1))

        # [3-2] Non-scared ghosts
        else:
            # [3-2-1] Dangerous state
            if ghostDistance <= GHOST_DIST_BOUND:
                score += (weights['distanceToGhost'] / (ghostDistance + 1))
                print(f"[Ghost] {weights['distanceToGhost'] / (ghostDistance + 1)} / ", end="") if debug else None
            # [3-2-2] Less dangerous state
            elif ghostDistance <= GHOST_DIST_BOUND + 2:
                score += (0.7 * weights['distanceToGhost'] / (ghostDistance + 1))
                print(f"[Ghost - Far] {0.5 * weights['distanceToGhost'] / (ghostDistance + 1)} / ", end="") if debug else None

    # [4] Capsules
    if capsuleList:
        capsuleDistances = [util.manhattanDist(pacmanPos, capsule) for capsule in capsuleList]
        nearestCapsuleDistance = min(capsuleDistances) if capsuleDistances else 0

        # [4-1] Go to capsule
        score += (weights['distanceToCapsule'] / (nearestCapsuleDistance + 1))
        print(f"[Capsule] {weights['distanceToCapsule'] / (nearestCapsuleDistance + 1)} / ", end="") if debug else None

    print(f"[Score] {score}") if debug else None

    return score

# Abbreviation
better = betterEvaluationFunction
