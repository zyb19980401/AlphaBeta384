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
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


def helper_FindTheMinDis(position, target_List):
    min_distance = None
    for item in target_List:
        item_distance = manhattanDistance(position, item)
        if min_distance is None:
            min_distance = item_distance
        if item_distance < min_distance:
            min_distance = item_distance
    if min_distance is None:
        return 0
    else:
        return min_distance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

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

        "*** YOUR CODE HERE ***"
        current_value = successorGameState.getScore()
        if action == Directions.STOP:
            return float("-inf")
        new_ghost_list = [i.getPosition() for i in newGhostStates]
        min_food_distance = helper_FindTheMinDis(newPos, newFood.asList())
        min_ghost_distance = helper_FindTheMinDis(newPos, new_ghost_list)
        if min_ghost_distance <= 1:
            return float('-inf')
        for food in currentGameState.getFood().asList():
            if manhattanDistance(newPos, food) == 0:
                return float('inf')
        result = current_value - min_food_distance + min_ghost_distance/10   #since the ghost is distance should be as
                                                                             # far away as posible. So we add it
        return result






def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        result = self.maxHelper(gameState, self.depth, 0)
        return result[0]

    def maxHelper(self, gamestate, maxdepth, depth):
        "base case: if max Depth is reached or the game is over"
        is_over = gamestate.isWin() or gamestate.isLose()
        if depth == maxdepth or is_over:
            return [None, self.evaluationFunction(gamestate)]
        states = []
        for action in gamestate.getLegalActions(0):
            acc_state = gamestate.generateSuccessor(0, action)
            states.append([action, (self.minHelper(acc_state, maxdepth, depth, 1))])
        max_answers = []
        for state in states:
            if max_answers == []:
                max_answers = state
            if state[1] > max_answers[1]:
                max_answers = state

        return max_answers


    def minHelper(self, gameState, maxdepth, depth,ghostIndex):

        "base case"
        isover = gameState.isWin() or gameState.isLose()
        if maxdepth == depth or isover:
            return self.evaluationFunction(gameState)

        new_states = []
        for action in gameState.getLegalActions(ghostIndex):
            acc_state = gameState.generateSuccessor(ghostIndex, action)
            "if this is the last ghost, we call maxHelper with depth + 1"
            if ghostIndex == (gameState.getNumAgents() - 1):
                new_states.append(self.maxHelper(acc_state, maxdepth, depth + 1)[1])
                "if this is not the last ghost, we add one more ghostindex to the answer"
            else:
                new_states.append(self.minHelper(acc_state, maxdepth, depth, ghostIndex + 1))
        min_answer = None
        for state in new_states:
            if min_answer is None:
                min_answer = state
            if state < min_answer:
                min_answer = state

        return min_answer







class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxdepth = self.depth
        alpha = float("-inf")
        beta = float("inf")
        result = self.maxhelper(gameState, maxdepth, 0, alpha, beta)
        return result[0]

    def maxhelper(self, gamestate, maxdepth, depth, alpha, beta):
        isover = gamestate.isWin() or gamestate.isLose()
        if isover or maxdepth == depth:
            return[None, self.evaluationFunction(gamestate)]
        new_states = []
        max_result = []

        for action in gamestate.getLegalActions(0):
            acc_action = gamestate.generateSuccessor(0, action)
            min_num = self.minhelper(acc_action, maxdepth, depth, alpha, beta, 1)
            new_states.append([action, min_num])
            for state in new_states:
                if max_result == []:
                    max_result = state
                if state[1] > max_result[1]:
                    max_result = state
            if max_result[1] >= beta:
                return max_result
            alpha = max(alpha, max_result[1]) #update
        return max_result

    def minhelper(self, gamestate, maxdepth, depth, alpha, beta, ghostindex):
        isover = gamestate.isLose() or gamestate.isWin()
        if isover or depth == maxdepth:
            return self.evaluationFunction(gamestate)

        new_states = []
        min_result = None

        for action in gamestate.getLegalActions(ghostindex):
            acc_state = gamestate.generateSuccessor(ghostindex, action)
            if ghostindex != (gamestate.getNumAgents() - 1):
                min_num = self.minhelper(acc_state, maxdepth, depth, alpha, beta, ghostindex + 1)
                new_states.append(min_num)
            else:
                max_num = self.maxhelper(acc_state, maxdepth, depth + 1, alpha, beta)[1]
                new_states.append(max_num)
            for state in new_states:
                if min_result is None:
                    min_result = state
                if state < min_result:
                    min_result = state
            if min_result <= alpha:
                return min_result
            beta = min(beta, min_result) #update
        return min_result



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxdepth = self.depth
        result = self.maxhelper(gameState, maxdepth, 0)
        return result[0]

    def maxhelper(self, gamestate, maxdepth, depth):
        isover = gamestate.isWin() or gamestate.isLose()
        if depth == maxdepth or isover:
            return [None, self.evaluationFunction(gamestate)]

        new_states = []
        for action in gamestate.getLegalActions(0):
            acc_state = gamestate.generateSuccessor(0, action)
            min_num = self.minhelper(acc_state, maxdepth, depth, 1)
            new_states.append([action, min_num])
        max_result = []
        for state in new_states:
            if max_result == []:
                max_result = state
            if state[1] > max_result[1]:
                max_result = state
        return max_result

    def minhelper(self, gamestate, maxdepth, depth, ghostindex):

        isover = gamestate.isLose() or gamestate.isWin()
        if depth == maxdepth or isover:
            return self.evaluationFunction(gamestate)
        new_states = []

        for action in gamestate.getLegalActions(ghostindex):
            acc_state = gamestate.generateSuccessor(ghostindex, action)
            if ghostindex != gamestate.getNumAgents() - 1:
                min_num = self.minhelper(acc_state, maxdepth, depth, ghostindex + 1)
                new_states.append(min_num)
            else:
                max_num = self.maxhelper(acc_state, maxdepth, depth + 1)[1]
                new_states.append(max_num)
        "average"
        sum = 0
        for state in new_states:
            sum += state
        return float(sum) / float(len(new_states))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: First, we add up all the ghost scare time together.
      This is very important. We can move wherever we want if the ghost scare time is bigger than 0.
      We loop over all the ghost. we seperate ghosts into two different types. one is close to the pacman,
      the other one is far frome the pacman. if the ghost is close to the pacman(manhattanDistance < 4), then
      we will choose to get away from the ghost than keey eating the food. If the ghost is far away from the pacman
      (manhattanDistance >4), we will want to eat as much food as posible instead of running away from the ghost. This
      implementation helps the pacman get moving while ghost is away and run away from the ghost while a ghost is close.
    """
    "*** YOUR CODE HERE ***"

    position = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostScaredTimes = 0

    for i in ghosts:
        ghostScaredTimes += i.scaredTimer

    score = currentGameState.getScore()

    for ghostState in ghosts:
        ghostPos = ghostState.getPosition()
        manhattanDistance = util.manhattanDistance(position, ghostPos)
        if 0 < manhattanDistance < 4:
            score -= 1.0 / manhattanDistance * 6
            for foodPos in foodList:
                fooddistance = util.manhattanDistance(position, foodPos)
                if fooddistance > 0:
                    score += 1.0 / fooddistance
            score += ghostScaredTimes
        if manhattanDistance > 4:
            score -= 1.0 / manhattanDistance
            for foodPos in foodList:
                fooddistance = util.manhattanDistance(position, foodPos)
                if fooddistance > 0:
                    score += 1.0 / fooddistance * 2
            score += ghostScaredTimes / 2
    return score

# Abbreviation
better = betterEvaluationFunction
