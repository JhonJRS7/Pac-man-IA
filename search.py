# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List
import random
import numpy as np


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()
        

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def euclideanDistance(position, problem):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    #util.raiseNotDefined()
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Usando uma fila
    states2explore = util.Queue()      
  
    startState = problem.getStartState()

    print("Start:", problem.getStartState())
    print("Goal:", problem.getGoalState())
    
    
    startNode = (startState, [], 0) #(state, action, cost)

    exploredStates = []
    
    # O estado inicial eh armazenado nos estados que serao explorados
    states2explore.push(startNode)
    
    while not states2explore.isEmpty():
        # Recupera o proximo estado
        currentState, actions, currentCost = states2explore.pop()
        
        if currentState not in exploredStates:
            #put popped node state into explored list
            exploredStates.append(currentState)
            #print("Is the current state goal?", problem.isGoalState(currentState))
            #print("Euclidean: ", euclideanDistance(currentState, problem))
            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                #print("Current successors:", problem.getSuccessors(currentState))
                successors = problem.getSuccessors(currentState)
                
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    states2explore.push(newNode)

    return actions

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Usando uma fila
    states2explore = util.Stack()      
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    exploredStates = []
    
    # O estado inicial eh armazenado nos estados que serao explorados
    states2explore.push(startNode)
    
    while not states2explore.isEmpty():
        # Recupera o proximo estado
        currentState, actions, currentCost = states2explore.pop()
        
        if currentState not in exploredStates:
            #put popped node state into explored list
            exploredStates.append(currentState)

            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
                
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    states2explore.push(newNode)

    return actions

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Usando uma fila
    states2explore = util.PriorityQueue()      
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    exploredStates = {}
    
    # O estado inicial eh armazenado nos estados que serao explorados
    states2explore.push(startNode, 0)
    
    while not states2explore.isEmpty():
        #begin exploring first (lowest-cost) node on states2explore
        currentState, actions, currentCost = states2explore.pop()
       
        if (currentState not in exploredStates) or (currentCost < exploredStates[currentState]):
            #put popped node's state into explored list
            exploredStates[currentState] = currentCost

            if problem.isGoalState(currentState):
                return actions
            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
               
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    states2explore.update(newNode, newCost)

    return actions


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def greedySearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    nextState = None
    
    startState = problem.getStartState()
    currentState = startState

    actions = []
    exploredStates = []
    currentCost = 0

    import sys
    
    while currentState != None:

        if problem.isGoalState(currentState) or currentState == None:
            return actions
        else:
            exploredStates.append(currentState)
            successors = problem.getSuccessors(currentState)

            if len(problem.getSuccessors(currentState)) == 0 and not problem.isGoalState(currentState):
                return actions
            
            bestAction = []
            bestCost = 0
            bestSucc = None
            currentHeuristic = sys.maxsize

            for succState, succAction, succCost in successors:
                if succState not in exploredStates:
                    distTemp = euclideanDistance(succState, problem)
                    if distTemp < currentHeuristic:
                        bestAction = [succAction]
                        bestCost = succCost
                        bestSucc = succState
                        currentHeuristic = distTemp

            actions = actions + bestAction
            currentCost = bestCost + currentCost            
            #currentState = (bestSucc, actions, currentCost)
            #print("Current:", currentState)
            #print("Next:", bestSucc)
            currentState = bestSucc

    #print(actions)
    return actions

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    states2explore = util.PriorityQueue()

    exploredNodes = [] #holds (state, cost)

    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    states2explore.push(startNode, 0)

    while not states2explore.isEmpty():

        #begin exploring first (lowest-combined (cost+heuristic) ) node on states2explore
        currentState, actions, currentCost = states2explore.pop()

        #put popped node into explored list
        exploredNodes.append((currentState, currentCost))

        if problem.isGoalState(currentState):
            return actions

        else:
            #list of (successor, action, stepCost)
            successors = problem.getSuccessors(currentState)

            #examine each successor
            for succState, succAction, succCost in successors:
                newAction = actions + [succAction]
                newCost = problem.getCostOfActions(newAction)
                newNode = (succState, newAction, newCost)

                #check if this successor has been explored
                already_explored = False
                for explored in exploredNodes:
                    #examine each explored node tuple
                    exploredState, exploredCost = explored

                    if (succState == exploredState) and (newCost >= exploredCost):
                        already_explored = True

                #if this successor not explored, put on states2explore and explored list
                if not already_explored:
                    states2explore.push(newNode, newCost + heuristic(succState, problem))
                    exploredNodes.append((succState, newCost))

    return actions

def geneticAlgorithm(problem: SearchProblem, population_size=50, generations=100, mutation_rate=0.1) -> List[Directions]:
    def get_legal_actions(state):
        return [action for _, action, _ in problem.getSuccessors(state)]

    def randomSolution():
        state = problem.getStartState()
        solution = []
        for _ in range(10000):
            legal_actions = get_legal_actions(state)
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            solution.append(action)
            state = [s for s, a, _ in problem.getSuccessors(state) if a == action][0]
        return solution

    def fitness(solution):
        current_state = problem.getStartState()
        score = 0
        for action in solution:
            if problem.isGoalState(current_state):
                score += 1000  # Reward for reaching the goal
                break
            successors = problem.getSuccessors(current_state)
            legal_actions = [a for _, a, _ in successors]
            if action in legal_actions:
                current_state = [s for s, a, _ in successors if a == action][0]
                score += 1
            else:
                break
        return score

    def crossover(parent1, parent2):
        crossover_point = random.randint(0, min(len(parent1), len(parent2)) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(solution):
        if not solution:
            return solution

        mutation_point = random.randint(0, len(solution) - 1)
        state = problem.getStartState()

        # Aplicar as ações até o ponto de mutação
        for action in solution[:mutation_point]:
            legal_actions = get_legal_actions(state)
            if action not in legal_actions: # Se a ação se tornou ilegal pare
                return solution
            state = [s for s, a, _ in problem.getSuccessors(state) if a == action][0]

        # Escolher uma nova ação legal para o ponto de mutação
        legal_actions = get_legal_actions(state)
        if legal_actions:
            solution[mutation_point] = random.choice(legal_actions)
        return solution

    population = [randomSolution() for _ in range(population_size)]

    for _ in range(generations):
        fitness_scores = [(fitness(individual), individual) for individual in population]
        fitness_scores.sort(reverse=True, key=lambda x: x[0])

        selected = [individual for _, individual in fitness_scores[:population_size // 2]]

        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            next_generation.append(child)

        population = next_generation

    best_solution = max(population, key=lambda ind: fitness(ind))
    return best_solution

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
gdy = greedySearch
astar = aStarSearch

ga = geneticAlgorithm