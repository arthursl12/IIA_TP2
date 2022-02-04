# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Value Iteration Algorithm (adapted from textbook)
        all_states = mdp.getStates()
        for i in range (self.iterations):
            values = self.values.copy()
            
            for state in all_states:
                # Lowest value is actually -infinite, initialization only
                self.values[state] = -float('inf')
                
                # Compute QValue locally using frozen values dict
                # and update current state value on the go
                As = mdp.getPossibleActions(state)
                for action in As:
                    P = self.mdp.getTransitionStatesAndProbs(state, action)
                    gamma = self.discount
                    total = 0
                    for (nextstate, prob) in P:
                        rs = self.mdp.getReward(state, action, nextstate)
                        term = prob * (rs + gamma * values[nextstate])
                        total += term
                    self.values[state] = max(self.values[state], total)
                
                # If the -infinite was not updated, we must correct it to 0
                if self.values[state] == -float('inf'):
                    self.values[state] = 0.0
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        P = self.mdp.getTransitionStatesAndProbs(state, action)
        gamma = self.discount
        total = 0
        for (nextstate, prob) in P:
            rs = self.mdp.getReward(state, action, nextstate)
            term = prob * (rs + gamma * self.values[nextstate])
            total += term
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Terminal state, i.e., there are no legal actions
        if self.mdp.isTerminal(state):
            return None
        
        # Solving Ties: the first state in possible action list is used
        possible_actions = self.mdp.getPossibleActions(state)
        max_action = None
        max_reward = -float('inf')
        for action in possible_actions:
            qval = self.getQValue(state, action)
            if (qval > max_reward):
                max_reward = qval
                max_action = action
        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
