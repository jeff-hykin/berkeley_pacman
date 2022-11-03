# pacman_agents.py
# ---------------
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


from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def get_action(self, state):
        legal = state.get_legal_pacman_actions()
        current = state.get_pacman_state().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, eval_fn="score_evaluation"):
        self.evaluation_function = util.lookup(eval_fn, globals())
        assert self.evaluation_function != None

    def get_action(self, state):
        # Generate candidate actions
        legal = state.get_legal_pacman_actions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generate_successor(0, action), action) for action in legal]
        scored = [
            (self.evaluation_function(state), action) for state, action in successors
        ]
        best_score = max(scored)[0]
        best_actions = [pair[1] for pair in scored if pair[0] == best_score]
        return random.choice(best_actions)


def score_evaluation(state):
    return state.get_score()
