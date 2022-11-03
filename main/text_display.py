from __future__ import print_function

# text_display.py
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


from builtins import str
from builtins import range
from builtins import object
import time

try:
    import pacman
except:
    pass

DRAW_EVERY = 1
SLEEP_TIME = 0  # This can be overwritten by __init__
DISPLAY_MOVES = False
QUIET = False  # Supresses output


class NullGraphics(object):
    def initialize(self, state, is_blue=False):
        pass

    def update(self, state):
        pass

    def check_null_display(self):
        return True

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    def update_distributions(self, dist):
        pass

    def finish(self):
        pass


class PacmanGraphics(object):
    def __init__(self, speed=None):
        if speed != None:
            global SLEEP_TIME
            SLEEP_TIME = speed

    def initialize(self, state, is_blue=False):
        self.draw(state)
        self.pause()
        self.turn = 0
        self.agent_counter = 0

    def update(self, state):
        num_agents = len(state.agent_states)
        self.agent_counter = (self.agent_counter + 1) % num_agents
        if self.agent_counter == 0:
            self.turn += 1
            if DISPLAY_MOVES:
                ghosts = [
                    pacman.nearest_point(state.get_ghost_position(i))
                    for i in range(1, num_agents)
                ]
                print(
                    "%4d) P: %-8s"
                    % (self.turn, str(pacman.nearest_point(state.get_pacman_position()))),
                    "| Score: %-5d" % state.score,
                    "| Ghosts:",
                    ghosts,
                )
            if self.turn % DRAW_EVERY == 0:
                self.draw(state)
                self.pause()
        if state._win or state._lose:
            self.draw(state)

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    def finish(self):
        pass
