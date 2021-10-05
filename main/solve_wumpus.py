from wumpus_world import Agent
import random

mr_bond = Agent()
print('Possible actions = ', mr_bond.valid_actions)
while mr_bond.is_alive and not mr_bond.escaped:
    # available data
    x, y, breeze, stench = mr_bond.observe()
    
    # 
    # example: Move randomly
    # 
    random_action = random.choice(mr_bond.valid_actions)
    mr_bond.take_action(random_action)