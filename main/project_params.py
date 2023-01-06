# project_params.py
# ----------------
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


choices = {
    "a_star_only": {
        "STUDENT_CODE_DEFAULT" : "search_agents.py,search.py",
        "PROJECT_TEST_CLASSES" : "search_test_classes.py",
        "PROJECT_NAME" : "Project 1: A Star",
        "BONUS_PIC" : False,
        "TEST_CASES": "a_star_test_cases",
    },
    "search_agent": {
        "STUDENT_CODE_DEFAULT" : "search_agents.py,search.py",
        "PROJECT_TEST_CLASSES" : "search_test_classes.py",
        "PROJECT_NAME" : "Project 1: Search",
        "BONUS_PIC" : False,
        "TEST_CASES": "search_agent_test_cases",
    },
    "multi_agent": {
        "STUDENT_CODE_DEFAULT" : 'multi_agents.py',
        "PROJECT_TEST_CLASSES" : 'multiagent_test_classes.py',
        "PROJECT_NAME" : 'Project 2: Multiagent search',
        "BONUS_PIC" : False,
        "TEST_CASES": "multi_agent_test_cases",
    }
}