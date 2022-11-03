from __future__ import print_function

# test_classes.py
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


# import modules from python standard library
from builtins import object
import inspect
import re
import sys


# Class which models a question in a project.  Note that questions have a
# maximum number of points they are worth, and are composed of a series of
# test cases
class Question(object):
    def raise_not_defined(self):
        print("Method not implemented: %s" % inspect.stack()[1][3])
        sys.exit(1)

    def __init__(self, question_dict, display):
        self.max_points = int(question_dict["max_points"])
        self.test_cases = []
        self.display = display

    def get_display(self):
        return self.display

    def get_max_points(self):
        return self.max_points

    # Note that 'thunk' must be a function which accepts a single argument,
    # namely a 'grading' object
    def add_test_case(self, test_case, thunk):
        self.test_cases.append((test_case, thunk))

    def execute(self, grades):
        self.raise_not_defined()


# Question in which all test cases must be passed in order to receive credit
class PassAllTestsQuestion(Question):
    def execute(self, grades):
        # TODO: is this the right way to use grades?  The autograder doesn't seem to use it.
        tests_failed = False
        grades.assign_zero_credit()
        for _, f in self.test_cases:
            if not f(grades):
                tests_failed = True
        if tests_failed:
            grades.fail("Tests failed.")
        else:
            grades.assign_full_credit()


class ExtraCreditPassAllTestsQuestion(Question):
    def __init__(self, question_dict, display):
        Question.__init__(self, question_dict, display)
        self.extra_points = int(question_dict["extra_points"])

    def execute(self, grades):
        # TODO: is this the right way to use grades?  The autograder doesn't seem to use it.
        tests_failed = False
        grades.assign_zero_credit()
        for _, f in self.test_cases:
            if not f(grades):
                tests_failed = True
        if tests_failed:
            grades.fail("Tests failed.")
        else:
            grades.assign_full_credit()
            grades.add_points(self.extra_points)


# Question in which predict credit is given for test cases with a ``points'' property.
# All other tests are mandatory and must be passed.
class HackedPartialCreditQuestion(Question):
    def execute(self, grades):
        # TODO: is this the right way to use grades?  The autograder doesn't seem to use it.
        grades.assign_zero_credit()

        points = 0
        passed = True
        for test_case, f in self.test_cases:
            test_result = f(grades)
            if "points" in test_case.test_dict:
                if test_result:
                    points += float(test_case.test_dict["points"])
            else:
                passed = passed and test_result

        ## FIXME: Below terrible hack to match q3's logic
        if int(points) == self.max_points and not passed:
            grades.assign_zero_credit()
        else:
            grades.add_points(int(points))


class Q6PartialCreditQuestion(Question):
    """Fails any test which returns False, otherwise doesn't effect the grades object.
    Partial credit tests will add the required points."""

    def execute(self, grades):
        grades.assign_zero_credit()

        results = []
        for _, f in self.test_cases:
            results.append(f(grades))
        if False in results:
            grades.assign_zero_credit()


class PartialCreditQuestion(Question):
    """Fails any test which returns False, otherwise doesn't effect the grades object.
    Partial credit tests will add the required points."""

    def execute(self, grades):
        grades.assign_zero_credit()

        for _, f in self.test_cases:
            if not f(grades):
                grades.assign_zero_credit()
                grades.fail("Tests failed.")
                return False


class NumberPassedQuestion(Question):
    """Grade is the number of test cases passed."""

    def execute(self, grades):
        grades.add_points([f(grades) for _, f in self.test_cases].count(True))


# Template modeling a generic test case
class TestCase(object):
    def raise_not_defined(self):
        print("Method not implemented: %s" % inspect.stack()[1][3])
        sys.exit(1)

    def get_path(self):
        return self.path

    def __init__(self, question, test_dict):
        self.question = question
        self.test_dict = test_dict
        self.path = test_dict["path"]
        self.messages = []

    def __str__(self):
        self.raise_not_defined()

    def execute(self, grades, module_dict, solution_dict):
        self.raise_not_defined()

    def write_solution(self, module_dict, file_path):
        self.raise_not_defined()
        return True

    # Tests should call the following messages for grading
    # to ensure a uniform format for test output.
    #
    # TODO: this is hairy, but we need to fix grading.py's interface
    # to get a nice hierarchical project - question - test structure,
    # then these should be moved into Question proper.
    def test_pass(self, grades):
        grades.add_message("PASS: %s" % (self.path,))
        for line in self.messages:
            grades.add_message("    %s" % (line,))
        return True

    def test_fail(self, grades):
        grades.add_message("FAIL: %s" % (self.path,))
        for line in self.messages:
            grades.add_message("    %s" % (line,))
        return False

    # This should really be question level?
    #
    def test_partial(self, grades, points, max_points):
        grades.add_points(points)
        extra_credit = max(0, points - max_points)
        regular_credit = points - extra_credit

        grades.add_message(
            "%s: %s (%s of %s points)"
            % (
                "PASS" if points >= max_points else "FAIL",
                self.path,
                regular_credit,
                max_points,
            )
        )
        if extra_credit > 0:
            grades.add_message("EXTRA CREDIT: %s points" % (extra_credit,))

        for line in self.messages:
            grades.add_message("    %s" % (line,))

        return True

    def add_message(self, message):
        self.messages.extend(message.split("\n"))
