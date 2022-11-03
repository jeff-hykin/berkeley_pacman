# grading.py
# ----------
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


"Common code for autograders"
from __future__ import print_function

from builtins import str
from builtins import object
import cgi
import html
import time
import sys
import json
import traceback
import pdb
from collections import defaultdict
import util


class Grades(object):
    "A data structure for project grades, along with formatting code to display them"

    def __init__(
        self,
        project_name,
        questions_and_maxes_list,
        gs_output=False,
        edx_output=False,
        mute_output=False,
    ):
        """
        Defines the grading scheme for a project
          project_name: project name
          questions_and_maxes_dict: a list of (question name, max points per question)
        """
        self.questions = [el[0] for el in questions_and_maxes_list]
        self.maxes = dict(questions_and_maxes_list)
        self.points = Counter()
        self.messages = dict([(q, []) for q in self.questions])
        self.project = project_name
        self.start = time.localtime()[1:6]
        self.sane = True  # Sanity checks
        self.current_question = None  # Which question we're grading
        self.edx_output = edx_output
        self.gs_output = gs_output  # GradeScope output
        self.mute = mute_output
        self.prereqs = defaultdict(set)

        # print 'Autograder transcript for %s' % self.project
        print("Starting on %d-%d at %d:%02d:%02d" % self.start)

    def add_prereq(self, question, prereq):
        self.prereqs[question].add(prereq)

    def grade(self, grading_module, exception_map={}, bonus_pic=False):
        """
        Grades each question
          grading_module: the module with all the grading functions (pass in with sys.modules[__name__])
        """

        completed_questions = set([])
        for q in self.questions:
            print("\nQuestion %s" % q)
            print("=" * (9 + len(q)))
            print()
            self.current_question = q

            incompleted = self.prereqs[q].difference(completed_questions)
            if len(incompleted) > 0:
                prereq = incompleted.pop()
                print(
                    """*** NOTE: Make sure to complete Question %s before working on Question %s,
*** because Question %s builds upon your answer for Question %s.
"""
                    % (prereq, q, q, prereq)
                )
                continue

            if self.mute:
                util.mute_print()
            try:
                util.TimeoutFunction(getattr(grading_module, q), 1800)(
                    self
                )  # Call the question's function
                # TimeoutFunction(getattr(grading_module, q),1200)(self) # Call the question's function
            except Exception as inst:
                self.add_exception_message(q, inst, traceback)
                self.add_error_hints(exception_map, inst, q[1])
            except:
                self.fail("FAIL: Terminated with a string exception.")
            finally:
                if self.mute:
                    util.unmute_print()

            if self.points[q] >= self.maxes[q]:
                completed_questions.add(q)

            print("\n### Question %s: %d/%d ###\n" % (q, self.points[q], self.maxes[q]))

        print("\nFinished at %d:%02d:%02d" % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
            print("Question %s: %d/%d" % (q, self.points[q], self.maxes[q]))
        print("------------------")
        print("Total: %d/%d" % (self.points.total_count(), sum(self.maxes.values())))
        if bonus_pic and self.points.total_count() == 25:
            print(
                """

                     ALL HAIL GRANDPAC.
              LONG LIVE THE GHOSTBUSTING KING.

                  ---      ----      ---
                  |  \    /  + \    /  |
                  | + \--/      \--/ + |
                  |   +     +          |
                  | +     +        +   |
                @@@@@@@@@@@@@@@@@@@@@@@@@@
              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            \   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
             \ /  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
              V   \   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
                   \ /  @@@@@@@@@@@@@@@@@@@@@@@@@@
                    V     @@@@@@@@@@@@@@@@@@@@@@@@
                            @@@@@@@@@@@@@@@@@@@@@@
                    /\      @@@@@@@@@@@@@@@@@@@@@@
                   /  \  @@@@@@@@@@@@@@@@@@@@@@@@@
              /\  /    @@@@@@@@@@@@@@@@@@@@@@@@@@@
             /  \ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            /    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                @@@@@@@@@@@@@@@@@@@@@@@@@@
                    @@@@@@@@@@@@@@@@@@

"""
            )
        print("""    (Don't forget to upload your project to receive credit)""")

        if self.edx_output:
            self.produce_output()
        if self.gs_output:
            self.produce_grade_scope_output()

    def add_exception_message(self, q, inst, traceback):
        """
        Method to format the exception message, this is more complicated because
        we need to cgi.escape the traceback but wrap the exception in a <pre> tag
        """
        self.fail("FAIL: Exception raised: %s" % inst)
        self.add_message("")
        for line in traceback.format_exc().split("\n"):
            self.add_message(line)

    def add_error_hints(self, exception_map, error_instance, question_num):
        type_of = str(type(error_instance))
        question_name = "q" + question_num
        error_hint = ""

        # question specific error hints
        if exception_map.get(question_name):
            question_map = exception_map.get(question_name)
            if question_map.get(type_of):
                error_hint = question_map.get(type_of)
        # fall back to general error messages if a question specific
        # one does not exist
        if exception_map.get(type_of):
            error_hint = exception_map.get(type_of)

        # dont include the HTML if we have no error hint
        if not error_hint:
            return ""

        for line in error_hint.split("\n"):
            self.add_message(line)

    def produce_grade_scope_output(self):
        out_dct = {}

        # total of entire submission
        total_possible = sum(self.maxes.values())
        total_score = sum(self.points.values())
        out_dct["score"] = total_score
        out_dct["max_score"] = total_possible
        out_dct["output"] = "Total score (%d / %d)" % (total_score, total_possible)

        # individual tests
        tests_out = []
        for name in self.questions:
            test_out = {}
            # test name
            test_out["name"] = name
            # test score
            test_out["score"] = self.points[name]
            test_out["max_score"] = self.maxes[name]
            # others
            is_correct = self.points[name] >= self.maxes[name]
            test_out["output"] = "  Question {num} ({points}/{max}) {correct}".format(
                num=(name[1] if len(name) == 2 else name),
                points=test_out["score"],
                max=test_out["max_score"],
                correct=("X" if not is_correct else ""),
            )
            test_out["tags"] = []
            tests_out.append(test_out)
        out_dct["tests"] = tests_out

        # file output
        with open("gradescope_response.json", "w") as outfile:
            json.dump(out_dct, outfile)
        return

    def produce_output(self):
        edx_output = open("edx_response.html", "w")
        edx_output.write("<div>")

        # first sum
        total_possible = sum(self.maxes.values())
        total_score = sum(self.points.values())
        check_or_x = '<span class="incorrect"/>'
        if total_score >= total_possible:
            check_or_x = '<span class="correct"/>'
        header = """
        <h3>
            Total score ({total_score} / {total_possible})
        </h3>
    """.format(
            total_score=total_score, total_possible=total_possible, check_or_x=check_or_x
        )
        edx_output.write(header)

        for q in self.questions:
            if len(q) == 2:
                name = q[1]
            else:
                name = q
            check_or_x = '<span class="incorrect"/>'
            if self.points[q] >= self.maxes[q]:
                check_or_x = '<span class="correct"/>'
            # messages = '\n<br/>\n'.join(self.messages[q])
            messages = "<pre>%s</pre>" % "\n".join(self.messages[q])
            output = """
        <div class="test">
          <section>
          <div class="shortform">
            Question {q} ({points}/{max}) {check_or_x}
          </div>
        <div class="longform">
          {messages}
        </div>
        </section>
      </div>
      """.format(
                q=name,
                max=self.maxes[q],
                messages=messages,
                check_or_x=check_or_x,
                points=self.points[q],
            )
            # print "*** output for Question %s " % q[1]
            # print output
            edx_output.write(output)
        edx_output.write("</div>")
        edx_output.close()
        edx_output = open("edx_grade", "w")
        edx_output.write(str(self.points.total_count()))
        edx_output.close()

    def fail(self, message, raw=False):
        "Sets sanity check bit to false and outputs a message"
        self.sane = False
        self.assign_zero_credit()
        self.add_message(message, raw)

    def assign_zero_credit(self):
        self.points[self.current_question] = 0

    def add_points(self, amt):
        self.points[self.current_question] += amt

    def deduct_points(self, amt):
        self.points[self.current_question] -= amt

    def assign_full_credit(self, message="", raw=False):
        self.points[self.current_question] = self.maxes[self.current_question]
        if message != "":
            self.add_message(message, raw)

    def add_message(self, message, raw=False):
        if not raw:
            # We assume raw messages, formatted for HTML, are printed separately
            if self.mute:
                util.unmute_print()
            print("*** " + message)
            if self.mute:
                util.mute_print()
            message = html.escape(message)
        self.messages[self.current_question].append(message)

    def add_message_to_email(self, message):
        print("WARNING**** add_message_to_email is deprecated %s" % message)
        for line in message.split("\n"):
            pass
            # print '%%% ' + line + ' %%%'
            # self.messages[self.current_question].append(line)


class Counter(dict):
    """
    Dict with default 0
    """

    def __getitem__(self, idx):
        try:
            return dict.__getitem__(self, idx)
        except KeyError:
            return 0

    def total_count(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())
