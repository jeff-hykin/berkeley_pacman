from __future__ import print_function

# autograder.py
# -------------
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


# imports from python standard library
import grading
import imp
import optparse
import os
import re
import sys
import random
from project_params import choices

random.seed(0)
try:
    from pacman import GameState
except:
    pass

# register arguments and set default values
def read_command(argv):
    parser = optparse.OptionParser(description="Run public tests on student code")
    parser.set_defaults(
        generate_solutions=False,
        edx_output=False,
        gs_output=False,
        mute_output=False,
        print_test_case=False,
        no_graphics=False,
    )
    parser.add_option(
        "--test-directory",
        dest="test_root",
        default=project_params["TEST_CASES"],
        help="Root test directory which contains subdirectories corresponding to each question",
    )
    parser.add_option(
        "--student-code",
        dest="student_code",
        default=project_params["STUDENT_CODE_DEFAULT"],
        help="comma separated list of student code files",
    )
    parser.add_option(
        "--code-directory",
        dest="code_root",
        default="",
        help="Root directory containing the student and test_class code",
    )
    parser.add_option(
        "--test-case-code",
        dest="test_case_code",
        default=project_params["PROJECT_TEST_CLASSES"],
        help="class containing test_class classes for this project",
    )
    parser.add_option(
        "--generate-solutions",
        dest="generate_solutions",
        action="store_true",
        help="Write solutions generated to .solution file",
    )
    parser.add_option(
        "--edx-output",
        dest="edx_output",
        action="store_true",
        help="Generate ed_x output files",
    )
    parser.add_option(
        "--gradescope-output",
        dest="gs_output",
        action="store_true",
        help="Generate GradeScope output files",
    )
    parser.add_option(
        "--mute",
        dest="mute_output",
        action="store_true",
        help="Mute output from executing tests",
    )
    parser.add_option(
        "--print-tests",
        "-p",
        dest="print_test_case",
        action="store_true",
        help="Print each test case before running them.",
    )
    parser.add_option(
        "--test",
        "-t",
        dest="run_test",
        default=None,
        help="Run one particular test.  Relative to test root.",
    )
    parser.add_option(
        "--question",
        "-q",
        dest="grade_question",
        default=None,
        help="Grade one particular question.",
    )
    parser.add_option(
        "--no-graphics",
        dest="no_graphics",
        action="store_true",
        help="No graphics display for pacman games.",
    )
    (options, args) = parser.parse_args(argv)
    return options


# confirm we should author solution files
def confirm_generate():
    print("WARNING: this action will overwrite any solution files.")
    print("Are you sure you want to proceed? (yes/no)")
    while True:
        ans = sys.stdin.readline().strip()
        if ans == "yes":
            break
        elif ans == "no":
            sys.exit(0)
        else:
            print('please answer either "yes" or "no"')


# TODO: Fix this so that it tracebacks work correctly
# Looking at source of the traceback module, presuming it works
# the same as the intepreters, it uses co_filename.  This is,
# however, a readonly attribute.
def set_module_name(module, filename):
    function_type = type(confirm_generate)
    class_type = type(optparse.Option)

    for i in dir(module):
        o = getattr(module, i)
        if hasattr(o, "__file__"):
            continue

        if type(o) == function_type:
            setattr(o, "__file__", filename)
        elif type(o) == class_type:
            setattr(o, "__file__", filename)
            # TODO: assign member __file__'s?
        # print i, type(o)


# from c_string_io import StringIO


def load_module_string(module_source):
    # Below broken, imp doesn't believe its being passed a file:
    #    ValueError: load_module arg#2 should be a file or None
    #
    # f = StringIO(module_code_dict[k])
    # tmp = imp.load_module(k, f, k, (".py", "r", imp.PY_SOURCE))
    tmp = imp.new_module(k)
    exec(module_code_dict[k], tmp.__dict__)
    set_module_name(tmp, k)
    return tmp


import py_compile


def load_module_file(module_name, file_path):
    with open(file_path, "r") as f:
        return imp.load_module(
            module_name, f, "%s.py" % module_name, (".py", "r", imp.PY_SOURCE)
        )


def read_file(path, root=""):
    "Read file from disk at specified path and return as string"
    with open(os.path.join(root, path), "r") as handle:
        return handle.read()


#######################################################################
# Error Hint Map
#######################################################################

# TODO: use these
ERROR_HINT_MAP = {
    "q1": {
        "<type 'exceptions.IndexError'>": """
      We noticed that your project threw an IndexError on q1.
      While many things may cause this, it may have been from
      assuming a certain number of successors from a state space
      or assuming a certain number of actions available from a given
      state. Try making your code more general (no hardcoded indices)
      and submit again!
    """
    },
    "q3": {
        "<type 'exceptions.AttributeError'>": """
        We noticed that your project threw an AttributeError on q3.
        While many things may cause this, it may have been from assuming
        a certain size or structure to the state space. For example, if you have
        a line of code assuming that the state is (x, y) and we run your code
        on a state space with (x, y, z), this error could be thrown. Try
        making your code more general and submit again!

    """
    },
}

import pprint


def split_strings(d):
    d2 = dict(d)
    for k in d:
        if k[0:2] == "__":
            del d2[k]
            continue
        if d2[k].find("\n") >= 0:
            d2[k] = d2[k].split("\n")
    return d2


def print_test(test_dict, solution_dict):
    pp = pprint.PrettyPrinter(indent=4)
    print("Test case:")
    for line in test_dict["__raw_lines__"]:
        print("   |", line)
    print("Solution:")
    for line in solution_dict["__raw_lines__"]:
        print("   |", line)


def run_test(test_name, module_dict, print_test_case=False, display=None):
    import test_parser
    import test_classes

    for module in module_dict:
        setattr(sys.modules[__name__], module, module_dict[module])

    test_dict = test_parser.TestParser(test_name + ".test").parse()
    solution_dict = test_parser.TestParser(test_name + ".solution").parse()
    test_out_file = os.path.join("%s.test_output" % test_name)
    test_dict["test_out_file"] = test_out_file
    test_class = getattr(project_test_classes, test_dict["class"])

    question_class = getattr(test_classes, "Question")
    question = question_class({"max_points": 0}, display)
    test_case = test_class(question, test_dict)

    if print_test_case:
        print_test(test_dict, solution_dict)

    # This is a fragile hack to create a stub grades object
    grades = grading.Grades(project_params["PROJECT_NAME"], [(None, 0)])
    test_case.execute(grades, module_dict, solution_dict)


# returns all the tests you need to run in order to run question
def get_depends(test_parser, test_root, question):
    all_deps = [question]
    question_dict = test_parser.TestParser(
        os.path.join(test_root, question, "CONFIG")
    ).parse()
    if "depends" in question_dict:
        depends = question_dict["depends"].split()
        for d in depends:
            # run dependencies first
            all_deps = get_depends(test_parser, test_root, d) + all_deps
    return all_deps


# get list of questions to grade
def get_test_subdirs(test_parser, test_root, question_to_grade):
    problem_dict = test_parser.TestParser(os.path.join(test_root, "CONFIG")).parse()
    if question_to_grade != None:
        questions = get_depends(test_parser, test_root, question_to_grade)
        if len(questions) > 1:
            print(
                "Note: due to dependencies, the following tests will be run: %s"
                % " ".join(questions)
            )
        return questions
    if "order" in problem_dict:
        return problem_dict["order"].split()
    return sorted(os.listdir(test_root))


# evaluate student code
def evaluate(
    generate_solutions,
    test_root,
    module_dict,
    exception_map=ERROR_HINT_MAP,
    edx_output=False,
    mute_output=False,
    gs_output=False,
    print_test_case=False,
    question_to_grade=None,
    display=None,
):
    # imports of testbench code.  note that the test_classes import must follow
    # the import of student code due to dependencies
    import test_parser
    import test_classes

    for module in module_dict:
        setattr(sys.modules[__name__], module, module_dict[module])

    questions = []
    question_dicts = {}
    test_subdirs = get_test_subdirs(test_parser, test_root, question_to_grade)
    for q in test_subdirs:
        subdir_path = os.path.join(test_root, q)
        if not os.path.isdir(subdir_path) or q[0] == ".":
            continue

        # create a question object
        question_dict = test_parser.TestParser(
            os.path.join(subdir_path, "CONFIG")
        ).parse()
        question_class = getattr(test_classes, question_dict["class"])
        question = question_class(question_dict, display)
        question_dicts[q] = question_dict

        # load test cases into question
        tests = [t for t in os.listdir(subdir_path) if re.match("[^#~.].*\.test\Z", t)]
        tests = [re.match("(.*)\.test\Z", t).group(1) for t in tests]
        for t in sorted(tests):
            test_file = os.path.join(subdir_path, "%s.test" % t)
            solution_file = os.path.join(subdir_path, "%s.solution" % t)
            test_out_file = os.path.join(subdir_path, "%s.test_output" % t)
            test_dict = test_parser.TestParser(test_file).parse()
            if test_dict.get("disabled", "false").lower() == "true":
                continue
            test_dict["test_out_file"] = test_out_file
            test_class = getattr(project_test_classes, test_dict["class"])
            test_case = test_class(question, test_dict)

            def makefun(test_case, solution_file):
                if generate_solutions:
                    # write solution file to disk
                    return lambda grades: test_case.write_solution(
                        module_dict, solution_file
                    )
                else:
                    # read in solution dictionary and pass as an argument
                    test_dict = test_parser.TestParser(test_file).parse()
                    solution_dict = test_parser.TestParser(solution_file).parse()
                    if print_test_case:
                        return lambda grades: print_test(
                            test_dict, solution_dict
                        ) or test_case.execute(grades, module_dict, solution_dict)
                    else:
                        return lambda grades: test_case.execute(
                            grades, module_dict, solution_dict
                        )

            question.add_test_case(test_case, makefun(test_case, solution_file))

        # Note extra function is necessary for scoping reasons
        def makefun(question):
            return lambda grades: question.execute(grades)

        setattr(sys.modules[__name__], q, makefun(question))
        questions.append((q, question.get_max_points()))

    grades = grading.Grades(
        project_params["PROJECT_NAME"],
        questions,
        gs_output=gs_output,
        edx_output=edx_output,
        mute_output=mute_output,
    )
    if question_to_grade == None:
        for q in question_dicts:
            for prereq in question_dicts[q].get("depends", "").split():
                grades.add_prereq(q, prereq)

    grades.grade(sys.modules[__name__], bonus_pic=project_params["BONUS_PIC"])
    return grades.points


def get_display(graphics_by_default, options=None):
    graphics = graphics_by_default
    if options is not None and options.no_graphics:
        graphics = False
    if graphics:
        try:
            import graphics_display

            return graphics_display.PacmanGraphics(1, frame_time=0.05)
        except ImportError:
            pass
    import text_display

    return text_display.NullGraphics()


if __name__ == "__main__":
    argv = sys.argv
    # 
    # select project
    # 
    which_tests = None
    if len(argv) > 1 and isinstance(argv[1], str) and argv[1][0:2] != "--":
        which_tests = argv.pop(1)
    if which_tests in choices.keys():
        project_params = choices[which_tests]
    else:
        options = list(choices.keys())
        argv.insert(1, options[0])
        example = " ".join(argv)
        print('=Error=====================================================')
        print('      When calling autograder.py')
        print('      please set the first argument to be one of these:')
        print('          '+str(options))
        print('      ex:')
        print('          python '+str(example))
        print('===========================================================')
        exit(1)
    
    options = read_command(argv)
    if options.generate_solutions:
        confirm_generate()
    code_paths = options.student_code.split(",")
    # module_code_dict = {}
    # for cp in code_paths:
    #     module_name = re.match('.*?([^/]*)\.py', cp).group(1)
    #     module_code_dict[module_name] = read_file(cp, root=options.code_root)
    # module_code_dict['project_test_classes'] = read_file(options.test_case_code, root=options.code_root)
    # module_dict = load_module_dict(module_code_dict)

    module_dict = {}
    for cp in code_paths:
        module_name = re.match(".*?([^/]*)\.py", cp).group(1)
        module_dict[module_name] = load_module_file(
            module_name, os.path.join(options.code_root, cp)
        )
    module_name = re.match(".*?([^/]*)\.py", options.test_case_code).group(1)
    module_dict["project_test_classes"] = load_module_file(
        module_name, os.path.join(options.code_root, options.test_case_code)
    )

    if options.run_test != None:
        run_test(
            options.run_test,
            module_dict,
            print_test_case=options.print_test_case,
            display=get_display(True, options),
        )
    else:
        evaluate(
            options.generate_solutions,
            options.test_root,
            module_dict,
            gs_output=options.gs_output,
            edx_output=options.edx_output,
            mute_output=options.mute_output,
            print_test_case=options.print_test_case,
            question_to_grade=options.grade_question,
            display=get_display(options.grade_question != None, options),
        )
