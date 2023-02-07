import subprocess
import shlex
from subprocess import Popen, PIPE
from threading import Timer

def run(cmd, timeout_sec):
    proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        return stdout.decode('utf-8')[0:-1], stderr.decode('utf-8')[0:-1]
    finally:
        timer.cancel()
    return None

def indent(string, by="    ", ignore_first=False):
    indent_string = (" "*by) if isinstance(by, int) else by
    string = string if isinstance(string, str) else f"{string}"
    start = indent_string if not ignore_first else ""
    return start + string.replace("\n", "\n"+indent_string)
        
# 
# auto install ez_yaml
# 
try:
    import ez_yaml
except Exception as error:
    print(subprocess.check_output(["python3", "-m", "pip", "install", "ez_yaml"]).decode('utf-8')[0:-1])
    import ez_yaml

# 
# parse args
# 
import sys
heuristics_to_test = sys.argv[1:]
if len(heuristics_to_test) == 0:
    print("please list the names of your heuristics as arguments")

base_tests = [
    [ "PositionSearchProblem", "tiny_maze"         , ],
    [ "PositionSearchProblem", "medium_maze"       , ],
    [ "PositionSearchProblem", "big_maze"          , ],
    [ "PositionSearchProblem", "huge_maze"         , ],
    [ "FoodSearchProblem"    , "food_search_1"     , ],
    [ "FoodSearchProblem"    , "box_search"        , ],
    [ "FoodSearchProblem"    , "food_search_2"     , ],
    [ "FoodSearchProblem"    , "food_search_3"     , ],
]
tests = []
for base_test in base_tests:
    for heuristic_name in heuristics_to_test:
        tests.append([ heuristic_name, *base_test ])

def run_and_extract_data(heuristic, problem, layout):
    output, error_output = run(" ".join(["python3", "pacman.py", "--timeout", "1", "--quiet_text_graphics", "-l", layout, "-p", "SearchAgent", "-a", f"prob={problem},heuristic={heuristic}", ]), timeout_sec=30)
    # seconds
    if not output:
        return [ None, None, None, "timed out" ]
    
    # output example:
    # """[SearchAgent] using problem type FoodSearchProblem
    # Path found with total cost of 51 in 0.0 seconds
    # Search nodes expanded: 579
    # Pacman emerges victorious! Score: 489
    # Average Score: 489.0
    # Scores:        489.0
    # Win Rate:      1/1 (1.00)
    # Record:        Win"""
    data = None
    pacman_score = None
    nodes_expanded = None
    solution_length = None
    seconds = None
    try:
        relevent_lines = output.split("\n")[-7:]
        output = "\n".join(relevent_lines)
        output = output.replace(f"Path found with total cost of","Path found with total cost of:")
        output = output.replace(f"Warning: this does not look like a regular search maze\n","")
        data  = ez_yaml.to_object(
            string=output
        )
        pacman_score          = data['Pacman emerges victorious! Score']
        nodes_expanded        = data['Search nodes expanded']
        solution_length, time = data['Path found with total cost of'].split(" in ")
        seconds, *_           = time.split(" ")
    except Exception as error:
        import os
        error_path = f"error/{heuristic}_{layout}.log"
        os.makedirs(os.path.dirname(error_path), exist_ok=True)
        with open(error_path, 'w') as the_file:
            if error_output:
                the_file.write(error_output)
            else:
                print("Tried to parse the output",                    file=the_file)
                print("I expected some output that includes:",        file=the_file)
                print("    Path found with total cost of",            file=the_file)
                print("    Pacman emerges victorious! Score: ",       file=the_file)
                print("    Search nodes expanded: ",                  file=the_file)
                print("    Path found with total cost of: ",          file=the_file)
                print("",                                             file=the_file)
                print("however instead I got this output:",           file=the_file)
                print(indent(output),                                 file=the_file)
                print("",                                             file=the_file)
                print("this is the error I got when parsing that:",   file=the_file)
                print(indent(f"{error}"),                             file=the_file)
                print("",                                             file=the_file)
                print("this is partially parsed data:",               file=the_file)
                print(f'''    pacman_score = {pacman_score}''',       file=the_file)
                print(f'''    nodes_expanded = {nodes_expanded}''',   file=the_file)
                print(f'''    solution_length = {solution_length}''', file=the_file)
                print(f'''    seconds = {seconds}''',                 file=the_file)
                print(f'''    data = {data}''',                       file=the_file)
            
        return [ None, None, None, error_path,  ]
    
    return pacman_score, nodes_expanded, int(solution_length), seconds

longest_name = max(*[ len(each) for each in  heuristics_to_test])+2

print()
print(f"""{f"HEURISTIC".rjust(longest_name)},              LAYOUT,   SECONDS, EXPANDED_NODE_COUNT, PACMAN_SCORE, SOLUTION_LENGTH""")
for heuristic, problem, layout in tests:
    pacman_score, nodes_expanded, solution_length, seconds = run_and_extract_data(heuristic, problem, layout)
    print(f"""{f"{heuristic},".rjust(longest_name+1)} {f"{layout}".rjust(19)}, {f"{seconds}".rjust(9)}, {f"{nodes_expanded}".rjust(19)}, {f"{pacman_score}".rjust(12)}, {f"{solution_length}".rjust(15)}""")
    
    
