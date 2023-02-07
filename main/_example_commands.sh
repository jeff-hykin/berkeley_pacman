# play pacman yourself
python pacman.py --layout big_safe_search

# autograder
python autograder.py a_star_only
python autograder.py a_star_only -q q1
python autograder.py a_star_only -q q2

# maze problem
python pacman.py -l medium_maze -p SearchAgent -a fn=a_star_search
python pacman.py -l medium_maze -p SearchAgent -a fn=a_star_search,heuristic=null_heuristic
python pacman.py -l medium_maze -p SearchAgent -a fn=a_star_search,heuristic=heuristic1

# food problem
python pacman.py -l food_search_1 -p SearchAgent -a prob=FoodSearchProblem
python pacman.py -l food_search_1 -p SearchAgent -a prob=FoodSearchProblem,heuristic=heuristic1

# comparision test
python ./benchmark.py null_heuristic your_heuristic 