class Agent:
    def __init__(self):
        # NOTE: test cases will change this
        self._wumpus_world = [
            [ "", "" , "" , ""], # Rooms [1, 1] to [4, 1]
            [ "", "W", "P", ""], # Rooms [1, 2] to [4, 2]
            [ "", "" , "" , ""], # Rooms [1, 3] to [4, 3]
            [ "", "" , "" , ""], # Rooms [1, 4] to [4, 4]
        ]
        self.current_location = [1, 1]
        self.is_alive = True
        self.escaped = False
        self.valid_actions = ["Up", "Down", "Left", "Right"]

    def _find_indices_for_location(self, loc):
        x, y = loc
        i, j = y - 1, x - 1
        return i, j

    def _check_for_pit_wumpus(self):
        ww = self._wumpus_world
        i, j = self._find_indices_for_location(self.current_location)
        if "P" in ww[i][j]:
            self.is_alive = False
            print("Agent died by falling in a Pit")
        elif "W" in ww[i][j]:
            self.is_alive = False
            print("Agent died by getting eaten by a Wumpus")
        return self.is_alive

    def take_action(
        self, action
    ):  # The function takes an action and returns whether the Agent is alive
        # after taking the action.
        
        assert action in self.valid_actions, f"Invalid Action.\nValid actions are {self.valid_actions}"
        if self.is_alive == False:
            print(
                "Action cannot be performed. Agent is DEAD. Location:{0}".format(
                    self.current_location
                )
            )
            return False
        if self.escaped == True:
            print(
                "Action cannot be performed. Agent has exited the Wumpus world.".format(
                    self.current_location
                )
            )
            return False

        index = self.valid_actions.index(action)
        valid_moves = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        move = valid_moves[index]
        new_location = []
        for v, inc in zip(self.current_location, move):
            z = v + inc  # increment location index
            z = (
                4 if z > 4 else 1 if z < 1 else z
            )  # Ensure that index is between 1 and 4
            new_location.append(z)
        self.current_location = new_location
        print(f"action: {action.ljust(5)}  current_location: {self.current_location}")
        if self.current_location[0] == 4 and self.current_location[1] == 4:
            self.escaped = True
        return self._check_for_pit_wumpus()

    def _find_adjacent_rooms(self):
        valid_moves = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        adj_rooms = []
        for v_m in valid_moves:
            room = []
            valid = True
            for v, inc in zip(self.current_location, v_m):
                z = v + inc
                if z < 1 or z > 4:
                    valid = False
                    break
                else:
                    room.append(z)
            if valid == True:
                adj_rooms.append(room)
        return adj_rooms

    def perceive_current_location(self):  # This function perceives the current location.
        # It tells whether breeze and stench are present in the current location.
        breeze, stench = False, False
        ww = self._wumpus_world
        if self.is_alive == False:
            print(
                "Agent cannot perceive. Agent is DEAD. Location:{0}".format(
                    self.current_location
                )
            )
            return [None, None]
        if self.escaped == True:
            print(
                "Agent cannot perceive. Agent has exited the Wumpus World.".format(
                    self.current_location
                )
            )
            return [None, None]

        adj_rooms = self._find_adjacent_rooms()
        for room in adj_rooms:
            i, j = self._find_indices_for_location(room)
            if "P" in ww[i][j]:
                breeze = True
            if "W" in ww[i][j]:
                stench = True
        return [breeze, stench]

    def observe(self):
        """
        x, y, breeze, stench = self.observe()
        """
        return *self.current_location, *self.perceive_current_location()
