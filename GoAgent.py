from copy import deepcopy

import numpy as np
import json
from GoRandom import RandomPlayer

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0
LEARNING_RATE = 0.7
DISCOUNT = 0.9
INITIAL_VAL = 0.5
EXP_RATE = 0.3


class Agent:
    GAME_NUM = 10_000

    def __init__(self, side=None):
        self.side = side
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT
        self.q_values = {}
        self.history_states = []
        self.initial_value = INITIAL_VAL

    def set_side(self, side):
        self.side = side

    # returns Q value of the state
    def Q(self, state):
        if state not in self.q_values:
            q_val = np.array([[0.05, 0.1, 0.1, 0.1, 0.05],
                              [0.1, 0.3, 0.3, 0.3, 0.1],
                              [0.1, 0.3, 0.3, 0.3, 0.1],
                              [0.1, 0.3, 0.3, 0.3, 0.1],
                              [0.05, 0.1, 0.1, 0.1, 0.05]])
            # q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def _select_best_move(self, board):
        state = board.encode_state()
        q_values = self.Q(state)
        result = self._find_max(q_values, board)
        if result == "PASS":
            return "PASS"
        else:
            return result[0], result[1]

    def _find_max(self, q_values, board):
        # finding out the candidates first
        candidates = []
        for i in range(0, 5):
            for j in range(0, 5):
                if board.is_valid_move(i, j, self.side):
                    candidates.append(tuple([i, j]))
                else:
                    q_values[i][j] = -1.0

        if not candidates:
            return "PASS"

        else:
            curr_max = -np.inf
            row, col = 0, 0
            for idx in candidates:
                if q_values[idx[0]][idx[1]] > curr_max:
                    curr_max = q_values[idx[0]][idx[1]]
                    row, col = idx[0], idx[1]
            return row, col

    def move(self, board):
        result = self._select_best_move(board)
        # print(f"Player move: {result}")
        if result == "PASS":
            return "PASS"
        else:
            row, col = result[0], result[1]
            self.history_states.append((board.encode_state(), (row, col)))
            board.move(row, col, self.side)
            # print("Playing Agent move: ", row, col)
            return row, col

    def learn(self, board):
        if board.game_result == 0:
            reward = DRAW_REWARD
        elif board.game_result == self.side:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            q = self.Q(state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)
        self.history_states = []

        # region Description
        # q_next = self.q_table[next_state]
        # q_next = np.zeros([self.action_size]) if done else q_next
        # q_target = reward + self.discount_rate * np.max(q_next)
        #
        # q_update = q_target - self.q_table[state, action]
        # self.q_table[state, action] += self.learning_rate * q_update
        # endregion

    def save_QValues(self):
        print(len(self.q_values))
        stringDIct = deepcopy(self.q_values)
        for key, values in stringDIct.items():
            stringDIct[key] = json.dumps(values, cls=MyEncoder)

        with open("Qval.txt", 'w') as f:
            print("writing in the file")
            f.write(json.dumps(stringDIct))
            f.write("\n")

    def load_QValues(self):
        with open("Qval.txt", 'r') as f:
            new_dictionary = {}
            for line in f:
                dicts_from_file = dict()
                dicts_from_file = (eval(line))
                for k, v in dicts_from_file.items():
                    # here we need to make v into list of 5 by 5
                    new_v = v[1:-1].split("[")
                    print("_"*60)
                    print(new_v)
                        # int_list = [float(j) for j in (new_v[i].rstrip("]")).split(",")]
                        # new_list.append(int_list)
                    # print(type(new_list))


                    new_dictionary[k] = np.array(v)
            return new_dictionary

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
