import numpy as np

DRAW_REWARD = 0.5
WIN_REWARD = 1.0
LOSS_REWARD = 0.0
LEARNING_RATE = 0.7
DISCOUNT = 0.9
INITIAL_VAL = 0.1


class Agent:
    GAME_NUMS = 100

    def __init__(self, side=None):
        self.side = side
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT
        # dictionary of Q-Values
        self.q_values = {}
        self.history_states = []
        self.initial_value = INITIAL_VAL

    # select the side the agent is on - won't be required for the final player
    def set_side(self, side):
        self.side = side

    # this function returns Q value of the given state
    # if it isn't there than initialize with INITIAL_VAL
    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((5, 5))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    # selecting the best possible move i.e max value
    # if there are no valid steps then return pass
    def _best_move(self, board):
        state = board.getHash()
        q_values = self.Q(state)
        num_steps = 0
        while num_steps < 24:
            i, j = self._find_max(q_values)
            if board.valid_place_check(i, j, self.side, board.board):
                return i, j
            else:
                q_values[i][j] = -1.0
                num_steps += 1

        if num_steps == 24:
            return "PASS"

    def _find_max(self, q_values):
        curr_max = -np.inf
        row, col = 0, 0
        for i in range(0, 5):
            for j in range(0, 5):
                if q_values[i][j] > curr_max:
                    curr_max = q_values[i][j]
                    row, col = i, j
        return row, col

    # if game is over then no move, otherwise select best move and play that move.
    # we also save the move in our history states
    def move(self, board):
        # print("Learner move")
        result = self._best_move(board)
        if result is "PASS":
            return "PASS", board.board, -1, board.board
        else:
            r, c = result[0], result[1]
            self.history_states.append((board.getHash(), (r, c)))
            a, b, c = board.move(r, c, self.side, board.board)
            return "MOVE", a, b, c

    # to learn we do the following
    # 1. Get the reward if same is complete.
    # 2. reverse the states and then propagate this q values to all the states.
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
