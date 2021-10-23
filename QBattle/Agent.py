import numpy as np

R_WIN = 1.0
R_LOSS = 0.0
LEARNING_RATE = 0.7
DISCOUNT = 0.9
INITIAL_VAL = 0.1


class Agent:
    GAME_NUMS = 10_000

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
    def _best_move(self, board):
        state = board.getHash()
        q_values = self.Q(state)
        result = np.where(q_values == np.amax(q_values))
        return result[0], result[1]

    # if game is over then no move, otherwise select best move and play that move.
    # we also save the move in our history states
    def move(self, board):
        if board.game_over():
            return
        r, c = self._best_move(board)
        self.history_states.append((board.getHash(), (r, c)))
        return board.move(r, c, self.side)

    # to learn we do the following
    # 1. Get the reward if same is complete.
    # 2. reverse the states and then propagate this q values to all the states.
    def learn(self, board):
        if board.game_result == self.side:
            reward = R_WIN
        else:
            reward = R_LOSS
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            x, y = move[0], move[1]
            q = self.Q(state)
            if max_q_value < 0:
                q[x, y] = reward
            else:
                q[x, y] = q[x, y] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)
        self.history_states = []
