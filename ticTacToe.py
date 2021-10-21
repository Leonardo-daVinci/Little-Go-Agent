# first we need a state class for board and a judge
# 3 major components  -  state, action, reward
import pickle

import numpy as np

# this is specific to the game, increase to 5 for Go
BOARD_ROWS = 3
BOARD_COLUMNS = 3
LEARNING_RATE = 0.2
DISCOUNT = 0.9
# epsilon denotes the probability of taking a random action or in short exploration
EPSILON = 0.3
ROUNDS = 100


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLUMNS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None  # not sure about this

        # since p1 always plays first
        self.playerSymbol = 1

    # this is used to hash the board table so that it can be stored in the state, value directory
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLUMNS * BOARD_ROWS))
        return self.boardHash

    # get the available positions in the board
    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLUMNS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    # add the player symbol in the board and then change the player
    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # check winner of the game and set the isEnd value
    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLUMNS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLUMNS)])
        diag_sum2 = sum([self.board[i, BOARD_COLUMNS - i - 1] for i in range(BOARD_COLUMNS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    # only when game ends - this is to be calculated for Go
    def giveReward(self):
        result = self.winner()
        print(f"Winner of the game is {result}")
        # back propagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)  # here tie is considered as bad result
            self.p2.feedReward(0.5)

    # we need to reset board after the game is over
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLUMNS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    # now we add the training against itself and learning the q-values in the process
    def play(self, rounds=ROUNDS):
        for i in range(rounds):
            if i % 10 == 0:
                print(f"Rounds:  {i}")
            while not self.isEnd:
                # for player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and update the board
                self.updateState(p1_action)
                boardHash = self.getHash()
                self.p1.addState(boardHash)

                # check if someone won the game or not
                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break


# player class to represent the player that can do the following actions:
# Choose action based on estimations
# record all states of game
# update Q values after each game
# save and load policy


class Player:
    def __init__(self, name, exp_rate=EPSILON):
        self.name = name
        self.states = []  # record all positions of player in each game
        self.lr = LEARNING_RATE
        self.exp_rate = exp_rate
        self.decay_gamma = DISCOUNT
        # we save state value pairs in form of a dictionary.
        self.states_value = {}

    def getHash(self, board):
        return str(board.reshape(BOARD_COLUMNS * BOARD_ROWS))

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # we take a random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -np.inf
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self, state):
        self.states.append(state)

    # this is to be done at the end of each game
    # where we propagate the values obtained as rewards in states_value dictionary
    def feedReward(self, reward):
        # as we are back propagating the values, we use reversed
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        with open('policy_' + str(self.name), 'wb') as fw:
            pickle.dump(self.states, fw)

    def loadPolicy(self, file):
        with open(file, 'rb') as f:
            self.states_value = pickle.load(f)


if __name__ == '__main__':
    # training the agents
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training the agents ...")
    st.play(50)
