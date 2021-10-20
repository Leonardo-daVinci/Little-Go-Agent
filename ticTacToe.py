# first we need a state class for board and a judge
# 3 major components  -  state, action, reward
import numpy as np

# this is specific to the game, increase to 5 for Go
BOARD_ROWS = 3
BOARD_COLUMNS = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros(BOARD_ROWS, BOARD_COLUMNS)
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None  # not sure about this

        # since p1 always plays first
        self.playerSymbol = 1

    # this is usef to hash the board table so that it can be stored in the state, value directory
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
        # back propagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1) # here tie is considered as bad result
            self.p2.feedReward(0.5)

    