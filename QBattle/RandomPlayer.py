from Board import Board
import numpy as np

DONE = 0
import random


class RandomPlayer:

    def __init__(self, side=None):
        self.side = side

    def set_side(self, side):
        self.side = side

    def move(self, board):
        # print("Random move")
        candidates = []
        for i in range(0, 5):
            for j in range(0, 5):
                if board.valid_place_check(i, j, self.side, board.board):
                    candidates.append(tuple([i, j]))

        if candidates:
            idx = np.random.randint(len(candidates))
            random_move = candidates[idx]
            row, col = random_move[0], random_move[1]
            a, b, c = board.move(row, col, self.side, board.board)
            return "MOVE", a, b, c
        else:
            board.game_result = DONE
            return "PASS", board.board, -1, board.board

    def learn(self, board):
        pass
