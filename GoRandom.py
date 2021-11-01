import numpy as np


class RandomPlayer:

    def __init__(self, side=None):
        self.side = side

    def set_side(self,side):
        self.side = side

    def move(self, board):
        candidates = []
        for i in range(0, 5):
            for j in range(0, 5):
                if board.is_valid_move(i, j, self.side):
                    candidates.append(tuple([i, j]))

        if candidates:
            idx = np.random.randint(len(candidates))
            random_move = candidates[idx]
            row, col = random_move[0], random_move[1]
            board.move(row, col, self.side)
            # print("Playing random move: ", row, col)
            return row, col
        else:
            # print("Random Pass")
            return "PASS"

    def learn(self, board):
        pass
